#!/usr/bin/env python3
"""Komondor GPU-partition entry point for Stage-2 training.

Hardware: `gpu` node — 64-core EPYC 7763, 256 GB, 4× A100 40 GB (16 cores/GPU).
Workers and DDP ranks follow the GPUs/CPUs Slurm actually gave this job.

Production (4 GPU, 1 day 16 hours)::

    sbatch --account=<account> --mail-user=YOU@email hpc/train_stage2.sbatch

Interactive smoke test (1 GPU, 16 cores, 1 hour)::

    srun -p gpu -A nr_hemo_ai1 --gres=gpu:1 -c 16 --mem-per-cpu=4000 \\
      --time=01:00:00 --pty bash
    # on the compute node:
    source ~/aneuxai_env/bin/activate   # or: conda activate aneurysmgnn
    cd "$SLURM_SUBMIT_DIR"              # or the git checkout
    export ANEUX_EPOCHS=1
    python 1test_encoder_decoder_only/train_pipeline/aneuxai_HPC.py

The ``test`` partition is the same GPU hardware with a 1-hour cap and often
a shorter queue: ``srun -p test -A nr_hemo_ai1 --gres=gpu:1 -c 16 --time=01:00:00 --pty bash``.

Do not run this file on the 3080 Ti. Use ``aneuxai.py`` at home.
"""
from __future__ import annotations

import atexit
import os
import sys

import torch

_HERE = os.path.abspath(os.path.dirname(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import aneuxai as _aneuxai
import config as _config
import latent_metrics as _latent_metrics
from aneuxai import build_tube_cache_only, run_stage2_training
from aneux_paths import CLEANDATA, EXPERIMENT_CACHE, EXPERIMENT_OUTPUT, REPO_ROOT
from dist_utils import (
    barrier,
    destroy_distributed,
    env_local_rank,
    init_distributed,
    is_main_process,
    world_size,
)
from hpc_runtime import (
    detect_project_account,
    job_workspace,
    persist_if_remote,
    project_root,
    scale_hpc_workers,
    slurm_cpu_count,
    stage_training_inputs,
    sync_run_back,
)
from run_report import make_run_dir

# Per-GPU batch 4. Global batch = 4 × n_gpu (16 on a full node). LR stays 2e-4.
# The R* sweep (hpc/sweep_rate_target.sbatch) runs one GPU at batch 16: the
# same global batch, so what it picks carries over to this production setup.
# The split has 493 training cases, so 25/GPU gave 5 optimiser steps an
# epoch and ~2.5k steps in total, far too few for the decoder to converge;
# 4/GPU gives ~31 steps an epoch (~15k over 500 epochs) at about the same
# wall time: ~94% of a step is SplineConv, whose cost is per sample, so an
# epoch costs the same whatever the batch (the old run: ~195 s an epoch).
# DataLoader / cache workers follow SLURM_CPUS_PER_TASK (2× oversubscribe on
# 16 cores/GPU). Override with ANEUX_NUM_WORKERS / ANEUX_CACHE_WORKERS.
BATCH_SIZE = int(os.environ.get("ANEUX_BATCH_SIZE", "4"))
ACCUM_STEPS = int(os.environ.get("ANEUX_ACCUM_STEPS", "1"))
EPOCHS = int(os.environ.get("ANEUX_EPOCHS", "500"))
VAL_EVERY = int(os.environ.get("ANEUX_VAL_EVERY", "1"))
LEARNING_RATE = float(os.environ.get("ANEUX_LR", "2e-4"))
# 0.01: the 2026-09-22 run (1e-4) memorised -- val recon peaked at epoch 110
# while train KL kept climbing.  EMA 0.996 averages ~250 steps (~8 epochs at
# 31 steps an epoch); 0.999 would lag the val curve by ~30 epochs.
WEIGHT_DECAY = float(os.environ.get("ANEUX_WEIGHT_DECAY", "0.01"))
EMA_DECAY = float(os.environ.get("ANEUX_EMA_DECAY", "0.996"))
# R* (nats per valid token) for the GECO hinge; unset keeps config.py's value.
RATE_TARGET = os.environ.get("ANEUX_RATE_TARGET", "").strip()
# a short name for this run, appended to the run folder (e.g. rstar0.5)
RUN_TAG = os.environ.get("ANEUX_RUN_TAG", "").strip()
LR_WARMUP_STEPS = 300
TORCH_THREADS = 1
PIN_MEMORY = True
PREFETCH_FACTOR = 1
PRELOAD_RAM = False
USE_GRADIENT_CHECKPOINTING = False  # A100 40 GB; do not checkpoint
GECO_ETA = 1e-3


def _resolve_worker_counts(n_gpu):
    scaled = scale_hpc_workers(n_gpu=n_gpu, n_cpu=slurm_cpu_count())
    num_workers = scaled["num_workers"]
    cache_workers = scaled["cache_build_workers"]
    if os.environ.get("ANEUX_NUM_WORKERS"):
        num_workers = int(os.environ["ANEUX_NUM_WORKERS"])
    if os.environ.get("ANEUX_CACHE_WORKERS"):
        cache_workers = int(os.environ["ANEUX_CACHE_WORKERS"])
    scaled["num_workers"] = max(0, num_workers)
    scaled["cache_build_workers"] = max(1, cache_workers)
    return scaled


def _activate_python_env_hints():
    """Print which interpreter we got. Env activation is the sbatch's job."""
    print(f"[hpc] python={sys.executable}  version={sys.version.split()[0]}")
    print(f"[hpc] torch={torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(
            f"[hpc] gpu0={torch.cuda.get_device_name(0)}  "
            f"count={torch.cuda.device_count()}"
        )


def _install_geco_eta(eta):
    _config.GECO_ETA = float(eta)


def _install_rate_target(value):
    """Set R* everywhere it is read: losses read config live, the other two copied it."""
    if not value:
        return float(_config.RATE_TARGET_NATS)
    r = float(value)
    if not r > 0.0:
        raise ValueError(f"ANEUX_RATE_TARGET must be > 0, got {value!r}")
    _config.RATE_TARGET_NATS = r
    _latent_metrics.RATE_TARGET_NATS = r
    _aneuxai.RATE_TARGET_NATS = r
    return r


def _cache_only_main():
    """ANEUX_CACHE_ONLY=1: fill ANEUX_CACHE in place from ANEUX_CLEANDATA, then exit."""
    scaled = _resolve_worker_counts(n_gpu=1)
    cache_dir = os.environ.get("ANEUX_CACHE", EXPERIMENT_CACHE)
    cleandata = os.environ.get("ANEUX_CLEANDATA", CLEANDATA)
    print(f"[hpc] cache-only: {cleandata} -> {cache_dir}  workers={scaled['cache_build_workers']}", flush=True)
    errors = build_tube_cache_only(
        cache_dir=cache_dir,
        cleandata_root=cleandata,
        cache_build_workers=scaled["cache_build_workers"],
    )
    # a handful of unbuildable cases is what training skips anyway; more means broken inputs
    if len(errors) > 10:
        raise SystemExit(f"[hpc] {len(errors)} cases failed to build; check cleandata")


def _maybe_reexec_torchrun():
    """Start DDP: one rank per GPU in a single NCCL process group.

    This is not four independent jobs. Each rank runs batch_size graphs on its
    GPU; DistributedSampler shards the dataset; DDP all-reduces gradients so
    the optimizer step is one global batch (batch_size × n_gpu). Skip if
    torchrun / WORLD_SIZE already launched us.
    """
    if os.environ.get("LOCAL_RANK") is not None:
        return
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        return
    nvis = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if nvis <= 1:
        return
    script = os.path.abspath(__file__)
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nvis}",
        "--standalone",
        script,
        *sys.argv[1:],
    ]
    print(f"[hpc] {nvis} GPUs visible; re-launching: {' '.join(cmd)}", flush=True)
    os.execvp(cmd[0], cmd)


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NCCL_DEBUG", os.environ.get("NCCL_DEBUG", "WARN"))
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("ANEUX_MONITOR_SEC", "15")

    if os.environ.get("ANEUX_CACHE_ONLY", "").strip() in ("1", "true", "yes"):
        _cache_only_main()
        return

    _maybe_reexec_torchrun()
    init_distributed()
    local = env_local_rank()
    nproc = max(1, world_size())
    if torch.cuda.is_available():
        nvis = torch.cuda.device_count()
        idx = 0 if nvis <= 1 else max(0, min(local, nvis - 1))
        torch.cuda.set_device(idx)
        device = f"cuda:{idx}"
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = "cpu"

    scaled = _resolve_worker_counts(n_gpu=nproc)
    num_workers = scaled["num_workers"]
    cache_build_workers = scaled["cache_build_workers"]

    main_rank = is_main_process()
    if main_rank:
        _activate_python_env_hints()
        print(
            f"[hpc] DDP ranks={nproc}  per-GPU batch={BATCH_SIZE}  "
            f"global batch={BATCH_SIZE * nproc * ACCUM_STEPS}  "
            f"(NCCL gradient sync, one optimizer step)",
            flush=True,
        )
        print(
            f"[hpc] scale: ngpu={scaled['n_gpu']} ncpu={scaled['n_cpu']} "
            f"dataloader_workers={num_workers} cache_workers={cache_build_workers}",
            flush=True,
        )

    account = detect_project_account()
    proj = project_root(account)
    repo = os.environ.get("ANEUX_REPO_ROOT", REPO_ROOT)
    cleandata_src = os.environ.get("ANEUX_CLEANDATA", CLEANDATA)
    cache_src = os.environ.get("ANEUX_CACHE", EXPERIMENT_CACHE)
    output_persist = os.environ.get("ANEUX_OUTPUT", EXPERIMENT_OUTPUT)

    workspace = None
    staged = None
    run_dir = None

    if main_rank:
        print(f"[hpc] account={account!r}  project_root={proj!r}")
        print(f"[hpc] repo={repo}")
        print(f"[hpc] cleandata_src={cleandata_src}")
        try:
            if os.environ.get("ANEUX_SKIP_STAGE", "").strip() in ("1", "true", "yes"):
                print("[hpc] ANEUX_SKIP_STAGE set; training on project disk")
                workspace = None
            else:
                workspace = job_workspace(account)
        except RuntimeError as exc:
            print(f"[hpc] scratch unavailable ({exc}); training on project disk")
            workspace = None
        if workspace:
            staged = stage_training_inputs(
                repo_root=repo,
                cleandata_src=cleandata_src,
                cache_src=cache_src,
                workspace=workspace,
            )
            cleandata_root = staged["cleandata"]
            cache_dir = staged["cache"]
            output_dir = staged["output"]
        else:
            cleandata_root = cleandata_src
            cache_dir = cache_src
            output_dir = output_persist
        os.makedirs(output_dir, exist_ok=True)
        job_tag = os.environ.get("SLURM_JOB_ID") or ""
        if RUN_TAG:
            job_tag = f"{job_tag}_{RUN_TAG}" if job_tag else RUN_TAG
        run_dir = make_run_dir(output_dir, job_id=job_tag or None)
        print(f"[hpc] run_dir={run_dir}")
    else:
        cleandata_root = cleandata_src
        cache_dir = cache_src
        output_dir = output_persist

    packed = [cleandata_root, cache_dir, output_dir, run_dir, workspace]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(packed, src=0)
        cleandata_root, cache_dir, output_dir, run_dir, workspace = packed
    barrier()

    _install_geco_eta(GECO_ETA)
    rate_target = _install_rate_target(RATE_TARGET)
    if main_rank:
        print(
            f"[hpc] R*={rate_target:g} nats/token  weight_decay={WEIGHT_DECAY:g}  "
            f"ema_decay={EMA_DECAY:g}  epochs={EPOCHS}  tag={RUN_TAG or '-'}",
            flush=True,
        )

    copy_back_done = {"yes": False}

    def _copy_back():
        if copy_back_done["yes"] or not main_rank:
            return
        copy_back_done["yes"] = True
        if cache_dir and os.path.isdir(cache_dir):
            try:
                persist_if_remote(cache_dir, cache_src, "tube_cache -> project")
            except Exception as exc:
                print(f"[hpc] tube_cache persist failed: {exc}", flush=True)
        # without a scratch workspace the run already lives on project disk
        if workspace and run_dir and os.path.isdir(run_dir):
            dest_root = os.path.join(os.path.abspath(output_persist), "runs")
            try:
                dest = sync_run_back(run_dir, dest_root)
                print(f"[hpc] copied run to {dest}", flush=True)
            except Exception as exc:
                print(f"[hpc] copy-back failed: {exc}", flush=True)

    atexit.register(_copy_back)

    extra = {
        "host": "komondor_gpu",
        "account": account,
        "workspace": workspace,
        "project_root": proj,
        "nproc_per_node": nproc,
        "partition": os.environ.get("SLURM_JOB_PARTITION", "gpu"),
        "docs": "https://docs.hpc.dkf.hu/AI/pytorch.html",
        "BATCH_SIZE_PER_GPU": BATCH_SIZE,
        "GLOBAL_BATCH": BATCH_SIZE * nproc * ACCUM_STEPS,
        "NUM_WORKERS": num_workers,
        "CACHE_BUILD_WORKERS": cache_build_workers,
        "n_cpu": scaled["n_cpu"],
        "GECO_ETA": GECO_ETA,
        "RATE_TARGET_NATS": rate_target,
        "RUN_TAG": RUN_TAG,
        "SLURM_ARRAY_JOB_ID": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "SLURM_ARRAY_TASK_ID": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }

    try:
        run_stage2_training(
            output_dir=output_dir,
            cache_dir=cache_dir,
            cleandata_root=cleandata_root,
            batch_size=BATCH_SIZE,
            accum_steps=ACCUM_STEPS,
            epochs=EPOCHS,
            num_workers=num_workers,
            cache_build_workers=cache_build_workers,
            torch_threads=TORCH_THREADS,
            pin_memory=PIN_MEMORY,
            prefetch_factor=PREFETCH_FACTOR,
            preload_ram=PRELOAD_RAM,
            gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            ema_decay=EMA_DECAY,
            lr_warmup_steps=LR_WARMUP_STEPS,
            val_every=VAL_EVERY,
            device=device,
            resume="auto",
            run_name_host="komondor_gpu",
            extra_meta=extra,
            run_dir=run_dir,
        )
    finally:
        destroy_distributed()
        _copy_back()


if __name__ == "__main__":
    main()
