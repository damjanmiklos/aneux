#!/usr/bin/env python3
"""Komondor GPU-partition entry point for Stage-2 training.

Hardware: one exclusive `gpu` node — 64-core EPYC 7763, 256 GB RAM, 4× A100 40 GB.
Launch with the sbatch in ``hpc/train_stage2.sbatch`` (torchrun, 4 processes).

Storage (https://docs.hpc.dkf.hu/storage/overview.html):
the git checkout and the scp'd ``cleandata/`` live on ``/project/<account>``
(HDD Lustre). At job start this script copies cleandata + tube_cache onto
``/scratch/<account>/...`` (NVMe Lustre, fastest tier) and copies the timestamped
run folder back to project when training finishes (or on SIGTERM).

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

import config as _config
from aneuxai import run_stage2_training
from aneux_paths import CLEANDATA, EXPERIMENT_CACHE, EXPERIMENT_OUTPUT, REPO_ROOT
from dist_utils import (
    barrier,
    destroy_distributed,
    env_local_rank,
    init_distributed,
    is_main_process,
)
from hpc_runtime import (
    detect_project_account,
    job_workspace,
    project_root,
    stage_training_inputs,
    sync_run_back,
)
from run_report import make_run_dir

# Real aneuxai.py batch on the 3080 Ti (~14k-node templates): peak 2.45 GiB
# at bs=2. A100 40 GB therefore has room for 10/GPU. Global batch 10×4 = 40
# (PC is 2×16 = 32). LR stays 2e-4; cosine is stretched over 500 epochs.
# 4 DDP ranks × 16 DataLoader workers = 64 processes. Cache warmup is 48
# short-lived processes, then they exit.
# Override at submit time: ANEUX_BATCH_SIZE, ANEUX_NUM_WORKERS, ANEUX_EPOCHS.
BATCH_SIZE = int(os.environ.get("ANEUX_BATCH_SIZE", "10"))
ACCUM_STEPS = int(os.environ.get("ANEUX_ACCUM_STEPS", "1"))
EPOCHS = int(os.environ.get("ANEUX_EPOCHS", "500"))
VAL_EVERY = int(os.environ.get("ANEUX_VAL_EVERY", "1"))
LEARNING_RATE = float(os.environ.get("ANEUX_LR", "2e-4"))
WEIGHT_DECAY = 1e-4
EMA_DECAY = 0.993
LR_WARMUP_STEPS = 300
NUM_WORKERS = int(os.environ.get("ANEUX_NUM_WORKERS", "16"))
CACHE_BUILD_WORKERS = int(os.environ.get("ANEUX_CACHE_WORKERS", "48"))
TORCH_THREADS = 1
PIN_MEMORY = True
PREFETCH_FACTOR = 1
PRELOAD_RAM = False
USE_GRADIENT_CHECKPOINTING = False  # A100 40 GB; do not checkpoint
GECO_ETA = 1e-3  # global 40 vs PC 32; leave η, do not retune from one factor


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


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NCCL_DEBUG", os.environ.get("NCCL_DEBUG", "WARN"))
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("ANEUX_MONITOR_SEC", "15")

    init_distributed()
    local = env_local_rank()
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

    main_rank = is_main_process()
    if main_rank:
        _activate_python_env_hints()

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
        run_dir = make_run_dir(output_dir, job_id=os.environ.get("SLURM_JOB_ID"))
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

    copy_back_done = {"yes": False}

    def _copy_back():
        if copy_back_done["yes"] or not main_rank:
            return
        copy_back_done["yes"] = True
        if run_dir and os.path.isdir(run_dir):
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
        "nproc_per_node": 4,
        "partition": "gpu",
        "docs": "https://docs.hpc.dkf.hu/AI/pytorch.html",
        "BATCH_SIZE_PER_GPU": BATCH_SIZE,
        "GLOBAL_BATCH": BATCH_SIZE * 4 * ACCUM_STEPS,
        "GECO_ETA": GECO_ETA,
    }

    try:
        run_stage2_training(
            output_dir=output_dir,
            cache_dir=cache_dir,
            cleandata_root=cleandata_root,
            batch_size=BATCH_SIZE,
            accum_steps=ACCUM_STEPS,
            epochs=EPOCHS,
            num_workers=NUM_WORKERS,
            cache_build_workers=CACHE_BUILD_WORKERS,
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
