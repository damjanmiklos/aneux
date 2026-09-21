# %% [markdown]
# # Hierarchical PointNeXt–SplineConv VAE for Aneurysm Mesh Deformation
# Stage 2 geometry autoencoder: tree-valued centerline latent + progressive tube decoder.

# %%
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import Subset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from aneux_paths import (
    CLEANDATA,
    EXPERIMENT_OUTPUT,
    EXPERIMENT_CACHE,
)

from config import (
    DEFAULT_LOSS_WEIGHTS,
    DECODER_HIDDEN_DIM,
    GECO_BETA_INIT,
    GECO_ETA,
    GRAD_CLIP,
    HIERARCHY_LEVELS,
    KL_WARMUP_EPOCHS,
    LAMBDA_KL,
    LATENT_DIM,
    LATENT_LEN,
    N_TRUE,
    RATE_TARGET_NATS,
    TUBE_RADIUS_MM,
    configure_stage2_precision,
    normalize_gradient_checkpointing,
)
from dataset import AneurysmDataset
from dist_utils import barrier, is_main_process
from model import GraphVAE
from run_report import dump_json, hardware_snapshot, make_run_dir
from train import train_model

# %% [markdown]
# ## 1. Configuration & Hyperparameters
# Meshes and centerlines come from `cleandata/` (uniform GT, original
# centerline, pregenerated template_mesh + template_centerline). Nothing is
# read from rawdata. The decoder starts from template_mesh, not a Bishop tube.

# %%
OUTPUT_DIR = EXPERIMENT_OUTPUT
CACHE_DIR = EXPERIMENT_CACHE
CLEANDATA_ROOT = CLEANDATA
# True only in vmtk_env: fill missing original/template centerlines with
# centerline_creation.py. Default False assumes those folders are already filled.
ENSURE_DERIVED = False

TUBE_RADIUS = TUBE_RADIUS_MM
N_LENGTH = HIERARCHY_LEVELS[-1][0]
N_RADIAL = HIERARCHY_LEVELS[-1][1]

# 3080 Ti dummy epoch (scratch/vram_probe_batch.py, 20 Sep 2026):
#   batch 1 = 3.53 GiB allocated / 5.07 GiB reserved, ~3.45 GiB/sample.
#   batch 2 is the largest that still leaves WDDM headroom on 12 GiB.
# Accumulate to a global batch of 32 (same as the old 4×8) so GECO / LR
# stay on the measured schedule.
BATCH_SIZE = 2
ACCUM_STEPS = 16
EPOCHS = 200
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
VAL_EVERY = 5
SEED = 31
SPLIT_JSON_NAME = "train_val_split.json"

# Training-hygiene defaults live here (item 11 / §8). train.py consumes them.
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
WD = WEIGHT_DECAY
EMA_DECAY = 0.993
EMA_WARMUP = True
# ~11 opt steps/epoch at 326 train / global-32. 300 steps ≈ 27 epochs of warmup.
LR_WARMUP_STEPS = 300
WD_EXCLUDE_BIAS = True
WD_EXCLUDE_NORMS = True
WD_EXCLUDE_GATES = True
WD_EXCLUDE_1D = True
WD_EXCLUDE_KEYWORDS = (
    "bias",
    "norm",
    "layernorm",
    ".ln.",
    "gate",
    "alpha_raw",
)

# 5950X = 16 cores / 32 threads, 32 GB RAM. 20 DataLoader workers as requested;
# main process keeps 1 torch thread so the workers actually get CPU time.
NUM_WORKERS = 20
CACHE_BUILD_WORKERS = 6
TORCH_THREADS = 1
PIN_MEMORY = False  # Windows WDDM: pinned host RAM fights the 3080 Ti
PREFETCH_FACTOR = 1
PRELOAD_RAM = False

# False = fully off (faster, more VRAM). True = checkpoint encoder + all decoder
# blocks. "fine" = only the 64k-node SplineConvs if a fat graph OOMs with False.
USE_GRADIENT_CHECKPOINTING = False

LOSS_WEIGHTS = dict(DEFAULT_LOSS_WEIGHTS)

# %% [markdown]
# ## 2. Data Preparation
# Paired uniformly_remeshed GT + pregenerated template_mesh / template_centerline
# from cleandata. Tokens and (u, θ) come from the template centerline. Hybrid
# far-from-centerline FPS for x_true, canonical ICA pose (mm preserved).

# %%
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_device_args(argv=None):
    """CLI `--device` / `--gpu`. Unknown args are ignored so notebooks still run."""
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--device",
        default=None,
        help="cpu, cuda, cuda:N, or N. cuda:0 is the first GPU listed in CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="Visible CUDA index (same as --device N).",
    )
    args, _unknown = parser.parse_known_args(argv)
    if args.device is not None:
        return args.device
    return args.gpu


def _parse_device_spec(spec):
    raw = str(spec).strip()
    if raw == "":
        return None
    low = raw.lower()
    if low == "cpu":
        return "cpu"
    if low in ("cuda", "gpu"):
        return "cuda:0"
    if low.startswith("cuda:"):
        rest = low.split(":", 1)[1]
        if not rest.isdigit():
            raise ValueError(f"Invalid device spec: {spec!r}")
        return f"cuda:{int(rest)}"
    if raw.isdigit():
        return f"cuda:{int(raw)}"
    raise ValueError(f"Invalid device spec: {spec!r} (use cpu, cuda, cuda:N, or N)")


def resolve_device(device_arg=None):
    """Pick device from a CLI/argument spec, else the first visible CUDA device.

    CUDA_VISIBLE_DEVICES is the physical-GPU policy: after it is applied,
    `cuda:0` is the first visible device. Never fall back to
    `gpu_index = 1 if n_gpu > 1 else 0`.
    """
    if device_arg is not None:
        parsed = _parse_device_spec(device_arg)
        if parsed is not None:
            return parsed
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def apply_device(device_arg=None):
    """Resolve and activate the device. Returns a `cpu` / `cuda:N` string."""
    spec = resolve_device(device_arg)
    if not spec.startswith("cuda") or not torch.cuda.is_available():
        if spec.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA not available; using CPU")
        elif device_arg is None:
            print("CUDA not available; using CPU")
        else:
            print("Using CPU")
        return "cpu"
    idx = int(spec.split(":")[-1])
    n_gpu = torch.cuda.device_count()
    if idx < 0 or idx >= n_gpu:
        raise ValueError(
            f"Requested {spec} but only {n_gpu} visible GPU(s). "
            "Set CUDA_VISIBLE_DEVICES or pass --device cuda:0."
        )
    torch.cuda.set_device(idx)
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    print(
        f"Using cuda:{idx} ({torch.cuda.get_device_name(idx)}); "
        f"{n_gpu} GPU(s) visible (CUDA_VISIBLE_DEVICES={vis})"
    )
    return f"cuda:{idx}"


def ema_decay_at_step(step, decay=None, warmup=None):
    """§8 EMA warm-up: min(d, (1 + n) / (10 + n)). `step` is 0-based update count."""
    d = float(EMA_DECAY if decay is None else decay)
    use_warmup = EMA_WARMUP if warmup is None else bool(warmup)
    if not use_warmup:
        return d
    n = float(step)
    return min(d, (1.0 + n) / (10.0 + n))


def weight_decay_excluded(name, param=None):
    """True if this parameter should get AdamW weight_decay=0 (§8)."""
    if WD_EXCLUDE_1D and param is not None and param.ndim <= 1:
        return True
    key = str(name).lower()
    if WD_EXCLUDE_BIAS and (key == "bias" or key.endswith(".bias")):
        return True
    if WD_EXCLUDE_NORMS and any(
        token in key for token in ("layernorm", ".ln.", "norm")
    ):
        return True
    if WD_EXCLUDE_GATES and any(token in key for token in ("gate", "alpha_raw")):
        return True
    if any(token in key for token in WD_EXCLUDE_KEYWORDS):
        return True
    return False


def adamw_param_groups(named_params, weight_decay=None):
    """AdamW param groups: decay vs excluded (bias / LayerNorm / gates / 1-D)."""
    wd = float(WEIGHT_DECAY if weight_decay is None else weight_decay)
    decay, no_decay = [], []
    for name, param in named_params:
        if not param.requires_grad:
            continue
        if weight_decay_excluded(name, param):
            no_decay.append(param)
        else:
            decay.append(param)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": wd})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def train_val_split(dataset, val_fraction, seed):
    """Two-way split kept for existing tests. Training uses train_val_test_split."""
    n = len(dataset)
    rng = np.random.RandomState(seed)
    if n <= 1:
        idxs = list(range(n))
        return Subset(dataset, idxs), Subset(dataset, [])
    n_val = max(1, int(n * val_fraction))
    n_val = min(n_val, n - 1)
    perm = rng.permutation(n).tolist()
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def hospital_group(dataset_id):
    """Source site from an AneuX id: SNF / p / UPF / USFD / ANSYS / …"""
    text = str(dataset_id).strip()
    if not text:
        return "other"
    token = text.split("_", 1)[0]
    i = 0
    while i < len(token) and token[i].isalpha():
        i += 1
    prefix = token[:i] if i else token
    return prefix if prefix else "other"


def _split_counts(n, val_fraction, test_fraction):
    """Counts for (n_test, n_val). Leaves at least one train index when n >= 1."""
    if n <= 0:
        return 0, 0
    if n == 1:
        return 0, 0
    if n == 2:
        return 0, 1
    n_test = max(1, int(n * float(test_fraction)))
    n_val = max(1, int(n * float(val_fraction)))
    if n_test + n_val >= n:
        n_test = min(n_test, n - 2)
        n_val = min(n_val, n - 1 - n_test)
        n_test = max(0, n_test)
        n_val = max(0, n_val)
    return n_test, n_val


def _split_one_group(ids, val_fraction, test_fraction, seed):
    unique = sorted(set(ids))
    n = len(unique)
    n_test, n_val = _split_counts(n, val_fraction, test_fraction)
    rng = np.random.RandomState(int(seed))
    perm = rng.permutation(n)
    ordered = [unique[i] for i in perm]
    test_ids = ordered[:n_test]
    val_ids = ordered[n_test : n_test + n_val]
    train_ids = ordered[n_test + n_val :]
    return train_ids, val_ids, test_ids


def split_ids(ids, val_fraction, test_fraction, seed):
    """Disjoint train/val/test IDs, stratified by hospital prefix."""
    groups = {}
    for key in ids:
        groups.setdefault(hospital_group(key), []).append(key)
    rng = np.random.RandomState(int(seed))
    train_ids, val_ids, test_ids = [], [], []
    for name in sorted(groups):
        g_seed = int(rng.randint(0, 2**31 - 1))
        tr, va, te = _split_one_group(
            groups[name], val_fraction, test_fraction, g_seed
        )
        train_ids.extend(tr)
        val_ids.extend(va)
        test_ids.extend(te)
    return train_ids, val_ids, test_ids


def train_val_test_indices(n, val_fraction, test_fraction, seed):
    """Index lists for a fixed seeded three-way split of `range(n)`."""
    train_ids, val_ids, test_ids = split_ids(
        list(range(int(n))), val_fraction, test_fraction, seed
    )
    return train_ids, val_ids, test_ids


def make_split_payload(train_ids, val_ids, test_ids, seed, val_fraction, test_fraction):
    return {
        "seed": int(seed),
        "val_fraction": float(val_fraction),
        "test_fraction": float(test_fraction),
        "stratify": "hospital",
        "train": list(train_ids),
        "val": list(val_ids),
        "test": list(test_ids),
    }


def _carve_ids(ids, n_take, seed):
    ordered = sorted(ids)
    if n_take <= 0 or not ordered:
        return list(ordered), []
    n_take = min(int(n_take), max(0, len(ordered) - 1))
    rng = np.random.RandomState(int(seed))
    perm = rng.permutation(len(ordered))
    taken = [ordered[i] for i in perm[:n_take]]
    rest = [ordered[i] for i in perm[n_take:]]
    return rest, taken


def ensure_three_way_payload(payload, val_fraction, test_fraction, seed):
    """Keep a stored three-way split. Extend a two-way file by carving test from train."""
    train_ids = list(payload.get("train") or [])
    val_ids = list(payload.get("val") or [])
    test_ids = list(payload.get("test") or [])
    if test_ids:
        return make_split_payload(
            train_ids,
            val_ids,
            test_ids,
            payload.get("seed", seed),
            payload.get("val_fraction", val_fraction),
            payload.get("test_fraction", test_fraction),
        )
    n = len(train_ids) + len(val_ids)
    n_test, _n_val = _split_counts(n, val_fraction, test_fraction)
    new_train, new_test = _carve_ids(train_ids, n_test, seed)
    return make_split_payload(
        new_train, val_ids, new_test, seed, val_fraction, test_fraction
    )


def subsets_from_ids(dataset, train_ids, val_ids, test_ids):
    id_to_idx = {
        dataset.samples[i]["dataset_id"]: i for i in range(len(dataset))
    }

    def _ix(id_list):
        return [id_to_idx[key] for key in id_list if key in id_to_idx]

    return (
        Subset(dataset, _ix(train_ids)),
        Subset(dataset, _ix(val_ids)),
        Subset(dataset, _ix(test_ids)),
    )


def train_val_test_split(dataset, val_fraction, test_fraction, seed):
    """Fixed seeded train/val/test Subsets. Same IDs + seed => same partitions."""
    ids = [dataset.samples[i]["dataset_id"] for i in range(len(dataset))]
    train_ids, val_ids, test_ids = split_ids(ids, val_fraction, test_fraction, seed)
    return subsets_from_ids(dataset, train_ids, val_ids, test_ids)


def load_or_create_fixed_split(
    dataset, split_path, val_fraction, test_fraction, seed, write=True
):
    """Load a stored hospital-stratified split, or create one and write it.

    An existing file is reused only if it already has ``stratify: hospital``.
    Older random two-way/three-way files are rebuilt.
    Rank-0 should pass ``write=True``; other DDP ranks wait on the barrier
    and load with ``write=False``.
    """
    parent = os.path.dirname(split_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = None
    if os.path.isfile(split_path):
        with open(split_path, "r") as f:
            stored = json.load(f)
        if stored.get("stratify") == "hospital" and stored.get("test"):
            payload = ensure_three_way_payload(
                stored, val_fraction, test_fraction, seed
            )
    if payload is None:
        if not write:
            raise FileNotFoundError(
                f"Split file missing at {split_path}; rank 0 must write it first"
            )
        ids = [dataset.samples[i]["dataset_id"] for i in range(len(dataset))]
        train_ids, val_ids, test_ids = split_ids(
            ids, val_fraction, test_fraction, seed
        )
        payload = make_split_payload(
            train_ids, val_ids, test_ids, seed, val_fraction, test_fraction
        )
    if write:
        with open(split_path, "w") as f:
            json.dump(payload, f, indent=4)
    train_ds, val_ds, test_ds = subsets_from_ids(
        dataset, payload["train"], payload["val"], payload["test"]
    )
    return train_ds, val_ds, test_ds, payload


def collect_hparams(**overrides):
    payload = {
        "host": "pc",
        "BATCH_SIZE": BATCH_SIZE,
        "ACCUM_STEPS": ACCUM_STEPS,
        "EPOCHS": EPOCHS,
        "VAL_SPLIT": VAL_SPLIT,
        "TEST_SPLIT": TEST_SPLIT,
        "VAL_EVERY": VAL_EVERY,
        "SEED": SEED,
        "LEARNING_RATE": LEARNING_RATE,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "EMA_DECAY": EMA_DECAY,
        "LR_WARMUP_STEPS": LR_WARMUP_STEPS,
        "NUM_WORKERS": NUM_WORKERS,
        "CACHE_BUILD_WORKERS": CACHE_BUILD_WORKERS,
        "TORCH_THREADS": TORCH_THREADS,
        "PIN_MEMORY": PIN_MEMORY,
        "PREFETCH_FACTOR": PREFETCH_FACTOR,
        "PRELOAD_RAM": PRELOAD_RAM,
        "USE_GRADIENT_CHECKPOINTING": USE_GRADIENT_CHECKPOINTING,
        "LATENT_DIM": LATENT_DIM,
        "LATENT_LEN": LATENT_LEN,
        "N_TRUE": N_TRUE,
        "GRAD_CLIP": GRAD_CLIP,
        "GECO_ETA": GECO_ETA,
        "GECO_BETA_INIT": GECO_BETA_INIT,
        "RATE_TARGET_NATS": RATE_TARGET_NATS,
        "GLOBAL_BATCH": int(BATCH_SIZE) * int(ACCUM_STEPS),
        "LOSS_WEIGHTS": dict(LOSS_WEIGHTS),
        "OUTPUT_DIR": OUTPUT_DIR,
        "CACHE_DIR": CACHE_DIR,
        "CLEANDATA_ROOT": CLEANDATA_ROOT,
    }
    payload.update(overrides)
    return payload


def run_stage2_training(
    *,
    output_dir=None,
    cache_dir=None,
    cleandata_root=None,
    batch_size=None,
    accum_steps=None,
    epochs=None,
    num_workers=None,
    cache_build_workers=None,
    torch_threads=None,
    pin_memory=None,
    prefetch_factor=None,
    preload_ram=None,
    gradient_checkpointing=None,
    learning_rate=None,
    weight_decay=None,
    ema_decay=None,
    lr_warmup_steps=None,
    val_every=None,
    device=None,
    resume="auto",
    run_name_host="pc",
    extra_meta=None,
    topk_train=3,
    topk_val=3,
    run_dir=None,
):
    """Shared Stage-2 launch used by aneuxai.py (PC) and aneuxai_HPC.py."""
    output_dir = os.path.abspath(output_dir or OUTPUT_DIR)
    cache_dir = os.path.abspath(cache_dir or CACHE_DIR)
    cleandata_root = os.path.abspath(cleandata_root or CLEANDATA_ROOT)
    batch_size = BATCH_SIZE if batch_size is None else int(batch_size)
    accum_steps = ACCUM_STEPS if accum_steps is None else int(accum_steps)
    epochs = EPOCHS if epochs is None else int(epochs)
    num_workers = NUM_WORKERS if num_workers is None else int(num_workers)
    cache_build_workers = (
        CACHE_BUILD_WORKERS if cache_build_workers is None else int(cache_build_workers)
    )
    torch_threads = TORCH_THREADS if torch_threads is None else int(torch_threads)
    pin_memory = PIN_MEMORY if pin_memory is None else bool(pin_memory)
    prefetch_factor = PREFETCH_FACTOR if prefetch_factor is None else int(prefetch_factor)
    preload_ram = PRELOAD_RAM if preload_ram is None else bool(preload_ram)
    ckpt_mode = normalize_gradient_checkpointing(
        USE_GRADIENT_CHECKPOINTING if gradient_checkpointing is None else gradient_checkpointing
    )
    learning_rate = LEARNING_RATE if learning_rate is None else float(learning_rate)
    weight_decay = WEIGHT_DECAY if weight_decay is None else float(weight_decay)
    ema_decay = EMA_DECAY if ema_decay is None else float(ema_decay)
    lr_warmup_steps = LR_WARMUP_STEPS if lr_warmup_steps is None else int(lr_warmup_steps)
    val_every = VAL_EVERY if val_every is None else int(val_every)

    probe_only = os.environ.get("ANEUX_VRAM_PROBE_ONLY", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if probe_only:
        epochs = 1
        num_workers = 0
        pin_memory = False
        prefetch_factor = 1
        preload_ram = False
        resume = False
        write_plots_flag = False
        output_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "scratch", "vram_aneuxai_probe")
        )
        if extra_meta is None:
            extra_meta = {}
        extra_meta = dict(extra_meta)
        extra_meta["vram_probe_only"] = True
    else:
        write_plots_flag = True

    seed_everything(SEED)
    configure_stage2_precision()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.set_num_threads(max(1, torch_threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, torch_threads)))
    os.environ.setdefault("MKL_NUM_THREADS", str(max(1, torch_threads)))

    resolved_device = apply_device(device if device is not None else parse_device_args())
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    job_id = os.environ.get("SLURM_JOB_ID")
    main = is_main_process()
    if run_dir:
        run_dir = os.path.abspath(run_dir)
        os.makedirs(os.path.join(run_dir, "data"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    elif main:
        run_dir = make_run_dir(output_dir, job_id=job_id)
    else:
        run_dir = None
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            packed = [run_dir]
            dist.broadcast_object_list(packed, src=0)
            run_dir = packed[0]
    except Exception:
        pass
    if not run_dir:
        run_dir = make_run_dir(output_dir, job_id=job_id)
    if main:
        print(f"Run directory: {run_dir}")

    if main:
        print("Initializing dataset...")
    dataset = AneurysmDataset(
        tube_radius=TUBE_RADIUS,
        n_length=N_LENGTH,
        n_radial=N_RADIAL,
        cache_dir=cache_dir,
        n_true=N_TRUE,
        cleandata_root=cleandata_root,
        require_templates=True,
        ensure_derived=ENSURE_DERIVED,
    )
    if main:
        print(f"Dataset loaded. Total samples: {len(dataset)}")

    split_path = os.path.join(output_dir, SPLIT_JSON_NAME)
    if probe_only:
        n_need = max(int(batch_size) * 2, 4)
        if main:
            print(
                f"ANEUX_VRAM_PROBE_ONLY: building at most {n_need} tube caches "
                f"(skipping samples that fail), then one batch"
            )
        ready = []
        for i in range(len(dataset)):
            if dataset._cache_file_ready(i):
                ready.append(i)
            else:
                sample_id = dataset.samples[i]["dataset_id"]
                try:
                    dataset._write_cache(i)
                    ready.append(i)
                    if main:
                        print(f"  cache built {sample_id}")
                except Exception as exc:
                    if main:
                        print(f"  skip {sample_id}: {exc}")
            if len(ready) >= n_need:
                break
        if len(ready) < int(batch_size):
            raise RuntimeError(
                f"VRAM probe needs {batch_size} cache-ready samples; got {len(ready)}. "
                "Need complete template_mesh + original_centerline with GroupIds."
            )
        dataset.samples = [dataset.samples[i] for i in ready]
        train_dataset = Subset(dataset, list(range(len(dataset))))
        val_dataset = Subset(dataset, [])
        test_dataset = Subset(dataset, [])
        split_payload = {
            "seed": int(SEED),
            "probe_only": True,
            "train": [s["dataset_id"] for s in dataset.samples],
            "val": [],
            "test": [],
        }
        if main:
            dump_json(os.path.join(run_dir, "data", "train_val_split.json"), split_payload)
            print(
                f"Probe subset: {len(train_dataset)} train graphs "
                f"(batch_size={batch_size})"
            )
        barrier()
    else:
        if main:
            train_dataset, val_dataset, test_dataset, split_payload = load_or_create_fixed_split(
                dataset, split_path, VAL_SPLIT, TEST_SPLIT, SEED, write=True
            )
            dump_json(os.path.join(run_dir, "data", "train_val_split.json"), split_payload)
            print(
                f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | "
                f"Test: {len(test_dataset)} (held-out, seed={split_payload['seed']})"
            )
            print(f"Saved train/val/test split to {split_path}")
        barrier()
        if not main:
            train_dataset, val_dataset, test_dataset, split_payload = load_or_create_fixed_split(
                dataset, split_path, VAL_SPLIT, TEST_SPLIT, SEED, write=False
            )
        if main:
            print("Warming tube cache (parallel raycast; not used as DataLoader workers)...")
            n_cached = dataset.warmup_cache(num_workers=cache_build_workers)
            print(f"Tube cache ready for {n_cached} samples")
        barrier()
    if preload_ram:
        n_ram = dataset.preload_ram()
        if main:
            print(f"RAM preload: {n_ram} graphs")

    if len(dataset) > 0:
        sample_data = dataset[0]
        print(f"Sample X_true shape: {sample_data.x_true.shape}")
        print(f"Sample X_tube (fine) shape: {sample_data.x.shape}")
        print(f"Sample mid shape: {sample_data.pos_mid.shape}")
        print(f"Sample coarse shape: {sample_data.pos_coarse.shape}")
        print(f"Sample Edge Index shape: {sample_data.edge_index.shape}")
        print(f"Sample face shape: {sample_data.face.shape}")
        print(
            f"Latent tokens: {sample_data.latent_pos.shape}  "
            f"n_tracts={int(sample_data.n_tracts)}"
        )

    print("Initializing Graph VAE model...")
    model = GraphVAE(
        latent_dim=LATENT_DIM,
        latent_len=LATENT_LEN,
        hidden_dim=DECODER_HIDDEN_DIM,
        tube_radius=TUBE_RADIUS,
        gradient_checkpointing=ckpt_mode,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {n_params:,}")
    print(f"Gradient checkpointing: {ckpt_mode}")

    meta = collect_hparams(
        host=run_name_host,
        BATCH_SIZE=batch_size,
        ACCUM_STEPS=accum_steps,
        EPOCHS=epochs,
        NUM_WORKERS=num_workers,
        CACHE_BUILD_WORKERS=cache_build_workers,
        TORCH_THREADS=torch_threads,
        PIN_MEMORY=pin_memory,
        PREFETCH_FACTOR=prefetch_factor,
        PRELOAD_RAM=preload_ram,
        USE_GRADIENT_CHECKPOINTING=ckpt_mode,
        LEARNING_RATE=learning_rate,
        WEIGHT_DECAY=weight_decay,
        EMA_DECAY=ema_decay,
        LR_WARMUP_STEPS=lr_warmup_steps,
        VAL_EVERY=val_every,
        OUTPUT_DIR=output_dir,
        CACHE_DIR=cache_dir,
        CLEANDATA_ROOT=cleandata_root,
        RUN_DIR=run_dir,
        n_params=n_params,
        n_train=len(train_dataset),
        n_val=len(val_dataset),
        n_test=len(test_dataset),
        device=resolved_device,
    )
    if extra_meta:
        meta.update(extra_meta)
    if main:
        dump_json(os.path.join(run_dir, "data", "run_config.json"), meta)
        dump_json(os.path.join(run_dir, "data", "hardware.json"), hardware_snapshot())

    print(f"Starting training on {resolved_device}...")
    trained_model, history = None, []
    if len(dataset) > 0:
        trained_model, history = train_model(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=epochs,
            batch_size=batch_size,
            lr=learning_rate,
            weights=LOSS_WEIGHTS,
            device=resolved_device,
            accum_steps=accum_steps,
            val_every=val_every,
            num_workers=num_workers,
            ckpt_dir=run_dir,
            grad_clip=GRAD_CLIP,
            weight_decay=weight_decay,
            kl_max=LAMBDA_KL,
            kl_warmup_epochs=KL_WARMUP_EPOCHS,
            ema_decay=ema_decay,
            resume=resume,
            lr_warmup_steps=lr_warmup_steps,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            topk_train=topk_train,
            topk_val=topk_val,
            write_plots=write_plots_flag,
            run_meta=meta,
        )
        print("Training complete.")
        print(f"Artifacts: {run_dir}")
    else:
        print("No samples found in cleandata/. Fill uniformly_remeshed, original_centerline, and template_mesh first.")
    return trained_model, history, run_dir


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass
    run_stage2_training()
