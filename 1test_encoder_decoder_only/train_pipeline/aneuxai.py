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
    GRAD_CLIP,
    HIERARCHY_LEVELS,
    KL_WARMUP_EPOCHS,
    LAMBDA_KL,
    LATENT_DIM,
    LATENT_LEN,
    N_TRUE,
    TUBE_RADIUS_MM,
    configure_stage2_precision,
    normalize_gradient_checkpointing,
)
from dataset import AneurysmDataset
from model import GraphVAE
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

# Real batch 4–8 fits (0.74 GiB/sample). Keep gradient accumulation as 1×8.
BATCH_SIZE = 4
ACCUM_STEPS = 8
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

NUM_WORKERS = 3
CACHE_BUILD_WORKERS = 8
TORCH_THREADS = 6

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
    dataset, split_path, val_fraction, test_fraction, seed
):
    """Load a stored hospital-stratified split, or create one and write it.

    An existing file is reused only if it already has ``stratify: hospital``.
    Older random two-way/three-way files are rebuilt.
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
        ids = [dataset.samples[i]["dataset_id"] for i in range(len(dataset))]
        train_ids, val_ids, test_ids = split_ids(
            ids, val_fraction, test_fraction, seed
        )
        payload = make_split_payload(
            train_ids, val_ids, test_ids, seed, val_fraction, test_fraction
        )
    with open(split_path, "w") as f:
        json.dump(payload, f, indent=4)
    train_ds, val_ds, test_ds = subsets_from_ids(
        dataset, payload["train"], payload["val"], payload["test"]
    )
    return train_ds, val_ds, test_ds, payload


if __name__ == "__main__":
    seed_everything(SEED)
    configure_stage2_precision()

    torch.set_num_threads(TORCH_THREADS)

    DEVICE = apply_device(parse_device_args())

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Initializing dataset...")
    dataset = AneurysmDataset(
        tube_radius=TUBE_RADIUS,
        n_length=N_LENGTH,
        n_radial=N_RADIAL,
        cache_dir=CACHE_DIR,
        n_true=N_TRUE,
        cleandata_root=CLEANDATA_ROOT,
        require_templates=True,
        ensure_derived=ENSURE_DERIVED,
    )

    print(f"Dataset loaded. Total samples: {len(dataset)}")

    split_path = os.path.join(OUTPUT_DIR, SPLIT_JSON_NAME)
    train_dataset, val_dataset, test_dataset, split_payload = load_or_create_fixed_split(
        dataset, split_path, VAL_SPLIT, TEST_SPLIT, SEED
    )
    print(
        f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | "
        f"Test: {len(test_dataset)} (held-out, seed={split_payload['seed']})"
    )
    print(f"Saved train/val/test split to {split_path}")
    print("Warming tube cache (parallel raycast; not used as DataLoader workers)...")
    n_cached = dataset.warmup_cache(num_workers=CACHE_BUILD_WORKERS)
    print(f"Tube cache ready for {n_cached} samples")

    if len(dataset) > 0:
        sample_data = dataset[0]
        print(f"Sample X_true shape: {sample_data.x_true.shape}")
        print(f"Sample X_tube (fine) shape: {sample_data.x.shape}")
        print(f"Sample mid shape: {sample_data.pos_mid.shape}")
        print(f"Sample coarse shape: {sample_data.pos_coarse.shape}")
        print(f"Sample Edge Index shape: {sample_data.edge_index.shape}")
        print(f"Sample face shape: {sample_data.face.shape}")
        print(f"Latent tokens: {sample_data.latent_pos.shape}  n_tracts={int(sample_data.n_tracts)}")
        print(f"pose_R: {tuple(sample_data.pose_R.shape)}  origin: {tuple(sample_data.origin_shift.shape)}")
        n_r = int(sample_data.r_star.numel())
        n_ok = int(sample_data.r_star_valid.sum().item()) if n_r else 0
        n_amb = int(getattr(sample_data, "r_star_ambiguous", torch.zeros(0, dtype=torch.bool)).sum().item()) if n_r else 0
        print(f"r_star valid: {n_ok}/{n_r} ({(n_ok / max(n_r, 1)):.3f})  ambiguous: {n_amb}")
        print(
            f"dtypes: x={sample_data.x.dtype} x_true={sample_data.x_true.dtype} "
            f"latent_pos={sample_data.latent_pos.dtype} "
            f"matmul={torch.get_float32_matmul_precision()} "
            f"tf32={torch.backends.cuda.matmul.allow_tf32}"
        )

    # %% [markdown]
    # ## 3. Model Initialization
    # PointNeXt encoder → tree latent Z ∈ R^{96×128}; progressive SplineConv decoder (Stage 2).

    # %%
    print("Initializing Graph VAE model...")
    ckpt_mode = normalize_gradient_checkpointing(USE_GRADIENT_CHECKPOINTING)
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

    # %% [markdown]
    # ## 4. Training Loop
    # Multi-scale Chamfer, sequence KL (annealed), displacement Dirichlet,
    # Laplacian smoothing, normal consistency. AdamW + cosine decay.

    # %%
    print(f"Starting training on {DEVICE}...")
    if len(dataset) > 0:
        trained_model, history = train_model(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            weights=LOSS_WEIGHTS,
            device=DEVICE,
            accum_steps=ACCUM_STEPS,
            val_every=VAL_EVERY,
            num_workers=NUM_WORKERS,
            ckpt_dir=OUTPUT_DIR,
            grad_clip=GRAD_CLIP,
            weight_decay=WEIGHT_DECAY,
            kl_max=LAMBDA_KL,
            kl_warmup_epochs=KL_WARMUP_EPOCHS,
            ema_decay=EMA_DECAY,
        )
        print("Training complete.")
    else:
        print("No samples found in cleandata/. Fill the five .vtp folders first.")
        trained_model, history = None, []

    # %% [markdown]
    # ## 5. Save the Model 
    # %%
    if trained_model is not None:
        model_path = os.path.join(OUTPUT_DIR, "graph_vae_aneurysm.pth")
        torch.save(trained_model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        history_path = os.path.join(OUTPUT_DIR, "history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"History saved to {history_path}")

        # %% [markdown]
        # ## 6. Plot Training History

        # %%
        import matplotlib.pyplot as plt

        epochs_range = range(1, len(history) + 1)
        train_loss = [h["loss"] for h in history]
        val_epochs = [i + 1 for i, h in enumerate(history) if "val_loss" in h]
        val_loss = [h["val_loss"] for h in history if "val_loss" in h]

        plt.figure(figsize=(14, 14))

        plt.subplot(3, 3, 1)
        plt.plot(epochs_range, train_loss, label="Train Total")
        if val_loss:
            plt.plot(val_epochs, val_loss, "ro-", label="Val Total")
        plt.title("Total Loss")
        plt.xlabel("Epoch")
        plt.legend()

        plt.subplot(3, 3, 2)
        plt.plot(epochs_range, [h["recon"] for h in history], label="Train Recon")
        plt.plot(val_epochs, [h["val_recon"] for h in history if "val_recon" in h], "ro-", label="Val Recon")
        plt.title("Reconstruction (Chamfer)")
        plt.xlabel("Epoch")
        plt.legend()

        plt.subplot(3, 3, 3)
        plt.plot(epochs_range, [h.get("rad", 0.0) for h in history], label="Train Rad")
        plt.plot(val_epochs, [h["val_rad"] for h in history if "val_rad" in h], "ro-", label="Val Rad")
        plt.title("Radial Huber")
        plt.xlabel("Epoch")
        plt.legend()

        plt.subplot(3, 3, 4)
        plt.plot(epochs_range, [h["kl"] for h in history], label="Train KL")
        plt.plot(val_epochs, [h["val_kl"] for h in history if "val_kl" in h], "ro-", label="Val KL")
        plt.title("KL Divergence")
        plt.xlabel("Epoch")
        plt.legend()

        plt.subplot(3, 3, 5)
        plt.plot(epochs_range, [h["disp"] for h in history], label="Train Disp")
        plt.plot(val_epochs, [h["val_disp"] for h in history if "val_disp" in h], "ro-", label="Val Disp")
        plt.title("Displacement Dirichlet")
        plt.xlabel("Epoch")
        plt.legend()

        plt.subplot(3, 3, 6)
        plt.plot(epochs_range, [h["lap"] for h in history], label="Train Lap")
        plt.plot(val_epochs, [h["val_lap"] for h in history if "val_lap" in h], "ro-", label="Val Lap")
        plt.title("Laplacian")
        plt.xlabel("Epoch")
        plt.legend()

        plt.subplot(3, 3, 7)
        plt.plot(epochs_range, [h["norm"] for h in history], label="Train Norm")
        plt.plot(val_epochs, [h["val_norm"] for h in history if "val_norm" in h], "ro-", label="Val Norm")
        plt.title("Normal Consistency")
        plt.xlabel("Epoch")
        plt.legend()

        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, "training_history.png")
        plt.savefig(plot_path)
        print(f"Training history plot saved as {plot_path}")
        plt.show()
