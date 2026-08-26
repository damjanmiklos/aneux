# %% [markdown]
# # Hierarchical PointNeXt–SplineConv VAE for Aneurysm Mesh Deformation
# Stage 2 geometry autoencoder: tree-valued centerline latent + progressive tube decoder.

# %%
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Subset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from aneux_paths import (
    CSV_PATH,
    VESSELS_AREA005,
    CENTERLINES,
    EXTRA_CENTERLINES,
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
    LEARNING_RATE,
    N_TRUE,
    TUBE_RADIUS_MM,
    WEIGHT_DECAY,
    configure_stage2_precision,
)
from dataset import AneurysmDataset
from model import GraphVAE
from train import train_model

# %% [markdown]
# ## 1. Configuration & Hyperparameters

# %%
OUTPUT_DIR = EXPERIMENT_OUTPUT
CACHE_DIR = EXPERIMENT_CACHE
VESSEL_DIR = VESSELS_AREA005
CENTERLINE_DIR = CENTERLINES
EXTRA_CENTERLINE_DIR = EXTRA_CENTERLINES

TUBE_RADIUS = TUBE_RADIUS_MM
N_LENGTH = HIERARCHY_LEVELS[-1][0]
N_RADIAL = HIERARCHY_LEVELS[-1][1]

BATCH_SIZE = 1
ACCUM_STEPS = 8
EPOCHS = 200
VAL_SPLIT = 0.15
VAL_EVERY = 5
SEED = 31

NUM_WORKERS = 2
CACHE_BUILD_WORKERS = 8
TORCH_THREADS = 4

LOSS_WEIGHTS = dict(DEFAULT_LOSS_WEIGHTS)

# %% [markdown]
# ## 2. Data Preparation
# ICA-filtered paired vessel/centerline meshes, unique-tract Bishop tubes,
# hybrid far-from-centerline FPS for x_true, canonical ICA pose (mm preserved).

# %%
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_split(dataset, val_fraction, seed):
    """Split indices by clinical location so each ICA site appears in both sets when possible."""
    by_loc = defaultdict(list)
    for i, sample in enumerate(dataset.samples):
        by_loc[sample.get("location", "unknown")].append(i)

    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []
    for idxs in by_loc.values():
        idxs = list(idxs)
        rng.shuffle(idxs)
        if len(idxs) >= 2:
            n_val = max(1, int(len(idxs) * val_fraction))
            n_val = min(n_val, len(idxs) - 1)
        else:
            n_val = 0
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])

    if not train_idx or not val_idx:
        n = len(dataset)
        n_val = max(1, int(n * val_fraction)) if n > 1 else 0
        perm = rng.permutation(n).tolist()
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


if __name__ == "__main__":
    seed_everything(SEED)
    configure_stage2_precision()

    torch.set_num_threads(TORCH_THREADS)

    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        gpu_index = 1 if n_gpu > 1 else 0
        DEVICE = f"cuda:{gpu_index}"
        torch.cuda.set_device(gpu_index)
        print(f"Using {DEVICE} ({torch.cuda.get_device_name(gpu_index)}); {n_gpu} GPU(s) visible")
    else:
        DEVICE = "cpu"
        print("CUDA not available; using CPU")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Initializing dataset...")
    dataset = AneurysmDataset(
        csv_path=CSV_PATH,
        vtp_vessel_dir=VESSEL_DIR,
        vtp_centerline_dir=CENTERLINE_DIR,
        tube_radius=TUBE_RADIUS,
        n_length=N_LENGTH,
        n_radial=N_RADIAL,
        extra_centerline_dir=EXTRA_CENTERLINE_DIR,
        cache_dir=CACHE_DIR,
        n_true=N_TRUE,
    )

    print(f"Dataset loaded. Total paired and filtered samples: {len(dataset)}")

    train_dataset, val_dataset = stratified_split(dataset, VAL_SPLIT, SEED)
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    print("Warming tube cache (parallel raycast; not used as DataLoader workers)...")
    n_cached = dataset.warmup_cache(num_workers=CACHE_BUILD_WORKERS)
    print(f"Tube cache ready for {n_cached} samples")

    train_ids = [dataset.samples[i]["dataset_id"] for i in train_dataset.indices]
    val_ids = [dataset.samples[i]["dataset_id"] for i in val_dataset.indices]
    split_path = os.path.join(OUTPUT_DIR, "train_val_split.json")
    with open(split_path, "w") as f:
        json.dump({"train": train_ids, "val": val_ids}, f, indent=4)
    print(f"Saved train/val split to {split_path}")

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
    model = GraphVAE(
        latent_dim=LATENT_DIM,
        latent_len=LATENT_LEN,
        hidden_dim=DECODER_HIDDEN_DIM,
        tube_radius=TUBE_RADIUS,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {n_params:,}")

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
        )
        print("Training complete.")
    else:
        print("No samples found! Please verify data paths and locations in the CSV.")
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
