# %% [markdown]
# # 3D Graph Variational Autoencoder for Aneurysm Mesh Deformation
# Phase 1: Autoencoder Proof-of-Concept

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

from dataset import AneurysmDataset
from model import GraphVAE
from train import train_model

torch.sparse.check_sparse_tensor_invariants.enable()

# %% [markdown]
# ## 1. Configuration & Hyperparameters

# %%
OUTPUT_DIR = EXPERIMENT_OUTPUT
CACHE_DIR = EXPERIMENT_CACHE
VESSEL_DIR = VESSELS_AREA005
CENTERLINE_DIR = CENTERLINES
EXTRA_CENTERLINE_DIR = EXTRA_CENTERLINES

TUBE_RADIUS = 2.0
N_LENGTH = 1000
N_RADIAL = 50

BATCH_SIZE = 1
ACCUM_STEPS = 8
EPOCHS = 100
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.15
VAL_EVERY = 5
SEED = 31

CPU_AFFINITY = [1, 2, 3]
NUM_WORKERS = 2

LOSS_WEIGHTS = {
    "recon": 1.0,
    "kl": 0.001,
    "geom": 0.1,
}

LATENT_DIM = 256
HIDDEN_DIM = 128
K_NEIGHBORS = 32


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
        n_val = int(len(idxs) * val_fraction)
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])

    if not train_idx or not val_idx:
        n = len(dataset)
        n_val = max(1, int(n * val_fraction)) if n > 1 else 0
        perm = rng.permutation(n).tolist()
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


# %% [markdown]
# ## 2. Data Preparation
# 1. Parse clinical.csv and filter ICA locations.
# 2. Match vessel and centerline `.vtp` files.
# 3. Generate (and cache) B-spline tube scaffolds.

# %%
if __name__ == "__main__":
    seed_everything(SEED)

    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, CPU_AFFINITY)
        torch.set_num_threads(len(CPU_AFFINITY))

    DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
    if DEVICE.startswith("cuda"):
        torch.cuda.set_device(int(DEVICE.split(":")[1]))

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
    )

    print(f"Dataset loaded. Total paired and filtered samples: {len(dataset)}")

    train_dataset, val_dataset = stratified_split(dataset, VAL_SPLIT, SEED)
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    train_ids = [dataset.samples[i]["dataset_id"] for i in train_dataset.indices]
    val_ids = [dataset.samples[i]["dataset_id"] for i in val_dataset.indices]
    split_path = os.path.join(OUTPUT_DIR, "train_val_split.json")
    with open(split_path, "w") as f:
        json.dump({"train": train_ids, "val": val_ids}, f, indent=4)
    print(f"Saved train/val split to {split_path}")

    if len(dataset) > 0:
        sample_data = dataset[0]
        print(f"Sample X_true shape: {sample_data.x_true.shape}")
        print(f"Sample X_tube shape: {sample_data.x.shape}")
        print(f"Sample Edge Index shape: {sample_data.edge_index.shape}")
        print(f"Sample face shape: {sample_data.face.shape}")

    # %% [markdown]
    # ## 3. Model Initialization
    # PointTransformer encoder on the vessel point cloud, SplineConv decoder
    # predicting Δx on the centerline tube.

    # %%
    print("Initializing Graph VAE model...")
    model = GraphVAE(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, k=K_NEIGHBORS)
    print(model)

    # %% [markdown]
    # ## 4. Training Loop
    # Chamfer Distance, KL Divergence, edge-length penalty, Laplacian smoothing.

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
        val_recon = [h["val_recon"] for h in history if "val_recon" in h]
        val_kl = [h["val_kl"] for h in history if "val_kl" in h]
        val_geom = [h["val_geom"] for h in history if "val_geom" in h]

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(epochs_range, train_loss, label="Train Total Loss")
        if val_loss:
            plt.plot(val_epochs, val_loss, "ro-", label="Val Total Loss")
        plt.title("Total Loss")
        plt.xlabel("Epoch")
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(epochs_range, [h["recon"] for h in history], label="Train Recon")
        if val_recon:
            plt.plot(val_epochs, val_recon, "ro-", label="Val Recon")
        plt.title("Reconstruction Loss (Chamfer)")
        plt.xlabel("Epoch")
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(epochs_range, [h["kl"] for h in history], label="Train KL")
        if val_kl:
            plt.plot(val_epochs, val_kl, "ro-", label="Val KL")
        plt.title("KL Divergence")
        plt.xlabel("Epoch")
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(epochs_range, [h["geom"] for h in history], label="Train Geom")
        if val_geom:
            plt.plot(val_epochs, val_geom, "ro-", label="Val Geom")
        plt.title("Geometric Regularization")
        plt.xlabel("Epoch")
        plt.legend()

        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, "training_history.png")
        plt.savefig(plot_path)
        print(f"Training history plot saved as {plot_path}")
        plt.show()
