# %% [markdown]
# # 3D Graph Variational Autoencoder for Aneurysm Mesh Deformation
# Phase 1: Autoencoder Proof-of-Concept

# %%
import torch
import os
from dataset import AneurysmDataset

# Opt-in to sparse tensor invariant checks globally to guarantee memory safety 
# and explicitly resolve PyTorch warnings.
torch.sparse.check_sparse_tensor_invariants.enable()
from model import GraphVAE
from train import train_model

# %% [markdown]
# ## 1. Configuration & Hyperparameters
# We set up paths to the user's specific dataset and define hyperparameters for the VAE.

# %%
CSV_PATH = "/home/dmiklos/aneux/rawdata/rawdata/data-v1.0/data/clinical.csv"
VESSEL_DIR = "/home/dmiklos/aneux/rawdata/rawdata/models-v1.0/models/vessels/remeshed/area-005"
CENTERLINE_DIR = "/home/dmiklos/aneux/rawdata/rawdata/models-v1.0/models/centerlines"
EXTRA_CENTERLINE_DIR = "/home/dmiklos/aneux/code/test_1stage_encoder_decoder_only/centerlines"  # Fallback for VMTK-extracted centerlines
# Tube scaffolding parameters
TUBE_RADIUS = 2.0
N_LENGTH = 1000
N_RADIAL = 50

# Training parameters
BATCH_SIZE = 1
ACCUM_STEPS = 8  # Gradient accumulation: effective batch size = BATCH_SIZE * ACCUM_STEPS
EPOCHS = 100
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.15  # 15% of data for validation
VAL_EVERY = 5     # Evaluate validation every N epochs

# Specific CPU threads/cores the process is allowed to run on
CPU_AFFINITY = [1, 2, 3]
NUM_WORKERS = 2
os.sched_setaffinity(0, CPU_AFFINITY)
torch.set_num_threads(len(CPU_AFFINITY))

DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
if 'cuda:1' in DEVICE:
    torch.cuda.set_device(1)

# Loss weights
LOSS_WEIGHTS = {
    'recon': 1.0,
    'kl': 1,
    'geom': 1
}

# Model parameters
LATENT_DIM = 256
HIDDEN_DIM = 128
K_NEIGHBORS = 32

# %% [markdown]
# ## 2. Data Preparation
# Initialize the custom PyTorch Dataset which handles:
# 1. Parsing the clinical.csv and filtering specific ICA locations.
# 2. Matching the full vessel and centerline `.vtp` files.
# 3. Generating the analytical B-spline base tube.

# %%
if __name__ == '__main__':
    print("Initializing dataset...")
    dataset = AneurysmDataset(
        csv_path=CSV_PATH,
        vtp_vessel_dir=VESSEL_DIR,
        vtp_centerline_dir=CENTERLINE_DIR,
        tube_radius=TUBE_RADIUS,
        n_length=N_LENGTH,
        n_radial=N_RADIAL,
        extra_centerline_dir=EXTRA_CENTERLINE_DIR
    )

    print(f"Dataset loaded. Total paired and filtered samples: {len(dataset)}")

    # Train/Validation split
    from torch.utils.data import random_split
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(31))
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # Save the split to a JSON file for later analysis
    import json
    train_ids = [dataset.samples[i]['dataset_id'] for i in train_dataset.indices]
    val_ids = [dataset.samples[i]['dataset_id'] for i in val_dataset.indices]

    split_dict = {
        'train': train_ids,
        'val': val_ids
    }

    with open("train_val_split.json", "w") as f:
        json.dump(split_dict, f, indent=4)
    print("Saved train/val split to train_val_split.json")

    if len(dataset) > 0:
        sample_data = dataset[0]
        print(f"Sample X_true shape: {sample_data.x_true.shape}")
        print(f"Sample X_tube shape: {sample_data.x.shape}")
        print(f"Sample Edge Index shape: {sample_data.edge_index.shape}")

    # %% [markdown]
    # ## 3. Model Initialization
    # We instantiate the GraphVAE which consists of:
    # - A robust PointNet-style encoder for the varying-size raw meshes.
    # - A GATv2Conv MPNN decoder to predict the deformations (Delta X).

    # %%
    print("Initializing Graph VAE model...")
    model = GraphVAE(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, k=K_NEIGHBORS)
    model = model.to(DEVICE)
    print(model)

    # %% [markdown]
    # ## 4. Training Loop
    # Execute the training sequence. The loss consists of Chamfer Distance, KL Divergence,
    # and Geometric Regularizations (Edge Length Penalty & Laplacian Smoothing).

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
            num_workers=NUM_WORKERS
        )
        print("Training complete.")
    else:
        print("No samples found! Please verify data paths and locations in the CSV.")

    # %% [markdown]
    # ## 5. Save the Model
    # Save the model weights for later inference or Phase 2 evaluation.

    # %%
    if len(dataset) > 0:
        torch.save(trained_model.state_dict(), "graph_vae_aneurysm.pth")
        print("Model saved to graph_vae_aneurysm.pth")
        
        # %% [markdown]
        # ## 6. Plot Training History
        # Visualize the loss curves for training and validation.
        
        # %%
        import matplotlib.pyplot as plt
        
        epochs_range = range(1, len(history) + 1)
        train_loss = [h['loss'] for h in history]
        val_epochs = [i+1 for i, h in enumerate(history) if 'val_loss' in h]
        val_loss = [h['val_loss'] for h in history if 'val_loss' in h]
        
        plt.figure(figsize=(12, 8))
        
        # Plot Total Loss
        plt.subplot(2, 2, 1)
        plt.plot(epochs_range, train_loss, label='Train Total Loss')
        if val_loss:
            plt.plot(val_epochs, val_loss, 'ro-', label='Val Total Loss')
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        # Plot Reconstruction Loss
        train_recon = [h['recon'] for h in history]
        plt.subplot(2, 2, 2)
        plt.plot(epochs_range, train_recon, label='Train Recon')
        plt.title('Reconstruction Loss (Chamfer)')
        plt.xlabel('Epoch')
        plt.legend()
        
        # Plot KL Divergence
        train_kl = [h['kl'] for h in history]
        plt.subplot(2, 2, 3)
        plt.plot(epochs_range, train_kl, label='Train KL')
        plt.title('KL Divergence')
        plt.xlabel('Epoch')
        plt.legend()
        
        # Plot Geometric Loss
        train_geom = [h['geom'] for h in history]
        plt.subplot(2, 2, 4)
        plt.plot(epochs_range, train_geom, label='Train Geom')
        plt.title('Geometric Regularization')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("training_history.png")
        print("Training history plot saved as training_history.png")
        plt.show()
