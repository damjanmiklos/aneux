# %% 
# ==========================================
# IMPORTS & CONFIGURATION
# ==========================================
import torch
import numpy as np
import pandas as pd
import json
import os
import sys
import pyvista as pv
from tqdm import tqdm
from torch_geometric.loader import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))
sys.path.insert(0, os.path.join(_HERE, "train_pipeline"))

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
    FOLLOW_BATCH,
    HIERARCHY_LEVELS,
    LATENT_DIM,
    LATENT_LEN,
    N_TRUE,
    TUBE_RADIUS_MM,
    configure_stage2_precision,
)
from dataset import AneurysmDataset
from model import GraphVAE
from train import losses_from_output, weighted_total

# --- CONFIGURATION ---
OUTPUT_DIR = EXPERIMENT_OUTPUT
CACHE_DIR = EXPERIMENT_CACHE
VESSEL_DIR = VESSELS_AREA005
CENTERLINE_DIR = CENTERLINES
EXTRA_CENTERLINE_DIR = EXTRA_CENTERLINES
TUBE_RADIUS = TUBE_RADIUS_MM
N_LENGTH = HIERARCHY_LEVELS[-1][0]
N_RADIAL = HIERARCHY_LEVELS[-1][1]

if torch.cuda.is_available():
    n_gpu = torch.cuda.device_count()
    gpu_index = 1 if n_gpu > 1 else 0
    DEVICE = f"cuda:{gpu_index}"
    torch.cuda.set_device(gpu_index)
else:
    DEVICE = "cpu"

SPLIT_FILE = os.path.join(OUTPUT_DIR, "train_val_split.json")
MODEL_FILE = os.path.join(OUTPUT_DIR, "graph_vae_aneurysm.pth")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "per_patient_losses.csv")

LOSS_WEIGHTS = dict(DEFAULT_LOSS_WEIGHTS)

configure_stage2_precision()


# %% 
# ==========================================
# CORE POST-PROCESSING FUNCTIONS
# ==========================================
def _faces_np(data):
    """Return triangle indices as [F, 3] from PyG `face` ([3, F]) or legacy `faces`."""
    face = getattr(data, "face", None)
    if face is not None and torch.is_tensor(face) and face.numel() > 0:
        arr = face.detach().cpu().numpy()
        if arr.shape[0] == 3:
            return arr.T
        return arr
    faces = getattr(data, "faces", None)
    if faces is None:
        raise AttributeError("Data object has neither face nor faces")
    if torch.is_tensor(faces):
        return faces.detach().cpu().numpy()
    return np.asarray(faces)


def tensor_to_vtp(x_pred_tensor, original_faces, output_filepath):
    """
    Write predicted tube coordinates plus scaffold triangles to a .vtp file.
    The scaffold is a set of open branch cylinders, not a watertight CFD surface.
    
    Args:
        x_pred_tensor (torch.Tensor): Tensor of shape [N, 3] with predicted coordinates.
        original_faces (np.ndarray): Array of shape [F, 3] with triangle vertex indices.
        output_filepath (str): Destination path for the .vtp file.
    """
    # Step 1 (Precision Cast):
    # Detach from autograd graph, move to CPU, strictly cast to float64, and convert to numpy.
    vertices = x_pred_tensor.detach().cpu().to(torch.float64).numpy()
    
    # Step 2 (VTK Face Padding):
    # PyVista requires faces array to be padded with the number of vertices per face.
    # Prepend '3' (since these are triangles) to every row.
    num_faces = original_faces.shape[0]
    padding = np.full((num_faces, 1), 3, dtype=original_faces.dtype)
    padded_faces = np.hstack((padding, original_faces))
    faces_vtk = padded_faces.flatten()
    
    # Step 3 (Mesh Construction):
    mesh = pv.PolyData(vertices, faces_vtk)
    
    # Step 4 (Normal Recomputation):
    # Compute new normals for the deformed geometry.
    mesh = mesh.compute_normals(cell_normals=True, point_normals=True, inplace=False)
    
    # Step 5 (Save):
    mesh.save(output_filepath)
    
    return mesh


# %% 
# ==========================================
# EVALUATE ALL SAMPLES & SAVE TO CSV
# ==========================================
# Run this cell to evaluate the model on the entire dataset and output per-patient losses.
def evaluate_all_samples():
    print("Loading dataset...")
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
    
    print(f"Loading split from {SPLIT_FILE}...")
    with open(SPLIT_FILE, 'r') as f:
        split_data = json.load(f)
    train_ids = set(split_data.get('train', []))
    val_ids = set(split_data.get('val', []))
    
    print("Initializing model...")
    model = GraphVAE(
        latent_dim=LATENT_DIM,
        latent_len=LATENT_LEN,
        hidden_dim=DECODER_HIDDEN_DIM,
        tube_radius=TUBE_RADIUS,
    ).to(DEVICE)
    
    print(f"Loading model weights from {MODEL_FILE}...")
    model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
    model.eval()
    
# Use num_workers=0 for stability during post-processing
    loader = DataLoader(dataset, batch_size=1, shuffle=False, follow_batch=FOLLOW_BATCH, num_workers=0)
    
    results = []
    
    print("Evaluating samples...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            batch = batch.to(DEVICE)
            
            # Find the patient ID for this sample
            patient_id = dataset.samples[i]['dataset_id']
            split = 'train' if patient_id in train_ids else ('val' if patient_id in val_ids else 'unknown')
            
            out = model(batch)
            terms = losses_from_output(out, batch)
            total_loss = weighted_total(terms, LOSS_WEIGHTS)
            
            results.append({
                'patient_id': patient_id,
                'split': split,
                'total_loss': total_loss.item(),
                'recon_loss': terms['recon'].item(),
                'kl_loss': terms['kl'].item(),
                'disp_loss': terms['disp'].item(),
                'lap_loss': terms['lap'].item(),
                'norm_loss': terms['norm'].item(),
            })
            
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"Saved per-patient losses to {RESULTS_CSV}")

#if __name__ == '__main__':
    # Run evaluation when executing the script directly
    #evaluate_all_samples()


# %%
# ==========================================
# GENERATE SINGLE VTP MESH
# ==========================================
# Run this cell to extract a specific VTP file for CFD visualization
def generate_vtp_for_sample(target_patient_id, output_filename=None):
    if output_filename is None:
        output_filename = os.path.join(OUTPUT_DIR, f"{target_patient_id}_predicted.vtp")
        
    print(f"Loading dataset to find {target_patient_id}...")
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
    
    # Find the index of the requested patient
    target_idx = -1
    for i, s in enumerate(dataset.samples):
        if s['dataset_id'] == target_patient_id:
            target_idx = i
            break
            
    if target_idx == -1:
        print(f"Patient {target_patient_id} not found in dataset!")
        return
        
    print("Initializing model...")
    model = GraphVAE(
        latent_dim=LATENT_DIM,
        latent_len=LATENT_LEN,
        hidden_dim=DECODER_HIDDEN_DIM,
        tube_radius=TUBE_RADIUS,
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
    model.eval()
    
    print(f"Processing sample {target_patient_id}...")
    data = dataset[target_idx]
    
    loader = DataLoader([data], batch_size=1, follow_batch=FOLLOW_BATCH)
    batch = next(iter(loader)).to(DEVICE)
    
    with torch.no_grad():
        out = model(batch)
        x_pred = out.x_pred
            
    print(f"Creating VTP file: {output_filename}")
    faces_np = _faces_np(data)
    origin = data.origin_shift.to(x_pred.device).reshape(1, 3)
    pose_R = getattr(data, "pose_R", None)
    x_world = x_pred
    if pose_R is not None:
        R = pose_R.to(x_pred.device).reshape(3, 3)
        x_world = x_pred @ R.t()
    tensor_to_vtp(x_world + origin, faces_np, output_filename)
    print("Done!")


if __name__ == "__main__":
    generate_vtp_for_sample("SNF00000419")
