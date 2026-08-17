# %% 
# ==========================================
# IMPORTS & CONFIGURATION
# ==========================================
import torch
import numpy as np
import pandas as pd
import json
import os
import pyvista as pv
from tqdm import tqdm
from torch_geometric.loader import DataLoader

from dataset import AneurysmDataset
from model import GraphVAE
from losses import compute_losses

# --- CONFIGURATION ---
CSV_PATH = "/home/dmiklos/aneux/rawdata/rawdata/data-v1.0/data/clinical.csv"
VESSEL_DIR = "/home/dmiklos/aneux/rawdata/rawdata/models-v1.0/models/vessels/remeshed/area-005"
CENTERLINE_DIR = "/home/dmiklos/aneux/rawdata/rawdata/models-v1.0/models/centerlines"
EXTRA_CENTERLINE_DIR = "/home/dmiklos/aneux/code/test_1stage_encoder_decoder_only/centerlines"
TUBE_RADIUS = 2.0
N_LENGTH = 1000
N_RADIAL = 50

# Model parameters
LATENT_DIM = 256
HIDDEN_DIM = 128
K_NEIGHBORS = 32

DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
if 'cuda:1' in DEVICE:
    torch.cuda.set_device(1)

OUTPUT_DIR = "/home/dmiklos/aneux/code/test_1stage_encoder_decoder_only/output"
SPLIT_FILE = os.path.join(OUTPUT_DIR, "train_val_split.json")
MODEL_FILE = os.path.join(OUTPUT_DIR, "graph_vae_aneurysm.pth")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "per_patient_losses.csv")

LOSS_WEIGHTS = {'recon': 1.0, 'kl': 1, 'geom': 1}

# Opt-in to sparse tensor invariant checks globally to guarantee memory safety 
torch.sparse.check_sparse_tensor_invariants.enable()


# %% 
# ==========================================
# CORE POST-PROCESSING FUNCTIONS
# ==========================================
def tensor_to_vtp(x_pred_tensor, original_faces, output_filepath):
    """
    Converts predicted continuous coordinates from the PyTorch GNN into a 
    physics-ready, watertight .vtp surface mesh for CFD.
    
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
        extra_centerline_dir=EXTRA_CENTERLINE_DIR
    )
    
    print(f"Loading split from {SPLIT_FILE}...")
    with open(SPLIT_FILE, 'r') as f:
        split_data = json.load(f)
    train_ids = set(split_data.get('train', []))
    val_ids = set(split_data.get('val', []))
    
    print("Initializing model...")
    model = GraphVAE(
        latent_dim=LATENT_DIM, 
        hidden_dim=HIDDEN_DIM, 
        k=K_NEIGHBORS
    ).to(DEVICE)
    
    print(f"Loading model weights from {MODEL_FILE}...")
    model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
    model.eval()
    
# Use num_workers=0 for stability during post-processing
    loader = DataLoader(dataset, batch_size=1, shuffle=False, follow_batch=['x_true'], num_workers=0)
    
    results = []
    
    print("Evaluating samples...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            batch = batch.to(DEVICE)
            
            # Find the patient ID for this sample
            patient_id = dataset.samples[i]['dataset_id']
            split = 'train' if patient_id in train_ids else ('val' if patient_id in val_ids else 'unknown')
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                x_pred, mu, logvar = model(batch)
                
                loss_recon, loss_kl, loss_geom = compute_losses(
                    x_pred, batch.x_true, mu, logvar, batch.x, batch.edge_index, batch.x_true_batch, batch.num_graphs,
                    batch.faces if hasattr(batch, 'faces') and batch.faces is not None else None
                )
                
                total_loss = LOSS_WEIGHTS['recon'] * loss_recon + LOSS_WEIGHTS['kl'] * loss_kl + LOSS_WEIGHTS['geom'] * loss_geom
            
            results.append({
                'patient_id': patient_id,
                'split': split,
                'total_loss': total_loss.item(),
                'recon_loss': loss_recon.item(),
                'kl_loss': loss_kl.item(),
                'geom_loss': loss_geom.item()
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
        extra_centerline_dir=EXTRA_CENTERLINE_DIR
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
        hidden_dim=HIDDEN_DIM, 
        k=K_NEIGHBORS
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
    model.eval()
    
    print(f"Processing sample {target_patient_id}...")
    data = dataset[target_idx]
    
    # Use PyG DataLoader to get proper batching structures (like batch_x_true)
    loader = DataLoader([data], batch_size=1, follow_batch=['x_true'])
    batch = next(iter(loader)).to(DEVICE)
    
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            x_pred, _, _ = model(batch)
            
    print(f"Creating VTP file: {output_filename}")
    # Extract the original faces (PyG stores them as a tensor)
    faces_np = data.faces.numpy() if isinstance(data.faces, torch.Tensor) else data.faces
    
    tensor_to_vtp(x_pred, faces_np, output_filename)
    print("Done!")

# Example usage (Uncomment and change patient ID to run):
generate_vtp_for_sample("SNF00000419")
