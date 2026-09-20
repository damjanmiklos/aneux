# %% 
# ==========================================
# IMPORTS & CONFIGURATION
# ==========================================
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pyvista as pv
import vtk
from vtk.util.numpy_support import vtk_to_numpy

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_HERE, "train_pipeline"))

from aneux_paths import (
    CLEANDATA,
    EXPERIMENT_OUTPUT,
    EXPERIMENT_CACHE,
)

# Training stack is optional so mesh_validity_metrics / remesh_for_cfd can run
# in hemomesh / vmtk_env (VMTK, no PyTorch). tensor_to_vtp and the evaluate /
# generate cells still need aneurysmgnn / aneuxai_env.
try:
    import pandas as pd
    import torch
    from tqdm import tqdm
    from torch_geometric.loader import DataLoader
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

    _HAS_TRAIN_STACK = True
except ImportError:
    torch = None
    pd = None
    tqdm = None
    DataLoader = None
    DEFAULT_LOSS_WEIGHTS = {}
    DECODER_HIDDEN_DIM = None
    FOLLOW_BATCH = None
    HIERARCHY_LEVELS = None
    LATENT_DIM = None
    LATENT_LEN = None
    N_TRUE = None
    TUBE_RADIUS_MM = 2.0
    configure_stage2_precision = None
    AneurysmDataset = None
    GraphVAE = None
    losses_from_output = None
    weighted_total = None
    _HAS_TRAIN_STACK = False

# --- CONFIGURATION ---
OUTPUT_DIR = EXPERIMENT_OUTPUT
CACHE_DIR = EXPERIMENT_CACHE
CLEANDATA_ROOT = CLEANDATA
TUBE_RADIUS = TUBE_RADIUS_MM
N_LENGTH = HIERARCHY_LEVELS[-1][0] if HIERARCHY_LEVELS else None
N_RADIAL = HIERARCHY_LEVELS[-1][1] if HIERARCHY_LEVELS else None

if _HAS_TRAIN_STACK and torch.cuda.is_available():
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

if _HAS_TRAIN_STACK:
    configure_stage2_precision()

# GT remesh target is 0.15 mm (AI training density). hemoMesh CFD remesh uses
# TargetArea = 0.025 mm^2; the matching equilateral edge is the CFD default.
_CFD_TARGET_AREA_MM2 = 0.025
HEMOMESH_CFD_TARGET_AREA_MM2 = _CFD_TARGET_AREA_MM2
CFD_REMESH_EDGE_MM = float(np.sqrt(4.0 * _CFD_TARGET_AREA_MM2 / np.sqrt(3.0)))
_INTERSECTION_LEN_MIN = 1e-8
_ANGLE_EDGE_MIN = 1e-15


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


def _as_polydata(mesh):
    if isinstance(mesh, pv.PolyData):
        return mesh
    if hasattr(mesh, "GetNumberOfPoints"):
        return pv.wrap(mesh)
    raise TypeError(
        "Expected pyvista.PolyData or vtkPolyData, got "
        f"{type(mesh).__name__}"
    )


def _triangle_points_faces(mesh):
    """Triangulated vertex array [N, 3] and face indices [F, 3] in VTK cell order."""
    poly = _as_polydata(mesh)
    if int(poly.n_cells) == 0:
        pts = np.asarray(poly.points, dtype=np.float64) if poly.n_points else np.zeros((0, 3), dtype=np.float64)
        return poly, pts, np.zeros((0, 3), dtype=np.int64)
    tri = poly.triangulate()
    pts = np.asarray(tri.points, dtype=np.float64)
    cells = tri.GetPolys()
    n_cells = int(cells.GetNumberOfCells()) if hasattr(cells, "GetNumberOfCells") else int(tri.n_cells)
    if n_cells == 0:
        return tri, pts, np.zeros((0, 3), dtype=np.int64)
    conn = np.asarray(vtk_to_numpy(cells.GetConnectivityArray()), dtype=np.int64)
    offsets = np.asarray(vtk_to_numpy(cells.GetOffsetsArray()), dtype=np.int64)
    sizes = np.diff(offsets)
    if sizes.size and np.all(sizes == 3) and conn.size == 3 * sizes.size:
        faces = conn.reshape(-1, 3)
    else:
        rows = []
        for i in range(len(sizes)):
            if int(sizes[i]) != 3:
                continue
            rows.append(conn[int(offsets[i]) : int(offsets[i + 1])])
        faces = np.asarray(rows, dtype=np.int64).reshape(-1, 3) if rows else np.zeros((0, 3), dtype=np.int64)
    return tri, pts, faces


def _n_connected_components(poly):
    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(poly)
    conn.SetExtractionModeToAllRegions()
    conn.Update()
    return int(conn.GetNumberOfExtractedRegions())


def _n_boundary_loops(poly):
    """Count openings as connected components of boundary edges (no VMTK)."""
    feat = vtk.vtkFeatureEdges()
    feat.SetInputData(poly)
    feat.BoundaryEdgesOn()
    feat.FeatureEdgesOff()
    feat.NonManifoldEdgesOff()
    feat.ManifoldEdgesOff()
    feat.ColoringOff()
    feat.Update()
    edges = feat.GetOutput()
    if int(edges.GetNumberOfCells()) == 0:
        return 0
    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(edges)
    conn.SetExtractionModeToAllRegions()
    conn.Update()
    return int(conn.GetNumberOfExtractedRegions())


def _min_triangle_angle_deg(pts, faces):
    if faces.size == 0 or pts.size == 0:
        return float("nan")
    a = pts[faces[:, 0]]
    b = pts[faces[:, 1]]
    c = pts[faces[:, 2]]
    e_ab = b - a
    e_bc = c - b
    e_ca = a - c
    n_ab = np.linalg.norm(e_ab, axis=1)
    n_bc = np.linalg.norm(e_bc, axis=1)
    n_ca = np.linalg.norm(e_ca, axis=1)
    valid = (n_ab > _ANGLE_EDGE_MIN) & (n_bc > _ANGLE_EDGE_MIN) & (n_ca > _ANGLE_EDGE_MIN)
    if not np.any(valid):
        return float("nan")
    u_ab = e_ab / n_ab[:, None]
    u_bc = e_bc / n_bc[:, None]
    u_ca = e_ca / n_ca[:, None]
    cos_a = np.clip(np.sum((-u_ca) * u_ab, axis=1), -1.0, 1.0)
    cos_b = np.clip(np.sum((-u_ab) * u_bc, axis=1), -1.0, 1.0)
    cos_c = np.clip(np.sum((-u_bc) * u_ca, axis=1), -1.0, 1.0)
    angles = np.degrees(np.arccos(np.stack((cos_a, cos_b, cos_c), axis=1)))
    return float(np.min(angles[valid]))


def _edge_adjacent_faces(faces):
    n = int(len(faces))
    adjacent = [set() for _ in range(n)]
    edge_to = defaultdict(list)
    for fi, (a, b, c) in enumerate(faces):
        ia, ib, ic = int(a), int(b), int(c)
        for e in ((min(ia, ib), max(ia, ib)), (min(ib, ic), max(ib, ic)), (min(ic, ia), max(ic, ia))):
            edge_to[e].append(fi)
    for ids in edge_to.values():
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                adjacent[ids[i]].add(ids[j])
                adjacent[ids[j]].add(ids[i])
    return adjacent


def _triangles_properly_intersect(tri_a, tri_b):
    """True if two triangles meet in a segment of positive length (a real fold)."""
    p1, q1, r1 = tri_a[0].tolist(), tri_a[1].tolist(), tri_a[2].tolist()
    p2, q2, r2 = tri_b[0].tolist(), tri_b[1].tolist(), tri_b[2].tolist()
    coplanar = vtk.mutable(0)
    pt1 = [0.0, 0.0, 0.0]
    pt2 = [0.0, 0.0, 0.0]
    surface_id = [0.0, 0.0]
    hit = vtk.vtkIntersectionPolyDataFilter.TriangleTriangleIntersection(
        p1, q1, r1, p2, q2, r2, coplanar, pt1, pt2, surface_id, 1e-8
    )
    if not hit:
        return False
    dx = pt1[0] - pt2[0]
    dy = pt1[1] - pt2[1]
    dz = pt1[2] - pt2[2]
    return (dx * dx + dy * dy + dz * dz) > (_INTERSECTION_LEN_MIN * _INTERSECTION_LEN_MIN)


def _self_intersection_count(poly, pts, faces):
    n = int(len(faces))
    if n < 2:
        return 0
    adjacent = _edge_adjacent_faces(faces)
    locator = vtk.vtkCellLocator()
    locator.SetDataSet(poly)
    locator.BuildLocator()
    id_list = vtk.vtkIdList()
    count = 0
    pad = 1e-9
    for i in range(n):
        tri = pts[faces[i]]
        bmin = tri.min(axis=0)
        bmax = tri.max(axis=0)
        bounds = (
            float(bmin[0] - pad),
            float(bmax[0] + pad),
            float(bmin[1] - pad),
            float(bmax[1] + pad),
            float(bmin[2] - pad),
            float(bmax[2] + pad),
        )
        id_list.Reset()
        locator.FindCellsWithinBounds(bounds, id_list)
        for k in range(id_list.GetNumberOfIds()):
            j = int(id_list.GetId(k))
            if j <= i or j >= n or j in adjacent[i]:
                continue
            if _triangles_properly_intersect(pts[faces[i]], pts[faces[j]]):
                count += 1
    return int(count)


def mesh_validity_metrics(mesh, n_profiles=None):
    """Validation counts for a decoded surface. Does not repair the mesh.

    §11 / §12: self-intersections, boundary loops (should equal n_profiles),
    connected components, and minimum triangle angle. A folded mesh with a
    good Chamfer is a failed sample — ``valid`` is False; nothing is remeshed
    or cleaned here.
    """
    tri, pts, faces = _triangle_points_faces(mesh)
    if faces.size:
        pad = np.full((len(faces), 1), 3, dtype=np.int64)
        poly = pv.PolyData(np.ascontiguousarray(pts), np.hstack((pad, faces)).ravel())
    else:
        poly = tri
    n_components = _n_connected_components(poly) if int(poly.n_cells) else 0
    n_loops = _n_boundary_loops(poly) if int(poly.n_cells) else 0
    n_isect = _self_intersection_count(poly, pts, faces)
    min_angle = _min_triangle_angle_deg(pts, faces)
    out = {
        "self_intersection_count": n_isect,
        "n_boundary_loops": n_loops,
        "n_components": n_components,
        "min_triangle_angle_deg": min_angle,
    }
    components_ok = n_components == 1
    intersection_free = n_isect == 0
    loops_ok = True
    if n_profiles is not None:
        expected = int(n_profiles)
        out["n_profiles"] = expected
        loops_ok = n_loops == expected
        out["boundary_loops_eq_n_profiles"] = bool(loops_ok)
    # Model selection: topology/fold failure is a failed sample even if Chamfer is good.
    out["valid"] = bool(components_ok and intersection_free and loops_ok)
    return out


def _load_remesh_surface_isotropically():
    template_dir = os.path.join(_REPO_ROOT, "datatransform", "template_creation")
    if template_dir not in sys.path:
        sys.path.insert(0, template_dir)
    try:
        from vessel_pipeline import remesh_surface_isotropically
    except ImportError as exc:
        raise RuntimeError(
            "remesh_for_cfd requires VMTK (conda env 'hemomesh' or 'vmtk_env'). "
            "mesh_validity_metrics does not need VMTK and is not a remesh. "
            "This function does not fake an isotropic remesh with VTK/PyVista."
        ) from exc
    return remesh_surface_isotropically


def remesh_for_cfd(mesh, edge_mm=CFD_REMESH_EDGE_MM):
    """Final isotropic remesh of the decoded surface before CFD.

    Uses the same ``remesh_surface_isotropically`` as the GT generator
    (edgelength mode, PreserveBoundaryEdges=1). Default ``edge_mm`` is the
    hemoMesh CFD area 0.025 mm^2 as an equilateral edge (~0.24 mm), not the
    0.15 mm AI-GT target. Restores triangle quality after large deformation;
    it is not a substitute for fold penalties or the validity metrics.
    """
    remesh_fn = _load_remesh_surface_isotropically()
    poly = _as_polydata(mesh)
    out = remesh_fn(poly, target_edge_length=float(edge_mm))
    return pv.wrap(out)


def _require_train_stack(fn_name):
    if _HAS_TRAIN_STACK:
        return
    raise RuntimeError(
        f"{fn_name} needs the Stage-2 training stack (PyTorch / torch_geometric). "
        "Use aneurysmgnn / aneuxai_env. Mesh validity metrics and remesh_for_cfd "
        "do not require it."
    )


# %% 
# ==========================================
# EVALUATE ALL SAMPLES & SAVE TO CSV
# ==========================================
# Run this cell to evaluate the model on the entire dataset and output per-patient losses.
def evaluate_all_samples():
    _require_train_stack("evaluate_all_samples")
    print("Loading dataset...")
    dataset = AneurysmDataset(
        tube_radius=TUBE_RADIUS,
        n_length=N_LENGTH,
        n_radial=N_RADIAL,
        cache_dir=CACHE_DIR,
        n_true=N_TRUE,
        cleandata_root=CLEANDATA_ROOT,
        require_templates=True,
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
                'rad_loss': terms['rad'].item() if 'rad' in terms else 0.0,
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
    _require_train_stack("generate_vtp_for_sample")
    if output_filename is None:
        output_filename = os.path.join(OUTPUT_DIR, f"{target_patient_id}_predicted.vtp")
        
    print(f"Loading dataset to find {target_patient_id}...")
    dataset = AneurysmDataset(
        tube_radius=TUBE_RADIUS,
        n_length=N_LENGTH,
        n_radial=N_RADIAL,
        cache_dir=CACHE_DIR,
        n_true=N_TRUE,
        cleandata_root=CLEANDATA_ROOT,
        require_templates=True,
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
    mesh = tensor_to_vtp(x_world + origin, faces_np, output_filename)
    metrics = mesh_validity_metrics(mesh)
    print("Mesh validity metrics (not repaired):", metrics)
    print("Done!")


if __name__ == "__main__":
    generate_vtp_for_sample("SNF00000419")
