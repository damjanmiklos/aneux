import os
import pandas as pd
import numpy as np
import pyvista as pv
from scipy.spatial import KDTree
from collections import defaultdict, deque
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc


class UnrealisticMeshError(ValueError):
    pass


# --- CONFIGURATION ---
CSV_PATH = r"C:\Users\miklo\OneDrive\UQ\aneux\rawdata\data-v1.0\data\clinical.csv"
VESSEL_DIR = r"C:\Users\miklo\OneDrive\UQ\aneux\rawdata\models-v1.0\models\vessels\remeshed\area-001"
CENTERLINE_DIR = r"C:\Users\miklo\OneDrive\UQ\aneux\rawdata\models-v1.0\models\centerlines"
OUTPUT_DIR = r"C:\Users\miklo\OneDrive\UQ\aneux\datatransform\cleaned_data\vessels_cleaned_and_decapped"

DISTANCE_THRESHOLD = 0.5  # in mm, to classify boundary as extension vs native
SMOOTHING_ITERATIONS = 5  # localized Laplacian iterations for boundary smoothing
SMOOTHING_FACTOR = 0.5    # alpha parameter for smoothing

# --- DECAPPING PARAMETERS ---
NORMALS_THRESHOLD = 0.05  # threshold of mean_dot to classify native vessel transition
MIN_DETECTION_DIST = 0.5  # in mm, to skip initial boundary normals transient


def build_adjacency(vessel):
    """
    Builds a vertex adjacency list from a triangulated PolyData mesh.
    Returns a defaultdict mapping vertex index -> list of unique neighbor indices.
    """
    faces = vessel.faces.reshape(-1, 4)[:, 1:]
    adj = defaultdict(list)
    for face in faces:
        v0, v1, v2 = face
        adj[v0].extend([v1, v2])
        adj[v1].extend([v0, v2])
        adj[v2].extend([v0, v1])
    for v in adj:
        adj[v] = list(set(adj[v]))
    return adj, faces


def get_outward_tangent(cell_coords, is_start):
    """
    Computes a stable outward-facing tangent vector at a centerline endpoint.
    """
    n_pts = len(cell_coords)
    k = min(5, n_pts - 1)
    if is_start:
        v = cell_coords[0] - cell_coords[k]
    else:
        v = cell_coords[-1] - cell_coords[-1 - k]
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else v



def uncap_vessel_mesh(vessel, centerline=None):
    if not vessel.is_all_triangles:
        vessel = vessel.triangulate()
        
    vessel = vessel.compute_normals(cell_normals=True, point_normals=False)
    cell_normals = vessel.cell_data['Normals']
    
    sized_vessel = vessel.compute_cell_sizes(length=False, area=True, volume=False)
    face_areas = sized_vessel.cell_data['Area']
        
    faces = vessel.faces.reshape(-1, 4)[:, 1:]
    verts = vessel.points
    
    # 1. Build Adjacency Graph (Edge to Faces)
    edge_to_faces = defaultdict(list)
    for f_idx, face in enumerate(faces):
        edges = [
            tuple(sorted([face[0], face[1]])),
            tuple(sorted([face[1], face[2]])),
            tuple(sorted([face[2], face[0]]))
        ]
        for edge in edges:
            edge_to_faces[edge].append(f_idx)
            
    face_neighbors = defaultdict(list)
    for f_list in edge_to_faces.values():
        if len(f_list) == 2:
            face_neighbors[f_list[0]].append(f_list[1])
            face_neighbors[f_list[1]].append(f_list[0])

    # --- REMOVED 4-TRIANGLE SMOOTHING ---
    # Smoothing blurs the sharp transition at the edge of the cap, 
    # causing the region growing to "leak" down the vessel wall.

    # 2. Local Region Growing
    # Tightened to 15 degrees. Caps are flat. 45 degrees allows the 
    # algorithm to wrap around the curved cylinder of the vessel wall.
    COS_ANGLE_THRESH = np.cos(np.radians(15)) 
    visited = np.zeros(len(faces), dtype=bool)
    patches = []
    
    for i in range(len(faces)):
        if visited[i]:
            continue
            
        queue = [i]
        visited[i] = True
        current_patch = [i]
        
        # We use the seed normal to strictly enforce flatness across the whole cap
        # rather than just comparing neighbor-to-neighbor (which allows slow creeping curves)
        seed_normal = cell_normals[i]
        
        head = 0
        while head < len(queue):
            curr = queue[head]
            head += 1
            
            for neighbor in face_neighbors[curr]:
                if not visited[neighbor]:
                    n_neighbor = cell_normals[neighbor]
                    
                    # Compare neighbor normal to the SEED normal. 
                    # This guarantees the entire patch remains flat.
                    if np.dot(seed_normal, n_neighbor) > COS_ANGLE_THRESH:
                        visited[neighbor] = True
                        queue.append(neighbor)
                        current_patch.append(neighbor)
                        
        patches.append(current_patch)
        
    # 3. The PCA & Flatness Tests
    faces_to_delete = set()
    for patch in patches:
        # Re-instituted a minimum triangle rule. A highly remeshed cap 
        # like the one in your image will easily have 50+ triangles.
        if len(patch) < 20 or len(patch) > 3000:
            continue
            
        patch = np.array(patch)
        p_normals = cell_normals[patch] 
        p_areas = face_areas[patch]
        
        weighted_normals = p_normals * p_areas[:, np.newaxis]
        sum_normals = np.sum(weighted_normals, axis=0)
        total_area = np.sum(p_areas)
        
        # Flatness test
        flatness = np.linalg.norm(sum_normals) / total_area
        if flatness < 0.70: # Slightly tightened since we aren't smoothing
            continue
            
        patch_verts_idx = np.unique(faces[patch].flatten())
        patch_points = verts[patch_verts_idx]
        
        if len(patch_points) < 3:
            continue
            
        mean_pt = np.mean(patch_points, axis=0)
        centered_pts = patch_points - mean_pt
        cov = np.cov(centered_pts.T)
        eigenvalues, _ = np.linalg.eigh(cov)
        
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        if eigenvalues[0] < 1e-8:
            continue
            
        s0 = np.sqrt(eigenvalues[0])
        s1 = np.sqrt(max(0, eigenvalues[1]))
        s2 = np.sqrt(max(0, eigenvalues[2]))
        
        aspect_ratio = s1 / s0
        thickness_ratio = s2 / s0    
        
        # aspect_ratio > 0.40 ensures it's somewhat circular (not a long thin strip)
        # thickness_ratio < 0.35 ensures it is flat (tightened from 0.50)
        if aspect_ratio > 0.40 and thickness_ratio < 0.35:
            faces_to_delete.update(patch)
            
    # 4. Cleanup and return
    del sized_vessel
    
    if faces_to_delete:
        keep_mask = np.ones(len(faces), dtype=bool)
        keep_mask[list(faces_to_delete)] = False
        kept_faces = faces[keep_mask]
        
        num_faces = kept_faces.shape[0]
        padding = np.full((num_faces, 1), 3, dtype=kept_faces.dtype)
        padded_faces = np.hstack((padding, kept_faces)).flatten()
        
        uncapped_raw = pv.PolyData(verts, padded_faces)
        uncapped_vessel = uncapped_raw.clean()
        
        del uncapped_raw
        gc.collect()
        
        return uncapped_vessel, len(faces_to_delete)
    else:
        gc.collect()
        return vessel, 0


def clean_vessel_extensions(vessel, centerline, has_extension_label=None, dataset_id=""):
    """
    Runs the 4-phase decapping pipeline on a single vessel mesh.
    """
    # Ensure vessel mesh consists entirely of triangles
    if not vessel.is_all_triangles:
        vessel = vessel.triangulate()

    # Build adjacency graph once for region growing
    adj, faces = build_adjacency(vessel)
    
    # Compute point normals on the vessel mesh
    vessel_with_normals = vessel.compute_normals(cell_normals=False, point_normals=True)
    normals = vessel_with_normals.point_data['Normals']
    verts_np = np.array(vessel.points)
    
    # -------------------------------------------------------------
    # Phase 1: Extension Detection & Endpoint Extraction
    # -------------------------------------------------------------
    # Extract boundary loops of the vessel mesh
    boundary_mesh = vessel.extract_feature_edges(
        boundary_edges=True,
        non_manifold_edges=False,
        feature_edges=False,
        manifold_edges=False
    )
    
    if boundary_mesh.n_points == 0:
        raise UnrealisticMeshError("Mesh has 0 open boundaries (completely closed)")
        
    loops = boundary_mesh.split_bodies()
    surf_tree = KDTree(vessel.points)
    
    boundary_loops = []
    for loop in loops:
        if loop is None or loop.n_points == 0:
            continue
        _, surf_indices = surf_tree.query(loop.points)
        
        # Fit plane using SVD to get the normal
        centroid = np.mean(loop.points, axis=0)
        centered = loop.points - centroid
        _, _, vh = np.linalg.svd(centered)
        normal = vh[2, :]
        
        # Ensure normal points OUTWARD (away from bulk of the mesh)
        v_centered = vessel.points - centroid
        projs = np.dot(v_centered, normal)
        if np.mean(projs) > 0:
            normal = -normal
            
        loop_radius = np.mean(np.linalg.norm(loop.points - centroid, axis=1))
        boundary_loops.append({
            'mesh_indices': surf_indices,
            'points': loop.points,
            'centroid': centroid,
            'normal': normal,
            'radius': loop_radius
        })
        
    if len(boundary_loops) <= 1:
        raise UnrealisticMeshError(f"Mesh has only {len(boundary_loops)} open boundary loop(s) (unrealistic)")
        
    # Extract centerline endpoints
    centerline_endpoints = []
    for i in range(centerline.n_cells):
        cell = centerline.GetCell(i)
        n_pts = cell.GetNumberOfPoints()
        if n_pts >= 2:
            cell_pt_ids = [cell.GetPointId(j) for j in range(n_pts)]
            centerline_endpoints.append(centerline.points[cell_pt_ids[0]])
            centerline_endpoints.append(centerline.points[cell_pt_ids[-1]])
            
    # Classify loops as extensions using centerline endpoint proximity
    loop_candidates = []
    loop_distances = []
    for loop in boundary_loops:
        min_dist = float('inf')
        for endpoint in centerline_endpoints:
            dist = np.linalg.norm(loop['centroid'] - endpoint)
            if dist < min_dist:
                min_dist = dist
        loop_distances.append(min_dist)
        loop_candidates.append((loop, min_dist))
        
    if has_extension_label is not None:
        has_extensions = (has_extension_label in [1, 2])
    else:
        has_extensions = np.median(loop_distances) > DISTANCE_THRESHOLD
    
    extensions_to_cut = []
    if has_extensions:
        for loop, min_dist in loop_candidates:
            if min_dist > DISTANCE_THRESHOLD:
                # Grow rings to find extension length L_ext
                visited = set(loop['mesh_indices'])
                current_ring = list(loop['mesh_indices'])
                
                detected_len = 0.0
                inward_normal = -loop['normal']
                
                for step in range(120):
                    next_ring = []
                    for v in current_ring:
                        for neighbor in adj[v]:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                next_ring.append(neighbor)
                    if not next_ring:
                        break
                        
                    if step % 20 == 0:
                        print(f"[{dataset_id}] Phase 1 growing ring {step}...", flush=True)
                        
                    ring_coords = verts_np[next_ring]
                    ring_centroid = np.mean(ring_coords, axis=0)
                    dist = np.dot(ring_centroid - loop['centroid'], inward_normal)
                    
                    # Compute mean dot product of vertex normals with inward axis
                    ring_normals = normals[next_ring]
                    mean_dot = np.mean(np.abs(np.dot(ring_normals, inward_normal)))
                    
                    if dist > MIN_DETECTION_DIST:
                        if mean_dot > NORMALS_THRESHOLD:
                            detected_len = dist
                            break
                            
                    detected_len = dist
                    current_ring = next_ring
                    
                extensions_to_cut.append({
                    'loop': loop,
                    'length': detected_len,
                    'radius': loop['radius']
                })
            
    if not extensions_to_cut:
        # No extensions detected, return original mesh
        return vessel, 0, True
        
    # -------------------------------------------------------------
    # Phase 2: Define Cutting Planes
    # -------------------------------------------------------------
    cutting_planes = []
    for ext in extensions_to_cut:
        normal = ext['loop']['normal']
        origin = ext['loop']['centroid'] - normal * ext['length']
        cutting_planes.append({
            'origin': origin,
            'normal': normal,
            'loop_indices': ext['loop']['mesh_indices'],
            'loop_centroid': ext['loop']['centroid'],
            'length': ext['length'],
            'radius': ext['radius']
        })
        
    # -------------------------------------------------------------
    # Phase 3: Region Growing & Cell Deletion
    # -------------------------------------------------------------
    all_verts_to_delete = set()
    for plane_idx, plane in enumerate(cutting_planes):
        queue = deque(plane['loop_indices'])
        selected_verts = set(plane['loop_indices'])
        
        # Define maximum BFS distance from loop centroid to prevent leak into other branches
        max_bfs_dist = plane['length'] + 2.0 * plane['radius']
        
        step = 0
        while queue:
            curr = queue.popleft()
            step += 1
            if step % 10000 == 0:
                print(f"[{dataset_id}] Phase 3 plane {plane_idx} growing: visited {step} vertices, queue size {len(queue)}", flush=True)
            for neighbor in adj[curr]:
                if neighbor not in selected_verts:
                    v_coord = verts_np[neighbor]
                    
                    # Localized check to prevent leaks
                    if np.linalg.norm(v_coord - plane['loop_centroid']) > max_bfs_dist:
                        continue
                        
                    dot_prod = np.dot(v_coord - plane['origin'], plane['normal'])
                    if dot_prod > 0:  # Vertex is on the extension side
                        selected_verts.add(neighbor)
                        queue.append(neighbor)
        all_verts_to_delete.update(selected_verts)
        
    # Mark faces for deletion
    faces_to_delete = []
    for idx, face in enumerate(faces):
        if any(v in all_verts_to_delete for v in face):
            faces_to_delete.append(idx)
            
    # Reconstruct mesh without extension faces
    keep_mask = np.ones(len(faces), dtype=bool)
    keep_mask[faces_to_delete] = False
    kept_faces = faces[keep_mask]
    
    num_faces = kept_faces.shape[0]
    padding = np.full((num_faces, 1), 3, dtype=kept_faces.dtype)
    padded_faces = np.hstack((padding, kept_faces)).flatten()
    
    cleaned_vessel_raw = pv.PolyData(vessel.points, padded_faces)
    cleaned_vessel = cleaned_vessel_raw.clean()  # clean floating vertices
    
    # -------------------------------------------------------------
    # Phase 4: Localized Smoothing and Integrity Check
    # -------------------------------------------------------------
    new_boundary_mesh = cleaned_vessel.extract_feature_edges(
        boundary_edges=True,
        non_manifold_edges=False,
        feature_edges=False,
        manifold_edges=False
    )
    
    if new_boundary_mesh.n_points > 0:
        new_tree = KDTree(cleaned_vessel.points)
        b_to_mesh_idx = new_tree.query(new_boundary_mesh.points)[1]
        
        b_lines = new_boundary_mesh.lines.reshape(-1, 3)[:, 1:]
        b_lines_mesh_idx = []
        for line in b_lines:
            pt0_coord = new_boundary_mesh.points[line[0]]
            pt1_coord = new_boundary_mesh.points[line[1]]
            
            is_near_cut = False
            for plane in cutting_planes:
                dist0 = np.linalg.norm(pt0_coord - plane['origin'])
                dist1 = np.linalg.norm(pt1_coord - plane['origin'])
                if dist0 < 5.0 and dist1 < 5.0:
                    d_plane0 = np.abs(np.dot(pt0_coord - plane['origin'], plane['normal']))
                    d_plane1 = np.abs(np.dot(pt1_coord - plane['origin'], plane['normal']))
                    if d_plane0 < 1.0 and d_plane1 < 1.0:
                        is_near_cut = True
                        break
                        
            if is_near_cut:
                u_mesh = b_to_mesh_idx[line[0]]
                v_mesh = b_to_mesh_idx[line[1]]
                b_lines_mesh_idx.append((u_mesh, v_mesh))
                
        b_adj = defaultdict(list)
        for u, v in b_lines_mesh_idx:
            b_adj[u].append(v)
            b_adj[v].append(u)
            
        new_points = cleaned_vessel.points.copy()
        for _ in range(SMOOTHING_ITERATIONS):
            temp_points = new_points.copy()
            for v_mesh, neighbors in b_adj.items():
                if len(neighbors) == 2:
                    n1, n2 = neighbors
                    temp_points[v_mesh] = (1.0 - SMOOTHING_FACTOR) * new_points[v_mesh] + SMOOTHING_FACTOR * (new_points[n1] + new_points[n2]) / 2.0
            new_points = temp_points
        cleaned_vessel.points = new_points
        
    # Integrity Check: Count boundary loops
    final_boundary_mesh = cleaned_vessel.extract_feature_edges(
        boundary_edges=True,
        non_manifold_edges=False,
        feature_edges=False,
        manifold_edges=False
    )
    final_num_loops = len(final_boundary_mesh.split_bodies()) if final_boundary_mesh.n_points > 0 else 0
    orig_num_loops = len(loops)
    
    integrity_passed = (final_num_loops == orig_num_loops)
    
    return cleaned_vessel, len(extensions_to_cut), integrity_passed


def process_patient_worker(dataset_id, location, vessel_file, centerline_file, output_file, has_extension_label=None):
    """
    Worker function to process a single patient. First uncaps if capped, then decaps extensions.
    Always saves the mesh (modified or unmodified) to output_file if input files exist.
    """
    import gc
    v_exists = os.path.exists(vessel_file)
    c_exists = os.path.exists(centerline_file)
    
    if not v_exists or not c_exists:
        if not v_exists and not c_exists:
            reason = "Vessel and centerline files missing"
        elif not v_exists:
            reason = "Vessel file missing"
        else:
            reason = "Centerline file missing"
            
        return {
            'dataset_id': dataset_id, 'location': location, 'status': 'Skipped', 'reason': reason,
            'cuts_made': 0, 'integrity_check': 'N/A', 'uncapped': False, 'cap_faces_deleted': 0
        }
        
    try:
        vessel = pv.read(vessel_file)
        centerline = pv.read(centerline_file)
        
        # 1. ALWAYS run uncapping. The new geometric algorithm is safe to run on all meshes.
        # It will gracefully return 0 if no caps are found.
        # print(f"[{dataset_id}] Running geometric cap detection...", flush=True)
        vessel, num_cap_faces_deleted = uncap_vessel_mesh(vessel)
        is_originally_capped = (num_cap_faces_deleted > 0)
            
        has_extension = (has_extension_label in [1, 2]) if has_extension_label is not None else False
        
        # 2. Check if we need to do extension decapping
        if not has_extension:
            vessel.save(output_file)
            if is_originally_capped:
                reason = f'Saved (Uncapped only, cap faces deleted: {num_cap_faces_deleted})'
            else:
                reason = 'Saved as-is (no extensions and no caps found)'
                
            # Memory Cleanup (Prevents multiprocessing NoneType errors)
            del vessel
            del centerline
            gc.collect()
            
            return {
                'dataset_id': dataset_id, 'location': location, 'status': 'Saved', 'reason': reason,
                'cuts_made': 0, 'integrity_check': 'Passed', 'uncapped': is_originally_capped, 'cap_faces_deleted': num_cap_faces_deleted
            }
                
        # 3. Run extension decapping
        # print(f"[{dataset_id}] Starting clean_vessel_extensions...", flush=True)
        cleaned_mesh, cuts_made, passed_check = clean_vessel_extensions(vessel, centerline, has_extension_label, dataset_id)
        
        # Always save the mesh if inputs existed
        cleaned_mesh.save(output_file)
        
        if not passed_check:
            reason = f"Saved (Warning: boundary mismatch, cuts: {cuts_made}, uncapped: {is_originally_capped})"
            chk_status = "Failed"
        else:
            if cuts_made > 0 or is_originally_capped:
                reason = f"Saved successfully (cuts: {cuts_made}, uncapped: {is_originally_capped})"
            else:
                reason = "Saved as-is (no extensions cut and no caps found)"
            chk_status = "Passed"
            
        # Memory Cleanup (Prevents multiprocessing NoneType errors)
        del vessel
        del centerline
        del cleaned_mesh
        gc.collect()
            
        return {
            'dataset_id': dataset_id, 'location': location, 'status': 'Saved', 'reason': reason,
            'cuts_made': cuts_made, 'integrity_check': chk_status, 'uncapped': is_originally_capped, 'cap_faces_deleted': num_cap_faces_deleted
        }
            
    except UnrealisticMeshError as ume:
        gc.collect()
        return {
            'dataset_id': dataset_id, 'location': location, 'status': 'Skipped', 'reason': str(ume),
            'cuts_made': 0, 'integrity_check': 'N/A', 'uncapped': False, 'cap_faces_deleted': 0
        }
    except Exception as e:
        gc.collect()
        return {
            'dataset_id': dataset_id, 'location': location, 'status': 'Skipped', 'reason': f"Error: {str(e)}",
            'cuts_made': 0, 'integrity_check': 'Error', 'uncapped': False, 'cap_faces_deleted': 0
        }


def main():
    import shutil
    if os.path.exists(OUTPUT_DIR):
        print(f"Deleting existing directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("C:\\Users\\miklo\\OneDrive\\UQ\\aneux\\datatransform\\label\\output", exist_ok=True)
    
    # Read clinical metadata and filter ICA locations
    df = pd.read_csv(CSV_PATH)
    locations = ['ICA pcom', 'ICA oph', 'ICA cav', 'ICA bif']
    df_filtered = df[df['location'].isin(locations)]
    
    print(f"Loaded CSV. Total filtered dataset rows: {len(df_filtered)}")
    
    processed_count = 0
    skipped_count = 0
    saved_count = 0
    uncapped_count = 0
    total_cap_faces_deleted = 0
    integrity_failed_count = 0
    cut_counts = []
    
    log_records = []
    
    # Load hasextension mapping
    has_ext_csv = "C:\\Users\\miklo\\OneDrive\\UQ\\aneux\\datatransform\\label\\hasextension.csv"
    has_ext_map = {}
    if os.path.exists(has_ext_csv):
        has_ext_df = pd.read_csv(has_ext_csv)
        has_ext_map = dict(zip(has_ext_df['Filename'], has_ext_df['Label']))
        print(f"Loaded hasextension.csv. Found {len(has_ext_map)} mappings.")
    else:
        print(f"Warning: {has_ext_csv} not found, defaulting to old behavior")

    tasks = []
    for _, row in df_filtered.iterrows():
        dataset_id = row['dataset']
        location = row['location']
        vessel_file = os.path.join(VESSEL_DIR, f"{dataset_id}.vtp")
        centerline_file = os.path.join(CENTERLINE_DIR, f"{dataset_id}.vtp")
        output_file = os.path.join(OUTPUT_DIR, f"{dataset_id}.vtp")
        
        label = has_ext_map.get(dataset_id, None)
        tasks.append((dataset_id, location, vessel_file, centerline_file, output_file, label))
        
    num_workers = min(os.cpu_count() or 1, 20)
    print(f"Starting multiprocessing pool with {num_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_patient_worker, *task): task
            for task in tasks
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Vessel Mesh Cleaning"):
            try:
                record = future.result()
                log_records.append(record)
                
                if record['status'] == 'Saved':
                    saved_count += 1
                    processed_count += 1
                    cut_counts.append(record['cuts_made'])
                    if record['uncapped']:
                        uncapped_count += 1
                        total_cap_faces_deleted += record['cap_faces_deleted']
                    if record['integrity_check'] == 'Failed':
                        integrity_failed_count += 1
                        print(f"\nWarning: Integrity check failed (boundary count mismatch) for {record['dataset_id']}")
                elif record['status'] == 'Skipped':
                    skipped_count += 1
                    if record['reason'].startswith("Error:"):
                        print(f"\nError processing patient {record['dataset_id']}: {record['reason']}")
            except Exception as e:
                print(f"\nWorker crashed with exception: {e}")
                
    # Write the log to Excel and CSV fallback
    log_df = pd.DataFrame(log_records)
    log_xlsx = "C:\\Users\\miklo\\OneDrive\\UQ\\aneux\\datatransform\\label\\output\\clean_and_uncap_log.xlsx"
    log_csv = "C:\\Users\\miklo\\OneDrive\\UQ\\aneux\\datatransform\\label\\output\\clean_and_uncap_log.csv"
    
    try:
        log_df.to_excel(log_xlsx, index=False)
        print(f"Saved Excel log to: {log_xlsx}")
    except Exception as ex:
        print(f"Failed to write Excel file (openpyxl might be missing): {ex}")
        
    try:
        log_df.to_csv(log_csv, index=False)
        print(f"Saved CSV log to: {log_csv}")
    except Exception as ex:
        print(f"Failed to write CSV file: {ex}")
        
    print("\n" + "="*50)
    print("CFD EXTENSION REMOVAL & UNCAPPING SUMMARY")
    print("="*50)
    print(f"Total Saved (modified/uncapped/cut): {saved_count}")
    print(f"  Vessels Uncapped: {uncapped_count} (deleted {total_cap_faces_deleted} cap faces)")
    print(f"  Total Cuts Made: {sum(cut_counts)}")
    print(f"Total Skipped (no changes or missing): {skipped_count}")
    print(f"Failed Integrity Checks: {integrity_failed_count}")
    print(f"Cleaned and uncapped meshes saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()
