import os
import sys

os.environ.setdefault("VTK_OFFSCREEN", "1")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VTK_NUMBER_OF_THREADS", "1")

import pandas as pd
import numpy as np
import pyvista as pv
from scipy.spatial import KDTree
from collections import defaultdict, deque
from tqdm import tqdm
import argparse
import multiprocessing

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from aneux_paths import (
    CSV_PATH,
    VESSELS_AREA001,
    CLEANED_VESSELS,
    LABEL_OUTPUT_DIR,
    HASEXTENSION_CSV,
    TEMPLATE_DIR,
)


class UnrealisticMeshError(ValueError):
    pass


# --- CONFIGURATION ---
VESSEL_DIR = VESSELS_AREA001
OUTPUT_DIR = CLEANED_VESSELS
# Voronoi in workers is memory-heavy; centerline_creation.py uses 2.
MAX_VMTK_WORKERS = 4
# If False, labeled no-extension cases are skipped (not copied into OUTPUT_DIR).
COPY_UNEXTENDED = False

DISTANCE_THRESHOLD = 0.5  # in mm, to classify boundary as extension vs native
SMOOTHING_ITERATIONS = 5  # localized Laplacian iterations for boundary smoothing
SMOOTHING_FACTOR = 0.5    # alpha parameter for smoothing

# --- EXTENSION CUT PARAMETERS ---
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


def clean_vessel_extensions(vessel, centerline, has_extension_label=None, dataset_id=""):
    """Cut CFD flow extensions from an already-open vessel mesh."""
    if not vessel.is_all_triangles:
        vessel = vessel.triangulate()

    adj, faces = build_adjacency(vessel)

    vessel_with_normals = vessel.compute_normals(cell_normals=False, point_normals=True)
    normals = vessel_with_normals.point_data['Normals']
    verts_np = np.array(vessel.points)

    # -------------------------------------------------------------
    # Phase 1: Extension Detection & Endpoint Extraction
    # -------------------------------------------------------------
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

        centroid = np.mean(loop.points, axis=0)
        centered = loop.points - centroid
        _, _, vh = np.linalg.svd(centered)
        normal = vh[2, :]

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

    centerline_endpoints = []
    for i in range(centerline.n_cells):
        cell = centerline.GetCell(i)
        n_pts = cell.GetNumberOfPoints()
        if n_pts >= 2:
            cell_pt_ids = [cell.GetPointId(j) for j in range(n_pts)]
            centerline_endpoints.append(centerline.points[cell_pt_ids[0]])
            centerline_endpoints.append(centerline.points[cell_pt_ids[-1]])

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

                    if np.linalg.norm(v_coord - plane['loop_centroid']) > max_bfs_dist:
                        continue

                    dot_prod = np.dot(v_coord - plane['origin'], plane['normal'])
                    if dot_prod > 0:
                        selected_verts.add(neighbor)
                        queue.append(neighbor)
        all_verts_to_delete.update(selected_verts)

    faces_to_delete = []
    for idx, face in enumerate(faces):
        if any(v in all_verts_to_delete for v in face):
            faces_to_delete.append(idx)

    keep_mask = np.ones(len(faces), dtype=bool)
    keep_mask[faces_to_delete] = False
    kept_faces = faces[keep_mask]

    num_faces = kept_faces.shape[0]
    padding = np.full((num_faces, 1), 3, dtype=kept_faces.dtype)
    padded_faces = np.hstack((padding, kept_faces)).flatten()

    cleaned_vessel_raw = pv.PolyData(vessel.points, padded_faces)
    cleaned_vessel = cleaned_vessel_raw.clean()

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


def _runtime_centerline(vessel, dataset_id):
    """Trace a Voronoi centerline on the open mesh. Not written to disk."""
    if TEMPLATE_DIR not in sys.path:
        sys.path.insert(0, TEMPLATE_DIR)
    from vessel_pipeline import compute_centerline_from_mesh, to_vtk_poly

    print(f"[{dataset_id}] Runtime centerline (centerline_creation pipeline)...", flush=True)
    cl = compute_centerline_from_mesh(to_vtk_poly(vessel))
    wrapped = pv.wrap(cl)
    if wrapped.n_points < 2 or wrapped.n_cells < 1:
        raise RuntimeError("runtime centerline is empty")
    return wrapped


def process_patient_worker(
    dataset_id,
    location,
    vessel_file,
    output_file,
    has_extension_label=None,
    copy_unextended=False,
):
    """Cut CFD flow extensions. Caps must already be removed."""
    import gc
    if not os.path.exists(vessel_file):
        return {
            'dataset_id': dataset_id, 'location': location, 'status': 'Skipped', 'reason': "Vessel file missing",
            'cuts_made': 0, 'integrity_check': 'N/A'
        }

    try:
        has_extension = (has_extension_label in [1, 2]) if has_extension_label is not None else False

        if not has_extension:
            if not copy_unextended:
                return {
                    'dataset_id': dataset_id, 'location': location, 'status': 'Skipped',
                    'reason': 'No extension (copy-unextended is off)',
                    'cuts_made': 0, 'integrity_check': 'N/A'
                }
            vessel = pv.read(vessel_file)
            vessel.save(output_file)
            del vessel
            gc.collect()
            return {
                'dataset_id': dataset_id, 'location': location, 'status': 'Saved',
                'reason': 'Saved as-is (no extensions)',
                'cuts_made': 0, 'integrity_check': 'Passed'
            }

        vessel = pv.read(vessel_file)
        centerline = _runtime_centerline(vessel, dataset_id)
        cleaned_mesh, cuts_made, passed_check = clean_vessel_extensions(
            vessel, centerline, has_extension_label, dataset_id
        )
        cleaned_mesh.save(output_file)

        if not passed_check:
            reason = f"Saved (Warning: boundary mismatch, cuts: {cuts_made})"
            chk_status = "Failed"
        elif cuts_made > 0:
            reason = f"Saved successfully (cuts: {cuts_made})"
            chk_status = "Passed"
        else:
            reason = "Saved as-is (no extensions cut)"
            chk_status = "Passed"

        del vessel
        del centerline
        del cleaned_mesh
        gc.collect()

        return {
            'dataset_id': dataset_id, 'location': location, 'status': 'Saved', 'reason': reason,
            'cuts_made': cuts_made, 'integrity_check': chk_status
        }

    except UnrealisticMeshError as ume:
        gc.collect()
        return {
            'dataset_id': dataset_id, 'location': location, 'status': 'Skipped', 'reason': str(ume),
            'cuts_made': 0, 'integrity_check': 'N/A'
        }
    except Exception as e:
        gc.collect()
        return {
            'dataset_id': dataset_id, 'location': location, 'status': 'Skipped', 'reason': f"Error: {str(e)}",
            'cuts_made': 0, 'integrity_check': 'Error'
        }


def _run_task(task):
    """Top-level Pool target so spawn workers can pickle the call."""
    return process_patient_worker(*task)


def _parse_args():
    parser = argparse.ArgumentParser(description="Remove CFD flow extensions from open vessels.")
    parser.add_argument(
        "--copy-unextended",
        action=argparse.BooleanOptionalAction,
        default=COPY_UNEXTENDED,
        help="Copy vessels with no extension label into the output folder (default: off).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_VMTK_WORKERS,
        help="Parallel spawn processes. Each case gets a fresh interpreter (VMTK isolation).",
    )
    return parser.parse_args()


def _consume_record(record, log_records, counts):
    log_records.append(record)
    if record['status'] == 'Saved':
        counts['saved'] += 1
        counts['cuts'].append(record['cuts_made'])
        if record['integrity_check'] == 'Failed':
            counts['integrity_failed'] += 1
            print(f"\nWarning: Integrity check failed (boundary count mismatch) for {record['dataset_id']}")
    elif record['status'] == 'Skipped':
        counts['skipped'] += 1
        if record['reason'].startswith("Error:"):
            print(f"\nError processing patient {record['dataset_id']}: {record['reason']}")


def main():
    import shutil

    args = _parse_args()
    copy_unextended = bool(args.copy_unextended)

    if os.path.exists(OUTPUT_DIR):
        print(f"Deleting existing directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LABEL_OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    locations = ['ICA pcom', 'ICA oph', 'ICA cav', 'ICA bif']
    df_filtered = df[df['location'].isin(locations)]

    print(f"Loaded CSV. Total filtered dataset rows: {len(df_filtered)}")
    print(f"Copy unextended vessels: {copy_unextended}")

    counts = {'saved': 0, 'skipped': 0, 'integrity_failed': 0, 'cuts': []}
    log_records = []

    has_ext_csv = HASEXTENSION_CSV
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
        output_file = os.path.join(OUTPUT_DIR, f"{dataset_id}.vtp")
        label = has_ext_map.get(dataset_id, None)
        has_extension = (label in [1, 2]) if label is not None else False
        if not has_extension and not copy_unextended:
            log_records.append({
                'dataset_id': dataset_id, 'location': location, 'status': 'Skipped',
                'reason': 'No extension (copy-unextended is off)',
                'cuts_made': 0, 'integrity_check': 'N/A',
            })
            counts['skipped'] += 1
            continue
        tasks.append((dataset_id, location, vessel_file, output_file, label, copy_unextended))

    num_workers = max(1, min(int(args.workers), os.cpu_count() or 1, len(tasks) or 1))
    print(f"Cases to run: {len(tasks)} | spawn workers: {num_workers} (maxtasksperchild=1)")

    if not tasks:
        print("No extension cases to process.")
    elif num_workers == 1:
        for task in tqdm(tasks, desc="Removing flow extensions"):
            _consume_record(_run_task(task), log_records, counts)
    else:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=num_workers, maxtasksperchild=1) as pool:
            for record in tqdm(
                pool.imap_unordered(_run_task, tasks),
                total=len(tasks),
                desc="Removing flow extensions",
            ):
                _consume_record(record, log_records, counts)

    log_df = pd.DataFrame(log_records)
    log_xlsx = os.path.join(LABEL_OUTPUT_DIR, "unextend_log.xlsx")
    log_csv = os.path.join(LABEL_OUTPUT_DIR, "unextend_log.csv")

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

    print("\n" + "=" * 50)
    print("CFD EXTENSION REMOVAL SUMMARY")
    print("=" * 50)
    print(f"Total Saved: {counts['saved']}")
    print(f"  Total Cuts Made: {sum(counts['cuts'])}")
    print(f"Total Skipped: {counts['skipped']}")
    print(f"Failed Integrity Checks: {counts['integrity_failed']}")
    print(f"Unextended meshes saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
