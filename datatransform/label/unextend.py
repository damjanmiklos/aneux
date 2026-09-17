import os
import sys

os.environ.setdefault("VTK_OFFSCREEN", "1")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VTK_NUMBER_OF_THREADS", "1")

import argparse
import multiprocessing
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import pyvista as pv
import vtk
from scipy.spatial import KDTree
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from aneux_paths import (
    CLEANED_VESSELS,
    CSV_PATH,
    HASCAPOREXTENSION_CSV,
    LABEL_OUTPUT_DIR,
    VESSELS_ORIGINAL,
)


class UnrealisticMeshError(ValueError):
    pass


# --- CONFIGURATION ---
VESSEL_DIR = VESSELS_ORIGINAL
OUTPUT_DIR = CLEANED_VESSELS
MAX_WORKERS = 16
# If False, cases that are not label "2" are skipped (not copied into OUTPUT_DIR).
COPY_UNEXTENDED = False
EXTENSION_LABEL = "2"

SMOOTHING_ITERATIONS = 5
SMOOTHING_FACTOR = 0.5

# A CFD extension is a straight constant-R tube. Native ostia flare or curve
# within a couple of millimetres. Thin tubes are judged in radii; the millimetre
# floor only rejects sub-millimetre noise, not 2 mm / 5 R perforators.
MIN_EXT_LENGTH_MM = 1.5
MIN_EXT_LENGTH_RADII = 2.0
CYL_RADIUS_TOL = 0.18
CYL_NORMAL_DOT = 0.22
CYL_AXIS_OFFSET = 0.35
RING_COLLAPSE = 0.60
RING_COLLAPSE_MIN_SEED = 16
MIN_DETECTION_DIST = 0.75
MAX_WALK_MM = 28.0
MAX_WALK_STEPS = 240
CLIP_RADIUS_FACTOR = 2.2
CLIP_FACTORS = (1.3, 1.8, 2.2, 3.0)
LENGTH_BACKOFF = (1.0, 0.94, 0.88, 0.80, 0.70)
MIN_KEEP_FRAC = 0.40
MIN_LOOP_POINTS = 8
MIN_LOOP_RADIUS_MM = 0.25
# Same cylinder walk used to report leftover %. Cut until that leftover is a
# short cuff, not until it is ~0 (that last millimetre is already native ostium).
KEEP_CUFF_MM = 0.80
KEEP_CUFF_RADII = 0.30
LEFTOVER_RECUT_MM = 1.40
LEFTOVER_RECUT_RADII = 0.70
LEFTOVER_SHRINK_MM = 0.25
MAX_CUT_PASSES = 2
# ---------------------


def _norm_label(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def keep_cuff(radius):
    """Leftover cylinder length we want to keep at a cut opening, in mm."""
    return max(KEEP_CUFF_MM, KEEP_CUFF_RADII * float(radius))


def leftover_too_long(cyl_len, radius):
    """True if the reported leftover is still a CFD remnant, not just the cuff."""
    return float(cyl_len) >= max(LEFTOVER_RECUT_MM, LEFTOVER_RECUT_RADII * float(radius))


def leftover_overcut(cyl_len, radius):
    """True if leftover cylinder is shorter than half the intended cuff."""
    return float(cyl_len) < 0.50 * keep_cuff(radius)


def _unit(vec):
    vec = np.asarray(vec, dtype=np.float64)
    nrm = float(np.linalg.norm(vec))
    if nrm < 1e-12:
        return vec
    return vec / nrm


def _as_surface(mesh):
    """Return a PolyData surface.

    Clip, split_bodies, and some VTK filters yield UnstructuredGrid. The rest
    of this pipeline (faces, is_all_triangles, .vtp save) requires PolyData.
    """
    if mesh is None:
        return mesh
    if isinstance(mesh, pv.PolyData):
        return mesh
    if getattr(mesh, "n_points", 0) == 0:
        return pv.PolyData()
    try:
        surf = mesh.extract_surface(
            pass_pointid=False,
            pass_cellid=False,
            algorithm="dataset_surface",
        )
    except TypeError:
        surf = mesh.extract_surface()
    except Exception:
        surf = pv.wrap(mesh).extract_surface()
    if not isinstance(surf, pv.PolyData):
        surf = pv.PolyData(np.asarray(surf.points), getattr(surf, "faces", None))
    return surf


def _as_triangles(mesh):
    mesh = _as_surface(mesh)
    if mesh is None or mesh.n_points == 0:
        return mesh
    if not mesh.is_all_triangles:
        mesh = mesh.triangulate()
    return _as_surface(mesh)


def resolve_vessel_file(folder, dataset_id):
    for ext in (".vtp", ".stl", ".vtk"):
        path = os.path.join(folder, f"{dataset_id}{ext}")
        if os.path.exists(path):
            return path
    return os.path.join(folder, f"{dataset_id}.vtp")


def build_adjacency(vessel):
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


def _boundary_loops(vessel):
    boundary_mesh = vessel.extract_feature_edges(
        boundary_edges=True,
        non_manifold_edges=False,
        feature_edges=False,
        manifold_edges=False,
    )
    if boundary_mesh.n_points == 0:
        return [], boundary_mesh
    loops = []
    for loop in boundary_mesh.split_bodies():
        if loop is None or loop.n_points < MIN_LOOP_POINTS:
            continue
        loops.append(loop)
    return loops, boundary_mesh


def _loop_frame(loop_points, vessel_points):
    centroid = np.mean(loop_points, axis=0)
    centered = np.asarray(loop_points, dtype=np.float64) - centroid
    cov = centered.T @ centered
    _evals, evecs = np.linalg.eigh(cov)
    normal = _unit(evecs[:, 0])
    sample = np.asarray(vessel_points, dtype=np.float64)
    if sample.shape[0] > 4000:
        sample = sample[:: max(1, sample.shape[0] // 4000)]
    if np.mean(np.dot(sample - centroid, normal)) > 0:
        normal = -normal
    radius = float(np.mean(np.linalg.norm(centered, axis=1)))
    return centroid, normal, radius


def _grow_cylinder(adj, verts, normals, seed_ids, centroid, outward, radius, stub=False):
    """Walk inward from an opening. Return cylindrical length and corrected outward."""
    inward = -outward
    visited = set(int(i) for i in seed_ids)
    current = [int(i) for i in seed_ids]
    flipped = False
    cyl_len = 0.0
    fail_streak = 0
    early_dots = []
    early_n = None
    min_ext = max(MIN_EXT_LENGTH_MM, MIN_EXT_LENGTH_RADII * radius)
    # Leftover cuffs are shorter than 2R, so ostium stop cannot wait for min_ext.
    min_stop = keep_cuff(radius) if stub else min_ext

    for _ in range(MAX_WALK_STEPS):
        nxt = []
        for v in current:
            for neighbor in adj.get(v, ()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    nxt.append(neighbor)
        if not nxt:
            break

        coords = verts[nxt]
        ring_c = np.mean(coords, axis=0)
        s = float(np.dot(ring_c - centroid, inward))
        if not flipped and abs(s) > 0.15:
            if s < 0.0:
                inward = -inward
                outward = -outward
                s = -s
            flipped = True
        if s > MAX_WALK_MM:
            break

        radial = np.linalg.norm(np.cross(coords - centroid, inward), axis=1)
        ring_r = float(np.mean(radial))
        offset = float(np.linalg.norm(np.cross(ring_c - centroid, inward)))
        mean_dot = float(np.mean(np.abs(np.dot(normals[nxt], inward))))
        n_seed = max(len(seed_ids), 1)
        r_ref = max(radius, 1e-3)
        ring_collapsed = (
            n_seed >= RING_COLLAPSE_MIN_SEED
            and len(nxt) < RING_COLLAPSE * n_seed
        )
        geom_drift = (
            abs(ring_r - radius) > max(0.08 * r_ref, 0.08)
            or offset > max(0.15 * r_ref, 0.15)
        )
        # A thin tube can drop to ~half the rim vertices and stay cylindrical.
        # Count ring collapse as the transition only when the tube also drifts.
        still_cyl = (
            s > 0.0
            and abs(ring_r - radius) <= CYL_RADIUS_TOL * r_ref
            and mean_dot <= CYL_NORMAL_DOT
            and offset <= CYL_AXIS_OFFSET * r_ref
            and not (ring_collapsed and geom_drift)
        )
        if still_cyl and fail_streak == 0 and 0.2 < s < 2.8:
            early_dots.append(mean_dot)
            if early_n is None:
                early_n = len(nxt)
        early_dot = float(np.median(early_dots)) if early_dots else 0.0
        flare = ring_r > radius * 1.08
        dot_rise = mean_dot > max(0.10, early_dot + 0.06)
        n_shrink = early_n is not None and len(nxt) < 0.90 * early_n
        # Parent ostium: the ring is still loosely cylindrical, but the wall
        # tilts and the ring thins, or the radius starts to flare.
        if (
            cyl_len >= min_stop
            and s > MIN_DETECTION_DIST
            and (flare or (dot_rise and n_shrink))
        ):
            break
        tight_recovery = (
            abs(ring_r - radius) <= 0.08 * r_ref
            and offset <= 0.20 * r_ref
        )
        if still_cyl:
            if fail_streak == 0 or tight_recovery:
                cyl_len = s
                fail_streak = 0
            else:
                # A looser tube after a fail is usually the parent branch.
                break
        elif s > MIN_DETECTION_DIST:
            fail_streak += 1
            if fail_streak >= 2:
                break
        current = nxt

    return cyl_len, outward


def is_extension_length(cyl_len, radius, stub=False):
    min_full = max(MIN_EXT_LENGTH_MM, MIN_EXT_LENGTH_RADII * radius)
    if stub:
        return leftover_too_long(cyl_len, radius)
    return cyl_len >= min_full


def detect_extensions(vessel, dataset_id="", stub=False):
    """Find openings whose first stretch is a straight constant-R tube."""
    vessel = _as_triangles(vessel)

    adj, faces = build_adjacency(vessel)
    vessel_with_normals = vessel.compute_normals(cell_normals=False, point_normals=True)
    normals = np.asarray(vessel_with_normals.point_data["Normals"])
    verts = np.asarray(vessel.points)

    loops, boundary_mesh = _boundary_loops(vessel)
    if boundary_mesh.n_points == 0:
        raise UnrealisticMeshError("Mesh has 0 open boundaries (completely closed)")
    if len(loops) <= 1:
        raise UnrealisticMeshError(
            f"Mesh has only {len(loops)} open boundary loop(s) (unrealistic)"
        )

    tree = KDTree(verts)
    detected = []
    for loop in loops:
        _, surf_idx = tree.query(loop.points)
        centroid, outward, radius = _loop_frame(loop.points, verts)
        if radius < MIN_LOOP_RADIUS_MM:
            continue
        cyl_len, outward = _grow_cylinder(
            adj, verts, normals, surf_idx, centroid, outward, radius, stub=False
        )
        if not is_extension_length(cyl_len, radius, stub=stub):
            continue
        cut_len = max(cyl_len - keep_cuff(radius), 0.0)
        detected.append({
            "centroid": centroid,
            "outward": outward,
            "radius": radius,
            "length": cut_len,
            "cyl_len": cyl_len,
            "loop_indices": surf_idx,
        })
        print(
            f"[{dataset_id}] extension R={radius:.2f} mm  cyl={cyl_len:.2f} mm "
            f"({cyl_len / max(radius, 1e-3):.1f}R)",
            flush=True,
        )
    return vessel, faces, detected, len(loops)


def _vtk_xyz(fn, pt):
    fn(float(pt[0]), float(pt[1]), float(pt[2]))


def _n_loops(mesh):
    loops, boundary = _boundary_loops(mesh)
    return len(loops) if boundary.n_points > 0 else 0


def _remove_cylindrical_stub(mesh, centroid, outward, radius, length, radius_factor=CLIP_RADIUS_FACTOR):
    """Delete the finite cylinder from the cut plane out through the opening.

    VTK treats implicit-function < 0 as inside. The clip keeps the outside of
    that region, so the region itself must be exactly the extension stub.
    """
    outward = _unit(outward)
    centroid = np.asarray(centroid, dtype=np.float64)
    origin = centroid - outward * float(length)
    far = centroid + outward * max(0.5 * float(radius), 0.3)
    cyl_r = max(float(radius) * float(radius_factor), float(radius) + 0.4)

    cylinder = vtk.vtkCylinder()
    _vtk_xyz(cylinder.SetCenter, origin)
    _vtk_xyz(cylinder.SetAxis, outward)
    cylinder.SetRadius(float(cyl_r))

    near_plane = vtk.vtkPlane()
    _vtk_xyz(near_plane.SetOrigin, origin)
    _vtk_xyz(near_plane.SetNormal, -outward)

    far_plane = vtk.vtkPlane()
    _vtk_xyz(far_plane.SetOrigin, far)
    _vtk_xyz(far_plane.SetNormal, outward)

    region = vtk.vtkImplicitBoolean()
    region.SetOperationTypeToIntersection()
    region.AddFunction(cylinder)
    region.AddFunction(near_plane)
    region.AddFunction(far_plane)

    poly = mesh.cast_to_unstructured_grid().extract_surface() if not isinstance(mesh, pv.PolyData) else mesh
    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(poly)
    clipper.SetClipFunction(region)
    clipper.InsideOutOff()
    clipper.Update()
    clipped = _as_surface(pv.wrap(clipper.GetOutput()))
    if clipped.n_points == 0:
        return _as_surface(mesh)
    return _as_surface(clipped.clean())


def _remove_stub_bfs(mesh, centroid, outward, radius, length):
    """Vertex flood-fill fallback when the implicit cylinder nicks the wall."""
    mesh = _as_triangles(mesh)
    adj, faces = build_adjacency(mesh)
    verts = np.asarray(mesh.points)
    loops, _ = _boundary_loops(mesh)
    if not loops:
        return mesh
    tree = KDTree(verts)
    best = None
    best_d = float("inf")
    for loop in loops:
        dist = float(np.linalg.norm(np.mean(loop.points, axis=0) - centroid))
        if dist < best_d:
            best_d = dist
            best = loop
    if best is None or best_d > 3.0 * max(float(radius), 0.5):
        return mesh

    _, surf_idx = tree.query(best.points)
    outward = _unit(outward)
    origin = np.asarray(centroid, dtype=np.float64) - outward * float(length)
    max_dist = float(length) + 1.25 * float(radius)
    max_radial = 1.4 * float(radius)
    queue = deque(int(i) for i in np.atleast_1d(surf_idx))
    selected = set(queue)
    while queue:
        curr = queue.popleft()
        for neighbor in adj.get(curr, ()):
            if neighbor in selected:
                continue
            pt = verts[neighbor]
            if np.linalg.norm(pt - centroid) > max_dist:
                continue
            if float(np.linalg.norm(np.cross(pt - centroid, outward))) > max_radial:
                continue
            if np.dot(pt - origin, outward) > 0.0:
                selected.add(neighbor)
                queue.append(neighbor)

    keep = np.array([not any(v in selected for v in face) for face in faces], dtype=bool)
    kept = faces[keep]
    if len(kept) < 20:
        return mesh
    padded = np.hstack((np.full((len(kept), 1), 3, dtype=kept.dtype), kept)).ravel()
    return _as_surface(pv.PolyData(mesh.points, padded).clean())


def _keep_largest(mesh):
    mesh = _as_surface(mesh)
    if mesh.n_points == 0:
        return mesh
    try:
        bodies = mesh.split_bodies()
    except Exception:
        return mesh
    if bodies.n_blocks <= 1:
        return mesh
    largest = max(bodies, key=lambda b: 0 if b is None else b.n_points)
    if largest is None:
        return mesh
    return _as_surface(pv.wrap(largest).clean())


def _smooth_cut_rims(cleaned_vessel, cutting_planes):
    new_boundary_mesh = cleaned_vessel.extract_feature_edges(
        boundary_edges=True,
        non_manifold_edges=False,
        feature_edges=False,
        manifold_edges=False,
    )
    if new_boundary_mesh.n_points == 0:
        return cleaned_vessel

    new_tree = KDTree(cleaned_vessel.points)
    b_to_mesh_idx = new_tree.query(new_boundary_mesh.points)[1]
    b_lines = new_boundary_mesh.lines.reshape(-1, 3)[:, 1:]
    b_adj = defaultdict(list)
    for line in b_lines:
        pt0 = new_boundary_mesh.points[line[0]]
        pt1 = new_boundary_mesh.points[line[1]]
        near = False
        for plane in cutting_planes:
            if (
                np.linalg.norm(pt0 - plane["origin"]) < 5.0
                and np.linalg.norm(pt1 - plane["origin"]) < 5.0
                and abs(np.dot(pt0 - plane["origin"], plane["normal"])) < 1.0
                and abs(np.dot(pt1 - plane["origin"], plane["normal"])) < 1.0
            ):
                near = True
                break
        if not near:
            continue
        u_mesh = int(b_to_mesh_idx[line[0]])
        v_mesh = int(b_to_mesh_idx[line[1]])
        b_adj[u_mesh].append(v_mesh)
        b_adj[v_mesh].append(u_mesh)

    if not b_adj:
        return cleaned_vessel

    new_points = np.asarray(cleaned_vessel.points).copy()
    for _ in range(SMOOTHING_ITERATIONS):
        temp_points = new_points.copy()
        for v_mesh, neighbors in b_adj.items():
            if len(neighbors) == 2:
                n1, n2 = neighbors
                temp_points[v_mesh] = (
                    (1.0 - SMOOTHING_FACTOR) * new_points[v_mesh]
                    + SMOOTHING_FACTOR * (new_points[n1] + new_points[n2]) / 2.0
                )
        new_points = temp_points
    cleaned_vessel.points = new_points
    return cleaned_vessel


def _cut_is_safe(candidate, n_before, loops_before):
    return (
        candidate.n_points >= 50
        and candidate.n_points >= MIN_KEEP_FRAC * n_before
        and _n_loops(candidate) == loops_before
    )


def _try_one_cut(mesh, ext, length, n_before, loops_before):
    for factor in CLIP_FACTORS:
        candidate = _keep_largest(
            _remove_cylindrical_stub(
                mesh,
                ext["centroid"],
                ext["outward"],
                ext["radius"],
                length,
                radius_factor=factor,
            )
        )
        if _cut_is_safe(candidate, n_before, loops_before):
            return candidate
    candidate = _keep_largest(
        _remove_stub_bfs(mesh, ext["centroid"], ext["outward"], ext["radius"], length)
    )
    if _cut_is_safe(candidate, n_before, loops_before):
        return candidate
    return None


def _same_branch(ext, prev):
    """True if `ext` is the leftover rim of a previous cut."""
    prev_c = np.asarray(prev["centroid"], dtype=np.float64)
    prev_n = _unit(prev["outward"])
    pred = prev_c - prev_n * float(prev["length"])
    cur = np.asarray(ext["centroid"], dtype=np.float64)
    r_ok = abs(float(ext["radius"]) - float(prev["radius"])) <= max(
        0.40, 0.30 * float(prev["radius"])
    )
    reach = max(
        3.0,
        2.5 * float(prev["radius"]),
        0.6 * float(prev.get("cyl_len", prev["length"])),
    )
    return r_ok and min(
        float(np.linalg.norm(cur - pred)),
        float(np.linalg.norm(cur - prev_c)),
    ) <= reach


def _opening_cyl_near(mesh, point, radius):
    """Cylindrical length of the opening nearest to `point` with similar radius."""
    mesh = _as_triangles(mesh)
    loops, boundary = _boundary_loops(mesh)
    if boundary.n_points == 0 or not loops:
        return None
    adj, _ = build_adjacency(mesh)
    normals = np.asarray(
        mesh.compute_normals(cell_normals=False, point_normals=True).point_data["Normals"]
    )
    verts = np.asarray(mesh.points)
    tree = KDTree(verts)
    point = np.asarray(point, dtype=np.float64)
    best = None
    best_score = float("inf")
    for loop in loops:
        centroid, outward, r = _loop_frame(loop.points, verts)
        score = float(np.linalg.norm(centroid - point)) + 2.0 * abs(r - radius)
        if score < best_score:
            best_score = score
            _, idx = tree.query(loop.points)
            best = (idx, centroid, outward, r)
    if best is None:
        return None
    idx, centroid, outward, r = best
    cyl, _ = _grow_cylinder(
        adj, verts, normals, idx, centroid, outward, r, stub=False
    )
    return cyl


def _apply_extension_cuts(mesh, extensions, dataset_id=""):
    cleaned = mesh
    cutting_planes = []
    applied = []
    for ext in sorted(extensions, key=lambda e: e["length"], reverse=True):
        n_before = cleaned.n_points
        loops_before = _n_loops(cleaned)
        accepted = None
        used_length = None
        overcut_fallback = None
        overcut_fallback_len = None
        for frac in LENGTH_BACKOFF:
            length = float(ext["length"]) * frac
            candidate = _try_one_cut(cleaned, ext, length, n_before, loops_before)
            if candidate is None:
                continue
            origin = ext["centroid"] - ext["outward"] * length
            after = _opening_cyl_near(candidate, origin, ext["radius"])
            if after is not None and leftover_overcut(after, ext["radius"]):
                overcut_fallback = candidate
                overcut_fallback_len = length
                continue
            accepted = candidate
            used_length = length
            if frac < 1.0:
                print(
                    f"[{dataset_id}] shortened cut to {length:.2f} mm "
                    f"(R={ext['radius']:.2f})",
                    flush=True,
                )
            break
        if accepted is None and overcut_fallback is not None:
            accepted = overcut_fallback
            used_length = overcut_fallback_len
        if accepted is None:
            print(
                f"[{dataset_id}] skipped a cut that punched an extra hole or "
                f"removed too much (R={ext['radius']:.2f} L={ext['length']:.2f})",
                flush=True,
            )
            continue
        cleaned = accepted
        origin = ext["centroid"] - ext["outward"] * used_length
        cutting_planes.append({"origin": origin, "normal": ext["outward"]})
        applied.append({**ext, "length": used_length})
    return cleaned, cutting_planes, applied


def _apply_leftover_cuts(mesh, extensions, previous, dataset_id=""):
    """Recut leftover CFD tube, but not native cylinder exposed by the first cut."""
    pending = [ext for ext in extensions if any(_same_branch(ext, prev) for prev in previous)]
    cleaned = mesh
    cutting_planes = []
    applied = []
    for ext in sorted(pending, key=lambda e: e["length"], reverse=True):
        n_before = cleaned.n_points
        loops_before = _n_loops(cleaned)
        accepted = None
        used_length = None
        grew = False
        nicked = False
        for frac in LENGTH_BACKOFF:
            length = float(ext["length"]) * frac
            candidate = _try_one_cut(cleaned, ext, length, n_before, loops_before)
            if candidate is None:
                continue
            origin = ext["centroid"] - ext["outward"] * length
            after = _opening_cyl_near(candidate, origin, ext["radius"])
            if after is not None and leftover_overcut(after, ext["radius"]):
                nicked = True
                continue
            if after is not None and after > ext["cyl_len"] - LEFTOVER_SHRINK_MM:
                grew = True
                continue
            accepted = candidate
            used_length = length
            if frac < 1.0:
                print(
                    f"[{dataset_id}] shortened leftover cut to {length:.2f} mm "
                    f"(R={ext['radius']:.2f})",
                    flush=True,
                )
            break
        if accepted is None:
            if nicked or grew:
                print(
                    f"[{dataset_id}] skipped leftover cut into native vessel "
                    f"(R={ext['radius']:.2f} L={ext['cyl_len']:.2f})",
                    flush=True,
                )
            else:
                print(
                    f"[{dataset_id}] skipped a cut that punched an extra hole or "
                    f"removed too much (R={ext['radius']:.2f} L={ext['length']:.2f})",
                    flush=True,
                )
            continue
        cleaned = accepted
        origin = ext["centroid"] - ext["outward"] * used_length
        cutting_planes.append({"origin": origin, "normal": ext["outward"]})
        applied.append({**ext, "length": used_length})
    return cleaned, cutting_planes, applied


def clean_vessel_extensions(vessel, dataset_id=""):
    """Cut CFD flow extensions from an already-open vessel mesh."""
    vessel, _faces, pending, orig_num_loops = detect_extensions(
        vessel, dataset_id=dataset_id
    )
    if not pending:
        return vessel, 0, True, []

    cleaned = vessel
    cutting_planes = []
    applied = []
    for pass_i in range(MAX_CUT_PASSES):
        if not pending:
            break
        if pass_i:
            print(
                f"[{dataset_id}] pass {pass_i + 1} on {len(pending)} leftover tube(s)",
                flush=True,
            )
            cleaned, planes, extra = _apply_leftover_cuts(
                cleaned, pending, applied, dataset_id
            )
        else:
            cleaned, planes, extra = _apply_extension_cuts(
                cleaned, pending, dataset_id
            )
        cutting_planes.extend(planes)
        applied.extend(extra)
        if not extra:
            break
        try:
            _, _, pending, _ = detect_extensions(
                cleaned, dataset_id=dataset_id, stub=True
            )
        except UnrealisticMeshError:
            pending = []

    if not applied:
        return vessel, 0, True, []

    cleaned = _as_triangles(cleaned)
    cleaned = _smooth_cut_rims(cleaned, cutting_planes)

    final_loops, final_boundary = _boundary_loops(cleaned)
    final_num_loops = len(final_loops) if final_boundary.n_points > 0 else 0
    integrity_passed = final_num_loops == orig_num_loops
    return cleaned, len(applied), integrity_passed, applied


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
            "dataset_id": dataset_id,
            "location": location,
            "status": "Skipped",
            "reason": "Vessel file missing",
            "cuts_made": 0,
            "integrity_check": "N/A",
        }

    try:
        is_extension_case = _norm_label(has_extension_label) == EXTENSION_LABEL
        if not is_extension_case:
            if not copy_unextended:
                return {
                    "dataset_id": dataset_id,
                    "location": location,
                    "status": "Skipped",
                    "reason": "Not label 2 (copy-unextended is off)",
                    "cuts_made": 0,
                    "integrity_check": "N/A",
                }
            vessel = pv.read(vessel_file)
            vessel.save(output_file)
            del vessel
            gc.collect()
            return {
                "dataset_id": dataset_id,
                "location": location,
                "status": "Saved",
                "reason": "Saved as-is (not an extension case)",
                "cuts_made": 0,
                "integrity_check": "Passed",
            }

        vessel = pv.read(vessel_file)
        cleaned_mesh, cuts_made, passed_check, applied = clean_vessel_extensions(
            vessel, dataset_id
        )
        cleaned_mesh = _as_triangles(cleaned_mesh)
        cleaned_mesh.save(output_file)

        lengths = ", ".join(f"{ext['cyl_len']:.1f}" for ext in applied)
        if not passed_check:
            reason = f"Saved (Warning: boundary mismatch, cuts: {cuts_made} @ {lengths} mm)"
            chk_status = "Failed"
        elif cuts_made > 0:
            reason = f"Saved successfully (cuts: {cuts_made} @ {lengths} mm)"
            chk_status = "Passed"
        else:
            reason = "Saved as-is (no cylindrical extensions found)"
            chk_status = "Passed"

        del vessel
        del cleaned_mesh
        gc.collect()
        return {
            "dataset_id": dataset_id,
            "location": location,
            "status": "Saved",
            "reason": reason,
            "cuts_made": cuts_made,
            "integrity_check": chk_status,
        }

    except UnrealisticMeshError as ume:
        gc.collect()
        return {
            "dataset_id": dataset_id,
            "location": location,
            "status": "Skipped",
            "reason": str(ume),
            "cuts_made": 0,
            "integrity_check": "N/A",
        }
    except Exception as exc:
        gc.collect()
        return {
            "dataset_id": dataset_id,
            "location": location,
            "status": "Skipped",
            "reason": f"Error: {exc}",
            "cuts_made": 0,
            "integrity_check": "Error",
        }


def _run_task(task):
    """Top-level Pool target so spawn workers can pickle the call."""
    return process_patient_worker(*task)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Remove CFD flow extensions from open vessels labeled 2."
    )
    parser.add_argument(
        "--copy-unextended",
        action=argparse.BooleanOptionalAction,
        default=COPY_UNEXTENDED,
        help="Copy vessels that are not label 2 into the output folder (default: off).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help="Parallel spawn processes.",
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        default=None,
        help="Optional dataset ids to process (still must be label 2 unless --copy-unextended).",
    )
    return parser.parse_args()


def _consume_record(record, log_records, counts):
    log_records.append(record)
    if record["status"] == "Saved":
        counts["saved"] += 1
        counts["cuts"].append(record["cuts_made"])
        if record["integrity_check"] == "Failed":
            counts["integrity_failed"] += 1
            print(
                f"\nWarning: Integrity check failed (boundary count mismatch) "
                f"for {record['dataset_id']}"
            )
    elif record["status"] == "Skipped":
        counts["skipped"] += 1
        if record["reason"].startswith("Error:"):
            print(f"\nError processing patient {record['dataset_id']}: {record['reason']}")


def _location_map():
    if not os.path.exists(CSV_PATH):
        return {}
    clinical = pd.read_csv(CSV_PATH)
    if "dataset" not in clinical.columns or "location" not in clinical.columns:
        return {}
    return dict(
        zip(
            clinical["dataset"].astype(str).str.strip(),
            clinical["location"].astype(str).str.strip(),
        )
    )


def main():
    import shutil

    args = _parse_args()
    copy_unextended = bool(args.copy_unextended)

    if not os.path.exists(HASCAPOREXTENSION_CSV):
        print(f"Label CSV not found: {HASCAPOREXTENSION_CSV}")
        return

    labels = pd.read_csv(HASCAPOREXTENSION_CSV)
    if "Filename" not in labels.columns or "Label" not in labels.columns:
        print(f"{HASCAPOREXTENSION_CSV} must have Filename and Label columns.")
        return
    labels["Filename"] = labels["Filename"].astype(str).str.strip()
    labels["Label"] = labels["Label"].map(_norm_label)

    if args.ids:
        wanted = set(args.ids)
        labels = labels[labels["Filename"].isin(wanted)]

    n_ext = int((labels["Label"] == EXTENSION_LABEL).sum())
    print(f"Loaded {HASCAPOREXTENSION_CSV}: {len(labels)} rows, {n_ext} labeled {EXTENSION_LABEL}")
    print(f"Copy non-extension vessels: {copy_unextended}")

    if os.path.exists(OUTPUT_DIR):
        print(f"Deleting existing directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LABEL_OUTPUT_DIR, exist_ok=True)

    locations = _location_map()
    counts = {"saved": 0, "skipped": 0, "integrity_failed": 0, "cuts": []}
    log_records = []
    tasks = []

    for _, row in labels.iterrows():
        dataset_id = row["Filename"]
        label = row["Label"]
        location = locations.get(dataset_id, "")
        vessel_file = resolve_vessel_file(VESSEL_DIR, dataset_id)
        output_file = os.path.join(OUTPUT_DIR, f"{dataset_id}.vtp")
        if label != EXTENSION_LABEL and not copy_unextended:
            log_records.append({
                "dataset_id": dataset_id,
                "location": location,
                "status": "Skipped",
                "reason": "Not label 2 (copy-unextended is off)",
                "cuts_made": 0,
                "integrity_check": "N/A",
            })
            counts["skipped"] += 1
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
    except Exception as exc:
        print(f"Failed to write Excel file (openpyxl might be missing): {exc}")

    try:
        log_df.to_csv(log_csv, index=False)
        print(f"Saved CSV log to: {log_csv}")
    except Exception as exc:
        print(f"Failed to write CSV file: {exc}")

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
