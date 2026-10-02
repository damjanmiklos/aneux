"""Uniform parent-vessel template with the aneurysm rebuilt from four spheres.

The parent artery is the centerline's inscribed-radius tube, with the sac spike
taken out of that radius. The sac and its neck are the compact patch where the
wall stands off that tube. Four inscribed spheres, spread through the sac and
nudged so their union tracks the sac wall, are stamped together with the parent
tube. One marching-cubes surface is smoothed hard (the raw union is lumpy) and
remeshed uniformly.

Inputs are ``cleandata/uniformly_remeshed`` and ``cleandata/original_centerline``.
Meshes land in ``datatransform/template_creation/output_vessel_aneurysm``.

Ostium planes come from the ``.ostium_frames.npz`` sitting next to each uniform
mesh (the same frames the variable remesh consumes). They are squared onto the
ground-truth rims, used both as the opening stubs and as the uncap cutters, and
the finished opening count has to match them. After the uniform remesh the
surface gets the same close-out as that path: fold sub-0.01 mm edges, polish
slivers and crossings, and wind the lumen outward. Edge lengths stay uniform.

The hot path is ``manifold_from_spheres``: training can call it again with new
sphere centers and radii without repeating detection.
"""
import os
import sys

os.environ["VTK_OFFSCREEN"] = "1"
os.environ["EGL_PLATFORM"] = "surfaceless"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VTK_NUMBER_OF_THREADS"] = "1"
os.environ["VTK_SMP_MAX_THREADS"] = "1"

import argparse
import heapq
import time
from collections import deque
from contextlib import contextmanager

import numpy as np
import pyvista as pv
import vtk
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from aneux_paths import (
    CLEANDATA_ORIGINAL_CENTERLINE,
    CLEANDATA_UNIFORM,
    TEMPLATE_OUTPUT_VESSEL_ANEURYSM as DEFAULT_OUTPUT_DIR,
)

from batch_run_log import (
    add_run_log_args,
    configure_batch_logging,
    finalize_run_logs,
    run_logged_case,
)
from surface_polish import orient_outward, polish_surface
from variable_remeshing import cut_frames_to_profiles, resolve_cut_frames
from vessel_pipeline import (
    DEFAULT_EXTENSION_LENGTH,
    DEFAULT_TARGET_EDGE_LENGTH,
    OPENING_EXTENSION_LENGTH_FACTOR,
    OPENING_EXTENSION_SPACING_FACTOR,
    TemplateQualityError,
    add_shared_cli_args,
    apply_taubin_smoothing,
    assert_template_quality,
    assert_template_scale,
    clip_flow_extensions_and_uncap,
    collapse_tiny_edges,
    decimate_dense_mc,
    enforce_min_edge,
    extra_opening_spheres,
    finalize_surface,
    force_manifold_triangles,
    inspect_openings,
    inspect_surface_topology,
    keep_largest_region,
    measure_open_profiles,
    MIN_EDGE_LENGTH_MM,
    read_polydata,
    recompute_point_normals,
    remesh_surface_verified,
    repair_nonmanifold_triangles,
    run_batch,
    save_polydata,
    square_frames_to_rims,
    stamp_polyball_image,
    surface_genus,
    to_vtk_poly,
    uniform_edge_length_for_profiles,
    WALL_PINHOLE_RADIUS_MM,
    with_dataset_id,
    _grow_solid_ball,
)

LOG_FOLDER = "vessel_aneurysm_logs"
# A finished template under half the vessel area is a fragment, not a sac that
# four spheres could not quite fill (those still land near 0.75).
MIN_TEMPLATE_AREA_RATIO = 0.50
POST_UNCAP_POINTS = 16000

# Coarser than the parent-tube modeller: the result is smoothed hard and then
# remeshed at ~0.5 mm, so a 0.22 mm voxel is enough and several times cheaper.
DEFAULT_GRID_SPACING = 0.22
DEFAULT_MAX_GRID_SIZE = 360
PARENT_SAMPLE_MM = 0.42
SMOOTH_PASSBAND = 0.04
SMOOTH_ITERS = 36
N_SPHERES = 4


def _faces_of(mesh):
    faces = np.asarray(pv.wrap(mesh).faces)
    if faces.size == 0 or faces.size % 4 != 0:
        raise TemplateQualityError("vessel mesh has no triangle faces")
    faces = faces.reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        raise TemplateQualityError("vessel mesh is not a triangle mesh")
    return np.ascontiguousarray(faces[:, 1:], dtype=np.int64)


def _dedup_centerline(cpts, rad, lines, tol=0.14):
    """One point per blob of duplicated tract samples, and the unique edges."""
    cpts = np.asarray(cpts, dtype=np.float64)
    rad = np.asarray(rad, dtype=np.float64).reshape(-1)
    key = np.round(cpts / tol).astype(np.int64)
    _uniq, inv = np.unique(key, axis=0, return_inverse=True)
    n = int(inv.max()) + 1 if len(inv) else 0
    up = np.zeros((n, 3), dtype=np.float64)
    ur = np.full(n, -1.0, dtype=np.float64)
    counts = np.zeros(n, dtype=np.int64)
    np.add.at(up, inv, cpts)
    counts += np.bincount(inv, minlength=n)
    # Maximal inscribed radius at a repeated sample.
    np.maximum.at(ur, inv, rad)
    up /= np.maximum(counts, 1)[:, None]
    ur = np.where(ur < 0.0, 0.35, ur)
    edges = []
    raw = np.asarray(lines, dtype=np.int64)
    offset = 0
    while offset < len(raw):
        count = int(raw[offset])
        ids = raw[offset + 1 : offset + 1 + count]
        offset += count + 1
        if count < 2:
            continue
        mapped = inv[ids]
        a = mapped[:-1]
        b = mapped[1:]
        keep = a != b
        if np.any(keep):
            edges.append(np.stack([a[keep], b[keep]], axis=1))
    if edges:
        e = np.unique(np.sort(np.vstack(edges), axis=1), axis=0)
    else:
        e = np.zeros((0, 2), dtype=np.int64)
    return up, ur, e


def _graph_filter(values, edges, steps, reduce):
    v = np.asarray(values, dtype=np.float64).copy()
    if len(edges) == 0 or steps <= 0:
        return v
    a = edges[:, 0]
    b = edges[:, 1]
    for _ in range(int(steps)):
        m = v.copy()
        src_b = v[b]
        src_a = v[a]
        if reduce == "min":
            np.minimum.at(m, a, src_b)
            np.minimum.at(m, b, src_a)
        else:
            np.maximum.at(m, a, src_b)
            np.maximum.at(m, b, src_a)
        v = m
    return v


def _parent_radius(ur, edges, up):
    """Opening of the radius along the centerline, so a short sac spike is removed."""
    if len(edges):
        length = np.linalg.norm(up[edges[:, 0]] - up[edges[:, 1]], axis=1)
        usable = length[length > 1e-4]
        step = float(np.median(usable)) if usable.size else 0.25
    else:
        step = 0.25
    k = int(np.clip(round(3.5 / max(step, 0.05)), 2, 48))
    opened = _graph_filter(_graph_filter(ur, edges, k, "min"), edges, k, "max")
    return np.maximum(opened, 0.25)


def _mesh_edges(faces):
    e = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    e.sort(axis=1)
    return np.unique(e, axis=0)


def _component_with_seed(mask, edges, n, seed):
    sel = np.flatnonzero(mask)
    out = np.zeros(n, dtype=bool)
    if sel.size == 0 or not mask[seed]:
        return out
    remap = np.full(n, -1, dtype=np.int64)
    remap[sel] = np.arange(sel.size)
    a = remap[edges[:, 0]]
    b = remap[edges[:, 1]]
    keep = (a >= 0) & (b >= 0)
    a = a[keep]
    b = b[keep]
    if a.size == 0:
        out[seed] = True
        return out
    graph = csr_matrix(
        (np.ones(a.size * 2, dtype=np.float32), (np.concatenate((a, b)), np.concatenate((b, a)))),
        shape=(sel.size, sel.size),
    )
    _nlab, labels = connected_components(graph, directed=False)
    out[sel[labels == labels[remap[seed]]]] = True
    return out


def _component_diameter(pts, idx):
    chunk = pts[idx]
    cen = chunk.mean(axis=0)
    radius = np.linalg.norm(chunk - cen, axis=1)
    return float(2.0 * radius.max()), cen


def _grow_neck(core, eligible, edges, median_edge, max_mm):
    """Walk a short distance off the sac through vertices that are still bulging."""
    max_hops = int(max(1, round(max_mm / max(median_edge, 1e-3))))
    n = len(core)
    adj = [[] for _ in range(n)]
    for i, j in edges:
        adj[int(i)].append(int(j))
        adj[int(j)].append(int(i))
    out = core.copy()
    hops = np.full(n, -1, dtype=np.int32)
    hops[core] = 0
    q = deque(int(i) for i in np.flatnonzero(core))
    while q:
        u = q.popleft()
        if hops[u] >= max_hops:
            continue
        nxt = hops[u] + 1
        for v in adj[u]:
            if hops[v] >= 0 or not eligible[v]:
                continue
            hops[v] = nxt
            out[v] = True
            q.append(v)
    return out


def detect_aneurysm(pts, faces, cpts, rad, lines):
    """Sac plus a short neck lip. Almost none of the parent wall.

    The dome is the vertex farthest outside the parent tube. Its level-set
    component is kept at the threshold whose compact, high bulge scores best,
    which is the step before the set spills along the parent. A geodesic lip
    of about one parent radius then picks up the neck.
    """
    t0 = time.perf_counter()
    up, ur, edges = _dedup_centerline(cpts, rad, lines)
    r_parent = _parent_radius(ur, edges, up)
    dist, nearest = cKDTree(up).query(pts, k=1, workers=1)
    nearest = np.asarray(nearest, dtype=np.int64)
    excess = dist - r_parent[nearest]
    med_edges = _mesh_edges(faces)
    elen = np.linalg.norm(pts[med_edges[:, 0]] - pts[med_edges[:, 1]], axis=1)
    median_edge = float(np.median(elen)) if elen.size else 0.4
    dome = int(np.argmax(excess))
    peak = float(excess[dome])
    r_dome = float(max(r_parent[nearest[dome]], 0.45))
    char = float(np.clip(1.15 * peak + 2.0 * r_dome, 4.0, 16.0))
    d_cap = float(np.clip(1.55 * char, 8.0, 22.0))
    levels = np.linspace(max(0.55, min(0.72 * peak, peak - 0.15)), 0.28, 8)
    best_score = -1.0
    core = np.zeros(len(pts), dtype=bool)
    core_diam = 0.0
    core_cen = pts[dome].copy()
    for level in levels:
        comp = _component_with_seed(excess >= float(level), med_edges, len(pts), dome)
        n_comp = int(comp.sum())
        if n_comp < 40:
            continue
        diam, cen = _component_diameter(pts, np.flatnonzero(comp))
        med_ex = float(np.median(excess[comp]))
        frac = n_comp / len(pts)
        if diam > d_cap or frac > 0.34 or med_ex < 0.45:
            continue
        score = (n_comp * med_ex) / (1.0 + (diam / char) ** 2)
        if score > best_score:
            best_score = score
            core = comp
            core_diam = diam
            core_cen = cen
    if not core.any():
        core = _component_with_seed(excess >= max(0.4, 0.35 * peak), med_edges, len(pts), dome)
        if core.any():
            core_diam, core_cen = _component_diameter(pts, np.flatnonzero(core))
    core_ids = nearest[core]
    r_neck = float(np.median(r_parent[core_ids])) if core_ids.size else r_dome
    neck_mm = float(np.clip(0.70 * r_neck, 0.55, 1.8))
    near_core = np.linalg.norm(pts - core_cen, axis=1) <= (0.5 * max(core_diam, 1.0) + neck_mm)
    eligible = near_core & (excess > 0.12) & ~core
    mask = _grow_neck(core, eligible, med_edges, median_edge, neck_mm)
    high = excess > max(0.9, 0.5 * peak)
    high_near = high & (np.linalg.norm(pts - core_cen, axis=1) <= max(d_cap, core_diam))
    info = {
        "peak_mm": peak,
        "frac": float(mask.mean()) if len(mask) else 0.0,
        "diameter_mm": float(core_diam),
        "median_excess_mm": float(np.median(excess[mask])) if mask.any() else 0.0,
        "neck_frac": float((mask & ~core).sum() / max(int(mask.sum()), 1)),
        "low_excess_frac": float((mask & (excess < 0.40)).sum() / max(int(mask.sum()), 1)),
        "local_recall": float((high_near & mask).sum() / max(int(high_near.sum()), 1)),
        "seconds": time.perf_counter() - t0,
    }
    return mask, excess, up, r_parent, edges, info


def _point_normals(pts, faces):
    mesh = pv.PolyData(
        np.ascontiguousarray(pts),
        np.hstack((np.full((len(faces), 1), 3, dtype=np.int64), faces)).ravel(),
    )
    filt = vtk.vtkPolyDataNormals()
    filt.SetInputData(mesh)
    filt.ComputePointNormalsOn()
    filt.SplittingOff()
    filt.ConsistencyOn()
    filt.AutoOrientNormalsOff()
    filt.Update()
    normals = np.asarray(vtk_to_numpy(filt.GetOutput().GetPointData().GetNormals()), dtype=np.float64)
    if normals.shape != pts.shape:
        raise TemplateQualityError("point normals could not be computed")
    return normals


def _resample_parent(up, ur, edges, spacing=PARENT_SAMPLE_MM):
    """Parent spheres every ``spacing`` mm. Overlapping balls do not need the 0.1 mm trace."""
    chunks_p = [up]
    chunks_r = [ur]
    spacing = float(spacing)
    for i, j in edges:
        p0 = up[i]
        p1 = up[j]
        dist = float(np.linalg.norm(p1 - p0))
        n = int(dist / spacing)
        if n <= 1:
            continue
        t = np.linspace(0.0, 1.0, n, endpoint=False)[1:]
        chunks_p.append(p0 + t[:, None] * (p1 - p0))
        chunks_r.append(ur[i] + t * (ur[j] - ur[i]))
    pts = np.vstack(chunks_p)
    rad = np.concatenate(chunks_r)
    key = np.round(pts / (0.55 * spacing)).astype(np.int64)
    _u, inv = np.unique(key, axis=0, return_inverse=True)
    n = int(inv.max()) + 1
    acc_p = np.zeros((n, 3), dtype=np.float64)
    acc_r = np.zeros(n, dtype=np.float64)
    counts = np.bincount(inv, minlength=n).astype(np.float64)
    np.add.at(acc_p, inv, pts)
    np.add.at(acc_r, inv, rad)
    acc_p /= counts[:, None]
    acc_r /= counts
    return acc_p, np.maximum(acc_r, 0.25)


def fit_aneurysm_spheres(pts, faces, mask, excess, up, r_parent, normals=None):
    """Four inscribed spheres whose union tracks the sac. Cheap, not exhaustive.

    Two starts (equal steps along the neck-to-dome axis, and a greedy cover by
    medial balls) are each slid toward the wall they still miss. The lower
    surface residual wins. Centers stay inside the lumen; radii are the
    distance to the wall.
    """
    t0 = time.perf_counter()
    sac_idx = np.flatnonzero(mask)
    if sac_idx.size < 30:
        raise TemplateQualityError("aneurysm mask is too small to place spheres")
    sac = pts[sac_idx]
    if normals is None:
        normals = _point_normals(pts, faces)
    wall_step = max(1, len(pts) // 40000)
    wall = np.ascontiguousarray(pts[::wall_step])
    nrm = np.ascontiguousarray(normals[::wall_step])
    wall_tree = cKDTree(wall)
    sac_s = np.ascontiguousarray(sac[:: max(1, len(sac) // 1800)])
    lo = sac.min(axis=0) - 1.0
    hi = sac.max(axis=0) + 1.0
    vox = 0.40
    if np.max(hi - lo) / vox > 52:
        vox = float(np.max(hi - lo) / 48.0)
    axes = [np.arange(lo[i], hi[i] + 0.5 * vox, vox) for i in range(3)]
    xx, yy, zz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    grid = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)
    d_wall, i_wall = wall_tree.query(grid, k=1, workers=1)
    inward = np.einsum("ij,ij->i", grid - wall[i_wall], nrm[i_wall])
    probe = up[:: max(1, len(up) // 300)]
    _dp, ip = wall_tree.query(probe, k=1, workers=1)
    orient = float(np.median(np.einsum("ij,ij->i", probe - wall[ip], nrm[ip])))
    if orient > 0.0:
        inward = -inward
    inside = inward < -0.05
    sac_tree = cKDTree(sac_s)
    d_sac, _ = sac_tree.query(grid, k=1, workers=1)
    in_sac = inside & (d_sac <= d_wall + 0.35) & (d_wall > 0.25)
    radii_grid = np.zeros(len(grid), dtype=np.float64)
    radii_grid[in_sac] = d_wall[in_sac]
    shape = (len(axes[0]), len(axes[1]), len(axes[2]))
    rvol = radii_grid.reshape(shape)
    is_max = (rvol > 0.45) & (rvol >= ndimage.maximum_filter(rvol, size=3) - 1e-8)
    is_max &= rvol > ndimage.minimum_filter(rvol, size=3) + 0.05
    maxima = np.argwhere(is_max)
    if len(maxima) == 0:
        flat = int(np.argmax(radii_grid))
        maxima = np.array([np.unravel_index(flat, shape)], dtype=np.int64)
    cand_c = np.stack([axes[k][maxima[:, k]] for k in range(3)], axis=1)
    cand_r = rvol[maxima[:, 0], maxima[:, 1], maxima[:, 2]]
    order = np.argsort(cand_r)[::-1]
    cand_c = cand_c[order]
    cand_r = cand_r[order]
    keep = []
    for i, (center, rad) in enumerate(zip(cand_c, cand_r)):
        if any(np.linalg.norm(center - cand_c[j]) < 0.55 * cand_r[j] for j in keep):
            continue
        keep.append(i)
        if len(keep) >= 16:
            break
    cand_c = cand_c[keep]
    cand_r = cand_r[keep]

    parent_tree = cKDTree(up)
    k_parent = min(5, len(up))
    _pd, pidx = parent_tree.query(sac_s, k=k_parent, workers=1)
    pidx = np.atleast_2d(np.asarray(pidx, dtype=np.int64))
    if pidx.shape[0] == 1 and sac_s.shape[0] != 1:
        pidx = pidx.T
    parent_res = np.min(
        np.linalg.norm(sac_s[:, None, :] - up[pidx], axis=2) - r_parent[pidx], axis=1
    )

    def sphere_residual(spheres):
        res = parent_res.copy()
        for center, rad in spheres:
            res = np.minimum(res, np.linalg.norm(sac_s - center, axis=1) - rad)
        return res

    def loss_of(spheres):
        res = sphere_residual(spheres)
        penalty = np.where(res < 0.0, 3.0 * res * res, res * res)
        return float(np.mean(penalty))

    def inscribed(center):
        center = np.asarray(center, dtype=np.float64)
        dist_w, j = wall_tree.query(center, k=1)
        side = float(np.dot(center - wall[j], nrm[j]))
        if orient > 0.0:
            side = -side
        if side >= -0.02:
            return None
        return float(dist_w)

    sac_grid = grid[in_sac]
    sac_r = radii_grid[in_sac]
    dome_pt = pts[sac_idx[int(np.argmax(excess[sac_idx]))]]
    neck_i = int(parent_tree.query(sac.mean(axis=0), k=1)[1])
    neck = up[neck_i]
    axis = dome_pt - neck
    axis_len = float(np.linalg.norm(axis))
    axis_dir = axis / axis_len if axis_len > 1e-6 else np.array([0.0, 0.0, 1.0])

    def axis_init():
        chosen = []
        if sac_grid.size == 0:
            return chosen
        proj = (sac_grid - neck) @ axis_dir
        for frac in (0.30, 0.50, 0.70, 0.88):
            target = frac * max(axis_len, 1.0)
            band = np.abs(proj - target) <= max(0.16 * axis_len, 0.7)
            pool_c = sac_grid[band] if np.any(band) else sac_grid
            pool_r = sac_r[band] if np.any(band) else sac_r
            j = int(np.argmax(pool_r))
            center = pool_c[j]
            rad = float(pool_r[j])
            if any(np.linalg.norm(center - oc) < 0.45 * max(rad, orad) for oc, orad in chosen):
                center = neck + min(target, axis_len * 0.92) * axis_dir
                rad = inscribed(center)
                if rad is None:
                    continue
            chosen.append((np.asarray(center, dtype=np.float64), float(rad)))
        return chosen[:N_SPHERES]

    def greedy_init():
        chosen = []
        used = np.zeros(len(cand_c), dtype=bool)
        for _ in range(N_SPHERES):
            base = sphere_residual(chosen) if chosen else parent_res
            best_i = None
            best_gain = 0.0
            for i, (center, rad) in enumerate(zip(cand_c, cand_r)):
                if used[i]:
                    continue
                hit = np.linalg.norm(sac_s - center, axis=1) - float(rad)
                gain = float(np.mean(np.clip(base, 0.0, None) - np.clip(np.minimum(base, hit), 0.0, None)))
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
            if best_i is None:
                break
            used[best_i] = True
            chosen.append((np.asarray(cand_c[best_i], dtype=np.float64), float(cand_r[best_i])))
        return chosen

    def fill_to_four(chosen):
        chosen = list(chosen)
        guard = 0
        while len(chosen) < N_SPHERES and guard < 6:
            guard += 1
            base = sphere_residual(chosen) if chosen else parent_res
            target = sac_s[int(np.argmax(np.clip(base, 0.0, None)))]
            center = 0.55 * target + 0.45 * sac.mean(axis=0)
            rad = inscribed(center)
            if rad is None and sac_grid.size:
                j = int(np.argmax(sac_r))
                center = sac_grid[j]
                rad = float(sac_r[j])
            if rad is None:
                break
            chosen.append((np.asarray(center, dtype=np.float64), float(rad)))
        return chosen[:N_SPHERES]

    def polish(chosen):
        chosen = [ (np.asarray(c, dtype=np.float64), float(r)) for c, r in chosen ]
        for _it in range(7):
            base_loss = loss_of(chosen)
            improved = False
            for si, (center, rad) in enumerate(list(chosen)):
                others = [sp for k, sp in enumerate(chosen) if k != si]
                others_res = parent_res.copy()
                for oc, orad in others:
                    others_res = np.minimum(others_res, np.linalg.norm(sac_s - oc, axis=1) - orad)
                miss = np.clip(others_res, 0.0, None)
                if float(miss.max()) < 0.15:
                    continue
                target = sac_s[int(np.argmax(miss))]
                direction = target - center
                dn = float(np.linalg.norm(direction))
                if dn < 1e-6:
                    continue
                direction /= dn
                best = (center, rad, base_loss)
                for step in (0.3, 0.65, 1.1, 1.7):
                    trial = center + step * direction
                    tr = inscribed(trial)
                    if tr is None or tr < 0.3:
                        continue
                    trial_set = list(chosen)
                    trial_set[si] = (trial, tr)
                    tl = loss_of(trial_set)
                    if tl + 1e-6 < best[2]:
                        best = (trial, tr, tl)
                if best[2] + 1e-4 < base_loss:
                    chosen[si] = (best[0], best[1])
                    base_loss = best[2]
                    improved = True
            if not improved:
                break
        return chosen

    starts = [fill_to_four(axis_init()), fill_to_four(greedy_init())]
    polished = [polish(s) for s in starts if s]
    if not polished:
        raise TemplateQualityError("could not place aneurysm spheres inside the sac")
    chosen = min(polished, key=loss_of)

    r_neck = float(r_parent[neck_i])
    def _meets_parent(center, rad):
        if np.linalg.norm(center - neck) < rad + r_neck - 0.15:
            return True
        near = parent_tree.query_ball_point(center, rad + 3.0)
        return any(
            np.linalg.norm(center - up[j]) < rad + float(r_parent[j]) - 0.15 for j in near
        )

    if chosen and not any(_meets_parent(c, rad) for c, rad in chosen):
        dists = [float(np.linalg.norm(c - neck)) for c, _rad in chosen]
        si = int(np.argmin(dists))
        center, _rad = chosen[si]
        direction = neck - center
        dn = float(np.linalg.norm(direction))
        if dn > 1e-6:
            for step in np.linspace(0.25, max(dn - 0.15, 0.25), 7):
                trial = center + (step / dn) * direction
                tr = inscribed(trial)
                if tr is not None and np.linalg.norm(trial - neck) < tr + r_neck - 0.1:
                    chosen[si] = (trial, tr)
                    break

    # Push each sphere toward the wall it still falls short of, and stop before
    # it crosses that wall. One inscribed ball misses the far side of a long sac.
    for _grow in range(5):
        res_now = sphere_residual(chosen)
        moved = False
        for si, (center, rad) in enumerate(list(chosen)):
            dist_s = np.linalg.norm(sac_s - center, axis=1) - rad
            owns = dist_s <= res_now + 1e-9
            if int(owns.sum()) < 12:
                continue
            short = float(np.median(res_now[owns]))
            if short < 0.25:
                continue
            trial_r = rad + min(0.55 * short, 0.7)
            trial = list(chosen)
            trial[si] = (center, trial_r)
            trial_res = sphere_residual(trial)
            if float(np.mean(trial_res < -0.3)) > 0.012:
                continue
            if float(np.mean(trial_res ** 2)) + 1e-6 < float(np.mean(res_now ** 2)):
                chosen[si] = (center, float(trial_r))
                moved = True
        if not moved:
            break

    final_res = sphere_residual(chosen)
    parent_rmse = float(np.sqrt(np.mean(np.clip(parent_res, 0.0, None) ** 2)))
    union_rmse = float(np.sqrt(np.mean(final_res ** 2)))
    centers = np.vstack([c for c, _r in chosen]) if chosen else np.zeros((0, 3))
    radii = np.array([r for _c, r in chosen], dtype=np.float64)
    info = {
        "n_spheres": int(len(chosen)),
        "parent_rmse_mm": parent_rmse,
        "union_rmse_mm": union_rmse,
        "cover_frac": float(np.mean(np.abs(final_res) <= 0.65)),
        "bulge_frac": float(np.mean(final_res < -0.35)),
        "radii_mm": radii,
        "neck_xyz": np.asarray(neck, dtype=np.float64),
        "neck_radius_mm": r_neck,
        "seconds": time.perf_counter() - t0,
    }
    return centers, radii, info


def _bridge_to_parent(centers, radii, parent_pts, parent_r, neck, neck_r):
    """Balls from the largest sphere to the neck, overlapping by a voxel or more.

    A center-distance test against every parent sphere calls a bend that merely
    passes the dome a connection, and marching cubes then keeps that kiss and
    drops the rest of the sac. The neck point is the centerline sample under
    the sac, so the chain joins the lumen that actually feeds it.
    """
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    radii = np.asarray(radii, dtype=np.float64).reshape(-1)
    if len(centers) == 0:
        return centers, radii
    neck = np.asarray(neck, dtype=np.float64).reshape(3)
    neck_r = float(neck_r)
    si = int(np.argmax(radii))
    gap = neck - centers[si]
    dist = float(np.linalg.norm(gap))
    if dist < 1e-6:
        return centers, radii
    margin = float(radii[si]) + neck_r - dist
    if margin >= 0.60:
        return centers, radii
    direction = gap / dist
    r_join = max(0.85 * neck_r, 0.55)
    step = max(0.45 * r_join, 0.3)
    extra_c = []
    extra_r = []
    travel = max(float(radii[si]) - 0.3, step)
    while travel < dist - 0.25 * neck_r and len(extra_c) < 10:
        extra_c.append(centers[si] + travel * direction)
        extra_r.append(r_join)
        travel += step
    extra_c.append(neck - min(0.4 * neck_r, 0.45 * dist) * direction)
    extra_r.append(max(neck_r, r_join))
    print(
        f"  Bridged the sac to the neck with {len(extra_c)} spheres "
        f"(gap {dist:.2f} mm, overlap {margin:.2f} mm)"
    )
    return (
        np.vstack((centers, np.asarray(extra_c, dtype=np.float64))),
        np.concatenate((radii, np.asarray(extra_r, dtype=np.float64))),
    )


def _parent_seed_zyx(parent_pts, parent_r, origin, spacing, field):
    """A voxel inside the thickest parent sphere, in (z, y, x) field order."""
    j = int(np.argmax(parent_r))
    p = np.asarray(parent_pts[j], dtype=np.float64)
    ix = int(np.round((p[0] - origin[0]) / spacing))
    iy = int(np.round((p[1] - origin[1]) / spacing))
    iz = int(np.round((p[2] - origin[2]) / spacing))
    nz, ny, nx = field.shape
    best = None
    best_d = 1e9
    for dz in range(-2, 3):
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                z, y, x = iz + dz, iy + dy, ix + dx
                if not (0 <= z < nz and 0 <= y < ny and 0 <= x < nx):
                    continue
                if field[z, y, x] >= 0.0:
                    continue
                d = dx * dx + dy * dy + dz * dz
                if d < best_d:
                    best_d = d
                    best = (z, y, x)
    return best


def _untangle_from_parent(field, spacing, seed_zyx):
    """Cut handles, starting in the parent lumen, and refuse a cut that guts it.

    The shared untangle seeds the most negative voxel. That is the aneurysm
    when one of its spheres is larger than the parent, and the neck is then
    too thin for the growth to cross, so the parent tube is deleted.
    """
    backup = np.array(field, copy=True)
    inside = field < 0.0
    idx = np.argwhere(inside)
    if len(idx) == 0:
        return 0, 0, False
    lo = np.maximum(idx.min(axis=0) - 1, 0)
    hi = idx.max(axis=0) + 2
    box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    sub = field[box]
    labeled, _nlab = ndimage.label(inside[box])
    if seed_zyx is None:
        comp_id = 0
    else:
        sz0, sy0, sx0 = (int(seed_zyx[k] - int(lo[k])) for k in range(3))
        if (
            0 <= sz0 < labeled.shape[0]
            and 0 <= sy0 < labeled.shape[1]
            and 0 <= sx0 < labeled.shape[2]
        ):
            comp_id = int(labeled[sz0, sy0, sx0])
        else:
            comp_id = 0
    if comp_id == 0:
        comp = inside[box]
    else:
        comp = labeled == comp_id
    solid = ndimage.binary_fill_holes(comp)
    filled = solid & ~comp
    depth = np.array(sub, copy=True)
    if seed_zyx is not None:
        sz, sy, sx = (int(seed_zyx[k] - int(lo[k])) for k in range(3))
        if (
            0 <= sz < depth.shape[0]
            and 0 <= sy < depth.shape[1]
            and 0 <= sx < depth.shape[2]
            and bool(solid[sz, sy, sx])
        ):
            depth[sz, sy, sx] = np.float32(-1.0e9)
    grown = _grow_solid_ball(solid, depth)
    cut = solid & ~grown
    n_cut = int(cut.sum())
    n_filled = int(filled.sum())
    # A real handle is a handful of voxels. Tens of thousands means the growth
    # never entered the other part of the vessel.
    if n_cut > max(800, int(0.02 * int(solid.sum()))):
        field[...] = backup
        return n_cut, n_filled, False
    gap = np.float32(0.25 * spacing * spacing)
    sub[filled] = -gap
    sub[cut] = np.maximum(-sub[cut], gap)
    return n_cut, n_filled, True


def _isosurface(image, level):
    """Flying-edges contour, including every component."""
    flying = vtk.vtkFlyingEdges3D()
    flying.SetInputData(image)
    flying.SetValue(0, float(level))
    flying.Update()
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(flying.GetOutput())
    tri.Update()
    return to_vtk_poly(tri.GetOutput())


def _component_point_sets(surface):
    """Point arrays of each connected component, largest first."""
    poly = to_vtk_poly(surface)
    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(poly)
    conn.SetExtractionModeToAllRegions()
    conn.ColorRegionsOn()
    conn.Update()
    labelled = conn.GetOutput()
    region = labelled.GetPointData().GetArray("RegionId")
    if region is None or int(conn.GetNumberOfExtractedRegions()) <= 1:
        if poly.GetPoints() is None:
            return [np.zeros((0, 3), dtype=np.float64)]
        return [np.asarray(vtk_to_numpy(poly.GetPoints().GetData()), dtype=np.float64)]
    ids = np.asarray(vtk_to_numpy(region), dtype=np.int64).reshape(-1)
    pts = np.asarray(vtk_to_numpy(labelled.GetPoints().GetData()), dtype=np.float64)
    groups = [np.ascontiguousarray(pts[ids == rid]) for rid in np.unique(ids)]
    groups.sort(key=len, reverse=True)
    return groups


def _paint_ball(field, grid_origin, spacing, center, radius):
    """Write one sphere into the polyball field (negative inside)."""
    nz, ny, nx = field.shape
    r = float(radius)
    reach = r + float(spacing)
    ox, oy, oz = (float(v) for v in grid_origin)
    i0 = max(0, int(np.floor((center[0] - reach - ox) / spacing)))
    i1 = min(nx, int(np.ceil((center[0] + reach - ox) / spacing)) + 1)
    j0 = max(0, int(np.floor((center[1] - reach - oy) / spacing)))
    j1 = min(ny, int(np.ceil((center[1] + reach - oy) / spacing)) + 1)
    k0 = max(0, int(np.floor((center[2] - reach - oz) / spacing)))
    k1 = min(nz, int(np.ceil((center[2] + reach - oz) / spacing)) + 1)
    if i1 <= i0 or j1 <= j0 or k1 <= k0:
        return
    xs = ox + np.arange(i0, i1, dtype=np.float64) * spacing
    ys = oy + np.arange(j0, j1, dtype=np.float64) * spacing
    zs = oz + np.arange(k0, k1, dtype=np.float64) * spacing
    dx = xs[None, None, :] - float(center[0])
    dy = ys[None, :, None] - float(center[1])
    dz = zs[:, None, None] - float(center[2])
    val = (dx * dx + dy * dy + dz * dz - r * r).astype(np.float32)
    slab = field[k0:k1, j0:j1, i0:i1]
    np.minimum(slab, val, out=slab)


def _paint_capsule(field, grid_origin, spacing, start, end, radius):
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    span = float(np.linalg.norm(end - start))
    step = max(0.45 * float(radius), float(spacing))
    n = max(int(np.ceil(span / step)), 1)
    for t in np.linspace(0.0, 1.0, n + 1):
        _paint_ball(field, grid_origin, spacing, start + t * (end - start), radius)


def _outlet_on_island(island, main_tree, ostia):
    """True when an ostium sits on this island and not on the main vessel."""
    if len(island) < 80 or ostia is None or len(ostia) == 0:
        return False
    sample = island[:: max(1, len(island) // 400)]
    for origin in np.asarray(ostia, dtype=np.float64).reshape(-1, 3):
        d_island = float(np.linalg.norm(sample - origin, axis=1).min())
        d_main = float(main_tree.query(origin)[0])
        # The ostium sitting on the island is that outlet, even when the cut
        # that tore it off left the main vessel just as close.
        if d_island < 1.2 or (d_island < 2.5 and d_island + 0.5 < d_main):
            return True
    return False


def _contour_keeping_outlets(field, image, level, ostia, grid_origin, spacing):
    """Contour, and neck an outlet back on if the contour tore it off.

    An inside contour that opens a one-voxel kiss also severs a thin outlet.
    Keeping only the largest piece then drops that outlet, and the case fails
    with a shut ostium on a genus-0 vessel. A short neck between the two
    pieces puts the outlet back without the kiss.

    Returns ``(surface, stranded)``. ``stranded`` means an ostium is still on
    a piece that is not the main vessel, so this contour must not be kept.
    """
    raw = _isosurface(image, level)
    groups = _component_point_sets(raw)
    if len(groups) <= 1:
        return keep_largest_region(raw), False
    main = groups[0]
    tree = cKDTree(main)
    necks = []
    for island in groups[1:]:
        if not _outlet_on_island(island, tree, ostia):
            continue
        sample = island[:: max(1, len(island) // 800)]
        dist, index = tree.query(sample)
        j = int(np.argmin(dist))
        necks.append((sample[j], main[int(index[j])], float(dist[j])))
    if not necks:
        return keep_largest_region(raw), False
    for start, end, _gap in necks:
        _paint_capsule(field, grid_origin, spacing, start, end, 0.45)
    scalars = image.GetPointData().GetScalars()
    scalars.Modified()
    image.Modified()
    rejoined = _isosurface(image, level)
    groups = _component_point_sets(rejoined)
    stranded = False
    if len(groups) > 1:
        tree = cKDTree(groups[0])
        stranded = any(_outlet_on_island(island, tree, ostia) for island in groups[1:])
    if stranded:
        return keep_largest_region(rejoined), True
    print(
        f"  Rejoined {len(necks)} outlet(s) the {level:.2f} contour had torn off"
    )
    return keep_largest_region(rejoined), False


def _apply_ostium_caps(field, grid_origin, spacing, ostia, parent_pts=None, parent_r=None):
    """Leave a thin outlet past each ostium and clear only the sac bulge around it.

    A sac sphere that bulges past the ostium plane hides the real end: the
    cutter is a narrow cylinder and only nicks that bulge, and the remesh then
    seals the slit. Only the lip just outside the outlet tube is cleared, a
    couple of millimetres out, so a branch further along the same plane stays.
    The outlet tube is written in firmly enough that a later inside contour
    cannot erase a thin one.
    """
    if not ostia:
        return
    nz, ny, nx = field.shape
    ox, oy, oz = (float(v) for v in grid_origin)
    spacing = float(spacing)
    jobs = []
    for origin, outward, radius in ostia:
        origin = np.asarray(origin, dtype=np.float64).reshape(3)
        outward = np.asarray(outward, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(outward))
        outward = outward / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])
        radius = max(float(radius), 0.2)
        clip_r = max(1.5 * radius, radius + 0.2)
        stub_r = min(max(radius, 0.36), 0.80 * clip_r)
        stub_len = max(1.6, 2.2 * radius)
        inward = max(1.4, 2.2 * radius)
        ends = np.vstack((
            origin - inward * outward,
            origin + (stub_len + 1.0) * outward,
        ))
        margin = 6.0
        lo = ends.min(axis=0) - margin
        hi = ends.max(axis=0) + margin
        i0 = max(0, int(np.floor((lo[0] - ox) / spacing)))
        i1 = min(nx, int(np.ceil((hi[0] - ox) / spacing)) + 1)
        j0 = max(0, int(np.floor((lo[1] - oy) / spacing)))
        j1 = min(ny, int(np.ceil((hi[1] - oy) / spacing)) + 1)
        k0 = max(0, int(np.floor((lo[2] - oz) / spacing)))
        k1 = min(nz, int(np.ceil((hi[2] - oz) / spacing)) + 1)
        if i1 <= i0 or j1 <= j0 or k1 <= k0:
            continue
        xs = ox + np.arange(i0, i1, dtype=np.float64) * spacing
        ys = oy + np.arange(j0, j1, dtype=np.float64) * spacing
        zs = oz + np.arange(k0, k1, dtype=np.float64) * spacing
        rel_x = xs[None, None, :] - origin[0]
        rel_y = ys[None, :, None] - origin[1]
        rel_z = zs[:, None, None] - origin[2]
        proj = rel_x * outward[0] + rel_y * outward[1] + rel_z * outward[2]
        rad2 = rel_x * rel_x + rel_y * rel_y + rel_z * rel_z - proj * proj
        slab = field[k0:k1, j0:j1, i0:i1]
        solid = (rad2 <= stub_r * stub_r) & (proj >= -inward) & (proj <= stub_len)
        inside = slab < 0.0
        carve = (
            (proj > 0.20)
            & (proj < stub_len + 0.8)
            & (rad2 > (stub_r * 1.25) ** 2)
            & (rad2 < 36.0)
            & inside
        )
        jobs.append((slab, solid, carve))
    # Carve every lip first, then write every stub. A neighbour's 6 mm carve
    # reaches the next branch; writing the stubs afterwards puts that branch
    # back. Writing them in the same pass left the later carve as the last word
    # and the outlet was gone before the cutter ran.
    for slab, _solid, carve in jobs:
        slab[carve] = np.float32(0.35)
    for slab, solid, _carve in jobs:
        slab[solid] = np.minimum(slab[solid], np.float32(-0.5))
    _separate_close_tips(field, (ox, oy, oz), spacing, ostia, parent_pts, parent_r)


def _parent_path_mm(adj, start, goal):
    """Shortest path along parent-sample links, or a huge number when none exists."""
    if start == goal:
        return 0.0
    best = {start: 0.0}
    heap = [(0.0, start)]
    while heap:
        dist, node = heapq.heappop(heap)
        if node == goal:
            return dist
        if dist > best.get(node, 1e18):
            continue
        for nxt, weight in adj[node]:
            nd = dist + weight
            if nd < best.get(nxt, 1e18):
                best[nxt] = nd
                heapq.heappush(heap, (nd, nxt))
    return 1.0e6


def _separate_close_tips(field, grid_origin, spacing, ostia, parent_pts, parent_r):
    """Open a gap where two different branch tips pass a few millimetres apart.

    The centerline reaches both ends, but the paths meet far upstream. The
    polyball still welds the tips into one blob, and the cutter that opens
    one of them then drops the other branch. The straight line between the
    tips leaves the parent tube, which is how a real fork is told from two
    openings of the same vessel.
    """
    if (
        not ostia or len(ostia) < 2
        or parent_pts is None or parent_r is None
        or len(parent_pts) == 0
    ):
        return
    parent_pts = np.asarray(parent_pts, dtype=np.float64).reshape(-1, 3)
    parent_r = np.asarray(parent_r, dtype=np.float64).reshape(-1)
    tree = cKDTree(parent_pts)
    # Parent samples are 0.42 mm apart. Linking them at 0.85 mm rebuilds each
    # branch without welding two tips that merely pass close.
    pairs = tree.query_pairs(0.85)
    adj = [[] for _ in range(len(parent_pts))]
    if pairs:
        for a, b in pairs:
            w = float(np.linalg.norm(parent_pts[int(a)] - parent_pts[int(b)]))
            adj[int(a)].append((int(b), w))
            adj[int(b)].append((int(a), w))
    specs = []
    for origin, outward, radius in ostia:
        origin = np.asarray(origin, dtype=np.float64).reshape(3)
        outward = np.asarray(outward, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(outward))
        outward = outward / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])
        specs.append((origin, outward, max(float(radius), 0.2)))
    n_split = 0
    for i, (oi, ni, ri) in enumerate(specs):
        for oj, nj, rj in specs[i + 1:]:
            if float(ni @ nj) < 0.80:
                continue
            rel = oj - oi
            along = float(rel @ ni)
            lat = float(np.linalg.norm(rel - along * ni))
            if lat < 1.5 or lat > 7.5:
                continue
            _da, ia = tree.query(oi)
            _db, ib = tree.query(oj)
            path = _parent_path_mm(adj, int(ia), int(ib))
            chord = float(np.linalg.norm(rel))
            # One vessel: the centerline path is about as short as the chord.
            # Two tips: the path goes back to the fork, tens of millimetres.
            if path < max(15.0, 3.0 * chord):
                continue
            if _clear_tip_lens(field, grid_origin, spacing, oi, ni, ri, oj, nj, rj, lat):
                n_split += 1
    if n_split:
        print(f"  Split {n_split} pair(s) of outlet tips that passed close")


def _clear_tip_lens(field, grid_origin, spacing, oi, ni, ri, oj, nj, rj, lat):
    """Set the fused lens between two tips outside, then rewrite each stub."""
    spacing = float(spacing)
    nz, ny, nx = field.shape
    ox, oy, oz = (float(v) for v in grid_origin)
    reach = 8.0
    ends = np.vstack((oi, oj))
    lo = ends.min(axis=0) - reach
    hi = ends.max(axis=0) + reach
    i0 = max(0, int(np.floor((lo[0] - ox) / spacing)))
    i1 = min(nx, int(np.ceil((hi[0] - ox) / spacing)) + 1)
    j0 = max(0, int(np.floor((lo[1] - oy) / spacing)))
    j1 = min(ny, int(np.ceil((hi[1] - oy) / spacing)) + 1)
    k0 = max(0, int(np.floor((lo[2] - oz) / spacing)))
    k1 = min(nz, int(np.ceil((hi[2] - oz) / spacing)) + 1)
    if i1 <= i0 or j1 <= j0 or k1 <= k0:
        return False
    xs = ox + np.arange(i0, i1, dtype=np.float64) * spacing
    ys = oy + np.arange(j0, j1, dtype=np.float64) * spacing
    zs = oz + np.arange(k0, k1, dtype=np.float64) * spacing

    def _proj_rad(origin, outward):
        rel_x = xs[None, None, :] - origin[0]
        rel_y = ys[None, :, None] - origin[1]
        rel_z = zs[:, None, None] - origin[2]
        proj = rel_x * outward[0] + rel_y * outward[1] + rel_z * outward[2]
        rad2 = rel_x * rel_x + rel_y * rel_y + rel_z * rel_z - proj * proj
        return proj, np.sqrt(np.maximum(rad2, 0.0))

    proji, radi = _proj_rad(oi, ni)
    projj, radj = _proj_rad(oj, nj)
    stub_i = max(1.6, 2.2 * ri)
    stub_j = max(1.6, 2.2 * rj)
    slab = field[k0:k1, j0:j1, i0:i1]
    # The overlap can sit a few millimetres upstream of either plane, where
    # the two tubes touch before they become separate ends. Voxels closer to
    # one axis than the other stay with that tip.
    lens = (
        (radi > 0.55 * ri)
        & (radj > 0.55 * rj)
        & (radi + radj < lat + 2.0)
        & (proji > -8.0)
        & (projj > -8.0)
        & (proji < stub_i + 2.0)
        & (projj < stub_j + 2.0)
        & (slab < 0.0)
    )
    n_lens = int(np.count_nonzero(lens))
    if n_lens < 8:
        return False
    slab[lens] = np.float32(0.35)

    def _rewrite(origin, outward, radius, stub_len, own_rad, other_rad):
        proj, _rad = _proj_rad(origin, outward)
        inward = max(1.4, 2.2 * radius)
        solid = (
            (own_rad <= radius)
            & (own_rad <= other_rad)
            & (proj >= -inward)
            & (proj <= stub_len)
        )
        slab[solid] = np.minimum(slab[solid], np.float32(-0.5))

    _rewrite(oi, ni, ri, stub_i, radi, radj)
    _rewrite(oj, nj, rj, stub_j, radj, radi)
    return True


def manifold_from_spheres(
    parent_pts,
    parent_r,
    sphere_centers,
    sphere_radii,
    opening_pts,
    opening_r,
    reference_bounds,
    grid_spacing=DEFAULT_GRID_SPACING,
    max_grid_size=DEFAULT_MAX_GRID_SIZE,
    ostia=None,
):
    """One watertight surface of the parent tube union the aneurysm spheres."""
    pts = np.asarray(parent_pts, dtype=np.float64).reshape(-1, 3)
    radii = np.asarray(parent_r, dtype=np.float64).reshape(-1)
    extra_c = []
    extra_r = []
    if sphere_centers is not None and len(sphere_centers):
        extra_c.append(np.asarray(sphere_centers, dtype=np.float64).reshape(-1, 3))
        extra_r.append(np.asarray(sphere_radii, dtype=np.float64).reshape(-1))
    if opening_pts is not None and len(opening_pts):
        extra_c.append(np.asarray(opening_pts, dtype=np.float64).reshape(-1, 3))
        extra_r.append(np.asarray(opening_r, dtype=np.float64).reshape(-1))
    if extra_c:
        pts = np.vstack((pts, np.vstack(extra_c)))
        radii = np.concatenate((radii, np.concatenate(extra_r)))
    radii = np.maximum(radii, 0.2)
    max_r = float(radii.max()) if radii.size else 1.0
    pad = 2.0 * max_r + 1.5
    b = reference_bounds
    model_bounds = [
        b[0] - pad, b[1] + pad,
        b[2] - pad, b[3] + pad,
        b[4] - pad, b[5] + pad,
    ]
    extents = [
        model_bounds[1] - model_bounds[0],
        model_bounds[3] - model_bounds[2],
        model_bounds[5] - model_bounds[4],
    ]
    spacing = float(grid_spacing)
    dims = [max(24, int(np.ceil(e / spacing)) + 1) for e in extents]
    hard = int(max_grid_size)
    if max(dims) > hard:
        spacing = max(extents) / float(hard - 1)
        dims = [max(24, int(np.round(e / spacing)) + 1) for e in extents]
        dims = [min(d, hard) for d in dims]
    image = stamp_polyball_image(pts, radii, model_bounds, dims, spacing)
    scalars = image.GetPointData().GetScalars()
    field = vtk_to_numpy(scalars).reshape(dims[2], dims[1], dims[0])
    grid_origin = (model_bounds[0], model_bounds[2], model_bounds[4])
    uncapped = np.array(field, copy=True)
    _apply_ostium_caps(field, grid_origin, spacing, ostia, parent_pts, parent_r)
    scalars.Modified()
    image.Modified()
    use_caps = True
    dropped = 0
    total = 1
    if ostia:
        raw = _isosurface(image, 0.0)
        groups = _component_point_sets(raw)
        total = sum(len(group) for group in groups)
        dropped = total - (len(groups[0]) if groups else 0)
        # A small island is an outlet the later contour can neck back on.
        # A large one is a branch the cap sliced off, and that union is discarded.
        if dropped > 0.08 * max(total, 1):
            print(
                f"  Ostium caps split off {dropped} pts; keeping the uncut union"
            )
            field[...] = uncapped
            scalars.Modified()
            image.Modified()
            use_caps = False
            dropped = 0
    surface = None
    if use_caps and ostia and dropped > 0:
        origins = np.asarray(
            [np.asarray(item[0], dtype=np.float64).reshape(3) for item in ostia],
            dtype=np.float64,
        )
        trial, stranded = _contour_keeping_outlets(
            field, image, 0.0, origins, grid_origin, spacing
        )
        if not stranded:
            surface = trial
    if surface is None:
        surface = _march(image)
    n_before = int(surface.GetNumberOfPoints())
    genus = surface_genus(surface)
    if genus > 0.4:
        scalars = image.GetPointData().GetScalars()
        field = vtk_to_numpy(scalars).reshape(dims[2], dims[1], dims[0])
        origin = (model_bounds[0], model_bounds[2], model_bounds[4])
        seed = _parent_seed_zyx(parent_pts, parent_r, origin, spacing, field)
        backup = np.array(field, copy=True)
        n_cut, n_filled, kept = _untangle_from_parent(field, spacing, seed)
        if not kept:
            print(
                f"  Untangle refused: it would cut {n_cut} voxels and drop the vessel "
                f"(genus was {genus:.1f})"
            )
        else:
            scalars.Modified()
            image.Modified()
            if use_caps:
                _apply_ostium_caps(field, origin, spacing, ostia, parent_pts, parent_r)
                scalars.Modified()
                image.Modified()
            if use_caps and ostia:
                origins = np.asarray(
                    [np.asarray(item[0], dtype=np.float64).reshape(3) for item in ostia],
                    dtype=np.float64,
                )
                trial, _stranded = _contour_keeping_outlets(
                    field, image, 0.0, origins, origin, spacing
                )
            else:
                trial = _march(image)
            if trial.GetNumberOfPoints() < 0.8 * max(n_before, 1):
                field[...] = backup
                scalars.Modified()
                image.Modified()
                print(
                    f"  Untangle shrank the surface "
                    f"({n_before} -> {trial.GetNumberOfPoints()} pts); keeping the fused one"
                )
            else:
                print(
                    f"  Untangled fused spheres: cut {n_cut}, filled {n_filled} "
                    f"(genus was {genus:.1f})"
                )
                surface = trial
    if surface_genus(surface) > 0.4:
        genus_now = surface_genus(surface)
        # A kiss that is only an isosurface touch is not a voxel tunnel, so the
        # growth cuts nothing and the handle stays. Contouring slightly inside
        # opens that contact. Keep it only when the vessel does not shrink and
        # no outlet was left on a piece that got thrown away.
        scalars = image.GetPointData().GetScalars()
        field = vtk_to_numpy(scalars).reshape(dims[2], dims[1], dims[0])
        backup = np.array(field, copy=True)
        grid_origin = (model_bounds[0], model_bounds[2], model_bounds[4])
        ostium_origins = None if not ostia else np.asarray(
            [np.asarray(item[0], dtype=np.float64).reshape(3) for item in ostia],
            dtype=np.float64,
        )
        for level in (-0.04, -0.12):
            if use_caps:
                _apply_ostium_caps(field, grid_origin, spacing, ostia, parent_pts, parent_r)
                scalars.Modified()
                image.Modified()
            trial, stranded = _contour_keeping_outlets(
                field, image, level, ostium_origins, grid_origin, spacing
            )
            genus_trial = surface_genus(trial)
            if (
                not stranded
                and genus_trial + 0.4 < genus_now
                and trial.GetNumberOfPoints() >= 0.85 * max(surface.GetNumberOfPoints(), 1)
            ):
                print(
                    f"  Inside contour at {level:.2f} dropped genus "
                    f"{genus_now:.1f} -> {genus_trial:.1f}"
                )
                surface = trial
                break
            field[...] = backup
            scalars.Modified()
            image.Modified()
    surface, n_nm = repair_nonmanifold_triangles(surface)
    if n_nm:
        raise TemplateQualityError(f"sphere union has {n_nm} non-manifold edges after marching cubes")
    if surface.GetNumberOfPoints() < 50:
        raise TemplateQualityError("marching cubes produced a degenerate surface")
    return surface, spacing


def _march(image, level=0.0):
    flying = vtk.vtkFlyingEdges3D()
    flying.SetInputData(image)
    flying.SetValue(0, float(level))
    flying.Update()
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(flying.GetOutput())
    tri.Update()
    return keep_largest_region(to_vtk_poly(tri.GetOutput()))


def _ostia_without_an_opening(openings, ostia):
    """Ostia that no finished opening lands on, as (index, radius, distance)."""
    if not ostia:
        return []
    centers = [np.asarray(op["center"], dtype=np.float64) for op in openings]
    used = set()
    missing = []
    for i, ostium in enumerate(ostia):
        if "origin" in ostium:
            origin = np.asarray(ostium["origin"], dtype=np.float64).reshape(3)
        else:
            origin = np.asarray(ostium["barycenter"], dtype=np.float64).reshape(3)
        radius = float(ostium["radius"])
        best_k = None
        best_d = 1.0e9
        for k, center in enumerate(centers):
            if k in used:
                continue
            dist = float(np.linalg.norm(center - origin))
            if dist < best_d:
                best_d = dist
                best_k = k
        if best_k is None or best_d > max(3.0 * radius, 1.5):
            missing.append((i, radius, best_d))
        else:
            used.add(best_k)
    return missing


def _openings_without_an_ostium(openings, ostia):
    """Radii of openings that no ostium claims, including a waived pinhole."""
    if not openings:
        return []
    claimed = set()
    centers = [np.asarray(op["center"], dtype=np.float64) for op in openings]
    for ostium in ostia:
        if "origin" in ostium:
            origin = np.asarray(ostium["origin"], dtype=np.float64).reshape(3)
        else:
            origin = np.asarray(ostium["barycenter"], dtype=np.float64).reshape(3)
        radius = float(ostium["radius"])
        best_k = None
        best_d = 1.0e9
        for k, center in enumerate(centers):
            if k in claimed:
                continue
            dist = float(np.linalg.norm(center - origin))
            if dist < best_d:
                best_d = dist
                best_k = k
        if best_k is not None and best_d <= max(3.0 * radius, 1.5):
            claimed.add(best_k)
    return [float(openings[k]["radius"]) for k in range(len(openings)) if k not in claimed]


def _stubs_from_frames(frames):
    """Balls just inside and just outside each ostium, along the frame normal.

    The outward run is what the pipe cutter trims. The inward run is what
    joins that cutter to a parent tube that stops short of the rim: an
    outward-only stub floats past the opening and is thrown away with the
    smaller component.
    """
    extra_pts = []
    extra_r = []
    for frame in frames:
        origin = np.asarray(frame["origin"], dtype=np.float64).reshape(3)
        outward = np.asarray(frame["normal"], dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(outward))
        outward = outward / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])
        radius = max(float(frame["radius"]), 1e-3)
        spacing = max(OPENING_EXTENSION_SPACING_FACTOR * radius, 0.1)
        out_len = OPENING_EXTENSION_LENGTH_FACTOR * radius
        in_len = max(2.5 * radius, 1.4)
        n_out = max(int(np.ceil(out_len / spacing)), 2)
        n_in = max(int(np.ceil(in_len / spacing)), 2)
        for i in range(1, n_out + 1):
            extra_pts.append(origin + i * spacing * outward)
            extra_r.append(radius)
        for i in range(1, n_in + 1):
            extra_pts.append(origin - i * spacing * outward)
            extra_r.append(radius)
    if not extra_pts:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    return np.asarray(extra_pts, dtype=np.float64), np.asarray(extra_r, dtype=np.float64)


@contextmanager
def _keep_ostium_holes(profiles):
    """Don't let the pinhole pass close a hole that sits on a known ostium.

    The shared test requires the hole to be at least half the frame radius.
    On a 0.22 mm voxel tube a 0.3 mm ostium is often cut smaller than that
    and then filled, which is the failure the variable remesh avoids by
    building the tube on a finer grid. The frames are the authority here.
    """
    import vessel_pipeline as vp

    orig = vp._is_wall_pinhole

    def _wrapped(loop, min_radius, profiles=None):
        radius, n_points, bary = loop
        if profiles and int(n_points) >= 4:
            point = np.asarray(bary, dtype=np.float64).reshape(3)
            for profile in profiles:
                r_p = max(float(profile["radius"]), 0.08)
                center = np.asarray(profile["barycenter"], dtype=np.float64).reshape(3)
                if float(np.linalg.norm(point - center)) > max(2.0 * r_p, 1.0):
                    continue
                if float(radius) >= 0.22 * r_p:
                    return False
        return orig(loop, min_radius, profiles)

    vp._is_wall_pinhole = _wrapped
    try:
        yield
    finally:
        vp._is_wall_pinhole = orig


def _thin_manifold(surface, min_points):
    """Decimate, and drop the extra sheet on a non-manifold edge instead of
    falling back to the dense mesh.

    A handful of non-manifold edges used to reject the whole decimate. The
    cutter and the remesh then ran on 40-70k points, which is most of the
    time and where a small ostium gets lost in the noise.
    """
    poly = to_vtk_poly(surface)
    n = int(poly.GetNumberOfPoints())
    floor = int(min_points)
    if n <= floor + 2000:
        return poly
    reduction = 1.0 - float(floor) / float(n)
    out = decimate_dense_mc(poly, target_reduction=reduction, min_points=floor)
    out, n_nm = repair_nonmanifold_triangles(out)
    if n_nm:
        out, _dropped = force_manifold_triangles(out)
        out, n_nm = repair_nonmanifold_triangles(out)
    if n_nm or int(out.GetNumberOfPoints()) >= n - 100:
        print("  Decimate left non-manifold edges; keeping the denser surface")
        return poly
    print(f"  Decimate {n} -> {out.GetNumberOfPoints()} pts")
    return out


def _prepare_inputs(v_file, centerline_path):
    mesh = read_polydata(v_file)
    if not os.path.isfile(centerline_path):
        raise TemplateQualityError(f"original centerline not found: {centerline_path}")
    cl = read_polydata(centerline_path)
    if "MaximumInscribedSphereRadius" not in cl.point_data:
        raise TemplateQualityError("centerline has no MaximumInscribedSphereRadius")
    pts = np.asarray(mesh.points, dtype=np.float64)
    faces = _faces_of(mesh)
    cpts = np.asarray(cl.points, dtype=np.float64)
    rad = np.asarray(cl.point_data["MaximumInscribedSphereRadius"], dtype=np.float64)
    lines = np.asarray(cl.lines)
    return mesh, cl, pts, faces, cpts, rad, lines


@with_dataset_id
def process_vessel_aneurysm_dataset(
    dataset_id,
    v_file,
    output_dir,
    centerline_path,
    target_edge_length=DEFAULT_TARGET_EDGE_LENGTH,
    extension_length=DEFAULT_EXTENSION_LENGTH,
    grid_spacing=DEFAULT_GRID_SPACING,
    max_grid_size=DEFAULT_MAX_GRID_SIZE,
):
    """Detect the sac, fit four spheres, and write one uniform manifold."""
    t_all = time.perf_counter()
    print(f"\n=========================================\nVessel+aneurysm remesh: {dataset_id}")
    mesh, centerline, pts, faces, cpts, rad, lines = _prepare_inputs(v_file, centerline_path)
    mask, _excess, up, r_parent, cl_edges, det = detect_aneurysm(pts, faces, cpts, rad, lines)
    print(
        f"  Sac: {100 * det['frac']:.1f}% of vertices, diameter {det['diameter_mm']:.1f} mm, "
        f"peak {det['peak_mm']:.2f} mm, median excess {det['median_excess_mm']:.2f} mm, "
        f"neck {100 * det['neck_frac']:.0f}%, low-excess {100 * det['low_excess_frac']:.0f}%, "
        f"recall {det['local_recall']:.2f} ({det['seconds']:.2f}s)"
    )
    if det["frac"] < 0.004 or det["local_recall"] < 0.5:
        print("  WARNING: sac mask is small or missed part of the bulge; spheres follow it anyway")
    centers, radii, fit = fit_aneurysm_spheres(pts, faces, mask, _excess, up, r_parent)
    print(
        f"  Spheres: n={fit['n_spheres']} R={np.round(radii, 2).tolist()} mm, "
        f"sac RMSE {fit['parent_rmse_mm']:.2f} -> {fit['union_rmse_mm']:.2f} mm, "
        f"cover {fit['cover_frac']:.2f}, bulge {fit['bulge_frac']:.3f} ({fit['seconds']:.2f}s)"
    )
    if fit["n_spheres"] != N_SPHERES:
        raise TemplateQualityError(f"expected {N_SPHERES} aneurysm spheres, got {fit['n_spheres']}")
    if fit["bulge_frac"] > 0.02:
        raise TemplateQualityError("aneurysm spheres cross the wall")

    parent_pts, parent_r = _resample_parent(up, r_parent, cl_edges)
    frames, frames_src = resolve_cut_frames(dataset_id, v_file)
    if frames:
        frames, n_squared = square_frames_to_rims(frames, mesh)
        profiles = cut_frames_to_profiles(frames)
        open_pts, open_r = _stubs_from_frames(frames)
        print(
            f"  Ostia: {len(frames)} ground-truth frames"
            + (f" ({n_squared} squared onto the rim)" if n_squared else "")
            + f" from {os.path.basename(str(frames_src))}"
        )
    else:
        print("  No ostium frames on disk; measuring openings on the vessel")
        profiles = measure_open_profiles(mesh)
        open_pts, open_r = extra_opening_spheres(centerline, profiles)
    if len(profiles) < 2:
        raise TemplateQualityError(f"need at least two openings, found {len(profiles)}")
    fit_centers = np.asarray(centers, dtype=np.float64)
    fit_radii = np.asarray(radii, dtype=np.float64)
    centers, radii = _bridge_to_parent(
        fit_centers,
        fit_radii,
        parent_pts,
        parent_r,
        fit["neck_xyz"],
        fit["neck_radius_mm"],
    )
    print(
        f"  Union: {len(parent_pts)} parent spheres + {len(radii)} aneurysm "
        f"+ {len(open_r)} opening stubs"
    )
    t_mc = time.perf_counter()
    if frames:
        ostia = [
            (
                np.asarray(frame["origin"], dtype=np.float64).reshape(3),
                np.asarray(frame["normal"], dtype=np.float64).reshape(3),
                float(frame["radius"]),
            )
            for frame in frames
        ]
    else:
        ostia = [
            (
                np.asarray(profile["barycenter"], dtype=np.float64).reshape(3),
                np.asarray(profile["normal"], dtype=np.float64).reshape(3),
                float(profile["radius"]),
            )
            for profile in profiles
        ]
    surface, spacing = manifold_from_spheres(
        parent_pts,
        parent_r,
        centers,
        radii,
        open_pts,
        open_r,
        mesh.bounds,
        grid_spacing=grid_spacing,
        max_grid_size=max_grid_size,
        ostia=ostia,
    )
    print(
        f"  Marching cubes {surface.GetNumberOfPoints()} pts at {spacing:.3f} mm "
        f"({time.perf_counter() - t_mc:.2f}s)"
    )
    # Cut before the strong smooth. Thirty-six Taubin passes pull a short outlet
    # back through its own ostium plane, and the cutter then finds no wall.
    if surface.GetNumberOfPoints() > POST_UNCAP_POINTS + 2000:
        surface = _thin_manifold(surface, POST_UNCAP_POINTS)
    t_un = time.perf_counter()
    with _keep_ostium_holes(profiles):
        surface, n_clipped = clip_flow_extensions_and_uncap(
            surface,
            profiles,
            extension_length=extension_length,
            centerline=None if frames else centerline,
            fast_uncap=True,
            cut_frames=frames,
            reference_surface=mesh if frames else None,
            collar_only=True,
        )
    print(f"  Uncap clipped {int(n_clipped)}/{len(profiles)} ({time.perf_counter() - t_un:.2f}s)")
    if surface.GetNumberOfPoints() > POST_UNCAP_POINTS + 2000:
        surface = _thin_manifold(surface, POST_UNCAP_POINTS)
    surface = apply_taubin_smoothing(
        surface, pass_band=SMOOTH_PASSBAND, n_iter=SMOOTH_ITERS, boundary_smoothing=False
    )
    # Tiny outlets would otherwise force a 0.15 mm edge and a minute of remeshing.
    # 0.34 mm still leaves several triangles around a small ostium and keeps the
    # wall uniform. This template is allowed to trade that last bit of outlet
    # roundness for speed.
    edge = min(
        float(target_edge_length),
        max(0.34, uniform_edge_length_for_profiles(profiles, target_edge_length)),
    )
    print(f"  Uniform remesh target edge {edge:.3f} mm")
    t_rm = time.perf_counter()
    surface = remesh_surface_verified(surface, edge, n_iter=4, connectivity_iter=8)
    print(f"  Uniform remesh -> {surface.GetNumberOfPoints()} pts ({time.perf_counter() - t_rm:.2f}s)")
    surface, _n_regions = finalize_surface(surface, profiles=profiles)
    folded = enforce_min_edge(surface, label="template")
    if folded is not surface:
        surface = recompute_point_normals(folded, auto_orient=False)
    surface, _polish = polish_surface(surface, label="template")
    surface, _flipped = orient_outward(surface, label="template")
    surface = recompute_point_normals(surface, auto_orient=False)
    # A pinched edge under 0.0001 mm is not an outlet. Fold it, including the
    # cut that a plain collapse refuses, and keep the result when the openings
    # stay put.
    if inspect_surface_topology(surface)["min_edge"] < MIN_EDGE_LENGTH_MM:
        n_open = len(inspect_openings(surface))
        cleaned = collapse_tiny_edges(
            surface, floor=MIN_EDGE_LENGTH_MM, allow_cuts=True
        )
        cleaned, _n_nm = repair_nonmanifold_triangles(cleaned)
        if (
            len(inspect_openings(cleaned)) == n_open
            and inspect_surface_topology(cleaned)["min_edge"] >= MIN_EDGE_LENGTH_MM
        ):
            surface = recompute_point_normals(cleaned, auto_orient=False)
    tpl_area, ref_area = assert_template_scale(surface, mesh, context=dataset_id)
    if ref_area > 1e-6 and tpl_area / ref_area < MIN_TEMPLATE_AREA_RATIO:
        raise TemplateQualityError(
            f"{dataset_id} template area is {tpl_area / ref_area:.2f}x the vessel "
            f"({tpl_area:.1f} vs {ref_area:.1f} mm^2); the surface collapsed"
        )
    openings = assert_template_quality(surface, context=dataset_id)
    counted = frames if frames else profiles
    # A hole under the pinhole radius cannot survive a 0.34 mm edge. It may
    # still be open when the cutter happened to keep it. Either result is
    # acceptable; a real ostium that is missing is not.
    required = [
        item for item in counted
        if float(item["radius"]) >= WALL_PINHOLE_RADIUS_MM
    ]
    skipped = [
        float(item["radius"])
        for item in counted
        if float(item["radius"]) < WALL_PINHOLE_RADIUS_MM
    ]
    if skipped:
        print(
            "  Not requiring "
            + ", ".join(f"r={r:.3f} mm" for r in skipped)
            + f" (under the {WALL_PINHOLE_RADIUS_MM:.2f} mm pinhole fill)"
        )
    missing = _ostia_without_an_opening(openings, required)
    extra = _openings_without_an_ostium(openings, counted)
    # When every ostium is required and the counts match, a frame that sits a
    # little off its rim is a warning. The greedy test used to fail cases whose
    # openings were all present. A waived pinhole changes the count, so the
    # match has to be checked or a real ostium can hide behind that hole.
    count_matches = len(openings) == len(required) and not skipped
    if missing and not count_matches:
        detail = ", ".join(f"r={r:.3f} mm" for _i, r, _d in missing)
        raise TemplateQualityError(
            f"{dataset_id}: template finished with {len(openings)} openings against "
            f"{len(required)} ostia; missing {detail}"
        )
    if extra and not count_matches:
        detail = ", ".join(f"r={r:.3f} mm" for r in extra)
        raise TemplateQualityError(
            f"{dataset_id}: {len(extra)} opening(s) are not an ostium ({detail})"
        )
    if missing:
        print(
            "  WARNING: "
            + ", ".join(f"ostium r={r:.3f} mm is {d:.2f} mm from its opening" for _i, r, d in missing)
        )
    topo = inspect_surface_topology(surface)
    if topo["n_nonmanifold"] or topo["n_bowtie"]:
        raise TemplateQualityError(
            f"template is not manifold (nonmanifold={topo['n_nonmanifold']}, bowtie={topo['n_bowtie']})"
        )
    genus = surface_genus(surface)
    if genus > 0.4:
        raise TemplateQualityError(f"template genus is {genus:.1f}")

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{dataset_id}.vtp")
    save_polydata(surface, out_file)
    np.savez(
        os.path.join(output_dir, f"{dataset_id}.spheres.npz"),
        centers=fit_centers,
        radii=fit_radii,
    )
    print(
        f"  Saved {out_file} ({surface.GetNumberOfPoints()} pts, "
        f"{len(openings)} openings, genus {genus:.1f}, "
        f"nonmanifold {topo['n_nonmanifold']}) in {time.perf_counter() - t_all:.2f}s"
    )
    return out_file


def _process_one(dataset_id, v_file, args):
    def work():
        centerline_path = os.path.join(args.centerline_dir, f"{dataset_id}.vtp")
        return process_vessel_aneurysm_dataset(
            dataset_id=dataset_id,
            v_file=v_file,
            output_dir=args.output_dir,
            centerline_path=centerline_path,
            target_edge_length=args.target_edge_length,
            extension_length=args.extension_length,
            grid_spacing=args.grid_spacing,
            max_grid_size=args.max_grid_size,
        )

    return run_logged_case(
        dataset_id,
        v_file,
        args,
        work,
        log_folder_name=LOG_FOLDER,
        default_output_dir=DEFAULT_OUTPUT_DIR,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Uniform remesh of the parent vessel with the aneurysm as four spheres"
    )
    add_shared_cli_args(parser, DEFAULT_OUTPUT_DIR, default_workers=4, include_remesh_grid=True)
    parser.add_argument(
        "--centerline-dir",
        type=str,
        default=CLEANDATA_ORIGINAL_CENTERLINE,
        help="Directory of original_centerline {id}.vtp files",
    )
    add_run_log_args(parser, LOG_FOLDER)
    parser.set_defaults(
        from_folder=True,
        vessel_dir=CLEANDATA_UNIFORM,
        grid_spacing=DEFAULT_GRID_SPACING,
        max_grid_size=DEFAULT_MAX_GRID_SIZE,
    )
    args = parser.parse_args()
    extra_log, on_worker_result = configure_batch_logging(
        args, LOG_FOLDER, DEFAULT_OUTPUT_DIR, keep_all_transcripts=False
    )
    extra = [
        "--target-edge-length", str(args.target_edge_length),
        "--extension-length", str(args.extension_length),
        "--sample-spacing", str(args.sample_spacing),
        "--grid-spacing", str(args.grid_spacing),
        "--max-grid-size", str(args.max_grid_size),
        "--centerline-dir", args.centerline_dir,
        "--from-folder",
    ] + extra_log
    try:
        run_batch(
            os.path.abspath(__file__),
            _process_one,
            args,
            extra,
            on_worker_result=on_worker_result,
        )
    finally:
        finalize_run_logs(args)


if __name__ == "__main__":
    main()
