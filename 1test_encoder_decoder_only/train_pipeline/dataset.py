import gc
import os
from collections import defaultdict, deque

import numpy as np
import pyvista as pv
import torch
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
from torch_geometric.data import Data
from torch.utils.data import Dataset

import config as _config
from murray import attach_murray_fields
from config import (
    CACHE_VERSION,
    DENSE_CL_SPACING_MM,
    FAR_CL_MARGIN_MM,
    HIERARCHY_LEVELS,
    JUNCTION_COUPLE_K,
    JUNCTION_COUPLE_RADIUS_MM,
    LATENT_LEN,
    MAX_TRACTS,
    MIN_RINGS_PER_BRANCH,
    N_TRUE,
    N_TRUE_FAR_FRAC,
    TEMPLATE_COARSE_K,
    TEMPLATE_COARSE_N_MIN,
    TEMPLATE_MID_K,
    TEMPLATE_MID_N_MIN,
    TUBE_RADIUS_MM,
)
from cleaned_io import (
    assert_not_rawdata_dir,
    ensure_sample_derived,
    list_cleandata_samples,
    list_vtp_ids,
    reject_rawdata_paths,
    sample_has_sources,
    sample_is_complete,
    summarize_cleandata,
    vtp_path,
)
from geometry import fps_metric, point_to_polyline_dist
from coarsen import build_core as build_coarsen_core
from coarsen import build_levels as build_template_levels
from raycast import (
    closest_cell_normals,
    compute_level_r_star,
    empty_r_star,
    mesh_r_star_edge_stats,
    signed_distance_to_oriented_surface,
    stretch_distance_r_star,
    template_ray_r_star,
    transform_vessel_mesh,
)

# Config agent lands LATENT_DIM=16 / LATENT_LEN=128 / TOKEN_SPACING_MM=2.0.
# getattr keeps this file working if an older config is still imported.
TOKEN_SPACING_MM = float(getattr(_config, "TOKEN_SPACING_MM", 2.0))
LATENT_DIM = int(getattr(_config, "LATENT_DIM", 16))  # dataset does not allocate codes
GROUPID_ENDPOINT_SNAP_MM = float(getattr(_config, "GROUPID_ENDPOINT_SNAP_MM", 1.0))
_LATENT_PAD_DEFAULT = int(getattr(_config, "LATENT_LEN", 128))


def _dedup_polyline(pts):
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return pts
    diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    mask = np.concatenate(([True], diffs > 1e-6))
    return pts[mask]


def _as_f64(x):
    return np.asarray(x, dtype=np.float64)


def _torch_f32(x):
    return torch.tensor(np.ascontiguousarray(x, dtype=np.float32), dtype=torch.float32)


def _torch_long(x):
    return torch.tensor(np.ascontiguousarray(x, dtype=np.int64), dtype=torch.long)


def _ensure_fp32_data(data):
    """Cast cached floating tensors to float32; leave index tensors as long."""
    keys_attr = getattr(data, "keys", None)
    keys = keys_attr() if callable(keys_attr) else keys_attr
    for key in list(keys):
        val = data[key]
        if torch.is_tensor(val) and val.is_floating_point() and val.dtype != torch.float32:
            data[key] = val.to(dtype=torch.float32)
    return data


def _as_aneurysm_data(data):
    if isinstance(data, AneurysmData):
        return data
    out = AneurysmData()
    keys_attr = getattr(data, "keys", None)
    keys = keys_attr() if callable(keys_attr) else keys_attr
    for key in list(keys):
        out[key] = data[key]
    flag = getattr(data, "has_true_normal", None)
    if flag is not None:
        out.has_true_normal = bool(flag)
    return out


def _load_cached_graph(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _infer_has_true_normal(data):
    flag = getattr(data, "has_true_normal", None)
    if flag is not None:
        if torch.is_tensor(flag):
            return bool((flag != 0).reshape(-1).any().item())
        return bool(flag)
    nrm = getattr(data, "x_true_normal", None)
    x_true = getattr(data, "x_true", None)
    if nrm is None or x_true is None or nrm.numel() == 0:
        return False
    if nrm.size(0) != x_true.size(0):
        return False
    nrm = nrm.float()
    return bool(torch.isfinite(nrm).all() and nrm.norm(dim=-1).mean() > 0.5)


def _refresh_r_star_smoothness(data):
    """Rewrite cached dθ/du from r* in millimetres.

    Older caches stored |Δr*| / edge_length. On the dense sac that slope is
    10–20, which zeroed the Dirichlet and Laplacian weights exactly where
    neighboring displacements have to stay coupled.
    """
    r_star = getattr(data, "r_star", None)
    valid = getattr(data, "r_star_valid", None)
    edge_index = getattr(data, "edge_index", None)
    pos = getattr(data, "x", None)
    if not all(torch.is_tensor(t) for t in (r_star, valid, edge_index, pos)):
        return data
    if pos.size(0) == 0 or edge_index.numel() == 0 or r_star.reshape(-1).numel() != pos.size(0):
        return data
    edges = edge_index.detach().cpu().numpy().T
    dth, du, med = mesh_r_star_edge_stats(
        pos.detach().cpu().numpy(),
        r_star.detach().cpu().numpy(),
        valid.detach().cpu().numpy().astype(bool),
        edges,
    )
    data.r_dth = torch.from_numpy(np.ascontiguousarray(dth)).float()
    data.r_du = torch.from_numpy(np.ascontiguousarray(du)).float()
    data.r_ring_med = torch.from_numpy(np.ascontiguousarray(med)).float()
    return data


def _finalize_item(data):
    if getattr(data, "face", None) is None and getattr(data, "faces", None) is not None:
        faces = data.faces
        data.face = faces.t().contiguous() if faces.size(-1) == 3 else faces
    if not isinstance(data, AneurysmData):
        data = _as_aneurysm_data(data)
    data = _ensure_fp32_data(data)
    data = _refresh_r_star_smoothness(data)
    # junction windows for the Murray term: derived from cached fields, so no
    # cache bump; fixed per case (pose jitter and mirroring leave radii alone)
    data = attach_murray_fields(data)
    data.has_true_normal = torch.tensor(
        1 if _infer_has_true_normal(data) else 0, dtype=torch.uint8
    )
    return data


_CPU_TENSOR_KEYS = frozenset(
    {
        "n_radial_fine",
        "n_radial_mid",
        "n_radial_coarse",
        "branch_nl_fine",
        "branch_nl_mid",
        "branch_nl_coarse",
        "branch_nl_fine_batch",
        "branch_nl_mid_batch",
        "branch_nl_coarse_batch",
        "cache_version",
        "n_tracts",
        "pose_R",
        "origin_shift",
        "has_true_normal",
    }
)


# Node sets not listed in config.FOLLOW_BATCH. Store a zeros `{name}_batch`
# index and increment it by 1 per graph so bs>1 slices stay correct.
_GT_FOLLOW_ATTRS = (
    "gt_points",
    "gt_normals",
    "gt_points_normal",
    "gt_cl_dist",
    "gt_template_sdf",
)
_GT_FOLLOW_BATCH_KEYS = frozenset(f"{name}_batch" for name in _GT_FOLLOW_ATTRS)


class AneurysmData(Data):
    """PyG Data with correct index offsets for the mid/coarse scaffold graphs."""

    def __inc__(self, key, value, *args, **kwargs):
        if key in ("edge_index", "face"):
            return int(self.x.size(0))
        if key in ("edge_index_mid", "face_mid"):
            return int(self.pos_mid.size(0))
        if key in ("edge_index_coarse", "face_coarse"):
            return int(self.pos_coarse.size(0))
        if key == "upsample_idx_mid":
            return int(self.pos_coarse.size(0))
        if key == "upsample_idx_fine":
            return int(self.pos_mid.size(0))
        if key == "gt_faces":
            n_gt = getattr(self, "gt_points", None)
            return int(n_gt.size(0)) if n_gt is not None else 0
        if key in _GT_FOLLOW_BATCH_KEYS:
            return 1
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in (
            "edge_index",
            "edge_index_mid",
            "edge_index_coarse",
            "face",
            "face_mid",
            "face_coarse",
            "gt_faces",
        ):
            return -1
        if key in ("pose_R", "origin_shift"):
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)

    def to(self, device=None, *args, **kwargs):
        cpu_vals = {}
        for key in _CPU_TENSOR_KEYS:
            if key in self:
                cpu_vals[key] = self[key]
                del self[key]
        try:
            out = super().to(device, *args, **kwargs)
        finally:
            for key, value in cpu_vals.items():
                self[key] = value
        for key, value in cpu_vals.items():
            if torch.is_tensor(value) and value.device.type != "cpu":
                value = value.cpu()
            out[key] = value
        flag = getattr(self, "has_true_normal", None)
        if flag is None:
            flag = getattr(out, "has_true_normal", None)
        if flag is not None:
            if torch.is_tensor(flag):
                flag = flag.detach().cpu()
                if flag.numel() <= 1:
                    out.has_true_normal = torch.tensor(
                        1 if bool((flag != 0).reshape(-1).any().item()) else 0,
                        dtype=torch.uint8,
                    )
                else:
                    out.has_true_normal = flag.to(dtype=torch.uint8).reshape(-1)
            else:
                out.has_true_normal = torch.tensor(
                    1 if bool(flag) else 0, dtype=torch.uint8
                )
        return out


def _ensure_gt_follow_indices(data):
    """Zeros `{attr}_batch` so collate + `__inc__==1` yields graph ids at bs>1."""
    for name in _GT_FOLLOW_ATTRS:
        val = getattr(data, name, None)
        if not torch.is_tensor(val) or val.numel() == 0:
            continue
        n = int(val.size(0))
        key = f"{name}_batch"
        if getattr(data, key, None) is None:
            data[key] = torch.zeros(n, dtype=torch.long)
    return data


def _arc_len(pts):
    pts = _as_f64(pts)
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def allocate_ring_counts(n_length, arc_lengths):
    """Split longitudinal rings across tracts by arc length so the counts sum to n_length."""
    n_branches = len(arc_lengths)
    if n_branches == 0:
        return []
    min_len = MIN_RINGS_PER_BRANCH
    if n_length < n_branches * min_len:
        min_len = max(1, n_length // n_branches)

    total_arc = float(sum(arc_lengths))
    if total_arc <= 1e-12:
        share = max(min_len, max(1, int(n_length) // n_branches))
        raw = [share] * n_branches
    else:
        raw = [
            max(min_len, int(round(n_length * (al / total_arc))))
            for al in arc_lengths
        ]

    raw = _rebalance_counts(raw, int(n_length), min_len)
    return raw


def _rebalance_counts(raw, target, min_len):
    raw = [int(v) for v in raw]
    if not raw:
        return raw
    target = int(target)
    min_len = max(0, int(min_len))
    total = sum(raw)
    guard = 0
    while total != target and guard < 100000:
        guard += 1
        if total > target:
            floor = min_len
            cands = [j for j, v in enumerate(raw) if v > floor]
            if not cands:
                floor = 1 if min_len > 1 else 0
                cands = [j for j, v in enumerate(raw) if v > floor]
            if not cands:
                break
            i = max(cands, key=lambda j: raw[j])
            raw[i] -= 1
            total -= 1
        else:
            i = max(range(len(raw)), key=lambda j: raw[j])
            raw[i] += 1
            total += 1
    return raw


def couple_ostium_edges(
    pos,
    tract_id,
    junction_incidents,
    junc_xyz,
    radius_mm=JUNCTION_COUPLE_RADIUS_MM,
    k=JUNCTION_COUPLE_K,
):
    """Bidirectional parent–daughter edges near each ostium (not a watertight Boolean).

    k-NN across tracts is a KD-tree query, not an (Na × Nb × 3) distance tensor.
    Short compact tracts can put 10k–20k fine-grid vertices inside `radius_mm`;
    the pairwise form was tens of GB per sample.
    """
    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
    tract_id = np.asarray(tract_id, dtype=np.int64).reshape(-1)
    if pos.shape[0] == 0 or not junction_incidents:
        return np.zeros((0, 2), dtype=np.int64)
    radius_mm = float(radius_mm)
    k = max(1, int(k))
    lim = radius_mm * 1.5
    pairs = []
    for nid, incident in junction_incidents.items():
        if nid not in junc_xyz:
            continue
        jp = np.asarray(junc_xyz[nid], dtype=np.float64).reshape(3)
        tids = sorted({int(t) for t in incident if 0 <= int(t) < MAX_TRACTS})
        near = {}
        for tid in tids:
            idx = np.where(tract_id == tid)[0]
            if idx.size == 0:
                continue
            dist = np.linalg.norm(pos[idx] - jp, axis=1)
            keep = idx[dist <= radius_mm]
            if keep.size == 0:
                n_keep = min(idx.size, max(k * 4, 4))
                keep = idx[np.argsort(dist)[:n_keep]]
            near[tid] = keep
        tids = [t for t in tids if t in near]
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                a = near[tids[i]]
                b = near[tids[j]]
                if a.size == 0 or b.size == 0:
                    continue
                _append_knn_pairs(pairs, pos, a, b, k, lim)
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(pairs, dtype=np.int64)


def _append_knn_pairs(pairs, pos, a, b, k, lim):
    """Add bidirectional edges from each point in `a` to its k nearest in `b` within `lim`."""
    kk = min(int(k), int(b.size))
    if kk < 1:
        return
    tree_b = cKDTree(pos[b])
    dists, nn = tree_b.query(pos[a], k=kk, distance_upper_bound=float(lim), workers=1)
    if kk == 1:
        dists = np.asarray(dists, dtype=np.float64).reshape(-1, 1)
        nn = np.asarray(nn, dtype=np.int64).reshape(-1, 1)
    else:
        dists = np.asarray(dists, dtype=np.float64)
        nn = np.asarray(nn, dtype=np.int64)
    n_b = int(b.size)
    for ia in range(a.size):
        for jb in range(kk):
            jloc = int(nn[ia, jb])
            dij = float(dists[ia, jb])
            if jloc < 0 or jloc >= n_b or not np.isfinite(dij) or dij > lim:
                continue
            ua, ub = int(a[ia]), int(b[jloc])
            pairs.append((ua, ub))
            pairs.append((ub, ua))


def extract_unique_tracts(centerline_mesh, snap=1e-4):
    """Unique centerline tracts from overlapping VMTK paths or already-split polylines.

    Builds a snapped undirected graph of polyline segments and splits at junctions
    (degree != 2). Duplicate parent paths collapse to a single edge set.
    """
    points = _as_f64(centerline_mesh.points)
    if points.shape[0] < 2:
        raise ValueError("Centerline has fewer than 2 points")

    keys = np.round(points / snap).astype(np.int64)
    canon = {}
    remap = np.empty(len(points), dtype=np.int64)
    for i, k in enumerate(map(tuple, keys)):
        if k not in canon:
            canon[k] = i
        remap[i] = canon[k]

    adj = defaultdict(set)
    n_cells = int(centerline_mesh.n_cells)
    if n_cells == 0:
        compact = []
        for i in range(len(points)):
            cid = int(remap[i])
            if not compact or compact[-1] != cid:
                compact.append(cid)
        for a, b in zip(compact, compact[1:]):
            adj[a].add(b)
            adj[b].add(a)
    else:
        for ci in range(n_cells):
            cell = centerline_mesh.GetCell(ci)
            n_pts = cell.GetNumberOfPoints()
            if n_pts < 2:
                continue
            ids = [int(remap[cell.GetPointId(j)]) for j in range(n_pts)]
            compact = [ids[0]]
            for vid in ids[1:]:
                if vid != compact[-1]:
                    compact.append(vid)
            for a, b in zip(compact, compact[1:]):
                adj[a].add(b)
                adj[b].add(a)

    if not adj:
        pts = _dedup_polyline(points)
        if len(pts) < 2:
            raise ValueError("Centerline has fewer than 2 unique points")
        return [pts], [(0, 1)], []

    def edge_key(a, b):
        return (a, b) if a < b else (b, a)

    special = [n for n, nbrs in adj.items() if len(nbrs) != 2]
    if not special:
        special = [min(adj)]

    used = set()
    path_ids = []
    for start in special:
        for nbr in list(adj[start]):
            ek = edge_key(start, nbr)
            if ek in used:
                continue
            path = [start, nbr]
            used.add(ek)
            prev, cur = start, nbr
            while len(adj[cur]) == 2:
                nxts = [x for x in adj[cur] if x != prev]
                if not nxts:
                    break
                nxt = nxts[0]
                ek2 = edge_key(cur, nxt)
                if ek2 in used:
                    break
                used.add(ek2)
                path.append(nxt)
                prev, cur = cur, nxt
            if len(path) >= 2:
                path_ids.append(path)

    if not path_ids:
        pts = _dedup_polyline(points)
        return [pts], [(0, 1)], []

    xyz_tracts = [_dedup_polyline(points[np.asarray(p, dtype=np.int64)]) for p in path_ids]
    keep = [(t, p) for t, p in zip(xyz_tracts, path_ids) if len(t) >= 2]
    if not keep:
        pts = _dedup_polyline(points)
        return [pts], [(0, 1)], []
    xyz_tracts, path_ids = zip(*keep)
    endpoints = [(int(p[0]), int(p[-1])) for p in path_ids]
    junctions = [n for n, nbrs in adj.items() if len(nbrs) > 2]
    return list(xyz_tracts), endpoints, junctions


def _point_data_array(mesh, name):
    pdata = getattr(mesh, "point_data", None)
    if pdata is not None and name in pdata:
        arr = np.asarray(pdata[name])
        if arr.size == 0:
            return None
        return arr.reshape(arr.shape[0], -1)[:, 0]
    getter = getattr(mesh, "GetPointData", None)
    if getter is None:
        return None
    vtk_arr = getter().GetArray(name)
    if vtk_arr is None:
        return None
    n = int(vtk_arr.GetNumberOfTuples())
    if n <= 0:
        return None
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = float(vtk_arr.GetComponent(i, 0))
    return out


def _cell_data_array(mesh, name):
    cdata = getattr(mesh, "cell_data", None)
    if cdata is not None and name in cdata:
        arr = np.asarray(cdata[name])
        if arr.size == 0:
            return None
        return arr.reshape(arr.shape[0], -1)[:, 0]
    getter = getattr(mesh, "GetCellData", None)
    if getter is None:
        return None
    vtk_arr = getter().GetArray(name)
    if vtk_arr is None:
        return None
    n = int(vtk_arr.GetNumberOfTuples())
    if n <= 0:
        return None
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = float(vtk_arr.GetComponent(i, 0))
    return out


def _iter_polyline_point_ids(mesh):
    n_cells = int(getattr(mesh, "n_cells", 0) or 0)
    if n_cells == 0 and hasattr(mesh, "GetNumberOfCells"):
        n_cells = int(mesh.GetNumberOfCells())
    for ci in range(n_cells):
        cell = mesh.GetCell(ci)
        n_pts = int(cell.GetNumberOfPoints())
        if n_pts < 2:
            continue
        yield ci, [int(cell.GetPointId(j)) for j in range(n_pts)]


def _longest_unique_polyline(pieces, snap=5e-2):
    """Collapse duplicate VMTK copies of one GroupId into a single polyline."""
    chunks = []
    line_ids = []
    offset = 0
    for piece in pieces:
        pts = _dedup_polyline(piece)
        if len(pts) < 2:
            continue
        n = len(pts)
        chunks.append(pts)
        line_ids.append(np.concatenate(([n], np.arange(offset, offset + n, dtype=np.int64))))
        offset += n
    if not chunks:
        return None
    points = np.concatenate(chunks, axis=0)
    lines = np.concatenate(line_ids)
    mesh = pv.PolyData(points, lines=lines)
    tracts, _, _ = extract_unique_tracts(mesh, snap=snap)
    if not tracts:
        return None
    return max(tracts, key=_arc_len)


def _select_group_polyline(pieces):
    """One polyline for a GroupId: longest full copy, or stitch LINE fragments."""
    pieces = [p for p in pieces if p is not None and len(p) >= 2]
    if not pieces:
        return None
    longest = max(pieces, key=_arc_len)
    n_full = sum(1 for p in pieces if len(p) >= 4)
    if n_full >= 1:
        return longest
    stitched = _longest_unique_polyline(pieces, snap=1e-4)
    if stitched is not None and _arc_len(stitched) >= _arc_len(longest) - 1e-9:
        return stitched
    return longest


def _snap_tract_endpoints(tracts, snap_mm=GROUPID_ENDPOINT_SNAP_MM):
    """Assign shared node ids to tract ends that lie within snap_mm."""
    snap_mm = float(snap_mm)
    clusters = []
    endpoints = []
    for pts in tracts:
        ids = []
        for xyz in (pts[0], pts[-1]):
            xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
            assigned = None
            for cid, cxyz in enumerate(clusters):
                if float(np.linalg.norm(xyz - cxyz)) <= snap_mm:
                    assigned = cid
                    clusters[cid] = 0.5 * (cxyz + xyz)
                    break
            if assigned is None:
                assigned = len(clusters)
                clusters.append(xyz.copy())
            ids.append(assigned)
        endpoints.append((int(ids[0]), int(ids[1])))
    deg = defaultdict(int)
    for a, b in endpoints:
        deg[a] += 1
        deg[b] += 1
    junctions = [n for n, d in deg.items() if d > 2]
    return endpoints, junctions


def _missing_groupids_error(centerline_mesh):
    n_cells = int(getattr(centerline_mesh, "n_cells", 0) or 0)
    return (
        "cleandata centerline is missing GroupIds (generator bug: "
        "clip_centerline_at_profiles dropped cell arrays). "
        f"Refusing extract_unique_tracts fallback. n_cells={n_cells}."
    )


def _polyline_runs_from_point_ids(ids, group_pt, blank_pt):
    """Split a polyline into consecutive (GroupId, Blanking) runs."""
    runs = []
    run_ids = []
    run_gid = None
    run_blank = False
    for pid in ids:
        gid = int(round(float(group_pt[pid])))
        blanked = False
        if blank_pt is not None:
            blanked = float(blank_pt[pid]) > 0.5
        if run_ids and (gid != run_gid or blanked != run_blank):
            runs.append((run_gid, run_blank, run_ids))
            run_ids = []
        run_gid = gid
        run_blank = blanked
        run_ids.append(pid)
    if run_ids:
        runs.append((run_gid, run_blank, run_ids))
    return runs


def _attach_blanked_runs(path_runs, points):
    """Prepend each blanked run to the daughter that follows it on this path."""
    pieces = []
    pending = None
    for gid, blanked, ids in path_runs:
        if len(ids) < 2:
            continue
        if blanked:
            pending = ids
            continue
        if pending is not None:
            ids = list(pending) + list(ids)
            pending = None
        pts = _dedup_polyline(points[np.asarray(ids, dtype=np.int64)])
        if len(pts) >= 2:
            pieces.append((int(gid), pts))
    return pieces


def extract_groupid_tracts(
    centerline_mesh,
    snap=1e-4,
    endpoint_snap_mm=None,
    require_groupids=False,
):
    """Validated GroupId tracts (§2.4.3 / prototype_branch_tracts.py).

    1. Group polyline runs by GroupIds; keep the longest copy per GroupId.
    2. Attach each blanked run to the daughter that follows it (same
       CenterlineId / TractId order, or along the polyline for point arrays).
    3. Snap endpoints at GROUPID_ENDPOINT_SNAP_MM (1 mm) to rebuild the tree.

    `require_groupids=True` (cleandata via `_build_data`) raises if GroupIds are
    missing. Synthetic tests pass `require_groupids=False` (the default) to keep
    the `extract_unique_tracts` fallback.
    """
    if endpoint_snap_mm is None:
        endpoint_snap_mm = GROUPID_ENDPOINT_SNAP_MM
    group_pt = _point_data_array(centerline_mesh, "GroupIds")
    group_cell = _cell_data_array(centerline_mesh, "GroupIds")
    if group_pt is None and group_cell is None:
        if require_groupids:
            raise ValueError(_missing_groupids_error(centerline_mesh))
        return extract_unique_tracts(centerline_mesh, snap=snap)

    blank_pt = _point_data_array(centerline_mesh, "Blanking")
    blank_cell = _cell_data_array(centerline_mesh, "Blanking")
    clid_cell = _cell_data_array(centerline_mesh, "CenterlineIds")
    tid_cell = _cell_data_array(centerline_mesh, "TractIds")
    points = _as_f64(centerline_mesh.points)
    by_group = defaultdict(list)

    polylines = list(_iter_polyline_point_ids(centerline_mesh))
    if not polylines:
        compact = list(range(len(points)))
        polylines = [(0, compact)] if len(compact) >= 2 else []

    use_cell_paths = (
        group_cell is not None
        and clid_cell is not None
        and len(clid_cell) >= max(len(polylines), 1)
    )
    if use_cell_paths:
        per_cl = defaultdict(list)
        for ci, ids in polylines:
            if ci >= len(group_cell):
                continue
            clid = int(round(float(clid_cell[ci])))
            tract = int(round(float(tid_cell[ci]))) if tid_cell is not None and ci < len(tid_cell) else ci
            gid = int(round(float(group_cell[ci])))
            blanked = False
            if blank_cell is not None and ci < len(blank_cell):
                blanked = float(blank_cell[ci]) > 0.5
            per_cl[clid].append((tract, gid, blanked, ids))
        for clid in sorted(per_cl):
            runs = [(g, b, ids) for _, g, b, ids in sorted(per_cl[clid], key=lambda t: t[0])]
            for gid, pts in _attach_blanked_runs(runs, points):
                by_group[gid].append(pts)
    else:
        for ci, ids in polylines:
            if group_pt is not None:
                runs = _polyline_runs_from_point_ids(ids, group_pt, blank_pt)
            elif group_cell is not None and ci < len(group_cell):
                gid = int(round(float(group_cell[ci])))
                blanked = False
                if blank_cell is not None and ci < len(blank_cell):
                    blanked = float(blank_cell[ci]) > 0.5
                runs = [(gid, blanked, ids)]
            else:
                continue
            for gid, pts in _attach_blanked_runs(runs, points):
                by_group[gid].append(pts)

    if not by_group:
        if require_groupids:
            raise ValueError(
                "GroupIds present but no non-blanked tract could be built "
                "(blanked-only centerline or empty runs)."
            )
        return extract_unique_tracts(centerline_mesh, snap=snap)

    # Longest copy per GroupId. Full VMTK polyline cells are kept as-is;
    # 2-point VTK_LINE fragments (pyvista.merge of test polylines) are
    # stitched. Do not snap-merge distinct VMTK copies of the same GroupId.
    tracts = []
    for gid in sorted(by_group):
        pts = _select_group_polyline(by_group[gid])
        if pts is not None and len(pts) >= 2:
            tracts.append(pts)
    if not tracts:
        if require_groupids:
            raise ValueError("GroupIds present but every GroupId copy was degenerate.")
        return extract_unique_tracts(centerline_mesh, snap=snap)

    endpoints, junctions = _snap_tract_endpoints(tracts, snap_mm=endpoint_snap_mm)
    return tracts, endpoints, junctions


def sample_x_true(
    gt_points,
    gt_normals,
    dense_cl,
    n_true=N_TRUE,
    n_far_frac=None,
    far_margin_mm=None,
    tube_radius=None,
):
    """Per-epoch resample of encoder/Chamfer GT from a cached full surface.

    `N_TRUE` FPS is intentionally *not* frozen in the cache (§2.2.3, §8).
    `train.py` should call this each epoch. `gt_normals` may be None.
    """
    gt_points = np.asarray(gt_points, dtype=np.float64).reshape(-1, 3)
    cl_xyz = np.asarray(dense_cl, dtype=np.float64).reshape(-1, 3)
    n_true = int(n_true)
    if n_far_frac is None:
        n_far_frac = N_TRUE_FAR_FRAC
    if far_margin_mm is None:
        far_margin_mm = FAR_CL_MARGIN_MM
    if tube_radius is None:
        tube_radius = TUBE_RADIUS_MM
    if gt_points.shape[0] == 0 or n_true < 1:
        empty = np.zeros((max(n_true, 1), 3), dtype=np.float32)
        return _torch_f32(empty[:1]), _torch_f32(np.zeros((1,), dtype=np.float32)), _torch_f32(empty[:1])

    d_all = point_to_polyline_dist(gt_points.astype(np.float32), cl_xyz).astype(np.float32)
    n_far = int(round(n_true * float(n_far_frac)))
    n_uni = max(1, n_true - n_far)
    uni = fps_metric(gt_points.astype(np.float32), n_uni)
    d_uni = point_to_polyline_dist(uni, cl_xyz).astype(np.float32)

    far_mask = d_all > (float(tube_radius) + float(far_margin_mm))
    far_pts = gt_points[far_mask]
    if far_pts.shape[0] == 0 or n_far <= 0:
        extra = fps_metric(gt_points.astype(np.float32), max(n_far, 1))[: max(n_far, 0)]
        d_extra = (
            point_to_polyline_dist(extra, cl_xyz).astype(np.float32)
            if len(extra)
            else np.zeros((0,), dtype=np.float32)
        )
    else:
        extra = fps_metric(far_pts.astype(np.float32), min(n_far, far_pts.shape[0]))
        d_extra = point_to_polyline_dist(extra, cl_xyz).astype(np.float32)
        if extra.shape[0] < n_far:
            pad = fps_metric(gt_points.astype(np.float32), n_far - extra.shape[0])
            extra = np.concatenate([extra, pad], axis=0)
            d_extra = np.concatenate(
                [d_extra, point_to_polyline_dist(pad, cl_xyz).astype(np.float32)], axis=0
            )

    x_true = np.concatenate([uni, extra], axis=0)[:n_true]
    d_true = np.concatenate([d_uni, d_extra], axis=0)[:n_true]
    if x_true.shape[0] < n_true:
        reps = int(np.ceil(n_true / max(x_true.shape[0], 1)))
        x_true = np.tile(x_true, (reps, 1))[:n_true]
        d_true = np.tile(d_true, reps)[:n_true]

    nrm_out = None
    if gt_normals is not None:
        gt_normals = np.asarray(gt_normals, dtype=np.float64).reshape(-1, 3)
        if gt_normals.shape[0] == gt_points.shape[0] and gt_normals.shape[0] > 0:
            _, idx = cKDTree(gt_points).query(np.asarray(x_true, dtype=np.float64), k=1, workers=1)
            nrm_out = gt_normals[np.asarray(idx, dtype=np.int64)]
            nrm_out = nrm_out / np.clip(np.linalg.norm(nrm_out, axis=1, keepdims=True), 1e-8, None)
    if nrm_out is None:
        nrm_out = np.zeros_like(x_true, dtype=np.float64)
        nrm_out[:, 2] = 1.0
    return _torch_f32(x_true), _torch_f32(d_true), _torch_f32(nrm_out)


def _choose_inlet(tracts, endpoints):
    """Parent tract = longest arm of the highest-degree node; inlet is that arm's leaf."""
    counts = defaultdict(int)
    for a, b in endpoints:
        counts[a] += 1
        counts[b] += 1
    if not counts:
        return 0, False
    root = max(counts, key=lambda n: (counts[n], -n))
    parent_i = 0
    parent_len = -1.0
    reverse = False
    for i, ((a, b), pts) in enumerate(zip(endpoints, tracts)):
        if counts[root] > 1 and a != root and b != root:
            continue
        L = _arc_len(pts)
        if L > parent_len:
            parent_len = L
            parent_i = i
            reverse = a == root
    if counts[root] <= 1:
        reverse = False
    return parent_i, reverse


def _orient_tracts(tracts, endpoints, inlet_i, reverse_inlet):
    tracts = [np.array(t, dtype=np.float64, copy=True) for t in tracts]
    endpoints = list(endpoints)
    if reverse_inlet:
        tracts[inlet_i] = tracts[inlet_i][::-1].copy()
        a, b = endpoints[inlet_i]
        endpoints[inlet_i] = (b, a)

    id_to_xyz = {}
    for (a, b), pts in zip(endpoints, tracts):
        id_to_xyz[a] = pts[0]
        id_to_xyz[b] = pts[-1]

    adj_t = defaultdict(list)
    for i, (a, b) in enumerate(endpoints):
        adj_t[a].append((i, b))
        adj_t[b].append((i, a))

    oriented = [False] * len(tracts)
    oriented[inlet_i] = True
    q = deque([endpoints[inlet_i][1], endpoints[inlet_i][0]])
    seen_node = set(q)
    while q:
        node = q.popleft()
        for ti, other in adj_t[node]:
            if oriented[ti]:
                continue
            a, b = endpoints[ti]
            if a != node:
                tracts[ti] = tracts[ti][::-1].copy()
                endpoints[ti] = (b, a)
            oriented[ti] = True
            if endpoints[ti][1] not in seen_node:
                seen_node.add(endpoints[ti][1])
                q.append(endpoints[ti][1])

    deg = defaultdict(int)
    for a, b in endpoints:
        deg[a] += 1
        deg[b] += 1
    junction_incidents = defaultdict(list)
    for i, (a, b) in enumerate(endpoints):
        if deg[a] >= 3:
            junction_incidents[a].append(i)
        if deg[b] >= 3:
            junction_incidents[b].append(i)

    junc_xyz = {nid: id_to_xyz[nid] for nid in junction_incidents}
    return tracts, endpoints, junction_incidents, junc_xyz


def _canonical_pose(inlet_pts):
    """Rotation sending inlet tangent to +Z and Bishop-like normal to +X."""
    pts = _as_f64(inlet_pts)
    k = min(5, len(pts) - 1)
    t0 = pts[k] - pts[0]
    t0 = t0 / (np.linalg.norm(t0) + 1e-12)
    v = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(np.dot(t0, v)) > 0.99:
        v = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    n0 = np.cross(t0, v)
    n0 = n0 / (np.linalg.norm(n0) + 1e-12)
    n0 = n0 - t0 * np.dot(n0, t0)
    n0 = n0 / (np.linalg.norm(n0) + 1e-12)
    b0 = np.cross(t0, n0)
    b0 = b0 / (np.linalg.norm(b0) + 1e-12)
    R = np.stack([n0, b0, t0], axis=1)
    return R


class AneurysmDataset(Dataset):
    def __init__(
        self,
        vtp_vessel_dir=None,
        vtp_centerline_dir=None,
        tube_radius=TUBE_RADIUS_MM,
        n_length=None,
        n_radial=None,
        extra_centerline_dir=None,
        cache_dir=None,
        n_true=N_TRUE,
        hierarchy=HIERARCHY_LEVELS,
        latent_len=LATENT_LEN,
        quiet=False,
        cleandata_root=None,
        require_templates=True,
        ensure_derived=False,
    ):
        super().__init__()
        if extra_centerline_dir:
            raise ValueError(
                "extra_centerline_dir is removed; Stage 2 reads original_centerline "
                "from cleandata/ (or generates it with centerline_creation.py)."
            )
        self._init_kwargs = dict(
            vtp_vessel_dir=vtp_vessel_dir,
            vtp_centerline_dir=vtp_centerline_dir,
            tube_radius=tube_radius,
            n_length=n_length,
            n_radial=n_radial,
            extra_centerline_dir=None,
            cache_dir=cache_dir,
            n_true=n_true,
            hierarchy=tuple(tuple(lv) for lv in hierarchy),
            latent_len=latent_len,
            cleandata_root=cleandata_root,
            require_templates=require_templates,
            ensure_derived=ensure_derived,
        )
        self.tube_radius = float(tube_radius)
        self.n_true = int(n_true)
        self.latent_len = int(latent_len) if latent_len is not None else _LATENT_PAD_DEFAULT
        self.token_spacing_mm = TOKEN_SPACING_MM
        self.hierarchy = tuple(tuple(lv) for lv in hierarchy)
        if n_length is not None or n_radial is not None:
            fine = list(self.hierarchy[-1])
            if n_length is not None:
                fine[0] = int(n_length)
            if n_radial is not None:
                fine[1] = int(n_radial)
            self.hierarchy = self.hierarchy[:-1] + (tuple(fine),)
        self.n_length = int(self.hierarchy[-1][0])
        self.n_radial = int(self.hierarchy[-1][1])
        self.cleandata_root = cleandata_root
        self.require_templates = bool(require_templates)
        self.ensure_derived = bool(ensure_derived)

        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tube_cache")
        self.cache_dir = os.path.abspath(cache_dir) if cache_dir else None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.samples = self._discover_samples(
            vtp_vessel_dir=vtp_vessel_dir,
            vtp_centerline_dir=vtp_centerline_dir,
            quiet=quiet,
        )

    def _discover_samples(self, vtp_vessel_dir, vtp_centerline_dir, quiet=False):
        if (vtp_vessel_dir is None) ^ (vtp_centerline_dir is None):
            raise ValueError("Pass both vtp_vessel_dir and vtp_centerline_dir, or neither.")
        if vtp_vessel_dir is not None:
            assert_not_rawdata_dir(vtp_vessel_dir, vtp_centerline_dir)
            return self._discover_from_dirs(
                vtp_vessel_dir, vtp_centerline_dir, quiet=quiet
            )
        complete, incomplete = list_cleandata_samples(
            root=self.cleandata_root,
            require_templates=self.require_templates,
            include_incomplete=True,
        )
        if self.ensure_derived:
            usable = []
            skipped = []
            for rec in complete + incomplete:
                if sample_has_sources(rec, require_templates=self.require_templates):
                    usable.append(rec)
                else:
                    skipped.append(rec["dataset_id"])
            samples = usable
            n_incomplete = len(incomplete)
        else:
            samples = complete
            n_incomplete = len(incomplete)
            skipped = [rec["dataset_id"] for rec in incomplete]
        if not quiet and os.environ.get("RANK", "0") in ("0", ""):
            _, counts = summarize_cleandata(self.cleandata_root)
            count_txt = ", ".join(f"{k}={v}" for k, v in counts.items())
            extra = ""
            if n_incomplete:
                extra = (
                    f", {n_incomplete} incomplete file sets"
                    + (
                        " (will run centerline_creation / uniform remesh at cache time)"
                        if self.ensure_derived
                        else " (pass ensure_derived=True under vmtk_env to fill them)"
                    )
                )
            print(
                f"AneurysmDataset: cleandata ({count_txt}); "
                f"{len(samples)} samples{extra}"
            )
            if skipped and not self.ensure_derived and n_incomplete <= 12:
                print("  incomplete ids: " + ", ".join(skipped))
        return samples

    def _discover_from_dirs(self, vtp_vessel_dir, vtp_centerline_dir, quiet=False):
        """Explicit GT+centerline folders (tests / debug). Still rejects rawdata."""
        ids = list_vtp_ids(vtp_vessel_dir) | list_vtp_ids(vtp_centerline_dir)
        samples = []
        missing = []
        for dataset_id in sorted(ids):
            rec = {
                "dataset_id": dataset_id,
                "vessel_file": vtp_path(vtp_vessel_dir, dataset_id),
                "centerline_file": vtp_path(vtp_centerline_dir, dataset_id),
                "template_mesh_file": None,
                "template_centerline_file": None,
            }
            reject_rawdata_paths(rec)
            if os.path.isfile(rec["vessel_file"]) and os.path.isfile(rec["centerline_file"]):
                samples.append(rec)
            else:
                missing.append(dataset_id)
        if not quiet:
            extra = f", {len(missing)} missing a vessel or centerline file" if missing else ""
            print(
                f"AneurysmDataset: {len(samples)} samples from explicit dirs{extra} "
                f"(vessel={vtp_vessel_dir}, centerline={vtp_centerline_dir})"
            )
        return samples

    def _cache_path(self, dataset_id):
        if not self.cache_dir:
            return None
        safe_id = str(dataset_id).replace("/", "_").replace("\\", "_")
        parts = "_".join(f"L{nl}R{nr}" for nl, nr in self.hierarchy)
        name = (
            f"{safe_id}_v{CACHE_VERSION}_{parts}_n{self.n_true}"
            f"_rad{self.tube_radius}_Z{self.latent_len}.pt"
        )
        return os.path.join(self.cache_dir, name)

    def _orthonormalize_frame(self, tangent, normal):
        t = _as_f64(tangent)
        n = _as_f64(normal)
        t = t / (np.linalg.norm(t) + 1e-12)
        n = n - t * np.dot(n, t)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-8:
            v = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(np.dot(t, v)) > 0.99:
                v = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            n = np.cross(t, v)
            n = n / (np.linalg.norm(n) + 1e-12)
        else:
            n = n / n_norm
        b = np.cross(t, n)
        b = b / (np.linalg.norm(b) + 1e-12)
        return t, n, b

    def _compute_parallel_transport_frames(self, derivatives):
        """Bishop frame along the curve via Rodrigues parallel transport."""
        derivatives = _as_f64(derivatives)
        t_norm = np.linalg.norm(derivatives, axis=1, keepdims=True)
        t_norm = np.maximum(t_norm, 1e-8)
        tangents = derivatives / t_norm
        n_pts = len(tangents)

        normals = np.zeros_like(tangents, dtype=np.float64)
        binormals = np.zeros_like(tangents, dtype=np.float64)

        t0 = tangents[0]
        v = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(t0, v)) > 0.99:
            v = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        n0 = np.cross(t0, v)
        t0, n0, b0 = self._orthonormalize_frame(t0, n0)
        tangents[0], normals[0], binormals[0] = t0, n0, b0

        eye3 = np.eye(3, dtype=np.float64)
        for i in range(1, n_pts):
            t_prev = tangents[i - 1]
            t_curr = tangents[i]
            axis = np.cross(t_prev, t_curr)
            sin_angle = np.linalg.norm(axis)
            cos_angle = float(np.clip(np.dot(t_prev, t_curr), -1.0, 1.0))

            if sin_angle > 1e-6:
                axis = axis / sin_angle
                k_mat = np.array(
                    [
                        [0.0, -axis[2], axis[1]],
                        [axis[2], 0.0, -axis[0]],
                        [-axis[1], axis[0], 0.0],
                    ],
                    dtype=np.float64,
                )
                rot = eye3 + sin_angle * k_mat + (1.0 - cos_angle) * (k_mat @ k_mat)
                n_i = rot @ normals[i - 1]
            else:
                n_i = normals[i - 1]

            t_i, n_i, b_i = self._orthonormalize_frame(t_curr, n_i)
            tangents[i], normals[i], binormals[i] = t_i, n_i, b_i

        return tangents, normals, binormals

    def _fit_centerline_spline(self, branch_points):
        branch_points = _as_f64(branch_points)
        n = len(branch_points)
        k = int(min(5, n - 1))
        if k < 1:
            raise ValueError("Need at least 2 points to fit a centerline spline")
        coords = [branch_points[:, 0], branch_points[:, 1], branch_points[:, 2]]
        try:
            tck, _ = splprep(coords, s=0, k=k)
        except Exception:
            tck, _ = splprep(coords, s=1.0, k=k)
        return tck

    def _arc_length_parameter(self, eval_points):
        eval_points = _as_f64(eval_points)
        seg = np.linalg.norm(np.diff(eval_points, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(cum[-1])
        if total <= 1e-12:
            return np.linspace(0.0, 1.0, len(eval_points), dtype=np.float64), 0.0
        return cum / total, total

    def _interp_by_u(self, u_src, values, u_q):
        u_src = np.asarray(u_src, dtype=np.float64)
        values = _as_f64(values)
        u_q = np.asarray(u_q, dtype=np.float64)
        if values.ndim == 1:
            return np.interp(u_q, u_src, values)
        return np.stack([np.interp(u_q, u_src, values[:, d]) for d in range(values.shape[1])], axis=1)

    def _fit_dense_tract(self, branch_points, spacing=DENSE_CL_SPACING_MM):
        branch_points = _dedup_polyline(branch_points)
        tck = self._fit_centerline_spline(branch_points)
        est = max(_arc_len(branch_points), spacing)
        n_dense = max(32, int(est / max(spacing, 1e-3)) + 1)
        u_sp = np.linspace(0.0, 1.0, n_dense, dtype=np.float64)
        xyz = np.vstack(splev(u_sp, tck)).T.astype(np.float64, copy=False)
        deriv = np.vstack(splev(u_sp, tck, der=1)).T.astype(np.float64, copy=False)
        dnorm = np.linalg.norm(deriv, axis=1, keepdims=True)
        if np.any(dnorm < 1e-8):
            fd = np.gradient(xyz, axis=0)
            deriv = np.where(dnorm < 1e-8, fd, deriv)
        tangents, normals, binormals = self._compute_parallel_transport_frames(deriv)
        u_arc, arc = self._arc_length_parameter(xyz)
        kappa = np.zeros(n_dense, dtype=np.float64)
        tau = np.zeros(n_dense, dtype=np.float64)
        try:
            d2 = np.vstack(splev(u_sp, tck, der=2)).T.astype(np.float64, copy=False)
            try:
                d3 = np.vstack(splev(u_sp, tck, der=3)).T.astype(np.float64, copy=False)
            except TypeError:
                d3 = np.zeros_like(d2)
            c12 = np.cross(deriv, d2)
            n12 = np.linalg.norm(c12, axis=1)
            n1 = np.linalg.norm(deriv, axis=1)
            kappa = n12 / np.clip(np.power(n1, 3), 1e-12, None)
            tau = np.einsum("ij,ij->i", c12, d3) / np.clip(n12 * n12, 1e-12, None)
            kappa = np.nan_to_num(kappa, nan=0.0, posinf=0.0, neginf=0.0)
            tau = np.nan_to_num(tau, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            pass
        return {
            "xyz": xyz,
            "t": tangents,
            "n": normals,
            "b": binormals,
            "u": u_arc,
            "arc": arc,
            "kappa": kappa,
            "tau": tau,
        }

    def _generate_branch_tube(self, branch_points, n_length_branch, n_radial):
        """Fit a dense Bishop spline once, then sample one tube level (tests / debug)."""
        dense = self._fit_dense_tract(branch_points)
        return self._tube_from_dense(dense, n_length_branch, n_radial)

    def _tube_from_dense(self, dense, n_length_branch, n_radial):
        u_q = np.linspace(0.0, 1.0, n_length_branch, dtype=np.float64)
        eval_points = self._interp_by_u(dense["u"], dense["xyz"], u_q)
        tangents = self._interp_by_u(dense["u"], dense["t"], u_q)
        normals = self._interp_by_u(dense["u"], dense["n"], u_q)
        binormals = self._interp_by_u(dense["u"], dense["b"], u_q)
        frames = [
            self._orthonormalize_frame(tangents[i], normals[i])
            for i in range(n_length_branch)
        ]
        tangents = np.stack([f[0] for f in frames], axis=0)
        normals = np.stack([f[1] for f in frames], axis=0)
        binormals = np.stack([f[2] for f in frames], axis=0)

        theta = (2.0 * np.pi * np.arange(n_radial, dtype=np.float64) / n_radial) - np.pi
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        n_v = (
            cos_t[None, :, None] * normals[:, None, :]
            + sin_t[None, :, None] * binormals[:, None, :]
        )
        b_v = (
            -sin_t[None, :, None] * normals[:, None, :]
            + cos_t[None, :, None] * binormals[:, None, :]
        )
        t_v = np.repeat(tangents[:, None, :], n_radial, axis=1)
        tube_nodes = eval_points[:, None, :] + self.tube_radius * n_v
        u_grid = np.repeat(u_q[:, None], n_radial, axis=1)
        th_grid = np.repeat(theta[None, :], n_length_branch, axis=0)
        return {
            "nodes": tube_nodes.reshape(-1, 3),
            "u_local": u_grid.reshape(-1),
            "theta": th_grid.reshape(-1),
            "n_v": n_v.reshape(-1, 3),
            "t_v": t_v.reshape(-1, 3),
            "b_v": b_v.reshape(-1, 3),
            "eval_points": eval_points,
            "u_cl": u_q,
            "arc": float(dense["arc"]),
        }

    def _generate_branch_topology(self, n_length_branch, n_radial, node_offset):
        """Quad-grid triangulation; graph diagonals match face diagonals (B–C)."""
        n_len = n_length_branch
        n_rad = n_radial
        ii, jj = np.meshgrid(np.arange(n_len), np.arange(n_rad), indexing="ij")
        idx = node_offset + ii * n_rad + jj
        next_j = (jj + 1) % n_rad
        idx_b_all = node_offset + ii * n_rad + next_j
        ring = np.stack([idx.ravel(), idx_b_all.ravel()], axis=1)

        if n_len > 1:
            idx_a = idx[:-1]
            idx_b = idx_b_all[:-1]
            idx_c = node_offset + (ii[:-1] + 1) * n_rad + jj[:-1]
            idx_d = node_offset + (ii[:-1] + 1) * n_rad + next_j[:-1]
            long = np.stack([idx_a.ravel(), idx_c.ravel()], axis=1)
            diag = np.stack([idx_b.ravel(), idx_c.ravel()], axis=1)
            faces = np.concatenate([
                np.stack([idx_a.ravel(), idx_b.ravel(), idx_c.ravel()], axis=1),
                np.stack([idx_b.ravel(), idx_d.ravel(), idx_c.ravel()], axis=1),
            ], axis=0)
            edges = np.concatenate([ring, long, diag], axis=0)
        else:
            edges = ring
            faces = np.zeros((0, 3), dtype=np.int64)

        edges = np.concatenate([edges, edges[:, ::-1]], axis=0)
        return edges, faces

    def _generate_level(
        self,
        dense_tracts,
        n_length,
        n_radial,
        arc_lengths,
        junction_incidents=None,
        junc_xyz=None,
    ):
        alloc = allocate_ring_counts(n_length, arc_lengths)
        node_chunks = []
        u_local_chunks = []
        theta_chunks = []
        n_chunks, t_chunks, b_chunks = [], [], []
        eval_chunks, u_cl_chunks = [], []
        tract_chunks = []
        step_chunks = []
        all_edges, all_faces = [], []
        node_offset = 0

        for tid, (dense, n_len, arc_b) in enumerate(zip(dense_tracts, alloc, arc_lengths)):
            tube = self._tube_from_dense(dense, n_len, n_radial)
            node_chunks.append(tube["nodes"])
            u_local_chunks.append(tube["u_local"])
            theta_chunks.append(tube["theta"])
            n_chunks.append(tube["n_v"])
            t_chunks.append(tube["t_v"])
            b_chunks.append(tube["b_v"])
            eval_chunks.append(tube["eval_points"])
            u_cl_chunks.append(tube["u_cl"])
            n_nodes = n_len * n_radial
            tract_chunks.append(np.full(n_nodes, tid, dtype=np.int64))
            u_step = 1.0 / max(n_len - 1, 1)
            step_chunks.append(np.full(n_nodes, u_step, dtype=np.float64))
            edges, faces = self._generate_branch_topology(n_len, n_radial, node_offset)
            all_edges.append(edges)
            all_faces.append(faces)
            node_offset += n_nodes

        pos = _torch_f32(np.concatenate(node_chunks, axis=0))
        u_local = _torch_f32(np.concatenate(u_local_chunks))
        theta = _torch_f32(np.concatenate(theta_chunks))
        n_v = _torch_f32(np.concatenate(n_chunks, axis=0))
        t_v = _torch_f32(np.concatenate(t_chunks, axis=0))
        b_v = _torch_f32(np.concatenate(b_chunks, axis=0))
        n_v = n_v / n_v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        t_v = t_v / t_v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        b_v = b_v / b_v.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        if all_edges:
            edge_index = torch.from_numpy(np.concatenate(all_edges, axis=0).T.copy()).long()
            extra = couple_ostium_edges(
                np.concatenate(node_chunks, axis=0),
                np.concatenate(tract_chunks),
                junction_incidents or {},
                junc_xyz or {},
                radius_mm=max(2.0 * self.tube_radius, JUNCTION_COUPLE_RADIUS_MM),
            )
            if extra.shape[0] > 0:
                extra_t = torch.from_numpy(np.ascontiguousarray(extra.T)).long()
                edge_index = torch.cat([edge_index, extra_t], dim=1)
            edge_index = torch.unique(edge_index, dim=1)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        if all_faces:
            faces = torch.from_numpy(np.concatenate(all_faces, axis=0).copy()).long()
        else:
            faces = torch.zeros((0, 3), dtype=torch.long)
        face = faces.t().contiguous()

        cl_dense = np.concatenate(eval_chunks, axis=0)
        cl_u = np.concatenate(u_cl_chunks)
        cl_tract = np.concatenate([
            np.full(len(u_cl), tid, dtype=np.int64)
            for tid, u_cl in enumerate(u_cl_chunks)
        ])

        return {
            "pos": pos,
            "u": u_local,
            "theta": theta,
            "normal": n_v,
            "tangent": t_v,
            "binormal": b_v,
            "edge_index": edge_index,
            "face": face,
            "branch_nl": torch.tensor(alloc, dtype=torch.long),
            "n_radial": torch.tensor(int(n_radial), dtype=torch.long),
            "cl_dense": _torch_f32(cl_dense),
            "cl_dense_u": _torch_f32(cl_u),
            "cl_tract_id": _torch_long(cl_tract),
            "tract_id": _torch_long(np.concatenate(tract_chunks)),
            "u_step": _torch_f32(np.concatenate(step_chunks)),
        }

    def _hybrid_true_points(self, vessel_pts, cl_xyz):
        x_true, d_true, _ = sample_x_true(
            vessel_pts,
            None,
            cl_xyz,
            n_true=self.n_true,
            tube_radius=self.tube_radius,
        )
        return x_true, d_true

    def _true_normals_at_points(self, query_pts, mesh_pts, mesh_normals):
        query_pts = np.asarray(query_pts, dtype=np.float64).reshape(-1, 3)
        mesh_pts = np.asarray(mesh_pts, dtype=np.float64).reshape(-1, 3)
        mesh_normals = np.asarray(mesh_normals, dtype=np.float64).reshape(-1, 3)
        if query_pts.shape[0] == 0:
            return _torch_f32(np.zeros((0, 3)))
        if mesh_pts.shape[0] == 0:
            nrm = np.zeros_like(query_pts)
            nrm[:, 2] = 1.0
            return _torch_f32(nrm)
        from scipy.spatial import cKDTree

        _, idx = cKDTree(mesh_pts).query(query_pts, k=1)
        nrm = mesh_normals[np.asarray(idx, dtype=np.int64)]
        nrm = nrm / np.clip(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-8, None)
        return _torch_f32(nrm)

    def _posed_mesh_normals(self, vessel_mesh, origin, R):
        if vessel_mesh is None:
            return None, None
        mesh = vessel_mesh
        try:
            mesh = mesh.compute_normals(point_normals=True, cell_normals=False, inplace=False)
        except Exception:
            pass
        pts = (_as_f64(mesh.points) - origin) @ R
        nrm = getattr(mesh, "point_normals", None)
        if nrm is None or len(nrm) != len(pts):
            return pts, None
        nrm = _as_f64(nrm) @ R
        nrm = nrm / np.clip(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-8, None)
        return pts, nrm

    def _build_latent_tokens(self, dense_tracts, junction_incidents, junc_xyz, arc_lengths):
        """Arc-length tokens at TOKEN_SPACING_MM; pad with zeros / False (§5.4).

        Per branch `n_tok = floor(L / spacing) + 1` at arc positions `k · 2 mm`
        (first token at the branch start). The daughter's first token *is* the
        junction — no separate junction slots. `LATENT_LEN` is a padding max.
        """
        del junction_incidents, junc_xyz  # no extra junction tokens (§5.4)
        spacing = float(getattr(self, "token_spacing_mm", TOKEN_SPACING_MM))
        pad = int(self.latent_len)
        n_tracts = min(len(dense_tracts), MAX_TRACTS)
        dense_tracts = dense_tracts[:n_tracts]
        arc_lengths = list(arc_lengths[:n_tracts])

        per_tract = []
        for tid, (dense, arc) in enumerate(zip(dense_tracts, arc_lengths)):
            arc = float(arc)
            n_tok = int(np.floor(arc / max(spacing, 1e-6))) + 1
            n_tok = max(1, n_tok)
            slots = []
            for k in range(n_tok):
                s_mm = k * spacing
                u = 0.0 if arc <= 1e-12 else float(min(s_mm / arc, 1.0))
                xyz = self._interp_by_u(dense["u"], dense["xyz"], np.asarray([u], dtype=np.float64))
                slots.append((k, u, xyz[0], tid))
            per_tract.append(slots)

        # Fill k=0 of every tract first so a short LATENT_LEN still covers the tree.
        ordered = []
        max_k = max((len(s) for s in per_tract), default=0)
        for k in range(max_k):
            for slots in per_tract:
                if k < len(slots):
                    ordered.append(slots[k])

        token_u = np.zeros(pad, dtype=np.float64)
        token_tract = np.zeros(pad, dtype=np.int64)
        token_is_junc = np.zeros(pad, dtype=np.int64)
        token_pos = np.zeros((pad, 3), dtype=np.float64)
        attend = np.zeros((pad, MAX_TRACTS), dtype=np.bool_)
        valid = np.zeros(pad, dtype=bool)
        n_keep = min(len(ordered), pad)
        for slot in range(n_keep):
            _k, u, xyz, tid = ordered[slot]
            token_u[slot] = u
            token_tract[slot] = int(tid)
            token_pos[slot] = xyz
            if 0 <= int(tid) < MAX_TRACTS:
                attend[slot, int(tid)] = True
            valid[slot] = True

        latent_u = _torch_f32(token_u)
        latent_pos = _torch_f32(token_pos)
        latent_tract = _torch_long(token_tract)
        latent_valid = torch.from_numpy(valid.copy()).bool()
        return {
            "latent_u": latent_u,
            "latent_tract_id": latent_tract,
            "latent_is_junction": _torch_long(token_is_junc),
            "latent_pos": latent_pos,
            "latent_valid": latent_valid,
            "token_u": latent_u.clone(),
            "token_pos": latent_pos.clone(),
            "token_tract_id": latent_tract.clone(),
            "token_attend": torch.from_numpy(attend.copy()).bool(),
            "n_tracts": torch.tensor(n_tracts, dtype=torch.long),
        }

    def _prepare_tracts(self, centerline_mesh, require_groupids=False):
        tracts, endpoints, _ = extract_groupid_tracts(
            centerline_mesh,
            endpoint_snap_mm=GROUPID_ENDPOINT_SNAP_MM,
            require_groupids=require_groupids,
        )
        if len(tracts) > MAX_TRACTS:
            order = np.argsort([-_arc_len(t) for t in tracts])[:MAX_TRACTS]
            tracts = [tracts[i] for i in order]
            endpoints = [endpoints[i] for i in order]
        inlet_i, reverse = _choose_inlet(tracts, endpoints)
        tracts, endpoints, junc_inc, junc_xyz = _orient_tracts(
            tracts, endpoints, inlet_i, reverse
        )
        cl_all = np.concatenate(tracts, axis=0)
        origin = cl_all.mean(axis=0)
        tracts = [t - origin for t in tracts]
        junc_xyz = {k: v - origin for k, v in junc_xyz.items()}
        R = _canonical_pose(tracts[inlet_i])
        tracts = [t @ R for t in tracts]
        junc_xyz = {k: v @ R for k, v in junc_xyz.items()}
        return tracts, junc_inc, junc_xyz, origin, R, inlet_i

    def _level_r_star(self, level, dense_tracts, gt_mesh):
        if gt_mesh is None:
            packed = empty_r_star(int(level["pos"].size(0)))
        else:
            packed = compute_level_r_star(
                level["pos"].numpy(),
                level["normal"].numpy(),
                level["u"].numpy(),
                level["tract_id"].numpy(),
                level["branch_nl"].numpy(),
                int(level["n_radial"].item()),
                dense_tracts,
                gt_mesh,
                tube_radius=self.tube_radius,
            )
        return {
            "r_star": _torch_f32(packed["r_star"]),
            "valid": torch.from_numpy(np.ascontiguousarray(packed["valid"])).bool(),
            "ambiguous": torch.from_numpy(
                np.ascontiguousarray(packed.get("ambiguous", np.zeros(packed["valid"].shape, dtype=bool)))
            ).bool(),
            "dth": _torch_f32(packed["dth"]),
            "du": _torch_f32(packed["du"]),
            "ring_med": _torch_f32(packed["ring_med"]),
        }

    def _attach_mesh_r_star_stats(self, packed, level):
        pos = level["pos"].numpy() if torch.is_tensor(level["pos"]) else np.asarray(level["pos"])
        edges = level["edge_index"].numpy().T if torch.is_tensor(level["edge_index"]) else np.asarray(level["edge_index"]).T
        if edges.ndim != 2 or edges.shape[1] != 2:
            edges = np.zeros((0, 2), dtype=np.int64)
        dth, du, ring_med = mesh_r_star_edge_stats(
            pos, packed["r_star"], packed["valid"], edges
        )
        packed["dth"] = dth
        packed["du"] = du
        packed["ring_med"] = ring_med
        return packed

    def _resample_point_scalar(self, src_pos, values, dst_pos):
        src_pos = _as_f64(src_pos).reshape(-1, 3)
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        dst_pos = _as_f64(dst_pos).reshape(-1, 3)
        if src_pos.shape[0] == 0 or values.shape[0] != src_pos.shape[0] or dst_pos.shape[0] == 0:
            return None
        _, idx = cKDTree(src_pos).query(dst_pos, k=1, workers=1)
        return values[np.asarray(idx, dtype=np.int64)]

    def _template_r_star(self, level, gt_mesh=None, stretch=None):
        pos = level["pos"].numpy()
        normal = level["normal"].numpy()
        r_local = level["r_local"].numpy() if "r_local" in level else np.zeros(pos.shape[0])
        n = int(pos.shape[0])
        packed = empty_r_star(n)
        stretch_arr = None if stretch is None else np.asarray(stretch, dtype=np.float64).reshape(-1)
        used_stretch = False
        if stretch_arr is not None and stretch_arr.shape[0] == n and np.isfinite(stretch_arr).any():
            mag = np.abs(stretch_arr[np.isfinite(stretch_arr)])
            frac = float(np.mean(mag > 0.3)) if mag.size else 0.0
            peak = float(np.max(mag)) if mag.size else 0.0
            # All-zero / 1-vertex exports are the §2.3 orientation bug, not a true field.
            used_stretch = frac > 0.005 or peak > 2.0
        if used_stretch:
            packed = stretch_distance_r_star(r_local, stretch_arr)
        elif gt_mesh is not None:
            packed = template_ray_r_star(pos, normal, r_local, gt_mesh)
        packed = self._attach_mesh_r_star_stats(packed, level)
        return {
            "r_star": _torch_f32(packed["r_star"]),
            "valid": torch.from_numpy(np.ascontiguousarray(packed["valid"])).bool(),
            "ambiguous": torch.from_numpy(
                np.ascontiguousarray(packed.get("ambiguous", np.zeros(packed["valid"].shape, dtype=bool)))
            ).bool(),
            "dth": _torch_f32(packed["dth"]),
            "du": _torch_f32(packed["du"]),
            "ring_med": _torch_f32(packed["ring_med"]),
        }

    def _dense_cl_pack(self, dense_tracts):
        xyz = np.concatenate([d["xyz"] for d in dense_tracts], axis=0)
        u = np.concatenate([d["u"] for d in dense_tracts], axis=0)
        tract = np.concatenate(
            [np.full(len(d["xyz"]), tid, dtype=np.int64) for tid, d in enumerate(dense_tracts)]
        )
        return {
            "cl_dense": _torch_f32(xyz),
            "cl_dense_u": _torch_f32(u),
            "cl_tract_id": _torch_long(tract),
        }

    def _project_points_to_tracts(self, points, dense_tracts):
        xyz = np.concatenate([d["xyz"] for d in dense_tracts], axis=0)
        u = np.concatenate([d["u"] for d in dense_tracts], axis=0)
        t = np.concatenate([d["t"] for d in dense_tracts], axis=0)
        n = np.concatenate([d["n"] for d in dense_tracts], axis=0)
        b = np.concatenate([d["b"] for d in dense_tracts], axis=0)
        tract_id = np.concatenate(
            [np.full(len(d["xyz"]), tid, dtype=np.int64) for tid, d in enumerate(dense_tracts)]
        )
        kappa = np.concatenate(
            [
                np.asarray(d.get("kappa", np.zeros(len(d["xyz"]))), dtype=np.float64).reshape(-1)
                for d in dense_tracts
            ]
        )
        tau = np.concatenate(
            [
                np.asarray(d.get("tau", np.zeros(len(d["xyz"]))), dtype=np.float64).reshape(-1)
                for d in dense_tracts
            ]
        )
        _, idx = cKDTree(xyz).query(points, k=1, workers=1)
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        cl = xyz[idx]
        rel = points - cl
        n_i, b_i, t_i = n[idx], b[idx], t[idx]
        x_nb = np.einsum("ij,ij->i", rel, n_i)
        y_nb = np.einsum("ij,ij->i", rel, b_i)
        theta = np.arctan2(y_nb, x_nb)
        return {
            "u": u[idx],
            "theta": theta,
            "tract_id": tract_id[idx],
            "t": t_i,
            "n_cl": n_i,
            "b_cl": b_i,
            "r_local": np.linalg.norm(rel, axis=1),
            "cl": cl,
            "kappa": kappa[idx],
            "tau": tau[idx],
        }

    def _vertex_frames_from_mesh(self, mesh_n, t_cl, n_cl, b_cl, theta):
        mesh_n = _as_f64(mesh_n).reshape(-1, 3)
        t_cl = _as_f64(t_cl).reshape(-1, 3)
        n_cl = _as_f64(n_cl).reshape(-1, 3)
        b_cl = _as_f64(b_cl).reshape(-1, 3)
        theta = np.asarray(theta, dtype=np.float64).reshape(-1)
        nrm = np.linalg.norm(mesh_n, axis=1, keepdims=True)
        n_v = mesh_n / np.clip(nrm, 1e-8, None)
        cos_t = np.cos(theta)[:, None]
        sin_t = np.sin(theta)[:, None]
        n_bishop = cos_t * n_cl + sin_t * b_cl
        n_bishop = n_bishop / np.clip(np.linalg.norm(n_bishop, axis=1, keepdims=True), 1e-8, None)
        weak = nrm.reshape(-1) < 0.5
        n_v = np.where(weak[:, None], n_bishop, n_v)
        t_v = t_cl - n_v * np.einsum("ij,ij->i", t_cl, n_v)[:, None]
        t_norm = np.linalg.norm(t_v, axis=1, keepdims=True)
        t_v = np.where(t_norm < 1e-6, t_cl, t_v / np.clip(t_norm, 1e-8, None))
        b_v = np.cross(n_v, t_v)
        b_norm = np.linalg.norm(b_v, axis=1, keepdims=True)
        bad = b_norm.reshape(-1) < 1e-6
        b_v = np.where(bad[:, None], np.cross(n_bishop, t_cl), b_v / np.clip(b_norm, 1e-8, None))
        t_v = np.cross(b_v, n_v)
        t_v = t_v / np.clip(np.linalg.norm(t_v, axis=1, keepdims=True), 1e-8, None)
        n_v = n_v / np.clip(np.linalg.norm(n_v, axis=1, keepdims=True), 1e-8, None)
        b_v = b_v / np.clip(np.linalg.norm(b_v, axis=1, keepdims=True), 1e-8, None)
        return n_v, t_v, b_v

    def _polydata_triangles(self, mesh):
        mesh = pv.wrap(mesh)
        if mesh.n_cells > 0 and not bool(mesh.is_all_triangles):
            mesh = mesh.triangulate()
        faces = None
        if hasattr(mesh, "regular_faces"):
            cand = np.asarray(mesh.regular_faces)
            if cand.ndim == 2 and cand.shape[1] == 3 and cand.shape[0] > 0:
                faces = np.asarray(cand, dtype=np.int64)
        if faces is None:
            raw = np.asarray(getattr(mesh, "faces", np.zeros(0, dtype=np.int64)), dtype=np.int64).reshape(-1)
            tris = []
            i = 0
            while i < raw.size:
                n = int(raw[i])
                if n == 3 and i + 3 < raw.size:
                    tris.append(raw[i + 1 : i + 4])
                i += max(n, 0) + 1
            faces = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
        return mesh, faces

    def _edges_from_faces(self, faces, n_points):
        if faces is None or len(faces) == 0:
            return np.zeros((0, 2), dtype=np.int64)
        e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
        e = np.sort(e, axis=1)
        e = np.unique(e, axis=0)
        e = e[(e[:, 0] >= 0) & (e[:, 1] >= 0) & (e[:, 0] < n_points) & (e[:, 1] < n_points)]
        if e.size == 0:
            return np.zeros((0, 2), dtype=np.int64)
        return np.concatenate([e, e[:, ::-1]], axis=0)

    def _knn_edges(self, pts, k=6):
        pts = _as_f64(pts).reshape(-1, 3)
        n = int(pts.shape[0])
        if n < 2:
            return np.zeros((0, 2), dtype=np.int64)
        kk = min(int(k) + 1, n)
        _, idx = cKDTree(pts).query(pts, k=kk, workers=1)
        idx = np.asarray(idx, dtype=np.int64).reshape(n, kk)
        pairs = []
        for i in range(n):
            for j in idx[i, 1:]:
                j = int(j)
                if 0 <= j < n and j != i:
                    pairs.append((i, j))
                    pairs.append((j, i))
        if not pairs:
            return np.zeros((0, 2), dtype=np.int64)
        return np.unique(np.asarray(pairs, dtype=np.int64), axis=0)

    def _u_step_from_edges(self, u, tract_id, edges):
        u = np.asarray(u, dtype=np.float64).reshape(-1)
        tract_id = np.asarray(tract_id, dtype=np.int64).reshape(-1)
        n = int(u.shape[0])
        step = np.full(n, 1e-3, dtype=np.float64)
        if edges is None or len(edges) == 0:
            return step
        src = np.asarray(edges[:, 0], dtype=np.int64)
        dst = np.asarray(edges[:, 1], dtype=np.int64)
        same = tract_id[src] == tract_id[dst]
        du = np.abs(u[src] - u[dst])
        acc = np.zeros(n, dtype=np.float64)
        cnt = np.zeros(n, dtype=np.float64)
        np.add.at(acc, src[same], du[same])
        np.add.at(cnt, src[same], 1.0)
        mask = cnt > 0
        step[mask] = np.maximum(acc[mask] / cnt[mask], 1e-4)
        return step

    def _template_levels(self, posed_tpl, fine_faces, fine):
        """Nested mid/coarse levels by sizing-field edge collapse (coarsen.py).

        Decimation erased the template's density: the flat, finely meshed sac
        went first, and on p131 the sac/parent edge ratio fell from 4.9 to
        0.83.  The collapse keeps the ratio, keeps >= N_min vertices around
        thin branches and rims, and prolongs by barycentric projection instead
        of a Euclidean kNN that reached across thin branches.
        """
        pts = _as_f64(posed_tpl.points)
        tel = _point_data_array(posed_tpl, "TargetEdgeLength")
        r_tpl = _point_data_array(posed_tpl, "R_template")
        if tel is None or tel.shape[0] != pts.shape[0]:
            # templates without a sizing field: their own mean incident edge length
            e = np.concatenate([fine_faces[:, [0, 1]], fine_faces[:, [1, 2]], fine_faces[:, [2, 0]]], axis=0)
            le = np.linalg.norm(pts[e[:, 0]] - pts[e[:, 1]], axis=1)
            acc = np.bincount(e[:, 0], le, minlength=pts.shape[0]) + np.bincount(e[:, 1], le, minlength=pts.shape[0])
            cnt = np.bincount(e[:, 0], minlength=pts.shape[0]) + np.bincount(e[:, 1], minlength=pts.shape[0])
            tel = acc / np.maximum(cnt, 1)
        if r_tpl is None or r_tpl.shape[0] != pts.shape[0]:
            r_tpl = fine["r_local"].numpy().astype(np.float64)
        tel = np.clip(np.asarray(tel, dtype=np.float64), 1e-3, None)
        r_tpl = np.clip(np.asarray(r_tpl, dtype=np.float64), 1e-2, None)
        lv = build_template_levels(
            pts, fine_faces, tel, r_tpl,
            k_mid=TEMPLATE_MID_K, k_coarse=TEMPLATE_COARSE_K,
            n_min_mid=TEMPLATE_MID_N_MIN, n_min_coarse=TEMPLATE_COARSE_N_MIN,
            min_rim_mid=TEMPLATE_MID_N_MIN, min_rim_coarse=TEMPLATE_COARSE_N_MIN,
        )

        def sub_mesh(keep, faces):
            cells = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).ravel()
            return pv.PolyData(pts[keep], cells)

        lv["mid_mesh"] = sub_mesh(lv["keep_mid"], lv["faces_mid"])
        lv["coarse_mesh"] = sub_mesh(lv["keep_coarse"], lv["faces_coarse"])
        return lv

    def _surface_normals(self, mesh, n_points):
        nrm = getattr(mesh, "point_normals", None)
        if nrm is None or len(nrm) != n_points:
            try:
                mesh = mesh.compute_normals(point_normals=True, cell_normals=False, inplace=False)
                nrm = getattr(mesh, "point_normals", None)
            except Exception:
                nrm = None
        if nrm is None or len(nrm) != n_points:
            return np.zeros((n_points, 3), dtype=np.float64)
        nrm = _as_f64(nrm)
        nrm = nrm / np.clip(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-8, None)
        return nrm

    def _orient_normals_outward(self, normals, pos, cl_xyz):
        """Flip n so n · (x − cl_nearest) ≥ 0 at every vertex (§10.2 step 2)."""
        normals = _as_f64(normals).reshape(-1, 3)
        pos = _as_f64(pos).reshape(-1, 3)
        cl_xyz = _as_f64(cl_xyz).reshape(-1, 3)
        if normals.shape[0] == 0:
            return normals
        if cl_xyz.shape[0] == 0:
            return normals
        _, idx = cKDTree(cl_xyz).query(pos, k=1, workers=1)
        radial = pos - cl_xyz[np.asarray(idx, dtype=np.int64)]
        sign = np.sign(np.einsum("ij,ij->i", normals, radial))
        sign[sign == 0.0] = 1.0
        out = normals * sign[:, None]
        nn = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(nn, 1e-8, None)

    def _face_topology(self, faces, n_points):
        """Return (n_components, n_nonmanifold_edges, n_boundary_loops)."""
        faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3) if faces is not None else np.zeros((0, 3), dtype=np.int64)
        n_points = int(n_points)
        if faces.size == 0 or n_points < 1:
            return 0, 0, 0
        e = np.sort(
            np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0),
            axis=1,
        )
        uniq, counts = np.unique(e, axis=0, return_counts=True)
        n_nonman = int(np.sum(counts > 2))
        parent = np.arange(n_points, dtype=np.int64)

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(a, b):
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[rb] = ra

        used = np.zeros(n_points, dtype=bool)
        for a, b, c in faces:
            union(a, b)
            union(b, c)
            used[int(a)] = True
            used[int(b)] = True
            used[int(c)] = True
        roots = {int(find(i)) for i in range(n_points) if used[i]}
        n_comp = int(len(roots))
        bmask = counts == 1
        bedges = uniq[bmask]
        if bedges.shape[0] == 0:
            return n_comp, n_nonman, 0
        adj = defaultdict(list)
        for a, b in bedges:
            a, b = int(a), int(b)
            adj[a].append(b)
            adj[b].append(a)
        seen = set()
        n_loops = 0
        for start in adj:
            if start in seen:
                continue
            n_loops += 1
            stack = [start]
            seen.add(start)
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v)
                        stack.append(v)
        return n_comp, n_nonman, n_loops

    def _assert_surface_topology(self, faces, n_points, name="template"):
        n_comp, n_nonman, n_loops = self._face_topology(faces, n_points)
        # Caps on VTK cylinders can be disconnected vertex islands; do not
        # fail the whole sample on component count. Non-manifold edges are
        # a hard topology break (§10.2 step 5).
        if n_nonman > 0:
            raise ValueError(f"{name}: {n_nonman} non-manifold edges")
        return n_comp, n_nonman, n_loops

    def _boundary_loop_vertices(self, faces, n_points):
        """Connected components of boundary edges as vertex-id arrays."""
        faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3) if faces is not None else np.zeros((0, 3), dtype=np.int64)
        n_points = int(n_points)
        if faces.size == 0 or n_points < 1:
            return []
        e = np.sort(
            np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0),
            axis=1,
        )
        uniq, counts = np.unique(e, axis=0, return_counts=True)
        bedges = uniq[counts == 1]
        if bedges.shape[0] == 0:
            return []
        adj = defaultdict(list)
        for a, b in bedges:
            a, b = int(a), int(b)
            adj[a].append(b)
            adj[b].append(a)
        seen = set()
        loops = []
        for start in adj:
            if start in seen:
                continue
            comp = []
            stack = [start]
            seen.add(start)
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v)
                        stack.append(v)
            if len(comp) >= 3:
                loops.append(np.asarray(comp, dtype=np.int64))
        return loops

    def _fit_loop_plane(self, pts, cl_xyz, cl_t):
        pts = _as_f64(pts).reshape(-1, 3)
        origin = pts.mean(axis=0)
        if pts.shape[0] < 3:
            nrm = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            c = pts - origin
            cov = (c.T @ c) / max(float(pts.shape[0]), 1.0)
            w, v = np.linalg.eigh(cov)
            nrm = v[:, int(np.argmin(w))]
        nrm = nrm / (np.linalg.norm(nrm) + 1e-12)
        if cl_xyz is not None and len(cl_xyz) > 0:
            _, idx = cKDTree(_as_f64(cl_xyz).reshape(-1, 3)).query(origin.reshape(1, 3), k=1)
            idx = int(np.asarray(idx).reshape(-1)[0])
            if cl_t is not None and len(cl_t) == len(cl_xyz):
                tang = _as_f64(cl_t)[idx]
            else:
                tang = origin - _as_f64(cl_xyz)[idx]
            if float(np.dot(nrm, tang)) < 0.0:
                nrm = -nrm
        return origin, nrm

    def _pose_ostium_frames(self, frames, origin, R):
        if frames is None:
            return None
        orig, nrm = frames
        orig = _as_f64(orig).reshape(-1, 3)
        nrm = _as_f64(nrm).reshape(-1, 3)
        if orig.shape[0] == 0:
            return None
        origin = _as_f64(origin).reshape(3)
        R = _as_f64(R).reshape(3, 3)
        orig_p = (orig - origin) @ R
        nrm_p = nrm @ R
        nn = np.linalg.norm(nrm_p, axis=1, keepdims=True)
        nrm_p = nrm_p / np.clip(nn, 1e-8, None)
        return orig_p, nrm_p

    def _match_frame_to_loop(self, centroid, frames, max_mm=8.0):
        if frames is None:
            return None, None
        orig, nrm = frames
        if orig.shape[0] == 0:
            return None, None
        d = np.linalg.norm(orig - centroid.reshape(1, 3), axis=1)
        j = int(np.argmin(d))
        if float(d[j]) > float(max_mm):
            return None, None
        return orig[j], nrm[j]

    def _level_boundary_and_geom(self, level, dense_tracts, suffix="", ostium_frames=None):
        """Planes + [curvature, torsion, d_ostium] for one scaffold level.

        Curvature/torsion are CL Frenet values at the nearest dense sample
        (not a per-vertex surface Frenet). Ostium distance is to the nearest
        boundary-loop vertex.
        """
        pos = level["pos"].numpy() if torch.is_tensor(level["pos"]) else np.asarray(level["pos"])
        n = int(pos.shape[0])
        face = level.get("face")
        if torch.is_tensor(face):
            faces = face.numpy().T if face.dim() == 2 and int(face.size(0)) == 3 else face.numpy()
        else:
            faces = np.zeros((0, 3), dtype=np.int64)
        faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        cl_xyz = np.concatenate([d["xyz"] for d in dense_tracts], axis=0) if dense_tracts else np.zeros((0, 3))
        cl_t = np.concatenate([d["t"] for d in dense_tracts], axis=0) if dense_tracts else np.zeros((0, 3))
        loops = self._boundary_loop_vertices(faces, n)
        origin_out = np.zeros((n, 3), dtype=np.float64)
        normal_out = np.zeros((n, 3), dtype=np.float64)
        mask = np.zeros(n, dtype=bool)
        rim_pts = []
        for loop in loops:
            loop = np.asarray(loop, dtype=np.int64)
            loop = loop[(loop >= 0) & (loop < n)]
            if loop.size < 3:
                continue
            centroid = pos[loop].mean(axis=0)
            fo, fn = self._match_frame_to_loop(centroid, ostium_frames)
            if fo is None:
                fo, fn = self._fit_loop_plane(pos[loop], cl_xyz, cl_t)
            origin_out[loop] = fo
            normal_out[loop] = fn
            mask[loop] = True
            rim_pts.append(pos[loop])
        d_ost = np.zeros(n, dtype=np.float64)
        if rim_pts:
            rim = np.concatenate(rim_pts, axis=0)
            _, idx = cKDTree(rim).query(pos, k=1, workers=1)
            d_ost = np.linalg.norm(pos - rim[np.asarray(idx, dtype=np.int64)], axis=1)
        elif n > 0 and "u" in level:
            u = level["u"].numpy() if torch.is_tensor(level["u"]) else np.asarray(level["u"])
            end = (u <= 0.02) | (u >= 0.98)
            if np.any(end):
                _, idx = cKDTree(pos[end]).query(pos, k=1, workers=1)
                d_ost = np.linalg.norm(pos - pos[end][np.asarray(idx, dtype=np.int64)], axis=1)
        proj = self._project_points_to_tracts(pos, dense_tracts) if dense_tracts else None
        kappa = proj["kappa"] if proj is not None else np.zeros(n)
        tau = proj["tau"] if proj is not None else np.zeros(n)
        out = {
            f"boundary_plane_origin{suffix}": _torch_f32(origin_out),
            f"boundary_plane_normal{suffix}": _torch_f32(normal_out),
            f"boundary_mask{suffix}": torch.from_numpy(np.ascontiguousarray(mask)).bool(),
            f"d_ostium{suffix}": _torch_f32(d_ost),
        }
        if suffix == "":
            out["curvature"] = _torch_f32(kappa)
            out["torsion"] = _torch_f32(tau)
        else:
            out[f"curvature{suffix}"] = _torch_f32(kappa)
            out[f"torsion{suffix}"] = _torch_f32(tau)
        return out

    def _load_ostium_frames_sidecar(self, sample):
        """Read `{id}.ostium_frames.npz` next to GT/template if present (read-only)."""
        dataset_id = str(sample.get("dataset_id", "") or "")
        if not dataset_id:
            return None
        name = f"{dataset_id}.ostium_frames.npz"
        candidates = []
        for key in ("vessel_file", "template_mesh_file"):
            path = sample.get(key)
            if path:
                candidates.append(os.path.join(os.path.dirname(os.path.abspath(path)), name))
        for path in candidates:
            if not os.path.isfile(path):
                continue
            try:
                blob = np.load(path)
                origin = np.asarray(blob["origin"], dtype=np.float64).reshape(-1, 3)
                normal = np.asarray(blob["normal"], dtype=np.float64).reshape(-1, 3)
                if origin.shape[0] == 0:
                    continue
                return origin, normal
            except Exception:
                continue
        return None

    def _ostium_frames_from_mesh(self, mesh):
        if mesh is None:
            return None
        fd = getattr(mesh, "field_data", None)
        if fd is None:
            return None
        for o_name, n_name in (
            ("ostium_origin", "ostium_normal"),
            ("cut_origin", "cut_normal"),
            ("plane_origin", "plane_normal"),
        ):
            if o_name in fd and n_name in fd:
                orig = np.asarray(fd[o_name], dtype=np.float64).reshape(-1, 3)
                nrm = np.asarray(fd[n_name], dtype=np.float64).reshape(-1, 3)
                if orig.shape[0] > 0:
                    return orig, nrm
        return None

    def _level_from_surface(self, mesh, dense_tracts):
        mesh, faces = self._polydata_triangles(mesh)
        pts = _as_f64(mesh.points)
        n = int(pts.shape[0])
        if n < 4:
            raise ValueError("template surface has fewer than 4 vertices")
        proj = self._project_points_to_tracts(pts, dense_tracts)
        nrm = self._surface_normals(mesh, n)
        nrm = self._orient_normals_outward(nrm, pts, proj["cl"])
        n_v, t_v, b_v = self._vertex_frames_from_mesh(
            nrm, proj["t"], proj["n_cl"], proj["b_cl"], proj["theta"]
        )
        edges = self._edges_from_faces(faces, n)
        if edges.shape[0] == 0:
            edges = self._knn_edges(pts)
        u_step = self._u_step_from_edges(proj["u"], proj["tract_id"], edges)
        if edges.shape[0] > 0:
            edge_index = torch.from_numpy(np.ascontiguousarray(edges.T)).long()
            edge_index = torch.unique(edge_index, dim=1)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        if faces is not None and len(faces) > 0:
            face = torch.from_numpy(np.ascontiguousarray(faces.T)).long()
        else:
            face = torch.zeros((3, 0), dtype=torch.long)
        return {
            "pos": _torch_f32(pts),
            "u": _torch_f32(proj["u"]),
            "theta": _torch_f32(proj["theta"]),
            "normal": _torch_f32(n_v),
            "tangent": _torch_f32(t_v),
            "binormal": _torch_f32(b_v),
            "edge_index": edge_index,
            "face": face,
            "branch_nl": torch.tensor([n], dtype=torch.long),
            "n_radial": torch.tensor(1, dtype=torch.long),
            "tract_id": _torch_long(proj["tract_id"]),
            "u_step": _torch_f32(u_step),
            "r_local": _torch_f32(proj["r_local"]),
            "kappa": _torch_f32(proj["kappa"]),
            "tau": _torch_f32(proj["tau"]),
        }

    def _gt_and_tokens(
        self,
        dense_tracts,
        junc_inc,
        junc_xyz,
        arc_lengths,
        origin,
        R,
        vessel_points,
        vessel_mesh,
        cl_pack=None,
    ):
        tokens = self._build_latent_tokens(dense_tracts, junc_inc, junc_xyz, arc_lengths)
        if cl_pack is None:
            cl_pack = self._dense_cl_pack(dense_tracts)
        cl_xyz = cl_pack["cl_dense"].numpy()
        gt_mesh = transform_vessel_mesh(vessel_mesh, origin, R) if vessel_mesh is not None else None
        if vessel_mesh is not None:
            vessel_points = np.asarray(vessel_mesh.points)
        gt_extra = {}
        if vessel_points is not None:
            vessel = (_as_f64(vessel_points) - origin) @ R
            if gt_mesh is not None:
                gt_pts = _as_f64(gt_mesh.points)
            else:
                gt_pts = vessel
            nrm = None
            if gt_mesh is not None:
                nrm = closest_cell_normals(gt_mesh, gt_pts)
            if nrm is None or float(np.linalg.norm(nrm, axis=1).mean()) <= 0.5:
                posed_pts, posed_nrm = self._posed_mesh_normals(vessel_mesh, origin, R)
                if posed_nrm is not None:
                    nrm = posed_nrm
                    if posed_pts.shape[0] == gt_pts.shape[0]:
                        gt_pts = posed_pts
            if nrm is None or nrm.shape[0] != gt_pts.shape[0]:
                nrm = np.zeros_like(gt_pts)
                nrm[:, 2] = 1.0
            nrm = self._orient_normals_outward(nrm, gt_pts, cl_xyz)
            gt_faces = None
            if gt_mesh is not None:
                _, gt_faces = self._polydata_triangles(gt_mesh)
            gt_extra = {
                "gt_points": _torch_f32(gt_pts),
                "gt_normals": _torch_f32(nrm),
                "gt_points_normal": _torch_f32(nrm),
            }
            cl_dist = point_to_polyline_dist(gt_pts.astype(np.float32), cl_xyz).astype(np.float64)
            gt_extra["gt_cl_dist"] = _torch_f32(cl_dist)
            if gt_faces is not None and len(gt_faces) > 0:
                face_t = torch.from_numpy(np.ascontiguousarray(np.asarray(gt_faces, dtype=np.int64).T)).long()
                gt_extra["gt_faces"] = face_t
            x_true, x_true_cl_dist, x_true_normal = sample_x_true(
                gt_pts, nrm, cl_xyz, n_true=self.n_true, tube_radius=self.tube_radius
            )
            if float(x_true_normal.float().norm(dim=-1).mean()) <= 0.5:
                query_nrm = closest_cell_normals(gt_mesh, x_true.numpy()) if gt_mesh is not None else None
                if query_nrm is not None and float(np.linalg.norm(query_nrm, axis=1).mean()) > 0.5:
                    x_true_normal = _torch_f32(query_nrm)
        else:
            x_true = torch.zeros((1, 3), dtype=torch.float32)
            x_true_cl_dist = torch.zeros((1,), dtype=torch.float32)
            x_true_normal = torch.zeros((1, 3), dtype=torch.float32)
        return tokens, cl_pack, gt_mesh, x_true, x_true_cl_dist, x_true_normal, gt_extra

    def _assemble_scaffold_data(
        self,
        fine,
        mid,
        coarse,
        tokens,
        cl_pack,
        x_true,
        x_true_cl_dist,
        x_true_normal,
        r_fine,
        r_mid,
        origin,
        R,
        extra=None,
    ):
        data = AneurysmData(
            x=fine["pos"],
            edge_index=fine["edge_index"],
            face=fine["face"],
            u=fine["u"],
            theta=fine["theta"],
            normal=fine["normal"],
            tangent=fine["tangent"],
            binormal=fine["binormal"],
            tract_id=fine["tract_id"],
            u_step=fine["u_step"],
            pos_mid=mid["pos"],
            edge_index_mid=mid["edge_index"],
            face_mid=mid["face"],
            u_mid=mid["u"],
            theta_mid=mid["theta"],
            normal_mid=mid["normal"],
            tangent_mid=mid["tangent"],
            binormal_mid=mid["binormal"],
            tract_id_mid=mid["tract_id"],
            u_step_mid=mid["u_step"],
            pos_coarse=coarse["pos"],
            edge_index_coarse=coarse["edge_index"],
            face_coarse=coarse["face"],
            u_coarse=coarse["u"],
            theta_coarse=coarse["theta"],
            normal_coarse=coarse["normal"],
            tangent_coarse=coarse["tangent"],
            binormal_coarse=coarse["binormal"],
            tract_id_coarse=coarse["tract_id"],
            u_step_coarse=coarse["u_step"],
            x_true=x_true,
            x_true_cl_dist=x_true_cl_dist,
            x_true_normal=x_true_normal,
            cl_dense=torch.cat([cl_pack["cl_dense"], cl_pack["cl_dense_u"].unsqueeze(-1)], dim=-1),
            cl_tract_id=cl_pack["cl_tract_id"],
            branch_nl_fine=fine["branch_nl"],
            branch_nl_mid=mid["branch_nl"],
            branch_nl_coarse=coarse["branch_nl"],
            n_radial_fine=fine["n_radial"],
            n_radial_mid=mid["n_radial"],
            n_radial_coarse=coarse["n_radial"],
            origin_shift=_torch_f32(origin),
            pose_R=_torch_f32(R),
            r_star=r_fine["r_star"],
            r_star_valid=r_fine["valid"],
            r_star_ambiguous=r_fine["ambiguous"],
            r_dth=r_fine["dth"],
            r_du=r_fine["du"],
            r_ring_med=r_fine["ring_med"],
            r_star_mid=r_mid["r_star"],
            r_star_valid_mid=r_mid["valid"],
            r_star_ambiguous_mid=r_mid["ambiguous"],
            cache_version=torch.tensor(CACHE_VERSION, dtype=torch.long),
            **tokens,
        )
        if extra:
            for key, value in extra.items():
                data[key] = value
        _ensure_gt_follow_indices(data)
        data.has_true_normal = torch.tensor(
            1
            if (
                x_true_normal is not None
                and x_true_normal.size(0) == x_true.size(0)
                and float(x_true_normal.float().norm(dim=-1).mean()) > 0.5
            )
            else 0,
            dtype=torch.uint8,
        )
        return data

    def _attach_sdf_and_level_geom(
        self, extra, fine, mid, coarse, dense_tracts, x_true, ostium_frames_posed=None, gt_pts=None
    ):
        extra.update(self._level_boundary_and_geom(fine, dense_tracts, "", ostium_frames_posed))
        extra.update(self._level_boundary_and_geom(mid, dense_tracts, "_mid", ostium_frames_posed))
        extra.update(self._level_boundary_and_geom(coarse, dense_tracts, "_coarse", ostium_frames_posed))
        tpl_pos = fine["pos"].numpy() if torch.is_tensor(fine["pos"]) else np.asarray(fine["pos"])
        tpl_nrm = fine["normal"].numpy() if torch.is_tensor(fine["normal"]) else np.asarray(fine["normal"])
        if gt_pts is not None:
            gt_pts = np.asarray(gt_pts, dtype=np.float64).reshape(-1, 3)
            if gt_pts.shape[0] > 0:
                sdf = signed_distance_to_oriented_surface(gt_pts, tpl_pos, tpl_nrm)
                extra["gt_template_sdf"] = _torch_f32(sdf)
        if x_true is not None and torch.is_tensor(x_true) and x_true.numel() > 0:
            extra["x_true_template_sdf"] = _torch_f32(
                signed_distance_to_oriented_surface(x_true.detach().cpu().numpy(), tpl_pos, tpl_nrm)
            )
        return extra

    def _build_template_scaffold(
        self,
        centerline_mesh,
        template_mesh,
        vessel_points=None,
        vessel_mesh=None,
        require_groupids=False,
        ostium_frames=None,
    ):
        tracts, junc_inc, junc_xyz, origin, R, _ = self._prepare_tracts(
            centerline_mesh, require_groupids=require_groupids
        )
        dense_tracts = [self._fit_dense_tract(t) for t in tracts]
        arc_lengths = [float(d["arc"]) if d["arc"] > 1e-12 else _arc_len(t) for d, t in zip(dense_tracts, tracts)]

        posed_tpl = transform_vessel_mesh(template_mesh, origin, R)
        if posed_tpl is None:
            raise ValueError("template_mesh has no triangulated surface")
        stretch = _point_data_array(posed_tpl, "StretchDistance")

        fine = self._level_from_surface(posed_tpl, dense_tracts)
        _, fine_faces = self._polydata_triangles(posed_tpl)
        self._assert_surface_topology(fine_faces, int(fine["pos"].size(0)), name="template fine")
        levels = self._template_levels(posed_tpl, fine_faces, fine)
        mid_mesh, coarse_mesh = levels["mid_mesh"], levels["coarse_mesh"]
        mid = self._level_from_surface(mid_mesh, dense_tracts)
        coarse = self._level_from_surface(coarse_mesh, dense_tracts)
        self._assert_surface_topology(levels["faces_mid"], int(mid["pos"].size(0)), name="template mid")
        self._assert_surface_topology(levels["faces_coarse"], int(coarse["pos"].size(0)), name="template coarse")

        idx_mid, w_mid = levels["upsample_idx_mid"], levels["upsample_w_mid"]
        idx_fine, w_fine = levels["upsample_idx_fine"], levels["upsample_w_fine"]
        extra = {
            "upsample_idx_mid": _torch_long(idx_mid),
            "upsample_w_mid": _torch_f32(w_mid),
            "upsample_idx_fine": _torch_long(idx_fine),
            "upsample_w_fine": _torch_f32(w_fine),
            "r_local": fine["r_local"],
            "r_local_mid": mid["r_local"],
            "r_local_coarse": coarse["r_local"],
        }

        tokens, cl_pack, gt_mesh, x_true, x_true_cl_dist, x_true_normal, gt_extra = self._gt_and_tokens(
            dense_tracts, junc_inc, junc_xyz, arc_lengths, origin, R, vessel_points, vessel_mesh
        )
        extra.update(gt_extra)
        frames = ostium_frames if ostium_frames is not None else self._ostium_frames_from_mesh(template_mesh)
        posed_frames = self._pose_ostium_frames(frames, origin, R)
        gt_pts = None
        if "gt_points" in extra:
            gt_pts = extra["gt_points"].numpy()
        extra = self._attach_sdf_and_level_geom(
            extra, fine, mid, coarse, dense_tracts, x_true,
            ostium_frames_posed=posed_frames, gt_pts=gt_pts,
        )
        stretch_fine = stretch
        tpl_pts = _as_f64(posed_tpl.points)
        if stretch is not None and stretch.shape[0] == tpl_pts.shape[0] and stretch.shape[0] != int(fine["pos"].size(0)):
            stretch_fine = self._resample_point_scalar(tpl_pts, stretch, fine["pos"].numpy())
        stretch_mid = None
        if stretch_fine is not None:
            stretch_mid = self._resample_point_scalar(
                fine["pos"].numpy(), stretch_fine, mid["pos"].numpy()
            )
        r_fine = self._template_r_star(fine, gt_mesh=gt_mesh, stretch=stretch_fine)
        r_mid = self._template_r_star(mid, gt_mesh=gt_mesh, stretch=stretch_mid)
        extra["r_dth_mid"] = r_mid["dth"]
        extra["r_du_mid"] = r_mid["du"]
        extra["r_ring_med_mid"] = r_mid["ring_med"]
        return self._assemble_scaffold_data(
            fine, mid, coarse, tokens, cl_pack, x_true, x_true_cl_dist, x_true_normal,
            r_fine, r_mid, origin, R, extra=extra,
        )

    def build_scaffold(
        self,
        centerline_mesh,
        vessel_points=None,
        vessel_mesh=None,
        template_mesh=None,
        require_groupids=False,
        ostium_frames=None,
    ):
        """Build decoder tensors. With `template_mesh`, identity stays on that surface."""
        if template_mesh is not None:
            return self._build_template_scaffold(
                centerline_mesh,
                template_mesh,
                vessel_points=vessel_points,
                vessel_mesh=vessel_mesh,
                require_groupids=require_groupids,
                ostium_frames=ostium_frames,
            )
        tracts, junc_inc, junc_xyz, origin, R, _ = self._prepare_tracts(
            centerline_mesh, require_groupids=require_groupids
        )
        dense_tracts = [self._fit_dense_tract(t) for t in tracts]
        arc_lengths = [float(d["arc"]) if d["arc"] > 1e-12 else _arc_len(t) for d, t in zip(dense_tracts, tracts)]

        names = ("coarse", "mid", "fine")
        levels = {}
        for name, (n_len, n_rad) in zip(names, self.hierarchy):
            levels[name] = self._generate_level(
                dense_tracts, n_len, n_rad, arc_lengths, junc_inc, junc_xyz
            )

        fine, mid, coarse = levels["fine"], levels["mid"], levels["coarse"]
        tokens, cl_pack, gt_mesh, x_true, x_true_cl_dist, x_true_normal, gt_extra = self._gt_and_tokens(
            dense_tracts,
            junc_inc,
            junc_xyz,
            arc_lengths,
            origin,
            R,
            vessel_points,
            vessel_mesh,
            cl_pack={
                "cl_dense": fine["cl_dense"],
                "cl_dense_u": fine["cl_dense_u"],
                "cl_tract_id": fine["cl_tract_id"],
            },
        )
        try:
            r_fine = self._level_r_star(fine, dense_tracts, gt_mesh)
            r_mid = self._level_r_star(mid, dense_tracts, gt_mesh)
        except Exception:
            r_fine = self._level_r_star(fine, dense_tracts, None)
            r_mid = self._level_r_star(mid, dense_tracts, None)
        extra = dict(gt_extra)
        if "edge_index" in fine:
            r_fine_pack = {
                "r_star": r_fine["r_star"].numpy(),
                "valid": r_fine["valid"].numpy(),
                "dth": r_fine["dth"].numpy(),
                "du": r_fine["du"].numpy(),
                "ring_med": r_fine["ring_med"].numpy(),
            }
            r_fine_pack = self._attach_mesh_r_star_stats(r_fine_pack, fine)
            r_fine["dth"] = _torch_f32(r_fine_pack["dth"])
            r_fine["du"] = _torch_f32(r_fine_pack["du"])
            r_fine["ring_med"] = _torch_f32(r_fine_pack["ring_med"])
        extra = self._attach_sdf_and_level_geom(
            extra, fine, mid, coarse, dense_tracts, x_true,
            ostium_frames_posed=self._pose_ostium_frames(ostium_frames, origin, R),
            gt_pts=extra["gt_points"].numpy() if "gt_points" in extra else None,
        )
        return self._assemble_scaffold_data(
            fine, mid, coarse, tokens, cl_pack, x_true, x_true_cl_dist, x_true_normal,
            r_fine, r_mid, origin, R, extra=extra,
        )

    def build_scaffold_from_centerline(self, centerline_mesh):
        """Stage-2 scaffold from a centerline only (no GT surface; used when Stage 1 supplies the tree)."""
        return self.build_scaffold(centerline_mesh, vessel_points=None)

    def __len__(self):
        return len(self.samples)

    def _build_data(self, sample):
        sample = dict(sample)
        if getattr(self, "ensure_derived", False) and not sample_is_complete(
            sample, require_templates=getattr(self, "require_templates", True)
        ):
            sample = ensure_sample_derived(sample)
        reject_rawdata_paths(sample)
        if not os.path.isfile(sample["vessel_file"]):
            raise FileNotFoundError(
                f"{sample['dataset_id']}: GT mesh missing ({sample['vessel_file']})"
            )
        if not os.path.isfile(sample["centerline_file"]):
            raise FileNotFoundError(
                f"{sample['dataset_id']}: original_centerline missing "
                f"({sample['centerline_file']}). Run centerline_creation.py or "
                "construct AneurysmDataset(ensure_derived=True) under vmtk_env."
            )
        vessel_mesh = pv.read(sample["vessel_file"])
        centerline_mesh = pv.read(sample["centerline_file"])
        tpl_mesh_path = sample.get("template_mesh_file")
        has_template = bool(tpl_mesh_path and os.path.isfile(tpl_mesh_path))
        if getattr(self, "require_templates", False) and not has_template:
            raise FileNotFoundError(
                f"{sample['dataset_id']}: template_mesh missing. "
                "Write it with variable_remeshing.py into cleandata/template_mesh. "
                "Tracts / pose / tokens use original_centerline (§2.5)."
            )
        template_mesh = pv.read(tpl_mesh_path) if has_template else None
        ostium_frames = self._load_ostium_frames_sidecar(sample)
        try:
            return self.build_scaffold(
                centerline_mesh,
                vessel_mesh=vessel_mesh,
                template_mesh=template_mesh,
                require_groupids=True,
                ostium_frames=ostium_frames,
            )
        finally:
            del vessel_mesh, centerline_mesh
            if template_mesh is not None:
                del template_mesh
            gc.collect()

    def _write_cache(self, idx):
        """Build and save one sample, then drop it. Used by cache warmup."""
        sample = self.samples[idx]
        cache_path = self._cache_path(sample["dataset_id"])
        if cache_path and os.path.isfile(cache_path):
            return
        data = self._build_data(sample)
        try:
            if cache_path:
                tmp = f"{cache_path}.{os.getpid()}.tmp"
                try:
                    torch.save(data, tmp)
                    os.replace(tmp, cache_path)
                except OSError:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                    raise
        finally:
            del data
            gc.collect()

    def warmup_cache(self, indices=None, num_workers=1, strict=False):
        """Write missing `.pt` caches. Existing versioned files are left untouched.

        Raycast is one *process* per sample (``OMP_NUM_THREADS=1`` inside the
        pool), not extra Python threads. ``num_workers>1`` runs samples in
        parallel. Each process exits after one sample so VTK/PyTorch heaps
        cannot accumulate. Training DataLoader workers are separate.

        Returns a list of ``(dataset_id, error)`` for samples that failed.
        ``strict=True`` raises if any failed; training should skip them instead.
        """
        from tqdm import tqdm

        idxs = list(range(len(self)) if indices is None else indices)
        missing = [i for i in idxs if not self._cache_file_ready(i)]
        n_hit = len(idxs) - len(missing)
        if not missing:
            print(f"Tube cache already complete ({n_hit} files)")
            return []

        workers = max(1, int(num_workers))
        workers = min(workers, len(missing))
        # compile the collapse core once here, not in every spawned worker
        build_coarsen_core()
        print(f"Tube cache: {n_hit} ready, {len(missing)} to build, {workers} process(es)")
        errors = []
        if workers == 1 or not getattr(self, "_init_kwargs", None):
            for i in tqdm(missing, desc="Tube cache"):
                try:
                    self._write_cache(i)
                except Exception as exc:
                    errors.append(
                        (self.samples[i]["dataset_id"], f"{type(exc).__name__}: {exc}")
                    )
            return self._warmup_result(len(idxs), missing, errors, strict=strict)

        import multiprocessing

        ctx = multiprocessing.get_context("spawn")
        prev_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        try:
            with ctx.Pool(
                processes=workers,
                initializer=_warmup_pool_init,
                initargs=(self._init_kwargs,),
                maxtasksperchild=1,
            ) as pool:
                for idx, err in tqdm(
                    pool.imap_unordered(_warmup_pool_build, missing, chunksize=1),
                    total=len(missing),
                    desc="Tube cache",
                ):
                    if err:
                        errors.append((self.samples[idx]["dataset_id"], err))
        finally:
            if prev_cvd is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = prev_cvd
        return self._warmup_result(len(idxs), missing, errors, strict=strict)

    def preload_ram(self, indices=None, max_items=None):
        """Hold finalized cache graphs in the parent process.

        On Linux, DataLoader workers started with ``fork`` share this via
        copy-on-write. After CUDA is initialized Komondor must use ``spawn``,
        so HPC training still reads ``.pt`` files from scratch NVMe; this
        preload then only helps the rank-0 sanity print and single-process
        debug runs. Clone on ``__getitem__`` so augmentations cannot mutate
        the stored graph.
        """
        from tqdm import tqdm

        idxs = list(range(len(self)) if indices is None else indices)
        if max_items is not None:
            idxs = idxs[: int(max_items)]
        self._ram_cache = {}
        for i in tqdm(idxs, desc="RAM preload"):
            if not self._cache_file_ready(i):
                continue
            sample = self.samples[i]
            path = self._cache_path(sample["dataset_id"])
            try:
                data = _finalize_item(_load_cached_graph(path))
            except Exception:
                continue
            self._ram_cache[i] = data
        return len(self._ram_cache)

    def _warmup_result(self, n_requested, missing, errors, strict=False):
        if errors:
            detail = "; ".join(f"{did}: {err}" for did, err in errors[:8])
            msg = (
                f"Tube cache failed for {len(errors)}/{len(missing)} samples ({detail})"
            )
            print(msg)
            if strict:
                raise RuntimeError(msg)
        n_ok = int(n_requested) - len(errors)
        if n_ok <= 0:
            raise RuntimeError(
                f"Tube cache produced no usable samples "
                f"({len(errors)} failed of {len(missing)} missing)"
            )
        return list(errors)

    def _cache_file_ready(self, idx):
        sample = self.samples[idx]
        path = self._cache_path(sample["dataset_id"])
        return bool(path and os.path.isfile(path))

    def __getitem__(self, idx):
        sample = self.samples[idx]
        cache_path = self._cache_path(sample["dataset_id"])

        ram = getattr(self, "_ram_cache", None)
        if ram is not None and idx in ram:
            return ram[idx].clone()

        if cache_path and os.path.exists(cache_path):
            try:
                data = _load_cached_graph(cache_path)
                if int(getattr(data, "cache_version", torch.tensor(-1))) == CACHE_VERSION:
                    return _finalize_item(data)
            except Exception:
                pass

        try:
            data = self._build_data(sample)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to build sample {sample['dataset_id']} "
                f"({sample['centerline_file']})"
            ) from exc

        if cache_path:
            tmp = f"{cache_path}.{os.getpid()}.tmp"
            try:
                torch.save(data, tmp)
                os.replace(tmp, cache_path)
            except OSError:
                if os.path.exists(tmp):
                    os.remove(tmp)
            gc.collect()

        return _finalize_item(data)


_WARMUP_DS = None


def _warmup_pool_init(init_kwargs):
    global _WARMUP_DS
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    kw = dict(init_kwargs)
    kw["quiet"] = True
    _WARMUP_DS = AneurysmDataset(**kw)


def _warmup_pool_build(idx):
    try:
        _WARMUP_DS._write_cache(idx)
        gc.collect()
        return idx, None
    except Exception as exc:
        return idx, f"{type(exc).__name__}: {exc}"
