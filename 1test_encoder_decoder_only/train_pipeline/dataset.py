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
    MIN_TOKENS_PER_TRACT,
    N_TRUE,
    N_TRUE_FAR_FRAC,
    TEMPLATE_COARSE_KEEP,
    TEMPLATE_MID_KEEP,
    TEMPLATE_MIN_COARSE,
    TEMPLATE_MIN_MID,
    TEMPLATE_UPSAMPLE_K,
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
from geometry import fps_metric, knn_upsample_tables, point_to_polyline_dist
from raycast import (
    closest_cell_normals,
    compute_level_r_star,
    empty_r_star,
    nearest_normal_offset_r_star,
    transform_vessel_mesh,
)

# Endpoint merge distance for one-polyline-per-GroupId tracts. VMTK blanked
# bifurcation blobs are dropped, so daughter ends sit ~1 MISR apart.
GROUPID_ENDPOINT_SNAP_MM = 1.0


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


def _finalize_item(data):
    if getattr(data, "face", None) is None and getattr(data, "faces", None) is not None:
        faces = data.faces
        data.face = faces.t().contiguous() if faces.size(-1) == 3 else faces
    if not isinstance(data, AneurysmData):
        data = _as_aneurysm_data(data)
    data = _ensure_fp32_data(data)
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
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in (
            "edge_index",
            "edge_index_mid",
            "edge_index_coarse",
            "face",
            "face_mid",
            "face_coarse",
        ):
            return -1
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
            out.has_true_normal = bool(flag)
        return out


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


def allocate_token_counts(n_tokens, arc_lengths, n_junctions, min_per=MIN_TOKENS_PER_TRACT):
    """Split a fixed token budget into per-tract tokens plus junction tokens."""
    n_tracts = len(arc_lengths)
    if n_tracts == 0:
        return [], 0
    min_per = max(1, int(min_per))
    n_junc = min(int(n_junctions), max(0, int(n_tokens) - n_tracts * min_per))
    budget = int(n_tokens) - n_junc
    alloc = allocate_ring_counts(budget, arc_lengths)
    if min_per != MIN_RINGS_PER_BRANCH:
        alloc = [max(min_per, a) for a in alloc]
        alloc = _rebalance_counts(alloc, budget, min_per)
    return alloc, n_junc


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


def extract_groupid_tracts(centerline_mesh, snap=1e-4, endpoint_snap_mm=GROUPID_ENDPOINT_SNAP_MM):
    """One representative polyline per VMTK GroupId from centerline_creation.py.

    `vmtkBranchExtractor` tags every point with GroupIds / Blanking. Parent
    segments still appear once per source→target path, but they share a GroupId.
    Blanking==1 groups are bifurcation blobs and are dropped. Endpoints of the
    remaining groups are snapped so `_orient_tracts` can rebuild the tree.

    Meshes without GroupIds (synthetic tests) fall back to `extract_unique_tracts`.
    """
    group_pt = _point_data_array(centerline_mesh, "GroupIds")
    group_cell = _cell_data_array(centerline_mesh, "GroupIds")
    if group_pt is None and group_cell is None:
        return extract_unique_tracts(centerline_mesh, snap=snap)

    blank_pt = _point_data_array(centerline_mesh, "Blanking")
    blank_cell = _cell_data_array(centerline_mesh, "Blanking")
    points = _as_f64(centerline_mesh.points)
    by_group = defaultdict(list)

    polylines = list(_iter_polyline_point_ids(centerline_mesh))
    if not polylines:
        compact = list(range(len(points)))
        polylines = [(0, compact)] if len(compact) >= 2 else []

    for ci, ids in polylines:
        runs = []
        if group_pt is not None:
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
        elif group_cell is not None and ci < len(group_cell):
            gid = int(round(float(group_cell[ci])))
            blanked = False
            if blank_cell is not None and ci < len(blank_cell):
                blanked = float(blank_cell[ci]) > 0.5
            runs.append((gid, blanked, ids))
        else:
            continue
        for gid, blanked, run_ids in runs:
            if blanked or len(run_ids) < 2:
                continue
            pts = _dedup_polyline(points[np.asarray(run_ids, dtype=np.int64)])
            if len(pts) < 2:
                continue
            by_group[gid].append(pts)

    if not by_group:
        return extract_unique_tracts(centerline_mesh, snap=snap)

    tracts = []
    for gid in sorted(by_group):
        pts = _longest_unique_polyline(by_group[gid])
        if pts is not None and len(pts) >= 2:
            tracts.append(pts)
    if not tracts:
        return extract_unique_tracts(centerline_mesh, snap=snap)

    endpoints, junctions = _snap_tract_endpoints(tracts, snap_mm=endpoint_snap_mm)
    return tracts, endpoints, junctions


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
        self.latent_len = int(latent_len)
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
        if not quiet:
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
        return {
            "xyz": xyz,
            "t": tangents,
            "n": normals,
            "b": binormals,
            "u": u_arc,
            "arc": arc,
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
        vessel_pts = np.asarray(vessel_pts, dtype=np.float32)
        cl_xyz = np.asarray(cl_xyz, dtype=np.float64)
        d_all = point_to_polyline_dist(vessel_pts, cl_xyz).astype(np.float32)
        n_far = int(round(self.n_true * N_TRUE_FAR_FRAC))
        n_uni = max(1, self.n_true - n_far)
        uni = fps_metric(vessel_pts, n_uni)
        d_uni = point_to_polyline_dist(uni, cl_xyz).astype(np.float32)

        far_mask = d_all > (self.tube_radius + FAR_CL_MARGIN_MM)
        far_pts = vessel_pts[far_mask]
        if far_pts.shape[0] == 0 or n_far <= 0:
            extra = fps_metric(vessel_pts, max(n_far, 1))[: max(n_far, 0)]
            d_extra = point_to_polyline_dist(extra, cl_xyz).astype(np.float32) if len(extra) else np.zeros((0,), dtype=np.float32)
        else:
            extra = fps_metric(far_pts, min(n_far, far_pts.shape[0]))
            d_extra = point_to_polyline_dist(extra, cl_xyz).astype(np.float32)
            if extra.shape[0] < n_far:
                pad = fps_metric(vessel_pts, n_far - extra.shape[0])
                extra = np.concatenate([extra, pad], axis=0)
                d_extra = np.concatenate(
                    [d_extra, point_to_polyline_dist(pad, cl_xyz).astype(np.float32)], axis=0
                )

        x_true = np.concatenate([uni, extra], axis=0)[: self.n_true]
        d_true = np.concatenate([d_uni, d_extra], axis=0)[: self.n_true]
        if x_true.shape[0] < self.n_true:
            reps = int(np.ceil(self.n_true / max(x_true.shape[0], 1)))
            x_true = np.tile(x_true, (reps, 1))[: self.n_true]
            d_true = np.tile(d_true, reps)[: self.n_true]
        return _torch_f32(x_true), _torch_f32(d_true)

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
        n_junc_avail = len(junction_incidents)
        alloc, n_junc = allocate_token_counts(self.latent_len, arc_lengths, n_junc_avail)
        n_tracts = min(len(dense_tracts), MAX_TRACTS)
        alloc = alloc[:n_tracts]
        dense_tracts = dense_tracts[:n_tracts]
        arc_lengths = arc_lengths[:n_tracts]

        token_u = []
        token_tract = []
        token_is_junc = []
        token_pos = []
        attend = np.zeros((self.latent_len, MAX_TRACTS), dtype=np.bool_)

        slot = 0
        for tid, (dense, n_tok) in enumerate(zip(dense_tracts, alloc)):
            n_tok = max(1, int(n_tok))
            u_q = np.linspace(0.0, 1.0, n_tok, dtype=np.float64)
            xyz = self._interp_by_u(dense["u"], dense["xyz"], u_q)
            for k in range(n_tok):
                if slot >= self.latent_len:
                    break
                token_u.append(u_q[k])
                token_tract.append(tid)
                token_is_junc.append(0)
                token_pos.append(xyz[k])
                attend[slot, tid] = True
                slot += 1

        junc_items = list(junction_incidents.items())[:n_junc]
        for nid, incident in junc_items:
            if slot >= self.latent_len:
                break
            xyz = np.asarray(junc_xyz[nid], dtype=np.float64).reshape(3)
            token_u.append(1.0)
            token_tract.append(-1)
            token_is_junc.append(1)
            token_pos.append(xyz)
            for tid in incident:
                if 0 <= tid < MAX_TRACTS:
                    attend[slot, tid] = True
            slot += 1

        while slot < self.latent_len:
            token_u.append(token_u[-1] if token_u else 0.0)
            token_tract.append(token_tract[-1] if token_tract else 0)
            token_is_junc.append(0)
            token_pos.append(token_pos[-1] if token_pos else np.zeros(3))
            if token_tract[-1] >= 0:
                attend[slot, min(int(token_tract[-1]), MAX_TRACTS - 1)] = True
            slot += 1

        return {
            "latent_u": _torch_f32(np.asarray(token_u[: self.latent_len])),
            "latent_tract_id": _torch_long(np.asarray(token_tract[: self.latent_len])),
            "latent_is_junction": _torch_long(np.asarray(token_is_junc[: self.latent_len])),
            "latent_pos": _torch_f32(np.stack(token_pos[: self.latent_len], axis=0)),
            "token_attend": torch.from_numpy(attend.copy()).bool(),
            "n_tracts": torch.tensor(n_tracts, dtype=torch.long),
        }

    def _prepare_tracts(self, centerline_mesh):
        tracts, endpoints, _ = extract_groupid_tracts(centerline_mesh)
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

    def _template_r_star(self, level, gt_pts):
        packed = nearest_normal_offset_r_star(
            level["pos"].numpy(),
            level["normal"].numpy(),
            gt_pts,
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

    def _decimate_keep(self, mesh, keep_frac, min_points):
        mesh = pv.wrap(mesh)
        n = int(mesh.n_points)
        target_n = max(int(min_points), int(round(n * float(keep_frac))))
        target_n = min(target_n, n)
        if target_n >= n or n < 8:
            return mesh
        reduction = float(np.clip(1.0 - (target_n / max(n, 1)), 0.0, 0.99))
        for fn in ("decimate", "decimate_pro"):
            try:
                out = getattr(mesh, fn)(reduction)
                if out is not None and int(out.n_points) >= 4:
                    return out
            except Exception:
                continue
        try:
            from ops import fps_indices

            keep = fps_indices(_torch_f32(mesh.points), target_n).detach().cpu().numpy()
            sub = mesh.extract_points(keep, adjacent_cells=True)
            if sub is not None and int(sub.n_points) >= 4:
                return sub.triangulate() if sub.n_cells > 0 else sub
        except Exception:
            pass
        return mesh

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

    def _level_from_surface(self, mesh, dense_tracts):
        mesh, faces = self._polydata_triangles(mesh)
        pts = _as_f64(mesh.points)
        n = int(pts.shape[0])
        if n < 4:
            raise ValueError("template surface has fewer than 4 vertices")
        proj = self._project_points_to_tracts(pts, dense_tracts)
        nrm = self._surface_normals(mesh, n)
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
        if vessel_points is not None:
            vessel = (_as_f64(vessel_points) - origin) @ R
            x_true, x_true_cl_dist = self._hybrid_true_points(vessel, cl_xyz)
            nrm = closest_cell_normals(gt_mesh, x_true.numpy()) if gt_mesh is not None else None
            if nrm is not None and float(np.linalg.norm(nrm, axis=1).mean()) > 0.5:
                x_true_normal = _torch_f32(nrm)
            else:
                posed_pts, posed_nrm = self._posed_mesh_normals(vessel_mesh, origin, R)
                if posed_nrm is not None:
                    x_true_normal = self._true_normals_at_points(x_true.numpy(), posed_pts, posed_nrm)
                else:
                    x_true_normal = torch.zeros_like(x_true)
        else:
            x_true = torch.zeros((1, 3), dtype=torch.float32)
            x_true_cl_dist = torch.zeros((1,), dtype=torch.float32)
            x_true_normal = torch.zeros((1, 3), dtype=torch.float32)
        return tokens, cl_pack, gt_mesh, x_true, x_true_cl_dist, x_true_normal

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

    def _build_template_scaffold(self, centerline_mesh, template_mesh, vessel_points=None, vessel_mesh=None):
        tracts, junc_inc, junc_xyz, origin, R, _ = self._prepare_tracts(centerline_mesh)
        dense_tracts = [self._fit_dense_tract(t) for t in tracts]
        arc_lengths = [float(d["arc"]) if d["arc"] > 1e-12 else _arc_len(t) for d, t in zip(dense_tracts, tracts)]

        posed_tpl = transform_vessel_mesh(template_mesh, origin, R)
        if posed_tpl is None:
            raise ValueError("template_mesh has no triangulated surface")
        try:
            posed_tpl = posed_tpl.compute_normals(point_normals=True, cell_normals=False, inplace=False)
        except Exception:
            pass

        fine = self._level_from_surface(posed_tpl, dense_tracts)
        mid_mesh = self._decimate_keep(posed_tpl, TEMPLATE_MID_KEEP, TEMPLATE_MIN_MID)
        coarse_mesh = self._decimate_keep(posed_tpl, TEMPLATE_COARSE_KEEP, TEMPLATE_MIN_COARSE)
        mid = self._level_from_surface(mid_mesh, dense_tracts)
        coarse = self._level_from_surface(coarse_mesh, dense_tracts)

        idx_mid, w_mid = knn_upsample_tables(coarse["pos"].numpy(), mid["pos"].numpy(), TEMPLATE_UPSAMPLE_K)
        idx_fine, w_fine = knn_upsample_tables(mid["pos"].numpy(), fine["pos"].numpy(), TEMPLATE_UPSAMPLE_K)
        extra = {
            "upsample_idx_mid": _torch_long(idx_mid),
            "upsample_w_mid": _torch_f32(w_mid),
            "upsample_idx_fine": _torch_long(idx_fine),
            "upsample_w_fine": _torch_f32(w_fine),
            "r_local": fine["r_local"],
            "r_local_mid": mid["r_local"],
            "r_local_coarse": coarse["r_local"],
        }

        tokens, cl_pack, gt_mesh, x_true, x_true_cl_dist, x_true_normal = self._gt_and_tokens(
            dense_tracts, junc_inc, junc_xyz, arc_lengths, origin, R, vessel_points, vessel_mesh
        )
        gt_pts = None if gt_mesh is None else np.asarray(gt_mesh.points, dtype=np.float64)
        r_fine = self._template_r_star(fine, gt_pts)
        r_mid = self._template_r_star(mid, gt_pts)
        return self._assemble_scaffold_data(
            fine, mid, coarse, tokens, cl_pack, x_true, x_true_cl_dist, x_true_normal,
            r_fine, r_mid, origin, R, extra=extra,
        )

    def build_scaffold(self, centerline_mesh, vessel_points=None, vessel_mesh=None, template_mesh=None):
        """Build decoder tensors. With `template_mesh`, identity stays on that surface."""
        if template_mesh is not None:
            return self._build_template_scaffold(
                centerline_mesh, template_mesh, vessel_points=vessel_points, vessel_mesh=vessel_mesh
            )
        tracts, junc_inc, junc_xyz, origin, R, _ = self._prepare_tracts(centerline_mesh)
        dense_tracts = [self._fit_dense_tract(t) for t in tracts]
        arc_lengths = [float(d["arc"]) if d["arc"] > 1e-12 else _arc_len(t) for d, t in zip(dense_tracts, tracts)]

        names = ("coarse", "mid", "fine")
        levels = {}
        for name, (n_len, n_rad) in zip(names, self.hierarchy):
            levels[name] = self._generate_level(
                dense_tracts, n_len, n_rad, arc_lengths, junc_inc, junc_xyz
            )

        fine, mid, coarse = levels["fine"], levels["mid"], levels["coarse"]
        tokens, cl_pack, gt_mesh, x_true, x_true_cl_dist, x_true_normal = self._gt_and_tokens(
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
        return self._assemble_scaffold_data(
            fine, mid, coarse, tokens, cl_pack, x_true, x_true_cl_dist, x_true_normal,
            r_fine, r_mid, origin, R,
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
        tpl_cl_path = sample.get("template_centerline_file")
        has_template = bool(
            tpl_mesh_path
            and os.path.isfile(tpl_mesh_path)
            and tpl_cl_path
            and os.path.isfile(tpl_cl_path)
        )
        if getattr(self, "require_templates", False) and not has_template:
            raise FileNotFoundError(
                f"{sample['dataset_id']}: template_mesh / template_centerline missing. "
                "Write them with variable_remeshing.py and centerline_creation.py "
                "into cleandata/template_mesh and cleandata/template_centerline."
            )
        template_mesh = None
        template_cl = None
        if has_template:
            template_mesh = pv.read(tpl_mesh_path)
            template_cl = pv.read(tpl_cl_path)
        try:
            return self.build_scaffold(
                template_cl if template_cl is not None else centerline_mesh,
                vessel_mesh=vessel_mesh,
                template_mesh=template_mesh,
            )
        finally:
            del vessel_mesh, centerline_mesh
            if template_mesh is not None:
                del template_mesh
            if template_cl is not None:
                del template_cl
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

    def warmup_cache(self, indices=None, num_workers=1):
        """Write missing `.pt` caches. Existing versioned files are left untouched.

        Raycast is one core per sample, so `num_workers>1` runs samples in
        parallel processes. Each process exits after one sample so VTK/PyTorch
        heaps cannot accumulate. Training DataLoader workers are separate.
        """
        from tqdm import tqdm

        idxs = list(range(len(self)) if indices is None else indices)
        missing = [i for i in idxs if not self._cache_file_ready(i)]
        n_hit = len(idxs) - len(missing)
        if not missing:
            print(f"Tube cache already complete ({n_hit} files)")
            return len(idxs)

        workers = max(1, int(num_workers))
        workers = min(workers, len(missing))
        print(f"Tube cache: {n_hit} ready, {len(missing)} to build, {workers} process(es)")
        if workers == 1 or not getattr(self, "_init_kwargs", None):
            for i in tqdm(missing, desc="Tube cache"):
                self._write_cache(i)
            return len(idxs)

        import multiprocessing

        errors = []
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
        if errors:
            detail = "; ".join(f"{did}: {err}" for did, err in errors[:8])
            raise RuntimeError(
                f"Tube cache failed for {len(errors)}/{len(missing)} samples ({detail})"
            )
        return len(idxs)

    def _cache_file_ready(self, idx):
        sample = self.samples[idx]
        path = self._cache_path(sample["dataset_id"])
        return bool(path and os.path.isfile(path))

    def __getitem__(self, idx):
        sample = self.samples[idx]
        cache_path = self._cache_path(sample["dataset_id"])

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
