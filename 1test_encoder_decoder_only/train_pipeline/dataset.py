import os
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import pyvista as pv
import torch
from scipy.interpolate import splprep, splev
from torch_geometric.data import Data
from torch.utils.data import Dataset

from config import (
    CACHE_VERSION,
    DENSE_CL_SPACING_MM,
    FAR_CL_MARGIN_MM,
    HIERARCHY_LEVELS,
    LATENT_LEN,
    MAX_TRACTS,
    MIN_RINGS_PER_BRANCH,
    MIN_TOKENS_PER_TRACT,
    N_TRUE,
    N_TRUE_FAR_FRAC,
    TUBE_RADIUS_MM,
)
from geometry import fps_metric, point_to_polyline_dist


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


class AneurysmData(Data):
    """PyG Data with correct index offsets for the mid/coarse scaffold graphs."""

    def __inc__(self, key, value, *args, **kwargs):
        if key in ("edge_index", "face"):
            return int(self.x.size(0))
        if key in ("edge_index_mid", "face_mid"):
            return int(self.pos_mid.size(0))
        if key in ("edge_index_coarse", "face_coarse"):
            return int(self.pos_coarse.size(0))
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
        csv_path,
        vtp_vessel_dir,
        vtp_centerline_dir,
        tube_radius=TUBE_RADIUS_MM,
        n_length=None,
        n_radial=None,
        extra_centerline_dir=None,
        cache_dir=None,
        n_true=N_TRUE,
        hierarchy=HIERARCHY_LEVELS,
        latent_len=LATENT_LEN,
    ):
        super().__init__()
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

        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tube_cache")
        self.cache_dir = os.path.abspath(cache_dir) if cache_dir else None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        df = pd.read_csv(csv_path)
        locations = ["ICA pcom", "ICA oph", "ICA cav", "ICA bif"]
        df_filtered = df[df["location"].isin(locations)]

        self.samples = []
        n_missing = 0
        for _, row in df_filtered.iterrows():
            dataset_id = row["dataset"]
            vessel_file = os.path.join(vtp_vessel_dir, f"{dataset_id}.vtp")
            centerline_file = os.path.join(vtp_centerline_dir, f"{dataset_id}.vtp")

            if not os.path.exists(centerline_file) and extra_centerline_dir:
                centerline_file = os.path.join(extra_centerline_dir, f"{dataset_id}.vtp")

            if os.path.exists(vessel_file) and os.path.exists(centerline_file):
                self.samples.append({
                    "dataset_id": dataset_id,
                    "location": row["location"],
                    "vessel_file": vessel_file,
                    "centerline_file": centerline_file,
                })
            else:
                n_missing += 1

        print(
            f"AneurysmDataset: {len(df)} CSV rows, {len(df_filtered)} ICA-filtered, "
            f"{len(self.samples)} with vessel+centerline, {n_missing} skipped (missing files)"
        )

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

    def _generate_level(self, dense_tracts, n_length, n_radial, arc_lengths):
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
        tracts, endpoints, _ = extract_unique_tracts(centerline_mesh)
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

    def build_scaffold(self, centerline_mesh, vessel_points=None):
        """Build tube / latent tensors from a centerline. `vessel_points` is optional (Stage-2)."""
        tracts, junc_inc, junc_xyz, origin, R, _ = self._prepare_tracts(centerline_mesh)
        dense_tracts = [self._fit_dense_tract(t) for t in tracts]
        arc_lengths = [float(d["arc"]) if d["arc"] > 1e-12 else _arc_len(t) for d, t in zip(dense_tracts, tracts)]

        names = ("coarse", "mid", "fine")
        levels = {}
        for name, (n_len, n_rad) in zip(names, self.hierarchy):
            levels[name] = self._generate_level(dense_tracts, n_len, n_rad, arc_lengths)

        fine, mid, coarse = levels["fine"], levels["mid"], levels["coarse"]
        tokens = self._build_latent_tokens(dense_tracts, junc_inc, junc_xyz, arc_lengths)
        cl_xyz = fine["cl_dense"].numpy()
        cl_dense = torch.cat([fine["cl_dense"], fine["cl_dense_u"].unsqueeze(-1)], dim=-1)

        if vessel_points is not None:
            vessel = (_as_f64(vessel_points) - origin) @ R
            x_true, x_true_cl_dist = self._hybrid_true_points(vessel, cl_xyz)
        else:
            x_true = torch.zeros((1, 3), dtype=torch.float32)
            x_true_cl_dist = torch.zeros((1,), dtype=torch.float32)

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
            cl_dense=cl_dense,
            cl_tract_id=fine["cl_tract_id"],
            branch_nl_fine=fine["branch_nl"],
            branch_nl_mid=mid["branch_nl"],
            branch_nl_coarse=coarse["branch_nl"],
            n_radial_fine=fine["n_radial"],
            n_radial_mid=mid["n_radial"],
            n_radial_coarse=coarse["n_radial"],
            origin_shift=_torch_f32(origin),
            pose_R=_torch_f32(R),
            cache_version=torch.tensor(CACHE_VERSION, dtype=torch.long),
            **tokens,
        )
        return data

    def build_scaffold_from_centerline(self, centerline_mesh):
        """Stage-2 contract: tube + tree tokens, no GT surface."""
        return self.build_scaffold(centerline_mesh, vessel_points=None)

    def __len__(self):
        return len(self.samples)

    def _build_data(self, sample):
        vessel_mesh = pv.read(sample["vessel_file"])
        centerline_mesh = pv.read(sample["centerline_file"])
        return self.build_scaffold(centerline_mesh, vessel_points=vessel_mesh.points)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        cache_path = self._cache_path(sample["dataset_id"])

        if cache_path and os.path.exists(cache_path):
            try:
                try:
                    data = torch.load(cache_path, map_location="cpu", weights_only=False)
                except TypeError:
                    data = torch.load(cache_path, map_location="cpu")
                if int(getattr(data, "cache_version", torch.tensor(-1))) == CACHE_VERSION:
                    if getattr(data, "face", None) is None and getattr(data, "faces", None) is not None:
                        faces = data.faces
                        data.face = faces.t().contiguous() if faces.size(-1) == 3 else faces
                    if not isinstance(data, AneurysmData):
                        data = _as_aneurysm_data(data)
                    return _ensure_fp32_data(data)
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

        return data
