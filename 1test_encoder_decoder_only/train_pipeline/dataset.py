import os

import numpy as np
import pandas as pd
import pyvista as pv
import torch
from scipy.interpolate import splprep, splev
from torch_geometric.data import Data
from torch.utils.data import Dataset

from config import (
    CACHE_VERSION,
    HIERARCHY_LEVELS,
    LATENT_LEN,
    MIN_RINGS_PER_BRANCH,
    N_TRUE,
    TUBE_RADIUS_MM,
)
from geometry import fps_metric


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


def _ensure_fp32_data(data):
    """Cast cached floating tensors to float32; leave index tensors as long."""
    for key in data.keys():
        val = data[key]
        if torch.is_tensor(val) and val.is_floating_point() and val.dtype != torch.float32:
            data[key] = val.to(dtype=torch.float32)
    return data


def allocate_ring_counts(n_length, arc_lengths):
    """Split longitudinal rings across branches by arc length."""
    n_branches = len(arc_lengths)
    if n_branches == 0:
        return []
    min_len = MIN_RINGS_PER_BRANCH
    if n_length < n_branches * min_len:
        min_len = max(1, n_length // n_branches)

    total_arc = float(sum(arc_lengths))
    if total_arc <= 1e-12:
        share = max(min_len, max(1, int(n_length) // n_branches))
        return [share] * n_branches

    raw = [
        max(min_len, int(round(n_length * (al / total_arc))))
        for al in arc_lengths
    ]
    return raw


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
        # Legacy single-level args still select the fine scaffold.
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

    def _extract_branches(self, centerline_mesh):
        branches = []
        for i in range(centerline_mesh.n_cells):
            cell = centerline_mesh.GetCell(i)
            n_pts = cell.GetNumberOfPoints()
            if n_pts < 2:
                continue
            point_ids = [cell.GetPointId(j) for j in range(n_pts)]
            pts = _dedup_polyline(centerline_mesh.points[point_ids])
            if len(pts) >= 2:
                branches.append(pts)

        if not branches:
            pts = _dedup_polyline(np.asarray(centerline_mesh.points))
            if len(pts) < 2:
                raise ValueError("Centerline has fewer than 2 unique points")
            branches.append(pts)
        return branches

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

    def _generate_branch_tube(self, branch_points, n_length_branch, n_radial):
        branch_points = _as_f64(branch_points)
        tck = self._fit_centerline_spline(branch_points)
        u_new = np.linspace(0.0, 1.0, n_length_branch, dtype=np.float64)
        eval_points = np.vstack(splev(u_new, tck)).T.astype(np.float64, copy=False)

        derivatives = np.vstack(splev(u_new, tck, der=1)).T.astype(np.float64, copy=False)
        dnorm = np.linalg.norm(derivatives, axis=1, keepdims=True)
        if np.any(dnorm < 1e-8):
            fd = np.gradient(eval_points, axis=0)
            derivatives = np.where(dnorm < 1e-8, fd, derivatives)

        tangents, normals, binormals = self._compute_parallel_transport_frames(derivatives)
        u_local, arc = self._arc_length_parameter(eval_points)

        theta = (2.0 * np.pi * np.arange(n_radial, dtype=np.float64) / n_radial) - np.pi
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # [L, R, 3]
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

        u_grid = np.repeat(u_local[:, None], n_radial, axis=1)
        th_grid = np.repeat(theta[None, :], n_length_branch, axis=0)

        return {
            "nodes": tube_nodes.reshape(-1, 3),
            "u_local": u_grid.reshape(-1),
            "theta": th_grid.reshape(-1),
            "n_v": n_v.reshape(-1, 3),
            "t_v": t_v.reshape(-1, 3),
            "b_v": b_v.reshape(-1, 3),
            "eval_points": eval_points,
            "u_cl": u_local,
            "arc": arc,
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

    def _generate_level(self, branches, n_length, n_radial, arc_lengths):
        alloc = allocate_ring_counts(n_length, arc_lengths)
        node_chunks = []
        u_local_chunks = []
        theta_chunks = []
        n_chunks, t_chunks, b_chunks = [], [], []
        eval_chunks, u_cl_chunks = [], []
        all_edges, all_faces = [], []
        node_offset = 0
        global_u_chunks = []
        total_arc = float(sum(arc_lengths)) or 1.0
        arc_cursor = 0.0

        for branch_pts, n_len, arc_b in zip(branches, alloc, arc_lengths):
            tube = self._generate_branch_tube(branch_pts, n_len, n_radial)
            node_chunks.append(tube["nodes"])
            u_local_chunks.append(tube["u_local"])
            theta_chunks.append(tube["theta"])
            n_chunks.append(tube["n_v"])
            t_chunks.append(tube["t_v"])
            b_chunks.append(tube["b_v"])
            eval_chunks.append(tube["eval_points"])
            u_cl_chunks.append(tube["u_cl"])
            span = (arc_b / total_arc) if total_arc > 0 else 0.0
            global_u_chunks.append(arc_cursor + tube["u_local"] * span)
            arc_cursor += span

            edges, faces = self._generate_branch_topology(n_len, n_radial, node_offset)
            all_edges.append(edges)
            all_faces.append(faces)
            node_offset += n_len * n_radial

        pos = _torch_f32(np.concatenate(node_chunks, axis=0))
        u_local = _torch_f32(np.concatenate(u_local_chunks))
        u_global = _torch_f32(np.concatenate(global_u_chunks))
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
        cl_u = []
        arc_cursor = 0.0
        for u_cl, arc_b in zip(u_cl_chunks, arc_lengths):
            span = (arc_b / total_arc) if total_arc > 0 else 0.0
            cl_u.append(arc_cursor + u_cl * span)
            arc_cursor += span
        cl_u = np.concatenate(cl_u)

        return {
            "pos": pos,
            "u": u_global,
            "u_local": u_local,
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
        }

    def _resample_centerline(self, cl_xyz, cl_u, n_out):
        cl_xyz = np.asarray(cl_xyz, dtype=np.float64)
        cl_u = np.asarray(cl_u, dtype=np.float64)
        order = np.argsort(cl_u)
        cl_xyz, cl_u = cl_xyz[order], cl_u[order]
        # Unique u for interpolation.
        _, uniq = np.unique(np.round(cl_u, 8), return_index=True)
        cl_xyz, cl_u = cl_xyz[uniq], cl_u[uniq]
        u_q = np.linspace(0.0, 1.0, n_out)
        if len(cl_u) == 1:
            xyz = np.repeat(cl_xyz, n_out, axis=0)
        else:
            xyz = np.stack([np.interp(u_q, cl_u, cl_xyz[:, d]) for d in range(3)], axis=1)
        return torch.tensor(xyz, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def _build_data(self, sample):
        vessel_mesh = pv.read(sample["vessel_file"])
        x_true_raw = _as_f64(vessel_mesh.points)

        centerline_mesh = pv.read(sample["centerline_file"])
        branches = [_as_f64(b) for b in self._extract_branches(centerline_mesh)]

        arc_lengths = []
        for pts in branches:
            if len(pts) < 2:
                arc_lengths.append(0.0)
            else:
                arc_lengths.append(float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))))

        cl_all = np.concatenate(branches, axis=0)
        origin = cl_all.mean(axis=0)
        branches = [b - origin for b in branches]
        x_true_raw = x_true_raw - origin

        x_true = _torch_f32(fps_metric(x_true_raw, self.n_true))

        levels = {}
        names = ("coarse", "mid", "fine")
        for name, (n_len, n_rad) in zip(names, self.hierarchy):
            levels[name] = self._generate_level(branches, n_len, n_rad, arc_lengths)

        fine = levels["fine"]
        mid = levels["mid"]
        coarse = levels["coarse"]

        cl_xyz = fine["cl_dense"].numpy()
        cl_u = fine["cl_dense_u"].numpy()
        cl_pos = self._resample_centerline(cl_xyz, cl_u, self.latent_len)
        cl_dense = torch.cat(
            [fine["cl_dense"], fine["cl_dense_u"].unsqueeze(-1)], dim=-1
        )

        data = Data(
            x=fine["pos"],
            edge_index=fine["edge_index"],
            face=fine["face"],
            u=fine["u"],
            u_local=fine["u_local"],
            theta=fine["theta"],
            normal=fine["normal"],
            tangent=fine["tangent"],
            binormal=fine["binormal"],
            pos_mid=mid["pos"],
            edge_index_mid=mid["edge_index"],
            face_mid=mid["face"],
            u_mid=mid["u"],
            theta_mid=mid["theta"],
            normal_mid=mid["normal"],
            tangent_mid=mid["tangent"],
            binormal_mid=mid["binormal"],
            pos_coarse=coarse["pos"],
            edge_index_coarse=coarse["edge_index"],
            face_coarse=coarse["face"],
            u_coarse=coarse["u"],
            theta_coarse=coarse["theta"],
            normal_coarse=coarse["normal"],
            tangent_coarse=coarse["tangent"],
            binormal_coarse=coarse["binormal"],
            x_true=x_true,
            cl_pos=cl_pos,
            cl_dense=cl_dense,
            branch_nl_fine=fine["branch_nl"],
            branch_nl_mid=mid["branch_nl"],
            branch_nl_coarse=coarse["branch_nl"],
            n_radial_fine=fine["n_radial"],
            n_radial_mid=mid["n_radial"],
            n_radial_coarse=coarse["n_radial"],
            origin_shift=_torch_f32(origin),
            cache_version=torch.tensor(CACHE_VERSION, dtype=torch.long),
        )
        return data

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
