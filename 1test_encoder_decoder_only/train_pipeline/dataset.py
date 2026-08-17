import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from scipy.interpolate import splprep, splev
import pyvista as pv


def _dedup_polyline(pts):
    if len(pts) < 2:
        return pts
    diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    mask = np.concatenate(([True], diffs > 1e-6))
    return pts[mask]


class AneurysmDataset(Dataset):
    def __init__(self, csv_path, vtp_vessel_dir, vtp_centerline_dir,
                 tube_radius=2.0, n_length=100, n_radial=50,
                 extra_centerline_dir=None, cache_dir=None):
        super().__init__()
        self.tube_radius = tube_radius
        self.n_length = n_length
        self.n_radial = n_radial

        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tube_cache")
        self.cache_dir = os.path.abspath(cache_dir) if cache_dir else None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        df = pd.read_csv(csv_path)
        locations = ['ICA pcom', 'ICA oph', 'ICA cav', 'ICA bif']
        df_filtered = df[df['location'].isin(locations)]

        self.samples = []
        n_missing = 0
        for _, row in df_filtered.iterrows():
            dataset_id = row['dataset']
            vessel_file = os.path.join(vtp_vessel_dir, f"{dataset_id}.vtp")
            centerline_file = os.path.join(vtp_centerline_dir, f"{dataset_id}.vtp")

            if not os.path.exists(centerline_file) and extra_centerline_dir:
                centerline_file = os.path.join(extra_centerline_dir, f"{dataset_id}.vtp")

            if os.path.exists(vessel_file) and os.path.exists(centerline_file):
                self.samples.append({
                    'dataset_id': dataset_id,
                    'location': row['location'],
                    'vessel_file': vessel_file,
                    'centerline_file': centerline_file
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
        name = f"{safe_id}_L{self.n_length}_R{self.n_radial}_rad{self.tube_radius}.pt"
        return os.path.join(self.cache_dir, name)

    def _extract_branches(self, centerline_mesh):
        """
        Extract ordered point coordinates for each branch (cell) in the centerline.
        Returns a list of numpy arrays, each [n_branch_pts, 3].
        """
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
        t = tangent / (np.linalg.norm(tangent) + 1e-12)
        n = normal - t * np.dot(normal, t)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-8:
            v = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(t, v)) > 0.99:
                v = np.array([0.0, 1.0, 0.0])
            n = np.cross(t, v)
            n = n / (np.linalg.norm(n) + 1e-12)
        else:
            n = n / n_norm
        b = np.cross(t, n)
        b = b / (np.linalg.norm(b) + 1e-12)
        return t, n, b

    def _compute_parallel_transport_frames(self, derivatives):
        """
        Bishop frame along the curve. Re-orthonormalize every step so drift
        does not accumulate, and replace a vanished normal on 180° reversals.
        """
        t_norm = np.linalg.norm(derivatives, axis=1, keepdims=True)
        t_norm = np.maximum(t_norm, 1e-8)
        tangents = derivatives / t_norm
        N = len(tangents)

        normals = np.zeros_like(tangents)
        binormals = np.zeros_like(tangents)

        t0 = tangents[0]
        v = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(t0, v)) > 0.99:
            v = np.array([0.0, 1.0, 0.0])
        n0 = np.cross(t0, v)
        t0, n0, b0 = self._orthonormalize_frame(t0, n0)
        tangents[0], normals[0], binormals[0] = t0, n0, b0

        for i in range(1, N):
            t_prev = tangents[i - 1]
            t_curr = tangents[i]
            axis = np.cross(t_prev, t_curr)
            sin_angle = np.linalg.norm(axis)
            cos_angle = float(np.clip(np.dot(t_prev, t_curr), -1.0, 1.0))

            if sin_angle > 1e-6:
                axis = axis / sin_angle
                K = np.array([
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]
                ])
                R = np.eye(3) + sin_angle * K + (1 - cos_angle) * (K @ K)
                n_i = R @ normals[i - 1]
            else:
                n_i = normals[i - 1]

            t_i, n_i, b_i = self._orthonormalize_frame(t_curr, n_i)
            tangents[i], normals[i], binormals[i] = t_i, n_i, b_i

        return tangents, normals, binormals

    def _fit_centerline_spline(self, branch_points):
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

    def _generate_branch_tube(self, branch_points, n_length_branch):
        """
        Generate tube coordinates for a single branch.
        Returns: (tube_nodes [n_length_branch * n_radial, 3],
                  eval_points [n_length_branch, 3])
        """
        tck = self._fit_centerline_spline(branch_points)

        u_new = np.linspace(0, 1, n_length_branch)
        eval_points = np.vstack(splev(u_new, tck)).T

        derivatives = np.vstack(splev(u_new, tck, der=1)).T
        dnorm = np.linalg.norm(derivatives, axis=1, keepdims=True)
        if np.any(dnorm < 1e-8):
            fd = np.gradient(eval_points, axis=0)
            derivatives = np.where(dnorm < 1e-8, fd, derivatives)

        _, normals, binormals = self._compute_parallel_transport_frames(derivatives)

        theta = np.linspace(0, 2 * np.pi, self.n_radial, endpoint=False)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        tube_nodes = (
            eval_points[:, None, :]
            + self.tube_radius * (
                cos_theta[None, :, None] * normals[:, None, :]
                + sin_theta[None, :, None] * binormals[:, None, :]
            )
        ).reshape(-1, 3)

        return tube_nodes, eval_points

    def _generate_branch_topology(self, n_length_branch, node_offset):
        """
        Edges and faces for one branch tube. Graph diagonals match the face
        diagonals (B–C of each quad A–B / C–D).
        Returns numpy arrays: edges [E, 2], faces [F, 3].
        """
        nr = self.n_radial
        n = n_length_branch
        i = np.arange(n)
        j = np.arange(nr)
        ii, jj = np.meshgrid(i, j, indexing="ij")
        idx = node_offset + ii * nr + jj
        next_j = (jj + 1) % nr
        idx_b_all = node_offset + ii * nr + next_j

        ring = np.stack([idx.ravel(), idx_b_all.ravel()], axis=1)

        if n > 1:
            idx_a = idx[:-1]
            idx_b = idx_b_all[:-1]
            idx_c = node_offset + (ii[:-1] + 1) * nr + jj[:-1]
            idx_d = node_offset + (ii[:-1] + 1) * nr + next_j[:-1]
            long = np.stack([idx_a.ravel(), idx_c.ravel()], axis=1)
            # Diagonal B–C, the shared edge of triangles ABC and BDC
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

    def _generate_branching_tube(self, branches):
        """
        Branching tube scaffold. Branches stay disjoint open cylinders (inlet/outlet
        topology, not a watertight CFD surface). N_LENGTH is split by arc length
        with a per-branch floor of 10.
        """
        MIN_LENGTH_PER_BRANCH = 10
        n_branches = len(branches)

        arc_lengths = []
        for pts in branches:
            if len(pts) < 2:
                arc_lengths.append(0.0)
            else:
                arc_lengths.append(float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))))
        total_arc = sum(arc_lengths)

        if total_arc <= 1e-12:
            share = max(MIN_LENGTH_PER_BRANCH, max(1, self.n_length // max(n_branches, 1)))
            raw_alloc = [share] * n_branches
        else:
            raw_alloc = [
                max(MIN_LENGTH_PER_BRANCH, int(round(self.n_length * (al / total_arc))))
                for al in arc_lengths
            ]

        all_tube_nodes = []
        all_edges = []
        all_faces = []
        node_offset = 0

        for branch_pts, n_len in zip(branches, raw_alloc):
            tube_nodes, _ = self._generate_branch_tube(branch_pts, n_len)
            all_tube_nodes.append(tube_nodes)
            edges, faces = self._generate_branch_topology(n_len, node_offset)
            all_edges.append(edges)
            all_faces.append(faces)
            node_offset += n_len * self.n_radial

        X_tube = torch.tensor(np.concatenate(all_tube_nodes, axis=0), dtype=torch.float32)

        if all_edges:
            edge_index = torch.from_numpy(np.concatenate(all_edges, axis=0).T.copy()).long()
            edge_index = torch.unique(edge_index, dim=1)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        if all_faces:
            faces = torch.from_numpy(np.concatenate(all_faces, axis=0).copy()).long()
        else:
            faces = torch.zeros((0, 3), dtype=torch.long)

        # PyG increments `face` [3, F] during batching; do not store [F, 3] as `faces`
        face = faces.t().contiguous()
        return X_tube, edge_index, face

    def __len__(self):
        return len(self.samples)

    def _build_data(self, sample):
        vessel_mesh = pv.read(sample['vessel_file'])
        X_true = torch.tensor(np.asarray(vessel_mesh.points), dtype=torch.float32)

        centerline_mesh = pv.read(sample['centerline_file'])
        branches = self._extract_branches(centerline_mesh)
        X_tube, edge_index, face = self._generate_branching_tube(branches)

        return Data(
            x=X_tube,
            edge_index=edge_index,
            x_true=X_true,
            face=face
        )

    def __getitem__(self, idx):
        sample = self.samples[idx]
        cache_path = self._cache_path(sample['dataset_id'])

        if cache_path and os.path.exists(cache_path):
            try:
                data = torch.load(cache_path, map_location="cpu")
                if getattr(data, "face", None) is None and getattr(data, "faces", None) is not None:
                    faces = data.faces
                    data.face = faces.t().contiguous() if faces.size(-1) == 3 else faces
                return data
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
