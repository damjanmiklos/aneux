import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.interpolate import splprep, splev
import pyvista as pv

class AneurysmDataset(Dataset):
    def __init__(self, csv_path, vtp_vessel_dir, vtp_centerline_dir, 
                 tube_radius=2.0, n_length=100, n_radial=50,
                 extra_centerline_dir=None):
        super().__init__()
        self.tube_radius = tube_radius
        self.n_length = n_length
        self.n_radial = n_radial
        
        # Load and filter CSV
        df = pd.read_csv(csv_path)
        locations = ['ICA pcom', 'ICA oph', 'ICA cav', 'ICA bif']
        df_filtered = df[df['location'].isin(locations)]
        
        self.samples = []r
        
        # Cross reference existing files
        for _, row in df_filtered.iterrows():
            dataset_id = row['dataset']
            vessel_file = os.path.join(vtp_vessel_dir, f"{dataset_id}.vtp")
            centerline_file = os.path.join(vtp_centerline_dir, f"{dataset_id}.vtp")
            
            # Check primary centerline dir first, then fallback to extra dir
            if not os.path.exists(centerline_file) and extra_centerline_dir:
                centerline_file = os.path.join(extra_centerline_dir, f"{dataset_id}.vtp")
            
            if os.path.exists(vessel_file) and os.path.exists(centerline_file):
                self.samples.append({
                    'dataset_id': dataset_id,
                    'vessel_file': vessel_file,
                    'centerline_file': centerline_file
                })

    # -------------------------------------------------------------------------
    # Branch extraction
    # -------------------------------------------------------------------------
    def _extract_branches(self, centerline_mesh):
        """
        Extract ordered point coordinates for each branch (cell) in the centerline.
        Returns a list of numpy arrays, each [n_branch_pts, 3].
        """
        branches = []
        for i in range(centerline_mesh.n_cells):
            cell = centerline_mesh.GetCell(i)
            n_pts = cell.GetNumberOfPoints()
            if n_pts < 6:
                continue  # Need at least k+1=6 points for quintic spline
            point_ids = [cell.GetPointId(j) for j in range(n_pts)]
            pts = centerline_mesh.points[point_ids]
            
            # Remove consecutive duplicate points
            diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            mask = np.concatenate(([True], diffs > 1e-6))
            pts = pts[mask]
            
            if len(pts) >= 6:
                branches.append(pts)
        
        # Fallback: if no valid branches, use all points as single branch
        if not branches:
            pts = centerline_mesh.points
            diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            mask = np.concatenate(([True], diffs > 1e-6))
            branches.append(pts[mask])
        
        return branches

    # -------------------------------------------------------------------------
    # Parallel transport frames
    # -------------------------------------------------------------------------
    def _compute_parallel_transport_frames(self, derivatives):
        """
        Compute parallel transport frame (Bishop frame) along the curve
        to avoid twisting that happens with Frenet-Serret frames.
        """
        tangents = derivatives / np.linalg.norm(derivatives, axis=1, keepdims=True)
        N = len(tangents)
        
        normals = np.zeros_like(tangents)
        binormals = np.zeros_like(tangents)
        
        # Initial frame
        t0 = tangents[0]
        # Choose an arbitrary vector not parallel to t0 to compute initial normal
        v = np.array([1.0, 0.0, 0.0])
        if np.abs(np.dot(t0, v)) > 0.99:
            v = np.array([0.0, 1.0, 0.0])
        
        n0 = np.cross(t0, v)
        n0 = n0 / np.linalg.norm(n0)
        b0 = np.cross(t0, n0)
        
        normals[0] = n0
        binormals[0] = b0
        
        # Propagate frame using parallel transport
        for i in range(1, N):
            t_prev = tangents[i-1]
            t_curr = tangents[i]
            
            # Rotation axis and angle between t_prev and t_curr
            axis = np.cross(t_prev, t_curr)
            sin_angle = np.linalg.norm(axis)
            cos_angle = np.dot(t_prev, t_curr)
            
            if sin_angle > 1e-6:
                axis = axis / sin_angle
                # Rodrigues' rotation formula components
                K = np.array([
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]
                ])
                R = np.eye(3) + sin_angle * K + (1 - cos_angle) * (K @ K)
                normals[i] = R @ normals[i-1]
            else:
                normals[i] = normals[i-1]
                
            binormals[i] = np.cross(t_curr, normals[i])
            
        return tangents, normals, binormals

    # -------------------------------------------------------------------------
    # Per-branch tube generation
    # -------------------------------------------------------------------------
    def _generate_branch_tube(self, branch_points, n_length_branch):
        """
        Generate tube coordinates for a single branch.
        Returns: (tube_nodes [n_length_branch * n_radial, 3], 
                  eval_points [n_length_branch, 3])
        """
        # Fit Quintic B-spline (k=5)
        tck, u = splprep([branch_points[:, 0], branch_points[:, 1], 
                          branch_points[:, 2]], s=0, k=5)
        
        # Evaluate spline at evenly spaced points
        u_new = np.linspace(0, 1, n_length_branch)
        eval_points = np.vstack(splev(u_new, tck)).T
        
        # Evaluate first derivative for tangents
        derivatives = np.vstack(splev(u_new, tck, der=1)).T
        
        # Compute frames
        tangents, normals, binormals = self._compute_parallel_transport_frames(derivatives)
        
        # Generate tube surface nodes
        theta = np.linspace(0, 2 * np.pi, self.n_radial, endpoint=False)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        tube_nodes = np.zeros((n_length_branch * self.n_radial, 3))
        
        for i in range(n_length_branch):
            center = eval_points[i]
            n = normals[i]
            b = binormals[i]
            
            for j in range(self.n_radial):
                idx = i * self.n_radial + j
                point = center + self.tube_radius * (cos_theta[j] * n + sin_theta[j] * b)
                tube_nodes[idx] = point
                
        return tube_nodes, eval_points

    def _generate_branch_topology(self, n_length_branch, node_offset):
        """
        Generate edges and faces for a single branch tube segment.
        
        Args:
            n_length_branch: number of cross-sections in this branch
            node_offset: global node index offset for this branch
            
        Returns: (edges list, faces list) with global indices
        """
        edges = []
        faces = []
        
        for i in range(n_length_branch):
            for j in range(self.n_radial):
                idx = node_offset + i * self.n_radial + j
                
                # Connect along the ring (circumferential)
                next_j = (j + 1) % self.n_radial
                idx_ring_next = node_offset + i * self.n_radial + next_j
                edges.append([idx, idx_ring_next])
                edges.append([idx_ring_next, idx])
                
                # Connect along the tube (longitudinal)
                if i < n_length_branch - 1:
                    idx_long_next = node_offset + (i + 1) * self.n_radial + j
                    edges.append([idx, idx_long_next])
                    edges.append([idx_long_next, idx])
                    
                    # Diagonal connections for triangulation
                    idx_diag = node_offset + (i + 1) * self.n_radial + next_j
                    edges.append([idx, idx_diag])
                    edges.append([idx_diag, idx])
                    
                    # Faces for Laplacian smoothing
                    faces.append([idx, idx_ring_next, idx_long_next])
                    faces.append([idx_ring_next, idx_diag, idx_long_next])
        
        return edges, faces

    # -------------------------------------------------------------------------
    # Full branching tube pipeline
    # -------------------------------------------------------------------------
    def _generate_branching_tube(self, branches):
        """
        Generate a complete branching tube scaffold from multiple centerline branches.
        
        Each branch gets its own spline and tube segment. N_LENGTH is distributed
        across branches proportionally to their arc length, with a minimum of 10.
        
        Returns: (X_tube tensor, edge_index tensor, faces tensor)
        """
        MIN_LENGTH_PER_BRANCH = 10
        
        # Compute arc lengths for proportional allocation
        arc_lengths = []
        for pts in branches:
            dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            arc_lengths.append(np.sum(dists))
        total_arc = sum(arc_lengths)
        
        # Allocate length samples proportionally, with minimum
        n_branches = len(branches)
        raw_alloc = [max(MIN_LENGTH_PER_BRANCH, 
                        int(round(self.n_length * (al / total_arc))))
                     for al in arc_lengths]
        
        # Adjust to hit target total (allow slight variation)
        # No strict enforcement — total nodes will vary per sample
        
        # Generate per-branch tubes and topology
        all_tube_nodes = []
        all_edges = []
        all_faces = []
        node_offset = 0
        
        for b_idx, (branch_pts, n_len) in enumerate(zip(branches, raw_alloc)):
            # Generate tube geometry for this branch
            tube_nodes, _ = self._generate_branch_tube(branch_pts, n_len)
            all_tube_nodes.append(tube_nodes)
            
            # Generate topology for this branch
            edges, faces = self._generate_branch_topology(n_len, node_offset)
            all_edges.extend(edges)
            all_faces.extend(faces)
            
            node_offset += n_len * self.n_radial
        
        # Concatenate all branch tubes
        X_tube = torch.tensor(np.concatenate(all_tube_nodes, axis=0), dtype=torch.float32)
        
        # Build edge_index
        if all_edges:
            edge_index = torch.tensor(all_edges, dtype=torch.long).t()
            edge_index = torch.unique(edge_index, dim=1)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        # Build faces
        if all_faces:
            faces = torch.tensor(all_faces, dtype=torch.long)
        else:
            faces = torch.zeros((0, 3), dtype=torch.long)
        
        return X_tube, edge_index, faces

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load vessel mesh point cloud (X_true)
        vessel_mesh = pv.read(sample['vessel_file'])
        X_true = torch.tensor(vessel_mesh.points, dtype=torch.float32)
        
        # Load centerline and extract branches
        centerline_mesh = pv.read(sample['centerline_file'])
        branches = self._extract_branches(centerline_mesh)
        
        # Generate branching tube scaffold with per-sample topology
        X_tube, edge_index, faces = self._generate_branching_tube(branches)
        
        from torch_geometric.data import Data
        
        # Each sample now has its own edge_index and faces since tube topology
        # varies with the number of branches per patient.
        # PyG's DataLoader will automatically batch and increment edge indices.
        data = Data(
            x=X_tube,
            edge_index=edge_index,
            x_true=X_true,
            faces=faces
        )
        return data
