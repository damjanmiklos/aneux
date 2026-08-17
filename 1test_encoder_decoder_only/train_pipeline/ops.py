"""Local geometric ops with PyG kernels when available and torch fallbacks otherwise."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MessagePassing


def _num_graphs(batch: Tensor) -> int:
    if batch is None or batch.numel() == 0:
        return 1
    return int(batch.max().item()) + 1


_PYTORCH3D_CUDA_FPS = None


def _no_autocast(x: Tensor):
    return torch.autocast(device_type="cuda" if x.is_cuda else "cpu", enabled=False)


def farthest_point_sample_torch(pts: Tensor, k: int) -> Tensor:
    """Iterative metric FPS on the tensor's device. Returns local indices [k]."""
    n = int(pts.size(0))
    k = min(int(k), n)
    selected = torch.empty(k, dtype=torch.long, device=pts.device)
    selected[0] = 0
    dist = torch.full((n,), float("inf"), device=pts.device, dtype=pts.dtype)
    last = pts[0]
    for i in range(1, k):
        dist = torch.minimum(dist, (pts - last).pow(2).sum(dim=-1))
        farthest = dist.argmax()
        selected[i] = farthest
        last = pts[farthest]
    return selected


def _pytorch3d_fps_ok(device: torch.device) -> bool:
    """True if pytorch3d FPS has CUDA kernels for this device (or the tensor is on CPU)."""
    global _PYTORCH3D_CUDA_FPS
    if device.type != "cuda":
        return True
    if _PYTORCH3D_CUDA_FPS is None:
        try:
            from pytorch3d.ops import sample_farthest_points

            with torch.autocast(device_type="cuda", enabled=False):
                probe = torch.zeros(1, 4, 3, device=device, dtype=torch.float32)
                sample_farthest_points(probe, K=2, random_start_point=False)
                torch.cuda.synchronize()
            _PYTORCH3D_CUDA_FPS = True
        except Exception as exc:
            msg = str(exc).lower()
            if "illegal" in msg or "cuda" in type(exc).__name__.lower():
                raise
            import warnings

            warnings.warn(f"pytorch3d CUDA FPS unavailable ({type(exc).__name__}: {exc})")
            _PYTORCH3D_CUDA_FPS = False
    return bool(_PYTORCH3D_CUDA_FPS)


def fps_indices(pts: Tensor, k: int) -> Tensor:
    """FPS indices for a single cloud [N, 3]. Prefers pytorch3d CUDA, else in-device torch FPS."""
    k = min(int(k), int(pts.size(0)))
    if k <= 0:
        return pts.new_zeros((0,), dtype=torch.long)
    if k == pts.size(0):
        return torch.arange(k, device=pts.device)
    with _no_autocast(pts):
        pts_f = pts.float().contiguous()
        if _pytorch3d_fps_ok(pts.device):
            from pytorch3d.ops import sample_farthest_points

            _, loc = sample_farthest_points(pts_f.unsqueeze(0), K=k, random_start_point=False)
            return loc.squeeze(0).long()
        return farthest_point_sample_torch(pts_f, k)


def _missing_pyg_lib(exc: BaseException) -> bool:
    return isinstance(exc, (ImportError, OSError)) or (
        isinstance(exc, RuntimeError) and "pyg-lib" in str(exc).lower()
    )


def ball_query_packed(
    support: Tensor,
    query: Tensor,
    radius: float,
    support_batch: Tensor,
    query_batch: Tensor,
    max_num_neighbors: int,
) -> Tensor:
    """Neighbors in `support` for each `query` point. Returns [2, E] = (support_idx, query_idx)."""
    try:
        from torch_geometric.nn import radius as _radius

        # pyg-lib radius is (query, support); PointNeXt grouping wants (support, query).
        with _no_autocast(support):
            return _radius(
                support.float().contiguous(),
                query.float().contiguous(),
                radius,
                support_batch,
                query_batch,
                max_num_neighbors=max_num_neighbors,
            ).flip(0)
    except Exception as exc:
        if not _missing_pyg_lib(exc):
            raise
        return _ball_query_torch(
            support, query, radius, support_batch, query_batch, max_num_neighbors
        )


def _ball_query_torch(
    support: Tensor,
    query: Tensor,
    radius: float,
    support_batch: Tensor,
    query_batch: Tensor,
    max_num_neighbors: int,
) -> Tensor:
    srcs, dsts = [], []
    n_graphs = max(_num_graphs(support_batch), _num_graphs(query_batch))
    for g in range(n_graphs):
        s_idx = (support_batch == g).nonzero(as_tuple=False).view(-1)
        q_idx = (query_batch == g).nonzero(as_tuple=False).view(-1)
        if s_idx.numel() == 0 or q_idx.numel() == 0:
            continue
        dist = torch.cdist(query[q_idx], support[s_idx])
        k = min(int(max_num_neighbors), int(s_idx.numel()))
        knn_dist, knn_loc = dist.topk(k, dim=1, largest=False)
        valid = knn_dist <= radius
        q_local, slot = torch.where(valid)
        if q_local.numel() == 0:
            continue
        srcs.append(s_idx[knn_loc[q_local, slot]])
        dsts.append(q_idx[q_local])
    if not srcs:
        return support.new_zeros((2, 0), dtype=torch.long)
    return torch.stack([torch.cat(srcs), torch.cat(dsts)], dim=0)


def radius_graph_packed(
    pos: Tensor,
    radius: float,
    batch: Tensor,
    loop: bool = True,
    max_num_neighbors: int = 32,
    flow: str = "source_to_target",
) -> Tensor:
    try:
        from torch_geometric.nn import radius_graph as _rg

        with _no_autocast(pos):
            return _rg(
                pos.float().contiguous(),
                r=radius,
                batch=batch,
                loop=loop,
                max_num_neighbors=max_num_neighbors,
                flow=flow,
            )
    except Exception as exc:
        if not _missing_pyg_lib(exc):
            raise
        ei = _ball_query_torch(pos, pos, radius, batch, batch, max_num_neighbors)
        # ei[0] = neighbors (support), ei[1] = centers (query)
        if flow != "source_to_target":
            ei = ei.flip(0)
        if loop:
            self = torch.arange(pos.size(0), device=pos.device)
            ei = torch.cat([ei, torch.stack([self, self], dim=0)], dim=1)
            ei = torch.unique(ei, dim=1)
        return ei


def _open_uniform_knots(n_ctrl: int, degree: int, device, dtype) -> Tensor:
    n_internal = n_ctrl - degree - 1
    zeros = torch.zeros(degree + 1, device=device, dtype=dtype)
    ones = torch.ones(degree + 1, device=device, dtype=dtype)
    if n_internal > 0:
        internal = torch.linspace(0, 1, n_internal + 2, device=device, dtype=dtype)[1:-1]
        return torch.cat([zeros, internal, ones], dim=0)
    return torch.cat([zeros, ones], dim=0)


def bspline_basis_1d(u: Tensor, n_ctrl: int, degree: int) -> Tensor:
    """Open B-spline basis of `degree` with `n_ctrl` functions. u in [0, 1] → [E, n_ctrl]."""
    u = u.clamp(0.0, 1.0 - 1e-6)
    knots = _open_uniform_knots(n_ctrl, degree, u.device, u.dtype)
    left = knots[:-1]
    right = knots[1:]
    basis = ((u.unsqueeze(1) >= left) & (u.unsqueeze(1) < right)).to(u.dtype)
    for p in range(1, degree + 1):
        n_fun = basis.size(1) - 1
        k_i = knots[:n_fun]
        k_ip = knots[p : p + n_fun]
        k_i1 = knots[1 : 1 + n_fun]
        k_ip1 = knots[p + 1 : p + 1 + n_fun]
        denom1 = k_ip - k_i
        denom2 = k_ip1 - k_i1
        w1 = torch.where(denom1 > 1e-8, (u.unsqueeze(1) - k_i) / denom1, torch.zeros_like(denom1))
        w2 = torch.where(denom2 > 1e-8, (k_ip1 - u.unsqueeze(1)) / denom2, torch.zeros_like(denom2))
        basis = w1 * basis[:, :-1] + w2 * basis[:, 1:]
    return basis


def tensor_bspline_basis(edge_attr: Tensor, kernel_size: int, degree: int) -> Tensor:
    b0 = bspline_basis_1d(edge_attr[:, 0], kernel_size, degree)
    b1 = bspline_basis_1d(edge_attr[:, 1], kernel_size, degree)
    b2 = bspline_basis_1d(edge_attr[:, 2], kernel_size, degree)
    return torch.einsum("ei,ej,ek->eijk", b0, b1, b2).reshape(edge_attr.size(0), -1)


class BSplineConv(MessagePassing):
    """Pure-PyTorch SplineCNN kernel (Fey et al. 2018) used when pyg-lib is unavailable."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim: int = 3,
        kernel_size: int = 5,
        degree: int = 2,
        aggr: str = "add",
        root_weight: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(aggr=aggr)
        del dim, kwargs
        self.kernel_size = int(kernel_size)
        self.degree = int(degree)
        n_basis = self.kernel_size ** 3
        self.weight = nn.Parameter(torch.empty(n_basis, in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight)
        self.root = nn.Linear(in_channels, out_channels, bias=False) if root_weight else None
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        if self.root is not None:
            out = out + self.root(x)
        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        basis = tensor_bspline_basis(edge_attr, self.kernel_size, self.degree)
        xw = torch.einsum("ei,pio->epo", x_j, self.weight)
        return torch.einsum("ep,epo->eo", basis, xw)


def make_spline_conv(in_channels: int, out_channels: int, **kwargs) -> nn.Module:
    try:
        from torch_geometric.nn.conv import SplineConv

        return SplineConv(in_channels, out_channels, **kwargs)
    except ImportError:
        return BSplineConv(in_channels, out_channels, **kwargs)
