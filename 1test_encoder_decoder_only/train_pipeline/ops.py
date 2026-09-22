"""Compiled geometric kernels: pytorch3d FPS, pyg-lib radius and SplineConv."""

from __future__ import annotations

import os

import torch
from torch import Tensor

_PYG_WHEEL_HINT = (
    "Install CUDA-matched wheels from "
    "https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html "
    "(pip install pyg-lib -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html)."
)


def _missing_package(name: str, hint: str) -> ImportError:
    return ImportError(f"Required package '{name}' is not installed. {hint}")


try:
    from pytorch3d.ops import sample_farthest_points
except ImportError as exc:
    raise _missing_package("pytorch3d", "Install it with: pip install pytorch3d") from exc

try:
    import pyg_lib  # noqa: F401
except ImportError as exc:
    raise _missing_package("pyg-lib", _PYG_WHEEL_HINT) from exc

from torch_geometric.nn import radius as _pyg_radius
from torch_geometric.nn import radius_graph as _pyg_radius_graph
from torch_geometric.nn.conv import SplineConv
from torch_geometric.typing import WITH_RADIUS, WITH_SPLINE

if not WITH_SPLINE:
    raise ImportError(
        "SplineConv requires pyg-lib spline operators (pyg-lib>=0.6.0; "
        "this replaced torch-spline-conv). " + _PYG_WHEEL_HINT
    )
if not WITH_RADIUS:
    raise ImportError(
        "radius / ball_query require pyg-lib radius operators. " + _PYG_WHEEL_HINT
    )


def _centroid_start_index(pts: Tensor) -> Tensor:
    c = pts.mean(dim=0, keepdim=True)
    return (pts - c).pow(2).sum(dim=-1).argmax()


def fps_indices(pts: Tensor, k: int) -> Tensor:
    """FPS indices for a single cloud [N, 3] via pytorch3d, starting at the centroid-farthest point."""
    k = min(int(k), int(pts.size(0)))
    if k <= 0:
        return pts.new_zeros((0,), dtype=torch.long)
    if k == pts.size(0):
        return torch.arange(k, device=pts.device)
    pts_f = pts.to(dtype=torch.float32).contiguous()
    start = int(_centroid_start_index(pts_f).item())
    perm = torch.arange(pts_f.size(0), device=pts_f.device)
    if start != 0:
        perm[0] = start
        perm[start] = 0
    swapped = pts_f[perm]
    _, loc = sample_farthest_points(swapped.unsqueeze(0), K=k, random_start_point=False)
    return perm[loc.squeeze(0).long()]


def ball_query_packed(
    support: Tensor,
    query: Tensor,
    radius: float,
    support_batch: Tensor,
    query_batch: Tensor,
    max_num_neighbors: int,
) -> Tensor:
    """Neighbors in `support` for each `query` point. Returns [2, E] = (support_idx, query_idx).

    All points with distance <= radius are returned, capped at `max_num_neighbors`
    nearest if a query has more than that many hits.
    """
    # pyg-lib radius is (query, support); PointNeXt grouping wants (support, query).
    return _pyg_radius(
        support.to(dtype=torch.float32).contiguous(),
        query.to(dtype=torch.float32).contiguous(),
        radius,
        support_batch,
        query_batch,
        max_num_neighbors=max_num_neighbors,
    ).flip(0)


def radius_graph_packed(
    pos: Tensor,
    radius: float,
    batch: Tensor,
    loop: bool = True,
    max_num_neighbors: int = 256,
    flow: str = "source_to_target",
) -> Tensor:
    return _pyg_radius_graph(
        pos.to(dtype=torch.float32).contiguous(),
        r=radius,
        batch=batch,
        loop=loop,
        max_num_neighbors=max_num_neighbors,
        flow=flow,
    )


def make_spline_conv(in_channels: int, out_channels: int, **kwargs) -> SplineConv:
    return SplineConv(in_channels, out_channels, **kwargs)


def assert_optimized_cuda_kernels(device):
    """Fail if FPS / radius / SplineConv are not the CUDA pyg-lib + pytorch3d path."""
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "Optimized kernels need a CUDA device (pytorch3d FPS, pyg-lib radius/SplineConv)."
        )
    if not WITH_SPLINE:
        raise RuntimeError("SplineConv is not using pyg-lib CUDA spline ops.")
    if not WITH_RADIUS:
        raise RuntimeError("radius / ball_query are not using pyg-lib CUDA radius ops.")
    torch.cuda.set_device(dev)
    pts = torch.randn(128, 3, device=dev)
    fps = fps_indices(pts, 16)
    if fps.device.type != "cuda":
        raise RuntimeError("pytorch3d FPS did not return CUDA indices.")
    batch = torch.zeros(pts.size(0), dtype=torch.long, device=dev)
    edges = radius_graph_packed(pts, radius=0.75, batch=batch, max_num_neighbors=16)
    if edges.device.type != "cuda":
        raise RuntimeError("pyg-lib radius_graph did not run on CUDA.")
    conv = make_spline_conv(8, 8, dim=3, kernel_size=5, degree=2, root_weight=False).to(dev)
    x = torch.randn(32, 8, device=dev)
    ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], device=dev)
    attr = torch.rand(ei.size(1), 3, device=dev)
    y = conv(x, ei, attr)
    if y.device.type != "cuda" or not torch.isfinite(y).all():
        raise RuntimeError("SplineConv CUDA forward failed.")
    if os.environ.get("RANK", "0") in ("0", ""):
        print(
            f"Kernels OK on {torch.cuda.get_device_name(dev)}: "
            f"pytorch3d FPS, pyg-lib radius, pyg-lib SplineConv "
            f"(WITH_SPLINE={WITH_SPLINE}, WITH_RADIUS={WITH_RADIUS})"
        )
    return True


def composed_radius(
    x: Tensor,
    x_tube: Tensor,
    normal: Tensor,
    tube_radius: float | Tensor,
) -> Tensor:
    """Local radius after displacement: ``r + n · (x − x_tube)``.

    ``tube_radius`` is the healthy radius at each vertex (cached ``r_local``).
    A Python scalar or a tensor of shape ``()``, ``(N,)``, or ``(N, 1)`` is
    accepted so existing scalar callers keep working. There is no hardcoded
    2 mm default inside this function.
    """
    offset = (normal.float() * (x.float() - x_tube.float())).sum(dim=-1)
    if not torch.is_tensor(tube_radius):
        r = offset.new_tensor(tube_radius)
    else:
        r = tube_radius.to(dtype=offset.dtype, device=offset.device)
    if r.numel() == 1:
        return r.reshape(()) + offset
    r = r.reshape(-1)
    if r.shape[0] != offset.shape[0]:
        raise ValueError(
            "composed_radius: tube_radius has "
            f"{int(r.shape[0])} values, expected 1 or {int(offset.shape[0])}"
        )
    return r + offset
