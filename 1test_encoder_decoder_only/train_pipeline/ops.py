"""Geometric kernels: on-device farthest-point, pyg-lib radius and SplineConv."""

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


def _centroid_start_perm(pts: Tensor) -> Tensor:
    """``[B, N]`` permutation that moves each cloud's centroid-farthest point to index 0.

    pytorch3d's farthest-point kernel always starts at index 0. Swapping that
    slot for the centroid-farthest point keeps the historical start without a
    random draw.
    """
    start = (pts - pts.mean(dim=1, keepdim=True)).square().sum(dim=-1).argmax(dim=1)
    b, n = int(pts.size(0)), int(pts.size(1))
    perm = torch.arange(n, device=pts.device).unsqueeze(0).expand(b, n).contiguous()
    rows = torch.arange(b, device=pts.device)
    first = perm[:, 0].clone()
    perm[rows, 0] = start
    perm[rows, start] = first
    return perm


def _fps_uniform_torch(pts: Tensor, k: int) -> Tensor:
    """Iterative farthest-point on ``pts``'s own device. Indices stay in ``[0, N)``.

    One selected point is written down per iteration, so a coincident neighbour
    cannot be chosen twice and the index buffer is never filled with ``-1``.
    """
    b, n, _ = pts.shape
    work = torch.nan_to_num(
        pts.detach().to(dtype=torch.float32).contiguous(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    rows = torch.arange(b, device=work.device)
    farthest = (work - work.mean(dim=1, keepdim=True)).square().sum(dim=-1).argmax(dim=1)
    dist = work.new_full((b, n), float("inf"))
    chosen = torch.empty((b, k), dtype=torch.long, device=work.device)
    for i in range(k):
        chosen[:, i] = farthest
        selected = work[rows, farthest]
        delta = work - selected.unsqueeze(1)
        dist = torch.minimum(dist, delta.square().sum(dim=-1))
        dist[rows, farthest] = -1.0
        farthest = dist.argmax(dim=1)
    return chosen


def _fps_uniform_cpu(pts: Tensor, k: int) -> Tensor:
    """CPU cache path. pytorch3d's CPU kernel is the fast sampler and stays in range."""
    cpu = torch.nan_to_num(
        pts.detach().to(device="cpu", dtype=torch.float32).contiguous(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    n = int(cpu.size(1))
    perm = _centroid_start_perm(cpu)
    swapped = torch.gather(cpu, 1, perm.unsqueeze(-1).expand_as(cpu))
    try:
        from pytorch3d.ops import sample_farthest_points
    except ImportError as exc:
        raise _missing_package("pytorch3d", "Install it with: pip install pytorch3d") from exc
    _, loc = sample_farthest_points(swapped, K=k, random_start_point=False)
    loc = loc.to(dtype=torch.long).clamp(0, n - 1)
    return torch.gather(perm, 1, loc)


def fps_uniform(pts: Tensor, k: int) -> Tensor:
    """Farthest-point indices for equal-sized clouds ``[B, N, C]``.

    CUDA uses the PyTorch loop above. pytorch3d's CUDA farthest-point kernel
    writes past the end of some clouds, and the fault is only reported at the
    next synchronizing ``nonzero``. That killed job 14506605 in resampling and
    job 14509551 in the encoder, after many epochs of the same kernel succeeding.
    CPU tensors keep pytorch3d's CPU kernel, which is what the tube cache uses.
    """
    if pts.dim() != 3:
        raise ValueError(f"fps_uniform expects [B, N, C], got {tuple(pts.shape)}")
    b, n = int(pts.size(0)), int(pts.size(1))
    k = min(int(k), n)
    out_device = pts.device
    if k <= 0:
        return torch.zeros((b, 0), dtype=torch.long, device=out_device)
    if k == n:
        return torch.arange(n, device=out_device).unsqueeze(0).expand(b, n).contiguous()
    if out_device.type == "cuda":
        return _fps_uniform_torch(pts, k)
    return _fps_uniform_cpu(pts, k).to(device=out_device)


def fps_indices(pts: Tensor, k: int) -> Tensor:
    """FPS indices for one cloud, starting at the point farthest from the centroid."""
    n = int(pts.size(0))
    k = min(int(k), n)
    if k <= 0:
        return pts.new_zeros((0,), dtype=torch.long)
    if k == n:
        return torch.arange(n, device=pts.device)
    return fps_uniform(pts.reshape(1, n, -1), k).reshape(-1)


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
    """Fail if radius / SplineConv are not the CUDA pyg-lib path."""
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "Optimized kernels need a CUDA device (pyg-lib radius/SplineConv)."
        )
    if not WITH_SPLINE:
        raise RuntimeError("SplineConv is not using pyg-lib CUDA spline ops.")
    if not WITH_RADIUS:
        raise RuntimeError("radius / ball_query are not using pyg-lib CUDA radius ops.")
    if dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(dev)
    pts = torch.randn(128, 3, device=dev)
    fps = fps_indices(pts, 16)
    if fps.device.type != "cuda":
        raise RuntimeError("FPS did not return CUDA indices.")
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
            f"on-device PyTorch FPS, pyg-lib radius, pyg-lib SplineConv "
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
