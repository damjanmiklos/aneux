"""Geometry helpers: Fourier encodings, FPS, cylindrical upsample, SplineConv pseudo-coords."""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import Tensor

from config import K_THETA, K_U
from ops import fps_indices


def harmonic_encoding_u(u: Tensor, k_u: int = K_U) -> Tensor:
    """γ(u) = [sin(2^i π u), cos(2^i π u)]_{i=0}^{K_u-1} ∈ R^{2 K_u}."""
    u = u.to(dtype=torch.float32).reshape(-1, 1)
    freqs = (2.0 ** torch.arange(k_u, device=u.device, dtype=torch.float32)) * math.pi
    ang = u * freqs.unsqueeze(0)
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


def harmonic_encoding_theta(theta: Tensor, k_theta: int = K_THETA) -> Tensor:
    """γ(θ) = [sin(2^i θ), cos(2^i θ)]_{i=0}^{K_θ-1} ∈ R^{2 K_θ}."""
    theta = theta.to(dtype=torch.float32).reshape(-1, 1)
    freqs = 2.0 ** torch.arange(k_theta, device=theta.device, dtype=torch.float32)
    ang = theta * freqs.unsqueeze(0)
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


def wrap_pi(delta: Tensor) -> Tensor:
    """Wrap angle differences into (-π, π]."""
    return torch.remainder(delta + math.pi, 2.0 * math.pi) - math.pi


def intrinsic_spline_pseudo_coords(
    u: Tensor,
    theta: Tensor,
    tract_id: Tensor,
    edge_index: Tensor,
    u_step: Tensor,
) -> Tensor:
    """Open-spline pseudo-coordinates in [0, 1]^3 from intrinsic (Δu, Δθ, kind).

    Δu is scaled so one longitudinal ring step maps to the cube edge.
    Δθ is wrapped to [-π, π] and mapped to [0, 1].
    The third channel is 0.5 on same-tract edges and 0 on any cross-tract edge.
    """
    u = u.to(dtype=torch.float32).reshape(-1)
    theta = theta.to(dtype=torch.float32).reshape(-1)
    u_step = u_step.to(dtype=torch.float32).reshape(-1).clamp_min(1e-4)
    src, dst = edge_index[0], edge_index[1]
    du = u[dst] - u[src]
    step = torch.maximum(u_step[src], u_step[dst])
    e_u = 0.5 + 0.5 * (du / step).clamp(-1.0, 1.0)

    dth = wrap_pi(theta[dst] - theta[src])
    e_th = 0.5 + 0.5 * (dth / math.pi).clamp(-1.0, 1.0)

    same = (tract_id[src] == tract_id[dst]).to(dtype=torch.float32)
    e_kind = 0.5 * same
    return torch.stack([e_u, e_th, e_kind], dim=-1)


def vertex_frames(theta: Tensor, n_cl: Tensor, t_cl: Tensor, b_cl: Tensor):
    """Local (n_v, t_v, b_v) at scaffold vertices from the Bishop frame and θ.

    n_v(θ) = cosθ n(u) + sinθ b(u)
    b_v(θ) = -sinθ n(u) + cosθ b(u)
    t_v     = t(u)
    """
    th = theta.to(dtype=torch.float32).reshape(-1, 1)
    n_cl = n_cl.to(dtype=torch.float32)
    t_cl = t_cl.to(dtype=torch.float32)
    b_cl = b_cl.to(dtype=torch.float32)
    cos_t = torch.cos(th)
    sin_t = torch.sin(th)
    n_v = cos_t * n_cl + sin_t * b_cl
    b_v = -sin_t * n_cl + cos_t * b_cl
    t_v = t_cl
    n_v = n_v / n_v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    b_v = b_v / b_v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    t_v = t_v / t_v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return n_v, t_v, b_v


def bilinear_cylindrical_upsample(
    field: Tensor,
    n_length_src: int,
    n_radial_src: int,
    n_length_dst: int,
    n_radial_dst: int,
) -> Tensor:
    """Bilinear upsample on a regular (u, θ) cylinder; θ is 2π-periodic.

    `field` is [n_length_src * n_radial_src, C], row-major (length, radial).
    """
    if n_length_src < 1 or n_radial_src < 1:
        raise ValueError("Source grid must be non-empty")
    field = field.to(dtype=torch.float32)
    c = field.shape[-1]
    src = field.reshape(n_length_src, n_radial_src, c)

    if n_length_dst == n_length_src and n_radial_dst == n_radial_src:
        return field

    device = field.device
    dtype = torch.float32
    if n_length_src == 1:
        i0 = torch.zeros(n_length_dst, dtype=torch.long, device=device)
        i1 = i0
        wu = torch.zeros(n_length_dst, dtype=dtype, device=device)
    else:
        u_idx = torch.linspace(0, n_length_src - 1, n_length_dst, device=device, dtype=dtype)
        i0 = u_idx.floor().long().clamp(0, n_length_src - 2)
        i1 = i0 + 1
        wu = (u_idx - i0.to(dtype)).reshape(-1, 1, 1)

    j_idx = (
        torch.arange(n_radial_dst, device=device, dtype=dtype) * (n_radial_src / n_radial_dst)
    )
    j0 = j_idx.floor().long() % n_radial_src
    j1 = (j0 + 1) % n_radial_src
    wv = (j_idx - j_idx.floor()).reshape(1, -1, 1)

    f00 = src[i0][:, j0]
    f01 = src[i0][:, j1]
    f10 = src[i1][:, j0]
    f11 = src[i1][:, j1]
    out = (1 - wu) * (1 - wv) * f00 + (1 - wu) * wv * f01 + wu * (1 - wv) * f10 + wu * wv * f11
    return out.reshape(n_length_dst * n_radial_dst, c)


def upsample_branch_concat(
    field: Tensor,
    n_length_src: Tensor,
    n_radial_src: int,
    n_length_dst: Tensor,
    n_radial_dst: int,
) -> Tensor:
    """Upsample a concatenation of per-branch regular grids."""
    pieces = []
    offset = 0
    nl_src = [int(v) for v in n_length_src.detach().cpu().reshape(-1).tolist()]
    nl_dst = [int(v) for v in n_length_dst.detach().cpu().reshape(-1).tolist()]
    if len(nl_src) != len(nl_dst):
        raise ValueError(
            f"Branch count mismatch during upsample: {len(nl_src)} vs {len(nl_dst)}"
        )
    for ns, nd in zip(nl_src, nl_dst):
        n_nodes = ns * n_radial_src
        pieces.append(
            bilinear_cylindrical_upsample(
                field[offset : offset + n_nodes], ns, n_radial_src, nd, n_radial_dst
            )
        )
        offset += n_nodes
    if not pieces:
        return field.new_zeros((0, field.shape[-1]))
    return torch.cat(pieces, dim=0)


def knn_upsample_tables(src_pts, dst_pts, k=3):
    """kNN inverse-distance tables from a coarser point set onto a finer one."""
    src = np.asarray(src_pts, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst_pts, dtype=np.float64).reshape(-1, 3)
    n_src = int(src.shape[0])
    n_dst = int(dst.shape[0])
    k = max(1, min(int(k), max(n_src, 1)))
    if n_dst == 0 or n_src == 0:
        return (
            np.zeros((n_dst, k), dtype=np.int64),
            np.zeros((n_dst, k), dtype=np.float64),
        )
    dist, idx = cKDTree(src).query(dst, k=k, workers=1)
    if k == 1:
        dist = np.asarray(dist, dtype=np.float64).reshape(-1, 1)
        idx = np.asarray(idx, dtype=np.int64).reshape(-1, 1)
    else:
        dist = np.asarray(dist, dtype=np.float64)
        idx = np.asarray(idx, dtype=np.int64)
    dist = np.maximum(dist, 1e-8)
    weight = 1.0 / dist
    weight = weight / np.clip(weight.sum(axis=1, keepdims=True), 1e-12, None)
    return idx.astype(np.int64, copy=False), weight.astype(np.float64, copy=False)


def knn_weighted_upsample(field: Tensor, index: Tensor, weight: Tensor) -> Tensor:
    """Gather `field[index]` and blend with inverse-distance weights."""
    field = field.to(dtype=torch.float32)
    index = index.long()
    weight = weight.to(dtype=field.dtype, device=field.device)
    if index.device != field.device:
        index = index.to(device=field.device)
    gathered = field[index]
    return (gathered * weight.unsqueeze(-1)).sum(dim=1)


def fps_metric(points: np.ndarray, n_samples: int) -> np.ndarray:
    """Metric-space FPS starting at the point farthest from the centroid."""
    pts = np.asarray(points, dtype=np.float32)
    n = int(pts.shape[0])
    if n == 0:
        raise ValueError("Cannot FPS an empty point set")
    k = min(int(n_samples), n)
    idx = fps_indices(torch.from_numpy(np.ascontiguousarray(pts)), k)
    sampled = pts[idx.detach().cpu().numpy()]

    if sampled.shape[0] < n_samples:
        reps = int(math.ceil(n_samples / sampled.shape[0]))
        sampled = np.tile(sampled, (reps, 1))[:n_samples]
    return sampled.astype(np.float32)


def point_to_polyline_dist(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Nearest distance from each point to a concatenation of polyline vertices."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    cl = np.asarray(polyline, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    if cl.shape[0] == 0:
        return np.full(pts.shape[0], np.inf, dtype=np.float64)
    dist, _ = cKDTree(cl).query(pts, k=1, workers=1)
    return np.asarray(dist, dtype=np.float64).reshape(-1)


def radial_bias_for_zero_init(r_margin: float) -> float:
    """Bias so softplus(bias) ≈ r_margin and Δr ≈ 0 at initialization."""
    r_margin = float(r_margin)
    if r_margin <= 0:
        return 0.0
    return float(math.log(math.expm1(r_margin)))


def decoupled_displacement(
    delta_r: Tensor,
    delta_s: Tensor,
    n_v: Tensor,
    t_v: Tensor,
    b_v: Tensor,
) -> Tensor:
    """x displacement from radial scalar and 2D shear in the local frame."""
    delta_r = delta_r.to(dtype=torch.float32)
    delta_s = delta_s.to(dtype=torch.float32)
    n_v = n_v.to(dtype=torch.float32)
    t_v = t_v.to(dtype=torch.float32)
    b_v = b_v.to(dtype=torch.float32)
    return delta_r * n_v + delta_s[:, 0:1] * t_v + delta_s[:, 1:2] * b_v


def clamp_residual_radial(
    delta_r: Tensor,
    dx_up: Tensor,
    n_v: Tensor,
    r_margin: float,
) -> Tensor:
    """Clamp residual Δr so the composed radial offset stays ≥ -r_margin.

    Coarse Δr already satisfies Δr ≥ -r_margin. Mid/fine add a residual on top of
    an upsampled Cartesian field `dx_up`. Requiring
    `Δr ≥ -(r_margin + n_v · dx_up)` restores `n_v · (dx_up + Δr n_v) ≥ -r_margin`.
    """
    delta_r = delta_r.to(dtype=torch.float32)
    if delta_r.dim() == 1:
        delta_r = delta_r.unsqueeze(-1)
    n_v = n_v.to(dtype=torch.float32)
    dx_up = dx_up.to(dtype=torch.float32)
    r_up = (n_v * dx_up).sum(dim=-1, keepdim=True)
    r_min = -(float(r_margin) + r_up)
    return torch.maximum(delta_r, r_min)
