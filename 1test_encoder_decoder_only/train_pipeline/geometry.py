"""Geometry helpers: Fourier encodings, FPS, cylindrical upsample, SplineConv pseudo-coords."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from config import EDGE_MAX_MM, K_THETA, K_U


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


def spline_pseudo_coords(
    pos: Tensor,
    edge_index: Tensor,
    r_edge_max: float = EDGE_MAX_MM,
) -> Tensor:
    """Open-spline pseudo-coordinates in [0, 1]^3 from rest-pose tube edges.

    e_ij = 0.5 + 0.5 * clamp((x_j - x_i) / (2 r_edge_max), -1, 1)
    """
    pos = pos.to(dtype=torch.float32)
    src, dst = edge_index[0], edge_index[1]
    delta = (pos[dst] - pos[src]) / (2.0 * r_edge_max)
    return 0.5 + 0.5 * delta.clamp(-1.0, 1.0)


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

    # Radial samples are equally spaced on the circle; wrap with modulo.
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
    nl_src = [int(v) for v in n_length_src.reshape(-1).tolist()]
    nl_dst = [int(v) for v in n_length_dst.reshape(-1).tolist()]
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


def fps_metric(points: np.ndarray, n_samples: int) -> np.ndarray:
    """Metric-space farthest point sampling in millimetres. Returns [n_samples, 3]."""
    pts = np.asarray(points, dtype=np.float32)
    n = int(pts.shape[0])
    if n == 0:
        raise ValueError("Cannot FPS an empty point set")
    k = min(int(n_samples), n)

    pts_t = torch.from_numpy(pts).unsqueeze(0)
    try:
        from pytorch3d.ops import sample_farthest_points

        sampled, _ = sample_farthest_points(pts_t, K=k, random_start_point=False)
        sampled = sampled.squeeze(0).numpy()
    except Exception:
        sampled = _fps_numpy(pts, k)

    if sampled.shape[0] < n_samples:
        reps = int(math.ceil(n_samples / sampled.shape[0]))
        sampled = np.tile(sampled, (reps, 1))[:n_samples]
    return sampled.astype(np.float32)


def _fps_numpy(pts: np.ndarray, k: int) -> np.ndarray:
    n = pts.shape[0]
    selected = np.empty(k, dtype=np.int64)
    selected[0] = 0
    dist = np.full(n, np.inf, dtype=np.float64)
    for i in range(1, k):
        last = pts[selected[i - 1]]
        dist = np.minimum(dist, np.linalg.norm(pts - last, axis=1))
        selected[i] = int(np.argmax(dist))
    return pts[selected]


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
