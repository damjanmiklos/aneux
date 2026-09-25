"""Stage-2 hierarchical PointNeXt encoder + progressive SplineConv decoder VAE.

Encoder follows PointNeXt (Qian et al., NeurIPS 2022): stem MLP, FPS set
abstraction, radius grouping with Δp / r, and inverted-residual MLP blocks.
The decoder is a geometry-aware progressive SplineConv deformer with a
tree-valued centerline latent. A future Stage 1 will produce the centerline
and Z that this decoder consumes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from torch_geometric.utils import scatter

import config as _cfg
from config import (
    ATTN_DIM,
    COARSE_ATTN_HEADS,
    DECODER_HIDDEN_DIM,
    GAMMA_THETA_DIM,
    GAMMA_U_DIM,
    INVRES_ALPHA_INIT,
    INVRES_EXPANSION,
    LATENT_DIM,
    LATENT_LEN,
    MAX_TRACTS,
    R_MARGIN_MM,
    SA_STAGES,
    SHEAR_MAX_MM,
    SKIP_GATE_INIT,
    STEM_DIM,
    TRACT_EMB_DIM,
    TUBE_RADIUS_MM,
    Z_ATTN_ALIBI,
    Z_ATTN_ALPHA_INIT,
    Z_ATTN_GATE_MAX,
    Z_ATTN_HEADS,
    Z_ATTN_RADIUS,
    COARSE_ATTN_ALPHA_INIT,
    COARSE_ATTN_GATE_MAX,
    COARSE_ATTN_OSTIUM_U,
    COARSE_ATTN_RINGS,
    normalize_gradient_checkpointing,
)
from geometry import (
    decoupled_displacement,
    harmonic_encoding_theta,
    harmonic_encoding_u,
    intrinsic_spline_pseudo_coords,
    knn_weighted_upsample,
    upsample_branch_concat,
)
from ops import ball_query_packed, fps_indices, make_spline_conv, radius_graph_packed


def _cfg_get(name: str, default):
    return getattr(_cfg, name, default)


def _spline_kernel_size():
    ks = _cfg_get("SPLINE_KERNEL_SIZE", (5, 5, 3))
    if isinstance(ks, int):
        return (5, 5, 3)
    return tuple(int(v) for v in ks)


def _spline_degree() -> int:
    """PyG SplineConv.degree is one int and must be < every kernel axis."""
    return int(_cfg_get("SPLINE_DEGREE", 2))


def _n_conv_per_level() -> int:
    n = _cfg_get("N_CONV_PER_LEVEL", None)
    if n is not None:
        return int(n)
    return 6


def _narrow_sa_stages(stages):
    """Item 29: shrink stage-3/4 width when the stale (256, 512) defaults remain."""
    stages = [list(s) for s in stages]
    if len(stages) >= 4:
        if int(stages[2][3]) == 256:
            stages[2][3] = 128
        if int(stages[3][3]) == 512:
            stages[3][3] = 256
    return tuple(tuple(s) for s in stages)


SIGMA_MIN = float(_cfg_get("SIGMA_MIN", 0.1))
SIGMA_MAX = float(_cfg_get("SIGMA_MAX", math.e))
TOKEN_ATTEND_K = int(_cfg_get("TOKEN_ATTEND_K", 5))
OSTIUM_NEIGHBOR_MM = float(_cfg_get("OSTIUM_NEIGHBOR_MM", 4.0))
RADIAL_FLOOR_FRAC = float(_cfg_get("RADIAL_FLOOR_FRAC", 0.8))
SHEAR_RLOCAL_K = float(_cfg_get("SHEAR_RLOCAL_K", 1.5))
LATENT_HEAD_LAYERS = int(_cfg_get("LATENT_HEAD_LAYERS", 3))
LATENT_HEAD_HEADS = int(_cfg_get("LATENT_HEAD_HEADS", 4))
LOGVAR_MIN = 2.0 * math.log(SIGMA_MIN)
LOGVAR_MAX = 2.0 * math.log(SIGMA_MAX)


def _ckpt_call(enabled, fn, *args):
    if enabled:
        return checkpoint(fn, *args, use_reentrant=False)
    return fn(*args)


def _num_graphs(batch: Tensor, n_graphs: int | None = None) -> int:
    if n_graphs is not None:
        return int(n_graphs)
    n = getattr(batch, "num_graphs", None)
    if n is not None and not torch.is_tensor(n):
        return int(n)
    if batch is None or batch.numel() == 0:
        return 1
    return int(batch.max().item()) + 1


def _ones_batch(n: int, device) -> Tensor:
    return torch.zeros(n, dtype=torch.long, device=device)


def _attr_batch(data, name: str, n: int) -> Tensor:
    b = getattr(data, f"{name}_batch", None)
    if b is not None:
        return b
    return _ones_batch(n, data.x.device)


def _as_bool_mask(t: Tensor) -> Tensor:
    if t.dtype == torch.bool:
        return t
    return t.bool()


def _soft_logvar(raw: Tensor, sigma_min: float = SIGMA_MIN, sigma_max: float = SIGMA_MAX) -> Tensor:
    """log σ² = log σmin² + (log σmax² − log σmin²) · sigmoid(raw). Gradient never dies."""
    log_min = 2.0 * math.log(float(sigma_min))
    log_max = 2.0 * math.log(float(sigma_max))
    return raw.new_tensor(log_min) + (log_max - log_min) * torch.sigmoid(raw)


def fps_packed(pos: Tensor, batch: Tensor, n_out: int, n_graphs: int | None = None) -> Tensor:
    """Farthest-point sample a packed cloud to exactly `n_out` points per graph."""
    n_graphs = _num_graphs(batch, n_graphs)
    if n_graphs == 1:
        k = min(int(n_out), int(pos.size(0)))
        if k <= 0:
            return pos.new_zeros((0,), dtype=torch.long)
        if k == pos.size(0):
            return torch.arange(pos.size(0), device=pos.device)
        return fps_indices(pos, k)
    pieces = []
    for g in range(n_graphs):
        node_idx = (batch == g).nonzero(as_tuple=False).view(-1)
        pts = pos[node_idx]
        k = min(int(n_out), int(pts.size(0)))
        if k <= 0:
            continue
        if k == pts.size(0):
            pieces.append(node_idx)
            continue
        loc = fps_indices(pts, k)
        pieces.append(node_idx[loc])
    if not pieces:
        return torch.arange(pos.size(0), device=pos.device)
    return torch.cat(pieces, dim=0)


def nearest_centerline_attr(
    pos: Tensor,
    pos_batch: Tensor,
    cl_dense: Tensor,
    cl_batch: Tensor,
    cl_tract: Tensor,
    n_graphs: int | None = None,
):
    """Nearest dense centerline sample → (u_local, tract_id) for each point."""
    u_out = pos.new_zeros(pos.size(0))
    t_out = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)
    n_graphs = _num_graphs(pos_batch, n_graphs)
    if n_graphs == 1:
        c = cl_dense
        if pos.size(0) == 0 or c.size(0) == 0:
            return u_out, t_out
        idx = torch.cdist(pos, c[:, :3]).argmin(dim=1)
        return c[idx, 3], cl_tract[idx]
    for g in range(n_graphs):
        p_mask = pos_batch == g
        c_mask = cl_batch == g
        p = pos[p_mask]
        c = cl_dense[c_mask]
        if p.numel() == 0 or c.numel() == 0:
            continue
        d = torch.cdist(p, c[:, :3])
        idx = d.argmin(dim=1)
        u_out[p_mask] = c[idx, 3]
        t_out[p_mask] = cl_tract[c_mask][idx]
    return u_out, t_out


def _reshape_token_field(value: Tensor, n_graphs: int, latent_len: int, trailing: int | None = None):
    if trailing is None:
        if value.dim() == 2 and value.size(0) == n_graphs:
            return value
        if value.numel() == n_graphs * latent_len:
            return value.reshape(n_graphs, latent_len)
        return value.reshape(1, -1)
    if value.dim() == 3 and value.size(0) == n_graphs:
        return value
    if value.numel() == n_graphs * latent_len * trailing:
        return value.reshape(n_graphs, latent_len, trailing)
    return value.reshape(1, value.size(0), trailing)


def _token_tables(data, n_graphs: int, latent_len: int):
    """Return [B, L] token descriptors from a possibly batched Data object."""
    u = data.latent_u
    tract = data.latent_tract_id
    attend = data.token_attend
    is_junc = data.latent_is_junction
    if attend.dtype != torch.bool:
        attend = attend.bool()
    valid = getattr(data, "latent_valid", None)
    if u.dim() == 2 and u.size(0) == n_graphs:
        if valid is None:
            valid = torch.ones(n_graphs, latent_len, dtype=torch.bool, device=u.device)
        else:
            valid = _reshape_token_field(_as_bool_mask(valid), n_graphs, latent_len)
        return u, tract, attend, is_junc, valid
    packed = getattr(data, "latent_u_batch", None) is not None or u.numel() == n_graphs * latent_len
    if packed:
        u = u.reshape(n_graphs, latent_len)
        tract = tract.reshape(n_graphs, latent_len)
        is_junc = is_junc.reshape(n_graphs, latent_len)
        attend = attend.reshape(n_graphs, latent_len, attend.size(-1))
        if valid is None:
            valid = torch.ones(n_graphs, latent_len, dtype=torch.bool, device=u.device)
        else:
            valid = _reshape_token_field(_as_bool_mask(valid), n_graphs, latent_len)
        return u, tract, attend, is_junc, valid
    u = u.reshape(1, -1)
    tract = tract.reshape(1, -1)
    is_junc = is_junc.reshape(1, -1)
    attend = attend.reshape(1, attend.size(0), attend.size(-1))
    if valid is None:
        valid = torch.ones(1, u.size(1), dtype=torch.bool, device=u.device)
    else:
        valid = _as_bool_mask(valid).reshape(1, -1)
    if n_graphs != 1:
        raise ValueError("Missing latent_u_batch for batched tree tokens")
    return u, tract, attend, is_junc, valid


def _token_pos_table(data, n_graphs: int, latent_len: int):
    pos = getattr(data, "latent_pos", None)
    if pos is None or not torch.is_tensor(pos):
        return None
    return _reshape_token_field(pos, n_graphs, latent_len, trailing=3)


def _token_depth_table(data, n_graphs: int, latent_len: int, device, dtype):
    depth = getattr(data, "latent_depth", None)
    if depth is None or not torch.is_tensor(depth):
        return torch.zeros(n_graphs, latent_len, 1, device=device, dtype=dtype)
    depth = _reshape_token_field(depth.to(device=device, dtype=dtype), n_graphs, latent_len)
    return depth.unsqueeze(-1)


def _first_matching_tensor(data, names, n: int):
    for name in names:
        v = getattr(data, name, None)
        if torch.is_tensor(v) and v.reshape(-1).numel() == n:
            return v.to(dtype=torch.float32).reshape(n)
    return None


def _level_r_local(data, n: int, level: str, fallback: float) -> Tensor:
    names = {
        "fine": ("r_local",),
        "mid": ("r_local_mid", "r_local"),
        "coarse": ("r_local_coarse", "r_local"),
    }[level]
    found = _first_matching_tensor(data, names, n)
    if found is not None:
        return found.clamp_min(1e-4)
    device = data.x.device if getattr(data, "x", None) is not None else "cpu"
    return torch.full((n,), float(fallback), dtype=torch.float32, device=device)


def _level_geom_features(data, n: int, level: str, fallback_r: float) -> Tensor:
    """[r_local, curvature, torsion, ostium distance] with zeros/r fallbacks."""
    r = _level_r_local(data, n, level, fallback_r)
    if level == "fine":
        k_names = ("curvature", "kappa", "curvature_fine", "kappa_fine")
        t_names = ("torsion", "tau", "torsion_fine")
        d_names = ("d_ostium", "ostium_dist", "dist_ostium", "ostium_distance")
    elif level == "mid":
        k_names = ("curvature_mid", "kappa_mid", "curvature", "kappa")
        t_names = ("torsion_mid", "tau_mid", "torsion", "tau")
        d_names = ("d_ostium_mid", "ostium_dist_mid", "d_ostium", "ostium_dist")
    else:
        k_names = ("curvature_coarse", "kappa_coarse", "curvature", "kappa")
        t_names = ("torsion_coarse", "tau_coarse", "torsion", "tau")
        d_names = ("d_ostium_coarse", "ostium_dist_coarse", "d_ostium", "ostium_dist")
    device = r.device
    k = _first_matching_tensor(data, k_names, n)
    tau = _first_matching_tensor(data, t_names, n)
    d_ost = _first_matching_tensor(data, d_names, n)
    if k is None:
        k = torch.zeros(n, device=device, dtype=r.dtype)
    if tau is None:
        tau = torch.zeros(n, device=device, dtype=r.dtype)
    if d_ost is None:
        d_ost = torch.zeros(n, device=device, dtype=r.dtype)
    return torch.stack([r, k, tau, d_ost], dim=-1)


def apply_boundary_plane_projection(x: Tensor, data, suffix: str = "") -> Tensor:
    """Slide rim vertices in their ostium cut plane. No-op when planes are missing.

    Accepts packed [N, 3] origin+normal (optionally with a boolean mask), or a
    packed-boundary layout [n_boundary, 3] plus a [N] mask of rim vertices.
    """
    if x.numel() == 0:
        return x
    n = int(x.size(0))
    origin = None
    normal = None
    mask = None
    for orig_name, nrm_name in (
        (f"boundary_plane_origin{suffix}", f"boundary_plane_normal{suffix}"),
        (f"plane_origin{suffix}", f"plane_normal{suffix}"),
        (f"ostium_plane_origin{suffix}", f"ostium_plane_normal{suffix}"),
    ):
        o = getattr(data, orig_name, None)
        nr = getattr(data, nrm_name, None)
        if torch.is_tensor(o) and torch.is_tensor(nr):
            origin, normal = o, nr
            break
    if origin is None or normal is None:
        return x
    for mask_name in (
        f"boundary_mask{suffix}",
        f"is_boundary{suffix}",
        f"rim_mask{suffix}",
        f"boundary_vertex{suffix}",
    ):
        m = getattr(data, mask_name, None)
        if torch.is_tensor(m) and m.reshape(-1).numel() == n:
            mask = _as_bool_mask(m.reshape(-1))
            break
    origin = origin.to(device=x.device, dtype=x.dtype)
    normal = normal.to(device=x.device, dtype=x.dtype)
    if origin.dim() == 1:
        origin = origin.unsqueeze(0)
    if normal.dim() == 1:
        normal = normal.unsqueeze(0)

    def _project(pts, org, nrm):
        nrm = F.normalize(nrm, dim=-1, eps=1e-8)
        return pts - nrm * ((pts - org) * nrm).sum(dim=-1, keepdim=True)

    if origin.size(0) == n and normal.size(0) == n:
        x_proj = _project(x, origin, normal)
        if mask is None:
            live = normal.norm(dim=-1, keepdim=True) > 1e-6
            return torch.where(live, x_proj, x)
        return torch.where(mask.unsqueeze(-1), x_proj, x)
    if mask is not None and origin.size(0) == int(mask.sum()) and normal.size(0) == int(mask.sum()):
        idx = mask.nonzero(as_tuple=False).view(-1)
        x = x.clone()
        x[idx] = _project(x[idx], origin, normal)
        return x
    return x


def _clamp_residual_radial(delta_r: Tensor, dx_up: Tensor, n_v: Tensor, floor: Tensor) -> Tensor:
    """Composed radial offset stays ≥ −floor, with per-vertex `floor` (0.8 r_local)."""
    delta_r = delta_r.to(dtype=torch.float32)
    if delta_r.dim() == 1:
        delta_r = delta_r.unsqueeze(-1)
    floor = floor.to(dtype=torch.float32, device=delta_r.device).reshape(-1, 1)
    n_v = n_v.to(dtype=torch.float32)
    dx_up = dx_up.to(dtype=torch.float32)
    r_up = (n_v * dx_up).sum(dim=-1, keepdim=True)
    return torch.maximum(delta_r, -(floor + r_up))


def _stem_features(data, x_true: Tensor) -> Tensor:
    """xyz + GT normals + template signed distance (3 → 7). Missing fields → zeros."""
    n = x_true.size(0)
    nrm = getattr(data, "x_true_normal", None)
    if not torch.is_tensor(nrm) or nrm.shape != x_true.shape:
        nrm = x_true.new_zeros(n, 3)
    else:
        nrm = nrm.to(dtype=x_true.dtype, device=x_true.device)
    sdf = getattr(data, "x_true_template_sdf", None)
    if sdf is None:
        sdf = getattr(data, "x_true_sdf", None)
    if not torch.is_tensor(sdf) or sdf.reshape(-1).numel() != n:
        sdf = x_true.new_zeros(n, 1)
    else:
        sdf = sdf.to(dtype=x_true.dtype, device=x_true.device).reshape(n, 1)
    return torch.cat([x_true, nrm, sdf], dim=-1)


class InvResMLP(nn.Module):
    """Inverted residual block (PointNeXt) with spec Δp/r, LayerNorm, and residual α."""

    def __init__(
        self,
        dim: int,
        radius: float,
        k_neighbors: int,
        expansion: int = INVRES_EXPANSION,
        alpha_init: float = INVRES_ALPHA_INIT,
    ):
        super().__init__()
        hidden = dim * expansion
        self.radius = float(radius)
        self.k_neighbors = int(k_neighbors)
        self.expand = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(inplace=True),
        )
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden, hidden),
        )
        self.reduce = nn.Sequential(
            nn.Linear(hidden, dim),
            nn.LayerNorm(dim),
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, h: Tensor, pos: Tensor, batch: Tensor) -> Tensor:
        h_mid = self.expand(h)
        edge_index = radius_graph_packed(
            pos,
            radius=self.radius,
            batch=batch,
            loop=True,
            max_num_neighbors=self.k_neighbors,
            flow="source_to_target",
        )
        src, dst = edge_index[0], edge_index[1]
        delta = ((pos[src] - pos[dst]) / self.radius).clamp(-1.0, 1.0)
        msg = h_mid[src] + self.pos_mlp(delta)
        h_agg = scatter(msg, dst, dim=0, dim_size=h.size(0), reduce="max")
        return h + self.alpha * self.reduce(h_agg)


class SetAbstraction(nn.Module):
    """FPS downsample + radius grouping + MLP, with Δp / r (PointNeXt Eq. 2)."""

    def __init__(self, in_dim: int, out_dim: int, n_out: int, radius: float, k_neighbors: int):
        super().__init__()
        self.n_out = int(n_out)
        self.radius = float(radius)
        self.k_neighbors = int(k_neighbors)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim + 3, out_dim),
            nn.LayerNorm(out_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, h: Tensor, pos: Tensor, batch: Tensor, n_graphs: int | None = None):
        idx = fps_packed(pos, batch, self.n_out, n_graphs=n_graphs)
        new_pos = pos[idx]
        new_batch = batch[idx]
        assign = ball_query_packed(
            pos,
            new_pos,
            self.radius,
            batch,
            new_batch,
            max_num_neighbors=self.k_neighbors,
        )
        nbr, qry = assign[0], assign[1]
        self_qry = torch.arange(new_pos.size(0), device=pos.device)
        nbr = torch.cat([nbr, idx], dim=0)
        qry = torch.cat([qry, self_qry], dim=0)
        delta = ((pos[nbr] - new_pos[qry]) / self.radius).clamp(-1.0, 1.0)
        feat = torch.cat([h[nbr], delta], dim=-1)
        feat = self.mlp(feat)
        new_h = scatter(feat, qry, dim=0, dim_size=new_pos.size(0), reduce="max")
        return new_h, new_pos, new_batch


class TokenSelfAttentionBlock(nn.Module):
    """Pre-norm token transformer block over the centerline tree."""

    def __init__(self, dim: int, n_heads: int, ff_mult: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True, dropout=0.0)
        self.ln2 = nn.LayerNorm(dim)
        hidden = dim * ff_mult
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: Tensor, key_padding_mask: Tensor | None) -> Tensor:
        if key_padding_mask is not None and key_padding_mask.all(dim=-1).any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[key_padding_mask.all(dim=-1), 0] = False
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + a
        return x + self.ff(self.ln2(x))


class CenterlineLatentHead(nn.Module):
    """Local pool of encoder centres onto tree tokens, then a small token transformer.

    Each encoder centre is assigned to its nearest token along the tree; features
    are max-pooled per token. 2–4 self-attention layers mix tokens with γ(u),
    tract, and branch-depth encodings. Soft σ bound replaces the dead-gradient
    logvar clamp.
    """

    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        latent_len: int,
        attn_dim: int,
        n_layers: int = LATENT_HEAD_LAYERS,
        n_heads: int = LATENT_HEAD_HEADS,
        sigma_min: float = SIGMA_MIN,
        sigma_max: float = SIGMA_MAX,
    ):
        super().__init__()
        self.latent_len = int(latent_len)
        self.attn_dim = int(attn_dim)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        n_heads = int(n_heads)
        if n_heads < 1 or self.attn_dim % n_heads != 0:
            n_heads = 1
        self.tract_emb = nn.Embedding(MAX_TRACTS + 1, attn_dim)
        pe_dim = GAMMA_U_DIM + attn_dim + 1
        self.fuse = nn.Sequential(
            nn.Linear(in_dim + pe_dim, attn_dim),
            nn.LayerNorm(attn_dim),
            nn.GELU(),
        )
        n_layers = max(2, min(4, int(n_layers)))
        self.blocks = nn.ModuleList(
            [TokenSelfAttentionBlock(attn_dim, n_heads) for _ in range(n_layers)]
        )
        self.out_ln = nn.LayerNorm(attn_dim)
        self.mu_head = nn.Linear(attn_dim, latent_dim)
        self.logvar_head = nn.Linear(attn_dim, latent_dim)

    def _tract_index(self, tract_id: Tensor) -> Tensor:
        idx = tract_id.clone()
        idx = torch.where(idx < 0, torch.full_like(idx, MAX_TRACTS), idx)
        return idx.clamp(0, MAX_TRACTS)

    def _assign_centres(
        self,
        u_pts: Tensor,
        tract_pts: Tensor,
        pos: Tensor | None,
        token_u: Tensor,
        token_tract: Tensor,
        token_pos: Tensor | None,
        valid: Tensor,
    ) -> Tensor:
        """Nearest token along the tree (same-tract |Δu|; 3-D fallback)."""
        n_c = u_pts.size(0)
        if n_c == 0:
            return u_pts.new_zeros((0,), dtype=torch.long)
        du = (u_pts.unsqueeze(1) - token_u.unsqueeze(0)).abs()
        same = tract_pts.unsqueeze(1) == token_tract.unsqueeze(0)
        large = du.new_tensor(1.0e6)
        dist = torch.where(same, du, large)
        dist = dist.masked_fill(~valid.unsqueeze(0), large)
        has_same = (same & valid.unsqueeze(0)).any(dim=1)
        if pos is not None and token_pos is not None and pos.size(0) == n_c:
            d3 = torch.cdist(pos, token_pos)
            d3 = d3.masked_fill(~valid.unsqueeze(0), large)
            dist = torch.where(has_same.unsqueeze(1), dist, d3)
        else:
            du_all = du.masked_fill(~valid.unsqueeze(0), large)
            dist = torch.where(has_same.unsqueeze(1), dist, du_all)
        return dist.argmin(dim=1)

    def _pool_graph(self, h, u_pts, tract_pts, pos, token_u, token_tract, token_pos, valid):
        l = token_u.size(0)
        pooled = h.new_zeros(l, h.size(-1))
        if h.size(0) == 0 or not bool(valid.any()):
            return pooled
        assign = self._assign_centres(u_pts, tract_pts, pos, token_u, token_tract, token_pos, valid)
        pooled = scatter(h, assign, dim=0, dim_size=l, reduce="max")
        return pooled * valid.unsqueeze(-1).to(dtype=pooled.dtype)

    def _encode_tokens(self, pooled, token_u, token_tract, depth, valid):
        b, l, _ = pooled.shape
        gamma_u = harmonic_encoding_u(token_u.reshape(-1)).reshape(b, l, -1)
        te = self.tract_emb(self._tract_index(token_tract))
        pe = torch.cat([gamma_u, te, depth], dim=-1)
        h = self.fuse(torch.cat([pooled, pe], dim=-1))
        pad = ~valid
        if pad.all():
            pad = pad.clone()
            pad[:, 0] = False
        for blk in self.blocks:
            h = blk(h, key_padding_mask=pad)
        h = self.out_ln(h)
        mu = self.mu_head(h)
        logvar = _soft_logvar(self.logvar_head(h), self.sigma_min, self.sigma_max)
        valid_f = valid.unsqueeze(-1).to(dtype=mu.dtype)
        mu = mu * valid_f
        logvar = torch.where(valid.unsqueeze(-1), logvar, torch.zeros_like(logvar))
        return mu, logvar

    def forward(self, h: Tensor, u_pts: Tensor, tract_pts: Tensor, batch: Tensor, data, pos: Tensor | None = None):
        n_graphs = int(getattr(data, "num_graphs", 1) or 1)
        token_u, token_tract, _, _, valid = _token_tables(data, n_graphs, self.latent_len)
        token_pos = _token_pos_table(data, n_graphs, self.latent_len)
        depth = _token_depth_table(data, n_graphs, self.latent_len, h.device, h.dtype)
        valid = valid.to(device=h.device)
        token_u = token_u.to(device=h.device, dtype=h.dtype)
        token_tract = token_tract.to(device=h.device)
        if token_pos is not None:
            token_pos = token_pos.to(device=h.device, dtype=h.dtype)
        pooled = h.new_zeros(n_graphs, self.latent_len, h.size(-1))
        if n_graphs == 1:
            tpos = None if token_pos is None else token_pos[0]
            pooled[0] = self._pool_graph(
                h, u_pts, tract_pts, pos, token_u[0], token_tract[0], tpos, valid[0],
            )
        else:
            for g in range(n_graphs):
                mask = batch == g
                tpos = None if token_pos is None else token_pos[g]
                gpos = None if pos is None else pos[mask]
                pooled[g] = self._pool_graph(
                    h[mask], u_pts[mask], tract_pts[mask], gpos,
                    token_u[g], token_tract[g], tpos, valid[g],
                )
        return self._encode_tokens(pooled, token_u, token_tract, depth, valid)


class PointNeXtEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        latent_len: int = LATENT_LEN,
        stem_dim: int = STEM_DIM,
        stages=SA_STAGES,
        gradient_checkpointing: str = "off",
        narrow_late_stages: bool = False,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.latent_len = int(latent_len)
        self.gradient_checkpointing = normalize_gradient_checkpointing(gradient_checkpointing)
        stages = tuple(stages)
        if narrow_late_stages:
            stages = _narrow_sa_stages(stages)
        self.stem = nn.Sequential(
            nn.Linear(7, stem_dim),
            nn.LayerNorm(stem_dim),
            nn.LeakyReLU(inplace=True),
        )
        sa, inv = [], []
        in_dim = stem_dim
        for n_out, radius, k_nb, hidden, n_blocks in stages:
            sa.append(SetAbstraction(in_dim, hidden, n_out, radius, k_nb))
            inv.append(nn.ModuleList(
                [InvResMLP(hidden, radius, k_nb) for _ in range(int(n_blocks))]
            ))
            in_dim = hidden
        self.sa_layers = nn.ModuleList(sa)
        self.inv_layers = nn.ModuleList(inv)
        self.latent_head = CenterlineLatentHead(in_dim, self.latent_dim, latent_len, attn_dim=in_dim)

    def forward(self, data):
        n_graphs = int(getattr(data, "num_graphs", 1) or 1)
        x_true = data.x_true
        x_true_batch = (
            data.x_true_batch
            if getattr(data, "x_true_batch", None) is not None
            else _ones_batch(x_true.size(0), x_true.device)
        )
        cl_batch = _attr_batch(data, "cl_dense", data.cl_dense.size(0))
        cl_tract = data.cl_tract_id
        h = self.stem(_stem_features(data, x_true))
        pos, batch = x_true, x_true_batch
        for sa, inv_blocks in zip(self.sa_layers, self.inv_layers):
            h, pos, batch = sa(h, pos, batch, n_graphs=n_graphs)
            for blk in inv_blocks:
                h = _ckpt_call(
                    self.training and self.gradient_checkpointing == "all",
                    blk,
                    h,
                    pos,
                    batch,
                )
        u_pts, tract_pts = nearest_centerline_attr(
            pos, batch, data.cl_dense, cl_batch, cl_tract, n_graphs=n_graphs
        )
        return self.latent_head(h, u_pts, tract_pts, batch, data, pos=pos)


def _decoder_token_allow(
    token_attend: Tensor,
    node_tract: Tensor,
    token_u: Tensor,
    latent_valid: Tensor,
    token_tract: Tensor | None,
    node_u: Tensor,
    token_pos: Tensor | None,
    node_pos: Tensor | None,
    k: int,
    ostium_mm: float,
):
    """[N, L] mask: K nearest same-branch tokens + ostium-neighbour tokens."""
    tract = node_tract.clamp(0, MAX_TRACTS - 1)
    allow = token_attend[:, tract].transpose(0, 1)
    allow = allow & latent_valid.unsqueeze(0)
    if token_tract is None:
        return allow
    same = tract.unsqueeze(1) == token_tract.unsqueeze(0)
    cand = allow & same
    du = (node_u.unsqueeze(1) - token_u.unsqueeze(0)).abs()
    large = du.new_tensor(1.0e6)
    dist = torch.where(cand, du, large)
    kk = min(max(int(k), 1), int(token_u.numel()))
    knn = torch.zeros_like(allow)
    if kk > 0 and node_u.numel() > 0:
        _, idx = dist.topk(kk, dim=-1, largest=False)
        knn.scatter_(1, idx, True)
        knn = knn & cand
    extra = torch.zeros_like(allow)
    ost_tok = ((token_u <= 0.05) | (token_u >= 0.95)) & latent_valid
    if token_pos is not None and node_pos is not None and bool(ost_tok.any()):
        d_tt = torch.cdist(token_pos, token_pos)
        near_ost = (d_tt <= float(ostium_mm)) & ost_tok.unsqueeze(0)
        diff_tr = token_tract.unsqueeze(1) != token_tract.unsqueeze(0)
        nbr = ((near_ost & diff_tr).any(dim=1) & latent_valid)
        d_vo = torch.cdist(node_pos, token_pos[ost_tok])
        v_near = (d_vo <= float(ostium_mm)).any(dim=1)
        extra = v_near.unsqueeze(1) & nbr.unsqueeze(0)
    else:
        v_near = (node_u <= 0.15) | (node_u >= 0.85)
        other = (token_tract.unsqueeze(0) != tract.unsqueeze(1)) & ost_tok.unsqueeze(0)
        extra = v_near.unsqueeze(1) & other
    return (knn | extra) & latent_valid.unsqueeze(0)


class LatentCrossAttention(nn.Module):
    """Scaffold nodes query the tree latent via Fourier (u, θ) with a local token mask."""

    def __init__(self, latent_dim: int, hidden_dim: int, attn_dim: int, latent_len: int):
        super().__init__()
        self.latent_len = int(latent_len)
        self.attn_dim = int(attn_dim)
        self.k_tokens = TOKEN_ATTEND_K
        self.ostium_mm = OSTIUM_NEIGHBOR_MM
        self.w_q = nn.Linear(GAMMA_U_DIM + GAMMA_THETA_DIM, attn_dim)
        self.w_k = nn.Linear(latent_dim + GAMMA_U_DIM, attn_dim)
        self.w_v = nn.Linear(latent_dim, attn_dim)
        self.out = nn.Linear(attn_dim + GAMMA_U_DIM + GAMMA_THETA_DIM, hidden_dim)

    def forward(
        self,
        z: Tensor,
        u: Tensor,
        theta: Tensor,
        node_batch: Tensor,
        node_tract: Tensor,
        token_u: Tensor,
        token_attend: Tensor,
        latent_valid: Tensor | None = None,
        token_tract: Tensor | None = None,
        token_pos: Tensor | None = None,
        node_pos: Tensor | None = None,
    ) -> Tensor:
        gamma_u = harmonic_encoding_u(u)
        gamma_th = harmonic_encoding_theta(theta)
        q = self.w_q(torch.cat([gamma_u, gamma_th], dim=-1))
        n_graphs = z.size(0)
        gamma_uk = harmonic_encoding_u(token_u.reshape(-1)).reshape(n_graphs, self.latent_len, -1)
        k = self.w_k(torch.cat([z, gamma_uk], dim=-1))
        v = self.w_v(z)
        scale = math.sqrt(self.attn_dim)
        tract = node_tract.clamp(0, MAX_TRACTS - 1)
        if latent_valid is None:
            latent_valid = torch.ones(n_graphs, self.latent_len, dtype=torch.bool, device=z.device)
        else:
            latent_valid = _as_bool_mask(latent_valid).to(device=z.device)
        restrict = token_tract is not None

        def _scores_and_weights(qg, kg, vg, g, node_idx):
            scores = qg.matmul(kg.transpose(0, 1)) / scale
            if restrict:
                tpos = None if token_pos is None else token_pos[g]
                npos = None if node_pos is None else node_pos[node_idx]
                allow = _decoder_token_allow(
                    token_attend[g],
                    tract[node_idx],
                    token_u[g],
                    latent_valid[g],
                    token_tract[g],
                    u[node_idx],
                    tpos,
                    npos,
                    self.k_tokens,
                    self.ostium_mm,
                )
            else:
                allow = token_attend[g][:, tract[node_idx]].transpose(0, 1)
                allow = allow & latent_valid[g].unsqueeze(0)
            scores = scores.masked_fill(~allow, -1.0e4)
            orphan = ~allow.any(dim=-1)
            scores = torch.where(orphan.unsqueeze(-1), torch.zeros_like(scores), scores)
            w = torch.softmax(scores, dim=-1)
            return w.matmul(vg)

        if n_graphs == 1:
            node_idx = torch.arange(q.size(0), device=q.device)
            a = _scores_and_weights(q, k[0], v[0], 0, node_idx)
            return self.out(torch.cat([a, gamma_u, gamma_th], dim=-1))
        a = q.new_empty(q.size(0), self.attn_dim)
        for g in range(n_graphs):
            mask = node_batch == g
            node_idx = mask.nonzero(as_tuple=False).view(-1)
            a[mask] = _scores_and_weights(q[mask], k[g], v[g], g, node_idx)
        return self.out(torch.cat([a, gamma_u, gamma_th], dim=-1))


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _bounded_gate(raw: Tensor, max_val: float) -> Tensor:
    return torch.sigmoid(raw).clamp(max=float(max_val))


def _zero_residual_out(linear: nn.Linear) -> None:
    nn.init.zeros_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class CoarsePositionalSelfAttention(nn.Module):
    """Per-graph multi-head self-attention on the coarse scaffold.

    Q/K concatenate [LN(h), γ(u), γ(θ), tract_emb]; LayerNorm is applied only to
    h. Values come from h. Same-tract scores are limited to a few rings along u;
    cross-tract mixing is allowed only near ostia (u near 0 or 1). Residual is
    sigmoid-gated and the output projection is zero-initialized.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int = COARSE_ATTN_HEADS,
        tract_emb_dim: int = TRACT_EMB_DIM,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim)
        n_heads = int(n_heads)
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim {hidden_dim} must be divisible by n_heads {n_heads}")
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.ln = nn.LayerNorm(hidden_dim)
        self.tract_emb = nn.Embedding(MAX_TRACTS + 1, int(tract_emb_dim))
        qk_in = hidden_dim + GAMMA_U_DIM + GAMMA_THETA_DIM + int(tract_emb_dim)
        self.w_q = nn.Linear(qk_in, hidden_dim)
        self.w_k = nn.Linear(qk_in, hidden_dim)
        self.w_v = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        _zero_residual_out(self.out)
        self.alpha_raw = nn.Parameter(torch.tensor(_logit(COARSE_ATTN_ALPHA_INIT)))
        self.gate_max = float(COARSE_ATTN_GATE_MAX)
        self.n_rings = int(COARSE_ATTN_RINGS)
        self.ostium_u = float(COARSE_ATTN_OSTIUM_U)

    def _tract_index(self, tract_id: Tensor) -> Tensor:
        idx = tract_id.clone()
        idx = torch.where(idx < 0, torch.full_like(idx, MAX_TRACTS), idx)
        return idx.clamp(0, MAX_TRACTS)

    def forward(
        self,
        h: Tensor,
        u: Tensor,
        theta: Tensor,
        tract_id: Tensor,
        batch: Tensor,
        u_step: Tensor | None = None,
        n_graphs: int | None = None,
    ) -> Tensor:
        if h.size(0) == 0:
            return h
        h_n = self.ln(h)
        gamma_u = harmonic_encoding_u(u)
        gamma_th = harmonic_encoding_theta(theta)
        te = self.tract_emb(self._tract_index(tract_id))
        qk = torch.cat([h_n, gamma_u, gamma_th, te], dim=-1)
        q = self.w_q(qk).view(-1, self.n_heads, self.head_dim)
        k = self.w_k(qk).view(-1, self.n_heads, self.head_dim)
        v = self.w_v(h).view(-1, self.n_heads, self.head_dim)
        mixed = h.new_zeros(h.size(0), self.n_heads, self.head_dim)
        scale = math.sqrt(self.head_dim)
        n_graphs = _num_graphs(batch, n_graphs)
        if n_graphs == 1:
            qg = q.transpose(0, 1)
            kg = k.transpose(0, 1)
            vg = v.transpose(0, 1)
            scores = torch.matmul(qg, kg.transpose(-2, -1)) / scale
            same = tract_id.unsqueeze(1) == tract_id.unsqueeze(0)
            du = (u.unsqueeze(1) - u.unsqueeze(0)).abs()
            if u_step is not None:
                sg = u_step.to(dtype=du.dtype).clamp_min(1e-4)
                lim = float(self.n_rings) * 0.5 * (sg.unsqueeze(1) + sg.unsqueeze(0))
            else:
                lim = du.new_ones(du.shape)
            ost = (u <= self.ostium_u) | (u >= 1.0 - self.ostium_u)
            allow = (same & (du <= lim)) | ((~same) & ost.unsqueeze(1) & ost.unsqueeze(0))
            scores = scores.masked_fill(~allow.unsqueeze(0), -1.0e9)
            w = torch.softmax(scores, dim=-1)
            mixed = torch.matmul(w, vg).transpose(0, 1)
        else:
            for g in range(n_graphs):
                mask = batch == g
                qg = q[mask]
                if qg.numel() == 0:
                    continue
                qg = qg.transpose(0, 1)
                kg = k[mask].transpose(0, 1)
                vg = v[mask].transpose(0, 1)
                scores = torch.matmul(qg, kg.transpose(-2, -1)) / scale
                ug = u[mask]
                tg = tract_id[mask]
                same = tg.unsqueeze(1) == tg.unsqueeze(0)
                du = (ug.unsqueeze(1) - ug.unsqueeze(0)).abs()
                if u_step is not None:
                    sg = u_step[mask].to(dtype=du.dtype).clamp_min(1e-4)
                    lim = float(self.n_rings) * 0.5 * (sg.unsqueeze(1) + sg.unsqueeze(0))
                else:
                    lim = du.new_ones(du.shape)
                local = same & (du <= lim)
                ost = (ug <= self.ostium_u) | (ug >= 1.0 - self.ostium_u)
                cross = (~same) & ost.unsqueeze(1) & ost.unsqueeze(0)
                allow = local | cross
                scores = scores.masked_fill(~allow.unsqueeze(0), -1.0e9)
                w = torch.softmax(scores, dim=-1)
                mixed[mask] = torch.matmul(w, vg).transpose(0, 1)
        delta = self.out(mixed.reshape(h.size(0), self.hidden_dim))
        return h + _bounded_gate(self.alpha_raw, self.gate_max) * delta


class LatentTractSelfAttention(nn.Module):
    """Mild per-tract residual mix of latent tokens *before* reparameterization.

    Q/K concatenate [LN(z), γ(u)]; values come from z. Junction tokens
    (tract_id < 0) and padded tokens (`latent_valid=False`) are left unchanged
    so KL still sees independent stations on those slots.
    """

    def __init__(
        self,
        latent_dim: int,
        n_heads: int = Z_ATTN_HEADS,
        alpha_init: float = Z_ATTN_ALPHA_INIT,
        radius: int = Z_ATTN_RADIUS,
        alibi: float = Z_ATTN_ALIBI,
        gate_max: float = Z_ATTN_GATE_MAX,
    ):
        super().__init__()
        latent_dim = int(latent_dim)
        n_heads = int(n_heads)
        if n_heads < 1 or latent_dim % n_heads != 0:
            n_heads = 1
        self.latent_dim = latent_dim
        self.n_heads = n_heads
        self.head_dim = latent_dim // n_heads
        self.radius = max(0, int(radius))
        self.alibi = float(alibi)
        self.gate_max = float(gate_max)
        self.ln = nn.LayerNorm(latent_dim)
        qk_in = latent_dim + GAMMA_U_DIM
        self.w_q = nn.Linear(qk_in, latent_dim)
        self.w_k = nn.Linear(qk_in, latent_dim)
        self.w_v = nn.Linear(latent_dim, latent_dim)
        self.out = nn.Linear(latent_dim, latent_dim)
        _zero_residual_out(self.out)
        self.alpha_raw = nn.Parameter(torch.tensor(_logit(alpha_init)))

    def forward(self, z: Tensor, data) -> Tensor:
        if z.numel() == 0:
            return z
        n_graphs, latent_len, _ = z.shape
        token_u, token_tract, _, is_junc, valid = _token_tables(data, n_graphs, latent_len)
        is_junc = is_junc.to(device=z.device)
        token_tract = token_tract.to(device=z.device)
        token_u = token_u.to(device=z.device, dtype=z.dtype)
        valid = valid.to(device=z.device)
        junc = is_junc.bool() | (token_tract < 0) | ~valid
        z_n = self.ln(z)
        gamma_u = harmonic_encoding_u(token_u).view(n_graphs, latent_len, -1)
        qk = torch.cat([z_n, gamma_u], dim=-1)
        q = self.w_q(qk).view(n_graphs, latent_len, self.n_heads, self.head_dim)
        k = self.w_k(qk).view(n_graphs, latent_len, self.n_heads, self.head_dim)
        v = self.w_v(z).view(n_graphs, latent_len, self.n_heads, self.head_dim)
        mixed = z.new_zeros(n_graphs, latent_len, self.n_heads, self.head_dim)
        scale = math.sqrt(self.head_dim)
        for g in range(n_graphs):
            active = token_tract[g][~junc[g]]
            if active.numel() == 0:
                continue
            for tid in active.unique().tolist():
                if int(tid) < 0:
                    continue
                mask = (token_tract[g] == tid) & ~junc[g]
                qg = q[g, mask]
                if qg.size(0) == 0:
                    continue
                qg = qg.transpose(0, 1)
                kg = k[g, mask].transpose(0, 1)
                vg = v[g, mask].transpose(0, 1)
                scores = torch.matmul(qg, kg.transpose(-2, -1)) / scale
                ug = token_u[g, mask]
                order = torch.argsort(ug)
                rank = torch.empty_like(order)
                rank[order] = torch.arange(order.numel(), device=z.device)
                local = (rank.unsqueeze(0) - rank.unsqueeze(1)).abs() <= self.radius
                du = (ug.unsqueeze(0) - ug.unsqueeze(1)).abs()
                scores = scores.masked_fill(~local.unsqueeze(0), -1.0e9)
                scores = scores - self.alibi * du.unsqueeze(0)
                w = torch.softmax(scores, dim=-1)
                mixed[g, mask] = torch.matmul(w, vg).transpose(0, 1)
        delta = self.out(mixed.reshape(n_graphs, latent_len, self.latent_dim))
        out = z + _bounded_gate(self.alpha_raw, self.gate_max) * delta
        return torch.where(junc.unsqueeze(-1), z, out)


class ResidualSplineConv(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size=None,
        degree=None,
    ):
        super().__init__()
        if kernel_size is None:
            kernel_size = _spline_kernel_size()
        if isinstance(kernel_size, int):
            kernel_size = _spline_kernel_size()
        if degree is None:
            degree = _spline_degree()
        degree = int(degree)
        if int(min(kernel_size)) <= degree:
            raise ValueError(
                f"SplineConv degree {degree} requires kernel_size > degree on every "
                f"axis, got {tuple(kernel_size)}. A smaller axis makes that "
                "B-spline basis identical for every edge."
            )
        self.norm = nn.LayerNorm(dim)
        self.conv = make_spline_conv(
            dim,
            dim,
            dim=3,
            kernel_size=list(kernel_size),
            degree=int(degree),
            aggr=str(_cfg_get("SPLINE_AGGR", "mean")),
            root_weight=bool(_cfg_get("SPLINE_ROOT_WEIGHT", True)),
        )

    def forward(self, h: Tensor, edge_index: Tensor, pseudo: Tensor) -> Tensor:
        h = h.to(dtype=torch.float32)
        pseudo = pseudo.to(dtype=torch.float32)
        return h + F.elu(self.conv(self.norm(h), edge_index, pseudo))


class DecoupledDisplacementHead(nn.Module):
    """Δr / Δs head with r_local-relative floor and shear cap.

    Floor ≈ −RADIAL_FLOOR_FRAC · r_local; shear ≤ max(SHEAR_MAX_MM, k · r_local).
    Identity at init: W = 0 and a per-vertex softplus offset so Δr = 0 when
    features are zero. Linear bias stays 0.
    """

    def __init__(
        self,
        hidden: int,
        r_margin: float = R_MARGIN_MM,
        s_max: float = SHEAR_MAX_MM,
        floor_frac: float = RADIAL_FLOOR_FRAC,
        shear_rlocal_k: float = SHEAR_RLOCAL_K,
    ):
        super().__init__()
        self.r_margin = float(r_margin)
        self.s_max = float(s_max)
        self.floor_frac = float(floor_frac)
        self.shear_rlocal_k = float(shear_rlocal_k)
        self.radial = nn.Linear(hidden, 1)
        self.shear = nn.Linear(hidden, 2)
        nn.init.zeros_(self.radial.weight)
        nn.init.zeros_(self.radial.bias)
        nn.init.zeros_(self.shear.weight)
        nn.init.zeros_(self.shear.bias)

    def _bounds(self, h: Tensor, r_local: Tensor | None):
        n = h.size(0)
        if r_local is None:
            r = h.new_full((n, 1), self.r_margin)
        else:
            r = r_local.to(dtype=h.dtype, device=h.device).reshape(n, 1).clamp_min(1e-4)
        floor = self.floor_frac * r
        s_max = torch.maximum(r.new_full((n, 1), self.s_max), self.shear_rlocal_k * r)
        return floor, s_max

    def forward(self, h: Tensor, r_local: Tensor | None = None):
        h = h.to(dtype=torch.float32)
        floor, s_max = self._bounds(h, r_local)
        raw_r = self.radial(h)
        offset = torch.log(torch.expm1(floor.clamp_min(1e-6)))
        delta_r = F.softplus(raw_r + offset) - floor
        delta_s = torch.tanh(self.shear(h)) * s_max
        return delta_r, delta_s


class FreeDisplacementHead(nn.Module):
    """Unconstrained 3-D displacement (coarse level). Identity Δx = 0 at init."""

    def __init__(self, hidden: int):
        super().__init__()
        self.disp = nn.Linear(hidden, 3)
        nn.init.zeros_(self.disp.weight)
        nn.init.zeros_(self.disp.bias)

    def forward(self, h: Tensor) -> Tensor:
        return self.disp(h.to(dtype=torch.float32))


def _branch_nl(data, name: str, graph: int, num_graphs: int) -> Tensor:
    nl = getattr(data, name)
    if num_graphs == 1:
        return nl
    batch = getattr(data, f"{name}_batch", None)
    if batch is None:
        raise ValueError(f"Missing {name}_batch on batched Data")
    if nl.device != batch.device:
        batch = batch.to(device=nl.device)
    return nl[batch == graph]


def _graph_int(value, graph: int) -> int:
    if not torch.is_tensor(value):
        return int(value)
    if value.device.type != "cpu":
        value = value.detach().cpu()
    if value.dim() == 0 or value.numel() == 1:
        return int(value.reshape(-1)[0].item())
    return int(value[graph].item())


def _upsample_tensor_names(nr_src_name: str):
    if nr_src_name == "n_radial_coarse":
        return "upsample_idx_mid", "upsample_w_mid"
    if nr_src_name == "n_radial_mid":
        return "upsample_idx_fine", "upsample_w_fine"
    return None, None


def _upsample_level(
    delta: Tensor,
    data,
    nr_src_name: str,
    nr_dst_name: str,
    nl_src_name: str,
    nl_dst_name: str,
    src_batch: Tensor,
    dst_batch: Tensor,
    num_graphs: int,
) -> Tensor:
    idx_name, w_name = _upsample_tensor_names(nr_src_name)
    index = getattr(data, idx_name, None) if idx_name else None
    weight = getattr(data, w_name, None) if w_name else None
    n_dst = int(dst_batch.size(0))
    if (
        index is not None
        and weight is not None
        and torch.is_tensor(index)
        and torch.is_tensor(weight)
        and index.size(0) == n_dst
        and weight.size(0) == n_dst
    ):
        return knn_weighted_upsample(delta, index, weight)
    if num_graphs == 1:
        nr_s = _graph_int(getattr(data, nr_src_name), 0)
        nr_d = _graph_int(getattr(data, nr_dst_name), 0)
        nl_s = _branch_nl(data, nl_src_name, 0, num_graphs)
        nl_d = _branch_nl(data, nl_dst_name, 0, num_graphs)
        return upsample_branch_concat(delta, nl_s, nr_s, nl_d, nr_d)
    out = delta.new_zeros(dst_batch.size(0), delta.size(-1))
    for g in range(num_graphs):
        src_mask = src_batch == g
        dst_mask = dst_batch == g
        nr_s = _graph_int(getattr(data, nr_src_name), g)
        nr_d = _graph_int(getattr(data, nr_dst_name), g)
        nl_s = _branch_nl(data, nl_src_name, g, num_graphs)
        nl_d = _branch_nl(data, nl_dst_name, g, num_graphs)
        out[dst_mask] = upsample_branch_concat(delta[src_mask], nl_s, nr_s, nl_d, nr_d)
    return out


class ProgressiveSplineDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        latent_len: int = LATENT_LEN,
        hidden_dim: int = DECODER_HIDDEN_DIM,
        attn_dim: int = ATTN_DIM,
        r_margin: float = R_MARGIN_MM,
        s_max: float = SHEAR_MAX_MM,
        gradient_checkpointing: str = "off",
        n_conv: int | None = None,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.latent_len = int(latent_len)
        self.r_margin = float(r_margin)
        self.gradient_checkpointing = normalize_gradient_checkpointing(gradient_checkpointing)
        n_conv = int(n_conv if n_conv is not None else _n_conv_per_level())
        self.cross_coarse = LatentCrossAttention(latent_dim, hidden_dim, attn_dim, latent_len)
        self.cross_mid = LatentCrossAttention(latent_dim, hidden_dim, attn_dim, latent_len)
        self.cross_fine = LatentCrossAttention(latent_dim, hidden_dim, attn_dim, latent_len)
        self.geom_fuse_c = nn.Linear(hidden_dim + 4, hidden_dim)
        self.geom_fuse_m = nn.Linear(hidden_dim + 4, hidden_dim)
        self.geom_fuse_f = nn.Linear(hidden_dim + 4, hidden_dim)
        self.coarse_attn = CoarsePositionalSelfAttention(hidden_dim)
        self.coarse_convs = nn.ModuleList(
            [ResidualSplineConv(hidden_dim) for _ in range(n_conv)]
        )
        self.mid_convs = nn.ModuleList(
            [ResidualSplineConv(hidden_dim) for _ in range(n_conv)]
        )
        self.fine_convs = nn.ModuleList(
            [ResidualSplineConv(hidden_dim) for _ in range(n_conv)]
        )
        self.mid_init = nn.Linear(3, hidden_dim)
        self.fine_init = nn.Linear(3, hidden_dim)
        self.alpha_c_raw = nn.Parameter(torch.tensor(_logit(SKIP_GATE_INIT)))
        self.alpha_m_raw = nn.Parameter(torch.tensor(_logit(SKIP_GATE_INIT)))
        self.coarse_head = FreeDisplacementHead(hidden_dim)
        self.mid_head = DecoupledDisplacementHead(hidden_dim, r_margin=r_margin, s_max=s_max)
        self.head = DecoupledDisplacementHead(hidden_dim, r_margin=r_margin, s_max=s_max)

    def _run_convs(self, h, edge_index, pseudo, convs, checkpoint_ok):
        use_ckpt = self.training and bool(checkpoint_ok)
        for conv in convs:
            h = _ckpt_call(use_ckpt, conv, h, edge_index, pseudo)
        return h

    def _run_attn(self, h, u, theta, tract, batch, u_step, n_graphs):
        attn = self.coarse_attn

        def _fn(h_in, u_in, th_in, tr_in, b_in, us_in):
            return attn(h_in, u_in, th_in, tr_in, b_in, us_in, n_graphs)

        return _ckpt_call(
            self.training and self.gradient_checkpointing == "all",
            _fn,
            h,
            u,
            theta,
            tract,
            batch,
            u_step,
        )

    def _cross(
        self,
        layer,
        z,
        u,
        theta,
        node_batch,
        tract,
        token_u,
        token_attend,
        latent_valid,
        token_tract,
        token_pos,
        node_pos,
    ):
        return layer(
            z, u, theta, node_batch, tract, token_u, token_attend,
            latent_valid=latent_valid,
            token_tract=token_tract,
            token_pos=token_pos,
            node_pos=node_pos,
        )

    def _fuse_geom(self, h, fuse, data, n, level):
        geom = _level_geom_features(data, n, level, self.r_margin)
        geom = geom.to(device=h.device, dtype=h.dtype)
        return fuse(torch.cat([h, geom], dim=-1))

    def forward(self, z: Tensor, data):
        n_graphs = z.size(0)
        token_u, token_tract, token_attend, _, latent_valid = _token_tables(
            data, n_graphs, self.latent_len
        )
        token_pos = _token_pos_table(data, n_graphs, self.latent_len)
        if token_pos is not None:
            token_pos = token_pos.to(device=z.device, dtype=z.dtype)
        token_u = token_u.to(device=z.device, dtype=z.dtype)
        token_tract = token_tract.to(device=z.device)
        latent_valid = latent_valid.to(device=z.device)

        pos_c = data.pos_coarse
        batch_c = _attr_batch(data, "pos_coarse", pos_c.size(0))
        pos_m = data.pos_mid
        batch_m = _attr_batch(data, "pos_mid", pos_m.size(0))
        pos_f = data.x
        batch_f = data.batch if getattr(data, "batch", None) is not None else _ones_batch(
            pos_f.size(0), pos_f.device
        )

        h_c = self._cross(
            self.cross_coarse, z, data.u_coarse, data.theta_coarse, batch_c,
            data.tract_id_coarse, token_u, token_attend, latent_valid, token_tract,
            token_pos, pos_c,
        )
        h_c = self._fuse_geom(h_c, self.geom_fuse_c, data, pos_c.size(0), "coarse")
        h_c = self._run_attn(
            h_c, data.u_coarse, data.theta_coarse, data.tract_id_coarse, batch_c,
            data.u_step_coarse, n_graphs,
        )
        r_c = _level_r_local(data, pos_c.size(0), "coarse", self.r_margin)
        pseudo_c = intrinsic_spline_pseudo_coords(
            data.u_coarse, data.theta_coarse, data.tract_id_coarse,
            data.edge_index_coarse, data.u_step_coarse,
            r_local=r_c, pos=pos_c,
        )
        h_c = self._run_convs(
            h_c, data.edge_index_coarse, pseudo_c, self.coarse_convs,
            self.gradient_checkpointing == "all",
        )
        dx_c = self.coarse_head(h_c)
        x_c = apply_boundary_plane_projection(pos_c + dx_c, data, suffix="_coarse")
        dx_c = x_c - pos_c

        dx_m0 = _upsample_level(
            dx_c, data,
            "n_radial_coarse", "n_radial_mid",
            "branch_nl_coarse", "branch_nl_mid",
            batch_c, batch_m, n_graphs,
        )
        h_c_up = _upsample_level(
            h_c, data,
            "n_radial_coarse", "n_radial_mid",
            "branch_nl_coarse", "branch_nl_mid",
            batch_c, batch_m, n_graphs,
        )
        h_m = self._cross(
            self.cross_mid, z, data.u_mid, data.theta_mid, batch_m,
            data.tract_id_mid, token_u, token_attend, latent_valid, token_tract,
            token_pos, pos_m,
        ) + self.mid_init(dx_m0) + torch.sigmoid(self.alpha_c_raw) * h_c_up
        h_m = self._fuse_geom(h_m, self.geom_fuse_m, data, pos_m.size(0), "mid")
        r_m = _level_r_local(data, pos_m.size(0), "mid", self.r_margin)
        pseudo_m = intrinsic_spline_pseudo_coords(
            data.u_mid, data.theta_mid, data.tract_id_mid,
            data.edge_index_mid, data.u_step_mid,
            r_local=r_m, pos=pos_m,
        )
        h_m = self._run_convs(
            h_m, data.edge_index_mid, pseudo_m, self.mid_convs,
            self.gradient_checkpointing == "all",
        )
        floor_m = RADIAL_FLOOR_FRAC * r_m
        dr_m, ds_m = self.mid_head(h_m, r_local=r_m)
        dr_m = _clamp_residual_radial(dr_m, dx_m0, data.normal_mid, floor_m)
        dx_m = dx_m0 + decoupled_displacement(
            dr_m, ds_m, data.normal_mid, data.tangent_mid, data.binormal_mid
        )
        x_m = apply_boundary_plane_projection(pos_m + dx_m, data, suffix="_mid")
        dx_m = x_m - pos_m

        dx_f0 = _upsample_level(
            dx_m, data,
            "n_radial_mid", "n_radial_fine",
            "branch_nl_mid", "branch_nl_fine",
            batch_m, batch_f, n_graphs,
        )
        h_m_up = _upsample_level(
            h_m, data,
            "n_radial_mid", "n_radial_fine",
            "branch_nl_mid", "branch_nl_fine",
            batch_m, batch_f, n_graphs,
        )
        h_f = self._cross(
            self.cross_fine, z, data.u, data.theta, batch_f,
            data.tract_id, token_u, token_attend, latent_valid, token_tract,
            token_pos, pos_f,
        ) + self.fine_init(dx_f0) + torch.sigmoid(self.alpha_m_raw) * h_m_up
        h_f = self._fuse_geom(h_f, self.geom_fuse_f, data, pos_f.size(0), "fine")
        r_f = _level_r_local(data, pos_f.size(0), "fine", self.r_margin)
        pseudo_f = intrinsic_spline_pseudo_coords(
            data.u, data.theta, data.tract_id, data.edge_index, data.u_step,
            r_local=r_f, pos=pos_f,
        )
        h_f = self._run_convs(
            h_f, data.edge_index, pseudo_f, self.fine_convs,
            self.gradient_checkpointing in ("all", "fine"),
        )
        floor_f = RADIAL_FLOOR_FRAC * r_f
        delta_r, delta_s = self.head(h_f, r_local=r_f)
        delta_r = _clamp_residual_radial(delta_r, dx_f0, data.normal, floor_f)
        dx_decoupled = decoupled_displacement(
            delta_r, delta_s, data.normal, data.tangent, data.binormal
        )
        delta_x = dx_f0 + dx_decoupled
        x_pred = apply_boundary_plane_projection(pos_f + delta_x, data, suffix="")
        delta_x = x_pred - pos_f
        return x_pred, delta_x, x_c, x_m, delta_r, delta_s, dx_c


@dataclass
class VAEOutput:
    x_pred: Tensor
    mu: Tensor
    logvar: Tensor
    z: Tensor
    delta_x: Tensor
    delta_r: Tensor
    delta_s: Tensor
    x_pred_coarse: Tensor
    x_pred_mid: Tensor
    mu_raw: Tensor | None = None
    logvar_raw: Tensor | None = None
    sampled: bool = False
    delta_x_coarse: Tensor | None = None


class GraphVAE(nn.Module):
    """Stage-2 deformation VAE. `decode(z, data)` maps latent Z and a centerline scaffold to a surface."""

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        latent_len: int = LATENT_LEN,
        hidden_dim: int = DECODER_HIDDEN_DIM,
        k: int = 32,
        tube_radius: float = TUBE_RADIUS_MM,
        r_margin: Optional[float] = None,
        sa_stages=None,
        gradient_checkpointing=None,
        **kwargs,
    ):
        super().__init__()
        del k, kwargs
        self.latent_dim = int(latent_dim)
        self.latent_len = int(latent_len)
        self.gradient_checkpointing = normalize_gradient_checkpointing(gradient_checkpointing)
        r_margin = float(tube_radius if r_margin is None else r_margin)
        self.encoder = PointNeXtEncoder(
            latent_dim=self.latent_dim,
            latent_len=self.latent_len,
            stages=sa_stages or SA_STAGES,
            gradient_checkpointing=self.gradient_checkpointing,
            narrow_late_stages=sa_stages is None,
        )
        self.decoder = ProgressiveSplineDecoder(
            latent_dim=self.latent_dim,
            latent_len=self.latent_len,
            hidden_dim=hidden_dim,
            r_margin=r_margin,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        self.z_attn = LatentTractSelfAttention(self.latent_dim)

    def reparameterize(self, mu: Tensor, logvar: Tensor, sample: bool | None = None) -> Tensor:
        """If `sample` is True, z = μ + σ·ε even in eval. If False, return μ.

        `sample=None` falls back to `self.training` so existing train.py calls
        keep working until they pass the flag explicitly.
        """
        if sample is None:
            sample = bool(self.training)
        if not sample:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def encode(self, data):
        return self.encoder(data)

    def mix_posterior(self, mu: Tensor, logvar: Tensor, data):
        """Apply the tract mixer to μ before sampling. logvar is already local/soft-bounded."""
        mu_mixed = self.z_attn(mu, data)
        return mu_mixed, logvar

    def decode(self, z: Tensor, data) -> Tensor:
        """Stage-2 decoder: map latent Z and scaffold `data` to surface coordinates.

        Does not run the mixer — Stage 1 / `forward` already produce decoder-ready codes.
        """
        x_pred, *_ = self.decoder(z, data)
        return x_pred

    def forward(self, data, sample: bool | None = None) -> VAEOutput:
        if sample is None:
            sample = bool(self.training)
        mu_raw, logvar_raw = self.encode(data)
        mu, logvar = self.mix_posterior(mu_raw, logvar_raw, data)
        z = self.reparameterize(mu, logvar, sample=sample)
        unpacked = self.decoder(z, data)
        x_pred, delta_x, x_coarse, x_mid, delta_r, delta_s = unpacked[:6]
        dx_c = unpacked[6] if len(unpacked) > 6 else None
        return VAEOutput(
            x_pred=x_pred,
            mu=mu,
            logvar=logvar,
            z=z,
            delta_x=delta_x,
            delta_r=delta_r,
            delta_s=delta_s,
            x_pred_coarse=x_coarse,
            x_pred_mid=x_mid,
            mu_raw=mu_raw,
            logvar_raw=logvar_raw,
            sampled=bool(sample),
            delta_x_coarse=dx_c,
        )
