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
    LOGVAR_CLAMP,
    MAX_TRACTS,
    N_SPLINE_COARSE,
    N_SPLINE_FINE,
    N_SPLINE_MID,
    R_MARGIN_MM,
    SA_STAGES,
    SHEAR_MAX_MM,
    SKIP_GATE_INIT,
    SPLINE_DEGREE,
    SPLINE_KERNEL_SIZE,
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
)
from geometry import (
    clamp_residual_radial,
    decoupled_displacement,
    harmonic_encoding_theta,
    harmonic_encoding_u,
    intrinsic_spline_pseudo_coords,
    radial_bias_for_zero_init,
    upsample_branch_concat,
)
from ops import ball_query_packed, fps_indices, make_spline_conv, radius_graph_packed


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


def _token_tables(data, n_graphs: int, latent_len: int):
    """Return [B, L] token descriptors from a possibly batched Data object."""
    u = data.latent_u
    tract = data.latent_tract_id
    attend = data.token_attend
    is_junc = data.latent_is_junction
    if attend.dtype != torch.bool:
        attend = attend.bool()
    if u.dim() == 2 and u.size(0) == n_graphs:
        return u, tract, attend, is_junc
    if getattr(data, "latent_u_batch", None) is not None or u.numel() == n_graphs * latent_len:
        u = u.reshape(n_graphs, latent_len)
        tract = tract.reshape(n_graphs, latent_len)
        is_junc = is_junc.reshape(n_graphs, latent_len)
        attend = attend.reshape(n_graphs, latent_len, attend.size(-1))
        return u, tract, attend, is_junc
    u = u.reshape(1, -1)
    tract = tract.reshape(1, -1)
    is_junc = is_junc.reshape(1, -1)
    attend = attend.reshape(1, attend.size(0), attend.size(-1))
    if n_graphs != 1:
        raise ValueError("Missing latent_u_batch for batched tree tokens")
    return u, tract, attend, is_junc


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


class CenterlineLatentHead(nn.Module):
    """Cross-attention from tree-valued centerline queries onto PointNeXt tokens."""

    def __init__(self, in_dim: int, latent_dim: int, latent_len: int, attn_dim: int):
        super().__init__()
        self.latent_len = int(latent_len)
        self.attn_dim = int(attn_dim)
        self.tract_emb = nn.Embedding(MAX_TRACTS + 1, attn_dim)
        self.w_q = nn.Linear(GAMMA_U_DIM + attn_dim, attn_dim)
        self.w_k = nn.Linear(in_dim + GAMMA_U_DIM + attn_dim, attn_dim)
        self.w_v = nn.Linear(in_dim, attn_dim)
        self.mu_head = nn.Linear(attn_dim, latent_dim)
        self.logvar_head = nn.Linear(attn_dim, latent_dim)

    def _tract_index(self, tract_id: Tensor) -> Tensor:
        idx = tract_id.clone()
        idx = torch.where(idx < 0, torch.full_like(idx, MAX_TRACTS), idx)
        return idx.clamp(0, MAX_TRACTS)

    def forward(self, h: Tensor, u_pts: Tensor, tract_pts: Tensor, batch: Tensor, data):
        n_graphs = int(getattr(data, "num_graphs", 1) or 1)
        token_u, token_tract, _, _ = _token_tables(data, n_graphs, self.latent_len)
        scale = math.sqrt(self.attn_dim)
        if n_graphs == 1:
            tg = self.tract_emb(self._tract_index(tract_pts))
            kg = self.w_k(torch.cat([h, harmonic_encoding_u(u_pts), tg], dim=-1))
            vg = self.w_v(h)
            tq = self.tract_emb(self._tract_index(token_tract[0]))
            q = self.w_q(torch.cat([harmonic_encoding_u(token_u[0]), tq], dim=-1))
            attn = torch.softmax(q @ kg.t() / scale, dim=-1) @ vg
            return self.mu_head(attn).unsqueeze(0), self.logvar_head(attn).clamp(*LOGVAR_CLAMP).unsqueeze(0)
        mu_out, lv_out = [], []
        for g in range(n_graphs):
            mask = batch == g
            hg = h[mask]
            ug = u_pts[mask]
            tg = self.tract_emb(self._tract_index(tract_pts[mask]))
            kg = self.w_k(torch.cat([hg, harmonic_encoding_u(ug), tg], dim=-1))
            vg = self.w_v(hg)
            uq = token_u[g]
            tq = self.tract_emb(self._tract_index(token_tract[g]))
            q = self.w_q(torch.cat([harmonic_encoding_u(uq), tq], dim=-1))
            attn = torch.softmax(q @ kg.t() / scale, dim=-1) @ vg
            mu_out.append(self.mu_head(attn))
            lv_out.append(self.logvar_head(attn).clamp(*LOGVAR_CLAMP))
        return torch.stack(mu_out, dim=0), torch.stack(lv_out, dim=0)


class PointNeXtEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        latent_len: int = LATENT_LEN,
        stem_dim: int = STEM_DIM,
        stages=SA_STAGES,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.latent_len = int(latent_len)
        stages = tuple(stages)
        self.stem = nn.Sequential(
            nn.Linear(3, stem_dim),
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
        self.latent_head = CenterlineLatentHead(in_dim, latent_dim, latent_len, attn_dim=in_dim)

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
        h = self.stem(x_true)
        pos, batch = x_true, x_true_batch
        for sa, inv_blocks in zip(self.sa_layers, self.inv_layers):
            h, pos, batch = sa(h, pos, batch, n_graphs=n_graphs)
            for blk in inv_blocks:
                if self.training:
                    h = checkpoint(blk, h, pos, batch, use_reentrant=False)
                else:
                    h = blk(h, pos, batch)
        u_pts, tract_pts = nearest_centerline_attr(
            pos, batch, data.cl_dense, cl_batch, cl_tract, n_graphs=n_graphs
        )
        return self.latent_head(h, u_pts, tract_pts, batch, data)


class LatentCrossAttention(nn.Module):
    """Scaffold nodes query the tree latent via Fourier (u, θ) with a tract mask."""

    def __init__(self, latent_dim: int, hidden_dim: int, attn_dim: int, latent_len: int):
        super().__init__()
        self.latent_len = int(latent_len)
        self.attn_dim = int(attn_dim)
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
        # Per-graph GEMM: scores = q_g @ k[g].T, a = softmax @ v[g].
        # Avoids materializing [N, L, D] broadcasts of k/v (~6 GiB at the fine scaffold).
        if n_graphs == 1:
            scores = q.matmul(k[0].transpose(0, 1)) / scale
            allow = token_attend[0][:, tract].transpose(0, 1)
            scores = scores.masked_fill(~allow, -1.0e4)
            orphan = ~allow.any(dim=-1)
            scores = torch.where(orphan.unsqueeze(-1), torch.zeros_like(scores), scores)
            a = torch.softmax(scores, dim=-1).matmul(v[0])
            return self.out(torch.cat([a, gamma_u, gamma_th], dim=-1))
        a = q.new_empty(q.size(0), self.attn_dim)
        for g in range(n_graphs):
            mask = node_batch == g
            qg = q[mask]
            scores = qg.matmul(k[g].transpose(0, 1)) / scale
            allow = token_attend[g][:, tract[mask]].transpose(0, 1)
            scores = scores.masked_fill(~allow, -1.0e4)
            orphan = ~allow.any(dim=-1)
            scores = torch.where(orphan.unsqueeze(-1), torch.zeros_like(scores), scores)
            w = torch.softmax(scores, dim=-1)
            a[mask] = w.matmul(v[g])
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
    """Mild per-tract residual mix of latent tokens after reparameterization.

    Q/K concatenate [LN(z), γ(u)]; values come from z. Junction tokens
    (tract_id < 0) are left unchanged so KL still sees independent stations.
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
        token_u, token_tract, _, is_junc = _token_tables(data, n_graphs, latent_len)
        is_junc = is_junc.to(device=z.device)
        token_tract = token_tract.to(device=z.device)
        token_u = token_u.to(device=z.device, dtype=z.dtype)
        junc = is_junc.bool() | (token_tract < 0)
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
        kernel_size: int = SPLINE_KERNEL_SIZE,
        degree: int = SPLINE_DEGREE,
    ):
        super().__init__()
        self.conv = make_spline_conv(
            dim,
            dim,
            dim=3,
            kernel_size=kernel_size,
            degree=degree,
            aggr="add",
            root_weight=False,
        )

    def forward(self, h: Tensor, edge_index: Tensor, pseudo: Tensor) -> Tensor:
        h = h.to(dtype=torch.float32)
        pseudo = pseudo.to(dtype=torch.float32)
        return h + F.elu(self.conv(h, edge_index, pseudo))


class DecoupledDisplacementHead(nn.Module):
    def __init__(self, hidden: int, r_margin: float = R_MARGIN_MM, s_max: float = SHEAR_MAX_MM):
        super().__init__()
        self.r_margin = float(r_margin)
        self.s_max = float(s_max)
        self.radial = nn.Linear(hidden, 1)
        self.shear = nn.Linear(hidden, 2)
        nn.init.zeros_(self.radial.weight)
        self.radial.bias.data.fill_(radial_bias_for_zero_init(self.r_margin))
        nn.init.zeros_(self.shear.weight)
        nn.init.zeros_(self.shear.bias)

    def forward(self, h: Tensor):
        h = h.to(dtype=torch.float32)
        delta_r = F.softplus(self.radial(h)) - self.r_margin
        delta_s = torch.tanh(self.shear(h)) * self.s_max
        return delta_r, delta_s


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
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.latent_len = int(latent_len)
        self.r_margin = float(r_margin)
        self.cross_coarse = LatentCrossAttention(latent_dim, hidden_dim, attn_dim, latent_len)
        self.cross_mid = LatentCrossAttention(latent_dim, hidden_dim, attn_dim, latent_len)
        self.cross_fine = LatentCrossAttention(latent_dim, hidden_dim, attn_dim, latent_len)
        self.coarse_attn = CoarsePositionalSelfAttention(hidden_dim)
        self.coarse_convs = nn.ModuleList(
            [ResidualSplineConv(hidden_dim) for _ in range(N_SPLINE_COARSE)]
        )
        self.mid_convs = nn.ModuleList(
            [ResidualSplineConv(hidden_dim) for _ in range(N_SPLINE_MID)]
        )
        self.fine_convs = nn.ModuleList(
            [ResidualSplineConv(hidden_dim) for _ in range(N_SPLINE_FINE)]
        )
        self.mid_init = nn.Linear(3, hidden_dim)
        self.fine_init = nn.Linear(3, hidden_dim)
        self.alpha_c_raw = nn.Parameter(torch.tensor(_logit(SKIP_GATE_INIT)))
        self.alpha_m_raw = nn.Parameter(torch.tensor(_logit(SKIP_GATE_INIT)))
        self.coarse_head = DecoupledDisplacementHead(hidden_dim, r_margin=r_margin, s_max=s_max)
        self.mid_head = DecoupledDisplacementHead(hidden_dim, r_margin=r_margin, s_max=s_max)
        self.head = DecoupledDisplacementHead(hidden_dim, r_margin=r_margin, s_max=s_max)

    def _run_convs(self, h, edge_index, pseudo, convs):
        for conv in convs:
            if self.training:
                h = checkpoint(conv, h, edge_index, pseudo, use_reentrant=False)
            else:
                h = conv(h, edge_index, pseudo)
        return h

    def _run_attn(self, h, u, theta, tract, batch, u_step, n_graphs):
        attn = self.coarse_attn

        def _fn(h_in, u_in, th_in, tr_in, b_in, us_in):
            return attn(h_in, u_in, th_in, tr_in, b_in, us_in, n_graphs)

        if self.training:
            return checkpoint(_fn, h, u, theta, tract, batch, u_step, use_reentrant=False)
        return attn(h, u, theta, tract, batch, u_step, n_graphs)

    def _cross(self, layer, z, u, theta, node_batch, tract, token_u, token_attend):
        return layer(z, u, theta, node_batch, tract, token_u, token_attend)

    def forward(self, z: Tensor, data):
        n_graphs = z.size(0)
        token_u, _, token_attend, _ = _token_tables(data, n_graphs, self.latent_len)

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
            data.tract_id_coarse, token_u, token_attend,
        )
        h_c = self._run_attn(
            h_c, data.u_coarse, data.theta_coarse, data.tract_id_coarse, batch_c,
            data.u_step_coarse, n_graphs,
        )
        pseudo_c = intrinsic_spline_pseudo_coords(
            data.u_coarse, data.theta_coarse, data.tract_id_coarse,
            data.edge_index_coarse, data.u_step_coarse,
        )
        h_c = self._run_convs(h_c, data.edge_index_coarse, pseudo_c, self.coarse_convs)
        dr_c, ds_c = self.coarse_head(h_c)
        dx_c = decoupled_displacement(dr_c, ds_c, data.normal_coarse, data.tangent_coarse, data.binormal_coarse)

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
            data.tract_id_mid, token_u, token_attend,
        ) + self.mid_init(dx_m0) + torch.sigmoid(self.alpha_c_raw) * h_c_up
        pseudo_m = intrinsic_spline_pseudo_coords(
            data.u_mid, data.theta_mid, data.tract_id_mid,
            data.edge_index_mid, data.u_step_mid,
        )
        h_m = self._run_convs(h_m, data.edge_index_mid, pseudo_m, self.mid_convs)
        dr_m, ds_m = self.mid_head(h_m)
        dr_m = clamp_residual_radial(dr_m, dx_m0, data.normal_mid, self.r_margin)
        dx_m = dx_m0 + decoupled_displacement(
            dr_m, ds_m, data.normal_mid, data.tangent_mid, data.binormal_mid
        )

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
            data.tract_id, token_u, token_attend,
        ) + self.fine_init(dx_f0) + torch.sigmoid(self.alpha_m_raw) * h_m_up
        pseudo_f = intrinsic_spline_pseudo_coords(
            data.u, data.theta, data.tract_id, data.edge_index, data.u_step,
        )
        h_f = self._run_convs(h_f, data.edge_index, pseudo_f, self.fine_convs)
        delta_r, delta_s = self.head(h_f)
        delta_r = clamp_residual_radial(delta_r, dx_f0, data.normal, self.r_margin)
        dx_decoupled = decoupled_displacement(
            delta_r, delta_s, data.normal, data.tangent, data.binormal
        )
        delta_x = dx_f0 + dx_decoupled
        x_pred = pos_f + delta_x
        return x_pred, delta_x, pos_c + dx_c, pos_m + dx_m, delta_r, delta_s


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
        **kwargs,
    ):
        super().__init__()
        del k, kwargs
        self.latent_dim = int(latent_dim)
        self.latent_len = int(latent_len)
        r_margin = float(tube_radius if r_margin is None else r_margin)
        self.encoder = PointNeXtEncoder(
            latent_dim=latent_dim, latent_len=latent_len, stages=sa_stages or SA_STAGES
        )
        self.decoder = ProgressiveSplineDecoder(
            latent_dim=latent_dim,
            latent_len=latent_len,
            hidden_dim=hidden_dim,
            r_margin=r_margin,
        )
        self.z_attn = LatentTractSelfAttention(self.latent_dim)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def encode(self, data):
        return self.encoder(data)

    def decode(self, z: Tensor, data) -> Tensor:
        """Stage-2 decoder: map latent Z and scaffold `data` to surface coordinates."""
        z = self.z_attn(z, data)
        x_pred, _, _, _, _, _ = self.decoder(z, data)
        return x_pred

    def forward(self, data) -> VAEOutput:
        mu, logvar = self.encode(data)
        z = self.reparameterize(mu, logvar)
        z_dec = self.z_attn(z, data)
        x_pred, delta_x, x_coarse, x_mid, delta_r, delta_s = self.decoder(z_dec, data)
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
        )
