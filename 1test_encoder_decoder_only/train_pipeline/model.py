"""Hierarchical PointNeXt encoder + progressive SplineConv decoder VAE.

Encoder follows PointNeXt (Qian et al., NeurIPS 2022): stem MLP, FPS set
abstraction, radius grouping with Δp / r, and inverted-residual MLP blocks.
The decoder is a geometry-aware progressive SplineConv deformer with a
tree-valued centerline latent.
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
    SPLINE_DEGREE,
    SPLINE_KERNEL_SIZE,
    STEM_DIM,
    TUBE_RADIUS_MM,
)
from geometry import (
    decoupled_displacement,
    harmonic_encoding_theta,
    harmonic_encoding_u,
    intrinsic_spline_pseudo_coords,
    radial_bias_for_zero_init,
    upsample_branch_concat,
)
from ops import ball_query_packed, fps_indices, make_spline_conv, radius_graph_packed


def _num_graphs(batch: Tensor) -> int:
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


def fps_packed(pos: Tensor, batch: Tensor, n_out: int) -> Tensor:
    """Farthest-point sample a packed cloud to exactly `n_out` points per graph."""
    n_graphs = _num_graphs(batch)
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
):
    """Nearest dense centerline sample → (u_local, tract_id) for each point."""
    u_out = pos.new_zeros(pos.size(0))
    t_out = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)
    n_graphs = _num_graphs(pos_batch)
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

    def forward(self, h: Tensor, pos: Tensor, batch: Tensor):
        idx = fps_packed(pos, batch, self.n_out)
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
        n_graphs = _num_graphs(batch)
        token_u, token_tract, _, _ = _token_tables(data, n_graphs, self.latent_len)
        scale = math.sqrt(self.attn_dim)
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
            h, pos, batch = sa(h, pos, batch)
            for blk in inv_blocks:
                if self.training:
                    h = checkpoint(blk, h, pos, batch, use_reentrant=False)
                else:
                    h = blk(h, pos, batch)
        u_pts, tract_pts = nearest_centerline_attr(
            pos, batch, data.cl_dense, cl_batch, cl_tract
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
        k_n = k[node_batch]
        v_n = v[node_batch]
        scale = math.sqrt(self.attn_dim)
        scores = (q.unsqueeze(1) * k_n).sum(-1) / scale
        attend_n = token_attend[node_batch]
        tract_idx = node_tract.clamp(0, MAX_TRACTS - 1).view(-1, 1, 1).expand(-1, self.latent_len, 1)
        allow = attend_n.gather(2, tract_idx).squeeze(-1)
        scores = scores.masked_fill(~allow, -1.0e4)
        orphan = ~allow.any(dim=-1)
        if orphan.any():
            scores = scores.clone()
            scores[orphan, 0] = 0.0
        w = torch.softmax(scores, dim=-1)
        a = (w.unsqueeze(-1) * v_n).sum(1)
        return self.out(torch.cat([a, gamma_u, gamma_th], dim=-1))


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
    batch = getattr(data, f"{name}_batch", None)
    if batch is None:
        if num_graphs == 1:
            return nl
        raise ValueError(f"Missing {name}_batch on batched Data")
    return nl[batch == graph]


def _graph_int(value: Tensor, graph: int) -> int:
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
        self.cross_coarse = LatentCrossAttention(latent_dim, hidden_dim, attn_dim, latent_len)
        self.cross_mid = LatentCrossAttention(latent_dim, hidden_dim, attn_dim, latent_len)
        self.cross_fine = LatentCrossAttention(latent_dim, hidden_dim, attn_dim, latent_len)
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
        h_m = self._cross(
            self.cross_mid, z, data.u_mid, data.theta_mid, batch_m,
            data.tract_id_mid, token_u, token_attend,
        ) + self.mid_init(dx_m0)
        pseudo_m = intrinsic_spline_pseudo_coords(
            data.u_mid, data.theta_mid, data.tract_id_mid,
            data.edge_index_mid, data.u_step_mid,
        )
        h_m = self._run_convs(h_m, data.edge_index_mid, pseudo_m, self.mid_convs)
        dr_m, ds_m = self.mid_head(h_m)
        dx_m = dx_m0 + decoupled_displacement(
            dr_m, ds_m, data.normal_mid, data.tangent_mid, data.binormal_mid
        )

        dx_f0 = _upsample_level(
            dx_m, data,
            "n_radial_mid", "n_radial_fine",
            "branch_nl_mid", "branch_nl_fine",
            batch_m, batch_f, n_graphs,
        )
        h_f = self._cross(
            self.cross_fine, z, data.u, data.theta, batch_f,
            data.tract_id, token_u, token_attend,
        ) + self.fine_init(dx_f0)
        pseudo_f = intrinsic_spline_pseudo_coords(
            data.u, data.theta, data.tract_id, data.edge_index, data.u_step,
        )
        h_f = self._run_convs(h_f, data.edge_index, pseudo_f, self.fine_convs)
        delta_r, delta_s = self.head(h_f)
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
    """Deformation VAE. `decode(z, data)` is the frozen Stage-2 inference contract."""

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

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def encode(self, data):
        return self.encoder(data)

    def decode(self, z: Tensor, data) -> Tensor:
        """Deterministic frozen-decoder forward for Stage-2 diffusion inference."""
        x_pred, _, _, _, _, _ = self.decoder(z, data)
        return x_pred

    def forward(self, data) -> VAEOutput:
        mu, logvar = self.encode(data)
        z = self.reparameterize(mu, logvar)
        x_pred, delta_x, x_coarse, x_mid, delta_r, delta_s = self.decoder(z, data)
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
