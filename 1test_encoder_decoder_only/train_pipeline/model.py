"""Hierarchical PointNeXt encoder + progressive SplineConv decoder VAE.

Encoder follows PointNeXt (Qian et al., NeurIPS 2022): stem MLP, FPS set
abstraction, radius grouping with Δp / r, and inverted-residual MLP blocks.
The decoder is the spec's geometry-aware progressive SplineConv deformer
with a 1D centerline latent trajectory.
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
    EDGE_MAX_MM,
    GAMMA_THETA_DIM,
    GAMMA_U_DIM,
    INVRES_ALPHA_INIT,
    INVRES_EXPANSION,
    LATENT_DIM,
    LATENT_LEN,
    LOGVAR_CLAMP,
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
    radial_bias_for_zero_init,
    spline_pseudo_coords,
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


def nearest_centerline_u(
    pos: Tensor,
    pos_batch: Tensor,
    cl_dense: Tensor,
    cl_batch: Tensor,
) -> Tensor:
    """Assign each point the arc-length u of its nearest dense centerline sample."""
    u_out = pos.new_zeros(pos.size(0))
    n_graphs = _num_graphs(pos_batch)
    for g in range(n_graphs):
        p = pos[pos_batch == g]
        c = cl_dense[cl_batch == g]
        if p.numel() == 0 or c.numel() == 0:
            continue
        d = torch.cdist(p, c[:, :3])
        idx = d.argmin(dim=1)
        u_out[pos_batch == g] = c[idx, 3]
    return u_out


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
    """Cross-attention from L centerline queries onto the coarsest PointNeXt tokens."""

    def __init__(self, in_dim: int, latent_dim: int, latent_len: int, attn_dim: int):
        super().__init__()
        self.latent_len = int(latent_len)
        self.attn_dim = int(attn_dim)
        self.w_q = nn.Linear(GAMMA_U_DIM, attn_dim)
        self.w_k = nn.Linear(in_dim + GAMMA_U_DIM, attn_dim)
        self.w_v = nn.Linear(in_dim, attn_dim)
        self.mu_head = nn.Linear(attn_dim, latent_dim)
        self.logvar_head = nn.Linear(attn_dim, latent_dim)

    def forward(self, h: Tensor, u_pts: Tensor, batch: Tensor):
        n_graphs = _num_graphs(batch)
        device = h.device
        u_q = torch.linspace(0.0, 1.0, self.latent_len, device=device)
        q = self.w_q(harmonic_encoding_u(u_q))
        mu_out, lv_out = [], []
        scale = math.sqrt(self.attn_dim)
        for g in range(n_graphs):
            mask = batch == g
            hg = h[mask]
            ug = u_pts[mask]
            kg = self.w_k(torch.cat([hg, harmonic_encoding_u(ug)], dim=-1))
            vg = self.w_v(hg)
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
        attn_dim: int = ATTN_DIM,
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

    def forward(self, x_true: Tensor, x_true_batch: Tensor, cl_dense: Tensor, cl_batch: Tensor):
        h = self.stem(x_true)
        pos, batch = x_true, x_true_batch
        for sa, inv_blocks in zip(self.sa_layers, self.inv_layers):
            h, pos, batch = sa(h, pos, batch)
            for blk in inv_blocks:
                if self.training:
                    h = checkpoint(blk, h, pos, batch, use_reentrant=False)
                else:
                    h = blk(h, pos, batch)
        u_pts = nearest_centerline_u(pos, batch, cl_dense, cl_batch)
        return self.latent_head(h, u_pts, batch)


class LatentCrossAttention(nn.Module):
    """Scaffold nodes query the 1D latent trajectory via Fourier (u, θ)."""

    def __init__(self, latent_dim: int, hidden_dim: int, attn_dim: int, latent_len: int):
        super().__init__()
        self.latent_len = int(latent_len)
        self.attn_dim = int(attn_dim)
        self.w_q = nn.Linear(GAMMA_U_DIM + GAMMA_THETA_DIM, attn_dim)
        self.w_k = nn.Linear(latent_dim + GAMMA_U_DIM, attn_dim)
        self.w_v = nn.Linear(latent_dim, attn_dim)
        self.out = nn.Linear(attn_dim + GAMMA_U_DIM + GAMMA_THETA_DIM, hidden_dim)

    def forward(self, z: Tensor, u: Tensor, theta: Tensor, node_batch: Tensor) -> Tensor:
        gamma_u = harmonic_encoding_u(u)
        gamma_th = harmonic_encoding_theta(theta)
        q = self.w_q(torch.cat([gamma_u, gamma_th], dim=-1))
        n_graphs = z.size(0)
        device = z.device
        u_k = torch.linspace(0.0, 1.0, self.latent_len, device=device)
        gamma_uk = harmonic_encoding_u(u_k).unsqueeze(0).expand(n_graphs, -1, -1)
        k = self.w_k(torch.cat([z, gamma_uk], dim=-1))
        v = self.w_v(z)
        k_n = k[node_batch]
        v_n = v[node_batch]
        scale = math.sqrt(self.attn_dim)
        scores = (q.unsqueeze(1) * k_n).sum(-1) / scale
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
        # pyg-lib spline CUDA kernels are float32-only; bf16 autocast illegal-accesses.
        device_type = "cuda" if h.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            h32 = h.float()
            out = h32 + F.elu(self.conv(h32, edge_index, pseudo.float()))
        return out.to(dtype=h.dtype)


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
        r_edge_max: float = EDGE_MAX_MM,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.r_edge_max = float(r_edge_max)
        self.cross_coarse = LatentCrossAttention(latent_dim, hidden_dim, attn_dim, latent_len)
        self.cross_mid = LatentCrossAttention(latent_dim, hidden_dim, attn_dim, latent_len)
        self.coarse_convs = nn.ModuleList(
            [ResidualSplineConv(hidden_dim) for _ in range(N_SPLINE_COARSE)]
        )
        self.mid_convs = nn.ModuleList(
            [ResidualSplineConv(hidden_dim) for _ in range(N_SPLINE_MID)]
        )
        self.fine_convs = nn.ModuleList(
            [ResidualSplineConv(hidden_dim) for _ in range(N_SPLINE_FINE)]
        )
        self.coarse_delta = nn.Linear(hidden_dim, 3)
        self.mid_delta = nn.Linear(hidden_dim, 3)
        self.fine_in = nn.Linear(3 + GAMMA_U_DIM + GAMMA_THETA_DIM, hidden_dim)
        self.mid_init = nn.Linear(3, hidden_dim)
        nn.init.zeros_(self.coarse_delta.weight)
        nn.init.zeros_(self.coarse_delta.bias)
        nn.init.zeros_(self.mid_delta.weight)
        nn.init.zeros_(self.mid_delta.bias)
        self.head = DecoupledDisplacementHead(hidden_dim, r_margin=r_margin, s_max=s_max)

    def _run_convs(self, h, edge_index, pseudo, convs):
        for conv in convs:
            if self.training:
                h = checkpoint(conv, h, edge_index, pseudo, use_reentrant=False)
            else:
                h = conv(h, edge_index, pseudo)
        return h

    def forward(self, z: Tensor, data):
        n_graphs = z.size(0)
        pos_c = data.pos_coarse
        batch_c = _attr_batch(data, "pos_coarse", pos_c.size(0))
        pos_m = data.pos_mid
        batch_m = _attr_batch(data, "pos_mid", pos_m.size(0))
        pos_f = data.x
        batch_f = data.batch if getattr(data, "batch", None) is not None else _ones_batch(
            pos_f.size(0), pos_f.device
        )

        h_c = self.cross_coarse(z, data.u_coarse, data.theta_coarse, batch_c)
        pseudo_c = spline_pseudo_coords(pos_c, data.edge_index_coarse, self.r_edge_max)
        h_c = self._run_convs(h_c, data.edge_index_coarse, pseudo_c, self.coarse_convs)
        dx_c = self.coarse_delta(h_c)

        dx_m0 = _upsample_level(
            dx_c, data,
            "n_radial_coarse", "n_radial_mid",
            "branch_nl_coarse", "branch_nl_mid",
            batch_c, batch_m, n_graphs,
        )
        h_m = self.cross_mid(z, data.u_mid, data.theta_mid, batch_m) + self.mid_init(dx_m0)
        pseudo_m = spline_pseudo_coords(pos_m, data.edge_index_mid, self.r_edge_max)
        h_m = self._run_convs(h_m, data.edge_index_mid, pseudo_m, self.mid_convs)
        dx_m = dx_m0 + self.mid_delta(h_m)

        dx_f0 = _upsample_level(
            dx_m, data,
            "n_radial_mid", "n_radial_fine",
            "branch_nl_mid", "branch_nl_fine",
            batch_m, batch_f, n_graphs,
        )
        h_f = self.fine_in(
            torch.cat(
                [dx_f0, harmonic_encoding_u(data.u), harmonic_encoding_theta(data.theta)],
                dim=-1,
            )
        )
        pseudo_f = spline_pseudo_coords(pos_f, data.edge_index, self.r_edge_max)
        h_f = self._run_convs(h_f, data.edge_index, pseudo_f, self.fine_convs)
        delta_r, delta_s = self.head(h_f)
        dx_decoupled = decoupled_displacement(
            delta_r, delta_s, data.normal, data.tangent, data.binormal
        )
        delta_x = dx_f0 + dx_decoupled
        x_pred = pos_f + delta_x
        return x_pred, delta_x, pos_c + dx_c, pos_m + dx_m


@dataclass
class VAEOutput:
    x_pred: Tensor
    mu: Tensor
    logvar: Tensor
    z: Tensor
    delta_x: Tensor
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
        x_true_batch = (
            data.x_true_batch
            if getattr(data, "x_true_batch", None) is not None
            else _ones_batch(data.x_true.size(0), data.x_true.device)
        )
        cl_batch = _attr_batch(data, "cl_dense", data.cl_dense.size(0))
        return self.encoder(data.x_true, x_true_batch, data.cl_dense, cl_batch)

    def decode(self, z: Tensor, data) -> Tensor:
        """Deterministic frozen-decoder forward for Stage-2 diffusion inference."""
        x_pred, _, _, _ = self.decoder(z, data)
        return x_pred

    def forward(self, data) -> VAEOutput:
        mu, logvar = self.encode(data)
        z = self.reparameterize(mu, logvar)
        x_pred, delta_x, x_coarse, x_mid = self.decoder(z, data)
        return VAEOutput(
            x_pred=x_pred,
            mu=mu,
            logvar=logvar,
            z=z,
            delta_x=delta_x,
            x_pred_coarse=x_coarse,
            x_pred_mid=x_mid,
        )
