"""Smoke tests for the hierarchical PointNeXt–SplineConv VAE."""

from __future__ import annotations

import math
import os
import shutil
import sys
import tempfile
import traceback

import numpy as np
import pyvista as pv
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CACHE_VERSION,
    FOLLOW_BATCH,
    GAMMA_THETA_DIM,
    GAMMA_U_DIM,
    K_THETA,
    K_U,
    LAMBDA_CD_COARSE,
    LOGVAR_CLAMP,
    MAX_TRACTS,
    SA_STAGES,
    configure_stage2_precision,
)
from dataset import (
    AneurysmDataset,
    allocate_ring_counts,
    allocate_token_counts,
    extract_unique_tracts,
)
from geometry import (
    bilinear_cylindrical_upsample,
    clamp_residual_radial,
    fps_metric,
    harmonic_encoding_theta,
    harmonic_encoding_u,
    intrinsic_spline_pseudo_coords,
    radial_bias_for_zero_init,
    upsample_branch_concat,
)
from losses import (
    _cl_radius,
    _mesh_normal_consistency,
    _uniform_laplacian_smoothing,
    _weighted_chamfer,
    compute_losses,
    displacement_dirichlet,
    displacement_dirichlet_local,
    vae_kl_loss,
)
from model import DecoupledDisplacementHead, GraphVAE, LatentCrossAttention
from ops import fps_indices, make_spline_conv
from train import kl_anneal_weight

TINY_HIERARCHY = ((8, 4), (16, 8), (32, 16))
TINY_SA = ((32, 4.0, 8, 32, 1), (8, 8.0, 8, 64, 1))


def _make_factory(radius=2.0, n_true=64, hierarchy=TINY_HIERARCHY, latent_len=8):
    ds = AneurysmDataset.__new__(AneurysmDataset)
    ds.tube_radius = float(radius)
    ds.n_true = int(n_true)
    ds.latent_len = int(latent_len)
    ds.hierarchy = tuple(tuple(lv) for lv in hierarchy)
    ds.n_length = int(ds.hierarchy[-1][0])
    ds.n_radial = int(ds.hierarchy[-1][1])
    ds.samples = []
    ds.cache_dir = None
    return ds


def _straight_branch(n=24, length=20.0, offset=(3.0, -2.0, 5.0)):
    z = np.linspace(0.0, length, n)
    pts = np.stack([np.zeros(n), np.zeros(n), z], axis=1).astype(np.float64)
    return pts + np.asarray(offset, dtype=np.float64)


def _polyline_mesh(pts):
    pts = np.asarray(pts, dtype=np.float64)
    if hasattr(pv, "lines_from_points"):
        return pv.lines_from_points(pts)
    n = len(pts)
    lines = np.hstack(([n], np.arange(n, dtype=np.int64)))
    return pv.PolyData(pts, lines=lines)


def _polylines_mesh(paths):
    meshes = [_polyline_mesh(p) for p in paths]
    out = meshes[0]
    for m in meshes[1:]:
        out = out.merge(m)
    return out


def _y_paths():
    parent = np.stack([np.zeros(12), np.zeros(12), np.linspace(0.0, 12.0, 12)], axis=1)
    child_a = np.stack([np.linspace(0.0, 8.0, 10), np.zeros(10), np.full(10, 12.0)], axis=1)
    child_b = np.stack([np.zeros(10), np.linspace(0.0, 8.0, 10), np.full(10, 12.0)], axis=1)
    path1 = np.concatenate([parent, child_a[1:]], axis=0)
    path2 = np.concatenate([parent, child_b[1:]], axis=0)
    return path1, path2


def _tube_cloud(polyline, radius=2.2, n_u=40, n_th=16, sac=False):
    poly = np.asarray(polyline, dtype=np.float64)
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= 1e-12:
        xyz = np.repeat(poly[:1], n_u, axis=0)
    else:
        u = np.linspace(0.0, 1.0, n_u)
        xyz = np.stack([np.interp(u, cum / total, poly[:, d]) for d in range(3)], axis=1)
    th = np.linspace(0.0, 2.0 * np.pi, n_th, endpoint=False)
    pts = []
    for p in xyz:
        for a in th:
            pts.append(p + radius * np.array([np.cos(a), np.sin(a), 0.0]))
    pts = np.asarray(pts, dtype=np.float32)
    if sac:
        center = xyz[len(xyz) // 2] + np.array([6.0, 0.0, 0.0])
        sac_pts = center + 0.4 * np.random.RandomState(0).randn(48, 3)
        pts = np.concatenate([pts, sac_pts.astype(np.float32)], axis=0)
    return pts


def make_synthetic_data(
    n_true=64,
    hierarchy=TINY_HIERARCHY,
    latent_len=8,
    radius=2.0,
    sac=True,
):
    factory = _make_factory(
        radius=radius, n_true=n_true, hierarchy=hierarchy, latent_len=latent_len
    )
    branch = _straight_branch()
    mesh = _polyline_mesh(branch)
    vessel = _tube_cloud(branch, sac=sac)
    return factory.build_scaffold(mesh, vessel_points=vessel)


def _tiny_model(latent_len=8, latent_dim=8):
    return GraphVAE(
        latent_dim=latent_dim,
        latent_len=latent_len,
        hidden_dim=16,
        tube_radius=2.0,
        sa_stages=TINY_SA,
    )


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_config_contracts():
    _assert(SA_STAGES[-1][0] == 64, f"last SA n_out {SA_STAGES[-1][0]}")
    _assert(LOGVAR_CLAMP == (-8.0, 2.0), LOGVAR_CLAMP)
    _assert(LAMBDA_CD_COARSE <= 0.05 + 1e-12, LAMBDA_CD_COARSE)
    _assert(CACHE_VERSION >= 4, CACHE_VERSION)


def test_fourier_shapes():
    u = torch.linspace(0, 1, 11)
    th = torch.linspace(-math.pi, math.pi, 13)[:-1]
    gu = harmonic_encoding_u(u)
    gt = harmonic_encoding_theta(th)
    _assert(gu.dtype == torch.float32 and gt.dtype == torch.float32, "Fourier dtype")
    _assert(gu.shape == (11, GAMMA_U_DIM), f"γ(u) shape {gu.shape}")
    _assert(gt.shape == (12, GAMMA_THETA_DIM), f"γ(θ) shape {gt.shape}")
    _assert(torch.isfinite(gu).all() and torch.isfinite(gt).all(), "non-finite Fourier")
    u_hi = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
    th_hi = torch.tensor([-math.pi, 0.0, math.pi], dtype=torch.float32)
    _assert(torch.isfinite(harmonic_encoding_u(u_hi)).all(), "γ(u) max-band")
    _assert(torch.isfinite(harmonic_encoding_theta(th_hi)).all(), "γ(θ) max-band")
    th0 = torch.tensor([-math.pi, math.pi - 1e-6])
    gt0 = harmonic_encoding_theta(th0)
    _assert((gt0[0] - gt0[1]).abs().max() < 1e-3, "θ encoding should be nearly 2π-periodic")
    _assert(K_U == 8 and K_THETA == 6, "frequency band counts")


def test_kl_and_anneal():
    mu = torch.zeros(2, 8, 4)
    logvar = torch.zeros(2, 8, 4)
    _assert(float(vae_kl_loss(mu, logvar)) < 1e-6, "KL(prior) should be 0")
    mu2 = torch.ones(2, 8, 4)
    _assert(float(vae_kl_loss(mu2, logvar)) > 0, "KL(shifted) should be > 0")
    _assert(abs(kl_anneal_weight(1, 5e-4, 20)) < 1e-12, "KL starts at 0")
    _assert(abs(kl_anneal_weight(20, 5e-4, 20) - 5e-4) < 1e-12, "KL reaches max at warmup")
    _assert(abs(kl_anneal_weight(10, 5e-4, 20) - 5e-4 * 9 / 19) < 1e-12, "KL mid schedule")


def test_pseudo_coords_range():
    data = make_synthetic_data(sac=False)
    e = intrinsic_spline_pseudo_coords(
        data.u, data.theta, data.tract_id, data.edge_index, data.u_step
    )
    _assert(e.dtype == torch.float32, "pseudo dtype")
    _assert(e.min() >= 0.0 and e.max() <= 1.0, f"pseudo coords out of [0,1]: {e.min()}, {e.max()}")
    src, dst = data.edge_index
    same = data.tract_id[src] == data.tract_id[dst]
    _assert(bool(same.all()), "synthetic single tract should have no cross-tract edges")
    _assert(torch.allclose(e[:, 2], torch.full((e.size(0),), 0.5)), "same-tract kind channel")


def test_bilinear_identity_and_wrap():
    nl, nr, c = 5, 8, 3
    field = torch.arange(nl * nr * c, dtype=torch.float32).reshape(nl * nr, c)
    out = bilinear_cylindrical_upsample(field, nl, nr, nl, nr)
    _assert(torch.allclose(out, field, atol=1e-5), "identity upsample failed")

    src = torch.zeros(2, 4, 1)
    src[0, 0, 0] = 1.0
    src[0, 3, 0] = 1.0
    up = bilinear_cylindrical_upsample(src.reshape(-1, 1), 2, 4, 2, 8)
    up = up.reshape(2, 8)
    _assert(up[0, 0] > 0.4 and up[0, 7] > 0.4, f"θ wrap failed: {up[0]}")

    concat = upsample_branch_concat(field, torch.tensor([nl]), nr, torch.tensor([nl]), nr)
    _assert(torch.allclose(concat, field, atol=1e-5), "branch concat identity failed")


def test_bishop_frames_orthonormal():
    factory = _make_factory()
    tube = factory._generate_branch_tube(_straight_branch(offset=(0.0, 0.0, 0.0)), 12, 6)
    n_v, t_v, b_v = tube["n_v"], tube["t_v"], tube["b_v"]
    _assert(n_v.dtype == np.float64, f"Bishop n dtype {n_v.dtype}")
    _assert(np.allclose(np.linalg.norm(n_v, axis=1), 1.0, atol=1e-5), "n not unit")
    _assert(np.allclose(np.linalg.norm(t_v, axis=1), 1.0, atol=1e-5), "t not unit")
    _assert(np.allclose(np.linalg.norm(b_v, axis=1), 1.0, atol=1e-5), "b not unit")
    _assert(np.allclose((n_v * t_v).sum(1), 0.0, atol=1e-4), "n·t")
    _assert(np.allclose((n_v * b_v).sum(1), 0.0, atol=1e-4), "n·b")
    _assert(np.allclose((t_v * b_v).sum(1), 0.0, atol=1e-4), "t·b")
    cross = np.cross(n_v, t_v)
    _assert(np.allclose(np.abs((cross * b_v).sum(1)), 1.0, atol=1e-4), "frame not orthonormal triad")
    _assert(tube["theta"].min() >= -np.pi - 1e-6 and tube["theta"].max() < np.pi + 1e-6, "θ range")

    dense = factory._fit_dense_tract(_straight_branch(offset=(0.0, 0.0, 0.0)))
    level = factory._generate_level([dense], 12, 6, [float(dense["arc"])])
    n, t, b = level["normal"], level["tangent"], level["binormal"]
    _assert(n.dtype == torch.float32, "packed normal dtype")
    _assert(float((n * t).sum(-1).abs().max()) < 1e-4, "packed n·t")
    _assert(float((n * b).sum(-1).abs().max()) < 1e-4, "packed n·b")
    _assert(torch.allclose(n.norm(dim=-1), torch.ones(n.size(0)), atol=1e-4), "packed ||n||")


def test_fps_count():
    pts = np.random.RandomState(0).randn(200, 3).astype(np.float32)
    out = fps_metric(pts, 32)
    _assert(out.shape == (32, 3), f"FPS shape {out.shape}")
    short = fps_metric(pts[:10], 32)
    _assert(short.shape == (32, 3), "FPS pad failed")


def test_fps_cuda_path():
    pts = torch.randn(128, 3)
    idx_cpu = fps_indices(pts, 16)
    _assert(idx_cpu.numel() == 16, idx_cpu.shape)
    _assert(idx_cpu.min() >= 0 and idx_cpu.max() < 128, "CPU FPS out of range")
    start = int((pts - pts.mean(0)).pow(2).sum(-1).argmax())
    _assert(int(idx_cpu[0]) == start, "FPS start")
    if not torch.cuda.is_available():
        return
    pts_g = pts.cuda()
    idx_g = fps_indices(pts_g, 16)
    _assert(idx_g.device.type == "cuda", f"expected CUDA indices, got {idx_g.device}")
    _assert(idx_g.numel() == 16, idx_g.shape)
    _assert(int(idx_g.max()) < 128, "CUDA FPS out of range")
    _assert(int(idx_g[0]) == start, "CUDA FPS start")


def test_ball_query_index_order():
    from ops import ball_query_packed

    support = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    query = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
    sb = torch.zeros(4, dtype=torch.long)
    qb = torch.zeros(2, dtype=torch.long)
    got = ball_query_packed(support, query, 1.5, sb, qb, 32)
    _assert(int(got[0].max()) < 4, f"support idx {got[0]}")
    _assert(int(got[1].max()) < 2, f"query idx {got[1]}")
    pairs = set(zip(got[0].tolist(), got[1].tolist()))
    _assert(pairs == {(0, 0), (1, 0), (2, 1), (3, 1)}, f"{got}")
    if torch.cuda.is_available():
        got_g = ball_query_packed(support.cuda(), query.cuda(), 1.5, sb.cuda(), qb.cuda(), 32)
        _assert(got_g.device.type == "cuda", got_g.device)
        _assert(int(got_g[0].max()) < 4 and int(got_g[1].max()) < 2, got_g)


def test_radius_all_in_ball():
    from ops import ball_query_packed

    ang = torch.linspace(0, 2 * math.pi, 9)[:-1]
    support = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)
    query = torch.zeros(1, 2)
    sb = torch.zeros(8, dtype=torch.long)
    qb = torch.zeros(1, dtype=torch.long)
    ei = ball_query_packed(support, query, 1.1, sb, qb, 256)
    _assert(ei.size(1) == 8, f"expected all 8 neighbors, got {ei.size(1)}")
    ei_cap = ball_query_packed(support, query, 1.1, sb, qb, 3)
    _assert(ei_cap.size(1) == 3, f"cap should keep 3 nearest, got {ei_cap.size(1)}")


def test_allocate_rings():
    cases = [
        (40, [10.0, 10.0]),
        (250, [1.0, 2.0, 3.0]),
        (1000, [5.0, 5.0, 5.0]),
        (10, [1.0, 1.0, 1.0, 1.0, 1.0]),
    ]
    for n_len, arcs in cases:
        alloc = allocate_ring_counts(n_len, arcs)
        _assert(len(alloc) == len(arcs), alloc)
        _assert(sum(alloc) == n_len, f"ring counts {alloc} sum to {sum(alloc)} != {n_len}")
        _assert(min(alloc) >= 1, alloc)
    alloc, n_junc = allocate_token_counts(64, [10.0, 10.0, 5.0], 2)
    _assert(sum(alloc) + n_junc == 64, (alloc, n_junc))
    _assert(n_junc == 2, n_junc)


def test_decoupled_head_no_inversion():
    head = DecoupledDisplacementHead(hidden=8, r_margin=2.0, s_max=3.0)
    h = torch.zeros(16, 8)
    dr, ds = head(h)
    _assert(torch.allclose(dr, torch.zeros_like(dr), atol=1e-4), f"Δr init {dr.mean()}")
    _assert(torch.allclose(ds, torch.zeros_like(ds), atol=1e-4), "Δs init")
    h2 = 50 * torch.randn(32, 8)
    dr2, ds2 = head(h2)
    _assert((dr2 > -2.0 - 1e-5).all(), f"Δr inverted: min={dr2.min()}")
    _assert((ds2.abs() <= 3.0 + 1e-5).all(), "shear clamp")
    _assert(abs(radial_bias_for_zero_init(2.0) - math.log(math.expm1(2.0))) < 1e-8, "bias formula")


def test_residual_radial_floor():
    n = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    dx_up = torch.tensor([[-1.5, 0.0, 0.0], [0.0, 0.5, 0.0]])
    dr = torch.tensor([[-1.8], [-1.8]])
    out = clamp_residual_radial(dr, dx_up, n, 2.0)
    _assert(torch.allclose(out[0], torch.tensor([-0.5]), atol=1e-5), f"inward floor {out[0]}")
    _assert(torch.allclose(out[1], torch.tensor([-1.8]), atol=1e-5), f"outward unchanged {out[1]}")
    composed = (n * (dx_up + out * n)).sum(dim=-1)
    _assert(bool((composed >= -2.0 - 1e-5).all()), f"composed radial {composed}")

    head = DecoupledDisplacementHead(hidden=4, r_margin=2.0, s_max=3.0)
    dr_h, _ = head(50 * torch.randn(2, 4))
    dr_c = clamp_residual_radial(dr_h, dx_up, n, 2.0)
    composed_h = (n * (dx_up + dr_c * n)).sum(dim=-1)
    _assert(bool((composed_h >= -2.0 - 1e-5).all()), f"head+clamp radial {composed_h}")


def test_decoder_composed_non_inversion():
    data = make_synthetic_data()
    loader = DataLoader([data], batch_size=1, follow_batch=FOLLOW_BATCH)
    batch = next(iter(loader))
    model = _tiny_model()
    model.eval()
    for head in (model.decoder.coarse_head, model.decoder.mid_head, model.decoder.head):
        torch.nn.init.zeros_(head.radial.weight)
        head.radial.bias.data.fill_(-80.0)
        torch.nn.init.zeros_(head.shear.weight)
        torch.nn.init.zeros_(head.shear.bias)
    z = torch.zeros(1, 8, 8)
    with torch.no_grad():
        _, delta_x, x_c, x_m, _, _ = model.decoder(z, batch)
    r_c = (batch.normal_coarse * (x_c - batch.pos_coarse)).sum(-1)
    r_m = (batch.normal_mid * (x_m - batch.pos_mid)).sum(-1)
    r_f = (batch.normal * delta_x).sum(-1)
    _assert(bool((r_c >= -2.0 - 1e-3).all()), f"coarse composed Δr min={float(r_c.min())}")
    _assert(bool((r_m >= -2.0 - 1e-3).all()), f"mid composed Δr min={float(r_m.min())}")
    _assert(bool((r_f >= -2.0 - 1e-3).all()), f"fine composed Δr min={float(r_f.min())}")


def test_orphan_attention_uniform():
    torch.manual_seed(0)
    latent_len, latent_dim, attn_dim, hidden = 5, 4, 8, 8
    layer = LatentCrossAttention(latent_dim, hidden, attn_dim, latent_len)
    layer.eval()
    n = 3
    z = torch.randn(1, latent_len, latent_dim)
    u = torch.full((n,), 0.4)
    theta = torch.zeros(n)
    node_batch = torch.zeros(n, dtype=torch.long)
    node_tract = torch.zeros(n, dtype=torch.long)
    token_u = torch.linspace(0, 1, latent_len).unsqueeze(0)
    token_attend = torch.zeros(1, latent_len, MAX_TRACTS, dtype=torch.bool)
    with torch.no_grad():
        out = layer(z, u, theta, node_batch, node_tract, token_u, token_attend)
        gamma_u = harmonic_encoding_u(u)
        gamma_th = harmonic_encoding_theta(theta)
        v = layer.w_v(z)[0]
        a = v.mean(dim=0, keepdim=True).expand(n, -1)
        expected = layer.out(torch.cat([a, gamma_u, gamma_th], dim=-1))
    _assert(torch.allclose(out, expected, atol=1e-5), "orphan queries must attend uniformly")


def _broadcast_latent_cross_attn(layer, z, u, theta, node_batch, node_tract, token_u, token_attend):
    """Old [N, L, D] broadcast formula; reference only (tiny N)."""
    gamma_u = harmonic_encoding_u(u)
    gamma_th = harmonic_encoding_theta(theta)
    q = layer.w_q(torch.cat([gamma_u, gamma_th], dim=-1))
    n_graphs = z.size(0)
    gamma_uk = harmonic_encoding_u(token_u.reshape(-1)).reshape(n_graphs, layer.latent_len, -1)
    k = layer.w_k(torch.cat([z, gamma_uk], dim=-1))
    v = layer.w_v(z)
    k_n = k[node_batch]
    v_n = v[node_batch]
    scale = math.sqrt(layer.attn_dim)
    scores = (q.unsqueeze(1) * k_n).sum(-1) / scale
    attend_n = token_attend[node_batch]
    tract_idx = node_tract.clamp(0, MAX_TRACTS - 1).view(-1, 1, 1).expand(-1, layer.latent_len, 1)
    allow = attend_n.gather(2, tract_idx).squeeze(-1)
    scores = scores.masked_fill(~allow, -1.0e4)
    orphan = ~allow.any(dim=-1)
    if orphan.any():
        scores = scores.clone()
        scores[orphan] = 0.0
    w = torch.softmax(scores, dim=-1)
    a = (w.unsqueeze(-1) * v_n).sum(1)
    return layer.out(torch.cat([a, gamma_u, gamma_th], dim=-1))


def test_cross_attention_gemm_matches_broadcast():
    torch.manual_seed(2)
    latent_len, latent_dim, attn_dim, hidden = 6, 4, 8, 8
    layer = LatentCrossAttention(latent_dim, hidden, attn_dim, latent_len)
    layer.eval()
    n0, n1 = 20, 12
    n = n0 + n1
    z = torch.randn(2, latent_len, latent_dim)
    u = torch.rand(n)
    theta = torch.rand(n) * 2 * math.pi - math.pi
    node_batch = torch.cat([torch.zeros(n0), torch.ones(n1)]).long()
    node_tract = torch.randint(0, 3, (n,))
    token_u = torch.rand(2, latent_len)
    token_attend = torch.zeros(2, latent_len, MAX_TRACTS, dtype=torch.bool)
    token_attend[0, :, 0] = True
    token_attend[0, :3, 1] = True
    token_attend[1, :, 1] = True
    token_attend[1, 2:, 2] = True
    with torch.no_grad():
        got = layer(z, u, theta, node_batch, node_tract, token_u, token_attend)
        ref = _broadcast_latent_cross_attn(
            layer, z, u, theta, node_batch, node_tract, token_u, token_attend
        )
    _assert(got.shape == ref.shape, f"shape {got.shape} vs {ref.shape}")
    _assert(torch.allclose(got, ref, atol=1e-5, rtol=1e-5), "GEMM path must match broadcast scores")


def test_cross_attention_two_graph_isolation():
    torch.manual_seed(3)
    latent_len, latent_dim, attn_dim, hidden = 6, 4, 8, 8
    layer = LatentCrossAttention(latent_dim, hidden, attn_dim, latent_len)
    layer.eval()
    n0, n1 = 18, 14
    z = torch.randn(2, latent_len, latent_dim)
    u = torch.rand(n0 + n1)
    theta = torch.rand(n0 + n1) * 2 * math.pi - math.pi
    node_batch = torch.cat([torch.zeros(n0), torch.ones(n1)]).long()
    node_tract = torch.cat([torch.zeros(n0), torch.ones(n1)]).long()
    token_u = torch.rand(2, latent_len)
    token_attend = torch.zeros(2, latent_len, MAX_TRACTS, dtype=torch.bool)
    token_attend[0, :, 0] = True
    token_attend[1, :, 1] = True
    with torch.no_grad():
        batched = layer(z, u, theta, node_batch, node_tract, token_u, token_attend)
        out0 = layer(
            z[0:1], u[:n0], theta[:n0],
            torch.zeros(n0, dtype=torch.long), node_tract[:n0],
            token_u[0:1], token_attend[0:1],
        )
        out1 = layer(
            z[1:2], u[n0:], theta[n0:],
            torch.zeros(n1, dtype=torch.long), node_tract[n0:],
            token_u[1:2], token_attend[1:2],
        )
    _assert(torch.allclose(batched[:n0], out0, atol=1e-5, rtol=1e-5), "graph 0 mixed with graph 1")
    _assert(torch.allclose(batched[n0:], out1, atol=1e-5, rtol=1e-5), "graph 1 mixed with graph 0")


def test_knn_chamfer_matches_cdist():
    torch.manual_seed(1)
    pred = torch.randn(17, 3)
    true = torch.randn(9, 3)
    pred_batch = torch.zeros(17, dtype=torch.long)
    true_batch = torch.zeros(9, dtype=torch.long)
    w_pred = torch.linspace(0.5, 1.5, 17)
    w_true = torch.linspace(0.8, 1.2, 9)
    knn_loss = _weighted_chamfer(pred, pred_batch, true, true_batch, w_pred, w_true, 1)
    dist = torch.cdist(pred, true, p=2).pow(2)
    min_true = dist.min(dim=1).values
    min_pred = dist.min(dim=0).values
    cdist_loss = 0.5 * (
        (w_pred * min_true).sum() / w_pred.sum()
        + (w_true * min_pred).sum() / w_true.sum()
    )
    _assert(torch.allclose(knn_loss, cdist_loss, atol=1e-4), f"chamfer {knn_loss} vs {cdist_loss}")

    cl = torch.randn(11, 4)
    rad_knn = _cl_radius(pred, cl)
    rad_cd = torch.cdist(pred, cl[:, :3]).min(dim=1).values
    _assert(torch.allclose(rad_knn, rad_cd, atol=1e-4), f"cl radius {rad_knn[:3]} vs {rad_cd[:3]}")


def test_mesh_losses_match_pytorch3d():
    from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency
    from pytorch3d.structures import Meshes

    data = make_synthetic_data()
    torch.manual_seed(2)
    verts = data.x + 0.05 * torch.randn_like(data.x)
    batch = torch.zeros(verts.size(0), dtype=torch.long)
    face = data.face
    meshes = Meshes(verts=[verts], faces=[face.t().contiguous()])
    lap_ref = mesh_laplacian_smoothing(meshes, method="uniform")
    norm_ref = mesh_normal_consistency(meshes)
    lap = _uniform_laplacian_smoothing(verts, face, batch, 1)
    nrm = _mesh_normal_consistency(verts, face, batch, 1)
    _assert(torch.allclose(lap, lap_ref, atol=1e-5, rtol=1e-4), f"lap {float(lap)} vs {float(lap_ref)}")
    _assert(torch.allclose(nrm, norm_ref, atol=1e-4, rtol=1e-3), f"norm {float(nrm)} vs {float(norm_ref)}")
    shifted = verts + torch.tensor([1.5, -0.7, 2.0])
    lap_shift = _uniform_laplacian_smoothing(shifted, face, batch, 1)
    _assert(torch.allclose(lap, lap_shift, atol=1e-5), "uniform Laplacian should ignore rigid translation")

    d1 = make_synthetic_data()
    d2 = make_synthetic_data(sac=True)
    loader = DataLoader([d1, d2], batch_size=2, follow_batch=FOLLOW_BATCH)
    packed = next(iter(loader))
    lap_b = _uniform_laplacian_smoothing(packed.x, packed.face, packed.batch, packed.num_graphs)
    nrm_b = _mesh_normal_consistency(packed.x, packed.face, packed.batch, packed.num_graphs)
    refs_lap, refs_nrm = [], []
    for i, di in enumerate((d1, d2)):
        m = Meshes(verts=[di.x], faces=[di.face.t().contiguous()])
        refs_lap.append(mesh_laplacian_smoothing(m, method="uniform"))
        refs_nrm.append(mesh_normal_consistency(m))
    _assert(torch.allclose(lap_b, 0.5 * (refs_lap[0] + refs_lap[1]), atol=1e-5, rtol=1e-4), "batched lap")
    _assert(torch.allclose(nrm_b, 0.5 * (refs_nrm[0] + refs_nrm[1]), atol=1e-4, rtol=1e-3), "batched norm")


def test_spline_conv_backend():
    conv = make_spline_conv(4, 4, dim=3, kernel_size=5, degree=2, root_weight=False)
    x = torch.randn(6, 4)
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]])
    attr = torch.rand(3, 3)
    y = conv(x, ei, attr)
    _assert(y.shape == (6, 4), y.shape)


def test_dirichlet_zero_on_rigid():
    delta = torch.ones(5, 3)
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]])
    _assert(float(displacement_dirichlet(delta, ei)) < 1e-8, "uniform Δx should have zero energy")
    dr = torch.ones(5, 1)
    ds = torch.zeros(5, 2)
    _assert(float(displacement_dirichlet_local(dr, ds, ei)) < 1e-8, "uniform Δr should have zero energy")


def test_unique_tracts_from_overlapping_paths():
    mesh = _polylines_mesh(list(_y_paths()))
    tracts, endpoints, junctions = extract_unique_tracts(mesh)
    _assert(len(tracts) == 3, f"expected 3 unique tracts, got {len(tracts)}")
    _assert(len(endpoints) == 3, endpoints)
    _assert(len(junctions) >= 1, "expected a junction node")


def test_tree_token_mask():
    factory = _make_factory(latent_len=16)
    paths = _y_paths()
    data = factory.build_scaffold(_polylines_mesh(list(paths)), vessel_points=_tube_cloud(paths[0]))
    n_tracts = int(data.n_tracts)
    _assert(n_tracts >= 2, f"Y-junction should have ≥2 tracts, got {n_tracts}")
    _assert(tuple(data.token_attend.shape) == (16, MAX_TRACTS), data.token_attend.shape)
    attend = data.token_attend.cpu().numpy()
    tract_ids = data.latent_tract_id.cpu().numpy()
    is_junc = data.latent_is_junction.cpu().numpy()
    for tid in range(n_tracts):
        _assert(bool(attend[:, tid].any()), f"tract {tid} has no attending tokens")
    exclusive = (tract_ids == 0) & (is_junc == 0)
    _assert(bool(exclusive.any()), "expected exclusive tract-0 tokens")
    slot0 = int(np.where(exclusive)[0][0])
    _assert(not bool(attend[slot0, 1]), "tract-0 token must not attend tract 1")
    node_t = int(data.tract_id[0].item())
    foreign = [t for t in range(n_tracts) if t != node_t]
    if foreign:
        foreign_ex = (tract_ids == foreign[0]) & (is_junc == 0)
        if foreign_ex.any():
            slot_f = int(np.where(foreign_ex)[0][0])
            _assert(not bool(attend[slot_f, node_t]), "foreign exclusive token attends this tract")


def test_pose_roundtrip():
    data = make_synthetic_data(sac=False)
    R = data.pose_R.numpy()
    _assert(np.allclose(R.T @ R, np.eye(3), atol=1e-4), "R not orthogonal")
    _assert(abs(float(np.linalg.det(R)) - 1.0) < 1e-3, f"det R {np.linalg.det(R)}")
    origin = data.origin_shift.numpy()
    x_can = data.x[:8].numpy()
    x_world = x_can @ R.T + origin
    x_back = (x_world - origin) @ R
    _assert(np.allclose(x_can, x_back, atol=1e-4), "pose round-trip failed")


def test_hybrid_far_points():
    data = make_synthetic_data(sac=True)
    _assert(data.x_true.size(0) == 64, data.x_true.shape)
    _assert(data.x_true_cl_dist.numel() == 64, data.x_true_cl_dist.shape)
    _assert(float(data.x_true_cl_dist.max()) > 2.0, "expected far-from-CL oversample")


def test_cache_hit():
    factory = _make_factory()
    factory.cache_dir = tempfile.mkdtemp(prefix="aneux_cache_")
    factory.samples = [{"dataset_id": "synthetic0", "vessel_file": "n/a", "centerline_file": "n/a"}]
    built = {"n": 0}

    def _build(_sample):
        built["n"] += 1
        return make_synthetic_data(sac=False)

    factory._build_data = _build
    try:
        a = factory[0]
        b = factory[0]
        _assert(built["n"] == 1, f"cache missed, built {built['n']} times")
        _assert(a.x.shape == b.x.shape, "cached shape")
        _assert(int(b.cache_version) == CACHE_VERSION, "cache version")
        path = factory._cache_path("synthetic0")
        stale = make_synthetic_data(sac=False)
        stale.cache_version = torch.tensor(-1)
        torch.save(stale, path)
        _ = factory[0]
        _assert(built["n"] == 2, "stale cache_version should rebuild")
    finally:
        shutil.rmtree(factory.cache_dir, ignore_errors=True)


def test_batch_inc():
    d1 = make_synthetic_data(sac=False)
    d2 = make_synthetic_data(sac=False)
    loader = DataLoader([d1, d2], batch_size=2, follow_batch=FOLLOW_BATCH)
    batch = next(iter(loader))
    _assert(batch.num_graphs == 2, batch.num_graphs)
    _assert(int(batch.edge_index.max()) < batch.x.size(0), "fine edge __inc__")
    _assert(int(batch.edge_index_mid.max()) < batch.pos_mid.size(0), "mid edge __inc__")
    _assert(int(batch.edge_index_coarse.max()) < batch.pos_coarse.size(0), "coarse edge __inc__")
    _assert(int(batch.face.max()) < batch.x.size(0), "fine face __inc__")
    _assert(int(batch.face_mid.max()) < batch.pos_mid.size(0), "mid face __inc__")
    _assert(int(batch.face_coarse.max()) < batch.pos_coarse.size(0), "coarse face __inc__")
    model = _tiny_model()
    model.eval()
    with torch.no_grad():
        out = model(batch)
    _assert(out.mu.shape[0] == 2, out.mu.shape)
    _assert(out.x_pred.size(0) == batch.x.size(0), "batched pred nodes")
    _assert(out.logvar.min() >= LOGVAR_CLAMP[0] - 1e-5, out.logvar.min())
    _assert(out.logvar.max() <= LOGVAR_CLAMP[1] + 1e-5, out.logvar.max())


def test_scaffold_decode_without_vessel():
    factory = _make_factory()
    data = factory.build_scaffold_from_centerline(_polyline_mesh(_straight_branch()))
    _assert(data.x_true.size(0) == 1, "Stage-2 scaffold should not require GT surface")
    model = _tiny_model()
    model.eval()
    z = torch.zeros(1, 8, 8)
    with torch.no_grad():
        x = model.decode(z, data)
    _assert(x.shape == data.x.shape, "decode contract shape")
    _assert(torch.isfinite(x).all(), "non-finite decode")


def test_forward_backward():
    data = make_synthetic_data()
    loader = DataLoader([data], batch_size=1, follow_batch=FOLLOW_BATCH)
    batch = next(iter(loader))
    model = _tiny_model()
    model.train()
    out = model(batch)
    _assert(out.x_pred.shape == batch.x.shape, f"pred {out.x_pred.shape} vs {batch.x.shape}")
    _assert(out.mu.shape == (1, 8, 8), f"mu {out.mu.shape}")
    _assert(out.logvar.shape == (1, 8, 8), f"logvar {out.logvar.shape}")
    _assert(out.z.shape == (1, 8, 8), f"z {out.z.shape}")
    _assert(out.x_pred_coarse.shape == batch.pos_coarse.shape, "coarse pred")
    _assert(out.x_pred_mid.shape == batch.pos_mid.shape, "mid pred")
    _assert(out.delta_r.shape[0] == batch.x.size(0), "delta_r")
    _assert(out.delta_s.shape == (batch.x.size(0), 2), "delta_s")
    _assert(torch.isfinite(out.x_pred).all(), "non-finite prediction")
    _assert(out.x_pred.dtype == torch.float32, "pred dtype")
    _assert(out.mu.dtype == torch.float32 and out.logvar.dtype == torch.float32, "latent dtype")
    _assert(not torch.is_autocast_enabled(), "autocast must stay off")
    for p in model.parameters():
        _assert(p.dtype == torch.float32, f"param dtype {p.dtype}")

    terms = compute_losses(
        out.x_pred,
        batch.x_true,
        out.mu,
        out.logvar,
        batch.x,
        batch.edge_index,
        batch.x_true_batch,
        batch.num_graphs,
        face=batch.face,
        batch_tube=batch.batch,
        delta_x=out.delta_x,
        delta_r=out.delta_r,
        delta_s=out.delta_s,
        x_pred_mid=out.x_pred_mid,
        batch_mid=batch.pos_mid_batch,
        x_pred_coarse=out.x_pred_coarse,
        batch_coarse=batch.pos_coarse_batch,
        x_true_cl_dist=batch.x_true_cl_dist,
        cl_dense=batch.cl_dense,
        cl_dense_batch=batch.cl_dense_batch,
    )
    loss = terms["recon"] + 0.001 * terms["kl"] + 0.1 * terms["disp"] + 0.05 * terms["lap"] + 0.02 * terms["norm"]
    loss.backward()
    grads = [p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None]
    _assert(len(grads) > 0 and sum(grads) > 0, "no gradients")

    model.eval()
    z = out.mu.detach()
    x_dec = model.decode(z, batch)
    _assert(x_dec.shape == batch.x.shape, "decode contract shape")


def test_stage2_precision_flags():
    configure_stage2_precision()
    _assert(torch.get_float32_matmul_precision() == "high", torch.get_float32_matmul_precision())
    if torch.cuda.is_available():
        _assert(bool(torch.backends.cuda.matmul.allow_tf32), "tf32 matmul")
        _assert(bool(torch.backends.cudnn.allow_tf32), "tf32 cudnn")
    _assert(not torch.sparse.check_sparse_tensor_invariants.is_enabled(), "sparse checks")
    _assert(not torch.is_autocast_enabled(), "autocast")


def test_synthetic_data_fp32():
    data = make_synthetic_data()
    for name in (
        "x",
        "x_true",
        "x_true_cl_dist",
        "latent_pos",
        "cl_dense",
        "pos_mid",
        "pos_coarse",
        "u",
        "theta",
        "normal",
        "pose_R",
        "origin_shift",
    ):
        t = getattr(data, name)
        _assert(t.dtype == torch.float32, f"{name} dtype {t.dtype}")
    _assert(data.edge_index.dtype == torch.long, "edge_index dtype")
    _assert(data.face.dtype == torch.long, "face dtype")
    _assert(data.token_attend.dtype == torch.bool, data.token_attend.dtype)
    _assert(not hasattr(data, "cl_pos") or getattr(data, "cl_pos") is None, "cl_pos should be gone")


def test_stratified_split_keeps_train():
    from aneuxai import stratified_split

    class Dummy:
        samples = [{"location": "ICA pcom"}] * 4 + [{"location": "ICA oph"}]

        def __len__(self):
            return len(self.samples)

    train, val = stratified_split(Dummy(), 0.15, 0)
    _assert(len(train) >= 1 and len(val) >= 1, (len(train), len(val)))
    oph = [i for i, s in enumerate(Dummy.samples) if s["location"] == "ICA oph"][0]
    _assert(oph in train.indices, "singleton location should stay in train")


def main():
    configure_stage2_precision()
    tests = [
        test_stage2_precision_flags,
        test_config_contracts,
        test_synthetic_data_fp32,
        test_fourier_shapes,
        test_kl_and_anneal,
        test_pseudo_coords_range,
        test_bilinear_identity_and_wrap,
        test_bishop_frames_orthonormal,
        test_fps_count,
        test_fps_cuda_path,
        test_ball_query_index_order,
        test_radius_all_in_ball,
        test_allocate_rings,
        test_decoupled_head_no_inversion,
        test_residual_radial_floor,
        test_decoder_composed_non_inversion,
        test_orphan_attention_uniform,
        test_cross_attention_gemm_matches_broadcast,
        test_cross_attention_two_graph_isolation,
        test_knn_chamfer_matches_cdist,
        test_mesh_losses_match_pytorch3d,
        test_spline_conv_backend,
        test_dirichlet_zero_on_rigid,
        test_unique_tracts_from_overlapping_paths,
        test_tree_token_mask,
        test_pose_roundtrip,
        test_hybrid_far_points,
        test_cache_hit,
        test_batch_inc,
        test_scaffold_decode_without_vessel,
        test_stratified_split_keeps_train,
        test_forward_backward,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
            print()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
