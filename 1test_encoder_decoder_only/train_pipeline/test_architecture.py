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
    CHAMFER_WEIGHT_CAP,
    DECODER_HIDDEN_DIM,
    FOLLOW_BATCH,
    GAMMA_THETA_DIM,
    GAMMA_U_DIM,
    K_THETA,
    K_U,
    LAMBDA_CD_COARSE,
    LATENT_DIM,
    LATENT_LEN,
    LEVEL_FINE,
    LOGVAR_CLAMP,
    LOGVAR_MIN,
    MAX_TRACTS,
    N_CONV_PER_LEVEL,
    N_TRUE,
    N_TRUE_FAR_FRAC,
    PLANE_HUBER_DELTA_MM,
    PLANE_L2_MIX,
    RADIAL_FLOOR_FRAC,
    SA_STAGES,
    SIGMA_MIN,
    SKIP_GATE_INIT,
    SMOOTH_W_AMBIGUOUS,
    SPLINE_DEGREE,
    SPLINE_KERNEL_SIZE,
    TOKEN_SPACING_MM,
    Z_ATTN_ALPHA_INIT,
    Z_ATTN_GATE_MAX,
    Z_ATTN_HEADS,
    Z_ATTN_RADIUS,
    configure_stage2_precision,
)
from dataset import (
    AneurysmDataset,
    allocate_ring_counts,
    couple_ostium_edges,
    extract_groupid_tracts,
    extract_unique_tracts,
    _arc_len,
)
from cleaned_io import (
    list_cleandata_samples,
    sample_is_complete,
)
from geometry import (
    bilinear_cylindrical_upsample,
    clamp_residual_radial,
    fps_metric,
    harmonic_encoding_theta,
    harmonic_encoding_u,
    intrinsic_spline_pseudo_coords,
    knn_weighted_upsample,
    radial_bias_for_zero_init,
    upsample_branch_concat,
)
from losses import (
    _cl_radius,
    _mesh_normal_consistency,
    _uniform_laplacian_smoothing,
    _weighted_chamfer,
    chamfer_distance_weights,
    compute_losses,
    displacement_dirichlet,
    displacement_dirichlet_local,
    huber,
    radial_huber_loss,
    smoothness_edge_weights,
    vae_kl_loss,
)
from model import (
    CoarsePositionalSelfAttention,
    DecoupledDisplacementHead,
    FreeDisplacementHead,
    GraphVAE,
    LatentCrossAttention,
    LatentTractSelfAttention,
    ResidualSplineConv,
    _attr_batch,
    _ones_batch,
    _soft_logvar,
    _upsample_level,
)
from ops import composed_radius, fps_indices, make_spline_conv
from raycast import (
    choose_normal_sign,
    compute_level_r_star,
    mesh_r_star_edge_stats,
    r_star_grid_stats,
    select_r_star_from_hits,
    template_ray_r_star,
    transform_vessel_mesh,
    voronoi_ok_hit,
)
from train import ModelEMA, kl_anneal_weight

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
    ds.ensure_derived = False
    ds.require_templates = False
    ds.cleandata_root = None
    ds.token_spacing_mm = float(TOKEN_SPACING_MM)
    return ds


class SkipTest(Exception):
    """Missing optional fixture; the architecture runner treats this as SKIP."""


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _vertex_r_local(data, pos=None):
    """Cached r_local, else distance to the dense centerline (generated tubes)."""
    pos = data.x if pos is None else pos
    r = getattr(data, "r_local", None)
    if torch.is_tensor(r) and r.reshape(-1).numel() == pos.size(0):
        return r.to(dtype=torch.float32).reshape(pos.size(0))
    cl = data.cl_dense[:, :3]
    return torch.cdist(pos.float(), cl.float()).min(dim=1).values.clamp_min(1e-4)


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


def _tiny_model(latent_len=8, latent_dim=8, gradient_checkpointing="off"):
    return GraphVAE(
        latent_dim=latent_dim,
        latent_len=latent_len,
        hidden_dim=16,
        tube_radius=2.0,
        sa_stages=TINY_SA,
        gradient_checkpointing=gradient_checkpointing,
    )


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_config_contracts():
    from aneuxai import BATCH_SIZE

    _assert(SA_STAGES[-1][0] == 64, f"last SA n_out {SA_STAGES[-1][0]}")
    _assert(abs(LOGVAR_CLAMP[0] - math.log(0.01)) < 1e-4, LOGVAR_CLAMP)
    _assert(abs(LOGVAR_CLAMP[1] - 2.0) < 1e-4, LOGVAR_CLAMP)
    _assert(abs(SIGMA_MIN - 0.1) < 1e-12, SIGMA_MIN)
    _assert(LAMBDA_CD_COARSE <= 0.05 + 1e-12, LAMBDA_CD_COARSE)
    _assert(CACHE_VERSION >= 10, CACHE_VERSION)
    _assert(LATENT_DIM == 16, LATENT_DIM)
    _assert(LATENT_LEN == 128, LATENT_LEN)
    _assert(abs(TOKEN_SPACING_MM - 2.0) < 1e-12, TOKEN_SPACING_MM)
    _assert(DECODER_HIDDEN_DIM == 128, DECODER_HIDDEN_DIM)
    _assert(N_TRUE == 16384, N_TRUE)
    _assert(LEVEL_FINE[1] == 64, LEVEL_FINE)
    _assert(abs(N_TRUE_FAR_FRAC - 0.25) < 1e-12, N_TRUE_FAR_FRAC)
    _assert(N_CONV_PER_LEVEL == 6, N_CONV_PER_LEVEL)
    _assert(tuple(SPLINE_KERNEL_SIZE) == (5, 5, 3), SPLINE_KERNEL_SIZE)
    _assert(min(SPLINE_KERNEL_SIZE) > SPLINE_DEGREE, (SPLINE_KERNEL_SIZE, SPLINE_DEGREE))
    _assert(CHAMFER_WEIGHT_CAP == 4.0, CHAMFER_WEIGHT_CAP)
    _assert(abs(SKIP_GATE_INIT - 0.1) < 1e-12, SKIP_GATE_INIT)
    _assert(Z_ATTN_HEADS == 4, Z_ATTN_HEADS)
    _assert(abs(Z_ATTN_ALPHA_INIT - 0.1) < 1e-12, Z_ATTN_ALPHA_INIT)
    _assert(Z_ATTN_RADIUS == 2, Z_ATTN_RADIUS)
    _assert(Z_ATTN_GATE_MAX <= 0.5 + 1e-12, Z_ATTN_GATE_MAX)
    _assert(isinstance(BATCH_SIZE, int) and BATCH_SIZE >= 1, BATCH_SIZE)
    from aneuxai import USE_GRADIENT_CHECKPOINTING
    from config import normalize_gradient_checkpointing

    _assert(
        normalize_gradient_checkpointing(USE_GRADIENT_CHECKPOINTING) in ("off", "fine", "all"),
        USE_GRADIENT_CHECKPOINTING,
    )
    _assert(normalize_gradient_checkpointing(False) == "off", "False -> off")
    _assert(normalize_gradient_checkpointing(True) == "all", "True -> all")
    _assert(normalize_gradient_checkpointing("fine") == "fine", "fine")


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

    torch.manual_seed(0)
    mu_v = torch.randn(1, 4, 8)
    lv_v = torch.randn(1, 4, 8) * 0.1
    loss_short, info_short = vae_kl_loss(mu_v, lv_v)
    mu_pad = torch.cat([mu_v, torch.full((1, 3, 8), 10.0)], dim=1)
    lv_pad = torch.cat([lv_v, torch.full((1, 3, 8), 5.0)], dim=1)
    valid = torch.zeros(1, 7, dtype=torch.bool)
    valid[:, :4] = True
    loss_pad, info_pad = vae_kl_loss(mu_pad, lv_pad, latent_valid=valid)
    _assert(
        torch.allclose(info_short["kl_mean_raw"], info_pad["kl_mean_raw"], atol=1e-6),
        f"KL mean must ignore padding: {info_short['kl_mean_raw']} vs {info_pad['kl_mean_raw']}",
    )
    _assert(torch.allclose(loss_short, loss_pad, atol=1e-6), f"KL loss {loss_short} vs {loss_pad}")
    _assert(int(info_pad["n_valid"].item()) == 4, info_pad["n_valid"])
    _assert(float(loss_pad) == float(loss_short), "float(KlLossResult) must match")


def test_pseudo_coords_range():
    data = make_synthetic_data(sac=False)
    r_local = _vertex_r_local(data)
    raised = False
    try:
        intrinsic_spline_pseudo_coords(
            data.u, data.theta, data.tract_id, data.edge_index, data.u_step
        )
    except ValueError:
        raised = True
    _assert(raised, "r_local is required; π-normalised Δθ is not a fallback")
    e = intrinsic_spline_pseudo_coords(
        data.u, data.theta, data.tract_id, data.edge_index, data.u_step,
        r_local=r_local, pos=data.x,
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


def test_knn_weighted_upsample():
    field = torch.tensor([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]], dtype=torch.float32)
    index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    weight = torch.tensor([[0.5, 0.5], [0.25, 0.75]], dtype=torch.float32)
    out = knn_weighted_upsample(field, index, weight)
    expect = torch.tensor([[2.0, 0.0], [4.5, 0.0]], dtype=torch.float32)
    _assert(torch.allclose(out, expect, atol=1e-5), out)


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
    _assert(
        isinstance(model.decoder.coarse_head, FreeDisplacementHead),
        type(model.decoder.coarse_head).__name__,
    )
    z = torch.zeros(1, 8, 8)
    with torch.no_grad():
        _, delta_x0, x_c0, _, _, *_ = model.decoder(z, batch)
    dx_c0 = x_c0 - batch.pos_coarse
    _assert(
        torch.allclose(dx_c0, torch.zeros_like(dx_c0), atol=1e-4),
        f"coarse identity Δx max={float(dx_c0.abs().max())}",
    )
    for head in (model.decoder.mid_head, model.decoder.head):
        torch.nn.init.zeros_(head.radial.weight)
        head.radial.bias.data.fill_(-80.0)
        torch.nn.init.zeros_(head.shear.weight)
        torch.nn.init.zeros_(head.shear.bias)
    with torch.no_grad():
        _, delta_x, x_c, x_m, _, *_ = model.decoder(z, batch)
    r_m = (batch.normal_mid * (x_m - batch.pos_mid)).sum(-1)
    r_f = (batch.normal * delta_x).sum(-1)
    floor = float(RADIAL_FLOOR_FRAC) * 2.0
    _assert(bool((r_m >= -floor - 1e-3).all()), f"mid composed Δr min={float(r_m.min())}")
    _assert(bool((r_f >= -floor - 1e-3).all()), f"fine composed Δr min={float(r_f.min())}")
    _assert(
        torch.allclose(x_c, batch.pos_coarse, atol=1e-4),
        "coarse FreeDisplacementHead must stay at identity when inverted mid/fine heads run",
    )


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


def test_point_to_plane_chamfer():
    true = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    n_true = torch.tensor([[0.0, 0.0, 1.0]] * 4, dtype=torch.float32)
    ones_p = torch.ones(4)
    ones_t = torch.ones(4)
    batch = torch.zeros(4, dtype=torch.long)
    tangent = true + torch.tensor([0.4, -0.25, 0.0])
    plane_t = _weighted_chamfer(tangent, batch, true, batch, ones_p, ones_t, 1, n_true=n_true)
    l2_t = _weighted_chamfer(tangent, batch, true, batch, ones_p, ones_t, 1)
    _assert(float(l2_t) > 0.05, f"L2 Chamfer must stay on without normals, got {float(l2_t)}")
    _assert(
        abs(float(plane_t) - float(PLANE_L2_MIX) * float(l2_t)) < 1e-4,
        f"in-plane should be mix*L2, got {float(plane_t)} vs {float(PLANE_L2_MIX) * float(l2_t)}",
    )
    normal = true + torch.tensor([0.0, 0.0, 3.0])
    plane_n = _weighted_chamfer(normal, batch, true, batch, ones_p, ones_t, 1, n_true=n_true)
    h_n = huber(torch.tensor([3.0]), delta=PLANE_HUBER_DELTA_MM)
    expected_n = (1.0 - PLANE_L2_MIX) * float(h_n) + PLANE_L2_MIX * 9.0
    _assert(abs(float(plane_n) - expected_n) < 1e-3, f"normal offset {float(plane_n)} vs {expected_n}")


def test_postprocess_checkpoint_and_split_summary():
    parent = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if parent not in sys.path:
        sys.path.insert(0, parent)
    import postprocess as pp

    live = {"weight": torch.zeros(2)}
    ema = {"weight": torch.ones(2)}
    last = {"epoch": 4, "geco_beta": 0.8, "model": live, "ema": {"decay": 0.99, "shadow": ema, "n_updates": 3}}
    state, source = pp.extract_state_dict(last, prefer_ema=True)
    _assert(source == "ema.shadow", source)
    _assert(torch.equal(state["weight"], torch.ones(2)), state["weight"])
    state, source = pp.extract_state_dict(last, prefer_ema=False)
    _assert(source == "model", source)
    _assert(torch.equal(state["weight"], torch.zeros(2)), state["weight"])
    best = {"epoch": 4, "score": 1.5, "model": ema}
    state, source = pp.extract_state_dict(best, prefer_ema=True)
    _assert(source == "model" and torch.equal(state["weight"], torch.ones(2)), source)
    beta, beta_src = pp.beta_for_checkpoint(last, run_dir=None)
    _assert(beta_src == "checkpoint" and abs(beta - 0.8) < 1e-9, (beta, beta_src))

    groups = {"train": {"a"}, "val": {"b"}, "test": {"c"}}
    _assert(pp.split_of("c", groups) == "test", pp.split_of("c", groups))
    _assert(pp.split_of("z", groups) == "unassigned", "missing id")

    rows = [
        {"status": "ok", "split": "train", "total": 2.0, "recon": 1.0, "rad": 1.0, "kl": 1.0},
        {"status": "ok", "split": "train", "total": 4.0, "recon": 3.0, "rad": 1.0, "kl": 1.0},
        {"status": "ok", "split": "val", "total": 10.0, "recon": 8.0, "rad": 1.0, "kl": 1.0},
        {"status": "error", "split": "test", "total": 99.0, "recon": 99.0, "error": "x"},
    ]
    summary = pp.summarize_splits(rows)
    recon = {row["split"]: row for row in summary if row["metric"] == "recon"}
    _assert(recon["train"]["n"] == 2 and abs(recon["train"]["mean"] - 2.0) < 1e-9, recon["train"])
    _assert(recon["val"]["n"] == 1 and abs(recon["val"]["mean"] - 8.0) < 1e-9, recon["val"])
    _assert("test" not in recon, recon)
    _assert(recon["all"]["n"] == 3, recon["all"])

    class Data:
        pass

    data = Data()
    data.pose_R = torch.eye(3)
    data.origin_shift = torch.tensor([10.0, 0.0, 0.0])
    world = pp.canonical_to_world(torch.tensor([[1.0, 2.0, 3.0]]), data)
    _assert(torch.allclose(world, torch.tensor([[11.0, 2.0, 3.0]])), world)


def test_stretch_identity_on_skinny_triangle():
    """Skinny template faces must not blow up σ + 1/σ when pred == template."""
    from losses import triangle_stretch_loss

    tpl = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1e-6, 0.0]],
        dtype=torch.float32,
    )
    face = torch.tensor([[0], [1], [2]], dtype=torch.long)
    same = float(triangle_stretch_loss(tpl, tpl, face))
    _assert(same < 1e-4, same)
    healthy = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    pred = healthy.clone()
    pred[:, :2] *= 2.0
    scaled = float(triangle_stretch_loss(pred, healthy, face))
    _assert(abs(scaled - 0.5) < 1e-3, scaled)
    collapsed = healthy.clone()
    collapsed[2] = collapsed[0]
    bad = float(triangle_stretch_loss(collapsed, healthy, face))
    _assert(math.isfinite(bad) and bad < 2000.0, bad)


def test_cross_attention_reads_world_direction():
    """The query carries the node normal, and the FiLM starts at identity."""
    from model import LatentCrossAttention

    torch.manual_seed(0)
    n, L, d = 12, 4, 8
    layer = LatentCrossAttention(d, 16, 16, L, use_dir=True)
    base = LatentCrossAttention(d, 16, 16, L, use_dir=False)
    z = torch.randn(1, L, d)
    u = torch.linspace(0.0, 1.0, n)
    theta = torch.zeros(n)
    tract = torch.zeros(n, dtype=torch.long)
    token_u = torch.linspace(0.0, 1.0, L).unsqueeze(0)
    attend = torch.ones(1, L, MAX_TRACTS, dtype=torch.bool)
    batch = torch.zeros(n, dtype=torch.long)
    nrm = torch.nn.functional.normalize(torch.randn(n, 3), dim=-1)
    args = (z, u, theta, batch, tract, token_u, attend)
    out_a = layer(*args, node_dir=nrm)
    out_b = layer(*args, node_dir=-nrm)
    _assert(out_a.shape == (n, 16), out_a.shape)
    _assert(not torch.allclose(out_a, out_b), "flipping the normal did not change the output")
    _assert(float(layer.dir_film.weight.abs().sum()) == 0.0, "dir FiLM must start at identity")
    _assert(base(*args).shape == (n, 16), "use_dir=False must keep the (u, θ)-only layer")
    try:
        layer(*args)
        raise AssertionError("use_dir=True without node_dir must raise")
    except ValueError:
        pass


def test_mesh_terms_have_finite_gradients_at_identity():
    """At init the heads are zero, so pred == template exactly.

    sqrt of a repeated eigenvalue (0) back-propagates 0 * inf = NaN; one such
    term poisons every parameter even at weight 0, since 0 * NaN is NaN.
    """
    from losses import conformal_distortion_loss, dihedral_fold_penalty, triangle_stretch_loss

    tpl = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.2]],
        dtype=torch.float32,
    )
    face = torch.tensor([[0, 1], [1, 3], [2, 2]], dtype=torch.long)
    batch = torch.zeros(4, dtype=torch.long)
    for name, fn in (
        ("stretch", lambda x: triangle_stretch_loss(x, tpl, face)),
        ("conf", lambda x: conformal_distortion_loss(x, tpl, face, batch, 1)),
        ("fold", lambda x: dihedral_fold_penalty(x, face, batch, 1)),
    ):
        x = tpl.clone().requires_grad_(True)
        (0.0 * fn(x)).backward()
        _assert(torch.isfinite(x.grad).all(), f"{name}: non-finite grad at identity {x.grad}")


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
    res = ResidualSplineConv(8)
    kraw = res.conv.kernel_size
    ks = tuple(int(v) for v in (kraw.tolist() if hasattr(kraw, "tolist") else kraw))
    _assert(ks == (5, 5, 3), ks)
    _assert(int(res.conv.degree) < min(ks), (res.conv.degree, ks))
    _assert(bool(res.conv.root_weight), "SplineConv root_weight")


def test_disp_sees_upsampled_spikes():
    """Flat Δr/Δs must not hide a spiky composed displacement."""
    n = 4
    x = torch.zeros(n, 3)
    x[:, 0] = torch.arange(n)
    true = x.clone()
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]])
    batch = torch.zeros(n, dtype=torch.long)
    mu = torch.zeros(1, 2, 4)
    logvar = torch.zeros(1, 2, 4)
    dr = torch.zeros(n, 1)
    ds = torch.zeros(n, 2)
    smooth = torch.zeros(n, 3)
    spiky = torch.zeros(n, 3)
    spiky[1, 2] = 8.0

    def _disp(delta_x):
        terms = compute_losses(
            x, true, mu, logvar, x, ei, batch, 1,
            delta_x=delta_x, delta_r=dr, delta_s=ds, batch_tube=batch,
        )
        return float(terms["disp"])

    _assert(_disp(smooth) < 1e-8, "flat field")
    _assert(_disp(spiky) > 1.0, "upsampled spike must enter disp")


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


def _polyline_with_groups(pts, group_ids, blanking=None):
    mesh = _polyline_mesh(pts)
    n = int(mesh.n_points)
    gids = np.asarray(group_ids, dtype=np.float64).reshape(-1)
    _assert(len(gids) == n, f"GroupIds {len(gids)} vs points {n}")
    mesh.point_data["GroupIds"] = gids
    if blanking is None:
        blanking = np.zeros(n, dtype=np.float64)
    mesh.point_data["Blanking"] = np.asarray(blanking, dtype=np.float64).reshape(-1)
    return mesh


def test_groupid_tracts_one_polyline_per_group():
    parent, child_a, child_b = (
        np.stack([np.zeros(12), np.zeros(12), np.linspace(0.0, 12.0, 12)], axis=1),
        np.stack([np.linspace(0.0, 8.0, 10), np.zeros(10), np.full(10, 12.0)], axis=1),
        np.stack([np.zeros(10), np.linspace(0.0, 8.0, 10), np.full(10, 12.0)], axis=1),
    )
    meshes = [
        _polyline_with_groups(parent, np.zeros(len(parent))),
        _polyline_with_groups(child_a, np.ones(len(child_a))),
        _polyline_with_groups(child_b, np.full(len(child_b), 2.0)),
    ]
    mesh = meshes[0]
    for extra in meshes[1:]:
        mesh = mesh.merge(extra)
    tracts, endpoints, junctions = extract_groupid_tracts(mesh)
    _assert(len(tracts) == 3, f"expected 3 GroupId tracts, got {len(tracts)}")
    _assert(len(endpoints) == 3, endpoints)
    _assert(len(junctions) >= 1, "expected a snapped junction")


def test_groupid_tracts_collapses_duplicate_parent():
    path1, path2 = _y_paths()
    n_parent = 12
    g1 = np.concatenate([np.zeros(n_parent), np.ones(len(path1) - n_parent)])
    g2 = np.concatenate([np.zeros(n_parent), np.full(len(path2) - n_parent, 2.0)])
    mesh = _polyline_with_groups(path1, g1).merge(_polyline_with_groups(path2, g2))
    tracts, endpoints, junctions = extract_groupid_tracts(mesh)
    _assert(len(tracts) == 3, f"duplicate parent GroupId should collapse, got {len(tracts)}")
    _assert(len(junctions) >= 1, "expected a junction after endpoint snap")


def test_groupid_skips_blanked_bifurcation():
    parent = np.stack([np.zeros(8), np.zeros(8), np.linspace(0.0, 8.0, 8)], axis=1)
    blank = np.stack([np.zeros(4), np.zeros(4), np.linspace(8.0, 8.4, 4)], axis=1)
    child = np.stack([np.linspace(0.0, 6.0, 8), np.zeros(8), np.full(8, 8.4)], axis=1)
    mesh = (
        _polyline_with_groups(parent, np.zeros(len(parent)))
        .merge(_polyline_with_groups(blank, np.full(len(blank), 9.0), blanking=np.ones(len(blank))))
        .merge(_polyline_with_groups(child, np.ones(len(child))))
    )
    tracts, endpoints, _junctions = extract_groupid_tracts(mesh)
    _assert(len(tracts) == 2, f"blanked group should be dropped, got {len(tracts)}")
    _assert(len(endpoints) == 2, endpoints)


def test_groupid_falls_back_without_arrays():
    mesh = _polylines_mesh(list(_y_paths()))
    tracts_g, _, _ = extract_groupid_tracts(mesh)
    tracts_u, _, _ = extract_unique_tracts(mesh)
    _assert(len(tracts_g) == len(tracts_u), (len(tracts_g), len(tracts_u)))


def _write_dummy_vtp(path, n=6):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng = np.random.RandomState(0)
    pv.PolyData(rng.randn(n, 3)).save(path)


def test_cleandata_discovery_and_complete():
    root = tempfile.mkdtemp(prefix="aneux_cleandata_")
    try:
        layout_dirs = {
            "uniformly_remeshed": os.path.join(root, "uniformly_remeshed"),
            "template_mesh": os.path.join(root, "template_mesh"),
            "original_centerline": os.path.join(root, "original_centerline"),
        }
        for folder in layout_dirs.values():
            os.makedirs(folder, exist_ok=True)
        for folder in layout_dirs.values():
            _write_dummy_vtp(os.path.join(folder, "CASE01.vtp"))
        _write_dummy_vtp(os.path.join(layout_dirs["uniformly_remeshed"], "CASE02.vtp"))
        complete = list_cleandata_samples(root=root, require_templates=True)
        _assert([s["dataset_id"] for s in complete] == ["CASE01"], complete)
        rec = complete[0]
        _assert(sample_is_complete(rec, require_templates=True), rec)
        _assert(os.path.basename(os.path.dirname(rec["vessel_file"])) == "uniformly_remeshed", rec["vessel_file"])
        _assert(os.path.basename(os.path.dirname(rec["centerline_file"])) == "original_centerline", rec["centerline_file"])
        _assert(os.path.basename(os.path.dirname(rec["template_mesh_file"])) == "template_mesh", rec["template_mesh_file"])
        rec_no_tpl_cl = dict(rec)
        rec_no_tpl_cl["template_centerline_file"] = os.path.join(
            root, "template_centerline", "CASE01.vtp"
        )
        _assert(
            sample_is_complete(rec_no_tpl_cl, require_templates=True),
            "completeness must not require template_centerline",
        )
        _assert(not os.path.isdir(os.path.join(root, "template_centerline")), "test must not create the dropped folder")
        ds = AneurysmDataset(
            cleandata_root=root,
            cache_dir=os.path.join(root, "cache"),
            quiet=True,
        )
        _assert(len(ds) == 1, len(ds))
        _assert(ds.samples[0]["dataset_id"] == "CASE01", ds.samples[0])
        _assert(ds.samples[0]["vessel_file"].endswith("CASE01.vtp"), ds.samples[0]["vessel_file"])
        _assert("coarse_file" not in ds.samples[0], ds.samples[0])
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _template_cylinder_pair(radius_tpl=2.0, radius_gt=2.4, n_sides=20, n_len=40):
    z = np.linspace(0.0, 20.0, n_len)
    pts = np.stack([np.zeros(n_len), np.zeros(n_len), z], axis=1)
    if hasattr(pv, "lines_from_points"):
        line = pv.lines_from_points(pts)
    else:
        line = _polyline_mesh(pts)
    tpl = line.tube(radius=radius_tpl, n_sides=n_sides, capping=True)
    gt = line.tube(radius=radius_gt, n_sides=n_sides + 4, capping=True)
    if not bool(tpl.is_all_triangles):
        tpl = tpl.triangulate()
    if not bool(gt.is_all_triangles):
        gt = gt.triangulate()
    return _polyline_mesh(pts), tpl, gt


def _open_cylinder_surface(radius=2.0, height=20.0, n_th=24, n_z=40, bulge=0.0):
    """Open cylinder (2 boundary loops). Optional +X Gaussian bulge on the wall."""
    z = np.linspace(0.0, height, n_z)
    th = np.linspace(0.0, 2.0 * np.pi, n_th, endpoint=False)
    zz, tt = np.meshgrid(z, th, indexing="ij")
    r = np.full_like(zz, float(radius), dtype=np.float64)
    if bulge:
        sigma = 3.0
        bump = float(bulge) * np.exp(-0.5 * ((zz - 0.5 * height) / sigma) ** 2)
        r = r + bump * np.clip(np.cos(tt), 0.0, None)
    pts = np.stack(
        [(r * np.cos(tt)).reshape(-1), (r * np.sin(tt)).reshape(-1), zz.reshape(-1)],
        axis=1,
    )
    faces = []
    for i in range(n_z - 1):
        for j in range(n_th):
            a = i * n_th + j
            b = i * n_th + (j + 1) % n_th
            c = (i + 1) * n_th + j
            d = (i + 1) * n_th + (j + 1) % n_th
            faces.extend([3, a, b, c, 3, b, d, c])
    return pv.PolyData(pts, np.asarray(faces, dtype=np.int64))


def _cell_y_centerline():
    """VMTK-like cells: duplicate parent, ~1.6 mm blank, two daughters."""
    parent = np.stack([np.zeros(90), np.zeros(90), np.linspace(0.0, 88.8, 90)], axis=1)
    blank = np.stack([np.zeros(8), np.zeros(8), np.linspace(88.8, 90.4, 8)], axis=1)
    d0 = np.stack([np.linspace(0.0, 13.3, 20), np.zeros(20), np.full(20, 90.4)], axis=1)
    d1 = np.stack([np.zeros(20), np.linspace(0.0, 12.9, 20), np.full(20, 90.4)], axis=1)
    chunks = [parent, blank, d0, parent.copy(), blank.copy(), d1]
    gids = [0, 1, 2, 0, 1, 3]
    blanks = [0, 1, 0, 0, 1, 0]
    clids = [0, 0, 0, 1, 1, 1]
    tids = [0, 1, 2, 0, 1, 2]
    pts = np.concatenate(chunks, axis=0)
    lines = []
    off = 0
    for ch in chunks:
        n = len(ch)
        lines.append(np.concatenate(([n], np.arange(off, off + n, dtype=np.int64))))
        off += n
    mesh = pv.PolyData(pts, lines=np.concatenate(lines))
    mesh.cell_data["GroupIds"] = np.asarray(gids, dtype=np.float64)
    mesh.cell_data["Blanking"] = np.asarray(blanks, dtype=np.float64)
    mesh.cell_data["CenterlineIds"] = np.asarray(clids, dtype=np.float64)
    mesh.cell_data["TractIds"] = np.asarray(tids, dtype=np.float64)
    return mesh


def _snf_unclipped_paths():
    return [
        os.path.join(_REPO_ROOT, "scratch", "cl_probe", "SNF00000100_branched_unclipped.vtp"),
        os.path.join(_REPO_ROOT, "scratch", "uniform_probe", "original_centerline", "SNF00000100.vtp"),
    ]


def _first_existing(paths):
    for path in paths:
        if os.path.isfile(path):
            return path
    return None


def test_template_scaffold_starts_from_mesh():
    factory = _make_factory(n_true=64, latent_len=8)
    cl, tpl, gt = _template_cylinder_pair()
    data = factory.build_scaffold(cl, vessel_mesh=gt, template_mesh=tpl)
    origin = data.origin_shift.detach().cpu().numpy().reshape(3)
    rot = data.pose_R.detach().cpu().numpy().reshape(3, 3)
    posed = (np.asarray(tpl.points, dtype=np.float64) - origin) @ rot
    _assert(data.x.size(0) == posed.shape[0], (data.x.size(0), posed.shape[0]))
    _assert(
        float(np.linalg.norm(data.x.numpy() - posed, axis=1).max()) < 1e-3,
        "fine scaffold must be the posed template_mesh",
    )
    _assert(hasattr(data, "upsample_idx_mid") and hasattr(data, "upsample_idx_fine"), "knn tables")
    _assert(data.upsample_idx_mid.size(0) == data.pos_mid.size(0), data.upsample_idx_mid.shape)
    _assert(data.upsample_idx_fine.size(0) == data.x.size(0), data.upsample_idx_fine.shape)
    _assert(int(data.n_radial_fine.item()) == 1, data.n_radial_fine)
    model = _tiny_model()
    model.eval()
    loader = DataLoader([data], batch_size=1, follow_batch=FOLLOW_BATCH)
    batch = next(iter(loader))
    with torch.no_grad():
        out = model(batch)
    _assert(out.x_pred.shape == batch.x.shape, out.x_pred.shape)
    _assert(torch.isfinite(out.x_pred).all(), "template decoder non-finite")

    d1 = factory.build_scaffold(cl, vessel_mesh=gt, template_mesh=tpl)
    d2 = factory.build_scaffold(cl, vessel_mesh=gt, template_mesh=tpl)
    batched = next(iter(DataLoader([d1, d2], batch_size=2, follow_batch=FOLLOW_BATCH)))
    _assert(int(batched.upsample_idx_mid.max()) < batched.pos_coarse.size(0), "mid knn __inc__")
    _assert(int(batched.upsample_idx_fine.max()) < batched.pos_mid.size(0), "fine knn __inc__")
    with torch.no_grad():
        bout = model(batched)
    _assert(bout.x_pred.size(0) == batched.x.size(0), bout.x_pred.shape)
    _assert(torch.isfinite(bout.x_pred).all(), "batched template decoder non-finite")


def test_dataset_rejects_rawdata_dirs():
    tmp = tempfile.mkdtemp(prefix="aneux_raw_reject_")
    try:
        vessel = os.path.join(tmp, "rawdata", "vessels")
        centerline = os.path.join(tmp, "rawdata", "centerlines")
        raised = False
        try:
            AneurysmDataset(
                vtp_vessel_dir=vessel,
                vtp_centerline_dir=centerline,
                cache_dir=os.path.join(tmp, "cache"),
                quiet=True,
            )
        except ValueError as exc:
            raised = "rawdata" in str(exc).lower()
        _assert(raised, "explicit rawdata dirs must be rejected")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


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


def test_junction_coupling_edges():
    factory = _make_factory()
    paths = _y_paths()
    data = factory.build_scaffold(_polylines_mesh(list(paths)), vessel_points=_tube_cloud(paths[0]))
    n = int(data.x.size(0))
    src, dst = data.edge_index
    _assert(int(src.max()) < n and int(dst.max()) < n, "coupling edges out of range")
    _assert(int(src.min()) >= 0 and int(dst.min()) >= 0, "negative coupling index")
    cross = data.tract_id[src] != data.tract_id[dst]
    _assert(bool(cross.any()), "Y-junction should have parent–daughter coupling edges")
    _assert(int(data.n_tracts) >= 2, "Y-junction needs ≥2 tracts")


def test_ostium_couple_linear_memory():
    """Dense ostium neighborhoods must not allocate an (Na × Nb × 3) distance tensor."""
    rng = np.random.default_rng(0)
    n = 12000
    pos = np.concatenate(
        [rng.normal(scale=0.8, size=(n, 3)) + np.array([0.4, 0.0, 0.0]),
         rng.normal(scale=0.8, size=(n, 3)) + np.array([-0.4, 0.0, 0.0])],
        axis=0,
    )
    tract_id = np.concatenate(
        [np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)]
    )
    extra = couple_ostium_edges(
        pos,
        tract_id,
        {0: [0, 1]},
        {0: np.zeros(3, dtype=np.float64)},
        radius_mm=4.0,
        k=2,
    )
    _assert(extra.ndim == 2 and extra.shape[1] == 2, extra.shape)
    _assert(extra.shape[0] > 0, "expected coupling edges in a dense ostium ball")
    _assert(int(extra.max()) < 2 * n, extra.max())
    _assert(int(extra.min()) >= 0, extra.min())


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


def test_gradient_checkpointing_modes():
    data = make_synthetic_data()
    loader = DataLoader([data], batch_size=1, follow_batch=FOLLOW_BATCH)
    batch = next(iter(loader))
    for mode in ("off", "fine", "all"):
        model = _tiny_model(gradient_checkpointing=mode)
        _assert(model.gradient_checkpointing == mode, mode)
        model.train()
        out = model(batch)
        loss = out.x_pred.float().pow(2).mean() + out.mu.float().pow(2).mean()
        loss.backward()
        _assert(torch.isfinite(loss), f"non-finite {mode}")
        grads = [p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None]
        _assert(len(grads) > 0 and sum(grads) > 0, f"no grads {mode}")


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
        r_star=getattr(batch, "r_star", None),
        r_star_valid=getattr(batch, "r_star_valid", None),
        r_dth=getattr(batch, "r_dth", None),
        r_du=getattr(batch, "r_du", None),
        r_ring_med=getattr(batch, "r_ring_med", None),
        normal=batch.normal,
    )
    loss = (
        terms["recon"]
        + 0.001 * terms["kl"]
        + 0.1 * terms["disp"]
        + 0.05 * terms["lap"]
        + 0.02 * terms["norm"]
        + terms["rad"]
    )
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
        "r_star",
        "r_dth",
        "r_star_mid",
        "x_true_normal",
    ):
        t = getattr(data, name)
        _assert(t.dtype == torch.float32, f"{name} dtype {t.dtype}")
    _assert(data.r_star_valid.dtype == torch.bool, data.r_star_valid.dtype)
    _assert(data.r_star_ambiguous.dtype == torch.bool, data.r_star_ambiguous.dtype)
    _assert(data.r_star_ambiguous.shape[0] == data.x.shape[0], "ambiguous must match fine nodes")
    _assert(data.r_star.shape[0] == data.x.shape[0], "r_star must match fine nodes")
    _assert(data.r_star_mid.shape[0] == data.pos_mid.shape[0], "r_star_mid must match mid nodes")
    _assert(data.edge_index.dtype == torch.long, "edge_index dtype")
    _assert(data.face.dtype == torch.long, "face dtype")
    _assert(data.token_attend.dtype == torch.bool, data.token_attend.dtype)
    _assert(data.latent_valid.dtype == torch.bool, data.latent_valid.dtype)
    _assert(data.latent_valid.shape[0] == data.latent_u.shape[0], "latent_valid length")
    _assert(data.x_true_normal.shape == data.x_true.shape, "x_true_normal shape")
    _assert(not hasattr(data, "cl_pos") or getattr(data, "cl_pos") is None, "cl_pos should be gone")


def test_train_val_split_covers_all():
    from aneuxai import hospital_group, split_ids, train_val_split

    class Dummy:
        samples = [{"dataset_id": f"c{i}"} for i in range(5)]

        def __len__(self):
            return len(self.samples)

    train, val = train_val_split(Dummy(), 0.2, 0)
    _assert(len(train) + len(val) == 5, (len(train), len(val)))
    _assert(len(val) >= 1 and len(train) >= 1, (len(train), len(val)))
    _assert(set(train.indices).isdisjoint(val.indices), (train.indices, val.indices))
    _assert(set(train.indices) | set(val.indices) == set(range(5)), (train.indices, val.indices))

    _assert(hospital_group("SNF00000100") == "SNF", hospital_group("SNF00000100"))
    _assert(hospital_group("p157_EgAYExEF") == "p", hospital_group("p157_EgAYExEF"))
    _assert(hospital_group("UPF_P0286.00_ID1") == "UPF", hospital_group("UPF_P0286.00_ID1"))
    _assert(hospital_group("USFD_UNIGE_0001") == "USFD", hospital_group("USFD_UNIGE_0001"))
    _assert(hospital_group("ANSYS_UNIGE_30_612") == "ANSYS", hospital_group("ANSYS_UNIGE_30_612"))

    ids = (
        [f"SNF{i:08d}" for i in range(20)]
        + [f"p{i:03d}_xxxx" for i in range(20)]
        + [f"UPF_P{i:04d}.00_ID1" for i in range(20)]
    )
    tr, va, te = split_ids(ids, 0.15, 0.15, 31)
    _assert(len(tr) + len(va) + len(te) == 60, (len(tr), len(va), len(te)))
    _assert(len(set(tr) & set(va)) == 0 and len(set(tr) & set(te)) == 0 and len(set(va) & set(te)) == 0, "overlaps")
    for prefix in ("SNF", "p", "UPF"):
        n_va = sum(1 for x in va if hospital_group(x) == prefix)
        n_te = sum(1 for x in te if hospital_group(x) == prefix)
        _assert(n_va >= 1, f"{prefix} missing from val")
        _assert(n_te >= 1, f"{prefix} missing from test")


def test_split_groups_patients_and_reconciles():
    import json
    import tempfile
    from aneuxai import load_or_create_fixed_split, patient_group, reconcile_split_payload, split_ids

    for key, want in (
        ("p447_GBQfAx_1", "p447"), ("p461_HwYbBA_RICA", "p461"), ("p420_Bg4cPh", "p420"),
        ("SNF00000049_01_3", "SNF00000049"), ("SNF00000365_02", "SNF00000365"), ("SNF00000074", "SNF00000074"),
        ("C0088b", "C0088"), ("C0002", "C0002"), ("UPF_P0211.00_ID2", "UPF_P0211"),
        ("ANSYS_UNIGE_17_10", "ANSYS_UNIGE_17"), ("ANSYS_UNIGE_16", "ANSYS_UNIGE_16"), ("USFD_0032", "USFD_0032"),
    ):
        _assert(patient_group(key) == want, f"{key} -> {patient_group(key)}, want {want}")

    ids = []
    for i in range(30):  # every third SNF patient has three keep-one variants
        ids += [f"SNF{i:08d}_01_{k}" for k in (1, 2, 3)] if i % 3 == 0 else [f"SNF{i:08d}_01"]
    ids += [f"C{i:04d}{s}" for i in range(20) for s in ("a", "b")]
    tr, va, te = split_ids(ids, 0.15, 0.15, 31)
    side = {i: k for k, v in (("train", tr), ("val", va), ("test", te)) for i in v}
    _assert(sorted(side) == sorted(ids), "every case in exactly one side")
    by_pat = {}
    for i, k in side.items():
        by_pat.setdefault(patient_group(i), set()).add(k)
    _assert(all(len(v) == 1 for v in by_pat.values()), "a patient on two sides")
    _assert(len(va) >= 1 and len(te) >= 1, (len(va), len(te)))

    stored = {"train": tr, "val": va, "test": te, "seed": 31, "stratify": "hospital", "group": "patient"}
    sibling = te[0].rsplit("_", 1)[0] + "_9" if te[0].count("_") == 2 else te[0][:-1] + "z"
    now = [i for i in ids if i != tr[0]] + [sibling, "SNF99999999_01"]
    out, ch = reconcile_split_payload(stored, now, 0.15, 0.15, 31)
    _assert(sibling in out["test"], f"new case {sibling} should join its patient's test side")
    _assert(tr[0] not in sum((out[k] for k in ("train", "val", "test")), []), "missing case dropped")
    _assert(ch["dropped"] == [tr[0]] and ch["added_new_patients"] == ["SNF99999999_01"], ch)
    for k in ("train", "val", "test"):
        _assert(set(stored[k]) - {tr[0]} <= set(out[k]), f"stored {k} assignments moved")

    class Dummy:
        def __init__(self, keys):
            self.samples = [{"dataset_id": k} for k in keys]

        def __len__(self):
            return len(self.samples)

    root = tempfile.mkdtemp()
    try:
        path = os.path.join(root, "split.json")
        with open(path, "w") as f:  # an old case-level split is rebuilt, not reused
            json.dump({"train": ids[:-1], "val": ids[-1:], "test": ids[:1], "stratify": "hospital"}, f)
        *_, p1 = load_or_create_fixed_split(Dummy(ids), path, 0.15, 0.15, 31)
        _assert(p1.get("group") == "patient", p1.keys())
        *_, p2 = load_or_create_fixed_split(Dummy(now), path, 0.15, 0.15, 31)
        placed = p2["train"] + p2["val"] + p2["test"]
        _assert(sorted(placed) == sorted(now), "reconciled split covers the current cases")
        _assert([n for n in os.listdir(root) if n.endswith(".tmp")] == [], "atomic write leaves no temp file")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_filter_split_payload_skips_failed_cache_ids():
    from aneuxai import filter_split_payload, subsets_from_ids

    payload = {
        "train": ["a", "b", "c"],
        "val": ["d"],
        "test": ["e"],
        "seed": 31,
    }
    out = filter_split_payload(payload, ["b", "e"])
    _assert(out["train"] == ["a", "c"], out["train"])
    _assert(out["val"] == ["d"], out["val"])
    _assert(out["test"] == [], out["test"])
    _assert(out["skipped_cache"] == ["b", "e"], out["skipped_cache"])
    _assert(payload["train"] == ["a", "b", "c"], "durable lists must stay")

    class Dummy:
        samples = [{"dataset_id": x} for x in "abcde"]

        def __len__(self):
            return len(self.samples)

    tr, va, te = subsets_from_ids(Dummy(), out["train"], out["val"], out["test"])
    _assert(tr.indices == [0, 2], tr.indices)
    _assert(va.indices == [3], va.indices)
    _assert(te.indices == [], te.indices)


def test_skipped_samples_report_is_loud():
    from aneuxai import skipped_sample_records
    from run_report import write_run_readme, write_skipped_samples_report

    payload = {"train": ["p347_x"], "val": ["p391_y"], "test": []}
    errors = [
        ("p347_x", "ValueError: template coarse: 2 non-manifold edges"),
        ("p391_y", "ValueError: template mid: 3 non-manifold edges"),
    ]
    records = skipped_sample_records(payload, errors)
    _assert(records[0]["split"] == "train", records[0])
    _assert(records[1]["split"] == "val", records[1])
    root = tempfile.mkdtemp(prefix="aneux_skip_")
    try:
        path = write_skipped_samples_report(root, records)
        _assert(os.path.basename(path) == "SKIPPED_SAMPLES.txt", path)
        text = open(path, encoding="utf-8").read()
        _assert("TRAINING SKIPPED 2 SAMPLE" in text, text[:200])
        _assert("p347_x" in text and "non-manifold" in text, text)
        _assert(os.path.isfile(os.path.join(root, "data", "skipped_cache.json")), "json")
        write_run_readme(root)
        readme = open(os.path.join(root, "README.txt"), encoding="utf-8").read()
        _assert("SAMPLES WERE SKIPPED" in readme, readme[:250])
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_warmup_result_skips_unless_strict():
    from dataset import AneurysmDataset

    errors = [
        ("p347_FAQEBBwJCx8WEwMDExgKAAMd", "ValueError: template coarse: 2 non-manifold edges"),
        ("p391_GgAcPhQdLQIbAAgKDAAAOhEN", "ValueError: template mid: 3 non-manifold edges"),
    ]
    out = AneurysmDataset._warmup_result(None, 695, list(range(695)), errors, strict=False)
    _assert(len(out) == 2, out)
    raised = False
    try:
        AneurysmDataset._warmup_result(None, 695, list(range(695)), errors, strict=True)
    except RuntimeError as exc:
        raised = True
        _assert("2/695" in str(exc), str(exc))
    _assert(raised, "strict=True must raise")
    raised = False
    try:
        AneurysmDataset._warmup_result(None, 2, [0, 1], errors, strict=False)
    except RuntimeError as exc:
        raised = True
        _assert("no usable" in str(exc), str(exc))
    _assert(raised, "zero usable caches must raise")


def test_dir_has_vtp_nested_cleandata_layout():
    from hpc_runtime import _dir_has_vtp

    root = tempfile.mkdtemp(prefix="aneux_vtp_")
    try:
        _assert(not _dir_has_vtp(root), "empty dir")
        nested = os.path.join(root, "uniformly_remeshed")
        os.makedirs(nested)
        _assert(not _dir_has_vtp(root), "subdir without vtp")
        with open(os.path.join(nested, "case.vtp"), "w", encoding="utf-8") as handle:
            handle.write("not a real mesh")
        _assert(_dir_has_vtp(root), "nested vtp must count")
        _assert(_dir_has_vtp(nested), "direct vtp must count")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_destroy_distributed_without_process_group():
    from dist_utils import destroy_distributed

    destroy_distributed()


def test_persist_if_remote_skips_same_path():
    from hpc_runtime import persist_if_remote

    root = tempfile.mkdtemp(prefix="aneux_persist_")
    try:
        _assert(persist_if_remote(root, root) == os.path.abspath(root), "same path")
        _assert(persist_if_remote(root, None) is None, "missing dest")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_geco_beta_not_clipped_to_zero_at_epoch_one():
    import config as cfg
    from losses import geco_beta_max_for_epoch, update_geco_beta

    hi1 = geco_beta_max_for_epoch(1, beta_max=10.0, warmup_epochs=20)
    _assert(hi1 >= cfg.GECO_BETA_MIN, hi1)
    _assert(abs(hi1 - cfg.GECO_BETA_INIT) < 1e-9, hi1)
    hi20 = geco_beta_max_for_epoch(20, beta_max=10.0, warmup_epochs=20)
    _assert(abs(hi20 - 10.0) < 1e-9, hi20)
    # KL below R* shrinks β slightly; a zero ceiling used to force β=0.
    b = update_geco_beta(1.0, kl_mean_raw=5.45, epoch=1, warmup_epochs=20)
    _assert(float(b) >= cfg.GECO_BETA_MIN, b)
    _assert(0.5 < float(b) <= hi1 + 1e-12, (b, hi1))


def test_scale_hpc_workers_follows_gpus():
    from hpc_runtime import scale_hpc_workers

    full = scale_hpc_workers(n_gpu=4, n_cpu=64)
    _assert(full["num_workers"] == 30, full)
    _assert(full["cache_build_workers"] == 63, full)
    _assert(full["cpus_per_gpu"] == 16, full)

    one = scale_hpc_workers(n_gpu=1, n_cpu=16)
    _assert(one["num_workers"] == 30, one)
    _assert(one["cache_build_workers"] == 30, one)

    tiny = scale_hpc_workers(n_gpu=1, n_cpu=1)
    _assert(tiny["num_workers"] == 0, tiny)
    _assert(tiny["cache_build_workers"] == 1, tiny)


def test_resource_monitor_snapshots_on_this_os():
    from resource_monitor import ResourceMonitor, _cpu_times, _loadavg, _meminfo_gib

    _assert(isinstance(_loadavg(), dict), "loadavg must not raise")
    mem = _meminfo_gib()
    _assert("mem_total_gib" in mem, mem)
    _assert(float(mem["mem_total_gib"]) > 0, mem)
    cpu = _cpu_times()
    _assert(cpu is not None and cpu[0] > 0, cpu)
    root = tempfile.mkdtemp(prefix="aneux_mon_")
    try:
        mon = ResourceMonitor(root, interval=5.0, disk_path=root)
        row = mon.take(write=True)
        _assert("elapsed_s" in row, row)
        _assert(os.path.isfile(mon.jsonl_path), mon.jsonl_path)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _mirror_pair_batch():
    import copy
    from torch_geometric.loader import DataLoader
    from config import FOLLOW_BATCH

    d = make_synthetic_data()
    d.theta = torch.remainder(d.theta + 1.3 + math.pi, 2 * math.pi) - math.pi  # no symmetric theta origin
    b = next(iter(DataLoader([d, copy.deepcopy(d)], batch_size=2, follow_batch=FOLLOW_BATCH)))
    return b


def _mirror_invariants(bt, g):
    """Handedness-free facts of graph g: all must survive a reflection unchanged."""
    from train import _wrap_pi

    mv = bt.batch == g
    x, n, t, bn, th = bt.x, bt.normal, bt.tangent, bt.binormal, bt.theta
    f = bt.face[:, mv[bt.face[0]]]
    fn = torch.cross(x[f[1]] - x[f[0]], x[f[2]] - x[f[0]], dim=-1)
    outward = ((fn * (n[f[0]] + n[f[1]] + n[f[2]])).sum(-1) > 0).float().mean()
    # +1 / -1: b = n x t (template path) or t x n (generated tubes); a mirror must keep it
    hand = (torch.cross(n[mv], t[mv], dim=-1) * bn[mv]).sum(-1).sign().mean()
    e = bt.edge_index[:, mv[bt.edge_index[0]]]
    dth = _wrap_pi(th[e[1]] - th[e[0]])
    ring = (dth.abs() > 1e-4) & (dth.abs() < 1.0) & ((bt.u[e[1]] - bt.u[e[0]]).abs() < 1e-5)
    dp = x[e[1]] - x[e[0]]
    # theta grows along t x n in either binormal convention
    t_x_n = torch.cross(t[e[0][ring]], n[e[0][ring]], dim=-1)
    theta_dir = (torch.sign(dth[ring]) == torch.sign((dp[ring] * t_x_n).sum(-1))).float().mean()
    gm = bt.gt_points_batch == g
    tpl2gt = torch.cdist(x[mv], bt.gt_points[gm]).min(dim=1).values.mean()
    return dict(outward=float(outward), hand=float(hand), theta_dir=float(theta_dir),
                n_ring=int(ring.sum()), tpl2gt=float(tpl2gt))


def test_mirror_reflects_one_graph_consistently():
    """A mirrored graph keeps outward winding, right-handed frames, theta direction and its GT fit."""
    import copy
    from train import apply_mirror

    b0 = _mirror_pair_batch()
    b = copy.deepcopy(b0)
    _, flip = apply_mirror(b, flip=[True, False])
    _assert(flip.tolist() == [True, False], flip)
    before, after = _mirror_invariants(b0, 0), _mirror_invariants(b, 0)
    _assert(before["n_ring"] > 100 and before["outward"] > 0.95 and abs(before["hand"]) == 1.0
            and before["theta_dir"] > 0.95, before)
    for k in ("outward", "hand", "theta_dir", "n_ring"):
        _assert(before[k] == after[k], (k, before, after))
    _assert(abs(before["tpl2gt"] - after["tpl2gt"]) < 1e-5, (before, after))

    m0 = b0.batch == 0
    _assert(torch.equal(b.x[m0], b0.x[m0] * torch.tensor([-1.0, 1.0, 1.0])), "x not reflected")
    _assert(torch.equal(b.binormal[m0], b0.binormal[m0] * torch.tensor([1.0, -1.0, -1.0])), "binormal")
    _assert(torch.equal(b.theta[m0], -b0.theta[m0]), "theta not negated")
    lp = b0.latent_pos_batch == 0
    _assert(torch.equal(b.latent_pos[lp, 0], -b0.latent_pos[lp, 0]), "latent_pos not reflected")
    _assert(torch.equal(b.cl_dense[b0.cl_dense_batch == 0, 3], b0.cl_dense[b0.cl_dense_batch == 0, 3]),
            "centerline radius must not change")
    _assert(abs(float(torch.linalg.det(b.pose_R.reshape(2, 3, 3)[0])) + 1.0) < 1e-5, "pose_R lost M")

    m1 = b0.batch == 1
    for key in ("x", "normal", "binormal", "theta", "tangent"):
        _assert(torch.equal(getattr(b, key)[m1], getattr(b0, key)[m1]), f"unflipped graph changed: {key}")


def test_mirror_twice_is_identity():
    import copy
    from train import apply_mirror

    b0 = _mirror_pair_batch()
    b0.theta[0] = -math.pi  # the one angle that leaves [-pi, pi) when negated
    b = copy.deepcopy(b0)
    apply_mirror(b, flip=[True, True])
    _assert(bool(b.theta[0] == b0.theta[0]), float(b.theta[0]))  # -pi maps to itself
    apply_mirror(b, flip=[True, True])
    for key, v in b0.items():
        if torch.is_tensor(v):
            _assert(torch.equal(v, b[key]), f"{key} not restored")
    raised = False
    try:
        apply_mirror(b, flip=[True])
    except ValueError:
        raised = True
    _assert(raised, "a flip mask of the wrong length must raise")


def test_add_meter_accepts_fold_and_stretch():
    from train import _add_meter, _flush_meters, _zero_tensor_meters

    acc = _zero_tensor_meters()
    _add_meter(acc, "loss", torch.tensor(1.0), 2.0)
    _add_meter(acc, "fold", torch.tensor(0.5), 2.0)
    _add_meter(acc, "stretch", torch.tensor(1.5), 2.0)
    out = _flush_meters(acc, 2.0)
    _assert(abs(out["fold"] - 0.5) < 1e-6, out)
    _assert(abs(out["stretch"] - 1.5) < 1e-6, out)


def test_chamfer_weight_cap():
    w = chamfer_distance_weights(torch.tensor([0.0, 2.0, 20.0]), radius=2.0, cap=4.0)
    _assert(torch.allclose(w[0], torch.tensor(1.0)), f"zero dist {w[0]}")
    _assert(torch.allclose(w[1], torch.tensor(2.0)), f"d=R {w[1]}")
    _assert(float(w[2]) == 4.0, f"cap failed {w[2]}")
    w2 = chamfer_distance_weights(torch.tensor([8.0]), radius=2.0, cap=4.0)
    _assert(float(w2[0]) == 4.0, "1+(d/R) must clamp at cap")
    dist = torch.tensor([0.0, 1.0, 2.0])
    r_loc = torch.tensor([1.0, 2.0, 0.5])
    wt = chamfer_distance_weights(dist, radius=r_loc, cap=8.0)
    _assert(torch.allclose(wt, torch.tensor([1.0, 1.5, 5.0])), f"tensor r_local weights {wt}")
    x = torch.tensor([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    tube = torch.zeros(2, 3)
    nrm = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    r_vec = torch.tensor([1.5, 2.5])
    cr = composed_radius(x, tube, nrm, r_vec)
    _assert(torch.allclose(cr, torch.tensor([4.5, 6.5])), f"composed_radius r_local {cr}")


def test_huber_and_radial_loss():
    diff = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    h = huber(diff, delta=1.0)
    _assert(torch.allclose(h[2], torch.tensor(0.0)), "Huber(0)")
    _assert(torch.allclose(h[1], torch.tensor(0.125)), f"quad {h[1]}")
    _assert(torch.allclose(h[0], torch.tensor(1.5)), f"lin {h[0]}")
    r_pred = torch.tensor([2.0, 3.0, 8.0])
    r_star = torch.tensor([2.0, 3.5, 7.0])
    valid = torch.tensor([True, True, False])
    loss = radial_huber_loss(r_pred, r_star, valid, delta=1.0)
    _assert(float(loss) > 0.0, "radial loss on valid verts")
    loss0 = radial_huber_loss(r_pred, r_star, torch.zeros(3, dtype=torch.bool), delta=1.0)
    _assert(float(loss0) == 0.0, "all-invalid radial must be 0")


def test_mesh_r_star_stats_stay_in_millimetres():
    """A short sac edge with a small |Δr*| must not look like a steep neck."""
    pos = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float64)
    r_star = np.array([2.0, 2.4], dtype=np.float64)
    valid = np.array([True, True])
    edges = np.array([[0, 1]], dtype=np.int64)
    dth, du, med = mesh_r_star_edge_stats(pos, r_star, valid, edges)
    _assert(abs(float(dth[0]) - 0.4) < 1e-6, dth)
    _assert(float(du[0]) == 0.0, du)
    _assert(abs(float(med[0]) - 2.2) < 1e-6, med)
    src = torch.tensor([0])
    dst = torch.tensor([1])
    w = smoothness_edge_weights(
        src,
        dst,
        torch.tensor(r_star),
        torch.tensor(valid),
        torch.tensor(dth),
        torch.tensor(du),
        torch.tensor(med),
    )
    _assert(float(w[0]) > 0.7, f"0.4 mm step must stay coupled, got {float(w[0])}")


def test_kl_penalty_is_excess_over_target():
    mu = torch.zeros(1, 2, 4)
    logvar = torch.zeros(1, 2, 4)
    under, info = vae_kl_loss(mu, logvar, beta=1.0)
    _assert(float(info["kl_mean_raw"]) < 12.0, info["kl_mean_raw"])
    _assert(float(under) == 0.0, f"under-target KL must not be penalised, got {float(under)}")
    mu_hi = torch.full((1, 2, 4), 3.0)
    over, info_hi = vae_kl_loss(mu_hi, logvar, beta=1.0)
    _assert(float(info_hi["kl_mean_raw"]) > 12.0, info_hi["kl_mean_raw"])
    _assert(float(over) > 0.0, over)


def test_smoothness_edge_weights():
    r_star = torch.tensor([2.0, 2.0, 8.0, 8.0])
    valid = torch.tensor([True, True, True, True])
    r_dth = torch.tensor([0.0, 0.0, 4.0, 4.0])
    r_du = torch.tensor([0.0, 0.0, 0.0, 0.0])
    ring = torch.tensor([2.0, 2.0, 8.0, 8.0])
    src = torch.tensor([0, 2])
    dst = torch.tensor([1, 3])
    w = smoothness_edge_weights(src, dst, r_star, valid, r_dth, r_du, ring)
    _assert(float(w[0]) > float(w[1]), f"neck should soften: {w}")
    invalid = torch.tensor([False, False, True, True])
    w_inv = smoothness_edge_weights(src, dst, r_star, invalid, r_dth, r_du, ring)
    _assert(float(w_inv[0]) == 1.0, "invalid endpoints fall back to w=1")
    amb = torch.tensor([False, False, True, False])
    valid_amb = torch.tensor([True, True, False, True])
    w_amb = smoothness_edge_weights(
        src, dst, r_star, valid_amb, r_dth, r_du, ring, ambiguous=amb
    )
    _assert(abs(float(w_amb[1]) - SMOOTH_W_AMBIGUOUS) < 1e-6, f"ambiguous should unlock {w_amb}")
    _assert(float(w_amb[0]) > 0.5, "valid parent edge should stay stiff")


def test_r_star_hit_selection_and_voronoi():
    _assert(voronoi_ok_hit(0, 0.5, 0, 0.5, arc_mm=20.0, max_ds_mm=2.0), "same station")
    _assert(not voronoi_ok_hit(1, 0.5, 0, 0.5, arc_mm=20.0, max_ds_mm=2.0), "cross-tract")
    _assert(not voronoi_ok_hit(0, 0.9, 0, 0.1, arc_mm=20.0, max_ds_mm=2.0), "siphon jump")

    hits = [
        {"t": 2.1, "voronoi_ok": True, "normal_dot": 0.9},
        {"t": 12.0, "voronoi_ok": True, "normal_dot": 0.9},
    ]
    val, ok, amb = select_r_star_from_hits(hits, ambiguous_mm=4.0)
    _assert(not ok and amb, "far double-hit should be ambiguous")

    hits2 = [
        {"t": -0.4, "voronoi_ok": True, "normal_dot": 0.9},
        {"t": 2.2, "voronoi_ok": True, "normal_dot": 0.9},
    ]
    val2, ok2, amb2 = select_r_star_from_hits(hits2, ambiguous_mm=4.0)
    _assert(ok2 and not amb2 and abs(val2 - 2.2) < 1e-6, f"prefer outward {val2}")

    hits3 = [{"t": 6.5, "voronoi_ok": True, "normal_dot": -0.95}]
    sign = choose_normal_sign([hits3])
    val3, ok3, amb3 = select_r_star_from_hits(hits3, normal_sign=sign)
    _assert(ok3 and not amb3 and abs(val3 - 6.5) < 1e-6, f"flipped normals {sign}, {val3}")
    val_miss, ok_miss, amb_miss = select_r_star_from_hits([])
    _assert(not ok_miss and not amb_miss, "no-hit is a miss, not a crease")


def test_r_star_grid_stats_and_cylinder_raycast():
    r = np.array([2.0, 2.0, 2.0, 2.0, 5.0, 5.0, 5.0, 5.0], dtype=np.float64)
    valid = np.ones(8, dtype=bool)
    dth, du, med = r_star_grid_stats(r, valid, [2], 4)
    _assert(float(dth.max()) < 1e-8, f"uniform ring Δθ {dth}")
    _assert(float(du[0]) == 3.0 and float(du[4]) == 3.0, f"longitudinal jump {du}")
    _assert(float(med[0]) == 2.0 and float(med[4]) == 5.0, f"ring med {med}")

    z = np.linspace(0.0, 20.0, 40)
    cl = np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)
    cylinder = pv.Cylinder(
        center=(0.0, 0.0, 10.0),
        direction=(0.0, 0.0, 1.0),
        radius=3.0,
        height=20.0,
        resolution=24,
        capping=True,
    ).triangulate()
    dense = {
        "xyz": cl,
        "u": np.linspace(0.0, 1.0, len(cl)),
        "arc": 20.0,
        "t": np.tile([0.0, 0.0, 1.0], (len(cl), 1)),
        "n": np.tile([1.0, 0.0, 0.0], (len(cl), 1)),
        "b": np.tile([0.0, 1.0, 0.0], (len(cl), 1)),
    }
    factory = _make_factory(radius=2.0, hierarchy=((8, 4), (16, 8), (24, 8)), latent_len=8)
    level = factory._generate_level([dense], 24, 8, [20.0])
    packed = compute_level_r_star(
        level["pos"].numpy(),
        level["normal"].numpy(),
        level["u"].numpy(),
        level["tract_id"].numpy(),
        level["branch_nl"].numpy(),
        int(level["n_radial"].item()),
        [dense],
        cylinder,
        tube_radius=2.0,
    )
    valid_frac = float(packed["valid"].mean())
    _assert(valid_frac > 0.5, f"cylinder valid frac {valid_frac}")
    if packed["valid"].any():
        mean_r = float(packed["r_star"][packed["valid"]].mean())
        _assert(2.4 < mean_r < 3.6, f"expected ~3 mm wall, got {mean_r}")


def test_coarse_attn_per_graph_and_finite():
    torch.manual_seed(0)
    hidden = 16
    attn = CoarsePositionalSelfAttention(hidden, n_heads=4, tract_emb_dim=8)
    attn.eval()
    torch.nn.init.xavier_uniform_(attn.out.weight)
    gate = float(torch.sigmoid(attn.alpha_raw).clamp(max=attn.gate_max).detach())
    _assert(0.0 < gate <= 0.5 + 1e-6, f"coarse gate {gate}")
    n0, n1 = 12, 10
    h = torch.randn(n0 + n1, hidden)
    u = torch.rand(n0 + n1)
    th = (torch.rand(n0 + n1) * 2 * math.pi) - math.pi
    tract = torch.cat([torch.zeros(n0), torch.ones(n1)]).long()
    batch = torch.cat([torch.zeros(n0), torch.ones(n1)]).long()
    with torch.no_grad():
        out = attn(h, u, th, tract, batch)
        out0 = attn(h[:n0], u[:n0], th[:n0], tract[:n0], torch.zeros(n0, dtype=torch.long))
        out1 = attn(h[n0:], u[n0:], th[n0:], tract[n0:], torch.zeros(n1, dtype=torch.long))
    _assert(torch.isfinite(out).all(), "coarse attn non-finite")
    _assert(torch.allclose(out[:n0], out0, atol=1e-5, rtol=1e-5), "graph 0 mixed")
    _assert(torch.allclose(out[n0:], out1, atol=1e-5, rtol=1e-5), "graph 1 mixed")


def _token_pack(u, tract, is_junc):
    n_graphs, latent_len = u.shape
    attend = torch.zeros(n_graphs, latent_len, MAX_TRACTS, dtype=torch.bool)
    for g in range(n_graphs):
        for i, tid in enumerate(tract[g].tolist()):
            if 0 <= int(tid) < MAX_TRACTS:
                attend[g, i, int(tid)] = True
    pack = type("Tok", (), {})()
    pack.latent_u = u
    pack.latent_tract_id = tract
    pack.latent_is_junction = is_junc
    pack.token_attend = attend
    return pack


def test_z_attn_per_tract_and_gate():
    torch.manual_seed(3)
    layer = LatentTractSelfAttention(latent_dim=8, n_heads=4)
    layer.eval()
    gate = float(torch.sigmoid(layer.alpha_raw).clamp(max=Z_ATTN_GATE_MAX).detach())
    _assert(0.0 < gate <= Z_ATTN_GATE_MAX + 1e-6, f"Z gate {gate}")
    _assert(abs(float(torch.sigmoid(layer.alpha_raw).detach()) - Z_ATTN_ALPHA_INIT) < 1e-4, f"Z gate init")
    u = torch.linspace(0.0, 1.0, 8).view(1, 8).repeat(2, 1)
    tract = torch.tensor([[0, 0, 0, 1, 1, 1, -1, -1], [0, 0, 0, 1, 1, 1, -1, -1]])
    is_junc = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 1, 1]])
    z = torch.randn(2, 8, 8)
    data = _token_pack(u, tract, is_junc)
    with torch.no_grad():
        ident = layer(z, data)
    _assert(torch.allclose(ident, z, atol=1e-5), "zero-init W_O must leave z unchanged")
    torch.nn.init.xavier_uniform_(layer.out.weight)
    with torch.no_grad():
        out = layer(z, data)
        z_zero = z.clone()
        z_zero[:, 3:6] = 0.0
        out_zero = layer(z_zero, data)
        out0 = layer(z[0:1], _token_pack(u[0:1], tract[0:1], is_junc[0:1]))
        out1 = layer(z[1:2], _token_pack(u[1:2], tract[1:2], is_junc[1:2]))
        u_long = torch.linspace(0.0, 1.0, 8).view(1, 8)
        tract_long = torch.tensor([[0, 0, 0, 0, 0, 0, -1, -1]])
        junc_long = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1]])
        z_long = torch.randn(1, 8, 8)
        pack_long = _token_pack(u_long, tract_long, junc_long)
        z_far = z_long.clone()
        z_far[:, 5] = 0.0
        near = layer(z_long, pack_long)
        near_far = layer(z_far, pack_long)
    _assert(torch.isfinite(out).all(), "Z attn non-finite")
    _assert(torch.allclose(out[:, :3], out_zero[:, :3], atol=1e-5, rtol=1e-5), "tract 0 mixed with tract 1")
    _assert(torch.allclose(out[:, 6:], z[:, 6:], atol=1e-6), "junction tokens must be unchanged")
    _assert(torch.allclose(out[0], out0[0], atol=1e-5, rtol=1e-5), "graph 0 mixed")
    _assert(torch.allclose(out[1], out1[0], atol=1e-5, rtol=1e-5), "graph 1 mixed")
    _assert(torch.allclose(near[:, :3], near_far[:, :3], atol=1e-5, rtol=1e-5), "local window leaked far token")


def test_true_normals_from_mesh():
    factory = _make_factory()
    branch = _straight_branch(offset=(0.0, 0.0, 0.0))
    cylinder = pv.Cylinder(
        center=(0.0, 0.0, 10.0),
        direction=(0.0, 0.0, 1.0),
        radius=3.0,
        height=20.0,
        resolution=24,
        capping=True,
    ).triangulate()
    data = factory.build_scaffold(_polyline_mesh(branch), vessel_mesh=cylinder)
    _assert(data.x_true_normal.shape == data.x_true.shape, data.x_true_normal.shape)
    nrm = data.x_true_normal.norm(dim=-1)
    _assert(float(nrm.mean()) > 0.5, f"mesh normals collapsed {float(nrm.mean())}")
    _assert(torch.isfinite(data.x_true_normal).all(), "non-finite GT normals")


def test_gated_hidden_upsample_shapes():
    data = make_synthetic_data()
    loader = DataLoader([data], batch_size=1, follow_batch=FOLLOW_BATCH)
    batch = next(iter(loader))
    model = _tiny_model()
    model.eval()
    dec = model.decoder
    _assert(0.0 < float(torch.sigmoid(dec.alpha_c_raw).detach()) < 1.0, "alpha_c must be bounded")
    _assert(0.0 < float(torch.sigmoid(dec.alpha_m_raw).detach()) < 1.0, "alpha_m must be bounded")
    _assert(abs(float(torch.sigmoid(dec.alpha_c_raw).detach()) - SKIP_GATE_INIT) < 1e-4, "alpha init")
    n_graphs = 1
    batch_c = _attr_batch(batch, "pos_coarse", batch.pos_coarse.size(0))
    batch_m = _attr_batch(batch, "pos_mid", batch.pos_mid.size(0))
    batch_f = batch.batch if getattr(batch, "batch", None) is not None else _ones_batch(
        batch.x.size(0), batch.x.device
    )
    h_c = torch.randn(batch.pos_coarse.size(0), dec.hidden_dim)
    h_m = torch.randn(batch.pos_mid.size(0), dec.hidden_dim)
    h_c_up = _upsample_level(
        h_c, batch, "n_radial_coarse", "n_radial_mid",
        "branch_nl_coarse", "branch_nl_mid", batch_c, batch_m, n_graphs,
    )
    h_m_up = _upsample_level(
        h_m, batch, "n_radial_mid", "n_radial_fine",
        "branch_nl_mid", "branch_nl_fine", batch_m, batch_f, n_graphs,
    )
    _assert(h_c_up.shape == (batch.pos_mid.size(0), dec.hidden_dim), h_c_up.shape)
    _assert(h_m_up.shape == (batch.x.size(0), dec.hidden_dim), h_m_up.shape)
    with torch.no_grad():
        z = torch.zeros(1, 8, 8)
        x_pred, *_ = dec(z, batch)
    _assert(torch.isfinite(x_pred).all(), "decoder with gated skips non-finite")


def test_ema_update_and_restore():
    model = _tiny_model()
    ema = ModelEMA(model, decay=0.5)
    before = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}
    with torch.no_grad():
        for p in model.parameters():
            if p.dtype.is_floating_point:
                p.add_(1.0)
    ema.update(model)
    key = next(iter(before))
    shadow = ema.shadow[key]
    expected = 0.5 * before[key] + 0.5 * (before[key] + 1.0)
    _assert(torch.allclose(shadow, expected, atol=1e-5), "EMA update")
    ema.store(model)
    ema.copy_to(model)
    _assert(torch.allclose(model.state_dict()[key], shadow), "EMA copy_to")
    ema.restore(model)
    _assert(torch.allclose(model.state_dict()[key], before[key] + 1.0), "EMA restore")


def test_full_capacity_model_forward():
    data = make_synthetic_data(latent_len=8)
    loader = DataLoader([data], batch_size=1, follow_batch=FOLLOW_BATCH)
    batch = next(iter(loader))
    model = GraphVAE(
        latent_dim=LATENT_DIM,
        latent_len=8,
        hidden_dim=DECODER_HIDDEN_DIM,
        tube_radius=2.0,
        sa_stages=TINY_SA,
    )
    model.train()
    out = model(batch)
    _assert(out.mu.shape[-1] == LATENT_DIM, out.mu.shape)
    _assert(out.x_pred.shape == batch.x.shape, out.x_pred.shape)
    _assert(len(model.decoder.mid_convs) == N_CONV_PER_LEVEL, len(model.decoder.mid_convs))
    _assert(len(model.decoder.fine_convs) == N_CONV_PER_LEVEL, len(model.decoder.fine_convs))
    _assert(len(model.decoder.coarse_convs) == N_CONV_PER_LEVEL, len(model.decoder.coarse_convs))
    kraw = model.decoder.fine_convs[0].conv.kernel_size
    ks = tuple(int(v) for v in (kraw.tolist() if hasattr(kraw, "tolist") else kraw))
    _assert(ks == tuple(SPLINE_KERNEL_SIZE) == (5, 5, 3), ks)
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
        r_star=batch.r_star,
        r_star_valid=batch.r_star_valid,
        r_dth=batch.r_dth,
        r_du=batch.r_du,
        r_ring_med=batch.r_ring_med,
        r_star_mid=batch.r_star_mid,
        r_star_valid_mid=batch.r_star_valid_mid,
        normal=batch.normal,
        normal_mid=batch.normal_mid,
        pos_mid=batch.pos_mid,
        x_true_normal=getattr(batch, "x_true_normal", None),
    )
    loss = sum(terms.values())
    loss.backward()
    _assert(torch.isfinite(loss), f"non-finite loss {loss}")
    grads = [p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None]
    _assert(len(grads) > 0 and sum(grads) > 0, "no gradients on capacity model")


def test_coaxial_rstar_healthy_and_bulge_ray():
    """§13: coaxial cylinders keep exact healthy r*; bulge r* follows the ray."""
    factory = _make_factory(n_true=64, latent_len=16)
    cl_pts = np.stack([np.zeros(41), np.zeros(41), np.linspace(0.0, 20.0, 41)], axis=1)
    cl = _polyline_mesh(cl_pts)
    tpl = _open_cylinder_surface(radius=2.0, height=20.0, n_th=24, n_z=40, bulge=0.0)
    gt_h = _open_cylinder_surface(radius=2.4, height=20.0, n_th=28, n_z=44, bulge=0.0)
    data_h = factory.build_scaffold(cl, vessel_mesh=gt_h, template_mesh=tpl)
    valid_h = data_h.r_star_valid.numpy()
    interior = (data_h.u.numpy() > 0.15) & (data_h.u.numpy() < 0.85)
    sel = valid_h & interior
    _assert(int(sel.sum()) > 20, f"healthy valid interior {int(sel.sum())}")
    mean_r = float(data_h.r_star.numpy()[sel].mean())
    _assert(2.2 < mean_r < 2.6, f"healthy coaxial r* should be ~2.4 mm, got {mean_r}")

    gt_b = _open_cylinder_surface(radius=2.4, height=20.0, n_th=28, n_z=44, bulge=6.0)
    data_b = factory.build_scaffold(cl, vessel_mesh=gt_b, template_mesh=tpl)
    origin = data_b.origin_shift.detach().cpu().numpy().reshape(3)
    rot = data_b.pose_R.detach().cpu().numpy().reshape(3, 3)
    posed_gt = transform_vessel_mesh(gt_b, origin, rot)
    r_loc = data_b.r_local.numpy()
    packed = template_ray_r_star(
        data_b.x.numpy(), data_b.normal.numpy(), r_loc, posed_gt
    )
    valid_b = data_b.r_star_valid.numpy() & packed["valid"]
    interior_b = (data_b.u.numpy() > 0.15) & (data_b.u.numpy() < 0.85)
    live = valid_b & interior_b
    _assert(int(live.sum()) > 20, f"bulge valid interior {int(live.sum())}")
    err = np.abs(data_b.r_star.numpy()[live] - packed["r_star"][live])
    _assert(float(np.median(err)) < 0.15, f"r* vs ray median err {float(np.median(err))}")
    bulge = live & (packed["r_star"] > 3.5)
    _assert(int(bulge.sum()) > 3, f"expected ray hits under the bulge, got {int(bulge.sum())}")
    bulge_r = float(data_b.r_star.numpy()[bulge].mean())
    _assert(bulge_r > 3.5, f"bulge r* should exceed healthy 2.4, got {bulge_r}")
    from scipy.spatial import cKDTree

    _, nn = cKDTree(np.asarray(posed_gt.points)).query(data_b.x.numpy()[bulge], k=1)
    nearest_r = r_loc[bulge] + np.linalg.norm(
        np.asarray(posed_gt.points)[nn] - data_b.x.numpy()[bulge], axis=1
    )
    ray_r = packed["r_star"][bulge]
    stored = data_b.r_star.numpy()[bulge]
    ray_err = np.abs(stored - ray_r)
    nn_err = np.abs(stored - nearest_r)
    _assert(
        float(np.median(ray_err)) <= float(np.median(nn_err)) + 0.05,
        f"r* under the bulge must follow the ray, not nearest-vertex "
        f"(ray median {float(np.median(ray_err)):.3f} vs nn {float(np.median(nn_err)):.3f})",
    )


def test_scaffold_normals_point_away_from_centerline():
    factory = _make_factory(n_true=64, latent_len=8)
    cl, tpl, gt = _template_cylinder_pair()
    data = factory.build_scaffold(cl, vessel_mesh=gt, template_mesh=tpl)
    cl_xyz = data.cl_dense.numpy()[:, :3]
    from scipy.spatial import cKDTree

    def _outward_frac(pos, nrm):
        _, idx = cKDTree(cl_xyz).query(pos, k=1)
        radial = pos - cl_xyz[idx]
        dots = np.einsum("ij,ij->i", nrm, radial)
        return float(np.mean(dots >= -1e-5)), float(dots.min())

    frac_f, min_f = _outward_frac(data.x.numpy(), data.normal.numpy())
    frac_m, min_m = _outward_frac(data.pos_mid.numpy(), data.normal_mid.numpy())
    frac_c, min_c = _outward_frac(data.pos_coarse.numpy(), data.normal_coarse.numpy())
    _assert(frac_f > 0.98, f"fine outward {frac_f} min={min_f}")
    _assert(frac_m > 0.98, f"mid outward {frac_m} min={min_m}")
    _assert(frac_c > 0.98, f"coarse outward {frac_c} min={min_c}")

    inward = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    pos = np.array([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]], dtype=np.float64)
    cl_line = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    flipped = factory._orient_normals_outward(inward, pos, cl_line)
    _assert(float(flipped[0, 0]) > 0.5, flipped[0])
    _assert(float(flipped[1, 0]) < -0.5, flipped[1])


def test_groupid_require_raises_without_arrays():
    mesh = _polylines_mesh(list(_y_paths()))
    raised = False
    try:
        extract_groupid_tracts(mesh, require_groupids=True)
    except ValueError as exc:
        raised = "GroupIds" in str(exc)
    _assert(raised, "require_groupids=True must refuse the unique-tract fallback")


def test_groupid_cell_arrays_blank_and_near_coincident():
    mesh = _cell_y_centerline()
    tracts, _endpoints, junctions = extract_groupid_tracts(mesh, require_groupids=True)
    arcs = sorted((_arc_len(t) for t in tracts), reverse=True)
    _assert(len(tracts) == 3, f"cell-array Y should be 3 tracts, got {len(tracts)} arcs={arcs}")
    _assert(len(junctions) == 1, f"expected 1 junction, got {junctions}")
    _assert(abs(arcs[0] - 88.8) < 1.5, f"parent arc {arcs[0]}")
    _assert(min(arcs[1], arcs[2]) > 13.0, f"blanked 1.6 mm should attach to daughters {arcs[1:]}")


def test_groupid_snf_unclipped_fixture():
    path = _first_existing(_snf_unclipped_paths())
    if path is None:
        raise SkipTest("no SNF00000100 unclipped centerline fixture under scratch/")
    mesh = pv.read(path)
    tracts, _endpoints, junctions = extract_groupid_tracts(mesh, require_groupids=True)
    _assert(len(tracts) == 3, f"{os.path.basename(path)} tracts {len(tracts)}")
    _assert(len(junctions) == 1, f"{os.path.basename(path)} junctions {junctions}")


def test_smoothness_weights_not_all_ones_on_bulge():
    factory = _make_factory(n_true=64, latent_len=16)
    cl_pts = np.stack([np.zeros(41), np.zeros(41), np.linspace(0.0, 40.0, 41)], axis=1)
    cl = _polyline_mesh(cl_pts)
    tpl = _open_cylinder_surface(radius=2.0, height=40.0, n_th=24, n_z=40, bulge=0.0)
    gt = _open_cylinder_surface(radius=2.0, height=40.0, n_th=24, n_z=40, bulge=8.0)
    data = factory.build_scaffold(cl, vessel_mesh=gt, template_mesh=tpl)
    src, dst = data.edge_index
    w = smoothness_edge_weights(
        src, dst, data.r_star, data.r_star_valid, data.r_dth, data.r_du, data.r_ring_med,
        ambiguous=data.r_star_ambiguous,
    )
    _assert(float(w.min()) < 1.0 - 1e-6, f"bulge smoothness weights were all ones: min={float(w.min())}")
    _assert(float((w < 1.0 - 1e-6).float().mean()) > 0.01, "too few softened edges on the bulge")


def test_token_spacing_independent_of_length():
    factory = _make_factory(latent_len=128)
    for length in (40.0, 90.0):
        z = np.linspace(0.0, length, max(8, int(length)))
        pts = np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)
        dense = factory._fit_dense_tract(pts)
        tok = factory._build_latent_tokens([dense], {}, {}, [float(dense["arc"])])
        valid = tok["latent_valid"].numpy()
        pos = tok["latent_pos"].numpy()[valid]
        n_tok = int(valid.sum())
        expect = int(np.floor(float(dense["arc"]) / TOKEN_SPACING_MM) + 1)
        _assert(n_tok == expect, f"L={length}: n_tok {n_tok} vs floor(L/2)+1={expect}")
        if n_tok > 1:
            ds = np.linalg.norm(np.diff(pos, axis=0), axis=1)
            _assert(abs(float(np.median(ds)) - TOKEN_SPACING_MM) < 0.15, f"L={length} spacing {float(np.median(ds))}")
        _assert(not bool(tok["token_attend"][~tok["latent_valid"]].any()), f"L={length} pad attend")
        _assert(n_tok != 96, "must not spread a fixed 96-slot budget")


def test_ring_neighbour_eth_at_cube_edge():
    data = make_synthetic_data(sac=False)
    r_local = _vertex_r_local(data)
    e = intrinsic_spline_pseudo_coords(
        data.u, data.theta, data.tract_id, data.edge_index, data.u_step,
        r_local=r_local, pos=data.x,
    )
    src, dst = data.edge_index
    same_u = (data.u[src] - data.u[dst]).abs() < 1e-4
    same_tr = data.tract_id[src] == data.tract_id[dst]
    dth = (data.theta[dst] - data.theta[src]).abs()
    ring = same_u & same_tr & (dth > 0.05)
    _assert(bool(ring.any()), "expected circumferential ring edges")
    e_th = e[ring, 1]
    dist01 = torch.minimum(e_th, (1.0 - e_th).abs())
    _assert(
        float(dist01.median()) < 0.12,
        f"ring-neighbour e_th should be ~0 or ~1, median dist={float(dist01.median())} values={e_th[:8]}",
    )


def test_template_levels_keep_loops_density_and_nest():
    """Sizing-field collapse: loops kept, manifold, nested, density ratio kept."""
    from coarsen import build_levels

    factory = _make_factory()
    mesh = _open_cylinder_surface(radius=2.0, height=20.0, n_th=48, n_z=160, bulge=0.0)
    mesh, faces0 = factory._polydata_triangles(mesh)
    pts = np.asarray(mesh.points, dtype=np.float64)
    _n0, _nm0, n_loops0 = factory._face_topology(faces0, pts.shape[0])
    _assert(n_loops0 == 2, f"open cylinder should have 2 loops, got {n_loops0}")
    # a dense band in the middle third, like a sac on a variable template
    edge0 = 2.0 * np.pi * 2.0 / 48
    dense = np.abs(pts[:, 2] - pts[:, 2].mean()) < 20.0 / 6
    tel = np.where(dense, edge0, 3.0 * edge0)
    lv = build_levels(pts, faces0, tel, np.full(pts.shape[0], 2.0))
    for name in ("mid", "coarse"):
        keep, f = lv["keep_" + name], lv["faces_" + name]
        _n, nman, n_loops = factory._face_topology(f, keep.shape[0])
        _assert(nman == 0, f"{name} non-manifold {nman}")
        _assert(n_loops == n_loops0, f"{name} loops {n_loops} vs {n_loops0}")
        _assert(keep.shape[0] < pts.shape[0], f"{name} did not coarsen")
        frac_dense = dense[keep].mean()
        _assert(frac_dense > 0.45, f"{name} lost the dense band: {frac_dense:.2f} of vertices in it")
    _assert(np.isin(lv["keep_coarse"], lv["keep_mid"]).all(), "coarse is not a subset of mid")
    for name, n_dst in (("mid", lv["keep_mid"].shape[0]), ("fine", pts.shape[0])):
        w = lv["upsample_w_" + name]
        _assert(lv["upsample_idx_" + name].shape == (n_dst, 3), f"{name} idx shape")
        _assert(np.allclose(w.sum(1), 1.0) and (w >= 0).all(), f"{name} weights not barycentric")
    # a nested vertex prolongs to itself
    pos_in_mid = np.searchsorted(lv["keep_mid"], lv["keep_mid"])
    w_self = lv["upsample_w_fine"][lv["keep_mid"]]
    i_self = lv["upsample_idx_fine"][lv["keep_mid"]]
    got = (w_self * (i_self == pos_in_mid[:, None])).sum(1)
    _assert(float(got.min()) > 0.999, f"nested vertex weight on itself {got.min():.4f}")


def test_forward_model_passes_sample_through_ddp_wrapper():
    """DDP hides GraphVAE.reparameterize; val-σ must still reach the inner forward."""
    import torch.nn as nn
    from train import _forward_model

    class Inner(nn.Module):
        def forward(self, data, sample=None):
            return sample

    class Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.module = Inner()

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    wrap = Wrap()
    wrap.eval()
    _assert(_forward_model(wrap, None, sample=True) is True, "sample=True")
    _assert(_forward_model(wrap, None, sample=False) is False, "sample=False")


def test_reparameterize_eval_samples_and_logvar_floor():
    model = _tiny_model()
    model.eval()
    mu = torch.zeros(1, 8, 8)
    logvar = torch.full((1, 8, 8), 2.0 * math.log(0.4))
    with torch.no_grad():
        z_mu = model.reparameterize(mu, logvar, sample=False)
        torch.manual_seed(11)
        z_s1 = model.reparameterize(mu, logvar, sample=True)
        torch.manual_seed(11)
        z_s2 = model.reparameterize(mu, logvar, sample=True)
    _assert(torch.allclose(z_mu, mu), "sample=False must return μ")
    _assert(torch.allclose(z_s1, z_s2), "same seed must match")
    _assert(not torch.allclose(z_s1, mu), "eval+sample=True must not always equal μ")
    raw = torch.linspace(-30.0, 30.0, 64)
    lv = _soft_logvar(raw)
    _assert((lv >= math.log(0.01) - 1e-5).all(), float(lv.min()))
    _assert((torch.exp(0.5 * lv) >= SIGMA_MIN - 1e-5).all(), float(torch.exp(0.5 * lv).min()))
    _assert(abs(float(LOGVAR_MIN) - math.log(0.01)) < 1e-5, LOGVAR_MIN)


def test_mixer_before_sampling():
    data = make_synthetic_data(sac=False)
    loader = DataLoader([data], batch_size=1, follow_batch=FOLLOW_BATCH)
    batch = next(iter(loader))
    model = _tiny_model()
    model.eval()
    with torch.no_grad():
        out_id = model(batch, sample=False)
    _assert(torch.allclose(out_id.mu, out_id.mu_raw, atol=1e-5), "zero-init mixer is identity")
    _assert(torch.allclose(out_id.z, out_id.mu, atol=1e-5), "sample=False z is mixed μ")
    torch.nn.init.xavier_uniform_(model.z_attn.out.weight)
    with torch.no_grad():
        out_mix = model(batch, sample=False)
        out_s = model(batch, sample=True)
    _assert(not torch.allclose(out_mix.mu, out_mix.mu_raw, atol=1e-5), "mixer must change μ before sampling")
    _assert(torch.allclose(out_mix.z, out_mix.mu, atol=1e-5), "sample=False z is mixed μ, not raw")
    _assert(bool(out_s.sampled), "sample=True must set sampled")
    _assert(not torch.allclose(out_s.z, out_s.mu, atol=1e-6), "eval+sample=True z differs from mixed μ")
    _assert((out_s.logvar >= math.log(0.01) - 1e-4).all(), float(out_s.logvar.min()))


def test_null_code_decodes_to_template():
    data = make_synthetic_data(sac=False)
    loader = DataLoader([data], batch_size=1, follow_batch=FOLLOW_BATCH)
    batch = next(iter(loader))
    model = _tiny_model()
    model.eval()
    z = torch.zeros(1, 8, 8)
    with torch.no_grad():
        x = model.decode(z, batch)
    _assert(x.shape == batch.x.shape, x.shape)
    _assert(torch.allclose(x, batch.x, atol=1e-3), f"null code Δ max={float((x - batch.x).abs().max())}")

    factory = _make_factory(n_true=64, latent_len=8)
    cl, tpl, gt = _template_cylinder_pair()
    tpl_data = factory.build_scaffold(cl, vessel_mesh=gt, template_mesh=tpl)
    tpl_batch = next(iter(DataLoader([tpl_data], batch_size=1, follow_batch=FOLLOW_BATCH)))
    with torch.no_grad():
        x_tpl = model.decode(z, tpl_batch)
    dmax = float((x_tpl - tpl_batch.x).abs().max())
    _assert(
        dmax < 0.02,
        f"template null-code Δ max={dmax} (identity at init, knn upsample)",
    )


def test_topk_checkpoints_keep_distinct_files():
    """A new best must not rename over the previous rank and erase it."""
    from run_report import TopKCheckpoints

    root = tempfile.mkdtemp(prefix="topk_")
    try:
        keeper = TopKCheckpoints(root, "best_val", k=3)
        for epoch, score in ((1, 4.0), (2, 3.0), (3, 2.0), (4, 1.0)):
            payload = {"epoch": epoch, "score": score, "model": {"w": torch.tensor([score])}}
            _assert(keeper.consider(score, payload, epoch), epoch)
        expect = [(1, 4, 1.0), (2, 3, 2.0), (3, 2, 3.0)]
        for rank, epoch, score in expect:
            path = os.path.join(root, f"best_val_{rank}.pt")
            _assert(os.path.isfile(path), path)
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            _assert(int(ckpt["epoch"]) == epoch, (rank, ckpt.get("epoch"), epoch))
            _assert(abs(float(ckpt["score"]) - score) < 1e-9, (rank, ckpt.get("score"), score))
        mid = {"epoch": 5, "score": 1.5, "model": {"w": torch.tensor([1.5])}}
        _assert(keeper.consider(1.5, mid, 5), "mid insert")
        expect = [(1, 4, 1.0), (2, 5, 1.5), (3, 3, 2.0)]
        for rank, epoch, score in expect:
            ckpt = torch.load(
                os.path.join(root, f"best_val_{rank}.pt"),
                map_location="cpu",
                weights_only=False,
            )
            _assert(int(ckpt["epoch"]) == epoch, (rank, ckpt.get("epoch")))
            _assert(abs(float(ckpt["score"]) - score) < 1e-9, (rank, ckpt.get("score")))
        _assert(not keeper.consider(9.0, {"epoch": 6, "score": 9.0}, 6), "worse score")
        pts = sorted(name for name in os.listdir(root) if name.endswith(".pt"))
        _assert(pts == ["best_val_1.pt", "best_val_2.pt", "best_val_3.pt"], pts)
        temps = [name for name in os.listdir(root) if name.startswith(".")]
        _assert(temps == [], temps)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_rim_normal_term_sees_only_the_projected_slide():
    import train as T
    from losses import rim_normal_residual_loss
    data = make_synthetic_data(latent_len=8)
    batch = next(iter(DataLoader([data], batch_size=1, follow_batch=FOLLOW_BATCH)))
    torch.manual_seed(0)
    model = _tiny_model()
    with torch.no_grad():  # the heads start at zero (identity decode); move them off it
        for prm in model.decoder.parameters():
            prm.add_(0.05 * torch.randn_like(prm))
    out = T._forward_model(model, batch, sample=False)
    _assert(out.rim_removed is not None and len(out.rim_removed) == 3, "decoder returns 3 removed levels")
    any_live = False
    for rem, sfx in zip(out.rim_removed, ("_coarse", "_mid", "")):
        m = getattr(batch, f"boundary_mask{sfx}")
        nrm = torch.nn.functional.normalize(getattr(batch, f"boundary_plane_normal{sfx}"), dim=-1)
        _assert(float(rem.detach()[~m].abs().max()) == 0.0, f"level {sfx or '_fine'}: nothing removed off the rim")
        r = rem[m].detach()
        tang = r - nrm[m] * (r * nrm[m]).sum(-1, keepdim=True)
        _assert(float(tang.abs().max()) < 1e-5, f"level {sfx or '_fine'}: removed part is along the plane normal")
        any_live |= float(r.abs().max()) > 1e-6
    _assert(any_live, "a perturbed decoder slides some rim vertex off its plane")
    terms = T.losses_from_output(out, batch)
    _assert("rim_normal" in terms and float(terms["rim_normal"]) > 0, f"term present: {terms.get('rim_normal')}")
    # nothing else in the objective pulls on this direction; this term must
    model.zero_grad()
    terms["rim_normal"].backward()
    g = sum(float(p.grad.abs().sum()) for p in model.decoder.parameters() if p.grad is not None)
    _assert(g > 0, "the term trains the decoder")
    # value: mean |removed|^2 over each level's rim, summed over levels
    rem = (torch.zeros(4, 3), torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 0.0], [5.0, 5.0, 5.0]]))
    masks = (torch.zeros(4, dtype=torch.bool), torch.tensor([True, True, False]))
    _assert(abs(float(rim_normal_residual_loss(rem, masks)) - 2.0) < 1e-6, "mean over rim only, empty level adds 0")
    w = dict(T.DEFAULT_LOSS_WEIGHTS)
    base = {k: torch.zeros(()) for k in ("recon", "kl", "disp", "lap", "norm")}
    _assert(abs(float(T.weighted_total({**base, "rim_normal": torch.tensor(3.0)}, w)) - 3.0 * w["rim_normal"]) < 1e-6,
            "weighted_total includes rim_normal")


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
        test_knn_weighted_upsample,
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
        test_point_to_plane_chamfer,
        test_postprocess_checkpoint_and_split_summary,
        test_topk_checkpoints_keep_distinct_files,
        test_stretch_identity_on_skinny_triangle,
        test_mesh_losses_match_pytorch3d,
        test_spline_conv_backend,
        test_disp_sees_upsampled_spikes,
        test_dirichlet_zero_on_rigid,
        test_unique_tracts_from_overlapping_paths,
        test_groupid_tracts_one_polyline_per_group,
        test_groupid_tracts_collapses_duplicate_parent,
        test_groupid_skips_blanked_bifurcation,
        test_groupid_falls_back_without_arrays,
        test_cleandata_discovery_and_complete,
        test_template_scaffold_starts_from_mesh,
        test_dataset_rejects_rawdata_dirs,
        test_tree_token_mask,
        test_junction_coupling_edges,
        test_ostium_couple_linear_memory,
        test_pose_roundtrip,
        test_hybrid_far_points,
        test_cache_hit,
        test_batch_inc,
        test_scaffold_decode_without_vessel,
        test_train_val_split_covers_all,
        test_split_groups_patients_and_reconciles,
        test_filter_split_payload_skips_failed_cache_ids,
        test_skipped_samples_report_is_loud,
        test_warmup_result_skips_unless_strict,
        test_dir_has_vtp_nested_cleandata_layout,
        test_destroy_distributed_without_process_group,
        test_persist_if_remote_skips_same_path,
        test_geco_beta_not_clipped_to_zero_at_epoch_one,
        test_scale_hpc_workers_follows_gpus,
        test_resource_monitor_snapshots_on_this_os,
        test_mirror_reflects_one_graph_consistently,
        test_mirror_twice_is_identity,
        test_add_meter_accepts_fold_and_stretch,
        test_chamfer_weight_cap,
        test_huber_and_radial_loss,
        test_mesh_r_star_stats_stay_in_millimetres,
        test_kl_penalty_is_excess_over_target,
        test_smoothness_edge_weights,
        test_r_star_hit_selection_and_voronoi,
        test_r_star_grid_stats_and_cylinder_raycast,
        test_coarse_attn_per_graph_and_finite,
        test_z_attn_per_tract_and_gate,
        test_true_normals_from_mesh,
        test_gated_hidden_upsample_shapes,
        test_ema_update_and_restore,
        test_full_capacity_model_forward,
        test_gradient_checkpointing_modes,
        test_forward_backward,
        test_coaxial_rstar_healthy_and_bulge_ray,
        test_scaffold_normals_point_away_from_centerline,
        test_groupid_require_raises_without_arrays,
        test_groupid_cell_arrays_blank_and_near_coincident,
        test_groupid_snf_unclipped_fixture,
        test_smoothness_weights_not_all_ones_on_bulge,
        test_token_spacing_independent_of_length,
        test_ring_neighbour_eth_at_cube_edge,
        test_template_levels_keep_loops_density_and_nest,
        test_forward_model_passes_sample_through_ddp_wrapper,
        test_reparameterize_eval_samples_and_logvar_floor,
        test_mixer_before_sampling,
        test_null_code_decodes_to_template,
        test_mesh_terms_have_finite_gradients_at_identity,
        test_cross_attention_reads_world_direction,
        test_rim_normal_term_sees_only_the_projected_slide,
    ]
    failed = 0
    skipped = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except SkipTest as exc:
            skipped += 1
            print(f"SKIP  {fn.__name__}  ({exc})")
        except Exception:
            failed += 1
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
            print()
    print(f"\n{len(tests) - failed - skipped}/{len(tests)} passed, {skipped} skipped, {failed} failed")
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
