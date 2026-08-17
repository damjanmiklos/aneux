"""Smoke tests for the hierarchical PointNeXt–SplineConv VAE."""

from __future__ import annotations

import math
import os
import sys
import traceback

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import FOLLOW_BATCH, GAMMA_THETA_DIM, GAMMA_U_DIM, K_THETA, K_U
from dataset import AneurysmDataset, allocate_ring_counts
from geometry import (
    bilinear_cylindrical_upsample,
    fps_metric,
    harmonic_encoding_theta,
    harmonic_encoding_u,
    radial_bias_for_zero_init,
    spline_pseudo_coords,
    upsample_branch_concat,
)
from losses import displacement_dirichlet, vae_kl_loss
from model import DecoupledDisplacementHead, GraphVAE
from ops import bspline_basis_1d, farthest_point_sample_torch, fps_indices
from train import kl_anneal_weight


class _TubeFactory(AneurysmDataset):
    def __init__(self, radius=2.0):
        self.tube_radius = radius


def _straight_branch(n=24, length=20.0):
    z = np.linspace(0.0, length, n)
    return np.stack([np.zeros(n), np.zeros(n), z], axis=1).astype(np.float32)


def make_synthetic_data(
    n_true=64,
    hierarchy=((8, 4), (16, 8), (32, 16)),
    latent_len=8,
    radius=2.0,
):
    factory = _TubeFactory(radius=radius)
    branch = _straight_branch()
    arc = [float(np.sum(np.linalg.norm(np.diff(branch, axis=0), axis=1)))]
    names = ("coarse", "mid", "fine")
    levels = {}
    for name, (nl, nr) in zip(names, hierarchy):
        levels[name] = factory._generate_level([branch], nl, nr, arc)

    fine, mid, coarse = levels["fine"], levels["mid"], levels["coarse"]
    rng = np.random.RandomState(0)
    x_true = torch.tensor(fine["pos"].numpy() + 0.1 * rng.randn(*fine["pos"].shape), dtype=torch.float32)
    if x_true.size(0) >= n_true:
        x_true = torch.tensor(fps_metric(x_true.numpy(), n_true), dtype=torch.float32)
    else:
        reps = int(math.ceil(n_true / x_true.size(0)))
        x_true = x_true.repeat(reps, 1)[:n_true]

    cl_xyz = fine["cl_dense"].numpy()
    cl_u = fine["cl_dense_u"].numpy()
    cl_pos = factory._resample_centerline(cl_xyz, cl_u, latent_len)
    cl_dense = torch.cat([fine["cl_dense"], fine["cl_dense_u"].unsqueeze(-1)], dim=-1)

    return Data(
        x=fine["pos"],
        edge_index=fine["edge_index"],
        face=fine["face"],
        u=fine["u"],
        u_local=fine["u_local"],
        theta=fine["theta"],
        normal=fine["normal"],
        tangent=fine["tangent"],
        binormal=fine["binormal"],
        pos_mid=mid["pos"],
        edge_index_mid=mid["edge_index"],
        face_mid=mid["face"],
        u_mid=mid["u"],
        theta_mid=mid["theta"],
        normal_mid=mid["normal"],
        tangent_mid=mid["tangent"],
        binormal_mid=mid["binormal"],
        pos_coarse=coarse["pos"],
        edge_index_coarse=coarse["edge_index"],
        face_coarse=coarse["face"],
        u_coarse=coarse["u"],
        theta_coarse=coarse["theta"],
        normal_coarse=coarse["normal"],
        tangent_coarse=coarse["tangent"],
        binormal_coarse=coarse["binormal"],
        x_true=x_true,
        cl_pos=cl_pos,
        cl_dense=cl_dense,
        branch_nl_fine=fine["branch_nl"],
        branch_nl_mid=mid["branch_nl"],
        branch_nl_coarse=coarse["branch_nl"],
        n_radial_fine=fine["n_radial"],
        n_radial_mid=mid["n_radial"],
        n_radial_coarse=coarse["n_radial"],
        origin_shift=torch.zeros(3),
    )


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_fourier_shapes():
    u = torch.linspace(0, 1, 11)
    th = torch.linspace(-math.pi, math.pi, 13)[:-1]
    gu = harmonic_encoding_u(u)
    gt = harmonic_encoding_theta(th)
    _assert(gu.shape == (11, GAMMA_U_DIM), f"γ(u) shape {gu.shape}")
    _assert(gt.shape == (12, GAMMA_THETA_DIM), f"γ(θ) shape {gt.shape}")
    _assert(torch.isfinite(gu).all() and torch.isfinite(gt).all(), "non-finite Fourier")
    # Periodicity in θ for the lowest band.
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
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    ei = torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]])
    e = spline_pseudo_coords(pos, ei, r_edge_max=2.0)
    _assert(e.min() >= 0.0 and e.max() <= 1.0, f"pseudo coords out of [0,1]: {e.min()}, {e.max()}")


def test_bilinear_identity_and_wrap():
    nl, nr, c = 5, 8, 3
    field = torch.arange(nl * nr * c, dtype=torch.float32).reshape(nl * nr, c)
    out = bilinear_cylindrical_upsample(field, nl, nr, nl, nr)
    _assert(torch.allclose(out, field, atol=1e-5), "identity upsample failed")

    # A spike on the last radial index should wrap to the first on a finer grid.
    src = torch.zeros(2, 4, 1)
    src[0, 0, 0] = 1.0
    src[0, 3, 0] = 1.0
    up = bilinear_cylindrical_upsample(src.reshape(-1, 1), 2, 4, 2, 8)
    up = up.reshape(2, 8)
    _assert(up[0, 0] > 0.4 and up[0, 7] > 0.4, f"θ wrap failed: {up[0]}")

    concat = upsample_branch_concat(field, torch.tensor([nl]), nr, torch.tensor([nl]), nr)
    _assert(torch.allclose(concat, field, atol=1e-5), "branch concat identity failed")


def test_bishop_frames_orthonormal():
    factory = _TubeFactory(radius=2.0)
    tube = factory._generate_branch_tube(_straight_branch(), n_length_branch=12, n_radial=6)
    n_v, t_v, b_v = tube["n_v"], tube["t_v"], tube["b_v"]
    _assert(np.allclose(np.linalg.norm(n_v, axis=1), 1.0, atol=1e-5), "n not unit")
    _assert(np.allclose(np.linalg.norm(t_v, axis=1), 1.0, atol=1e-5), "t not unit")
    _assert(np.allclose(np.linalg.norm(b_v, axis=1), 1.0, atol=1e-5), "b not unit")
    _assert(np.allclose((n_v * t_v).sum(1), 0.0, atol=1e-4), "n·t")
    _assert(np.allclose((n_v * b_v).sum(1), 0.0, atol=1e-4), "n·b")
    _assert(np.allclose((t_v * b_v).sum(1), 0.0, atol=1e-4), "t·b")
    # Right-handed: n × t should align with... t × n = b? stored t, n, b:
    # n_v × t_v should be related to b. Local basis (t, n, b) at centerline;
    # vertex: t_v, n_v, b_v should be right-handed.
    cross = np.cross(n_v, t_v)
    # n × t = n × tangent; for θ=0, n_v=n, t_v=t, n×t = -t×n = -b. Sign depends.
    _assert(np.allclose(np.abs((cross * b_v).sum(1)), 1.0, atol=1e-4), "frame not orthonormal triad")
    _assert(tube["theta"].min() >= -np.pi - 1e-6 and tube["theta"].max() < np.pi + 1e-6, "θ range")


def test_fps_count():
    pts = np.random.RandomState(0).randn(200, 3).astype(np.float32)
    out = fps_metric(pts, 32)
    _assert(out.shape == (32, 3), f"FPS shape {out.shape}")
    short = fps_metric(pts[:10], 32)
    _assert(short.shape == (32, 3), "FPS pad failed")


def test_fps_cuda_path():
    from ops import _pytorch3d_fps_ok

    pts = torch.randn(128, 3)
    idx_cpu = fps_indices(pts, 16)
    _assert(idx_cpu.numel() == 16, idx_cpu.shape)
    _assert(idx_cpu.min() >= 0 and idx_cpu.max() < 128, "CPU FPS out of range")
    idx_loop = farthest_point_sample_torch(pts, 16)
    _assert(idx_loop.numel() == 16, idx_loop.shape)
    if not torch.cuda.is_available():
        return
    pts_g = pts.cuda()
    _assert(_pytorch3d_fps_ok(pts_g.device), "pytorch3d CUDA FPS unavailable")
    idx_g = fps_indices(pts_g, 16)
    _assert(idx_g.device.type == "cuda", f"expected CUDA indices, got {idx_g.device}")
    _assert(idx_g.numel() == 16, idx_g.shape)
    _assert(int(idx_g.max()) < 128, "CUDA FPS out of range")


def test_ball_query_index_order():
    from ops import ball_query_packed, _ball_query_torch

    support = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    query = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
    sb = torch.zeros(4, dtype=torch.long)
    qb = torch.zeros(2, dtype=torch.long)
    ref = _ball_query_torch(support, query, 1.5, sb, qb, 32)
    got = ball_query_packed(support, query, 1.5, sb, qb, 32)
    _assert(int(got[0].max()) < 4, f"support idx {got[0]}")
    _assert(int(got[1].max()) < 2, f"query idx {got[1]}")
    pairs = lambda ei: set(zip(ei[0].tolist(), ei[1].tolist()))
    _assert(pairs(got) == pairs(ref), f"{got} vs {ref}")
    if torch.cuda.is_available():
        got_g = ball_query_packed(support.cuda(), query.cuda(), 1.5, sb.cuda(), qb.cuda(), 32)
        _assert(got_g.device.type == "cuda", got_g.device)
        _assert(int(got_g[0].max()) < 4 and int(got_g[1].max()) < 2, got_g)


def test_allocate_rings():
    alloc = allocate_ring_counts(40, [10.0, 10.0])
    _assert(len(alloc) == 2 and min(alloc) >= 2, alloc)
    alloc2 = allocate_ring_counts(10, [1.0, 1.0, 1.0, 1.0, 1.0])
    _assert(len(alloc2) == 5, alloc2)


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


def test_bspline_partition_of_unity():
    u = torch.linspace(0, 1, 64)
    basis = bspline_basis_1d(u, n_ctrl=5, degree=2)
    _assert(basis.shape == (64, 5), basis.shape)
    _assert(torch.allclose(basis.sum(dim=1), torch.ones(64), atol=1e-4), "partition of unity")
    _assert((basis >= -1e-5).all(), "basis should be non-negative")


def test_dirichlet_zero_on_rigid():
    delta = torch.ones(5, 3)
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]])
    loss = displacement_dirichlet(delta, ei)
    _assert(float(loss) < 1e-8, "uniform displacement should have zero Dirichlet energy")


def test_forward_backward():
    data = make_synthetic_data()
    loader = DataLoader([data], batch_size=1, follow_batch=FOLLOW_BATCH)
    batch = next(iter(loader))
    model = GraphVAE(
        latent_dim=8,
        latent_len=8,
        hidden_dim=16,
        tube_radius=2.0,
        sa_stages=((32, 4.0, 8, 32, 1), (8, 8.0, 8, 64, 1)),
    )
    model.train()
    out = model(batch)
    _assert(out.x_pred.shape == batch.x.shape, f"pred {out.x_pred.shape} vs {batch.x.shape}")
    _assert(out.mu.shape == (1, 8, 8), f"mu {out.mu.shape}")
    _assert(out.logvar.shape == (1, 8, 8), f"logvar {out.logvar.shape}")
    _assert(out.z.shape == (1, 8, 8), f"z {out.z.shape}")
    _assert(out.x_pred_coarse.shape == batch.pos_coarse.shape, "coarse pred")
    _assert(out.x_pred_mid.shape == batch.pos_mid.shape, "mid pred")
    _assert(torch.isfinite(out.x_pred).all(), "non-finite prediction")

    from losses import compute_losses

    terms = compute_losses(
        out.x_pred, batch.x_true, out.mu, out.logvar, batch.x, batch.edge_index,
        batch.x_true_batch, batch.num_graphs, face=batch.face, batch_tube=batch.batch,
        delta_x=out.delta_x, x_pred_mid=out.x_pred_mid, batch_mid=batch.pos_mid_batch,
        x_pred_coarse=out.x_pred_coarse, batch_coarse=batch.pos_coarse_batch,
    )
    loss = terms["recon"] + 0.001 * terms["kl"] + 0.1 * terms["disp"] + 0.05 * terms["lap"] + 0.02 * terms["norm"]
    loss.backward()
    grads = [p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None]
    _assert(len(grads) > 0 and sum(grads) > 0, "no gradients")

    model.eval()
    z = out.mu.detach()
    x_dec = model.decode(z, batch)
    _assert(x_dec.shape == batch.x.shape, "decode contract shape")


def main():
    tests = [
        test_fourier_shapes,
        test_kl_and_anneal,
        test_pseudo_coords_range,
        test_bilinear_identity_and_wrap,
        test_bishop_frames_orthonormal,
        test_fps_count,
        test_fps_cuda_path,
        test_ball_query_index_order,
        test_allocate_rings,
        test_decoupled_head_no_inversion,
        test_bspline_partition_of_unity,
        test_dirichlet_zero_on_rigid,
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
