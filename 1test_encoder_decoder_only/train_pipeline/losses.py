import math

import torch
import torch.nn.functional as F
from pytorch3d.ops import knn_points

import config as _config
from config import (
    CHAMFER_WEIGHT_CAP,
    CROSS_TRACT_SMOOTH_W,
    LAMBDA_CD_COARSE,
    LAMBDA_CD_MID,
    LAMBDA_RAD_MID,
    PLANE_HUBER_DELTA_MM,
    PLANE_L2_MIX,
    RADIAL_HUBER_DELTA_MM,
    SMOOTH_BETA_R,
    SMOOTH_BETA_THETA,
    SMOOTH_BETA_U,
    SMOOTH_DELTA_R_MM,
    SMOOTH_W_AMBIGUOUS,
    TUBE_RADIUS_MM,
)
from ops import composed_radius


def _cfg(name, default):
    return getattr(_config, name, default)


def _flag_true(flag):
    if flag is None:
        return None
    if torch.is_tensor(flag):
        t = flag.detach()
        if t.device.type != "cpu":
            t = t.cpu()
        return bool((t != 0).reshape(-1).any().item())
    return bool(flag)


class KlLossResult(tuple):
    """Unpackable ``(loss, info)`` that still accepts ``float()`` in older tests."""

    def __new__(cls, loss, info):
        return super().__new__(cls, (loss, info))

    def __float__(self):
        return float(self[0].detach().cpu())

    def item(self):
        return self[0].item()


def _token_valid_weight(mu, latent_valid):
    """Float mask of shape ``mu.shape[:-1]`` (1 = keep)."""
    token_shape = mu.shape[:-1]
    n_tok = int(mu.reshape(-1, mu.size(-1)).size(0))
    if latent_valid is None:
        return mu.new_ones(token_shape)
    if torch.is_tensor(latent_valid):
        v = latent_valid
    else:
        v = mu.new_tensor(1.0 if bool(latent_valid) else 0.0)
    v = v.to(device=mu.device)
    if v.dtype == torch.bool:
        v = v.to(dtype=mu.dtype)
    else:
        v = (v != 0).to(dtype=mu.dtype)
    v = v.reshape(-1)
    if v.numel() == 1:
        return v.reshape(()).expand(token_shape)
    if v.numel() != n_tok:
        raise ValueError(
            "vae_kl_loss: latent_valid has "
            f"{int(v.numel())} values, expected 1 or {n_tok}"
        )
    return v.reshape(token_shape)


def apply_token_kl_floor(kl_per_token, latent_valid=None, lambda_tok=None):
    """Masked mean of ``max(λ_tok, Σ_j KL_j)``.

    Apply this to the *accumulated* token batch (``bs × accum``), not to each
    micro-step, so λ_tok ≈ 0.5 nats is warm-up insurance rather than per-step
    free bits (§5.3.6 item 3).
    """
    kl_per_token = kl_per_token.float()
    if lambda_tok is None:
        lambda_tok = _cfg("TOKEN_KL_FLOOR_NATS", 0.5)
    floored = torch.clamp(kl_per_token, min=float(lambda_tok))
    w = _token_valid_weight(floored.unsqueeze(-1), latent_valid)
    return (floored * w).sum() / w.sum().clamp_min(1e-8)


def geco_beta_max_for_epoch(
    epoch,
    beta_max=None,
    warmup_epochs=None,
    beta_min=None,
    beta_start=None,
):
    """Ramp the GECO ceiling from ``beta_start`` at epoch 1 to ``beta_max``.

    Must never return 0: clipping β to a zero ceiling kills the KL term after
    the first optimiser step (epoch-1 logs then show β=0 and val KL=0).
    """
    if beta_max is None:
        beta_max = _cfg("GECO_BETA_MAX", 10.0)
    if warmup_epochs is None:
        warmup_epochs = _cfg("KL_WARMUP_EPOCHS", 20)
    if beta_min is None:
        beta_min = _cfg("GECO_BETA_MIN", 1e-4)
    if beta_start is None:
        beta_start = _cfg("GECO_BETA_INIT", 1.0)
    beta_max = float(beta_max)
    beta_min = float(beta_min)
    start = max(float(beta_start), beta_min)
    warmup_epochs = int(warmup_epochs)
    if warmup_epochs <= 1:
        return max(beta_max, beta_min)
    t = min(1.0, max(0.0, (int(epoch) - 1) / float(warmup_epochs - 1)))
    return max(beta_min, start + t * (beta_max - start))


def update_geco_beta(
    beta,
    kl_mean_raw,
    rate_target=None,
    eta=None,
    beta_min=None,
    beta_max=None,
    epoch=None,
    warmup_epochs=None,
):
    """One optimiser-step dual update: ``β ← clip(β · exp(η · (KL̄_raw − R*)))``.

    Constraint is on the mean raw KL over valid tokens. Pass ``epoch`` so the
    20-epoch warm-up ramps β_max rather than a fixed λ. The floor ``β_min``
    always holds; the ceiling is never dropped below it.
    """
    if rate_target is None:
        rate_target = _cfg("RATE_TARGET_NATS", 12.0)
    if eta is None:
        eta = _cfg("GECO_ETA", 1e-3)
    if beta_min is None:
        beta_min = _cfg("GECO_BETA_MIN", 1e-4)
    if beta_max is None:
        beta_max = _cfg("GECO_BETA_MAX", 10.0)
    kl = float(kl_mean_raw.detach().cpu()) if torch.is_tensor(kl_mean_raw) else float(kl_mean_raw)
    b = float(beta.detach().cpu()) if torch.is_tensor(beta) else float(beta)
    new_b = b * math.exp(float(eta) * (kl - float(rate_target)))
    lo = float(beta_min)
    hi = float(beta_max)
    if epoch is not None:
        hi = geco_beta_max_for_epoch(
            epoch, beta_max=hi, warmup_epochs=warmup_epochs, beta_min=lo
        )
    hi = max(hi, lo)
    new_b = min(hi, max(lo, new_b))
    if torch.is_tensor(beta):
        return beta.detach().new_tensor(new_b)
    return new_b


def vae_kl_loss(
    mu,
    logvar,
    latent_valid=None,
    beta=None,
    token_floor=None,
    use_lambda_kl=False,
):
    """Per-token KL of a diagonal Gaussian against N(0, I), masked by ``latent_valid``.

    mu, logvar: ``[B, L, D]`` or ``[B, D]``. logvar is log(σ²), not a lognormal.
    The mean is over *valid* tokens so β and R* mean the same on short and long
    trees. Raw per-token KL is unclamped and reported before β.

    Returns ``(loss, info_dict)``. ``loss`` is ``β · mean_valid(KL)`` when ``beta``
    is set; otherwise the unweighted valid-token mean (unit-test fallback).
    ``LAMBDA_KL`` is not the primary weight; pass ``use_lambda_kl=True`` only
    for callers that still multiply by the old fixed λ inside this function.
    Optional ``token_floor`` applies ``max(λ_tok, Σ_j KL_j)`` to this call's
    tokens — prefer :func:`apply_token_kl_floor` on the accumulated batch.
    """
    mu = mu.float()
    logvar = logvar.float()
    kl_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    kl_token = kl_dim.sum(dim=-1)
    w = _token_valid_weight(mu, latent_valid)
    w_sum = w.sum().clamp_min(1e-8)
    kl_mean_raw = (kl_token * w).sum() / w_sum
    if token_floor is not None:
        kl_for_loss = (torch.clamp(kl_token, min=float(token_floor)) * w).sum() / w_sum
    else:
        kl_for_loss = kl_mean_raw
    if beta is None and use_lambda_kl:
        beta = _cfg("LAMBDA_KL", 5e-4)
    if beta is None:
        loss = kl_for_loss
        beta_used = kl_for_loss.new_zeros(())
    else:
        if torch.is_tensor(beta):
            b = beta.to(dtype=kl_for_loss.dtype, device=kl_for_loss.device)
        else:
            b = kl_for_loss.new_tensor(float(beta))
        loss = b * kl_for_loss
        beta_used = b.detach() if torch.is_tensor(b) else kl_for_loss.new_tensor(float(b))
    n_valid = w_sum.detach()
    ln2 = math.log(2.0)
    token_view = (kl_token * w).reshape(mu.size(0), -1)
    bits_per_case = token_view.sum(dim=-1).detach() / ln2
    info = {
        "kl_raw_per_token": kl_token.detach(),
        "kl_mean_raw": kl_mean_raw.detach(),
        "n_valid": n_valid,
        "beta": beta_used,
        "rate_target": float(_cfg("RATE_TARGET_NATS", 12.0)),
        "rate_gap": (kl_mean_raw.detach() - float(_cfg("RATE_TARGET_NATS", 12.0))),
        "bits_per_case": bits_per_case,
    }
    return KlLossResult(loss, info)


def displacement_dirichlet(delta_x, edge_index):
    """Cartesian Dirichlet energy (uniform Δx has zero energy)."""
    if edge_index.numel() == 0:
        return delta_x.new_zeros(())
    src, dst = edge_index[0], edge_index[1]
    return torch.mean((delta_x[src] - delta_x[dst]).pow(2).sum(dim=-1))


def displacement_dirichlet_local(delta_r, delta_s, edge_index, edge_weight=None):
    """Smooth Δr and Δs on the scaffold graph, not Cartesian Δx."""
    if edge_index.numel() == 0:
        return delta_r.new_zeros(())
    src, dst = edge_index[0], edge_index[1]
    err = (delta_r.reshape(-1)[src] - delta_r.reshape(-1)[dst]).pow(2) + (
        delta_s[src] - delta_s[dst]
    ).pow(2).sum(dim=-1)
    if edge_weight is None:
        return err.mean()
    w = edge_weight.to(dtype=err.dtype).reshape(-1)
    return (w * err).sum() / w.sum().clamp_min(1e-8)


def chamfer_distance_weights(dist, radius, cap=CHAMFER_WEIGHT_CAP):
    """Linear Chamfer weights 1 + d/R, clamped so a large sac cannot dominate.

    ``radius`` is the local healthy radius: a Python float or a tensor
    broadcastable to ``dist`` (per-vertex ``r_local``).
    """
    d = dist.float()
    cap = float(cap)
    if torch.is_tensor(radius):
        r = radius.to(dtype=d.dtype, device=d.device)
        if r.numel() == 1:
            w = 1.0 + d / r.reshape(()).clamp_min(1e-8)
        else:
            r = r.reshape(-1)
            d_flat = d.reshape(-1)
            if r.shape[0] != d_flat.shape[0]:
                raise ValueError(
                    "chamfer_distance_weights: radius has "
                    f"{int(r.shape[0])} values, expected 1 or {int(d_flat.shape[0])}"
                )
            w = (1.0 + d_flat / r.clamp_min(1e-8)).reshape(d.shape)
    else:
        w = 1.0 + d / float(radius)
    return w.clamp(max=cap)


def huber(diff, delta=RADIAL_HUBER_DELTA_MM):
    """Elementwise Huber; `delta` is in millimetres."""
    delta = float(delta)
    abs_d = diff.abs()
    quad = 0.5 * diff.pow(2)
    lin = delta * (abs_d - 0.5 * delta)
    return torch.where(abs_d <= delta, quad, lin)


def radial_huber_loss(r_pred, r_star, valid, delta=RADIAL_HUBER_DELTA_MM):
    if r_pred is None or r_star is None or valid is None:
        return r_pred.new_zeros(()) if r_pred is not None else torch.zeros(())
    if r_pred.size(0) == 0:
        return r_pred.new_zeros(())
    w = valid.to(dtype=r_pred.dtype).reshape(-1)
    diff = huber(r_pred.reshape(-1) - r_star.float().reshape(-1), delta=delta)
    return (diff * w).sum() / w.sum().clamp_min(1e-8)


def smoothness_edge_weights(
    src,
    dst,
    r_star,
    valid,
    r_dth,
    r_du,
    r_ring_med,
    beta_th=SMOOTH_BETA_THETA,
    beta_u=SMOOTH_BETA_U,
    beta_r=SMOOTH_BETA_R,
    delta_r=SMOOTH_DELTA_R_MM,
    ambiguous=None,
    w_ambiguous=SMOOTH_W_AMBIGUOUS,
):
    """Per-edge smoothness weight in (0, 1].

    Siphon / Voronoi misses (`valid=False`, not ambiguous) keep w=1.
    Double-hit creases (`ambiguous=True`) drop to `w_ambiguous` so Dirichlet
    does not glue the neck.
    """
    w = r_star.new_ones(src.size(0))
    if valid is None or r_dth is None or r_du is None or r_ring_med is None:
        return w
    both = valid.bool()[src] & valid.bool()[dst]
    dth = torch.maximum(r_dth.float()[src], r_dth.float()[dst])
    du = torch.maximum(r_du.float()[src], r_du.float()[dst])
    rdev = torch.maximum(
        F.relu((r_star.float()[src] - r_ring_med.float()[src]).abs() - float(delta_r)),
        F.relu((r_star.float()[dst] - r_ring_med.float()[dst]).abs() - float(delta_r)),
    )
    w_on = 1.0 / (1.0 + float(beta_th) * dth + float(beta_u) * du + float(beta_r) * rdev)
    w = torch.where(both, w_on, w)
    if ambiguous is not None:
        amb = ambiguous.bool()[src] | ambiguous.bool()[dst]
        w = torch.where(amb, w.new_full((), float(w_ambiguous)), w)
    return w


def _cross_tract_smooth(weight, src, dst, tract_id, scale=CROSS_TRACT_SMOOTH_W):
    """Down-weight parent–daughter coupling edges in Dirichlet / Laplacian."""
    if tract_id is None or src.numel() == 0:
        return weight
    if weight is None:
        weight = src.new_ones(src.size(0), dtype=torch.float32)
    cross = tract_id[src] != tract_id[dst]
    return torch.where(cross, weight.to(dtype=torch.float32) * float(scale), weight.to(dtype=torch.float32))


def _as_face_index(face):
    if face is None or face.numel() == 0:
        return None
    if face.dim() != 2:
        return None
    if face.size(0) == 3:
        return face
    if face.size(1) == 3:
        return face.t().contiguous()
    return None


def _knn_idx_and_sq(src, dst):
    """Nearest-neighbour index and squared L2 from each `src` point to `dst`."""
    src = src.to(dtype=torch.float32).contiguous().unsqueeze(0)
    dst = dst.to(dtype=torch.float32).contiguous().unsqueeze(0)
    knn = knn_points(src, dst, K=1, return_nn=False)
    return knn.idx.reshape(-1), knn.dists.reshape(-1)


def _knn_min_sq(src, dst):
    """Squared L2 distance from each `src` point to its nearest `dst` point."""
    _, dist = _knn_idx_and_sq(src, dst)
    return dist


def _signed_plane(src, dst, n_at_dst):
    """Signed point-to-plane distance using unit normals at nearest `dst` points."""
    idx, _ = _knn_idx_and_sq(src, dst)
    n = n_at_dst[idx].to(dtype=torch.float32)
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    diff = src.to(dtype=torch.float32) - dst.to(dtype=torch.float32)[idx]
    return (diff * n).sum(dim=-1), idx


def _plane_l2_pair(src, dst, n_true, n_is_at_dst=True, mix=PLANE_L2_MIX, delta=PLANE_HUBER_DELTA_MM):
    """Huber point-to-plane mixed with a small L2 Chamfer (stops in-plane sliding)."""
    mix = float(mix)
    idx, l2 = _knn_idx_and_sq(src, dst)
    if n_is_at_dst:
        n = n_true[idx].to(dtype=torch.float32)
        n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        signed = ((src.to(dtype=torch.float32) - dst.to(dtype=torch.float32)[idx]) * n).sum(dim=-1)
    else:
        n = n_true.to(dtype=torch.float32)
        n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        signed = ((src.to(dtype=torch.float32) - dst.to(dtype=torch.float32)[idx]) * n).sum(dim=-1)
    plane = huber(signed, delta=delta)
    return (1.0 - mix) * plane + mix * l2


def _cl_radius(points, cl_xyz):
    if cl_xyz is None or cl_xyz.numel() == 0:
        return points.new_zeros(points.size(0))
    if points.size(0) == 0:
        return points.new_zeros(0)
    return _knn_min_sq(points, cl_xyz[:, :3]).sqrt()


def _chamfer_pair(pred, true, w_pred, w_true, n_true=None):
    if pred.size(0) == 0 or true.size(0) == 0:
        return pred.new_zeros(())
    use_plane = (
        n_true is not None
        and n_true.dim() == 2
        and n_true.size(0) == true.size(0)
        and n_true.size(-1) == 3
    )
    if use_plane:
        min_true = _plane_l2_pair(pred, true, n_true, n_is_at_dst=True)
        min_pred = _plane_l2_pair(true, pred, n_true, n_is_at_dst=False)
    else:
        min_true = _knn_min_sq(pred, true)
        min_pred = _knn_min_sq(true, pred)
    loss_p = (w_pred * min_true).sum() / w_pred.sum().clamp_min(1e-8)
    loss_t = (w_true * min_pred).sum() / w_true.sum().clamp_min(1e-8)
    return 0.5 * (loss_p + loss_t)


def _weighted_chamfer(pred, pred_batch, true, true_batch, w_pred, w_true, num_graphs, n_true=None):
    if int(num_graphs) == 1:
        return _chamfer_pair(pred, true, w_pred, w_true, n_true=n_true)
    loss = pred.new_zeros(())
    n_ok = 0
    for i in range(num_graphs):
        p = pred[pred_batch == i]
        t = true[true_batch == i]
        if p.size(0) == 0 or t.size(0) == 0:
            continue
        n_t = n_true[true_batch == i] if n_true is not None else None
        loss = loss + _chamfer_pair(
            p, t, w_pred[pred_batch == i], w_true[true_batch == i], n_true=n_t
        )
        n_ok += 1
    if n_ok > 0:
        loss = loss / n_ok
    return loss


def _oriented_face_edges(face):
    """Stack the three edges of each triangle. `face` is [3, F]; returns [3F, 2]."""
    v0, v1, v2 = face[0], face[1], face[2]
    return torch.stack(
        (torch.stack((v0, v1), dim=1), torch.stack((v1, v2), dim=1), torch.stack((v2, v0), dim=1)),
        dim=0,
    ).reshape(-1, 2)


def _unique_undirected_edges(face, n_verts):
    """Bidirectional edge index [2, 2E] from triangle faces [3, F]."""
    edges, _ = _oriented_face_edges(face).sort(dim=1)
    key = edges[:, 0] * int(n_verts) + edges[:, 1]
    uniq = torch.unique(key)
    e0 = torch.div(uniq, n_verts, rounding_mode="floor")
    e1 = torch.remainder(uniq, n_verts)
    src = torch.cat([e0, e1], dim=0)
    dst = torch.cat([e1, e0], dim=0)
    return src, dst


def _uniform_laplacian_smoothing(verts, face, batch, num_graphs, edge_weight=None):
    """Uniform mesh Laplacian ||LV||, mean per vertex then mean over graphs.

    Matches pytorch3d `mesh_laplacian_smoothing(..., method='uniform')` without
    constructing a Meshes object (no CPU sync from packed-list conversion).
    L V[i] = mean_{j ~ i}(V[j]) - V[i]. Isolated vertices contribute ||V[i]||.
    Optional `edge_weight` is aligned with the bidirectional unique-edge list
    (src, dst); it both reweights the stencil and scales the per-vertex penalty.
    """
    if face is None or face.numel() == 0 or verts.size(0) == 0:
        return verts.new_zeros(())
    n = verts.size(0)
    src, dst = _unique_undirected_edges(face, n)
    if edge_weight is None:
        w = torch.ones(dst.size(0), device=verts.device, dtype=verts.dtype)
    else:
        w = edge_weight.to(device=verts.device, dtype=verts.dtype).reshape(-1)
        if w.size(0) != dst.size(0):
            w = torch.ones(dst.size(0), device=verts.device, dtype=verts.dtype)
    deg = verts.new_zeros(n).index_add(0, dst, w)
    nb = verts.new_zeros(n, verts.size(-1)).index_add(0, dst, w.unsqueeze(-1) * verts[src])
    lap = nb / deg.clamp_min(1.0).unsqueeze(-1) - verts
    per = lap.norm(dim=-1)
    if edge_weight is not None:
        mean_w = verts.new_zeros(n).index_add(0, dst, w) / deg.clamp_min(1.0)
        per = per * mean_w
    counts = torch.bincount(batch, minlength=num_graphs).to(dtype=verts.dtype).clamp_min(1.0)
    return (per * counts[batch].reciprocal()).sum() / float(num_graphs)


def _adjacent_face_pairs(face, n_verts):
    """Face-index pairs [2, P] that share an undirected edge."""
    F = face.size(1)
    if F == 0:
        return face.new_zeros((2, 0))
    edges, _ = _oriented_face_edges(face).sort(dim=1)
    key = edges[:, 0] * int(n_verts) + edges[:, 1]
    face_ids = torch.arange(F, device=face.device, dtype=torch.long).repeat(3)
    order = torch.argsort(key)
    key = key[order]
    face_ids = face_ids[order]
    same = key[1:] == key[:-1]
    a = face_ids[:-1][same]
    b = face_ids[1:][same]
    ok = a != b
    a, b = a[ok], b[ok]
    if a.numel() == 0:
        return face.new_zeros((2, 0))
    return torch.stack((a, b), dim=0)


def _mesh_normal_consistency(verts, face, batch, num_graphs):
    """Mean `1 - cos(n_i, n_j)` over faces that share an edge, then over graphs.

    GPU-only stand-in for pytorch3d `mesh_normal_consistency`. Adjacent faces on
    this scaffold share winding, so both normals point outward and no extra
    sign flip is applied.
    """
    if face is None or face.numel() == 0 or verts.size(0) == 0:
        return verts.new_zeros(())
    pairs = _adjacent_face_pairs(face, verts.size(0))
    if pairs.size(1) == 0:
        return verts.new_zeros(())
    v0 = verts[face[0]]
    v1 = verts[face[1]]
    v2 = verts[face[2]]
    normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    loss = 1.0 - F.cosine_similarity(normals[pairs[0]], normals[pairs[1]], dim=-1, eps=1e-8)
    pair_batch = batch[face[0, pairs[0]]]
    counts = torch.bincount(pair_batch, minlength=num_graphs).to(dtype=verts.dtype).clamp_min(1.0)
    return (loss * counts[pair_batch].reciprocal()).sum() / float(num_graphs)


def _face_normals(verts, face):
    v0 = verts[face[0]]
    v1 = verts[face[1]]
    v2 = verts[face[2]]
    return torch.cross(v1 - v0, v2 - v0, dim=-1)


def _barycentric_interpolate(attr, face, face_idx, bary):
    """Interpolate a per-vertex field onto surface samples."""
    if attr is None:
        return None
    a0 = attr[face[0, face_idx]]
    a1 = attr[face[1, face_idx]]
    a2 = attr[face[2, face_idx]]
    if attr.dim() == 1:
        return bary[:, 0] * a0 + bary[:, 1] * a1 + bary[:, 2] * a2
    b = bary.unsqueeze(-1)
    return b[:, 0] * a0 + b[:, 1] * a1 + b[:, 2] * a2


def _face_areas(verts, face):
    nrm = _face_normals(verts.float(), face)
    return 0.5 * nrm.norm(dim=-1)


def sample_mesh_surface(
    verts,
    face,
    n_samples,
    batch=None,
    num_graphs=1,
    area_weighted=True,
):
    """Uniform or area-weighted samples on triangles. ``face`` is ``[3, F]``.

    Returns ``(points, sample_batch, barycentric, face_index)``. Face indices
    are detached; barycentric combinations keep a gradient to ``verts``.
    """
    face = _as_face_index(face)
    n_samples = int(n_samples)
    if (
        face is None
        or face.numel() == 0
        or verts.size(0) == 0
        or n_samples <= 0
    ):
        empty = verts.new_zeros((0, verts.size(-1) if verts.dim() == 2 else 3))
        idx = verts.new_zeros((0,), dtype=torch.long)
        return empty, idx, verts.new_zeros((0, 3)), idx
    if batch is None:
        batch = verts.new_zeros(verts.size(0), dtype=torch.long)
        num_graphs = 1
    n_g = max(int(num_graphs), 1)
    pts_out, batch_out, bary_out, fidx_out = [], [], [], []
    face_batch = batch[face[0]]
    for g in range(n_g):
        local_ids = (face_batch == g).nonzero(as_tuple=False).reshape(-1)
        if local_ids.numel() == 0:
            continue
        g_face = face[:, local_ids]
        area = _face_areas(verts, g_face)
        n_f = max(int(area.numel()), 1)
        uniform = torch.full_like(area, 1.0 / float(n_f))
        if area_weighted:
            total = area.sum().clamp_min(0.0)
            probs = torch.where(total > 0, area / total.clamp_min(1e-12), uniform)
        else:
            probs = uniform
        pick = torch.multinomial(probs, n_samples, replacement=True)
        u = torch.rand(n_samples, device=verts.device, dtype=verts.dtype)
        v = torch.rand(n_samples, device=verts.device, dtype=verts.dtype)
        fold = (u + v) > 1
        u = torch.where(fold, 1 - u, u)
        v = torch.where(fold, 1 - v, v)
        bary = torch.stack((1 - u - v, u, v), dim=-1)
        v0 = verts[g_face[0, pick]]
        v1 = verts[g_face[1, pick]]
        v2 = verts[g_face[2, pick]]
        pts = bary[:, 0:1] * v0 + bary[:, 1:2] * v1 + bary[:, 2:3] * v2
        pts_out.append(pts)
        batch_out.append(
            torch.full((n_samples,), g, device=verts.device, dtype=torch.long)
        )
        bary_out.append(bary)
        fidx_out.append(local_ids[pick])
    if not pts_out:
        empty = verts.new_zeros((0, verts.size(-1)))
        idx = verts.new_zeros((0,), dtype=torch.long)
        return empty, idx, verts.new_zeros((0, 3)), idx
    return (
        torch.cat(pts_out, dim=0),
        torch.cat(batch_out, dim=0),
        torch.cat(bary_out, dim=0),
        torch.cat(fidx_out, dim=0),
    )


def _mean_over_faces(per_face, face, batch, num_graphs, like):
    if per_face.numel() == 0:
        return like.new_zeros(())
    if batch is None:
        return per_face.mean()
    face_batch = batch[face[0]]
    counts = torch.bincount(face_batch, minlength=num_graphs).to(dtype=like.dtype).clamp_min(1.0)
    return (per_face * counts[face_batch].reciprocal()).sum() / float(num_graphs)


def fold_penalty(x_pred, x_template, face, batch=None, num_graphs=1):
    """Hinge on ``n_pred · n_template < 0`` per triangle (§7.5, §11 item 1)."""
    face = _as_face_index(face)
    if face is None or face.numel() == 0 or x_pred.size(0) == 0:
        return x_pred.new_zeros(())
    n_pred = F.normalize(_face_normals(x_pred.float(), face), dim=-1, eps=1e-8)
    n_tpl = F.normalize(_face_normals(x_template.float(), face), dim=-1, eps=1e-8)
    pen = F.relu(-(n_pred * n_tpl).sum(dim=-1))
    return _mean_over_faces(pen, face, batch, num_graphs, x_pred)


def _symmetric_eigvals_2x2(mat):
    """Eigenvalues of symmetric ``[..., 2, 2]`` without ``eigvalsh``.

    For ``[[p, q], [q, r]]`` the roots are
    ``((p+r) ± sqrt((p-r)**2 + 4q**2)) / 2``. CUDA ``eigvalsh`` on ~1e5
    tiny matrices tries to allocate a 30+ GiB MAGMA workspace.
    """
    p = mat[..., 0, 0]
    q = mat[..., 0, 1]
    r = mat[..., 1, 1]
    tr = p + r
    disc = ((p - r).square() + 4.0 * q.square()).clamp_min(0.0).sqrt()
    return torch.stack((0.5 * (tr + disc), 0.5 * (tr - disc)), dim=-1)


def triangle_stretch_loss(
    x_pred,
    x_template,
    face,
    batch=None,
    num_graphs=1,
    method="svd",
):
    """Per-triangle stretch against the template (§7.5, §11 item 1).

    ``method='svd'``: singular values of the 2-D deformation gradient
    (eigenvalues of the rest-metric Cauchy–Green tensor). ``method='edge'``:
    edge-length ratios. Both use ``σ + 1/σ − 2``, which is 0 at identity.
    """
    face = _as_face_index(face)
    if face is None or face.numel() == 0 or x_pred.size(0) == 0:
        return x_pred.new_zeros(())
    pred = x_pred.float()
    tpl = x_template.float()
    method = str(method).lower()
    if method in ("edge", "edge_length", "ratio"):
        def _el(verts):
            v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
            return torch.stack(
                ((v1 - v0).norm(dim=-1), (v2 - v1).norm(dim=-1), (v0 - v2).norm(dim=-1)),
                dim=-1,
            )
        ratio = _el(pred) / _el(tpl).clamp_min(1e-8)
        per = (ratio + ratio.clamp_min(1e-8).reciprocal() - 2.0).mean(dim=-1)
        return _mean_over_faces(per, face, batch, num_graphs, x_pred)

    d_tpl = torch.stack(
        (tpl[face[1]] - tpl[face[0]], tpl[face[2]] - tpl[face[0]]), dim=-1
    )
    d_pred = torch.stack(
        (pred[face[1]] - pred[face[0]], pred[face[2]] - pred[face[0]]), dim=-1
    )
    c0 = d_tpl.transpose(-1, -2) @ d_tpl
    c1 = d_pred.transpose(-1, -2) @ d_pred
    eye = torch.eye(2, device=pred.device, dtype=pred.dtype).expand(c0.size(0), 2, 2)
    c0 = c0 + 1e-8 * eye
    a = torch.linalg.solve(c0, c1)
    a = 0.5 * (a + a.transpose(-1, -2))
    ev = _symmetric_eigvals_2x2(a).clamp_min(0.0)
    sigma = ev.sqrt().clamp_min(1e-8)
    per = (sigma + sigma.reciprocal() - 2.0).mean(dim=-1)
    area = _face_areas(tpl, face)
    per = torch.where(area > 1e-12, per, torch.zeros_like(per))
    per = torch.nan_to_num(per, nan=0.0, posinf=0.0, neginf=0.0)
    return _mean_over_faces(per, face, batch, num_graphs, x_pred)


def _resolve_true_cloud(
    x_true,
    batch_x_true,
    x_true_cl_dist,
    x_true_normal,
    has_true_normal,
    gt_points,
    gt_normals,
    gt_batch,
    gt_cl_dist,
    num_graphs,
):
    """Full GT (``gt_points`` / ``gt_normals``) when present, else ``x_true``."""
    if gt_points is None:
        true_pts = x_true
        true_batch = batch_x_true
        true_cl = x_true_cl_dist
        true_n = x_true_normal
        plane_flag = has_true_normal
    else:
        true_pts = gt_points.float()
        if gt_batch is not None:
            true_batch = gt_batch
        elif int(num_graphs) == 1:
            true_batch = torch.zeros(
                true_pts.size(0), dtype=torch.long, device=true_pts.device
            )
        elif true_pts.size(0) == x_true.size(0):
            true_batch = batch_x_true
        else:
            raise ValueError("gt_points requires gt_batch when num_graphs > 1")
        true_cl = gt_cl_dist
        true_n = gt_normals if gt_normals is not None else None
        plane_flag = True if gt_normals is not None else has_true_normal
    n_true = None
    plane = _flag_true(plane_flag)
    if (
        true_n is not None
        and true_n.dim() == 2
        and true_n.size(0) == true_pts.size(0)
        and true_n.size(-1) == 3
        and plane is not False
    ):
        nrm = true_n.float()
        if plane is True:
            n_true = nrm
        elif nrm.device.type == "cpu" and torch.isfinite(nrm).all() and nrm.norm(dim=-1).mean() > 0.5:
            n_true = nrm
    return true_pts, true_batch, true_cl, n_true


def compute_losses(
    x_pred,
    x_true,
    mu,
    logvar,
    x_tube,
    edge_index,
    batch_x_true,
    num_graphs,
    face=None,
    batch_tube=None,
    faces=None,
    delta_x=None,
    delta_r=None,
    delta_s=None,
    x_pred_mid=None,
    batch_mid=None,
    x_pred_coarse=None,
    batch_coarse=None,
    x_true_cl_dist=None,
    cl_dense=None,
    cl_dense_batch=None,
    tube_radius=TUBE_RADIUS_MM,
    lambda_cd_mid=LAMBDA_CD_MID,
    lambda_cd_coarse=LAMBDA_CD_COARSE,
    r_star=None,
    r_star_valid=None,
    r_dth=None,
    r_du=None,
    r_ring_med=None,
    r_star_mid=None,
    r_star_valid_mid=None,
    normal=None,
    normal_mid=None,
    pos_mid=None,
    pos_coarse=None,
    x_true_normal=None,
    has_true_normal=None,
    r_star_ambiguous=None,
    tract_id=None,
    lambda_rad_mid=LAMBDA_RAD_MID,
    chamfer_weight_cap=CHAMFER_WEIGHT_CAP,
    radial_huber_delta=RADIAL_HUBER_DELTA_MM,
    normal_coarse=None,
    r_local=None,
    r_local_mid=None,
    r_local_coarse=None,
    latent_valid=None,
    kl_beta=None,
    kl_token_floor=None,
    gt_points=None,
    gt_normals=None,
    gt_batch=None,
    gt_cl_dist=None,
    x_template=None,
    chamfer_pred_samples=None,
    chamfer_face_sample_mode=None,
    stretch_method="svd",
):
    """Return a dict of unweighted loss terms."""
    x_pred = x_pred.float()
    x_true = x_true.float()
    x_tube = x_tube.float()
    mu = mu.float()
    logvar = logvar.float()
    if delta_x is None:
        delta_x = x_pred - x_tube
    else:
        delta_x = delta_x.float()
    if x_pred_mid is not None:
        x_pred_mid = x_pred_mid.float()
    if x_pred_coarse is not None:
        x_pred_coarse = x_pred_coarse.float()
    if r_local is not None:
        r_local = r_local.float()
    if r_local_mid is not None:
        r_local_mid = r_local_mid.float()
    if r_local_coarse is not None:
        r_local_coarse = r_local_coarse.float()

    if batch_tube is None:
        raise ValueError("compute_losses requires batch_tube (the PyG batch vector for tube nodes)")
    if face is None:
        face = faces
    face = _as_face_index(face)
    x_template = x_tube if x_template is None else x_template.float()

    r_scalar = tube_radius if not torch.is_tensor(tube_radius) else tube_radius
    cap = float(chamfer_weight_cap)
    n_samp = chamfer_pred_samples
    if n_samp is None:
        n_samp = _cfg("CHAMFER_PRED_SAMPLES", _cfg("N_TRUE", 16384))
    n_samp = int(n_samp)
    mode = chamfer_face_sample_mode
    if mode is None:
        mode = _cfg("CHAMFER_FACE_SAMPLE_MODE", "area")
    area_weighted = str(mode).lower() not in ("uniform", "equal", "faces")

    true_pts, true_batch, true_cl, n_true = _resolve_true_cloud(
        x_true,
        batch_x_true,
        x_true_cl_dist,
        x_true_normal,
        has_true_normal,
        gt_points,
        gt_normals,
        gt_batch,
        gt_cl_dist,
        num_graphs,
    )
    cl_xyz = cl_dense
    if cl_xyz is not None and cl_dense_batch is None:
        cl_dense_batch = torch.zeros(cl_xyz.size(0), dtype=torch.long, device=cl_xyz.device)
    if true_cl is None:
        if cl_xyz is not None and true_pts.size(0) > 0:
            if int(num_graphs) == 1:
                true_cl = _cl_radius(true_pts, cl_xyz)
            else:
                true_cl = true_pts.new_zeros(true_pts.size(0))
                for g in range(int(num_graphs)):
                    tm = true_batch == g
                    cm = cl_dense_batch == g if cl_dense_batch is not None else slice(None)
                    if int(tm.sum()) == 0:
                        continue
                    true_cl[tm] = _cl_radius(true_pts[tm], cl_xyz[cm])
        else:
            true_cl = true_pts.new_zeros(true_pts.size(0))
    r_true = r_local.mean() if r_local is not None else r_scalar
    w_true = chamfer_distance_weights(true_cl.float(), r_true, cap=cap)

    def pred_weights(points, point_batch, tube=None, nrm=None, rloc=None):
        radius = rloc if rloc is not None else r_scalar
        if (
            nrm is not None
            and tube is not None
            and nrm.size(0) == points.size(0)
            and tube.size(0) == points.size(0)
        ):
            rad = composed_radius(points, tube, nrm, radius).abs()
            return chamfer_distance_weights(rad, radius, cap=cap)
        w = points.new_ones(points.size(0))
        if cl_xyz is None or points.size(0) == 0:
            return w
        n_g = int(num_graphs)
        if n_g == 1:
            return chamfer_distance_weights(_cl_radius(points, cl_xyz), radius, cap=cap)
        for g in range(n_g):
            pm = point_batch == g
            cm = cl_dense_batch == g if cl_dense_batch is not None else slice(None)
            pts = points[pm]
            if pts.size(0) == 0:
                continue
            rad = _cl_radius(pts, cl_xyz[cm])
            if torch.is_tensor(radius) and radius.numel() == points.size(0):
                r_g = radius.reshape(-1)[pm]
            else:
                r_g = radius
            w[pm] = chamfer_distance_weights(rad, r_g, cap=cap)
        return w

    def face_sampled_pred(verts, v_batch, faces, nrm, tube, rloc):
        if faces is None or n_samp <= 0:
            return verts, v_batch, nrm, tube, rloc
        pts, s_batch, bary, fidx = sample_mesh_surface(
            verts,
            faces,
            n_samp,
            batch=v_batch,
            num_graphs=num_graphs,
            area_weighted=area_weighted,
        )
        if pts.size(0) == 0:
            return verts, v_batch, nrm, tube, rloc
        nrm_s = _barycentric_interpolate(nrm, faces, fidx, bary)
        if nrm_s is not None:
            nrm_s = F.normalize(nrm_s, dim=-1, eps=1e-8)
        tube_s = _barycentric_interpolate(tube, faces, fidx, bary)
        rloc_s = _barycentric_interpolate(rloc, faces, fidx, bary)
        return pts, s_batch, nrm_s, tube_s, rloc_s

    pred_cd, batch_cd, nrm_cd, tube_cd, rloc_cd = face_sampled_pred(
        x_pred, batch_tube, face, normal, x_tube, r_local
    )
    w_pred = pred_weights(pred_cd, batch_cd, tube=tube_cd, nrm=nrm_cd, rloc=rloc_cd)
    loss_recon = _weighted_chamfer(
        pred_cd, batch_cd, true_pts, true_batch, w_pred, w_true, num_graphs, n_true=n_true
    )
    if x_pred_mid is not None and batch_mid is not None:
        w_mid = pred_weights(
            x_pred_mid, batch_mid, tube=pos_mid, nrm=normal_mid, rloc=r_local_mid
        )
        loss_recon = loss_recon + lambda_cd_mid * _weighted_chamfer(
            x_pred_mid, batch_mid, true_pts, true_batch, w_mid, w_true, num_graphs, n_true=n_true
        )
    if x_pred_coarse is not None and batch_coarse is not None:
        w_c = pred_weights(
            x_pred_coarse, batch_coarse, tube=pos_coarse, nrm=normal_coarse, rloc=r_local_coarse
        )
        loss_recon = loss_recon + lambda_cd_coarse * _weighted_chamfer(
            x_pred_coarse, batch_coarse, true_pts, true_batch, w_c, w_true, num_graphs, n_true=n_true
        )

    loss_kl, _kl_info = vae_kl_loss(
        mu, logvar, latent_valid=latent_valid, beta=kl_beta, token_floor=kl_token_floor
    )
    disp_w = None
    if r_star is not None and r_star_valid is not None and r_dth is not None:
        src_d, dst_d = edge_index[0], edge_index[1]
        if src_d.numel():
            disp_w = smoothness_edge_weights(
                src_d, dst_d, r_star, r_star_valid, r_dth, r_du, r_ring_med,
                ambiguous=r_star_ambiguous,
            )
            disp_w = _cross_tract_smooth(disp_w, src_d, dst_d, tract_id)
    elif tract_id is not None and edge_index.numel():
        src_d, dst_d = edge_index[0], edge_index[1]
        disp_w = _cross_tract_smooth(None, src_d, dst_d, tract_id)
    if delta_r is not None and delta_s is not None:
        loss_disp = displacement_dirichlet_local(delta_r, delta_s, edge_index, edge_weight=disp_w)
    else:
        src, dst = edge_index[0], edge_index[1]
        if edge_index.numel() == 0:
            loss_disp = delta_x.new_zeros(())
        else:
            err = (delta_x[src] - delta_x[dst]).pow(2).sum(dim=-1)
            if disp_w is None:
                loss_disp = err.mean()
            else:
                loss_disp = (disp_w * err).sum() / disp_w.sum().clamp_min(1e-8)

    lap_w = None
    if face is not None and r_star is not None and r_star_valid is not None and r_dth is not None:
        src_l, dst_l = _unique_undirected_edges(face, x_pred.size(0))
        if src_l.numel():
            lap_w = smoothness_edge_weights(
                src_l, dst_l, r_star, r_star_valid, r_dth, r_du, r_ring_med,
                ambiguous=r_star_ambiguous,
            )
            lap_w = _cross_tract_smooth(lap_w, src_l, dst_l, tract_id)
    loss_lap = _uniform_laplacian_smoothing(x_pred, face, batch_tube, num_graphs, edge_weight=lap_w)
    loss_norm = _mesh_normal_consistency(x_pred, face, batch_tube, num_graphs)
    loss_fold = fold_penalty(x_pred, x_template, face, batch=batch_tube, num_graphs=num_graphs)
    loss_stretch = triangle_stretch_loss(
        x_pred, x_template, face, batch=batch_tube, num_graphs=num_graphs, method=stretch_method
    )

    loss_rad = x_pred.new_zeros(())
    if r_star is not None and normal is not None:
        r_rad = r_local if r_local is not None else r_scalar
        r_pred = composed_radius(x_pred, x_tube, normal, r_rad)
        loss_rad = radial_huber_loss(r_pred, r_star, r_star_valid, delta=radial_huber_delta)
        if (
            x_pred_mid is not None
            and r_star_mid is not None
            and normal_mid is not None
            and pos_mid is not None
        ):
            r_rad_m = r_local_mid if r_local_mid is not None else r_scalar
            r_pred_m = composed_radius(x_pred_mid, pos_mid, normal_mid, r_rad_m)
            loss_rad = loss_rad + float(lambda_rad_mid) * radial_huber_loss(
                r_pred_m, r_star_mid, r_star_valid_mid, delta=radial_huber_delta
            )

    return {
        "recon": loss_recon,
        "kl": loss_kl,
        "disp": loss_disp,
        "lap": loss_lap,
        "norm": loss_norm,
        "rad": loss_rad,
        "fold": loss_fold,
        "stretch": loss_stretch,
    }
