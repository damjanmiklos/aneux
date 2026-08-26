import torch
import torch.nn.functional as F
from pytorch3d.ops import knn_points

from config import (
    CHAMFER_WEIGHT_CAP,
    CROSS_TRACT_SMOOTH_W,
    LAMBDA_CD_COARSE,
    LAMBDA_CD_MID,
    LAMBDA_RAD_MID,
    LOGVAR_CLAMP,
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


def _flag_true(flag):
    if flag is None:
        return None
    if torch.is_tensor(flag):
        t = flag.detach()
        if t.device.type != "cpu":
            t = t.cpu()
        return bool((t != 0).reshape(-1).any().item())
    return bool(flag)


def vae_kl_loss(mu, logvar):
    """KL of a diagonal Gaussian posterior against N(0, I).

    mu, logvar: [B, L, D] or [B, D]. Averaged over batch and latent tokens.
    logvar is log(σ²) of a Gaussian, not a lognormal.
    """
    logvar = torch.clamp(logvar, LOGVAR_CLAMP[0], LOGVAR_CLAMP[1])
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    return kl.sum(dim=-1).mean()


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
    """Linear Chamfer weights 1 + d/R, clamped so a large sac cannot dominate."""
    r = float(radius)
    return (1.0 + dist.float() / r).clamp(max=float(cap))


def huber(diff, delta=RADIAL_HUBER_DELTA_MM):
    """Elementwise Huber; `delta` is in millimetres."""
    delta = float(delta)
    abs_d = diff.abs()
    quad = 0.5 * diff.pow(2)
    lin = delta * (abs_d - 0.5 * delta)
    return torch.where(abs_d <= delta, quad, lin)


def composed_radius(x, x_tube, normal, tube_radius):
    """Radial distance from the centerline sample: R + n · (x - x_tube)."""
    return float(tube_radius) + (normal.float() * (x.float() - x_tube.float())).sum(dim=-1)


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

    if batch_tube is None:
        raise ValueError("compute_losses requires batch_tube (the PyG batch vector for tube nodes)")
    if face is None:
        face = faces

    r = float(tube_radius)
    cap = float(chamfer_weight_cap)
    if x_true_cl_dist is None:
        x_true_cl_dist = x_true.new_zeros(x_true.size(0))
    w_true = chamfer_distance_weights(x_true_cl_dist.float(), r, cap=cap)

    cl_xyz = cl_dense
    if cl_xyz is not None and cl_dense_batch is None:
        cl_dense_batch = torch.zeros(cl_xyz.size(0), dtype=torch.long, device=cl_xyz.device)

    def pred_weights(points, point_batch, tube=None, nrm=None):
        if (
            nrm is not None
            and tube is not None
            and nrm.size(0) == points.size(0)
            and tube.size(0) == points.size(0)
        ):
            rad = composed_radius(points, tube, nrm, r).abs()
            return chamfer_distance_weights(rad, r, cap=cap)
        w = points.new_ones(points.size(0))
        if cl_xyz is None or points.size(0) == 0:
            return w
        n_g = int(num_graphs)
        if n_g == 1:
            return chamfer_distance_weights(_cl_radius(points, cl_xyz), r, cap=cap)
        for g in range(n_g):
            pm = point_batch == g
            cm = cl_dense_batch == g if cl_dense_batch is not None else slice(None)
            pts = points[pm]
            if pts.size(0) == 0:
                continue
            rad = _cl_radius(pts, cl_xyz[cm])
            w[pm] = chamfer_distance_weights(rad, r, cap=cap)
        return w

    w_pred = pred_weights(x_pred, batch_tube, tube=x_tube, nrm=normal)
    n_true = None
    plane = _flag_true(has_true_normal)
    if (
        x_true_normal is not None
        and x_true_normal.dim() == 2
        and x_true_normal.size(0) == x_true.size(0)
        and x_true_normal.size(-1) == 3
        and plane is not False
    ):
        nrm = x_true_normal.float()
        if plane is True:
            n_true = nrm
        elif nrm.device.type == "cpu" and torch.isfinite(nrm).all() and nrm.norm(dim=-1).mean() > 0.5:
            n_true = nrm
    loss_recon = _weighted_chamfer(
        x_pred, batch_tube, x_true, batch_x_true, w_pred, w_true, num_graphs, n_true=n_true
    )
    if x_pred_mid is not None and batch_mid is not None:
        w_mid = pred_weights(x_pred_mid, batch_mid, tube=pos_mid, nrm=normal_mid)
        loss_recon = loss_recon + lambda_cd_mid * _weighted_chamfer(
            x_pred_mid, batch_mid, x_true, batch_x_true, w_mid, w_true, num_graphs, n_true=n_true
        )
    if x_pred_coarse is not None and batch_coarse is not None:
        w_c = pred_weights(x_pred_coarse, batch_coarse, tube=pos_coarse, nrm=normal_coarse)
        loss_recon = loss_recon + lambda_cd_coarse * _weighted_chamfer(
            x_pred_coarse, batch_coarse, x_true, batch_x_true, w_c, w_true, num_graphs, n_true=n_true
        )

    loss_kl = vae_kl_loss(mu, logvar)
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

    face = _as_face_index(face)
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

    loss_rad = x_pred.new_zeros(())
    if r_star is not None and normal is not None:
        r_pred = composed_radius(x_pred, x_tube, normal, r)
        loss_rad = radial_huber_loss(r_pred, r_star, r_star_valid, delta=radial_huber_delta)
        if (
            x_pred_mid is not None
            and r_star_mid is not None
            and normal_mid is not None
            and pos_mid is not None
        ):
            r_pred_m = composed_radius(x_pred_mid, pos_mid, normal_mid, r)
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
    }
