import torch
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency
from pytorch3d.ops import knn_points
from pytorch3d.structures import Meshes

from config import LAMBDA_CD_COARSE, LAMBDA_CD_MID, LOGVAR_CLAMP, TUBE_RADIUS_MM


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


def displacement_dirichlet_local(delta_r, delta_s, edge_index):
    """Smooth Δr and Δs on the scaffold graph, not Cartesian Δx."""
    if edge_index.numel() == 0:
        return delta_r.new_zeros(())
    src, dst = edge_index[0], edge_index[1]
    dr = (delta_r.reshape(-1)[src] - delta_r.reshape(-1)[dst]).pow(2).mean()
    ds = (delta_s[src] - delta_s[dst]).pow(2).sum(dim=-1).mean()
    return dr + ds


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


def _knn_min_sq(src, dst):
    """Squared L2 distance from each `src` point to its nearest `dst` point."""
    src = src.to(dtype=torch.float32).contiguous().unsqueeze(0)
    dst = dst.to(dtype=torch.float32).contiguous().unsqueeze(0)
    return knn_points(src, dst, K=1, return_nn=False).dists.reshape(-1)


def _cl_radius(points, cl_xyz):
    if cl_xyz is None or cl_xyz.numel() == 0:
        return points.new_zeros(points.size(0))
    if points.size(0) == 0:
        return points.new_zeros(0)
    return _knn_min_sq(points, cl_xyz[:, :3]).sqrt()


def _weighted_chamfer(pred, pred_batch, true, true_batch, w_pred, w_true, num_graphs):
    loss = pred.new_zeros(())
    n_ok = 0
    for i in range(num_graphs):
        p = pred[pred_batch == i]
        t = true[true_batch == i]
        if p.size(0) == 0 or t.size(0) == 0:
            continue
        wp = w_pred[pred_batch == i]
        wt = w_true[true_batch == i]
        min_true = _knn_min_sq(p, t)
        min_pred = _knn_min_sq(t, p)
        loss_p = (wp * min_true).sum() / wp.sum().clamp_min(1e-8)
        loss_t = (wt * min_pred).sum() / wt.sum().clamp_min(1e-8)
        loss = loss + 0.5 * (loss_p + loss_t)
        n_ok += 1
    if n_ok > 0:
        loss = loss / n_ok
    return loss


def _meshes_from_batch(verts, faces, batch_tube, num_graphs):
    face = _as_face_index(faces)
    if face is None:
        return None
    counts = torch.bincount(batch_tube, minlength=num_graphs)
    ptr = torch.zeros(num_graphs + 1, device=verts.device, dtype=torch.long)
    ptr[1:] = torch.cumsum(counts, dim=0)
    face_owner = batch_tube[face[0]]
    x_f32 = verts.to(torch.float32)
    verts_list, faces_list = [], []
    for i in range(num_graphs):
        f_i = face[:, face_owner == i] - ptr[i]
        if f_i.numel() == 0:
            continue
        verts_list.append(x_f32[ptr[i]:ptr[i + 1]])
        faces_list.append(f_i.t().contiguous())
    if not verts_list:
        return None
    return Meshes(verts=verts_list, faces=faces_list)


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
    if x_true_cl_dist is None:
        x_true_cl_dist = x_true.new_zeros(x_true.size(0))
    w_true = 1.0 + (x_true_cl_dist.float() / r).pow(2)

    cl_xyz = cl_dense
    if cl_xyz is not None and cl_dense_batch is None:
        cl_dense_batch = torch.zeros(cl_xyz.size(0), dtype=torch.long, device=cl_xyz.device)

    def pred_weights(points, point_batch):
        w = points.new_ones(points.size(0))
        if cl_xyz is None:
            return w
        n_graphs = int(point_batch.max().item()) + 1 if point_batch.numel() else 1
        for g in range(n_graphs):
            pm = point_batch == g
            cm = cl_dense_batch == g if cl_dense_batch is not None else slice(None)
            if not torch.any(pm):
                continue
            rad = _cl_radius(points[pm], cl_xyz[cm])
            w[pm] = 1.0 + (rad / r).pow(2)
        return w

    w_pred = pred_weights(x_pred, batch_tube)
    loss_recon = _weighted_chamfer(
        x_pred, batch_tube, x_true, batch_x_true, w_pred, w_true, num_graphs
    )
    if x_pred_mid is not None and batch_mid is not None:
        w_mid = pred_weights(x_pred_mid, batch_mid)
        loss_recon = loss_recon + lambda_cd_mid * _weighted_chamfer(
            x_pred_mid, batch_mid, x_true, batch_x_true, w_mid, w_true, num_graphs
        )
    if x_pred_coarse is not None and batch_coarse is not None:
        w_c = pred_weights(x_pred_coarse, batch_coarse)
        loss_recon = loss_recon + lambda_cd_coarse * _weighted_chamfer(
            x_pred_coarse, batch_coarse, x_true, batch_x_true, w_c, w_true, num_graphs
        )

    loss_kl = vae_kl_loss(mu, logvar)
    if delta_r is not None and delta_s is not None:
        loss_disp = displacement_dirichlet_local(delta_r, delta_s, edge_index)
    else:
        src, dst = edge_index[0], edge_index[1]
        loss_disp = (
            torch.mean((delta_x[src] - delta_x[dst]).pow(2).sum(dim=-1))
            if edge_index.numel()
            else delta_x.new_zeros(())
        )

    loss_lap = x_pred.new_zeros(())
    loss_norm = x_pred.new_zeros(())
    meshes = _meshes_from_batch(x_pred, face, batch_tube, num_graphs)
    if meshes is not None:
        loss_lap = mesh_laplacian_smoothing(meshes, method="uniform")
        loss_norm = mesh_normal_consistency(meshes)

    return {
        "recon": loss_recon,
        "kl": loss_kl,
        "disp": loss_disp,
        "lap": loss_lap,
        "norm": loss_norm,
    }
