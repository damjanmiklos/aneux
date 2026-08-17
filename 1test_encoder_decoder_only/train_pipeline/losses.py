import torch
from pytorch3d.loss import chamfer_distance, mesh_laplacian_smoothing, mesh_normal_consistency
from pytorch3d.structures import Meshes

from config import LAMBDA_CD_COARSE, LAMBDA_CD_MID, LOGVAR_CLAMP


def vae_kl_loss(mu, logvar):
    """Sequence-wise KL of a diagonal Gaussian posterior against N(0, I).

    mu, logvar: [B, L, D] or [B, D]. Averaged over batch and latent tokens.
    """
    logvar = torch.clamp(logvar, LOGVAR_CLAMP[0], LOGVAR_CLAMP[1])
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    return kl.sum(dim=-1).mean()


def displacement_dirichlet(delta_x, edge_index):
    """Penalize displacement spikes between adjacent scaffold vertices."""
    if edge_index.numel() == 0:
        return delta_x.new_zeros(())
    src, dst = edge_index[0], edge_index[1]
    return torch.mean((delta_x[src] - delta_x[dst]).pow(2).sum(dim=-1))


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


def _batched_chamfer(pred, pred_batch, true, true_batch, num_graphs):
    loss = pred.new_zeros(())
    n_ok = 0
    for i in range(num_graphs):
        p = pred[pred_batch == i].unsqueeze(0)
        t = true[true_batch == i].unsqueeze(0)
        if p.size(1) == 0 or t.size(1) == 0:
            continue
        cd, _ = chamfer_distance(p, t)
        loss = loss + cd
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
    x_pred_mid=None,
    batch_mid=None,
    x_pred_coarse=None,
    batch_coarse=None,
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

    loss_recon = _batched_chamfer(x_pred, batch_tube, x_true, batch_x_true, num_graphs)
    if x_pred_mid is not None and batch_mid is not None:
        loss_recon = loss_recon + lambda_cd_mid * _batched_chamfer(
            x_pred_mid, batch_mid, x_true, batch_x_true, num_graphs
        )
    if x_pred_coarse is not None and batch_coarse is not None:
        loss_recon = loss_recon + lambda_cd_coarse * _batched_chamfer(
            x_pred_coarse, batch_coarse, x_true, batch_x_true, num_graphs
        )

    loss_kl = vae_kl_loss(mu, logvar)
    loss_disp = displacement_dirichlet(delta_x, edge_index)

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
