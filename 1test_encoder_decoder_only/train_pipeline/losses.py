import torch
from pytorch3d.loss import chamfer_distance, mesh_laplacian_smoothing
from pytorch3d.structures import Meshes


def vae_kl_loss(mu, logvar):
    """
    KL divergence of a diagonal Gaussian posterior against N(0, I).
    logvar is clamped to keep exp() finite.
    """
    logvar = torch.clamp(logvar, -30.0, 20.0)
    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
    return kld_loss


def edge_length_penalty(x_pred, x_tube, edge_index):
    """
    Penalizes edges that stretch too much compared to the base tube.
    """
    src_pred, dst_pred = x_pred[edge_index[0]], x_pred[edge_index[1]]
    src_tube, dst_tube = x_tube[edge_index[0]], x_tube[edge_index[1]]

    len_pred = torch.norm(dst_pred - src_pred, dim=1)
    len_tube = torch.norm(dst_tube - src_tube, dim=1)

    loss_edge = torch.mean((len_pred - len_tube) ** 2)
    return loss_edge


def _as_face_index(face):
    """Return face indices as [3, F], or None."""
    if face is None or face.numel() == 0:
        return None
    if face.dim() != 2:
        return None
    if face.size(0) == 3:
        return face
    if face.size(1) == 3:
        return face.t().contiguous()
    return None


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
):
    """
    Combined loss for variable-size tube graphs.

    x_pred / x_tube: [total_nodes, 3]
    x_true: [M_batched, 3]
    edge_index: [2, E] (PyG-incremented)
    batch_x_true: [M_batched]
    batch_tube: [total_nodes] PyG batch vector for tube nodes (required for batch_size > 1)
    face: [3, F] PyG-incremented face index, or [F, 3]
    faces: legacy alias for face
    """
    if batch_tube is None:
        raise ValueError("compute_losses requires batch_tube (the PyG batch vector for tube nodes)")

    if face is None:
        face = faces
    face = _as_face_index(face)

    loss_recon = x_pred.new_zeros(())
    n_recon = 0
    for i in range(num_graphs):
        x_pred_i = x_pred[batch_tube == i].unsqueeze(0)
        x_true_i = x_true[batch_x_true == i].unsqueeze(0)
        if x_pred_i.size(1) == 0 or x_true_i.size(1) == 0:
            continue
        loss_chamfer, _ = chamfer_distance(x_pred_i, x_true_i)
        loss_recon = loss_recon + loss_chamfer
        n_recon += 1
    if n_recon > 0:
        loss_recon = loss_recon / n_recon

    loss_kl = vae_kl_loss(mu, logvar)
    loss_edge = edge_length_penalty(x_pred, x_tube, edge_index)

    device_type = "cuda" if x_pred.is_cuda else "cpu"
    loss_laplacian = x_pred.new_zeros(())
    if face is not None:
        counts = torch.bincount(batch_tube, minlength=num_graphs)
        ptr = torch.zeros(num_graphs + 1, device=x_pred.device, dtype=torch.long)
        ptr[1:] = torch.cumsum(counts, dim=0)
        face_owner = batch_tube[face[0]]
        x_f32 = x_pred.to(torch.float32)

        verts_list = []
        faces_list = []
        for i in range(num_graphs):
            f_i = face[:, face_owner == i] - ptr[i]
            if f_i.numel() == 0:
                continue
            verts_list.append(x_f32[ptr[i]:ptr[i + 1]])
            faces_list.append(f_i.t().contiguous())

        if verts_list:
            with torch.autocast(device_type=device_type, enabled=False):
                meshes = Meshes(verts=verts_list, faces=faces_list)
                loss_laplacian = mesh_laplacian_smoothing(meshes, method="uniform")

    loss_geom = loss_edge + loss_laplacian
    return loss_recon, loss_kl, loss_geom
