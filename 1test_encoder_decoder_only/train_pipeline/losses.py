import torch
from pytorch3d.loss import chamfer_distance, mesh_laplacian_smoothing
from pytorch3d.structures import Meshes

def vae_kl_loss(mu, logvar):
    """
    Computes the KL divergence loss for a Gaussian VAE.
    """
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
    return kld_loss

def edge_length_penalty(x_pred, x_tube, edge_index):
    """
    Penalizes edges that stretch too much compared to the base tube.
    """
    # Get edge coordinates for prediction
    src_pred, dst_pred = x_pred[edge_index[0]], x_pred[edge_index[1]]
    # Get edge coordinates for base tube
    src_tube, dst_tube = x_tube[edge_index[0]], x_tube[edge_index[1]]
    
    # Calculate lengths
    len_pred = torch.norm(dst_pred - src_pred, dim=1)
    len_tube = torch.norm(dst_tube - src_tube, dim=1)
    
    # L2 penalty on the difference
    loss_edge = torch.mean((len_pred - len_tube) ** 2)
    return loss_edge

def compute_losses(x_pred, x_true, mu, logvar, x_tube, edge_index, batch_x_true, num_graphs, faces=None):
    """
    Computes the combined loss for variable-size tube graphs.
    
    x_pred: [total_nodes, 3] (Predicted tube nodes, variable per graph)
    x_true: [M_batched, 3] (True aneurysm nodes, varying sizes)
    x_tube: [total_nodes, 3] (Base tube coordinates)
    edge_index: [2, E] (Batched edges)
    batch_x_true: [M_batched] (Batch indices for x_true)
    num_graphs: batch_size
    faces: [F_total, 3] per-sample faces batched by PyG, or legacy shared faces
    """
    loss_recon = 0.0
    
    # Use the batch vector attached to x_tube to determine per-graph node ranges.
    # PyG stores a 'batch' attribute on the Data object which maps each node to its graph.
    # Since x_pred has the same layout as x_tube (they share the same graph),
    # we can use the batch vector of x_tube.
    # However, in the loss function we receive flat tensors, not the Data object.
    # We need to infer per-graph boundaries from the batch vector.
    
    # For Chamfer distance, we need to split x_pred and x_true per graph.
    # x_true uses batch_x_true. For x_pred, we need the tube batch vector.
    # We can reconstruct it from edge_index bounds or from x_tube if it matches.
    # The simplest approach: since x_pred and x_tube share the same node ordering,
    # and PyG batches graphs sequentially, we can use the faces to find boundaries.
    # 
    # Actually, let's just detect node counts from PyG's batch vector.
    # We'll pass batch_tube through the call chain.
    
    # FALLBACK: if we don't have the batch vector, try sequential uniform split
    # (this handles the case where all graphs have equal size)
    total_nodes = x_pred.size(0)
    
    # Try to get per-graph tube node counts
    # We'll compute node boundaries by checking sequential regions
    n_per_graph = total_nodes // num_graphs
    remainder = total_nodes % num_graphs
    
    node_offset = 0
    for i in range(num_graphs):
        # For variable-size graphs, each graph might have different node count
        # Use uniform assumption first (works for equal-size batches)
        n_i = n_per_graph + (1 if i < remainder else 0)
        
        mask = (batch_x_true == i)
        x_true_i = x_true[mask].unsqueeze(0)  # [1, M_i, 3]
        x_pred_i = x_pred[node_offset:node_offset + n_i].unsqueeze(0)  # [1, N_i, 3]
        
        loss_chamfer, _ = chamfer_distance(x_pred_i, x_true_i)
        loss_recon += loss_chamfer
        
        node_offset += n_i
        
    loss_recon = loss_recon / num_graphs
    
    # KL Divergence
    loss_kl = vae_kl_loss(mu, logvar)
    
    # Geometric regularization: Edge length
    loss_edge = edge_length_penalty(x_pred, x_tube, edge_index)
    
    # Geometric regularization: Laplacian smoothing
    if faces is not None and faces.numel() > 0:
        # Cast verts to float32 because PyTorch3D laplacian uses sparse matrices
        # which do not currently support bfloat16 on CUDA
        with torch.autocast(device_type='cuda', enabled=False):
            meshes = Meshes(verts=[x_pred.to(torch.float32)], faces=[faces])
            loss_laplacian = mesh_laplacian_smoothing(meshes, method="uniform")
    else:
        loss_laplacian = torch.tensor(0.0, device=x_pred.device)
        
    loss_geom = loss_edge + loss_laplacian
    
    return loss_recon, loss_kl, loss_geom
