import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool, PointTransformerConv, SplineConv
from torch_cluster import knn_graph

from torch.utils.checkpoint import checkpoint

def MLP(channels):
    return nn.Sequential(
        nn.Linear(channels[0], channels[1]),
        nn.LeakyReLU(),
        nn.LayerNorm(channels[1]),
        nn.Linear(channels[1], channels[2]),
        nn.LeakyReLU(),
        nn.LayerNorm(channels[2])
    )

class Encoder(nn.Module):
    def __init__(self, latent_dim=128, k=32):
        super().__init__()
        self.k = k
        
        # Initial projection of 3D coordinates
        self.lin_in = nn.Linear(3, 32)
        
        # PointTransformer layers
        pos_nn1 = MLP([3, 32, 32])
        attn_nn1 = MLP([32, 32, 32])
        self.conv1 = PointTransformerConv(32, 32, pos_nn=pos_nn1, attn_nn=attn_nn1)
        
        pos_nn2 = MLP([3, 64, 64])
        attn_nn2 = MLP([64, 64, 64])
        self.conv2 = PointTransformerConv(32, 64, pos_nn=pos_nn2, attn_nn=attn_nn2)
        
        pos_nn3 = MLP([3, 128, 128])
        attn_nn3 = MLP([128, 128, 128])
        self.conv3 = PointTransformerConv(64, 128, pos_nn=pos_nn3, attn_nn=attn_nn3)
        
        self.lin_out = nn.Linear(128, latent_dim)
        
        # VAE Bottleneck
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)

    def forward(self, x, batch):
        """
        x: [M, 3] raw point cloud coordinates
        batch: [M] batch assignment vector for PyG
        """
        # Dynamic KNN graph construction
        edge_index = knn_graph(x, k=self.k, batch=batch, loop=True)
        
        # Initial feature embedding
        h = self.lin_in(x)
        
        # Define wrapper functions for checkpointing
        def run_conv1(h_in, pos, edge_idx):
            return F.leaky_relu(self.conv1(h_in, pos, edge_idx))
            
        def run_conv2(h_in, pos, edge_idx):
            return F.leaky_relu(self.conv2(h_in, pos, edge_idx))
            
        def run_conv3(h_in, pos, edge_idx):
            return F.leaky_relu(self.conv3(h_in, pos, edge_idx))
        
        # PointTransformer message passing using Gradient Checkpointing
        # use_reentrant=False is the modern PyTorch standard
        h = checkpoint(run_conv1, h, x, edge_index, use_reentrant=False)
        h = checkpoint(run_conv2, h, x, edge_index, use_reentrant=False)
        h = checkpoint(run_conv3, h, x, edge_index, use_reentrant=False)
        
        h = F.leaky_relu(self.lin_out(h))
        
        # Global max pooling
        h_global = global_max_pool(h, batch) # [batch_size, 256]
        
        # Project to latent parameters
        mu = self.fc_mu(h_global)
        logvar = self.fc_logvar(h_global)
        
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=64):
        super().__init__()
        # Input features: latent_dim (128) + 3D coordinates (3) = 131
        in_channels = latent_dim + 3
        
        # SplineConv layers
        self.conv1 = SplineConv(in_channels, hidden_dim, dim=3, kernel_size=5, degree=2)
        self.conv2 = SplineConv(hidden_dim, hidden_dim, dim=3, kernel_size=5, degree=2)
        self.conv3 = SplineConv(hidden_dim, hidden_dim, dim=3, kernel_size=5, degree=2)
        self.conv4 = SplineConv(hidden_dim, hidden_dim, dim=3, kernel_size=5, degree=2)
        
        # Output head projecting back to 3 dimensions (XYZ displacement)
        self.out_head = nn.Linear(hidden_dim, 3)

    def forward(self, z, x_tube, edge_index_tube, batch_tube, **kwargs):
        """
        z: [batch_size, latent_dim] sampled latent vectors
        x_tube: [total_nodes, 3] batched tube coordinates (variable per graph)
        edge_index_tube: [2, E_batched] batched adjacency
        batch_tube: [total_nodes] batch assignment vector for tube nodes
        """
        # Feature injection: Expand Z per node using batch assignment
        z_expand = z[batch_tube]  # [total_nodes, latent_dim]
        h = torch.cat([x_tube, z_expand], dim=-1)  # [total_nodes, latent_dim + 3]
        
        # Calculate continuous pseudo-coordinates for SplineConv
        row, col = edge_index_tube
        pseudo = x_tube[col] - x_tube[row]
        
        # Normalize pseudo to [0, 1] as required by SplineConv
        pseudo_min = pseudo.min(dim=0, keepdim=True)[0]
        pseudo_max = pseudo.max(dim=0, keepdim=True)[0]
        pseudo = (pseudo - pseudo_min) / (pseudo_max - pseudo_min + 1e-8)
        
        # MPNN layers with geometry-aware SplineConv
        h = F.elu(self.conv1(h, edge_index_tube, pseudo))
        h = F.elu(self.conv2(h, edge_index_tube, pseudo))
        h = F.elu(self.conv3(h, edge_index_tube, pseudo))
        h = F.elu(self.conv4(h, edge_index_tube, pseudo))
        
        # Final linear projection
        delta_x = self.out_head(h) # [batch_size * N, 3]
        
        # Geometric Operation
        x_pred = x_tube + delta_x
        
        return x_pred

class GraphVAE(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=64, k=32):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim, k=k)
        self.decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim)
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, data):
        # Encode
        mu, logvar = self.encoder(data.x_true, data.x_true_batch)
        
        # Sample
        z = self.reparameterize(mu, logvar)
        
        # Decode using batch vector for variable-size tube graphs
        x_pred = self.decoder(z, data.x, data.edge_index, data.batch)
        
        return x_pred, mu, logvar
