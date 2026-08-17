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

        self.lin_in = nn.Linear(3, 32)

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

        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)

    def _run_conv(self, conv, h_in, pos, edge_idx):
        return F.leaky_relu(conv(h_in, pos, edge_idx))

    def forward(self, x, batch):
        """
        x: [M, 3] raw point cloud coordinates
        batch: [M] batch assignment vector for PyG
        """
        edge_index = knn_graph(x, k=self.k, batch=batch, loop=True)
        h = self.lin_in(x)

        def run_conv1(h_in, pos, edge_idx):
            return self._run_conv(self.conv1, h_in, pos, edge_idx)

        def run_conv2(h_in, pos, edge_idx):
            return self._run_conv(self.conv2, h_in, pos, edge_idx)

        def run_conv3(h_in, pos, edge_idx):
            return self._run_conv(self.conv3, h_in, pos, edge_idx)

        if self.training:
            h = checkpoint(run_conv1, h, x, edge_index, use_reentrant=False)
            h = checkpoint(run_conv2, h, x, edge_index, use_reentrant=False)
            h = checkpoint(run_conv3, h, x, edge_index, use_reentrant=False)
        else:
            h = run_conv1(h, x, edge_index)
            h = run_conv2(h, x, edge_index)
            h = run_conv3(h, x, edge_index)

        h = F.leaky_relu(self.lin_out(h))
        h_global = global_max_pool(h, batch)  # [batch_size, latent_dim]

        mu = self.fc_mu(h_global)
        logvar = torch.clamp(self.fc_logvar(h_global), -30.0, 20.0)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=64):
        super().__init__()
        in_channels = latent_dim + 3

        self.conv1 = SplineConv(in_channels, hidden_dim, dim=3, kernel_size=5, degree=2)
        self.conv2 = SplineConv(hidden_dim, hidden_dim, dim=3, kernel_size=5, degree=2)
        self.conv3 = SplineConv(hidden_dim, hidden_dim, dim=3, kernel_size=5, degree=2)
        self.conv4 = SplineConv(hidden_dim, hidden_dim, dim=3, kernel_size=5, degree=2)

        self.out_head = nn.Linear(hidden_dim, 3)

    def _normalize_pseudo(self, pseudo, edge_batch, num_graphs):
        """Min-max normalize SplineConv pseudo-coords independently per graph."""
        normed = torch.empty_like(pseudo)
        for g in range(num_graphs):
            mask = edge_batch == g
            if not mask.any():
                continue
            p = pseudo[mask]
            pmin = p.min(dim=0, keepdim=True).values
            pmax = p.max(dim=0, keepdim=True).values
            normed[mask] = (p - pmin) / (pmax - pmin + 1e-8)
        return normed

    def forward(self, z, x_tube, edge_index_tube, batch_tube):
        """
        z: [batch_size, latent_dim]
        x_tube: [total_nodes, 3]
        edge_index_tube: [2, E_batched]
        batch_tube: [total_nodes]
        """
        z_expand = z[batch_tube]
        h = torch.cat([x_tube, z_expand], dim=-1)

        row, col = edge_index_tube
        pseudo = x_tube[col] - x_tube[row]
        num_graphs = int(batch_tube.max().item()) + 1 if batch_tube.numel() else 1
        pseudo = self._normalize_pseudo(pseudo, batch_tube[row], num_graphs)

        h = F.elu(self.conv1(h, edge_index_tube, pseudo))
        h = h + F.elu(self.conv2(h, edge_index_tube, pseudo))
        h = h + F.elu(self.conv3(h, edge_index_tube, pseudo))
        h = h + F.elu(self.conv4(h, edge_index_tube, pseudo))

        delta_x = self.out_head(h)
        return x_tube + delta_x


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
        return mu

    def forward(self, data):
        mu, logvar = self.encoder(data.x_true, data.x_true_batch)
        z = self.reparameterize(mu, logvar)
        x_pred = self.decoder(z, data.x, data.edge_index, data.batch)
        return x_pred, mu, logvar
