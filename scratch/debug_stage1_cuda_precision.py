"""One CUDA train step under Stage-2 FP32+TF32 (no autocast)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "1test_encoder_decoder_only", "train_pipeline"))

import torch
from torch_geometric.loader import DataLoader

from config import FOLLOW_BATCH, configure_stage2_precision
from losses import compute_losses
from test_architecture import GraphVAE, make_synthetic_data

configure_stage2_precision()
assert torch.cuda.is_available()
assert torch.get_float32_matmul_precision() == "high"
assert torch.backends.cuda.matmul.allow_tf32

data = make_synthetic_data()
batch = next(iter(DataLoader([data], batch_size=1, follow_batch=FOLLOW_BATCH))).cuda()
model = GraphVAE(
    latent_dim=8,
    latent_len=8,
    hidden_dim=16,
    tube_radius=2.0,
    sa_stages=((32, 4.0, 8, 32, 1), (8, 8.0, 8, 64, 1)),
).cuda()
model.train()
out = model(batch)
assert out.x_pred.dtype == torch.float32
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
    x_pred_mid=out.x_pred_mid,
    batch_mid=batch.pos_mid_batch,
    x_pred_coarse=out.x_pred_coarse,
    batch_coarse=batch.pos_coarse_batch,
)
loss = terms["recon"] + terms["kl"] + terms["disp"] + terms["lap"] + terms["norm"]
loss.backward()
torch.cuda.synchronize()
print("cuda_ok", out.x_pred.dtype, float(loss), torch.get_float32_matmul_precision())
print("finite", {k: bool(torch.isfinite(v).all()) for k, v in terms.items()})
