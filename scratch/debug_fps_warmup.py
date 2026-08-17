"""Confirm pytorch3d FPS needs a warmup before the first encoder stem."""
import os
import random
import sys

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "1test_encoder_decoder_only", "train_pipeline"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from aneux_paths import CENTERLINES, CSV_PATH, EXPERIMENT_CACHE, EXTRA_CENTERLINES, VESSELS_AREA005
from aneuxai import stratified_split
from config import FOLLOW_BATCH
from dataset import AneurysmDataset
from model import GraphVAE
from pytorch3d.ops import sample_farthest_points


def p3d(tag, x=None):
    if x is None:
        x = torch.zeros(1, 4, 3, device="cuda")
        k = 2
    else:
        x = x.unsqueeze(0) if x.dim() == 2 else x
        k = 32
    try:
        sample_farthest_points(x, K=k, random_start_point=False)
        torch.cuda.synchronize()
        print(tag, "OK", flush=True)
    except Exception as exc:
        print(tag, "FAIL", type(exc).__name__, flush=True)


random.seed(31)
np.random.seed(31)
torch.manual_seed(31)
ds = AneurysmDataset(
    csv_path=CSV_PATH,
    vtp_vessel_dir=VESSELS_AREA005,
    vtp_centerline_dir=CENTERLINES,
    extra_centerline_dir=EXTRA_CENTERLINES,
    cache_dir=EXPERIMENT_CACHE,
    n_true=4096,
)
train_ds, _ = stratified_split(ds, 0.15, 31)
batch = next(iter(DataLoader(train_ds, batch_size=1, shuffle=True, follow_batch=FOLLOW_BATCH))).to("cuda")
model = GraphVAE().cuda()
opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

print("no warmup path", flush=True)
h = model.encoder.stem(batch.x_true)
torch.cuda.synchronize()
print("stem ok", flush=True)
p3d("p3d after stem, no warmup")
