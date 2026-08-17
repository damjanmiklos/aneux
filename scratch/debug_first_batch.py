"""Reproduce aneuxai's first training batch and print the FPS probe error."""
from __future__ import annotations

import os
import random
import sys
import traceback

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "1test_encoder_decoder_only", "train_pipeline"))
sys.path.insert(0, ROOT)

from aneux_paths import CENTERLINES, CSV_PATH, EXPERIMENT_CACHE, EXTRA_CENTERLINES, VESSELS_AREA005
from aneuxai import stratified_split
from config import FOLLOW_BATCH
from dataset import AneurysmDataset
from model import GraphVAE
from ops import _pytorch3d_fps_ok, fps_indices
from train import _autocast, losses_from_output, weighted_total
from config import DEFAULT_LOSS_WEIGHTS

SEED = 31


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    ds = AneurysmDataset(
        csv_path=CSV_PATH,
        vtp_vessel_dir=VESSELS_AREA005,
        vtp_centerline_dir=CENTERLINES,
        extra_centerline_dir=EXTRA_CENTERLINES,
        cache_dir=EXPERIMENT_CACHE,
        n_true=4096,
    )
    train_ds, _ = stratified_split(ds, 0.15, SEED)
    loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        follow_batch=FOLLOW_BATCH,
        pin_memory=True,
        num_workers=0,
    )
    batch = next(iter(loader)).to(device)
    x = batch.x_true
    print(
        "first shuffled sample",
        "x_true", tuple(x.shape), x.dtype,
        "finite", bool(torch.isfinite(x).all()),
        "min", float(x.min()), "max", float(x.max()),
        "tube", tuple(batch.x.shape),
        "edges", tuple(batch.edge_index.shape),
        flush=True,
    )
    ei = batch.edge_index
    print("edge min/max", int(ei.min()), int(ei.max()), "n", batch.x.size(0), flush=True)

    print("FPS probe", _pytorch3d_fps_ok(device), flush=True)
    idx = fps_indices(x, 1024)
    print("fps", idx.shape, int(idx.min()), int(idx.max()), flush=True)

    model = GraphVAE().to(device)
    model.train()
    print("running model under autocast...", flush=True)
    try:
        with _autocast(device):
            out = model(batch)
        terms = losses_from_output(out, batch)
        loss = weighted_total(terms, DEFAULT_LOSS_WEIGHTS)
        loss.backward()
        torch.cuda.synchronize()
        print("ok", {k: float(v.detach()) for k, v in terms.items()}, flush=True)
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()
