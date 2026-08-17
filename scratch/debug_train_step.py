"""Mimic aneuxai train_epoch on one real batch (autocast + full losses)."""
from __future__ import annotations

import os
import sys
import traceback

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
from torch_geometric.loader import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "1test_encoder_decoder_only", "train_pipeline"))
sys.path.insert(0, ROOT)

torch.sparse.check_sparse_tensor_invariants.enable()

from aneux_paths import CENTERLINES, CSV_PATH, EXPERIMENT_CACHE, EXTRA_CENTERLINES, VESSELS_AREA005
from config import FOLLOW_BATCH
from dataset import AneurysmDataset
from model import GraphVAE
from train import _autocast, losses_from_output, weighted_total
from config import DEFAULT_LOSS_WEIGHTS


def main() -> None:
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    ds = AneurysmDataset(
        csv_path=CSV_PATH,
        vtp_vessel_dir=VESSELS_AREA005,
        vtp_centerline_dir=CENTERLINES,
        extra_centerline_dir=EXTRA_CENTERLINES,
        cache_dir=EXPERIMENT_CACHE,
        n_true=4096,
    )
    loader = DataLoader([ds[0]], batch_size=1, follow_batch=FOLLOW_BATCH, pin_memory=True)
    batch = next(iter(loader)).to(device)

    model = GraphVAE().to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

    print("forward+loss+backward under train autocast (no prior FPS probe)...", flush=True)
    try:
        with _autocast(device):
            out = model(batch)
            terms = losses_from_output(out, batch)
            loss = weighted_total(terms, DEFAULT_LOSS_WEIGHTS)
        print("forward ok", {k: float(v.detach()) for k, v in terms.items()}, flush=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        torch.cuda.synchronize()
        print("backward+step ok", float(loss.detach()), flush=True)
    except Exception:
        traceback.print_exc()
        return

    print("TRAIN STEP PASSED")


if __name__ == "__main__":
    main()
