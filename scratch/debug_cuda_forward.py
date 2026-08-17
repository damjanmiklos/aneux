"""Isolate which CUDA op illegal-accesses on a real cached vessel sample."""
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

from aneux_paths import (
    CENTERLINES,
    CSV_PATH,
    EXPERIMENT_CACHE,
    EXTRA_CENTERLINES,
    VESSELS_AREA005,
)
from config import FOLLOW_BATCH
from dataset import AneurysmDataset
from model import GraphVAE, SetAbstraction
from ops import ball_query_packed, fps_indices, radius_graph_packed


def _sync(label: str) -> None:
    torch.cuda.synchronize()
    print(f"  ok  {label}", flush=True)


def main() -> None:
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    print("device", torch.cuda.get_device_name(0))
    print("pyg-lib", end=" ")
    try:
        import pyg_lib

        print(pyg_lib.__version__)
    except Exception as exc:
        print("FAILED", exc)

    print("Loading dataset (cache)...", flush=True)
    ds = AneurysmDataset(
        csv_path=CSV_PATH,
        vtp_vessel_dir=VESSELS_AREA005,
        vtp_centerline_dir=CENTERLINES,
        extra_centerline_dir=EXTRA_CENTERLINES,
        cache_dir=EXPERIMENT_CACHE,
        n_true=4096,
    )
    print("n samples", len(ds), "cache", EXPERIMENT_CACHE)
    data = ds[0]
    x = data.x_true
    print(
        "x_true",
        tuple(x.shape),
        x.dtype,
        "finite",
        bool(torch.isfinite(x).all()),
        "min",
        float(x.min()),
        "max",
        float(x.max()),
    )

    loader = DataLoader([data], batch_size=1, follow_batch=FOLLOW_BATCH)
    batch = next(iter(loader)).to(device)
    pos = batch.x_true.contiguous()
    bvec = batch.x_true_batch.contiguous()
    print("batch x_true", pos.shape, pos.dtype, "batch", bvec[:8], bvec[-1])

    print("\n== kernels on real x_true ==")
    try:
        idx = fps_indices(pos, 1024)
        _sync("pytorch3d/path fps 4096->1024")
        print("    idx", idx.dtype, int(idx.min()), int(idx.max()), idx.shape)
    except Exception:
        traceback.print_exc()
        return

    try:
        from pytorch3d.ops import sample_farthest_points

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loc = sample_farthest_points(pos.unsqueeze(0), K=1024, random_start_point=False)
        _sync("pytorch3d fps under bf16 autocast")
        print("    loc dtype", loc.dtype)
    except Exception:
        traceback.print_exc()

    new_pos = pos[idx]
    try:
        assign = ball_query_packed(pos, new_pos, 1.5, bvec, bvec[idx], 32)
        _sync("ball_query r=1.5 k=32")
        print("    edges", assign.shape, int(assign[0].max()), int(assign[1].max()))
    except Exception:
        traceback.print_exc()
        return

    try:
        ei = radius_graph_packed(new_pos, 1.5, bvec[idx], loop=True, max_num_neighbors=32)
        _sync("radius_graph 1024 r=1.5")
        print("    edges", ei.shape)
    except Exception:
        traceback.print_exc()
        return

    print("\n== SetAbstraction stage 0 ==")
    try:
        sa = SetAbstraction(32, 64, 1024, 1.5, 32).to(device)
        h = torch.randn(pos.size(0), 32, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hout, pout, bout = sa(h, pos, bvec)
        _sync("SA0 under autocast")
        print("    h", hout.shape, hout.dtype)
    except Exception:
        traceback.print_exc()
        print("retry SA0 without autocast")
        try:
            torch.cuda.empty_cache()
            sa = SetAbstraction(32, 64, 1024, 1.5, 32).to(device)
            h = torch.randn(pos.size(0), 32, device=device)
            hout, pout, bout = sa(h, pos, bvec)
            _sync("SA0 fp32")
        except Exception:
            traceback.print_exc()
            return

    print("\n== full GraphVAE forward (fp32) ==")
    try:
        model = GraphVAE().to(device)
        model.train()
        out = model(batch)
        _sync("full forward fp32")
        print("    x_pred", out.x_pred.shape, float(out.x_pred.abs().mean()))
    except Exception:
        traceback.print_exc()
        return

    print("\n== full GraphVAE forward+backward (bf16 autocast) ==")
    try:
        model = GraphVAE().to(device)
        model.train()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(batch)
            loss = out.x_pred.float().pow(2).mean() + out.mu.float().pow(2).mean()
        loss.backward()
        _sync("full backward autocast")
        print("    loss", float(loss))
    except Exception:
        traceback.print_exc()
        return

    print("\nALL STEPS PASSED")


if __name__ == "__main__":
    main()
