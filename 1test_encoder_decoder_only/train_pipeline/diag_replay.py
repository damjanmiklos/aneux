"""Replay one training step dumped by crash_diag (``crash_rank*_e*_s*.pt``).

Single process, single GPU, ``CUDA_LAUNCH_BLOCKING=1`` so a fault is raised by
the kernel that caused it. Run it under ``compute-sanitizer`` to get the
offending kernel, thread and address even when the access happens to hit mapped
memory::

    compute-sanitizer --tool memcheck --launch-timeout 0 \\
        python diag_replay.py crash_rank1_e9_s31.pt --ckpt last.pt

The step re-seeds the CPU and CUDA generators with the states saved at the start
of the crashed step, so mirror / pose jitter / x_true resampling draw the same
numbers. ``--ckpt`` loads the model weights from the epoch before (last.pt);
without it the model is freshly initialised, which still exercises every kernel
on the same data. ``--repeat`` runs the step more than once (the augmentation
RNG is restored each time).
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", help="crash_rank*_e*_s*.pt written by crash_diag")
    ap.add_argument("--ckpt", default=None, help="last.pt (model weights) from the epoch before the crash")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--no-aug", action="store_true", help="skip the train augmentations")
    args = ap.parse_args()

    from config import (
        DECODER_HIDDEN_DIM, DEFAULT_LOSS_WEIGHTS as LOSS_WEIGHTS, LATENT_DIM, LATENT_LEN, TUBE_RADIUS_MM,
    )
    from model import GraphVAE
    from train import train_epoch

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    try:
        entry = torch.load(args.dump, map_location="cpu", weights_only=False)
    except TypeError:
        entry = torch.load(args.dump, map_location="cpu")
    print(f"dump: epoch {entry['epoch']} step {entry['step']} ids={entry['ids']}")
    flags = [f for f in entry.get("flags", []) if f]
    if flags:
        print(f"scan flags recorded for this batch: {flags}")

    model = GraphVAE(
        latent_dim=LATENT_DIM,
        latent_len=LATENT_LEN,
        hidden_dim=DECODER_HIDDEN_DIM,
        tube_radius=TUBE_RADIUS_MM,
        gradient_checkpointing="off",
    ).to(device)
    if args.ckpt:
        try:
            ck = torch.load(args.ckpt, map_location=device, weights_only=False)
        except TypeError:
            ck = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ck["model"] if "model" in ck else ck, strict=True)
        print(f"loaded weights from {args.ckpt} (epoch {ck.get('epoch')})")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

    for i in range(max(1, args.repeat)):
        torch.set_rng_state(entry["rng_cpu"])
        if entry.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(entry["rng_cuda"], device)
        batch = entry["batch"].clone()
        print(f"--- replay {i + 1}/{args.repeat}", flush=True)
        train_epoch(
            model, [batch], optimizer, LOSS_WEIGHTS, device,
            augment=not args.no_aug, epoch=int(entry["epoch"]) + 1, geco_beta=None,
        )
        torch.cuda.synchronize()
        print("step completed without a CUDA error", flush=True)


if __name__ == "__main__":
    main()
