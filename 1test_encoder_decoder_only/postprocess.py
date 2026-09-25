#!/usr/bin/env python3
"""Reconstruct every cached case with a Stage-2 checkpoint.

Uses the same μ-path forward and loss as validation. Meshes are written in
the original scanner frame (undo the canonical pose). One row per case goes
to ``per_case_losses.csv``; train / val / test means go to
``split_comparison.csv`` and the terminal.

Default weights are the EMA validation winner ``checkpoints/best_val_1.pt``.
``last.pt`` uses the EMA shadow when it is present.

Run from the repo (aneurysmgnn locally, aneuxai_env on Komondor)::

    python 1test_encoder_decoder_only/postprocess.py --run-dir path/to/run

Missing tube-cache files are built first. A ``.pt`` that is already in
``--cache-dir`` is loaded as-is and not written again.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys

import numpy as np
import pyvista as pv
import torch
from torch_geometric.loader import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_PIPELINE = os.path.join(_HERE, "train_pipeline")
for _path in (_REPO_ROOT, _PIPELINE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from aneux_paths import CLEANDATA, EXPERIMENT_CACHE, EXPERIMENT_OUTPUT
from config import (
    DECODER_HIDDEN_DIM,
    DEFAULT_LOSS_WEIGHTS,
    FOLLOW_BATCH,
    GECO_BETA_INIT,
    HIERARCHY_LEVELS,
    LATENT_DIM,
    LATENT_LEN,
    N_TRUE,
    TUBE_RADIUS_MM,
    configure_stage2_precision,
)
from dataset import AneurysmDataset, _finalize_item, _load_cached_graph
from model import GraphVAE
from train import _keep_meta_on_cpu, losses_from_output, weighted_total

LOSS_KEYS = (
    "recon",
    "rad",
    "kl",
    "kl_mean_raw",
    "rate_gap",
    "disp",
    "lap",
    "norm",
    "fold",
    "conf",
    "fold_tpl",
    "stretch",
)
WEIGHT_KEYS = ("recon", "rad", "kl", "disp", "lap", "norm", "fold", "conf")
SUMMARY_KEYS = ("total",) + LOSS_KEYS + tuple(f"w_{key}" for key in WEIGHT_KEYS)
SPLIT_ORDER = ("train", "val", "test")


def _is_state_dict(obj):
    if not isinstance(obj, dict) or not obj:
        return False
    if "shadow" in obj and "decay" in obj:
        return False
    return all(torch.is_tensor(value) for value in obj.values())


def extract_state_dict(ckpt, prefer_ema=True):
    """Return ``(state_dict, source)`` from a raw dict or a training checkpoint.

    ``best_val_*.pt`` stores the EMA weights under ``model``. ``last.pt``
    stores the live weights under ``model`` and the EMA under ``ema.shadow``.
    """
    if _is_state_dict(ckpt):
        return ckpt, "state_dict"
    if not isinstance(ckpt, dict):
        raise TypeError(f"checkpoint must be a dict, got {type(ckpt).__name__}")
    ema = ckpt.get("ema")
    if prefer_ema and isinstance(ema, dict) and _is_state_dict(ema.get("shadow")):
        return ema["shadow"], "ema.shadow"
    model = ckpt.get("model")
    if _is_state_dict(model):
        # best_val_*.pt stores the EMA weights in this key. last.pt stores
        # the live weights here; its EMA is taken from ema.shadow above.
        return model, "model"
    if isinstance(ema, dict) and _is_state_dict(ema.get("shadow")):
        return ema["shadow"], "ema.shadow"
    raise KeyError("checkpoint has no model state_dict or ema.shadow")


def _checkpoint_epoch(ckpt):
    if not isinstance(ckpt, dict):
        return None
    epoch = ckpt.get("epoch")
    if epoch is None and isinstance(ckpt.get("metrics"), dict):
        epoch = ckpt["metrics"].get("epoch")
    try:
        return None if epoch is None else int(epoch)
    except (TypeError, ValueError):
        return None


def beta_for_checkpoint(ckpt, run_dir=None):
    """β that validation used at the saved epoch, else ``GECO_BETA_INIT``."""
    if isinstance(ckpt, dict):
        direct = ckpt.get("geco_beta")
        if direct is None and isinstance(ckpt.get("metrics"), dict):
            metrics = ckpt["metrics"]
            direct = metrics.get("geco_beta", metrics.get("beta"))
        if direct is not None:
            try:
                return float(direct), "checkpoint"
            except (TypeError, ValueError):
                pass
    epoch = _checkpoint_epoch(ckpt)
    history = os.path.join(run_dir or "", "data", "epoch_metrics.jsonl")
    if epoch is not None and os.path.isfile(history):
        with open(history, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    row_epoch = int(row.get("epoch", -1))
                except (TypeError, ValueError):
                    continue
                if row_epoch != epoch:
                    continue
                beta = row.get("geco_beta", row.get("beta"))
                if beta is not None:
                    return float(beta), "epoch_metrics"
    return float(GECO_BETA_INIT), "GECO_BETA_INIT"


def find_checkpoint(run_dir):
    names = (
        os.path.join("checkpoints", "best_val_1.pt"),
        "best_val_1.pt",
        os.path.join("checkpoints", "last.pt"),
        "last.pt",
    )
    for name in names:
        path = os.path.join(run_dir, name)
        if os.path.isfile(path):
            return path
    return None


def latest_run_dir(output_root):
    runs = os.path.join(output_root, "runs")
    if not os.path.isdir(runs):
        return None
    found = []
    for name in os.listdir(runs):
        path = os.path.join(runs, name)
        if os.path.isdir(path) and find_checkpoint(path):
            found.append(path)
    if not found:
        return None
    found.sort(key=os.path.getmtime, reverse=True)
    return found[0]


def load_split(path):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    groups = {}
    for name in SPLIT_ORDER:
        ids = payload.get(name) or []
        groups[name] = {str(item) for item in ids}
    return groups


def split_of(case_id, groups):
    key = str(case_id)
    for name in SPLIT_ORDER:
        if key in groups.get(name, ()):
            return name
    return "unassigned"


def find_split_file(run_dir, output_root):
    candidates = [
        os.path.join(run_dir, "data", "train_val_split.json"),
        os.path.join(run_dir, "train_val_split.json"),
        os.path.join(output_root, "train_val_split.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def canonical_to_world(x_pred, data):
    """Undo ``(world - origin) @ R`` so the mesh matches cleandata."""
    x = x_pred.detach()
    pose = getattr(data, "pose_R", None)
    if torch.is_tensor(pose) and pose.numel() >= 9:
        rotation = pose.detach().to(device=x.device, dtype=x.dtype).reshape(-1, 3, 3)[0]
        x = x @ rotation.transpose(0, 1)
    origin = getattr(data, "origin_shift", None)
    if torch.is_tensor(origin) and origin.numel() >= 3:
        shift = origin.detach().to(device=x.device, dtype=x.dtype).reshape(-1, 3)[0]
        x = x + shift
    return x


def _faces_np(data):
    face = getattr(data, "face", None)
    if face is None or not torch.is_tensor(face) or face.numel() == 0:
        raise AttributeError("sample has no faces")
    arr = face.detach().cpu().numpy()
    if arr.shape[0] == 3:
        return arr.T
    return arr


def write_predicted_vtp(x_world, faces, path):
    vertices = x_world.detach().cpu().to(torch.float64).numpy()
    faces = np.asarray(faces)
    pad = np.full((faces.shape[0], 1), 3, dtype=np.int64)
    mesh = pv.PolyData(vertices, np.hstack((pad, faces.astype(np.int64, copy=False))).ravel())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mesh.save(path)
    return mesh


def _as_float(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        return float(value.detach().float().reshape(-1).mean().cpu())
    return float(value)


def loss_row(terms, weights, kl_beta):
    row = {}
    for key in LOSS_KEYS:
        row[key] = _as_float(terms.get(key))
    for key in WEIGHT_KEYS:
        raw = row.get(key)
        weight = float(weights.get(key, 0.0))
        row[f"w_{key}"] = None if raw is None else raw * weight
    row["total"] = _as_float(weighted_total(terms, weights))
    row["beta"] = float(kl_beta)
    return row


def summarize_splits(rows):
    """Per-split mean / std / median / min / max for successful cases."""
    ok = [row for row in rows if row.get("status") == "ok"]
    groups = {name: [row for row in ok if row.get("split") == name] for name in SPLIT_ORDER}
    groups["all"] = ok
    summary = []
    for split, members in groups.items():
        if not members:
            continue
        for metric in SUMMARY_KEYS:
            values = []
            for row in members:
                value = row.get(metric)
                if value is None:
                    continue
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if number == number and number not in (float("inf"), float("-inf")):
                    values.append(number)
            if not values:
                continue
            summary.append(
                {
                    "split": split,
                    "metric": metric,
                    "n": len(values),
                    "mean": statistics.fmean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                }
            )
    return summary


def format_comparison(summary):
    metrics = ("total", "recon", "rad", "kl", "kl_mean_raw", "lap", "norm")
    splits = []
    for name in ("train", "val", "test", "all"):
        if any(row["split"] == name for row in summary):
            splits.append(name)
    lines = [
        f"{'split':<8} {'n':>5} "
        + " ".join(f"{metric:>12}" for metric in metrics)
    ]
    for split in splits:
        by_metric = {row["metric"]: row for row in summary if row["split"] == split}
        n = next(iter(by_metric.values()))["n"] if by_metric else 0
        cells = []
        for metric in metrics:
            row = by_metric.get(metric)
            cells.append(f"{row['mean']:12.4f}" if row else f"{'—':>12}")
        lines.append(f"{split:<8} {n:5d} " + " ".join(cells))
    return "\n".join(lines)


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.8g}"
    return value


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def default_cache_workers():
    cpus = os.cpu_count() or 2
    return max(1, min(15, cpus - 1))


def load_cached_case(dataset, index):
    """Read an existing cache file. Does not call the mesh builder."""
    sample = dataset.samples[index]
    path = dataset._cache_path(sample["dataset_id"])
    return _finalize_item(_load_cached_graph(path))


def build_model(device):
    model = GraphVAE(
        latent_dim=LATENT_DIM,
        latent_len=LATENT_LEN,
        hidden_dim=DECODER_HIDDEN_DIM,
        tube_radius=TUBE_RADIUS_MM,
        gradient_checkpointing="off",
    )
    return model.to(device)


def resolve_device(name):
    if name:
        device = torch.device(name)
    elif torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")
    if device.type == "cuda":
        index = 0 if device.index is None else int(device.index)
        torch.cuda.set_device(index)
        device = torch.device("cuda", index)
    return device


def reconstruct_all(
    run_dir,
    checkpoint,
    out_dir,
    split_path,
    cache_dir,
    cleandata_root,
    device,
    prefer_ema=True,
    cache_workers=None,
):
    configure_stage2_precision()
    groups = load_split(split_path)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state, source = extract_state_dict(ckpt, prefer_ema=prefer_ema)
    kl_beta, beta_source = beta_for_checkpoint(ckpt, run_dir)
    weights = dict(DEFAULT_LOSS_WEIGHTS)
    weights["kl"] = 1.0

    model = build_model(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    n_fine = HIERARCHY_LEVELS[-1]
    dataset = AneurysmDataset(
        tube_radius=TUBE_RADIUS_MM,
        n_length=n_fine[0],
        n_radial=n_fine[1],
        cache_dir=cache_dir,
        n_true=N_TRUE,
        cleandata_root=cleandata_root,
        require_templates=True,
        ensure_derived=False,
    )
    workers = default_cache_workers() if cache_workers is None else max(1, int(cache_workers))
    print(f"Tube cache: {cache_dir}  (build missing only, {workers} workers)")
    cache_errors = {
        str(case_id): err
        for case_id, err in dataset.warmup_cache(num_workers=workers, strict=False)
    }

    os.makedirs(out_dir, exist_ok=True)
    meta = {
        "run_dir": os.path.abspath(run_dir),
        "checkpoint": os.path.abspath(checkpoint),
        "weight_source": source,
        "epoch": _checkpoint_epoch(ckpt),
        "kl_beta": kl_beta,
        "kl_beta_source": beta_source,
        "split_file": os.path.abspath(split_path),
        "cache_dir": os.path.abspath(cache_dir),
        "device": str(device),
        "forward": "eval μ (sample=False)",
        "loss_weights": weights,
    }
    with open(os.path.join(out_dir, "run_meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print(
        f"checkpoint={checkpoint}\n"
        f"weights={source}  epoch={meta['epoch']}  β={kl_beta:.6g} ({beta_source})\n"
        f"split={split_path}\n"
        f"writing {out_dir}"
    )

    rows = []
    case_fields = [
        "case_id",
        "split",
        "status",
        "mesh_path",
        "n_vertices",
        "n_faces",
        "total",
        *LOSS_KEYS,
        *[f"w_{key}" for key in WEIGHT_KEYS],
        "beta",
        "error",
    ]
    n = len(dataset)
    for index in range(n):
        case_id = str(dataset.samples[index]["dataset_id"])
        split = split_of(case_id, groups)
        base = {
            "case_id": case_id,
            "split": split,
            "status": "ok",
            "mesh_path": "",
            "n_vertices": "",
            "n_faces": "",
            "beta": kl_beta,
            "error": "",
        }
        prefix = f"[{index + 1}/{n}] {split} {case_id}"
        if not dataset._cache_file_ready(index):
            base["status"] = "missing_cache"
            base["error"] = cache_errors.get(case_id, "tube cache was not built")
            rows.append(base)
            print(f"{prefix}  {base['error']}")
            continue
        try:
            data = load_cached_case(dataset, index)
            loader = DataLoader([data], batch_size=1, follow_batch=FOLLOW_BATCH)
            batch = _keep_meta_on_cpu(next(iter(loader)).to(device))
            with torch.no_grad():
                out = model(batch, sample=False)
                terms = losses_from_output(out, batch, kl_beta=kl_beta)
            losses = loss_row(terms, weights, kl_beta)
            faces = _faces_np(data)
            mesh_path = os.path.join(out_dir, "meshes", split, f"{case_id}.vtp")
            write_predicted_vtp(canonical_to_world(out.x_pred, data), faces, mesh_path)
            base.update(losses)
            base["mesh_path"] = mesh_path
            base["n_vertices"] = int(out.x_pred.shape[0])
            base["n_faces"] = int(faces.shape[0])
            rows.append(base)
            print(
                f"{prefix}  total={losses['total']:.4f}  "
                f"recon={losses['recon']:.4f}  rad={losses['rad']:.4f}  "
                f"kl={losses['kl']:.4f}"
            )
        except Exception as exc:
            base["status"] = "error"
            base["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(base)
            print(f"{prefix}  ERROR {base['error']}")

    case_csv = os.path.join(out_dir, "per_case_losses.csv")
    write_csv(case_csv, rows, case_fields)
    summary = summarize_splits(rows)
    summary_csv = os.path.join(out_dir, "split_comparison.csv")
    write_csv(
        summary_csv,
        summary,
        ["split", "metric", "n", "mean", "std", "median", "min", "max"],
    )
    n_ok = sum(1 for row in rows if row["status"] == "ok")
    n_bad = len(rows) - n_ok
    print(f"\nSaved {n_ok} meshes, {n_bad} not reconstructed")
    print(f"Per-case losses: {case_csv}")
    print(f"Split comparison: {summary_csv}\n")
    print(format_comparison(summary))
    return rows, summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Copied run folder (contains checkpoints/ and data/train_val_split.json). "
        "Defaults to the newest run under 1test_encoder_decoder_only/output/runs.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Override the checkpoint. Default: checkpoints/best_val_1.pt, else last.pt.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output folder. Default: <run-dir>/reconstructions.",
    )
    parser.add_argument("--split", default=None, help="Override train_val_split.json.")
    parser.add_argument("--cache-dir", default=EXPERIMENT_CACHE)
    parser.add_argument(
        "--cache-workers",
        type=int,
        default=None,
        help="Processes for missing tube-cache files only. Existing files are not rebuilt.",
    )
    parser.add_argument("--cleandata", default=CLEANDATA)
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu.")
    parser.add_argument(
        "--live-weights",
        action="store_true",
        help="On last.pt, use the live weights instead of the EMA shadow.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_dir = args.run_dir or latest_run_dir(EXPERIMENT_OUTPUT)
    if not run_dir or not os.path.isdir(run_dir):
        raise SystemExit(
            "No run folder. Pass --run-dir pointing at the copied "
            "output/runs/<stamp>_job<id> directory."
        )
    run_dir = os.path.abspath(run_dir)
    checkpoint = args.checkpoint or find_checkpoint(run_dir)
    if not checkpoint or not os.path.isfile(checkpoint):
        raise SystemExit(f"No checkpoint under {run_dir}. Pass --checkpoint.")
    split_path = args.split or find_split_file(run_dir, EXPERIMENT_OUTPUT)
    if not split_path or not os.path.isfile(split_path):
        raise SystemExit(
            "No train_val_split.json. Expected "
            f"{os.path.join(run_dir, 'data', 'train_val_split.json')}."
        )
    out_dir = os.path.abspath(args.out_dir or os.path.join(run_dir, "reconstructions"))
    device = resolve_device(args.device)
    reconstruct_all(
        run_dir=run_dir,
        checkpoint=os.path.abspath(checkpoint),
        out_dir=out_dir,
        split_path=os.path.abspath(split_path),
        cache_dir=os.path.abspath(args.cache_dir),
        cleandata_root=os.path.abspath(args.cleandata),
        device=device,
        prefer_ema=not args.live_weights,
        cache_workers=args.cache_workers,
    )


if __name__ == "__main__":
    main()
