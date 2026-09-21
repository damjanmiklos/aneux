"""Timestamped run directories, top-k checkpoints, and training plots/CSV dumps."""
from __future__ import annotations

import csv
import json
import math
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone

import numpy as np

try:
    import torch
except ImportError:
    torch = None


PLOT_SERIES = (
    ("loss", "val_loss", "Total loss"),
    ("recon", "val_recon", "Reconstruction (Chamfer)"),
    ("rad", "val_rad", "Radial Huber"),
    ("kl", "val_kl", "KL (weighted)"),
    ("kl_mean_raw", "val_kl_mean_raw", "KL raw nats/token"),
    ("disp", "val_disp", "Displacement Dirichlet"),
    ("lap", "val_lap", "Laplacian"),
    ("norm", "val_norm", "Normal consistency"),
    ("stretch", "val_stretch", "Triangle stretch"),
    ("fold", "val_fold", "Fold penalty"),
    ("geco_beta", None, "GECO β"),
    ("rate_gap", "val_rate_gap", "Rate gap KL̄−R*"),
    ("lr", None, "Learning rate"),
    ("ema_decay_used", None, "EMA decay used"),
    ("samples_per_sec", None, "Throughput (samples/s)"),
    ("epoch_seconds", None, "Wall seconds / epoch"),
    ("peak_vram_gib", None, "Peak allocated VRAM (GiB)"),
    ("val_recon_sample_gap", None, "recon(σ) − recon(μ)"),
)


def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def make_run_dir(output_root, job_id=None, stamp=None):
    """Create ``<output_root>/runs/<stamp>[_job<id>]/`` with plots/ and checkpoints/."""
    stamp = stamp or utc_stamp()
    if job_id:
        name = f"{stamp}_job{job_id}"
    else:
        name = stamp
    run_dir = os.path.join(os.path.abspath(output_root), "runs", name)
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "tb"), exist_ok=True)
    return run_dir


def dump_json(path, payload):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
    return path


def _json_default(value):
    if torch is not None and torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def hardware_snapshot():
    info = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "env": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "LOCAL_RANK",
                "RANK",
                "WORLD_SIZE",
                "SLURM_JOB_ID",
                "SLURM_JOB_NAME",
                "SLURM_JOB_ACCOUNT",
                "SLURM_JOB_PARTITION",
                "SLURM_GPUS_ON_NODE",
                "SLURM_CPUS_ON_NODE",
                "SLURM_JOB_NUM_NODES",
                "SLURM_NTASKS",
                "SLURM_MEM_PER_NODE",
                "SLURM_MEM_PER_CPU",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NCCL_DEBUG",
            )
            if os.environ.get(key)
        },
    }
    if torch is not None:
        info["torch"] = getattr(torch, "__version__", None)
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            gpus = []
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                free_b, total_b = torch.cuda.mem_get_info(idx)
                gpus.append(
                    {
                        "index": idx,
                        "name": torch.cuda.get_device_name(idx),
                        "total_gib": round(props.total_memory / 1024**3, 3),
                        "driver_free_gib": round(free_b / 1024**3, 3),
                        "driver_total_gib": round(total_b / 1024**3, 3),
                        "major": props.major,
                        "minor": props.minor,
                    }
                )
            info["gpus"] = gpus
    try:
        info["nvidia_smi"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,utilization.memory,temperature.gpu",
                "--format=csv",
            ],
            text=True,
            timeout=20,
        )
    except Exception as exc:
        info["nvidia_smi_error"] = str(exc)
    for cmd, key in (
        (["lscpu"], "lscpu"),
        (["free", "-h"], "free"),
        (["uname", "-a"], "uname"),
        (["df", "-h", "/scratch", "/project", "/home"], "df"),
    ):
        try:
            info[key] = subprocess.check_output(cmd, text=True, timeout=15)
        except Exception:
            pass
    return info


def capture_slurm_job_stats(job_id=None):
    """Best-effort seff / sacct / jobstats / sstat snapshot.

    Called from Python at the end of training, so the Slurm job is still
    RUNNING. ``seff`` CPU/memory efficiency is then incomplete (Komondor
    docs). GPU time-series and a usable efficiency report come from
    ``jobstats`` (https://jobstats.komondor.hinfra.hu) and from the sbatch
    dump after ``srun`` returns. ``sstat`` MaxDiskRead/Write is the IO
    accounting Slurm exposes while the step is alive.
    """
    job_id = job_id or os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return None
    out = {"job_id": str(job_id), "note": "job still allocated; seff may be incomplete"}
    queries = (
        (["seff", str(job_id)], "seff"),
        (
            [
                "sacct",
                "-j",
                str(job_id),
                "-o",
                "JobID,State,ExitCode,Elapsed,MaxRSS,ReqMem,AllocTRES,TotalCPU,MaxDiskRead,MaxDiskWrite",
                "--parsable2",
            ],
            "sacct",
        ),
        (["jobstats", str(job_id)], "jobstats"),
        (
            [
                "sstat",
                "--allsteps",
                "-j",
                str(job_id),
                "-o",
                "JobID,MaxRSS,MaxVMSize,AveCPU,MaxDiskRead,MaxDiskWrite",
            ],
            "sstat",
        ),
    )
    for cmd, key in queries:
        try:
            out[key] = subprocess.check_output(cmd, text=True, timeout=60)
        except Exception as exc:
            out[f"{key}_error"] = str(exc)
    return out


class TopKCheckpoints:
    """Keep the k lowest scores on disk as ``{prefix}_1.pt`` … ``{prefix}_k.pt``."""

    def __init__(self, directory, prefix, k=3):
        self.directory = directory
        self.prefix = prefix
        self.k = int(k)
        self.entries = []
        os.makedirs(directory, exist_ok=True)

    def consider(self, score, payload, epoch, extra=None):
        score = float(score)
        if not math.isfinite(score):
            return False
        if len(self.entries) >= self.k and score >= self.entries[-1][0]:
            return False
        tmp = os.path.join(
            self.directory, f".{self.prefix}_epoch{int(epoch)}_{os.getpid()}.tmp"
        )
        torch.save(payload, tmp)
        self.entries.append((score, tmp, int(epoch), extra or {}))
        self.entries.sort(key=lambda row: (row[0], row[2]))
        dropped = self.entries[self.k :]
        self.entries = self.entries[: self.k]
        for _s, path, _e, _x in dropped:
            try:
                os.remove(path)
            except OSError:
                pass
        self._relabel()
        return True

    def _relabel(self):
        for rank_i, (score, path, epoch, extra) in enumerate(self.entries, start=1):
            dest = os.path.join(self.directory, f"{self.prefix}_{rank_i}.pt")
            if os.path.abspath(path) != os.path.abspath(dest):
                try:
                    os.replace(path, dest)
                except OSError:
                    shutil.copy2(path, dest)
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                self.entries[rank_i - 1] = (score, dest, epoch, extra)
        self._write_index()

    def _write_index(self):
        dump_json(
            os.path.join(self.directory, f"{self.prefix}_index.json"),
            {
                "prefix": self.prefix,
                "k": self.k,
                "entries": [
                    {
                        "rank": i,
                        "score": score,
                        "epoch": epoch,
                        "path": os.path.basename(path),
                        **(extra or {}),
                    }
                    for i, (score, path, epoch, extra) in enumerate(self.entries, start=1)
                ],
            },
        )


def selection_score(metrics, weights, prefix=""):
    """Same criterion as the old best.pt: recon + λ_rad · rad."""
    recon = float(metrics.get(f"{prefix}recon", metrics.get("recon", float("inf"))))
    rad = float(metrics.get(f"{prefix}rad", metrics.get("rad", 0.0)))
    lam = float((weights or {}).get("rad", 1.0))
    return recon + lam * rad


def write_history_tables(history, run_dir):
    data_dir = os.path.join(run_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    dump_json(os.path.join(data_dir, "history.json"), history)
    jsonl_path = os.path.join(data_dir, "epoch_metrics.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for row in history:
            handle.write(json.dumps(row, default=_json_default) + "\n")
    csv_path = os.path.join(data_dir, "epoch_metrics.csv")
    fields = []
    for row in history:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in history:
            writer.writerow({k: row.get(k, "") for k in fields})
    return csv_path


def _series(history, key):
    xs, ys = [], []
    for i, row in enumerate(history, start=1):
        epoch = int(row.get("epoch", i))
        if key in row and row[key] is not None:
            try:
                val = float(row[key])
            except (TypeError, ValueError):
                continue
            if math.isfinite(val):
                xs.append(epoch)
                ys.append(val)
    return xs, ys


def plot_training_history(history, run_dir):
    """Write a grid of curves plus one PNG per metric. No-op if matplotlib is missing."""
    if not history:
        return []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    written = []

    n = len(PLOT_SERIES)
    cols = 3
    rows = int(math.ceil(n / float(cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.6 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (train_key, val_key, title) in zip(axes, PLOT_SERIES):
        tx, ty = _series(history, train_key)
        if tx:
            ax.plot(tx, ty, label=train_key)
        if val_key:
            vx, vy = _series(history, val_key)
            if vx:
                ax.plot(vx, vy, "o-", label=val_key)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        if tx or (val_key and _series(history, val_key)[0]):
            ax.legend(fontsize=8)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    grid_path = os.path.join(plot_dir, "training_overview.png")
    fig.savefig(grid_path, dpi=140)
    plt.close(fig)
    written.append(grid_path)

    for train_key, val_key, title in PLOT_SERIES:
        tx, ty = _series(history, train_key)
        vx, vy = _series(history, val_key) if val_key else ([], [])
        if not tx and not vx:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if tx:
            ax.plot(tx, ty, label=train_key)
        if vx:
            ax.plot(vx, vy, "o-", label=val_key)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        dest = os.path.join(plot_dir, f"{train_key}.png")
        fig.savefig(dest, dpi=140)
        plt.close(fig)
        written.append(dest)

    # Train vs val gap for the reconstruction terms.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    drawn = False
    for train_key, val_key, title in (
        ("recon", "val_recon", "recon"),
        ("loss", "val_loss", "total"),
        ("rad", "val_rad", "rad"),
    ):
        tx, ty = _series(history, train_key)
        vx, vy = _series(history, val_key)
        if not tx or not vx:
            continue
        vmap = dict(zip(vx, vy))
        xs, gap = [], []
        for e, tval in zip(tx, ty):
            if e in vmap:
                xs.append(e)
                gap.append(vmap[e] - tval)
        if xs:
            ax.plot(xs, gap, label=f"val-train {title}")
            drawn = True
    if drawn:
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_title("Validation − train (overfitting gap)")
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        dest = os.path.join(plot_dir, "overfit_gap.png")
        fig.savefig(dest, dpi=140)
        written.append(dest)
    plt.close(fig)
    return written


def write_run_summary(run_dir, history, extra=None):
    """Compact end-of-job JSON: best scores, last epoch, checkpoint inventory."""
    inf = float("inf")

    def _best(key):
        rows = [r for r in history if r.get(key) is not None]
        if not rows:
            return None
        row = min(rows, key=lambda r: float(r.get(key, inf)))
        return {"epoch": row.get("epoch"), key: row.get(key), "loss": row.get("loss")}

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    ckpts = []
    if os.path.isdir(ckpt_dir):
        for name in sorted(os.listdir(ckpt_dir)):
            path = os.path.join(ckpt_dir, name)
            if os.path.isfile(path):
                ckpts.append({"name": name, "bytes": os.path.getsize(path)})
    payload = {
        "n_epochs_logged": len(history),
        "last": history[-1] if history else None,
        "best_train_recon": _best("recon"),
        "best_val_recon": _best("val_recon"),
        "best_train_score_recon_plus_rad": None,
        "best_val_score_recon_plus_rad": None,
        "checkpoints": ckpts,
        "plot_dir": os.path.join(run_dir, "plots"),
        "data_dir": os.path.join(run_dir, "data"),
    }
    if extra:
        payload.update(extra)
    return dump_json(os.path.join(run_dir, "data", "run_summary.json"), payload)


def write_run_readme(run_dir, extra_lines=None):
    lines = [
        f"Stage-2 run directory created {utc_stamp()} UTC",
        "",
        "data/epoch_metrics.csv     per-epoch scalars (train + val + latent + throughput)",
        "data/epoch_metrics.jsonl   same rows, one JSON object per epoch",
        "data/history.json          full history list",
        "data/run_config.json       hyperparameters and paths used for this job",
        "data/run_summary.json      best train/val scores and checkpoint inventory",
        "data/resource_snapshots.jsonl  CPU/GPU/IO every 15s (ANEUX_MONITOR_SEC)",
        "data/resource_snapshots.csv    same snapshots as a table",
        "plots/resource_util.png        GPU util, CPU util, IO MiB/s vs time",
        "plots/resource_gpu_mem.png     GPU memory vs time",
        "data/slurm_jobstats.json   seff/sacct/jobstats/sstat (partial while job still runs)",
        "data/seff.txt              written by sbatch after srun (CPU/mem efficiency)",
        "data/jobstats.txt          Komondor jobstats CLI (CPU/mem/GPU over time)",
        "data/sstat.txt             MaxRSS / AveCPU / MaxDiskRead / MaxDiskWrite",
        "checkpoints/last.pt        resumable (model + EMA + AdamW + RNG + GECO β)",
        "checkpoints/best_train_1..3.pt   three lowest train recon+λ·rad (live weights)",
        "checkpoints/best_val_1..3.pt     three lowest val recon+λ·rad (EMA μ path)",
        "plots/                     one PNG per metric plus training_overview.png",
        "tb/                        TensorBoard scalars if torch.utils.tensorboard is present",
        "",
    ]
    if extra_lines:
        lines.extend(extra_lines)
        lines.append("")
    path = os.path.join(run_dir, "README.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path
