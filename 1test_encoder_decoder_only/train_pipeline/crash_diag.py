"""Opt-in crash diagnostics for Stage-2 training (``ANEUX_DIAG=1``).

Nothing here runs unless ``ANEUX_DIAG`` is set, so the normal path is untouched.

Why it exists: jobs 14506605 and 14509551 died with ``CUDA error 700`` (illegal
memory access) that CUDA reported at the first synchronizing call after the real
fault, while the other ranks sat in an all-reduce until the 10-minute NCCL
watchdog. This module records enough to name the kernel and the sample:

* per-sample scan (in the DataLoader workers): non-finite floats, index tensors
  outside the point set they address, spline pseudo-coordinates that are not
  finite or leave [0, 1], and tensor sizes;
* per-step breadcrumb (one JSON line per rank per step): sample ids, sizes, GPU
  memory, loss and gradient norm;
* the current stage (augment / forward / loss / backward / optimizer / ...);
* on any exception: traceback, stage, sample ids, GPU state, and the CPU copy of
  the failing and previous batches plus the RNG states, so the step can be
  replayed with ``diag_replay.py`` (optionally under ``compute-sanitizer``);
* a stall watchdog that dumps every thread's stack if a rank stops making
  progress (a rank waiting in an all-reduce for a dead peer);
* a hard exit after the dump, so torchrun tears the job down at once instead of
  waiting for the NCCL watchdog.

Files go to ``ANEUX_DIAG_DIR`` (set it to a project-disk path: a rank that is
killed never runs the scratch copy-back).
"""

from __future__ import annotations

import faulthandler
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from collections import deque

import torch


def enabled() -> bool:
    return os.environ.get("ANEUX_DIAG", "").strip().lower() in ("1", "true", "yes")


# (index tensor, the point set it addresses)
_INDEX_PAIRS = (
    ("face", "x"),
    ("face_mid", "pos_mid"),
    ("face_coarse", "pos_coarse"),
    ("gt_faces", "gt_points"),
    ("edge_index", "x"),
    ("edge_index_mid", "pos_mid"),
    ("edge_index_coarse", "pos_coarse"),
    ("upsample_idx_fine", "pos_mid"),
    ("upsample_idx_mid", "pos_coarse"),
)

# (level, points, edges, u, theta, tract, u_step, r_local)
_SPLINE_LEVELS = (
    ("fine", "x", "edge_index", "u", "theta", "tract_id", "u_step", "r_local"),
    ("mid", "pos_mid", "edge_index_mid", "u_mid", "theta_mid", "tract_id_mid", "u_step_mid", "r_local_mid"),
    (
        "coarse",
        "pos_coarse",
        "edge_index_coarse",
        "u_coarse",
        "theta_coarse",
        "tract_id_coarse",
        "u_step_coarse",
        "r_local_coarse",
    ),
)

# Checked on the GPU after augmentation.
_DEVICE_CHECK_KEYS = ("x_true", "gt_points", "x", "pos_mid", "pos_coarse", "latent_pos", "cl_dense")

_BIG_COORD = 1.0e4  # mm; a coordinate this large is not a vessel


def _keys(data):
    keys = getattr(data, "keys", None)
    keys = keys() if callable(keys) else keys
    return list(keys)


def scan_sample(data):
    """Return ``(problems, sizes)`` for one cached graph. Never raises."""
    problems = []
    sizes = {}
    try:
        for key in _keys(data):
            val = data[key]
            if torch.is_tensor(val) and val.is_floating_point() and val.numel():
                bad = int((~torch.isfinite(val)).sum())
                if bad:
                    problems.append(f"nonfinite:{key}={bad}/{val.numel()}")
        for ik, pk in _INDEX_PAIRS:
            if ik not in data or pk not in data:
                continue
            idx, pts = data[ik], data[pk]
            if not (torch.is_tensor(idx) and torch.is_tensor(pts)) or idx.numel() == 0:
                continue
            n = int(pts.size(0))
            lo, hi = int(idx.min()), int(idx.max())
            if lo < 0 or hi >= n:
                bad = int(((idx < 0) | (idx >= n)).sum())
                problems.append(f"index:{ik} range [{lo},{hi}] vs {pk}={n} ({bad} bad)")
        for key in ("x", "pos_mid", "pos_coarse", "gt_points", "x_true"):
            val = data[key] if key in data else None
            if torch.is_tensor(val) and val.numel():
                big = float(val.abs().max())
                if big > _BIG_COORD:
                    problems.append(f"huge_coord:{key}={big:.3g}")
        for key in ("r_local", "r_local_mid", "r_local_coarse"):
            val = data[key] if key in data else None
            if torch.is_tensor(val) and val.numel():
                nonpos = int((val <= 0).sum())
                if nonpos:
                    problems.append(f"nonpositive:{key}={nonpos}")
        problems.extend(_scan_spline_pseudo(data))
        for key, name in (
            ("x", "n_fine"), ("pos_mid", "n_mid"), ("pos_coarse", "n_coarse"),
            ("gt_points", "n_gt"), ("x_true", "n_x_true"),
        ):
            if key in data and torch.is_tensor(data[key]):
                sizes[name] = int(data[key].size(0))
        for key, name in (("edge_index", "e_fine"), ("edge_index_mid", "e_mid"), ("edge_index_coarse", "e_coarse")):
            if key in data and torch.is_tensor(data[key]):
                sizes[name] = int(data[key].size(-1))
        for key, name in (("face", "f_fine"), ("gt_faces", "f_gt")):
            if key in data and torch.is_tensor(data[key]):
                sizes[name] = int(data[key].size(-1)) if data[key].size(0) == 3 else int(data[key].size(0))
    except Exception as exc:  # a scan failure is itself worth knowing
        problems.append(f"scan_error:{type(exc).__name__}:{exc}")
    return problems, sizes


def _scan_spline_pseudo(data):
    """Pseudo-coordinates the SplineConv kernels will see, on the CPU."""
    out = []
    try:
        from geometry import intrinsic_spline_pseudo_coords
    except Exception as exc:
        return [f"scan_error:pseudo import {exc}"]
    for level, pk, ek, uk, tk, trk, sk, rk in _SPLINE_LEVELS:
        if any(k not in data for k in (pk, ek, uk, tk, trk, sk, rk)):
            continue
        try:
            pseudo = intrinsic_spline_pseudo_coords(
                data[uk], data[tk], data[trk], data[ek], data[sk], r_local=data[rk], pos=data[pk]
            )
        except Exception as exc:
            out.append(f"pseudo_error:{level}:{type(exc).__name__}:{exc}")
            continue
        if pseudo.numel() == 0:
            continue
        nonfinite = int((~torch.isfinite(pseudo)).sum())
        lo, hi = float(torch.nan_to_num(pseudo, nan=0.0).min()), float(torch.nan_to_num(pseudo, nan=0.0).max())
        if nonfinite:
            out.append(f"pseudo_nonfinite:{level}={nonfinite}")
        if lo < 0.0 or hi > 1.0:
            out.append(f"pseudo_range:{level} [{lo:.4g},{hi:.4g}]")
    return out


class Recorder:
    """Per-rank recorder. One instance per training process."""

    def __init__(self):
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        root = os.environ.get("ANEUX_DIAG_DIR") or os.path.join(os.getcwd(), "diag_out")
        os.makedirs(root, exist_ok=True)
        self.dir = root
        self.log = open(os.path.join(root, f"rank{self.rank}.log"), "a", buffering=1)
        self.stacks = open(os.path.join(root, f"rank{self.rank}.stacks.txt"), "a", buffering=1)
        self.stage = "init"
        self.epoch = -1
        self.step = -1
        self.recent = deque(maxlen=2)
        self.stall_s = float(os.environ.get("ANEUX_DIAG_STALL_S", "1200"))
        self.n_flagged = 0
        self._event(
            "start",
            host=socket.gethostname(),
            pid=os.getpid(),
            torch=torch.__version__,
            cuda_launch_blocking=os.environ.get("CUDA_LAUNCH_BLOCKING"),
            alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        )

    # ------------------------------------------------------------------ log
    def _event(self, kind, **fields):
        row = {"t": round(time.time(), 3), "kind": kind, "rank": self.rank,
               "epoch": self.epoch, "step": self.step, "stage": self.stage}
        row.update(fields)
        try:
            self.log.write(json.dumps(row, default=str) + "\n")
        except Exception:
            pass

    def _arm(self):
        try:
            faulthandler.cancel_dump_traceback_later()
            faulthandler.dump_traceback_later(self.stall_s, repeat=False, file=self.stacks)
        except Exception:
            pass

    # --------------------------------------------------------------- steps
    def begin(self, epoch, step, cpu_batch):
        """Call with the CPU batch, before ``.to(device)``."""
        self.epoch, self.step, self.stage = int(epoch), int(step), "load"
        ids = list(getattr(cpu_batch, "sample_id", None) or [])
        flags = [f for f in (getattr(cpu_batch, "diag_flags", None) or [])]
        sizes = [s for s in (getattr(cpu_batch, "diag_sizes", None) or [])]
        entry = {"epoch": self.epoch, "step": self.step, "ids": ids, "flags": flags,
                 "sizes": sizes, "batch": cpu_batch, "rng_cpu": torch.get_rng_state()}
        try:
            entry["rng_cuda"] = torch.cuda.get_rng_state()
        except Exception:
            entry["rng_cuda"] = None
        self.recent.append(entry)
        mem = {}
        try:
            mem = {"alloc_gib": round(torch.cuda.memory_allocated() / 2**30, 3),
                   "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 3)}
        except Exception:
            pass
        flagged = [f for f in flags if f]
        self._event("step", ids=ids, flags=flagged or None, sizes=sizes, **mem)
        if flagged:
            self.n_flagged += 1
            self._event("WARN_sample_flags", ids=ids, flags=flagged)
        self._arm()

    def mark(self, stage):
        self.stage = stage

    def check_device(self, batch, label):
        """Finite check on the GPU tensors the kernels consume (syncs)."""
        bad = {}
        for key in _DEVICE_CHECK_KEYS:
            val = getattr(batch, key, None)
            if torch.is_tensor(val) and val.is_floating_point() and val.numel():
                n_bad = int((~torch.isfinite(val)).sum().item())
                if n_bad:
                    bad[key] = n_bad
        if bad:
            ids = list(getattr(batch, "sample_id", None) or [])
            self._event("WARN_device_nonfinite", where=label, bad=bad, ids=ids)

    def after_step(self, loss, grad_norm):
        fields = {}
        try:
            fields["loss"] = float(loss.detach())
            if grad_norm is not None:
                fields["grad_norm"] = float(grad_norm)
        except Exception:
            return
        self._event("done", **fields)
        if not all(torch.isfinite(torch.tensor(v)) for v in fields.values()):
            self._event("WARN_nonfinite_loss_or_grad", **fields)

    # --------------------------------------------------------------- crash
    def dump_crash(self, exc, where=""):
        self._event("CRASH", where=where, exc=f"{type(exc).__name__}: {exc}")
        path = os.path.join(self.dir, f"crash_rank{self.rank}.txt")
        lines = [
            f"time      {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"host      {socket.gethostname()}  rank {self.rank}  local_rank {self.local_rank}  pid {os.getpid()}",
            f"where     {where}",
            f"stage     {self.stage}   epoch {self.epoch}   step {self.step}",
            f"exception {type(exc).__name__}: {exc}",
            "",
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            "",
        ]
        for entry in self.recent:
            lines.append(
                f"batch epoch {entry['epoch']} step {entry['step']}  ids={entry['ids']}  "
                f"flags={[f for f in entry['flags'] if f]}  sizes={entry['sizes']}"
            )
        try:
            lines.append(
                f"cuda mem  allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB  "
                f"reserved {torch.cuda.memory_reserved() / 2**30:.2f} GiB"
            )
        except Exception as mem_exc:
            lines.append(f"cuda mem  unavailable ({type(mem_exc).__name__})")
        try:
            smi = subprocess.run(
                ["nvidia-smi", "-i", str(self.local_rank), "--query-gpu=name,ecc.errors.uncorrected.volatile.total,"
                 "ecc.errors.corrected.volatile.total,retired_pages.pending,memory.used,temperature.gpu,clocks_throttle_reasons.active",
                 "--format=csv"],
                capture_output=True, text=True, timeout=30,
            )
            lines.append("nvidia-smi " + (smi.stdout or smi.stderr).strip())
        except Exception as smi_exc:
            lines.append(f"nvidia-smi unavailable ({type(smi_exc).__name__})")
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        try:
            faulthandler.dump_traceback(file=self.stacks, all_threads=True)
        except Exception:
            pass
        for entry in self.recent:
            out = os.path.join(self.dir, f"crash_rank{self.rank}_e{entry['epoch']}_s{entry['step']}.pt")
            try:
                torch.save(entry, out)
            except Exception as save_exc:
                self._event("WARN_batch_save_failed", path=out, exc=str(save_exc))
        try:
            self.log.flush()
        except Exception:
            pass


_REC = None


def recorder():
    """The per-process recorder, or ``None`` when diagnostics are off."""
    global _REC
    if not enabled():
        return None
    if _REC is None:
        _REC = Recorder()
    return _REC


def fatal(exc, where=""):
    """Dump everything, then exit this process immediately (no hang in teardown)."""
    rec = recorder()
    if rec is None:
        return
    if rec is not None:
        try:
            rec.dump_crash(exc, where=where)
        finally:
            try:
                sys.stderr.write(f"[diag] rank {rec.rank} crashed in stage {rec.stage}; wrote {rec.dir}\n")
                sys.stderr.flush()
                sys.stdout.flush()
            except Exception:
                pass
    os._exit(1)
