"""Komondor job staging: copy training data from /project onto /scratch.

Storage (https://docs.hpc.dkf.hu/storage/overview.html):

- `/project/<account>` — persistent HDD Lustre. Git clone and the manually
  scp'd `cleandata/` live here.
- `/scratch/<account>` — NVMe Lustre, fastest tier, quota'd and temporary.
  The job copies cleandata + tube_cache here before training and copies the
  run directory back to project at the end.

Python packages stay in the existing conda env (`aneuxai_env` or
`aneurysmgnn`). Komondor docs warn against installing conda onto Lustre;
we do not copy the env (that would burn RAM in /tmp). We only stage the
large `.vtp` / `.pt` working set.
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import time


def detect_project_account():
    for key in (
        "ANEUX_ACCOUNT",
        "ANEUX_PROJECT_ID",
        "SLURM_JOB_ACCOUNT",
        "SLURM_ACCOUNT",
    ):
        val = os.environ.get(key)
        if val:
            return val.strip()
    cwd = os.path.abspath(os.getcwd())
    for root in ("/project", "/scratch"):
        prefix = root.rstrip("/") + os.sep
        if cwd.startswith(prefix):
            rest = cwd[len(prefix) :].split(os.sep, 1)[0]
            if rest:
                return rest
    if os.path.isdir("/project"):
        try:
            names = sorted(
                n
                for n in os.listdir("/project")
                if not n.startswith(".") and os.path.isdir(os.path.join("/project", n))
            )
        except OSError:
            names = []
        if len(names) == 1:
            return names[0]
    return None


def project_root(account=None):
    explicit = os.environ.get("ANEUX_PROJECT_ROOT")
    if explicit:
        return os.path.abspath(explicit)
    account = account or detect_project_account()
    if account and os.path.isdir(os.path.join("/project", account)):
        return os.path.join("/project", account)
    return None


def scratch_root(account=None):
    explicit = os.environ.get("ANEUX_SCRATCH_ROOT")
    if explicit:
        return os.path.abspath(explicit)
    account = account or detect_project_account()
    if account and os.path.isdir("/scratch"):
        path = os.path.join("/scratch", account)
        return path
    # Local / non-Komondor fallback: never invent a writable path under rawdata.
    return os.environ.get("ANEUX_LOCAL_SCRATCH")


def job_workspace(account=None, job_id=None, user=None):
    """Per-job directory on the fastest filesystem available."""
    job_id = str(job_id or os.environ.get("SLURM_JOB_ID") or os.getpid())
    user = user or os.environ.get("USER") or os.environ.get("USERNAME") or "user"
    scratch = scratch_root(account)
    if scratch:
        path = os.path.join(scratch, user, "aneux", job_id)
        os.makedirs(path, exist_ok=True)
        return os.path.abspath(path)
    raise RuntimeError(
        "No scratch filesystem found. On Komondor this is /scratch/<account>. "
        "Set ANEUX_SCRATCH_ROOT or ANEUX_ACCOUNT."
    )


def _first_positive_int(value, default=None):
    if value is None or value == "":
        return default
    if isinstance(value, (int, float)):
        n = int(value)
        return n if n > 0 else default
    digits = "".join(ch if ch.isdigit() else " " for ch in str(value))
    parts = [p for p in digits.split() if p]
    if not parts:
        return default
    n = int(parts[0])
    return n if n > 0 else default


def slurm_cpu_count():
    for key in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        n = _first_positive_int(os.environ.get(key))
        if n:
            return n
    return os.cpu_count() or 1


def slurm_gpu_count(fallback=1):
    for key in ("SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_NODE", "SLURM_STEP_GPUS"):
        n = _first_positive_int(os.environ.get(key))
        if n:
            return n
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible not in ("-1", "NoDevFiles"):
        n = len([p for p in visible.split(",") if p.strip() != ""])
        if n:
            return n
    return fallback


def scale_hpc_workers(n_gpu=None, n_cpu=None):
    """DataLoader / cache process counts from the Slurm allocation.

    Komondor GPU nodes are 64 cores / 4 A100s, so 1 GPU comes with 16 cores.
    One core per rank is left for the trainer. Cache warmup is rank-0 only and
    uses all but one allocated core (GPU is idle). Env overrides
    ``ANEUX_NUM_WORKERS`` / ``ANEUX_CACHE_WORKERS`` still win at the caller.
    """
    n_gpu = max(1, int(n_gpu if n_gpu is not None else slurm_gpu_count(1)))
    n_cpu = max(1, int(n_cpu if n_cpu is not None else slurm_cpu_count()))
    per_gpu = max(1, n_cpu // n_gpu)
    num_workers = max(0, per_gpu - 1)
    cache_build_workers = max(1, n_cpu - 1)
    return {
        "n_gpu": n_gpu,
        "n_cpu": n_cpu,
        "cpus_per_gpu": per_gpu,
        "num_workers": num_workers,
        "cache_build_workers": cache_build_workers,
    }


def _which(name):
    return shutil.which(name)


def copy_tree(src, dst, description="tree"):
    """Copy `src` → `dst`. Prefer rsync on Linux; shutil elsewhere."""
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    t0 = time.time()
    rsync = _which("rsync")
    if rsync and os.name != "nt":
        os.makedirs(dst, exist_ok=True)
        cmd = [
            rsync,
            "-a",
            "--human-readable",
            "--info=stats2",
            src.rstrip("/") + "/",
            dst.rstrip("/") + "/",
        ]
        print(f"[hpc] rsync {description}: {src} -> {dst}", flush=True)
        subprocess.check_call(cmd)
    else:
        print(f"[hpc] copy {description}: {src} -> {dst}", flush=True)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
    dt = time.time() - t0
    print(f"[hpc] {description} copy finished in {dt:.1f}s", flush=True)
    return dst


def _dir_has_vtp(path):
    """True if ``path`` or a near subdirectory contains a ``.vtp``.

    ``cleandata/`` stores meshes in ``uniformly_remeshed/``, ``template_mesh/``,
    and ``original_centerline/`` — a top-level ``listdir`` misses them.
    """
    if not path or not os.path.isdir(path):
        return False
    try:
        for root, dirnames, files in os.walk(path):
            for name in files:
                if name.lower().endswith(".vtp"):
                    return True
            rel = os.path.relpath(root, path)
            depth = 0 if rel == os.curdir else rel.count(os.sep) + 1
            if depth >= 2:
                dirnames.clear()
    except OSError:
        return False
    return False


def persist_if_remote(src, dest, description="tree"):
    """Copy ``src`` → ``dest`` when they are different existing/creatable paths."""
    if not src or not dest:
        return None
    src = os.path.abspath(src)
    dest = os.path.abspath(dest)
    if src == dest or not os.path.isdir(src):
        return dest if src == dest else None
    return copy_tree(src, dest, description)


def stage_training_inputs(
    *,
    repo_root,
    cleandata_src,
    cache_src,
    workspace,
    copy_cleandata=True,
    copy_cache=True,
):
    """Place cleandata and tube_cache on scratch. Code stays on the git checkout."""
    staged = {
        "workspace": workspace,
        "repo_root": os.path.abspath(repo_root),
        "cleandata": os.path.abspath(cleandata_src),
        "cache": os.path.abspath(cache_src) if cache_src else None,
    }
    if copy_cleandata and _dir_has_vtp(cleandata_src):
        dest = os.path.join(workspace, "cleandata")
        copy_tree(cleandata_src, dest, "cleandata")
        staged["cleandata"] = dest
    elif copy_cleandata:
        print(
            f"[hpc] cleandata at {cleandata_src} has no .vtp files yet; "
            "training will see whatever is there.",
            flush=True,
        )
    if copy_cache and cache_src and os.path.isdir(cache_src):
        dest = os.path.join(workspace, "tube_cache")
        copy_tree(cache_src, dest, "tube_cache")
        staged["cache"] = dest
    else:
        dest = os.path.join(workspace, "tube_cache")
        os.makedirs(dest, exist_ok=True)
        staged["cache"] = dest
    staged["output"] = os.path.join(workspace, "output")
    os.makedirs(staged["output"], exist_ok=True)
    return staged


def sync_run_back(run_dir, dest_root):
    """Copy the timestamped run folder back to persistent project storage."""
    if not run_dir or not os.path.isdir(run_dir):
        print("[hpc] no run_dir to copy back", flush=True)
        return None
    dest_root = os.path.abspath(dest_root)
    os.makedirs(dest_root, exist_ok=True)
    dest = os.path.join(dest_root, os.path.basename(run_dir.rstrip(os.sep)))
    copy_tree(run_dir, dest, "run output -> project")
    return dest


def write_sync_marker(path, payload):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for key, val in payload.items():
            handle.write(f"{key}={val}\n")


def chmod_writable(path):
    try:
        os.chmod(path, os.stat(path).st_mode | stat.S_IWUSR)
    except OSError:
        pass


def python_exe():
    return sys.executable
