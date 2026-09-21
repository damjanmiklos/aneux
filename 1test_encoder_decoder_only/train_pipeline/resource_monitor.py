"""Periodic CPU / GPU / IO snapshots during training.

Rank 0 writes ``data/resource_snapshots.jsonl`` plus a CSV and PNG plots.
Interval defaults to 15 s (``ANEUX_MONITOR_SEC``). GPU rows come from
``nvidia-smi`` (all visible devices). IO is the sum of ``/proc/<pid>/io``
over this Slurm job's processes, or the trainer process tree on a PC.
"""
from __future__ import annotations

import csv
import json
import os
import threading
import time
from datetime import datetime, timezone


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _read_proc_io(pid):
    path = os.path.join("/proc", str(pid), "io")
    out = {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if ":" not in line:
                    continue
                key, val = line.split(":", 1)
                out[key.strip()] = int(val.strip())
    except (OSError, ValueError):
        return None
    return out


def _sum_io(pids):
    keys = ("rchar", "wchar", "read_bytes", "write_bytes")
    totals = {k: 0 for k in keys}
    n = 0
    for pid in pids:
        rec = _read_proc_io(pid)
        if not rec:
            continue
        n += 1
        for key in keys:
            totals[key] += int(rec.get(key, 0))
    totals["n_pids"] = n
    return totals


def _descendant_pids(root):
    found = []
    stack = [int(root)]
    seen = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        found.append(pid)
        task_dir = os.path.join("/proc", str(pid), "task")
        try:
            tids = os.listdir(task_dir)
        except OSError:
            continue
        for tid in tids:
            child_path = os.path.join(task_dir, tid, "children")
            try:
                with open(child_path, "r", encoding="utf-8") as handle:
                    stack.extend(int(x) for x in handle.read().split())
            except (OSError, ValueError):
                pass
    return found


def _job_pids(job_id):
    """All PIDs whose environ contains this SLURM_JOB_ID."""
    if not job_id or not os.path.isdir("/proc"):
        return _descendant_pids(os.getpid())
    needle = f"SLURM_JOB_ID={job_id}".encode("ascii")
    pids = []
    try:
        names = os.listdir("/proc")
    except OSError:
        return _descendant_pids(os.getpid())
    for name in names:
        if not name.isdigit():
            continue
        env_path = os.path.join("/proc", name, "environ")
        try:
            with open(env_path, "rb") as handle:
                blob = handle.read()
        except OSError:
            continue
        if needle in blob:
            pids.append(int(name))
    return pids or _descendant_pids(os.getpid())


def _cpu_times_windows():
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class FILETIME(ctypes.Structure):
        _fields_ = [
            ("dwLowDateTime", wintypes.DWORD),
            ("dwHighDateTime", wintypes.DWORD),
        ]

    idle = FILETIME()
    kernel = FILETIME()
    user = FILETIME()
    if not ctypes.windll.kernel32.GetSystemTimes(
        ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
    ):
        return None

    def _quad(ft):
        return (int(ft.dwHighDateTime) << 32) | int(ft.dwLowDateTime)

    idle_t = _quad(idle)
    total = _quad(kernel) + _quad(user)
    return total, idle_t


def _cpu_times():
    path = "/proc/stat"
    try:
        with open(path, "r", encoding="utf-8") as handle:
            parts = handle.readline().split()
        nums = [int(x) for x in parts[1:8]]
        idle = nums[3] + (nums[4] if len(nums) > 4 else 0)
        total = sum(nums)
        return total, idle
    except (OSError, ValueError, IndexError):
        pass
    return _cpu_times_windows()


def _meminfo_windows():
    if os.name != "nt":
        return {}
    import ctypes
    from ctypes import wintypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_uint64),
            ("ullAvailPhys", ctypes.c_uint64),
            ("ullTotalPageFile", ctypes.c_uint64),
            ("ullAvailPageFile", ctypes.c_uint64),
            ("ullTotalVirtual", ctypes.c_uint64),
            ("ullAvailVirtual", ctypes.c_uint64),
            ("ullAvailExtendedVirtual", ctypes.c_uint64),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return {}
    total = float(status.ullTotalPhys) / 1024**3
    avail = float(status.ullAvailPhys) / 1024**3
    return {
        "mem_total_gib": round(total, 3),
        "mem_available_gib": round(avail, 3),
        "mem_used_gib": round(total - avail, 3),
    }


def _meminfo_gib():
    path = "/proc/meminfo"
    info = {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if ":" not in line:
                    continue
                key, val = line.split(":", 1)
                bits = val.split()
                if not bits:
                    continue
                try:
                    kib = float(bits[0])
                except ValueError:
                    continue
                info[key] = kib / (1024.0 * 1024.0)
    except OSError:
        info = {}
    if info:
        return {
            "mem_total_gib": round(info.get("MemTotal", 0.0), 3),
            "mem_available_gib": round(info.get("MemAvailable", 0.0), 3),
            "mem_used_gib": round(
                info.get("MemTotal", 0.0) - info.get("MemAvailable", 0.0), 3
            ),
        }
    return _meminfo_windows()


def _loadavg():
    getter = getattr(os, "getloadavg", None)
    if getter is None:
        return {}
    try:
        a, b, c = getter()
        return {"load1": round(a, 3), "load5": round(b, 3), "load15": round(c, 3)}
    except OSError:
        return {}


def _nvidia_gpus():
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu",
        "--format=csv,nounits,noheader",
    ]
    try:
        raw = subprocess_output(cmd, timeout=8)
    except Exception:
        return []
    gpus = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 8:
            continue
        def _num(text):
            try:
                return float(text)
            except ValueError:
                return None

        gpus.append(
            {
                "index": int(float(parts[0])) if parts[0].replace(".", "", 1).isdigit() else parts[0],
                "name": parts[1],
                "util_gpu": _num(parts[2]),
                "util_mem": _num(parts[3]),
                "mem_used_mib": _num(parts[4]),
                "mem_total_mib": _num(parts[5]),
                "power_w": _num(parts[6]),
                "temp_c": _num(parts[7]),
            }
        )
    return gpus


def subprocess_output(cmd, timeout=8):
    import subprocess

    return subprocess.check_output(cmd, text=True, timeout=timeout, stderr=subprocess.DEVNULL)


def _disk_gib(path):
    if not path:
        return {}
    try:
        usage = os.statvfs(path) if hasattr(os, "statvfs") else None
    except OSError:
        usage = None
    if usage is not None:
        total = usage.f_frsize * usage.f_blocks
        free = usage.f_frsize * usage.f_bavail
        return {
            "disk_total_gib": round(total / 1024**3, 3),
            "disk_free_gib": round(free / 1024**3, 3),
            "disk_used_gib": round((total - free) / 1024**3, 3),
        }
    try:
        import shutil

        du = shutil.disk_usage(path)
        return {
            "disk_total_gib": round(du.total / 1024**3, 3),
            "disk_free_gib": round(du.free / 1024**3, 3),
            "disk_used_gib": round(du.used / 1024**3, 3),
        }
    except OSError:
        return {}


class ResourceMonitor:
    def __init__(self, data_dir, interval=15.0, disk_path=None):
        self.data_dir = os.path.abspath(data_dir)
        self.interval = max(5.0, float(interval))
        self.disk_path = disk_path or self.data_dir
        self.jsonl_path = os.path.join(self.data_dir, "resource_snapshots.jsonl")
        self.csv_path = os.path.join(self.data_dir, "resource_snapshots.csv")
        self._stop = threading.Event()
        self._thread = None
        self._stopped = False
        self._lock = threading.Lock()
        self._rows = []
        self._t0 = time.time()
        self._prev_cpu = None
        self._prev_io = None
        self._prev_t = None
        os.makedirs(self.data_dir, exist_ok=True)

    def start(self):
        self.take(write=True)
        self._thread = threading.Thread(
            target=self._loop, name="aneux-resource-monitor", daemon=True
        )
        self._thread.start()
        import atexit

        atexit.register(self.stop)
        print(
            f"Resource monitor every {self.interval:.0f}s -> {self.jsonl_path}",
            flush=True,
        )
        return self

    def stop(self):
        if self._stopped:
            return self.jsonl_path
        self._stopped = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=min(10.0, self.interval + 2))
            self._thread = None
        try:
            self.take(write=True)
        except Exception:
            pass
        try:
            self.write_csv()
        except Exception:
            pass
        try:
            self.plot()
        except Exception:
            pass
        return self.jsonl_path

    def _loop(self):
        while not self._stop.wait(self.interval):
            try:
                self.take(write=True)
            except Exception:
                continue

    def take(self, write=True):
        now = time.time()
        elapsed = now - self._t0
        gpus = _nvidia_gpus()
        io = _sum_io(_job_pids(os.environ.get("SLURM_JOB_ID")))
        cpu = _cpu_times()
        row = {
            "t_utc": _utc(),
            "elapsed_s": round(elapsed, 3),
            "pid": os.getpid(),
            "n_gpu": len(gpus),
        }
        row.update(_loadavg())
        row.update(_meminfo_gib())
        row.update(_disk_gib(self.disk_path))
        nproc = os.cpu_count() or 1
        if row.get("load1") is not None:
            row["cpu_load_pct"] = round(100.0 * float(row["load1"]) / float(nproc), 2)

        if cpu is not None:
            total, idle = cpu
            if self._prev_cpu is not None:
                dtot = total - self._prev_cpu[0]
                didle = idle - self._prev_cpu[1]
                if dtot > 0:
                    row["cpu_util_pct"] = round(100.0 * (1.0 - didle / dtot), 2)
            self._prev_cpu = (total, idle)

        for key, val in io.items():
            row[f"io_{key}"] = val
        if self._prev_io is not None and self._prev_t is not None:
            dt = max(1e-6, now - self._prev_t)
            for key in ("read_bytes", "write_bytes", "rchar", "wchar"):
                prev = self._prev_io.get(key, 0)
                cur = io.get(key, 0)
                row[f"io_{key}_mib_s"] = round((cur - prev) / dt / (1024.0 * 1024.0), 3)
        self._prev_io = io
        self._prev_t = now

        utils = []
        mems = []
        for gpu in gpus:
            idx = gpu.get("index", 0)
            row[f"gpu{idx}_util"] = gpu.get("util_gpu")
            row[f"gpu{idx}_mem_util"] = gpu.get("util_mem")
            row[f"gpu{idx}_mem_used_mib"] = gpu.get("mem_used_mib")
            row[f"gpu{idx}_power_w"] = gpu.get("power_w")
            row[f"gpu{idx}_temp_c"] = gpu.get("temp_c")
            if gpu.get("util_gpu") is not None:
                utils.append(float(gpu["util_gpu"]))
            if gpu.get("mem_used_mib") is not None:
                mems.append(float(gpu["mem_used_mib"]))
        if utils:
            row["gpu_util_mean"] = round(sum(utils) / len(utils), 2)
            row["gpu_util_min"] = round(min(utils), 2)
            row["gpu_util_max"] = round(max(utils), 2)
        if mems:
            row["gpu_mem_used_mib_sum"] = round(sum(mems), 1)
        row["gpus"] = gpus

        with self._lock:
            self._rows.append(row)
            if write:
                with open(self.jsonl_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, default=str) + "\n")
        return row

    def write_csv(self):
        with self._lock:
            rows = list(self._rows)
        if not rows:
            return None
        skip = {"gpus"}
        fields = []
        for row in rows:
            for key in row:
                if key not in skip and key not in fields:
                    fields.append(key)
        with open(self.csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fields})
        return self.csv_path

    def plot(self):
        with self._lock:
            rows = list(self._rows)
        if len(rows) < 2:
            return []
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return []

        plot_dir = os.path.join(os.path.dirname(self.data_dir), "plots")
        os.makedirs(plot_dir, exist_ok=True)
        xs = [r["elapsed_s"] / 60.0 for r in rows]
        written = []

        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        ax = axes[0]
        if any(r.get("gpu_util_mean") is not None for r in rows):
            ax.plot(xs, [r.get("gpu_util_mean") for r in rows], label="GPU util mean %")
        n_gpu = max((int(r.get("n_gpu") or 0) for r in rows), default=0)
        for idx in range(n_gpu):
            ys = [r.get(f"gpu{idx}_util") for r in rows]
            if any(v is not None for v in ys):
                ax.plot(xs, ys, alpha=0.5, label=f"GPU {idx}")
        ax.set_ylabel("GPU util %")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        ax = axes[1]
        if any(r.get("cpu_util_pct") is not None for r in rows):
            ax.plot(xs, [r.get("cpu_util_pct") for r in rows], label="CPU util %")
        if any(r.get("cpu_load_pct") is not None for r in rows):
            ax.plot(xs, [r.get("cpu_load_pct") for r in rows], label="load1 / ncpu %")
        ax.set_ylabel("CPU %")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        ax = axes[2]
        if any(r.get("io_read_bytes_mib_s") is not None for r in rows):
            ax.plot(xs, [r.get("io_read_bytes_mib_s") for r in rows], label="read MiB/s")
        if any(r.get("io_write_bytes_mib_s") is not None for r in rows):
            ax.plot(xs, [r.get("io_write_bytes_mib_s") for r in rows], label="write MiB/s")
        ax.set_ylabel("IO MiB/s")
        ax.set_xlabel("minutes")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        dest = os.path.join(plot_dir, "resource_util.png")
        fig.savefig(dest, dpi=120)
        plt.close(fig)
        written.append(dest)

        fig, ax = plt.subplots(figsize=(10, 4))
        if any(r.get("gpu_mem_used_mib_sum") is not None for r in rows):
            ax.plot(
                xs,
                [r.get("gpu_mem_used_mib_sum") for r in rows],
                label="GPU mem used (sum MiB)",
            )
        for idx in range(n_gpu):
            ys = [r.get(f"gpu{idx}_mem_used_mib") for r in rows]
            if any(v is not None for v in ys):
                ax.plot(xs, ys, alpha=0.5, label=f"GPU {idx} MiB")
        ax.set_ylabel("MiB")
        ax.set_xlabel("minutes")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        dest = os.path.join(plot_dir, "resource_gpu_mem.png")
        fig.savefig(dest, dpi=120)
        plt.close(fig)
        written.append(dest)
        return written
