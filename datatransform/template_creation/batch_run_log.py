"""Per-case error logs for template_creation batch scripts.

Each run writes ``<output-dir>/<log_folder>/run_<timestamp>/`` with per-case
JSON, ``errors/<id>.txt`` (traceback + stdout), worker transcripts on failure,
and ``summary.csv`` / ``summary.xlsx``.
"""
import argparse
import csv
import json
import os
import sys
import time
import traceback
from contextvars import ContextVar
from datetime import datetime, timezone

LOG_EXCEL_TRACEBACK_CHARS = 30000
CASE_LOG: ContextVar[dict | None] = ContextVar("batch_case_log", default=None)

SUMMARY_FIELDS = [
    "dataset_id",
    "status",
    "step",
    "error_type",
    "error_message",
    "warnings",
    "duration_s",
    "started_at",
    "finished_at",
    "input_file",
    "output_file",
    "returncode",
    "traceback",
]


class StreamTee:
    """Mirror writes to the original stream and keep a text copy."""

    def __init__(self, stream):
        self._stream = stream
        self.parts = []

    def write(self, data):
        if not isinstance(data, str):
            data = data.decode("utf-8", "replace")
        self.parts.append(data)
        self._stream.write(data)
        return len(data)

    def flush(self):
        self._stream.flush()

    def isatty(self):
        return False

    def fileno(self):
        fn = getattr(self._stream, "fileno", None)
        if fn is None:
            raise OSError("fileno")
        return fn()

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", "utf-8")

    def getvalue(self):
        return "".join(self.parts)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def set_step(name):
    rec = CASE_LOG.get()
    if rec is not None:
        rec["step"] = name


def warn(msg):
    text = str(msg)
    print(f"  WARNING: {text}")
    rec = CASE_LOG.get()
    if rec is not None:
        rec.setdefault("warnings", []).append(text)


def harvest_warning_lines(text, warnings, max_lines=50):
    seen = set(warnings)
    extra = 0
    for line in (text or "").splitlines():
        marker = (
            "WARNING" in line
            or "WARN|" in line
            or " ERR|" in line
            or "ERROR:" in line
        )
        if not marker:
            continue
        cleaned = line.strip()
        if not cleaned or cleaned in seen:
            continue
        if len(warnings) >= max_lines:
            extra += 1
            continue
        warnings.append(cleaned)
        seen.add(cleaned)
    if extra:
        warnings.append(f"... {extra} more warning/error lines")
    return warnings


def last_step_from_output(text):
    last = ""
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("Step "):
            last = stripped.split("...", 1)[0].strip()
    return last


def add_run_log_args(parser, log_folder_name):
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help=(
            "Directory for this run's JSON/CSV/Excel error logs "
            f"(default: <output-dir>/{log_folder_name}/run_<timestamp>)."
        ),
    )
    parser.add_argument(
        "--skip-log-merge",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def resolve_log_dir(args, log_folder_name, default_output_dir=None):
    log_dir = getattr(args, "log_dir", None)
    if log_dir:
        return os.path.abspath(log_dir)
    output_dir = getattr(args, "output_dir", None) or default_output_dir
    if not output_dir:
        output_dir = os.getcwd()
    output_dir = os.path.abspath(output_dir)
    if getattr(args, "case", None):
        return os.path.join(output_dir, log_folder_name, "live")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_dir, log_folder_name, f"run_{stamp}")


def configure_batch_logging(args, log_folder_name, default_output_dir=None):
    """Set ``args.log_dir`` and return (extra_cli_flags, on_worker_result)."""
    args.log_dir = resolve_log_dir(args, log_folder_name, default_output_dir)
    os.makedirs(args.log_dir, exist_ok=True)
    print(f"Run log directory: {args.log_dir}")
    extra = ["--log-dir", str(args.log_dir), "--skip-log-merge"]

    def on_worker_result(dataset_id, returncode, output):
        if returncode != 0:
            write_worker_transcript(args.log_dir, dataset_id, returncode, output)

    return extra, on_worker_result


def finalize_run_logs(args):
    if getattr(args, "skip_log_merge", False):
        return None
    log_dir = getattr(args, "log_dir", None)
    if not log_dir:
        return None
    return merge_run_logs(log_dir)


def write_case_log(log_dir, record):
    """Write one case JSON and, on failure, a plain-text traceback file."""
    os.makedirs(log_dir, exist_ok=True)
    dataset_id = str(record["dataset_id"])
    payload = dict(record)
    payload["warnings"] = list(payload.get("warnings") or [])
    stdout = payload.pop("stdout", "") or ""
    dest = os.path.join(log_dir, f"{dataset_id}.json")
    tmp = dest + f".{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
        handle.write("\n")
    os.replace(tmp, dest)

    if payload.get("status") != "success":
        err_dir = os.path.join(log_dir, "errors")
        os.makedirs(err_dir, exist_ok=True)
        err_path = os.path.join(err_dir, f"{dataset_id}.txt")
        warning_block = "\n".join(payload["warnings"]) or "(none)"
        with open(err_path, "w", encoding="utf-8") as handle:
            handle.write(
                f"dataset_id: {dataset_id}\n"
                f"status: {payload.get('status')}\n"
                f"step: {payload.get('step')}\n"
                f"started_at: {payload.get('started_at')}\n"
                f"finished_at: {payload.get('finished_at')}\n"
                f"duration_s: {payload.get('duration_s')}\n"
                f"input_file: {payload.get('input_file')}\n"
                f"error_type: {payload.get('error_type')}\n"
                f"error_message: {payload.get('error_message')}\n"
                f"\n--- warnings ---\n{warning_block}\n"
                f"\n--- traceback ---\n{payload.get('traceback') or '(none)'}\n"
            )
            if stdout.strip():
                handle.write(f"\n--- stdout ---\n{stdout}")
                if not stdout.endswith("\n"):
                    handle.write("\n")
        print(f"  Logged error to {err_path}")
    return dest


def write_worker_transcript(log_dir, dataset_id, returncode, output):
    """Persist captured worker stdout (includes VTK C++ warnings)."""
    output = output or ""
    path = None
    if output:
        trans_dir = os.path.join(log_dir, "transcripts")
        os.makedirs(trans_dir, exist_ok=True)
        path = os.path.join(trans_dir, f"{dataset_id}.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(f"dataset_id: {dataset_id}\nreturncode: {returncode}\n\n")
            handle.write(output)
            if not output.endswith("\n"):
                handle.write("\n")
    json_path = os.path.join(log_dir, f"{dataset_id}.json")
    rec = None
    if os.path.isfile(json_path):
        try:
            with open(json_path, encoding="utf-8") as handle:
                rec = json.load(handle)
        except (OSError, json.JSONDecodeError):
            rec = None
    if rec is None:
        rec = {
            "dataset_id": dataset_id,
            "input_file": "",
            "output_file": "",
            "status": "success" if returncode == 0 else "error",
            "step": last_step_from_output(output),
            "warnings": [],
            "error_type": "" if returncode == 0 else "WorkerExit",
            "error_message": "" if returncode == 0 else f"worker exited with code {returncode}",
            "traceback": "",
            "stdout": output,
            "started_at": "",
            "finished_at": utc_now(),
            "duration_s": "",
        }
    rec["returncode"] = returncode
    if path:
        rec["transcript_file"] = path
    if returncode != 0:
        rec["status"] = "error"
        rec["error_type"] = rec.get("error_type") or "WorkerExit"
        rec["error_message"] = rec.get("error_message") or f"worker exited with code {returncode}"
        if output and len(output) >= len(rec.get("stdout") or ""):
            rec["stdout"] = output
        if not rec.get("step"):
            rec["step"] = last_step_from_output(output)
    rec["warnings"] = harvest_warning_lines(output, list(rec.get("warnings") or []))
    write_case_log(log_dir, rec)
    return path


def _log_rows_from_dir(log_dir):
    rows = []
    if not log_dir or not os.path.isdir(log_dir):
        return rows
    for name in sorted(os.listdir(log_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(log_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                rec = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        warnings = rec.get("warnings") or []
        if isinstance(warnings, str):
            warning_text = warnings
        else:
            warning_text = " | ".join(str(w) for w in warnings)
        traceback_text = str(rec.get("traceback") or "")
        if len(traceback_text) > LOG_EXCEL_TRACEBACK_CHARS:
            traceback_text = traceback_text[:LOG_EXCEL_TRACEBACK_CHARS] + "\n...[truncated]"
        rows.append(
            {
                "dataset_id": rec.get("dataset_id", os.path.splitext(name)[0]),
                "status": rec.get("status", ""),
                "step": rec.get("step", ""),
                "error_type": rec.get("error_type", ""),
                "error_message": rec.get("error_message", ""),
                "warnings": warning_text,
                "duration_s": rec.get("duration_s", ""),
                "started_at": rec.get("started_at", ""),
                "finished_at": rec.get("finished_at", ""),
                "input_file": rec.get("input_file", ""),
                "output_file": rec.get("output_file", ""),
                "returncode": rec.get("returncode", ""),
                "traceback": traceback_text,
            }
        )
    rows.sort(key=lambda r: (0 if r.get("status") == "error" else 1, str(r.get("dataset_id"))))
    return rows


def merge_run_logs(log_dir):
    """Build CSV/Excel/text summaries for every case JSON in ``log_dir``."""
    log_dir = os.path.abspath(log_dir)
    rows = _log_rows_from_dir(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, "summary.csv")
    xlsx_path = os.path.join(log_dir, "summary.xlsx")
    errors_path = os.path.join(log_dir, "errors.txt")
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    n_err = sum(1 for r in rows if r.get("status") == "error")
    n_ok = sum(1 for r in rows if r.get("status") == "success")
    with open(errors_path, "w", encoding="utf-8") as handle:
        handle.write(f"errors={n_err} success={n_ok} total={len(rows)}\n\n")
        for rec in rows:
            if rec.get("status") != "error":
                continue
            handle.write(
                f"{rec['dataset_id']}\n"
                f"  step: {rec.get('step')}\n"
                f"  {rec.get('error_type')}: {rec.get('error_message')}\n"
                f"  detail: {os.path.join(log_dir, 'errors', rec['dataset_id'] + '.txt')}\n\n"
            )

    try:
        import pandas as pd

        pd.DataFrame(rows).to_excel(xlsx_path, index=False)
        print(f"Saved Excel log to: {xlsx_path}")
    except Exception as exc:
        print(f"Failed to write Excel log ({xlsx_path}): {exc}")
        xlsx_path = None

    print(
        f"Run log: {n_ok} success, {n_err} error, {len(rows)} recorded. "
        f"See {log_dir}"
    )
    return {
        "log_dir": log_dir,
        "csv": csv_path,
        "xlsx": xlsx_path,
        "errors": errors_path,
        "n_success": n_ok,
        "n_error": n_err,
        "n_total": len(rows),
    }


def run_logged_case(dataset_id, v_file, args, work, log_folder_name, default_output_dir=None):
    """Run ``work()`` for one case, writing success/error logs under ``log_dir``."""
    log_dir = resolve_log_dir(args, log_folder_name, default_output_dir)
    rec = {
        "dataset_id": dataset_id,
        "input_file": v_file,
        "output_file": "",
        "status": "running",
        "step": "start",
        "warnings": [],
        "error_type": "",
        "error_message": "",
        "traceback": "",
        "stdout": "",
        "started_at": utc_now(),
        "finished_at": "",
        "duration_s": "",
    }
    token = CASE_LOG.set(rec)
    t0 = time.perf_counter()
    old_out, old_err = sys.stdout, sys.stderr
    tee_out, tee_err = StreamTee(old_out), StreamTee(old_err)
    sys.stdout, sys.stderr = tee_out, tee_err
    try:
        out_file = work()
        rec["status"] = "success"
        rec["output_file"] = out_file or ""
        return out_file
    except Exception as exc:
        rec["status"] = "error"
        cause = exc.__cause__ if exc.__cause__ is not None else exc
        rec["error_type"] = type(cause).__name__
        rec["error_message"] = str(exc)
        rec["traceback"] = traceback.format_exc()
        raise
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        captured = tee_out.getvalue() + tee_err.getvalue()
        rec["warnings"] = harvest_warning_lines(captured, list(rec.get("warnings") or []))
        if rec.get("status") != "success":
            rec["stdout"] = captured
        if rec.get("step") in ("", "start"):
            rec["step"] = last_step_from_output(captured) or rec["step"]
        rec["finished_at"] = utc_now()
        rec["duration_s"] = round(time.perf_counter() - t0, 3)
        try:
            write_case_log(log_dir, rec)
        except Exception as log_exc:
            print(f"  WARNING: failed to write case log for {dataset_id}: {log_exc}")
        CASE_LOG.reset(token)
