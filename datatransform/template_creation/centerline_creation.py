"""AneuX centerline extraction from remeshed vessel surfaces.

Writes ``original_centerline`` only. ``template_centerline`` is dropped
(§15 item 3): it matched ``original_centerline`` to within 0.12 mm, and
training parametrises from the original.

Each run writes debug logs under ``<output-dir>/centerline_logs/run_<timestamp>/``:
per-case JSON, ``errors/<id>.txt`` with traceback, worker transcripts, and
``summary.csv`` / ``summary.xlsx``.
"""
import os
import sys

os.environ["VTK_OFFSCREEN"] = "1"
os.environ["EGL_PLATFORM"] = "surfaceless"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VTK_NUMBER_OF_THREADS"] = "1"

import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from aneux_paths import (
    CLEAN_CENTERLINE as DEFAULT_OUTPUT_DIR,
    CLEAN_UNIFORM_MESH,
)

from batch_run_log import (
    add_run_log_args,
    configure_batch_logging,
    finalize_run_logs,
    run_logged_case,
)
from process_case import is_template_centerline_dir
from vessel_pipeline import (
    add_shared_cli_args,
    process_centerline_dataset as _vessel_process_centerline_dataset,
    run_batch,
)

LOG_FOLDER = "centerline_logs"


def process_centerline_dataset(dataset_id, v_file, output_dir, **kwargs):
    """Write one original_centerline ``.vtp``. Never write ``template_centerline``."""
    if is_template_centerline_dir(output_dir):
        print(
            f"Skipping {dataset_id}: template_centerline is dropped "
            "(§15 item 3); parametrise from original_centerline."
        )
        return None
    allowed = {}
    for key in ("extension_length", "sample_spacing"):
        if key in kwargs:
            allowed[key] = kwargs[key]
    return _vessel_process_centerline_dataset(
        dataset_id=dataset_id,
        v_file=v_file,
        output_dir=output_dir,
        **allowed,
    )


def _process_one(dataset_id, v_file, args):
    def work():
        return process_centerline_dataset(
            dataset_id=dataset_id,
            v_file=v_file,
            output_dir=args.output_dir,
            extension_length=args.extension_length,
            sample_spacing=args.sample_spacing,
        )

    return run_logged_case(
        dataset_id,
        v_file,
        args,
        work,
        log_folder_name=LOG_FOLDER,
        default_output_dir=DEFAULT_OUTPUT_DIR,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="AneuX centerline extraction pipeline")
    add_shared_cli_args(parser, DEFAULT_OUTPUT_DIR, default_workers=25, include_remesh_grid=False)
    add_run_log_args(parser, LOG_FOLDER)
    parser.set_defaults(vessel_dir=CLEAN_UNIFORM_MESH, from_folder=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    extra_log, on_worker_result = configure_batch_logging(
        args, LOG_FOLDER, DEFAULT_OUTPUT_DIR
    )
    extra = [
        "--extension-length",
        str(args.extension_length),
        "--sample-spacing",
        str(args.sample_spacing),
        "--vessel-dir",
        str(args.vessel_dir),
    ] + extra_log
    try:
        run_batch(
            os.path.abspath(__file__),
            _process_one,
            args,
            extra,
            on_worker_result=on_worker_result,
        )
    finally:
        finalize_run_logs(args)


if __name__ == "__main__":
    main()
