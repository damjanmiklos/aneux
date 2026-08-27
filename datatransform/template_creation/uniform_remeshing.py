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
from aneux_paths import TEMPLATE_OUTPUT_REMESHED as DEFAULT_OUTPUT_DIR

from vessel_pipeline import add_shared_cli_args, process_uniform_dataset, run_batch


def _process_one(dataset_id, v_file, args):
    process_uniform_dataset(
        dataset_id=dataset_id,
        v_file=v_file,
        output_dir=args.output_dir,
        target_edge_length=args.target_edge_length,
        extension_length=args.extension_length,
        sample_spacing=args.sample_spacing,
        grid_spacing=args.grid_spacing,
        max_grid_size=args.max_grid_size,
    )


def main():
    parser = argparse.ArgumentParser(description="AneuX uniform surface remeshing pipeline")
    add_shared_cli_args(parser, DEFAULT_OUTPUT_DIR, default_workers=2, include_remesh_grid=True)
    args = parser.parse_args()
    extra = [
        "--target-edge-length", str(args.target_edge_length),
        "--extension-length", str(args.extension_length),
        "--sample-spacing", str(args.sample_spacing),
        "--grid-spacing", str(args.grid_spacing),
        "--max-grid-size", str(args.max_grid_size),
    ]
    run_batch(os.path.abspath(__file__), _process_one, args, extra)


if __name__ == "__main__":
    main()
