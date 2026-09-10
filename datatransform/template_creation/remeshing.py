#!/usr/bin/env python3
"""Ground-truth uniform remesh of the original vessel (aneurysm kept).

hemoMesh remeshes for CFD: area-based elements, stronger smoothing, flow
extensions left on. ``uniform_remeshing.py`` remeshes a MISR polyball parent
tube, so the sac is not in the output.

This script remeshes the *original* raw surface for AI training:
  - constant target edge length (not triangle area)
  - as little Taubin as will still let VMTK remesh robustly
  - pipe-section ostia perpendicular to the centerline tangent
    (same cutter as aneurysm-removal / template uncap)
  - never writes into ``rawdata/``

Centerlines still use a sanitised working copy (originals have zero-length
edges that break Voronoi). That copy is discarded; the remeshed surface is
the cleaned original.
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

import numpy as np
import pyvista as pv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from aneux_paths import (
    CLEANDATA_UNIFORM as DEFAULT_OUTPUT_DIR,
    VESSELS_ORIGINAL,
    ensure_cleandata_layout,
)

from vessel_pipeline import (
    TemplateQualityError,
    add_flow_extensions,
    add_shared_cli_args,
    apply_taubin_smoothing,
    assert_template_quality,
    clean_triangulate,
    clip_flow_extensions_and_uncap,
    drop_degenerate_triangles,
    drop_tiny_islands,
    extract_boundary_loops,
    extract_branches,
    extract_centerlines_for_tube,
    finalize_surface,
    inspect_openings,
    log_profiles,
    measure_open_profiles,
    opening_clip_frames,
    remesh_surface_isotropically,
    repair_nonmanifold_triangles,
    resample_centerline,
    run_batch,
    sanitize_vessel_for_vmtk,
    save_polydata,
    seed_points_from_profiles,
    smooth_centerline_preserve_misr,
    uniform_edge_length_for_profiles,
    with_dataset_id,
)

# Edge-length remesh near AneuX area-001 (~0.13 mm median edge from 0.01 mm^2 cells).
DEFAULT_GT_EDGE_LENGTH_MM = 0.15
# VMTK default remesh loop is 10; 20 extra split/collapse/relax passes at bends.
GT_REMESH_N_ITER = 20
GT_REMESH_CONNECTIVITY_ITER = 20
# vtkWindowedSinc PassBand is [0, 2]: 0 = strongest, 2 = none, 0.1 = VTK default.
# 1.5 is light (weaker than VMTK's 1.0); 0.9 would still be strong smoothing.
GT_TAUBIN_PASS_BAND = 1.5
GT_TAUBIN_ITER = 5
# Losing the sac drops ~9–16% area on the polyball path; fail before that.
GT_MIN_AREA_RATIO = 0.88
GT_MAX_AREA_RATIO = 1.20
OPENING_PLANARITY_STD_MM = 0.25


def prepare_gt_surface(vessel_mesh):
    """Keep original tessellation density; only drop degenerates and flaps."""
    poly = clean_triangulate(vessel_mesh)
    poly = drop_degenerate_triangles(poly)
    poly, n_nm = repair_nonmanifold_triangles(poly)
    if n_nm > 0:
        print(f"  WARNING: {n_nm} non-manifold edges remain on the original after repair")
    return poly


def assert_gt_remesh_scale(surface, reference_mesh, context="gt-remesh"):
    """GT remesh must stay near the original area (aneurysm still present)."""
    out_area = float(pv.wrap(surface).area)
    ref_area = float(pv.wrap(reference_mesh).area)
    if ref_area <= 1e-6:
        return out_area, ref_area
    ratio = out_area / ref_area
    print(f"  GT area={out_area:.1f} mm^2 vs original {ref_area:.1f} mm^2 (ratio {ratio:.2f})")
    if ratio < GT_MIN_AREA_RATIO:
        raise TemplateQualityError(
            f"{context} remesh area is {ratio:.2f}x the original "
            f"({out_area:.1f} vs {ref_area:.1f} mm^2); aneurysm or a branch was lost."
        )
    if ratio > GT_MAX_AREA_RATIO:
        raise TemplateQualityError(
            f"{context} remesh area is {ratio:.2f}x the original "
            f"({out_area:.1f} vs {ref_area:.1f} mm^2); flow extensions were not clipped off."
        )
    return out_area, ref_area


def log_opening_planarity(surface, frames):
    """Report how planar each ostium is along the centerline tangent."""
    loops = extract_boundary_loops(surface)
    rims = []
    for i in range(loops.GetNumberOfCells()):
        cell = loops.GetCell(i)
        n = cell.GetNumberOfPoints()
        if n == 0:
            continue
        pts = np.array([cell.GetPoints().GetPoint(j) for j in range(n)], dtype=np.float64)
        rims.append(pts)
    if not rims or not frames:
        return
    for origin, outward, radius in frames:
        origin = np.asarray(origin, dtype=np.float64)
        outward = np.asarray(outward, dtype=np.float64)
        rim = min(rims, key=lambda pts: float(np.linalg.norm(pts.mean(axis=0) - origin)))
        axial = (rim - origin) @ outward
        axial_std = float(np.std(axial))
        flag = "" if axial_std <= OPENING_PLANARITY_STD_MM else "  (not planar)"
        print(
            f"  Ostium planarity: origin {np.round(origin, 2)} "
            f"r={radius:.3f} mm axial_std={axial_std:.3f} mm{flag}"
        )


def _gt_centerline(work_vessel, extension_length, sample_spacing):
    """Sanitised + strongly smoothed copy is only used to trace the lumen."""
    print("Step 2: Working copy for centerlines (sanitise + strong Taubin, discarded later)...")
    work = sanitize_vessel_for_vmtk(work_vessel)
    work = apply_taubin_smoothing(work)
    anatomical_profiles = measure_open_profiles(work)
    log_profiles(anatomical_profiles, label="Anatomical")
    seed_points_from_profiles(anatomical_profiles)

    print("Step 3: Flow extensions on the working copy, then Voronoi centerline...")
    extended_work = add_flow_extensions(work, extension_length=extension_length)
    extended_profiles = measure_open_profiles(extended_work)
    log_profiles(extended_profiles, label="Extended")
    if len(extended_profiles) != len(anatomical_profiles):
        print(
            f"  WARNING: opening count changed after extensions "
            f"({len(anatomical_profiles)} -> {len(extended_profiles)})."
        )
    centerline = extract_centerlines_for_tube(
        extended_work, work, anatomical_profiles, extended_profiles
    )
    resampled = resample_centerline(centerline, sample_spacing=sample_spacing)
    smooth_cl = smooth_centerline_preserve_misr(resampled)
    branched = extract_branches(smooth_cl)
    return work, anatomical_profiles, branched


@with_dataset_id
def process_gt_remesh_dataset(
    dataset_id,
    v_file,
    output_dir,
    target_edge_length=DEFAULT_GT_EDGE_LENGTH_MM,
    extension_length=5.0,
    sample_spacing=0.1,
):
    print(f"\n=========================================\nProcessing GT remesh: {dataset_id}")
    ensure_cleandata_layout()
    original = pv.read(v_file)

    print("Step 1: Preparing original surface (keep detail, drop degenerates)...")
    gt_surface = prepare_gt_surface(original)
    print(f"  GT working surface: {gt_surface.GetNumberOfPoints()} points")
    print("Step 1b: Anatomical openings on the detailed original...")
    gt_profiles = measure_open_profiles(gt_surface)
    log_profiles(gt_profiles, label="GT anatomical")
    seed_points_from_profiles(gt_profiles)

    _work, work_profiles, branched = _gt_centerline(
        gt_surface, extension_length, sample_spacing
    )
    if len(gt_profiles) != len(work_profiles):
        print(
            f"  WARNING: opening count differs on GT vs sanitised working copy "
            f"({len(gt_profiles)} vs {len(work_profiles)}). Clipping uses GT loops."
        )

    print("Step 4: Flow extensions on the detailed original, then pipe-section ostia...")
    extended_gt = add_flow_extensions(gt_surface, extension_length=extension_length)
    opened_gt, n_clipped = clip_flow_extensions_and_uncap(
        extended_gt,
        gt_profiles,
        extension_length=extension_length,
        centerline=branched,
    )
    n_in = len(gt_profiles)
    if n_clipped < n_in:
        print(
            f"  WARNING: pipe-section uncap opened {n_clipped}/{n_in} ends; "
            "remaining ostia keep the original rim."
        )
    if len(inspect_openings(opened_gt)) < 2:
        raise TemplateQualityError(
            f"GT remesh has fewer than 2 openings after uncap ({n_clipped}/{n_in} clipped)."
        )
    opened_gt, _n_regions_pre = drop_tiny_islands(opened_gt)
    frames = opening_clip_frames(branched, gt_profiles)
    log_opening_planarity(opened_gt, frames)

    print(
        f"Step 5: Light Taubin (pass_band={GT_TAUBIN_PASS_BAND}, "
        f"n_iter={GT_TAUBIN_ITER}, boundary off)..."
    )
    opened_gt = apply_taubin_smoothing(
        opened_gt,
        pass_band=GT_TAUBIN_PASS_BAND,
        n_iter=GT_TAUBIN_ITER,
        boundary_smoothing=False,
    )

    effective_edge = uniform_edge_length_for_profiles(gt_profiles, target_edge_length)
    r_min = min(p["radius"] for p in gt_profiles)
    print(
        f"Step 6: Isotropic remesh ElementSizeMode=edgelength "
        f"(TargetEdgeLength={effective_edge:.3f} mm, n_iter={GT_REMESH_N_ITER}, "
        f"R_min={r_min:.3f} mm, PreserveBoundaryEdges=1)..."
    )
    remeshed = remesh_surface_isotropically(
        opened_gt,
        target_edge_length=effective_edge,
        n_iter=GT_REMESH_N_ITER,
        connectivity_iter=GT_REMESH_CONNECTIVITY_ITER,
    )
    print(f"  -> Remeshed surface points: {remeshed.GetNumberOfPoints()}")

    final_surface, _n_regions = finalize_surface(remeshed)
    assert_gt_remesh_scale(final_surface, original, context=dataset_id)
    openings = assert_template_quality(
        final_surface, n_expected_openings=n_in, context=dataset_id
    )
    log_opening_planarity(final_surface, frames)

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{dataset_id}.vtp")
    save_polydata(final_surface, out_file)
    read_back = pv.read(out_file)
    print(f"Successfully saved GT remesh to: {out_file}")
    print(
        f"  -> Verified Saved Mesh: {read_back.n_points} points, {read_back.n_cells} cells, "
        f"disk size={os.path.getsize(out_file)} bytes"
    )
    print(
        f"  -> Verified Open Boundaries Count: {len(openings)} "
        f"(anatomical profiles {n_in}, pipe-section clipped {n_clipped})"
    )
    return out_file


def _process_one(dataset_id, v_file, args):
    process_gt_remesh_dataset(
        dataset_id=dataset_id,
        v_file=v_file,
        output_dir=args.output_dir,
        target_edge_length=args.target_edge_length,
        extension_length=args.extension_length,
        sample_spacing=args.sample_spacing,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Remesh original AneuX vessels to high-detail ground-truth surfaces "
            "(constant edge length, perpendicular ostia). Does not build a parent tube."
        )
    )
    add_shared_cli_args(
        parser,
        DEFAULT_OUTPUT_DIR,
        default_workers=2,
        include_remesh_grid=False,
    )
    parser.add_argument(
        "--target-edge-length",
        type=float,
        default=DEFAULT_GT_EDGE_LENGTH_MM,
        help="Uniform target edge length in mm (default 0.15).",
    )
    parser.set_defaults(vessel_dir=VESSELS_ORIGINAL)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    extra = [
        "--target-edge-length",
        str(args.target_edge_length),
        "--extension-length",
        str(args.extension_length),
        "--sample-spacing",
        str(args.sample_spacing),
        "--vessel-dir",
        str(args.vessel_dir),
    ]
    run_batch(os.path.abspath(__file__), _process_one, args, extra)


if __name__ == "__main__":
    main()
