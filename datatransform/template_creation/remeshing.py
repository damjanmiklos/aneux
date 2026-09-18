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

Each run writes debug logs under ``<output-dir>/gt_remesh_logs/run_<timestamp>/``:
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

import numpy as np
import pyvista as pv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from aneux_paths import (
    CLEAN_UNIFORM_MESH as DEFAULT_OUTPUT_DIR,
    TOTAL_CLEAN_ORIGINAL_MESH,
    ensure_cleandata_layout,
)

from batch_run_log import (
    add_run_log_args,
    configure_batch_logging,
    finalize_run_logs,
    merge_run_logs,
    record,
    run_logged_case,
    set_step,
    warn,
    write_case_log,
    write_worker_transcript,  # re-exported for tests
)
from vessel_pipeline import (
    TemplateQualityError,
    add_flow_extensions,
    weld_degenerate_vertices,
    patch_wall_pinholes,
    force_manifold_triangles,
    uncap_closed_surface,
    add_shared_cli_args,
    apply_taubin_smoothing,
    assert_template_quality,
    cap_unmatched_loops,
    clean_triangulate,
    clip_flow_extensions_and_uncap,
    drop_boundary_ear_triangles,
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
    remesh_surface_verified,
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
# 20/20 was meant to give extra split/collapse/relax passes at bends. It does the
# opposite: VMTK's vertex relocation does not converge on these surfaces, it
# oscillates. On p129 the area grows 5.45x while the cell count rises only 1.6x
# and the mean edge *grows* to 0.2593 mm -- a crumpling surface, not a finer one.
# Sweeping the same input: 20/20 -> 5.452x area, CV 2.3181; 10/10 -> 1.372x,
# CV 1.2631; 6/10 -> 1.024x, CV 0.3952 and mean edge 0.1253 mm, matching the
# dataset-wide healthy 0.1254 mm. Fewer iterations is better on every objective
# -- uniform edge length, texture kept, and area preserved -- and it is also
# ~3x faster, which matters because this step is 68-96% of a case's runtime.
# These now match REMESH_N_ITER / REMESH_CONNECTIVITY_ITER used everywhere else.
GT_REMESH_N_ITER = 6
GT_REMESH_CONNECTIVITY_ITER = 10
# vtkWindowedSinc PassBand is [0, 2]: 0 = strongest, 2 = none, 0.1 = VTK default.
# 1.5 is light (weaker than VMTK's 1.0); 0.9 would still be strong smoothing.
GT_TAUBIN_PASS_BAND = 1.5
GT_TAUBIN_ITER = 5
# Losing the sac drops ~9–16% area on the polyball path; fail before that.
GT_MIN_AREA_RATIO = 0.88
GT_MAX_AREA_RATIO = 1.20
OPENING_PLANARITY_STD_MM = 0.25
LOG_FOLDER = "gt_remesh_logs"
_set_step = set_step
_warn = warn


def prepare_gt_surface(vessel_mesh):
    """Keep original tessellation density; only repair what breaks VMTK.

    Nothing here resamples the surface, so the aneurysm texture is untouched.
    What is removed is exactly what the rest of the pipeline cannot survive:
    micron-scale edges (VMTK's boundary-preserving remesh keeps them and the
    final quality gate then rejects the case), non-manifold sheets, ears on the
    ostium rims, and wall punctures small enough that vmtkFlowExtensions would
    grow a spurious 5 mm tube out of them.
    """
    poly = clean_triangulate(vessel_mesh)
    poly = drop_degenerate_triangles(poly)
    poly = drop_boundary_ear_triangles(poly)
    poly, n_nm = repair_nonmanifold_triangles(poly)
    if n_nm > 0:
        poly, n_forced = force_manifold_triangles(poly)
        poly, n_nm = repair_nonmanifold_triangles(poly)
        if n_forced:
            _warn(
                f"cut {n_forced} triangle(s) to make the original manifold "
                f"({n_nm} non-manifold edges left)"
            )
    if n_nm > 0:
        _warn(f"{n_nm} non-manifold edges remain on the original after repair")
    poly, min_edge = weld_degenerate_vertices(poly)
    poly = drop_boundary_ear_triangles(poly)
    poly, n_pin = patch_wall_pinholes(poly, label="original")
    if n_pin:
        _warn(f"patched {n_pin} wall pinhole(s) on the original before flow extensions")
    poly = uncap_closed_surface(poly)
    print(f"  Original min edge after welding: {min_edge:.6f} mm")
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
    # One rim per ostium. Picking each frame's nearest rim independently let two
    # frames share one rim and let a frame borrow a different ostium's rim
    # entirely, which reported an axial spread larger than the opening's own
    # radius -- a planar cut cannot do that, so the warning was measuring the
    # wrong hole rather than a malformed one.
    centers = [pts.mean(axis=0) for pts in rims]
    pairs = sorted(
        (
            float(np.linalg.norm(centers[j] - np.asarray(frames[i][0], dtype=np.float64))),
            i,
            j,
        )
        for i in range(len(frames))
        for j in range(len(rims))
    )
    assigned = {}
    taken = set()
    for dist, i, j in pairs:
        if i in assigned or j in taken:
            continue
        assigned[i] = (j, dist)
        taken.add(j)

    for i, (origin, outward, radius) in enumerate(frames):
        origin = np.asarray(origin, dtype=np.float64)
        outward = np.asarray(outward, dtype=np.float64)
        if i not in assigned:
            _warn(
                f"ostium has no rim of its own: origin {np.round(origin, 2)} "
                f"r={radius:.3f} mm"
            )
            print(
                f"  Ostium planarity: origin {np.round(origin, 2)} "
                f"r={radius:.3f} mm (no rim matched)"
            )
            continue
        j, dist = assigned[i]
        rim = rims[j]
        axial = (rim - origin) @ outward
        axial_std = float(np.std(axial))
        # A rim further away than the opening is wide is not this ostium.
        reach = max(2.0 * float(radius), 1.0)
        flag = ""
        if dist > reach:
            flag = f"  (nearest rim is {dist:.2f} mm away; not this ostium)"
            _warn(
                f"ostium has no rim within {reach:.2f} mm: origin "
                f"{np.round(origin, 2)} r={radius:.3f} mm d={dist:.3f} mm"
            )
        elif axial_std > OPENING_PLANARITY_STD_MM:
            flag = "  (not planar)"
            _warn(
                f"ostium not planar: origin {np.round(origin, 2)} "
                f"r={radius:.3f} mm axial_std={axial_std:.3f} mm d={dist:.3f} mm"
            )
        print(
            f"  Ostium planarity: origin {np.round(origin, 2)} "
            f"r={radius:.3f} mm axial_std={axial_std:.3f} mm d={dist:.3f} mm{flag}"
        )


def _gt_centerline(work_vessel, extension_length, sample_spacing, gt_profiles=None):
    """Sanitised + strongly smoothed copy is only used to trace the lumen."""
    print("Step 2: Working copy for centerlines (sanitise + strong Taubin, discarded later)...")
    _set_step("2_centerline_working_copy")
    work = sanitize_vessel_for_vmtk(work_vessel)
    work = apply_taubin_smoothing(work)
    # Decimating to a quarter of the points tears the wall wherever the input
    # was already punctured, and merges neighbouring punctures into holes far
    # bigger than either -- a 4-point, 1.2 mm quad on p375. Those are not
    # openings, and a Voronoi diagram that has to route around them leaves the
    # lumen. Which loops are real is not a judgement call here: the ostia were
    # measured on the detailed original one step earlier, so anything on this
    # copy that matches none of them is damage, and this copy is discarded
    # anyway.
    work, n_junk = cap_unmatched_loops(work, gt_profiles, label="centerline copy")
    if n_junk:
        _warn(f"capped {n_junk} torn opening(s) on the centerline copy")
    anatomical_profiles = measure_open_profiles(work)
    log_profiles(anatomical_profiles, label="Anatomical")
    seed_points_from_profiles(anatomical_profiles)

    print("Step 3: Flow extensions on the working copy, then Voronoi centerline...")
    _set_step("3_voronoi_centerline")
    extended_work = add_flow_extensions(work, extension_length=extension_length)
    extended_profiles = measure_open_profiles(extended_work)
    log_profiles(extended_profiles, label="Extended")
    if len(extended_profiles) != len(anatomical_profiles):
        _warn(
            f"opening count changed after extensions "
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
    _set_step("1_prepare_original")
    gt_surface = prepare_gt_surface(original)
    print(f"  GT working surface: {gt_surface.GetNumberOfPoints()} points")
    print("Step 1b: Anatomical openings on the detailed original...")
    _set_step("1b_gt_openings")
    gt_profiles = measure_open_profiles(gt_surface)
    log_profiles(gt_profiles, label="GT anatomical")
    seed_points_from_profiles(gt_profiles)

    _work, work_profiles, branched = _gt_centerline(
        gt_surface, extension_length, sample_spacing, gt_profiles=gt_profiles
    )
    if len(gt_profiles) != len(work_profiles):
        _warn(
            f"opening count differs on GT vs sanitised working copy "
            f"({len(gt_profiles)} vs {len(work_profiles)}). Clipping uses GT loops."
        )

    print("Step 4: Flow extensions on the detailed original, then pipe-section ostia...")
    _set_step("4_pipe_section_ostia")
    extended_gt = add_flow_extensions(gt_surface, extension_length=extension_length)
    opened_gt, n_clipped = clip_flow_extensions_and_uncap(
        extended_gt,
        gt_profiles,
        extension_length=extension_length,
        centerline=branched,
        unextended_surface=gt_surface,
    )
    n_in = len(gt_profiles)
    if n_clipped < n_in:
        _warn(
            f"pipe-section uncap opened {n_clipped}/{n_in} ends; "
            "remaining ostia keep the original rim."
        )
    if len(inspect_openings(opened_gt)) < 2:
        raise TemplateQualityError(
            f"GT remesh has fewer than 2 openings after uncap ({n_clipped}/{n_in} clipped)."
        )
    opened_gt, _n_regions_pre = drop_tiny_islands(opened_gt)
    # The pipe-section uncap cuts fresh triangles at every rim and leaves
    # degeneracies behind: on p097 it took the shortest edge from 0.005032 to
    # 0.000037 mm and made 12 triangles of aspect ratio over 50, none of which
    # were in the surface it was handed. This weld clears the worst of that at a
    # fixed 1e-3 mm. It is not what saves the remesh -- p097 still diverged
    # 1.476x with it in place, and only a tolerance scaled to the mesh's own
    # mean edge fixed that (see REMESH_WELD_FRACTIONS) -- but it keeps the
    # quality gates honest about what the clip left behind.
    opened_gt, min_edge_after_clip = weld_degenerate_vertices(opened_gt)
    print(f"  Min edge after the uncap and weld: {min_edge_after_clip:.6f} mm")
    frames = opening_clip_frames(branched, gt_profiles)
    log_opening_planarity(opened_gt, frames)

    print(
        f"Step 5: Light Taubin (pass_band={GT_TAUBIN_PASS_BAND}, "
        f"n_iter={GT_TAUBIN_ITER}, boundary off)..."
    )
    _set_step("5_light_taubin")
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
    _set_step("6_isotropic_remesh")
    remeshed = remesh_surface_verified(
        opened_gt,
        target_edge_length=effective_edge,
        n_iter=GT_REMESH_N_ITER,
        connectivity_iter=GT_REMESH_CONNECTIVITY_ITER,
        label="GT surface",
    )
    print(f"  -> Remeshed surface points: {remeshed.GetNumberOfPoints()}")

    _set_step("7_finalize_and_save")
    final_surface, _n_regions = finalize_surface(remeshed, profiles=gt_profiles)
    assert_gt_remesh_scale(final_surface, original, context=dataset_id)
    openings = assert_template_quality(final_surface, context=dataset_id)
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
    if len(openings) != n_in:
        # This used to print and ship anyway, which is how p489 went out with 9
        # openings against 5 profiles and p129 with 6 against 4. Neither number
        # was real: extract_boundary_loops was splitting rims, and the split
        # also misled patch_wall_pinholes into re-opening a hole it had just
        # patched, which is what put the one genuine tear in p129. With the
        # count refereed by connectivity both cases now agree with their
        # profiles, so what reaches this branch is a rim this pipeline actually
        # tore -- the opposite of the well-made openings the dataset exists to
        # provide, and not something to ship quietly.
        raise TemplateQualityError(
            f"{dataset_id} finished with {len(openings)} openings against "
            f"{n_in} anatomical profiles ({n_clipped} pipe-section clipped); "
            f"the difference is torn rims, not ostia."
        )
    record(n_openings=len(openings), n_profiles=n_in, n_clipped=n_clipped)
    return out_file


def _process_one(dataset_id, v_file, args):
    def work():
        return process_gt_remesh_dataset(
            dataset_id=dataset_id,
            v_file=v_file,
            output_dir=args.output_dir,
            target_edge_length=args.target_edge_length,
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
    parser = argparse.ArgumentParser(
        description=(
            "Remesh original AneuX vessels to high-detail ground-truth surfaces "
            "(constant edge length, perpendicular ostia). Does not build a parent tube."
        )
    )
    add_shared_cli_args(
        parser,
        DEFAULT_OUTPUT_DIR,
        default_workers=25,
        include_remesh_grid=False,
    )
    parser.add_argument(
        "--target-edge-length",
        type=float,
        default=DEFAULT_GT_EDGE_LENGTH_MM,
        help="Uniform target edge length in mm (default 0.15).",
    )
    add_run_log_args(parser, LOG_FOLDER)
    parser.set_defaults(vessel_dir=TOTAL_CLEAN_ORIGINAL_MESH, from_folder=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    extra_log, on_worker_result = configure_batch_logging(
        args, LOG_FOLDER, DEFAULT_OUTPUT_DIR
    )
    extra = [
        "--target-edge-length",
        str(args.target_edge_length),
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
