"""Shared VMTK surface/centerline pipeline for template creation.

VTK/VMTK filters are crashy and orientation-sensitive. This module prefers
preflight checks over hoping a C++ filter no-ops safely.
"""
import os
import sys

os.environ.setdefault("VTK_OFFSCREEN", "1")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VTK_NUMBER_OF_THREADS", "1")

import argparse
import subprocess
import threading
from queue import Empty, Queue

import numpy as np
import pandas as pd
import pyvista as pv
import vtk
from tqdm import tqdm

try:
    from vmtk import vmtkscripts
    from vmtk import vtkvmtk
except ImportError as exc:
    raise ImportError(
        "Required package 'vmtk' is not installed. "
        "Install VMTK Python bindings so that `from vmtk import vmtkscripts` succeeds "
        "(e.g. conda install -c vmtk vmtk)."
    ) from exc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from aneux_paths import CSV_PATH as DEFAULT_CSV_PATH, VESSELS_AREA005 as DEFAULT_VESSEL_DIR

DEFAULT_TARGET_EDGE_LENGTH = 0.5
DEFAULT_EXTENSION_LENGTH = 5.0
DEFAULT_SAMPLE_SPACING = 0.1
DEFAULT_GRID_SPACING = 0.08
DEFAULT_MAX_GRID_SIZE = 250
DEFAULT_CAP_DISPLACEMENT = 0.1
MISR_FLOOR_MM = 0.35
# Voronoi MISR inside a sac can be tens of mm. Using that as a polyball radius
# rebuilds the aneurysm as a blob and explodes the modeller AABB. Parent-tube
# spheres are capped from anatomical openings / vessel size, not from sac MISR.
MISR_PARENT_OPENING_FACTOR = 1.5
MISR_PARENT_EXTENT_FRACTION = 0.20
R_TEMPLATE_FLOOR_MM = 0.30
PINHOLE_HOLE_SIZE_MM = 0.12
MIN_OPENING_RADIUS_MM = 0.08
MIN_OPENING_LOOP_POINTS = 6
MIN_EDGE_LENGTH_MM = 0.001
SLIVER_Q01_THRESHOLD = 0.3
FILTER_LOCATIONS = ["ICA pcom", "ICA oph", "ICA cav", "ICA bif"]


class TemplateQualityError(RuntimeError):
    """Raised when a case cannot be turned into a usable template."""


def _unit(vec):
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    nrm = float(np.linalg.norm(arr))
    if nrm < 1e-12:
        return arr
    return arr / nrm


def _vec3(xyz):
    arr = np.asarray(xyz, dtype=np.float64).reshape(-1)
    return [float(arr[0]), float(arr[1]), float(arr[2])]


def _set_vec3(setter, xyz):
    """VTK Python wrappers disagree on SetFoo(x,y,z) vs SetFoo([x,y,z])."""
    vals = _vec3(xyz)
    try:
        setter(vals)
    except TypeError:
        setter(vals[0], vals[1], vals[2])


def to_vtk_poly(mesh):
    """Detached vtkPolyData copy for VMTK (never pass a live PyVista pipeline)."""
    if mesh is None:
        return vtk.vtkPolyData()
    if isinstance(mesh, pv.UnstructuredGrid):
        mesh = mesh.extract_surface()
    vtk_poly = vtk.vtkPolyData()
    vtk_poly.DeepCopy(mesh)
    return vtk_poly


def clean_triangulate(surface):
    vtk_poly = to_vtk_poly(surface)
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(vtk_poly)
    cleaner.Update()
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(cleaner.GetOutputPort())
    tri.PassLinesOff()
    tri.PassVertsOff()
    tri.Update()
    return to_vtk_poly(tri.GetOutput())


def mesh_center(surface):
    b = surface.GetBounds()
    return np.array(
        [0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3]), 0.5 * (b[4] + b[5])],
        dtype=np.float64,
    )


def strip_all_arrays(surface):
    """Drop every point/cell/field array, including unnamed VTK leftovers."""
    vtk_poly = to_vtk_poly(surface)
    pd = vtk_poly.GetPointData()
    while pd.GetNumberOfArrays() > 0:
        pd.RemoveArray(0)
    cd = vtk_poly.GetCellData()
    while cd.GetNumberOfArrays() > 0:
        cd.RemoveArray(0)
    vtk_poly.GetFieldData().Initialize()
    pd.SetTCoords(None)
    cd.SetTCoords(None)
    pd.SetNormals(None)
    cd.SetNormals(None)
    return vtk_poly


def recompute_point_normals(surface, auto_orient=False):
    vtk_poly = to_vtk_poly(surface)
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(vtk_poly)
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOff()
    normals.ConsistencyOn()
    normals.SplittingOff()
    if auto_orient:
        normals.AutoOrientNormalsOn()
    else:
        normals.AutoOrientNormalsOff()
    normals.Update()
    return to_vtk_poly(normals.GetOutput())


def apply_taubin_smoothing(surface_mesh, pass_band=0.1, n_iter=15, feature_angle=45.0):
    """Volume-preserving smoothing to reduce Voronoi jitter without collapsing the wall."""
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(to_vtk_poly(surface_mesh))
    smoother.SetNumberOfIterations(n_iter)
    smoother.SetPassBand(pass_band)
    smoother.SetFeatureAngle(feature_angle)
    smoother.FeatureEdgeSmoothingOff()
    smoother.BoundarySmoothingOn()
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    return to_vtk_poly(smoother.GetOutput())


def extract_boundary_loops(surface):
    extractor = vtkvmtk.vtkvmtkPolyDataBoundaryExtractor()
    extractor.SetInputData(to_vtk_poly(surface))
    extractor.Update()
    return extractor.GetOutput()


def measure_open_profiles(surface):
    """Open-boundary loops with radius, barycenter, and (when available) outward normals."""
    vtk_poly = to_vtk_poly(surface)
    ref_sys = vtkvmtk.vtkvmtkBoundaryReferenceSystems()
    ref_sys.SetInputData(vtk_poly)
    ref_sys.SetBoundaryNormalsArrayName("BoundaryNormals")
    ref_sys.SetBoundaryRadiusArrayName("BoundaryRadius")
    ref_sys.SetPoint1ArrayName("Point1")
    ref_sys.SetPoint2ArrayName("Point2")
    ref_sys.Update()
    ref_poly = ref_sys.GetOutput()
    n_b = ref_poly.GetNumberOfPoints()
    if n_b == 0:
        raise TemplateQualityError("No open boundary profiles found on vessel mesh.")

    normals_array = ref_poly.GetPointData().GetArray("BoundaryNormals")
    radii_array = ref_poly.GetPointData().GetArray("BoundaryRadius")
    profiles = []
    for i in range(n_b):
        pos = np.array(ref_poly.GetPoint(i), dtype=np.float64)
        radius = float(radii_array.GetComponent(i, 0)) if radii_array else 0.0
        if normals_array is not None:
            normal = _unit(np.array(normals_array.GetTuple(i), dtype=np.float64))
        else:
            normal = np.zeros(3)
        profiles.append(
            {
                "index": i,
                "barycenter": pos,
                "normal": normal,
                "radius": radius,
            }
        )
    profiles.sort(key=lambda p: p["radius"], reverse=True)
    return profiles


def log_profiles(profiles, label=""):
    prefix = f"  {label} " if label else "  "
    print(f"{prefix}Boundary profiles found: {len(profiles)}")
    for p in profiles:
        print(
            f"{prefix} Profile {p['index']}: radius={p['radius']:.3f} mm, "
            f"center={np.round(p['barycenter'], 2)}, normal={np.round(p['normal'], 3)}"
        )
    print(
        f"{prefix} -> Selected inlet: Profile {profiles[0]['index']} "
        f"(radius={profiles[0]['radius']:.3f} mm)"
    )


def seed_points_from_profiles(profiles):
    if len(profiles) < 2:
        raise TemplateQualityError(
            f"Need at least 2 open boundaries for inlet/outlet seeding, found {len(profiles)}."
        )
    inlet = [profiles[0]["barycenter"]]
    outlets = [p["barycenter"] for p in profiles[1:]]
    return inlet, outlets


def add_flow_extensions(open_surface, extension_length=DEFAULT_EXTENSION_LENGTH):
    """Extrude cylinders on an OPEN surface. Must not be capped first or this is a no-op."""
    vtk_poly = clean_triangulate(open_surface)
    n_open = extract_boundary_loops(vtk_poly).GetNumberOfCells()
    if n_open == 0:
        raise TemplateQualityError(
            "Flow extensions require open boundaries; input surface is already closed."
        )

    extender = vmtkscripts.vmtkFlowExtensions()
    extender.Surface = vtk_poly
    extender.ExtensionLength = float(extension_length)
    extender.ExtensionMode = "boundarynormal"
    extender.InterpolationMode = "linear"
    extender.AdaptiveExtensionLength = 0
    extender.Interactive = 0
    extender.Execute()
    extended = clean_triangulate(extender.Surface)
    n_after = extract_boundary_loops(extended).GetNumberOfCells()
    if n_after == 0:
        raise TemplateQualityError("Flow extensions produced a closed surface (unexpected).")
    if extended.GetNumberOfPoints() <= vtk_poly.GetNumberOfPoints():
        print("  WARNING: flow-extension point count did not increase; VMTK may have skipped ends.")
    print(
        f"  Flow extensions: {n_open} openings in, {n_after} openings out, "
        f"{vtk_poly.GetNumberOfPoints()} -> {extended.GetNumberOfPoints()} points"
    )
    return extended


def cap_surface(open_surface, displacement=DEFAULT_CAP_DISPLACEMENT):
    """Close openings with a slight cap displacement so Delaunay tets at caps are non-degenerate."""
    vtk_poly = clean_triangulate(open_surface)
    capper = vtkvmtk.vtkvmtkCapPolyData()
    capper.SetInputData(vtk_poly)
    capper.SetDisplacement(float(displacement))
    capper.SetInPlaneDisplacement(0.0)
    capper.SetCellEntityIdsArrayName("CellEntityIds")
    capper.Update()
    capped = clean_triangulate(capper.GetOutput())
    n_open = extract_boundary_loops(capped).GetNumberOfCells()
    if n_open != 0:
        raise TemplateQualityError(f"Capping left {n_open} openings; cannot run centerlines.")
    return recompute_point_normals(capped, auto_orient=True)


def extract_voronoi_centerlines(closed_surface, source_points, target_points):
    if not target_points:
        raise TemplateQualityError("No outlet seed points for vmtkCenterlines.")
    vtk_poly = to_vtk_poly(closed_surface)
    centerlines = vmtkscripts.vmtkCenterlines()
    centerlines.Surface = vtk_poly
    centerlines.SeedSelectorName = "pointlist"
    centerlines.SourcePoints = [float(c) for pt in source_points for c in pt]
    centerlines.TargetPoints = [float(c) for pt in target_points for c in pt]
    centerlines.Interactive = 0
    centerlines.AppendEndPoints = 1
    centerlines.CapDisplacement = float(DEFAULT_CAP_DISPLACEMENT)
    centerlines.Execute()
    result = to_vtk_poly(centerlines.Centerlines)
    if result.GetNumberOfPoints() < 2 or result.GetNumberOfCells() < 1:
        raise TemplateQualityError("vmtkCenterlines returned an empty centerline.")
    return result


def resample_centerline(centerline, sample_spacing=DEFAULT_SAMPLE_SPACING):
    resampler = vmtkscripts.vmtkCenterlineResampling()
    resampler.Centerlines = to_vtk_poly(centerline)
    resampler.Length = float(sample_spacing)
    resampler.Execute()
    return to_vtk_poly(resampler.Centerlines)


def smooth_centerline_preserve_misr(centerline, smoothing_factor=0.1, iterations=100):
    """Laplacian-smooth geometry, then copy MISR from the pre-smooth line (VMTK does not update it)."""
    pre = to_vtk_poly(centerline)
    smoother = vmtkscripts.vmtkCenterlineSmoothing()
    smoother.Centerlines = pre
    smoother.SmoothingFactor = float(smoothing_factor)
    smoother.NumberOfSmoothingIterations = int(iterations)
    smoother.Execute()
    smoothed = to_vtk_poly(smoother.Centerlines)

    misr_pre = pre.GetPointData().GetArray("MaximumInscribedSphereRadius")
    misr_sm = smoothed.GetPointData().GetArray("MaximumInscribedSphereRadius")
    if misr_pre is None or misr_sm is None:
        return smoothed

    locator = vtk.vtkPointLocator()
    locator.SetDataSet(pre)
    locator.BuildLocator()
    for i in range(smoothed.GetNumberOfPoints()):
        pid = locator.FindClosestPoint(smoothed.GetPoint(i))
        misr_sm.SetComponent(i, 0, misr_pre.GetComponent(pid, 0))
    return smoothed


def extract_branches(centerline):
    extractor = vmtkscripts.vmtkBranchExtractor()
    extractor.Centerlines = to_vtk_poly(centerline)
    extractor.Execute()
    return to_vtk_poly(extractor.Centerlines)


def _misr_array_or_raise(centerline):
    misr = centerline.GetPointData().GetArray("MaximumInscribedSphereRadius")
    if misr is None:
        raise TemplateQualityError("Centerline has no MaximumInscribedSphereRadius array.")
    return misr


def _misr_values(misr_array):
    n = int(misr_array.GetNumberOfTuples())
    vals = np.empty(n, dtype=np.float64)
    for i in range(n):
        vals[i] = float(misr_array.GetComponent(i, 0))
    return vals


def parent_tube_misr_cap(profiles, reference_bounds):
    """Max sphere radius allowed when rasterizing the parent tube."""
    b = np.asarray(reference_bounds, dtype=np.float64).reshape(-1)
    extents = [float(b[1] - b[0]), float(b[3] - b[2]), float(b[5] - b[4])]
    extent_cap = MISR_PARENT_EXTENT_FRACTION * max(extents)
    opening_cap = extent_cap
    if profiles:
        opening_cap = MISR_PARENT_OPENING_FACTOR * max(float(p["radius"]) for p in profiles)
    return float(max(MISR_FLOOR_MM, min(opening_cap, extent_cap)))


def clamp_misr_for_parent_tube(misr_array, r_cap, r_floor=MISR_FLOOR_MM):
    raw = _misr_values(misr_array)
    n_floor = n_cap = 0
    max_r = 0.0
    for i, r_val in enumerate(raw):
        if r_val < r_floor:
            r_val = r_floor
            n_floor += 1
        if r_val > r_cap:
            r_val = r_cap
            n_cap += 1
        misr_array.SetComponent(i, 0, r_val)
        max_r = max(max_r, r_val)
    print(
        f"  MISR raw min/median/max={raw.min():.3f}/{np.median(raw):.3f}/{raw.max():.3f} mm; "
        f"parent-tube cap={r_cap:.3f} mm (floored {n_floor}, capped {n_cap} of {raw.size} points)"
    )
    return max_r


def generate_base_surface(
    branched_centerline,
    grid_spacing=DEFAULT_GRID_SPACING,
    max_grid_size=DEFAULT_MAX_GRID_SIZE,
    reference_bounds=None,
    profiles=None,
    extension_length=DEFAULT_EXTENSION_LENGTH,
):
    """Polyball tube on an isotropic grid. Discrete spheres: PolyBallLine is too slow on large volumes."""
    if not isinstance(branched_centerline, vtk.vtkPolyData):
        branched_centerline = to_vtk_poly(branched_centerline)
    misr_array = _misr_array_or_raise(branched_centerline)
    if reference_bounds is None:
        reference_bounds = pv.wrap(branched_centerline).bounds
    r_cap = parent_tube_misr_cap(profiles, reference_bounds)
    max_r = clamp_misr_for_parent_tube(misr_array, r_cap)
    vtk_cl = to_vtk_poly(branched_centerline)

    pad = 2.0 * max(max_r, MISR_FLOOR_MM) + float(extension_length) + 1.0
    model_bounds = [
        reference_bounds[0] - pad, reference_bounds[1] + pad,
        reference_bounds[2] - pad, reference_bounds[3] + pad,
        reference_bounds[4] - pad, reference_bounds[5] + pad,
    ]
    extents = [
        model_bounds[1] - model_bounds[0],
        model_bounds[3] - model_bounds[2],
        model_bounds[5] - model_bounds[4],
    ]
    spacing = float(grid_spacing)
    dims = [max(32, int(np.ceil(e / spacing)) + 1) for e in extents]
    peak = max(dims)
    if peak > int(max_grid_size):
        spacing = max(extents) / float(int(max_grid_size) - 1)
        dims = [max(32, int(round(e / spacing)) + 1) for e in extents]
        dims = [min(d, int(max_grid_size)) for d in dims]
    print(f"  CenterlineModeller grid dimensions: {dims} (isotropic spacing ~{spacing:.4f} mm)")

    modeller = vtkvmtk.vtkvmtkPolyBallModeller()
    modeller.SetInputData(vtk_cl)
    modeller.SetRadiusArrayName("MaximumInscribedSphereRadius")
    modeller.UsePolyBallLineOff()
    try:
        modeller.SetSampleDimensions(dims)
    except TypeError:
        modeller.SetSampleDimensions(int(dims[0]), int(dims[1]), int(dims[2]))
    try:
        modeller.SetModelBounds(model_bounds)
    except TypeError:
        modeller.SetModelBounds(*[float(v) for v in model_bounds])
    modeller.SetNegateFunction(0)
    modeller.Update()

    mc = vmtkscripts.vmtkMarchingCubes()
    mc.Image = modeller.GetOutput()
    mc.Level = 0.0
    mc.Connectivity = 1
    mc.Execute()
    raw = to_vtk_poly(mc.Surface)
    kept = keep_largest_region(raw)
    if kept.GetNumberOfPoints() < 50:
        raise TemplateQualityError("Marching cubes produced a degenerate tube surface.")
    slack = float(extension_length) + 2.0 * max_r + 2.0
    b = kept.GetBounds()
    axes = (
        ("x", b[0], b[1], reference_bounds[0], reference_bounds[1]),
        ("y", b[2], b[3], reference_bounds[2], reference_bounds[3]),
        ("z", b[4], b[5], reference_bounds[4], reference_bounds[5]),
    )
    for axis, s_lo, s_hi, v_lo, v_hi in axes:
        if s_lo < v_lo - slack - 1e-6 or s_hi > v_hi + slack + 1e-6:
            raise TemplateQualityError(
                f"Parent tube exceeded vessel bounds on {axis} "
                f"(surface {s_lo:.1f}..{s_hi:.1f} vs vessel {v_lo:.1f}..{v_hi:.1f}, slack={slack:.1f} mm)."
            )
    return kept


def keep_largest_region(surface):
    vtk_poly = to_vtk_poly(surface)
    n_regions = count_connected_regions(vtk_poly)
    if n_regions <= 1:
        return clean_triangulate(vtk_poly)
    largest = vtk.vtkPolyDataConnectivityFilter()
    largest.SetInputData(vtk_poly)
    largest.SetExtractionModeToLargestRegion()
    largest.Update()
    main = clean_triangulate(largest.GetOutput())
    print(
        f"  Kept largest of {n_regions} regions "
        f"({main.GetNumberOfPoints()} pts, dropped {vtk_poly.GetNumberOfPoints() - main.GetNumberOfPoints()} pts)."
    )
    return main


def fill_pinholes(surface, hole_size=PINHOLE_HOLE_SIZE_MM):
    filler = vtk.vtkFillHolesFilter()
    filler.SetInputData(to_vtk_poly(surface))
    filler.SetHoleSize(float(hole_size))
    filler.Update()
    return to_vtk_poly(filler.GetOutput())


def _plane_has_sign_change(surface, origin, normal):
    origin = np.asarray(origin, dtype=np.float64)
    normal = _unit(normal)
    vtk_poly = to_vtk_poly(surface)
    n_pts = vtk_poly.GetNumberOfPoints()
    if n_pts == 0:
        return False
    signs = np.empty(n_pts, dtype=np.float64)
    for i in range(n_pts):
        p = np.array(vtk_poly.GetPoint(i), dtype=np.float64)
        signs[i] = float(np.dot(normal, p - origin))
    vtk_poly.BuildCells()
    for ci in range(vtk_poly.GetNumberOfCells()):
        cell = vtk_poly.GetCell(ci)
        n = cell.GetNumberOfPoints()
        if n < 2:
            continue
        ids = [cell.GetPointId(j) for j in range(n)]
        for a, b in zip(ids, ids[1:] + ids[:1]):
            if signs[a] == 0.0 or signs[b] == 0.0:
                return True
            if signs[a] * signs[b] < 0.0:
                return True
    return False


def _clip_origin_candidates(surface, profile, search_mm):
    bary = np.asarray(profile["barycenter"], dtype=np.float64)
    outward = _unit(profile["normal"])
    inward = -outward
    radius = max(float(profile["radius"]), 0.2)
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(surface)
    locator.BuildLocator()
    closest_id = locator.FindClosestPoint(bary)
    closest = np.array(surface.GetPoint(closest_id), dtype=np.float64)
    if np.linalg.norm(closest - bary) > search_mm + 4.0 * radius:
        print(
            f"  [Uncap] Profile {profile['index']} has no nearby tube wall "
            f"(closest {np.linalg.norm(closest - bary):.2f} mm, r={radius:.3f} mm). Skipping."
        )
        return [], inward
    origins = [bary, closest]
    n_steps = 10
    for i in range(1, n_steps + 1):
        origins.append(bary + inward * (search_mm * i / n_steps))
        origins.append(closest + inward * (radius * i / n_steps))
    return origins, inward


def clip_one_profile(surface, profile, body_point, search_mm):
    """Local seam clip at one opening. Never call TopologicalSeamFilter if the plane misses."""
    outward = _unit(profile["normal"])
    plane_normal = -outward
    origins, _inward = _clip_origin_candidates(surface, profile, search_mm)
    if not origins:
        return surface, False
    origin = None
    for cand in origins:
        if _plane_has_sign_change(surface, cand, plane_normal):
            origin = cand
            break
    if origin is None:
        print(
            f"  [Uncap] Profile {profile['index']} plane does not intersect the tube "
            f"(r={profile['radius']:.3f} mm). Skipping this end."
        )
        return surface, False

    plane = vtk.vtkPlane()
    _set_vec3(plane.SetOrigin, origin)
    _set_vec3(plane.SetNormal, plane_normal)

    seam_filter = vtkvmtk.vtkvmtkTopologicalSeamFilter()
    seam_filter.SetInputData(surface)
    _set_vec3(seam_filter.SetClosestPoint, origin)
    seam_filter.SetSeamScalarsArrayName("SeamScalars")
    seam_filter.SetSeamFunction(plane)

    clipper = vtk.vtkClipPolyData()
    clipper.SetInputConnection(seam_filter.GetOutputPort())
    clipper.SetValue(0.0)
    clipper.InsideOutOff()
    clipper.GenerateClipScalarsOff()
    clipper.GenerateClippedOutputOff()

    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputConnection(clipper.GetOutputPort())
    connectivity.SetExtractionModeToClosestPointRegion()
    _set_vec3(connectivity.SetClosestPoint, body_point)

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(connectivity.GetOutputPort())
    cleaner.Update()
    candidate = to_vtk_poly(cleaner.GetOutput())

    n_prev = surface.GetNumberOfPoints()
    n_cand = candidate.GetNumberOfPoints()
    n_cells = candidate.GetNumberOfCells()
    if n_cand < 50 or n_cells < 50:
        print(
            f"  [Uncap] Profile {profile['index']} clip is degenerate "
            f"({n_cand} pts, {n_cells} cells). Keeping previous surface."
        )
        return surface, False
    if n_cand < 0.45 * n_prev:
        print(
            f"  [Uncap] Profile {profile['index']} clip discarded too much of the mesh "
            f"({n_cand}/{n_prev} pts). Keeping previous surface."
        )
        return surface, False
    matched = min(
        inspect_openings(candidate) or [{"radius": 1e9, "center": origin}],
        key=lambda op: float(np.linalg.norm(np.asarray(op["center"]) - np.asarray(profile["barycenter"]))),
    )
    max_r = max(4.0 * float(profile["radius"]), float(profile["radius"]) + 2.5)
    if float(matched.get("radius", 0.0)) > max_r:
        print(
            f"  [Uncap] Profile {profile['index']} clip opened a too-large loop "
            f"(r={matched['radius']:.3f} mm, cap {max_r:.3f} mm). Keeping previous surface."
        )
        return surface, False
    return candidate, True


def clip_flow_extensions_and_uncap(base_surface, profiles, extension_length=DEFAULT_EXTENSION_LENGTH):
    current = to_vtk_poly(base_surface)
    body_pt = mesh_center(current)
    n_clipped = 0
    for profile in profiles:
        search_mm = float(extension_length) + 3.0 * max(float(profile["radius"]), 0.5)
        current, ok = clip_one_profile(current, profile, body_pt, search_mm)
        if ok:
            n_clipped += 1
            body_pt = mesh_center(current)
    print(f"  Uncap: clipped {n_clipped}/{len(profiles)} openings")
    current = fill_pinholes(current)
    current = clean_triangulate(current)
    post = inspect_openings(current)
    print(
        "  Openings after uncap/pinhole-fill: "
        + ", ".join(f"r={op['radius']:.3f}mm n={op['n_points']}" for op in post)
    )
    return current, n_clipped


def _polyline_cells(centerline):
    vtk_cl = to_vtk_poly(centerline)
    vtk_cl.BuildCells()
    cells = []
    for ci in range(vtk_cl.GetNumberOfCells()):
        cell = vtk_cl.GetCell(ci)
        if cell.GetCellType() not in (vtk.VTK_LINE, vtk.VTK_POLY_LINE):
            continue
        n = cell.GetNumberOfPoints()
        if n < 2:
            continue
        pts = np.array([vtk_cl.GetPoint(cell.GetPointId(j)) for j in range(n)], dtype=np.float64)
        cells.append(pts)
    return cells, vtk_cl


def _trim_polyline_end(pts, origin, plane_normal, max_end_dist):
    origin = np.asarray(origin, dtype=np.float64)
    plane_normal = _unit(plane_normal)
    if np.linalg.norm(pts[0] - origin) <= max_end_dist:
        i = 0
        while i < len(pts) - 2 and float(np.dot(plane_normal, pts[i] - origin)) < 0.0:
            i += 1
        pts = pts[i:]
    if len(pts) < 2:
        return pts
    if np.linalg.norm(pts[-1] - origin) <= max_end_dist:
        i = len(pts) - 1
        while i > 1 and float(np.dot(plane_normal, pts[i] - origin)) < 0.0:
            i -= 1
        pts = pts[: i + 1]
    return pts


def clip_centerline_at_profiles(centerline, profiles, extension_length=DEFAULT_EXTENSION_LENGTH):
    """Trim only polyline ends that belong to a nearby opening (never a global AABB)."""
    cells, vtk_cl = _polyline_cells(centerline)
    kept = []
    for pts in cells:
        trimmed = pts
        for profile in profiles:
            origin = profile["barycenter"]
            plane_normal = -_unit(profile["normal"])
            max_end_dist = float(extension_length) + 3.0 * max(float(profile["radius"]), 0.5)
            trimmed = _trim_polyline_end(trimmed, origin, plane_normal, max_end_dist)
            if len(trimmed) < 2:
                break
        if len(trimmed) >= 2:
            kept.append(trimmed)
    if not kept:
        raise TemplateQualityError("Centerline clip removed every tract.")

    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    for pts in kept:
        ids = vtk.vtkIdList()
        for p in pts:
            ids.InsertNextId(points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2])))
        lines.InsertNextCell(ids)
    out = vtk.vtkPolyData()
    out.SetPoints(points)
    out.SetLines(lines)

    locator = vtk.vtkPointLocator()
    locator.SetDataSet(vtk_cl)
    locator.BuildLocator()
    for ai in range(vtk_cl.GetPointData().GetNumberOfArrays()):
        src = vtk_cl.GetPointData().GetArray(ai)
        name = src.GetName()
        if not name:
            continue
        dst = vtk.vtkDoubleArray()
        dst.SetName(name)
        dst.SetNumberOfComponents(src.GetNumberOfComponents())
        dst.SetNumberOfTuples(out.GetNumberOfPoints())
        for i in range(out.GetNumberOfPoints()):
            pid = locator.FindClosestPoint(out.GetPoint(i))
            dst.SetTuple(i, src.GetTuple(pid))
        out.GetPointData().AddArray(dst)
    return out


def compute_template_local_radii(template_mesh, branched_centerline):
    """R_template from the polyball line (closest point on the polyline), not nearest vertex."""
    vtk_template = to_vtk_poly(template_mesh)
    vtk_cl = to_vtk_poly(branched_centerline)
    n_pts = vtk_template.GetNumberOfPoints()
    r_template = np.full(n_pts, R_TEMPLATE_FLOOR_MM, dtype=np.float64)

    if vtk_cl.GetNumberOfPoints() == 0:
        return r_template

    vtk_cl.BuildCells()
    vtk_cl.BuildLinks()
    polyball = vtkvmtk.vtkvmtkPolyBallLine()
    try:
        polyball.SetInput(vtk_cl)
    except (TypeError, AttributeError):
        polyball.SetInputData(vtk_cl)
    polyball.SetPolyBallRadiusArrayName("MaximumInscribedSphereRadius")
    polyball.UseRadiusInformationOn()

    locator = vtk.vtkPointLocator()
    locator.SetDataSet(vtk_cl)
    locator.BuildLocator()
    misr_arr = vtk_cl.GetPointData().GetArray("MaximumInscribedSphereRadius")

    for i in range(n_pts):
        p = vtk_template.GetPoint(i)
        try:
            polyball.EvaluateFunction(p)
            r_val = float(polyball.GetLastPolyBallCenterRadius())
        except Exception:
            r_val = 0.0
        if r_val <= 1e-6 and misr_arr is not None:
            pid = locator.FindClosestPoint(p)
            r_val = float(misr_arr.GetComponent(pid, 0))
        r_template[i] = max(R_TEMPLATE_FLOOR_MM, r_val)
    return r_template


def compute_raycast_stretch_distances(template_mesh, ground_truth_mesh, r_template=None, max_ray_length=25.0, tol=1e-4):
    """Outward MISR-tube stretch vs GT. `n = -template_normals` is required: VTK normals are inward here."""
    vtk_template = to_vtk_poly(template_mesh)

    template_normals_filter = vtk.vtkPolyDataNormals()
    template_normals_filter.SetInputData(vtk_template)
    template_normals_filter.ComputePointNormalsOn()
    template_normals_filter.ComputeCellNormalsOff()
    template_normals_filter.ConsistencyOn()
    template_normals_filter.SplittingOff()
    template_normals_filter.AutoOrientNormalsOff()
    template_normals_filter.Update()
    template_mesh_with_normals = to_vtk_poly(template_normals_filter.GetOutput())

    pv_template = pv.wrap(template_mesh_with_normals)

    gt_normals_filter = vtk.vtkPolyDataNormals()
    gt_normals_filter.SetInputData(to_vtk_poly(ground_truth_mesh))
    gt_normals_filter.ComputeCellNormalsOn()
    gt_normals_filter.ComputePointNormalsOff()
    gt_normals_filter.ConsistencyOn()
    gt_normals_filter.SplittingOff()
    gt_normals_filter.Update()
    gt_mesh_with_normals = to_vtk_poly(gt_normals_filter.GetOutput())
    gt_cell_normals = gt_mesh_with_normals.GetCellData().GetNormals()

    locator = vtk.vtkCellLocator()
    locator.SetDataSet(gt_mesh_with_normals)
    locator.BuildLocator()

    template_pts = pv_template.points
    template_normals = pv_template.point_normals
    n_pts = pv_template.n_points
    distances = np.zeros(n_pts, dtype=np.float64)
    t = vtk.mutable(0.0)
    x = [0.0, 0.0, 0.0]
    pcoords = [0.0, 0.0, 0.0]
    subId = vtk.mutable(0)
    cellId = vtk.mutable(0)

    for i in range(n_pts):
        p = template_pts[i]
        nrm = np.asarray(template_normals[i], dtype=np.float64)
        n = -_unit(nrm)
        r_local = r_template[i] if r_template is not None else 1.0

        p_inward = p - n * 1.5
        hit_inward = locator.IntersectWithLine(p, p_inward, tol, t, x, pcoords, subId, cellId)
        if hit_inward:
            d_inward = np.linalg.norm(np.array(x) - p)
            if d_inward < 0.4:
                distances[i] = 0.0
                continue

        p_end = p + n * max_ray_length
        hit = locator.IntersectWithLine(p, p_end, tol, t, x, pcoords, subId, cellId)
        if hit:
            d = np.linalg.norm(np.array(x) - p)
            if 0.10 < d <= (3.5 * r_local):
                cid = cellId.get()
                if gt_cell_normals:
                    cell_normal = np.array(gt_cell_normals.GetTuple(cid))
                    if np.dot(n, cell_normal) > 0.2:
                        distances[i] = d
                else:
                    distances[i] = d
    return distances


def build_target_edge_array(template_mesh, distances, r_template, base_edge=0.50, min_edge=0.01):
    vtk_poly = to_vtk_poly(template_mesh)
    n_pts = vtk_poly.GetNumberOfPoints()
    stretch_factors = 1.0 + (distances / np.maximum(R_TEMPLATE_FLOOR_MM, r_template))
    target_edge_lengths = np.maximum(min_edge, base_edge / (stretch_factors ** 1.5))

    vtk_target_array = vtk.vtkDoubleArray()
    vtk_target_array.SetName("TargetEdgeLength")
    vtk_target_array.SetNumberOfTuples(n_pts)
    for i in range(n_pts):
        vtk_target_array.SetValue(i, float(target_edge_lengths[i]))
    vtk_poly.GetPointData().AddArray(vtk_target_array)
    return vtk_poly, target_edge_lengths, stretch_factors


def remesh_surface_adaptively(open_surface_with_array, edge_array_name="TargetEdgeLength", n_iter=10):
    remesher = vmtkscripts.vmtkSurfaceRemeshing()
    remesher.Surface = to_vtk_poly(open_surface_with_array)
    remesher.ElementSizeMode = "edgelengtharray"
    remesher.TargetEdgeLengthArrayName = edge_array_name
    remesher.PreserveBoundaryEdges = 1
    remesher.NumberOfIterations = n_iter
    remesher.Execute()
    return to_vtk_poly(remesher.Surface)


def remesh_surface_isotropically(open_surface, target_edge_length=0.5, n_iter=10):
    remesher = vmtkscripts.vmtkSurfaceRemeshing()
    remesher.Surface = to_vtk_poly(open_surface)
    remesher.ElementSizeMode = "edgelength"
    remesher.TargetEdgeLength = float(target_edge_length)
    remesher.PreserveBoundaryEdges = 1
    remesher.NumberOfIterations = n_iter
    remesher.Execute()
    return to_vtk_poly(remesher.Surface)


def uniform_edge_length_for_profiles(profiles, target_edge_length):
    """Only refine globally when the smallest opening is smaller than the target edge."""
    if not profiles:
        return float(target_edge_length)
    r_min = min(float(p["radius"]) for p in profiles)
    return float(min(target_edge_length, max(0.15, r_min)))


def count_connected_regions(surface):
    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(to_vtk_poly(surface))
    conn.SetExtractionModeToAllRegions()
    conn.Update()
    return int(conn.GetNumberOfExtractedRegions())


def inspect_openings(surface):
    loops = extract_boundary_loops(surface)
    info = []
    for i in range(loops.GetNumberOfCells()):
        cell = loops.GetCell(i)
        n = cell.GetNumberOfPoints()
        if n == 0:
            continue
        pts = np.array([cell.GetPoints().GetPoint(j) for j in range(n)], dtype=np.float64)
        center = pts.mean(axis=0)
        radius = float(np.mean(np.linalg.norm(pts - center, axis=1)))
        info.append({"index": i, "n_points": n, "radius": radius, "center": center})
    return info


def inspect_surface_topology(surface):
    """Edge usage, shortest edge, and triangle quality (radius-ratio)."""
    vtk_poly = to_vtk_poly(surface)
    vtk_poly.BuildCells()
    edge_count = {}
    min_edge = float("inf")
    n_tri = 0
    for ci in range(vtk_poly.GetNumberOfCells()):
        cell = vtk_poly.GetCell(ci)
        n = cell.GetNumberOfPoints()
        if n < 2:
            continue
        ids = [cell.GetPointId(j) for j in range(n)]
        pts = [np.asarray(vtk_poly.GetPoint(pid), dtype=np.float64) for pid in ids]
        for a in range(n):
            b = (a + 1) % n if n > 2 else a + 1
            if b >= n:
                continue
            i0, i1 = ids[a], ids[b]
            key = (i0, i1) if i0 < i1 else (i1, i0)
            edge_count[key] = edge_count.get(key, 0) + 1
            if n == 3:
                min_edge = min(min_edge, float(np.linalg.norm(pts[a] - pts[b])))
        if n == 3:
            n_tri += 1
    n_boundary = sum(1 for c in edge_count.values() if c == 1)
    n_nonmanifold = sum(1 for c in edge_count.values() if c > 2)
    if not np.isfinite(min_edge) or n_tri == 0:
        min_edge = 0.0

    med_q01 = None
    frac_sliver = None
    try:
        quality = vtk.vtkMeshQuality()
        quality.SetInputData(vtk_poly)
        quality.SetTriangleQualityMeasureToRadiusRatio()
        quality.Update()
        arr = quality.GetOutput().GetCellData().GetArray("Quality")
        if arr is not None and arr.GetNumberOfTuples() > 0:
            rr = np.array([arr.GetValue(i) for i in range(arr.GetNumberOfTuples())], dtype=np.float64)
            q01 = 1.0 / np.maximum(rr, 1e-12)
            med_q01 = float(np.median(q01))
            frac_sliver = float(np.mean(q01 < SLIVER_Q01_THRESHOLD))
    except Exception:
        pass
    return {
        "n_triangles": n_tri,
        "n_boundary_edges": n_boundary,
        "n_nonmanifold": n_nonmanifold,
        "min_edge": float(min_edge),
        "median_q01": med_q01,
        "frac_sliver": frac_sliver,
    }


def drop_tiny_islands(surface):
    main = keep_largest_region(surface)
    return main, count_connected_regions(main)


def finalize_surface(surface):
    cleaned = fill_pinholes(clean_triangulate(surface))
    cleaned = strip_all_arrays(cleaned)
    cleaned, n_regions = drop_tiny_islands(cleaned)
    cleaned = strip_all_arrays(cleaned)
    cleaned = recompute_point_normals(cleaned, auto_orient=False)
    return cleaned, n_regions


def assert_template_scale(surface, reference_mesh, context="template", max_area_ratio=2.5):
    tpl_area = float(pv.wrap(to_vtk_poly(surface)).area)
    ref_area = float(pv.wrap(to_vtk_poly(reference_mesh)).area)
    if ref_area <= 1e-6:
        return tpl_area, ref_area
    ratio = tpl_area / ref_area
    print(f"  Template area={tpl_area:.1f} mm^2 vs vessel {ref_area:.1f} mm^2 (ratio {ratio:.2f})")
    if ratio > max_area_ratio:
        raise TemplateQualityError(
            f"{context} parent tube area is {ratio:.2f}x the vessel "
            f"({tpl_area:.1f} vs {ref_area:.1f} mm^2); MISR blob or modeller overflow."
        )
    return tpl_area, ref_area


def assert_template_quality(surface, n_expected_openings, context="template"):
    vtk_poly = to_vtk_poly(surface)
    n_regions = count_connected_regions(vtk_poly)
    openings = inspect_openings(vtk_poly)
    issues = []
    if n_regions != 1:
        issues.append(f"{n_regions} connected components")
    if len(openings) != int(n_expected_openings):
        issues.append(f"{len(openings)} openings (expected {n_expected_openings})")
    for op in openings:
        if op["radius"] < MIN_OPENING_RADIUS_MM or op["n_points"] < MIN_OPENING_LOOP_POINTS:
            issues.append(
                f"pinhole/degenerate opening r={op['radius']:.4f} mm npts={op['n_points']}"
            )
    pd = vtk_poly.GetPointData()
    cd = vtk_poly.GetCellData()
    for i in range(pd.GetNumberOfArrays()):
        name = pd.GetArrayName(i)
        if not name or name.startswith("Array "):
            issues.append(f"unnamed point array {name!r}")
    for i in range(cd.GetNumberOfArrays()):
        name = cd.GetArrayName(i)
        if not name or name.startswith("Array "):
            issues.append(f"unnamed cell array {name!r}")
        elif name != "Normals":
            pass
    if vtk_poly.GetNumberOfPoints() < 100 or vtk_poly.GetNumberOfCells() < 100:
        issues.append("mesh too small")
    topo = inspect_surface_topology(vtk_poly)
    sliver_txt = (
        f"median_q={topo['median_q01']:.3f}, slivers(q<{SLIVER_Q01_THRESHOLD})={100.0 * topo['frac_sliver']:.2f}%"
        if topo["median_q01"] is not None and topo["frac_sliver"] is not None
        else "triangle quality n/a"
    )
    print(
        f"  Mesh topology: nonmanifold={topo['n_nonmanifold']}, "
        f"min_edge={topo['min_edge']:.4f} mm, {sliver_txt}"
    )
    if topo["n_nonmanifold"] > 0:
        issues.append(f"{topo['n_nonmanifold']} non-manifold edges")
    if topo["n_triangles"] > 0 and topo["min_edge"] < MIN_EDGE_LENGTH_MM:
        issues.append(
            f"degenerate min edge {topo['min_edge']:.6f} mm (floor {MIN_EDGE_LENGTH_MM} mm)"
        )
    if issues:
        raise TemplateQualityError(f"{context} quality failed: " + "; ".join(issues))
    return openings


def save_polydata(surface, out_file):
    vtk_poly = to_vtk_poly(surface)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(out_file)
    writer.SetInputData(vtk_poly)
    writer.SetDataModeToBinary()
    if writer.Write() != 1:
        raise TemplateQualityError(f"Failed to write {out_file}")


def build_parent_tube(
    vessel_mesh,
    extension_length=DEFAULT_EXTENSION_LENGTH,
    sample_spacing=DEFAULT_SAMPLE_SPACING,
    grid_spacing=DEFAULT_GRID_SPACING,
    max_grid_size=DEFAULT_MAX_GRID_SIZE,
):
    """Shared path: smooth -> extend -> cap -> centerline -> polyball tube -> uncap at anatomy."""
    print("Step 1: Applying Taubin surface smoothing...")
    smoothed_vessel = apply_taubin_smoothing(vessel_mesh)

    print("Step 1b: Detecting anatomical inlet/outlet boundaries...")
    anatomical_profiles = measure_open_profiles(smoothed_vessel)
    log_profiles(anatomical_profiles, label="Anatomical")
    _inlet, _outlets = seed_points_from_profiles(anatomical_profiles)

    print("Step 2: Adding flow extensions on the open surface...")
    extended_vessel = add_flow_extensions(smoothed_vessel, extension_length=extension_length)

    print("Step 2b: Detecting extended-end seeds...")
    extended_profiles = measure_open_profiles(extended_vessel)
    log_profiles(extended_profiles, label="Extended")
    if len(extended_profiles) != len(anatomical_profiles):
        print(
            f"  WARNING: opening count changed after extensions "
            f"({len(anatomical_profiles)} -> {len(extended_profiles)}). Using extended ends as seeds."
        )
    source_pts, target_pts = seed_points_from_profiles(extended_profiles)

    print("Step 2c: Capping extended surface for Delaunay centerlines...")
    closed_extended = cap_surface(extended_vessel)

    print("Step 3: Extracting Voronoi centerline and MISR...")
    centerline = extract_voronoi_centerlines(closed_extended, source_pts, target_pts)

    print(f"Step 4: Spline resampling ({sample_spacing} mm) and trajectory smoothing...")
    resampled = resample_centerline(centerline, sample_spacing=sample_spacing)
    smooth_centerline = smooth_centerline_preserve_misr(resampled)

    print("Step 5: Extracting branches...")
    branched_centerline = extract_branches(smooth_centerline)

    print("Step 6: Generating multi-branch base surface (vmtkCenterlineModeller)...")
    base_surface = generate_base_surface(
        branched_centerline,
        grid_spacing=grid_spacing,
        max_grid_size=max_grid_size,
        reference_bounds=smoothed_vessel.GetBounds(),
        profiles=anatomical_profiles,
        extension_length=extension_length,
    )

    print("Step 7: Uncapping open boundaries and removing flow extensions...")
    open_base_surface, n_clipped = clip_flow_extensions_and_uncap(
        base_surface, anatomical_profiles, extension_length=extension_length
    )
    print(f"  -> Open base surface points: {open_base_surface.GetNumberOfPoints()}")
    if n_clipped < 2:
        raise TemplateQualityError(f"Uncap opened only {n_clipped} ends; need at least inlet and one outlet.")
    return {
        "smoothed_vessel": smoothed_vessel,
        "anatomical_profiles": anatomical_profiles,
        "branched_centerline": branched_centerline,
        "open_base_surface": open_base_surface,
        "n_clipped": n_clipped,
    }


def process_variable_dataset(
    dataset_id,
    v_file,
    output_dir,
    target_edge_length=DEFAULT_TARGET_EDGE_LENGTH,
    extension_length=DEFAULT_EXTENSION_LENGTH,
    sample_spacing=DEFAULT_SAMPLE_SPACING,
    grid_spacing=DEFAULT_GRID_SPACING,
    max_grid_size=DEFAULT_MAX_GRID_SIZE,
):
    print(f"\n=========================================\nProcessing Adaptive Variable Remeshing Case: {dataset_id}")
    vessel_mesh = pv.read(v_file)
    built = build_parent_tube(
        vessel_mesh,
        extension_length=extension_length,
        sample_spacing=sample_spacing,
        grid_spacing=grid_spacing,
        max_grid_size=max_grid_size,
    )
    open_base_surface = built["open_base_surface"]
    branched_centerline = built["branched_centerline"]
    anatomical_profiles = built["anatomical_profiles"]

    print("Step 8a: Computing local tube radius and raycasting stretch vs ground truth...")
    r_template = compute_template_local_radii(open_base_surface, branched_centerline)
    stretch_distances = compute_raycast_stretch_distances(
        open_base_surface, vessel_mesh, r_template=r_template
    )
    min_edge = 0.01
    print(
        f"Step 8b: Building stretch metric k = 1 + d / R_template "
        f"(Base={target_edge_length} mm, Min={min_edge:.2f} mm)..."
    )
    surface_with_array, _edge_lengths, stretch_factors = build_target_edge_array(
        open_base_surface,
        stretch_distances,
        r_template,
        base_edge=target_edge_length,
        min_edge=min_edge,
    )
    healthy = stretch_factors[stretch_distances < 0.3]
    stretched = stretch_factors[stretch_distances >= 0.3]
    mean_k_healthy = float(np.mean(healthy)) if healthy.size else 1.0
    mean_k_stretch = float(np.mean(stretched)) if stretched.size else 1.0
    print(
        f"  -> Stretch Factor k metrics: Max k={float(np.max(stretch_factors)):.2f}, "
        f"Healthy Vessel k={mean_k_healthy:.2f}, Stretched Zone k={mean_k_stretch:.2f}"
    )

    print("Step 8c: Adaptively remeshing surface (ElementSizeMode='edgelengtharray')...")
    remeshed_surface = remesh_surface_adaptively(surface_with_array, edge_array_name="TargetEdgeLength")
    print(f"  -> Adaptive remeshed surface points: {remeshed_surface.GetNumberOfPoints()}")
    remesh_openings = inspect_openings(remeshed_surface)
    print(
        "  Openings after remesh: "
        + ", ".join(f"r={op['radius']:.3f}mm n={op['n_points']}" for op in remesh_openings)
    )

    final_surface, _n_regions = finalize_surface(remeshed_surface)
    assert_template_scale(final_surface, vessel_mesh, context=dataset_id)
    openings = assert_template_quality(
        final_surface, n_expected_openings=int(built["n_clipped"]), context=dataset_id
    )

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{dataset_id}.vtp")
    save_polydata(final_surface, out_file)
    read_back = pv.read(out_file)
    print(f"Successfully saved variable remeshed surface to: {out_file}")
    print(
        f"  -> Verified Saved Mesh: {read_back.n_points} points, {read_back.n_cells} cells, "
        f"disk size={os.path.getsize(out_file)} bytes"
    )
    print(
        f"  -> Verified Open Boundaries Count: {len(openings)} "
        f"(expected {int(built['n_clipped'])} clipped openings; {len(anatomical_profiles)} anatomical profiles)"
    )
    return out_file


def process_uniform_dataset(
    dataset_id,
    v_file,
    output_dir,
    target_edge_length=DEFAULT_TARGET_EDGE_LENGTH,
    extension_length=DEFAULT_EXTENSION_LENGTH,
    sample_spacing=DEFAULT_SAMPLE_SPACING,
    grid_spacing=DEFAULT_GRID_SPACING,
    max_grid_size=DEFAULT_MAX_GRID_SIZE,
):
    print(f"\n=========================================\nProcessing Uniform Remeshing Case: {dataset_id}")
    vessel_mesh = pv.read(v_file)
    built = build_parent_tube(
        vessel_mesh,
        extension_length=extension_length,
        sample_spacing=sample_spacing,
        grid_spacing=grid_spacing,
        max_grid_size=max_grid_size,
    )
    open_base_surface = built["open_base_surface"]
    anatomical_profiles = built["anatomical_profiles"]
    effective_edge = uniform_edge_length_for_profiles(anatomical_profiles, target_edge_length)
    r_min = min(p["radius"] for p in anatomical_profiles)
    print(
        f"Step 8: Isotropically remeshing surface "
        f"(TargetEdgeLength={effective_edge:.3f} mm, R_min={r_min:.3f} mm)..."
    )
    remeshed_surface = remesh_surface_isotropically(
        open_base_surface, target_edge_length=effective_edge
    )
    print(f"  -> Remeshed surface points: {remeshed_surface.GetNumberOfPoints()}")

    final_surface, _n_regions = finalize_surface(remeshed_surface)
    assert_template_scale(final_surface, vessel_mesh, context=dataset_id)
    openings = assert_template_quality(
        final_surface, n_expected_openings=int(built["n_clipped"]), context=dataset_id
    )

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{dataset_id}.vtp")
    save_polydata(final_surface, out_file)
    read_back = pv.read(out_file)
    print(f"Successfully saved uniform remeshed surface to: {out_file}")
    print(
        f"  -> Verified Saved Mesh: {read_back.n_points} points, {read_back.n_cells} cells, "
        f"disk size={os.path.getsize(out_file)} bytes"
    )
    print(
        f"  -> Verified Open Boundaries Count: {len(openings)} "
        f"(expected {int(built['n_clipped'])} clipped openings; {len(anatomical_profiles)} anatomical profiles)"
    )
    return out_file


def process_centerline_dataset(
    dataset_id,
    v_file,
    output_dir,
    extension_length=DEFAULT_EXTENSION_LENGTH,
    sample_spacing=DEFAULT_SAMPLE_SPACING,
):
    print(f"\n=========================================\nProcessing Centerline Case: {dataset_id}")
    vessel_mesh = pv.read(v_file)
    print("Step 1: Applying Taubin surface smoothing...")
    smoothed_vessel = apply_taubin_smoothing(vessel_mesh)
    print("Step 1b: Detecting anatomical inlet/outlet boundaries...")
    anatomical_profiles = measure_open_profiles(smoothed_vessel)
    log_profiles(anatomical_profiles, label="Anatomical")
    seed_points_from_profiles(anatomical_profiles)

    print("Step 2: Adding flow extensions on the open surface...")
    extended_vessel = add_flow_extensions(smoothed_vessel, extension_length=extension_length)
    extended_profiles = measure_open_profiles(extended_vessel)
    source_pts, target_pts = seed_points_from_profiles(extended_profiles)

    print("Step 2c: Capping extended surface for Delaunay centerlines...")
    closed_extended = cap_surface(extended_vessel)

    print("Step 3: Extracting Voronoi centerline and MISR...")
    centerline = extract_voronoi_centerlines(closed_extended, source_pts, target_pts)
    print(f"Step 4: Spline resampling ({sample_spacing} mm) and trajectory smoothing...")
    resampled = resample_centerline(centerline, sample_spacing=sample_spacing)
    smooth_centerline = smooth_centerline_preserve_misr(resampled)
    print("Step 5: Extracting branches...")
    branched_centerline = extract_branches(smooth_centerline)
    print("Step 6: Trimming flow-extension ends at anatomical profile planes...")
    final_centerline = clip_centerline_at_profiles(
        branched_centerline, anatomical_profiles, extension_length=extension_length
    )
    if final_centerline.GetNumberOfCells() < 1:
        raise TemplateQualityError("Clipped centerline has no cells.")

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{dataset_id}.vtp")
    save_polydata(final_centerline, out_file)
    print(f"Successfully saved centerline to: {out_file}")
    return out_file


def load_valid_datasets(csv_path, vessel_dir, limit=None, case_ids=None):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Clinical CSV file not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    df["location"] = df["location"].astype(str).str.strip()
    df_filtered = df[df["location"].isin(FILTER_LOCATIONS)]
    wanted = None
    if case_ids:
        wanted = {str(x) for x in case_ids}
    valid = []
    for _, row in df_filtered.iterrows():
        dataset_id = str(row["dataset"])
        if wanted is not None and dataset_id not in wanted:
            continue
        v_file = os.path.join(vessel_dir, f"{dataset_id}.vtp")
        if os.path.exists(v_file):
            valid.append((dataset_id, v_file))
    if limit is not None:
        valid = valid[: int(limit)]
    return valid


def add_shared_cli_args(parser, default_output_dir, default_workers, include_remesh_grid=True):
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV_PATH, help="Path to clinical.csv")
    parser.add_argument("--vessel-dir", type=str, default=DEFAULT_VESSEL_DIR, help="Directory of input vessel .vtp files")
    parser.add_argument("--output-dir", type=str, default=default_output_dir, help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of dataset meshes to process")
    parser.add_argument("--workers", type=int, default=default_workers, help="Number of parallel worker processes")
    parser.add_argument("--case", type=str, default=None, help="Process a single dataset id")
    parser.add_argument("--vessel-file", type=str, default=None, help="Explicit input .vtp for --case")
    parser.add_argument("--skip-existing", action="store_true", help="Skip cases whose output .vtp already exists")
    parser.add_argument("--cases", type=str, nargs="*", default=None, help="Optional subset of dataset ids")
    parser.add_argument("--extension-length", type=float, default=DEFAULT_EXTENSION_LENGTH, help="Flow extension length in mm")
    parser.add_argument("--sample-spacing", type=float, default=DEFAULT_SAMPLE_SPACING, help="Centerline resampling spacing in mm")
    if include_remesh_grid:
        parser.add_argument("--target-edge-length", type=float, default=DEFAULT_TARGET_EDGE_LENGTH, help="Base / uniform target edge length in mm")
        parser.add_argument("--grid-spacing", type=float, default=DEFAULT_GRID_SPACING, help="Requested modeller voxel size in mm")
        parser.add_argument("--max-grid-size", type=int, default=DEFAULT_MAX_GRID_SIZE, help="Max voxels along the longest axis (spacing stays isotropic)")
    return parser


def run_batch(script_path, process_one, args, extra_cli_flags):
    os.makedirs(args.output_dir, exist_ok=True)
    if args.case:
        v_file = args.vessel_file or os.path.join(args.vessel_dir, f"{args.case}.vtp")
        if not os.path.exists(v_file):
            raise FileNotFoundError(f"Vessel file not found: {v_file}")
        process_one(args.case, v_file, args)
        return

    valid_datasets = load_valid_datasets(args.csv, args.vessel_dir, limit=args.limit, case_ids=args.cases)
    if args.skip_existing:
        valid_datasets = [
            (did, path)
            for did, path in valid_datasets
            if not os.path.exists(os.path.join(args.output_dir, f"{did}.vtp"))
        ]
    num_cases = len(valid_datasets)
    if num_cases == 0:
        print("No matching dataset files to process.")
        return
    num_workers = max(1, min(int(args.workers), num_cases))
    print(f"CSV Path: {args.csv}")
    print(f"Vessel Dir: {args.vessel_dir}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Processing limit: {args.limit} samples | Valid dataset cases: {num_cases}")
    print(f"Parallel Workers: {num_workers} (Requested={args.workers}, Active Workers={num_workers})")

    failures = []
    if num_workers == 1:
        for dataset_id, v_file in tqdm(valid_datasets, desc="Processing"):
            try:
                process_one(dataset_id, v_file, args)
            except Exception as exc:
                print(f"Error processing case {dataset_id}: {exc}")
                failures.append((dataset_id, str(exc)))
    else:
        print(f"Spawning worker pool with max active concurrent workers = {num_workers}...")
        task_queue = Queue()
        for item in valid_datasets:
            task_queue.put(item)
        pbar = tqdm(total=num_cases, desc="Processing Parallel")
        lock = threading.Lock()

        def worker_thread():
            while True:
                try:
                    dataset_id, v_file = task_queue.get_nowait()
                except Empty:
                    break
                cmd = [
                    sys.executable,
                    script_path,
                    "--case",
                    str(dataset_id),
                    "--vessel-file",
                    v_file,
                    "--output-dir",
                    args.output_dir,
                ] + extra_cli_flags
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                out, _ = proc.communicate()
                if proc.returncode != 0:
                    print(f"\n[ERROR] Case {dataset_id} failed (code {proc.returncode}):\n{out}")
                    with lock:
                        failures.append((dataset_id, out[-2000:] if out else f"exit {proc.returncode}"))
                pbar.update(1)

        threads = [threading.Thread(target=worker_thread) for _ in range(num_workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        pbar.close()
        print("Parallel execution complete. All worker processes finished.")

    if failures:
        fail_path = os.path.join(args.output_dir, "failures.txt")
        with open(fail_path, "w", encoding="utf-8") as handle:
            for dataset_id, msg in failures:
                handle.write(f"{dataset_id}\n{msg}\n\n")
        print(f"Failed {len(failures)}/{num_cases} cases. See {fail_path}")
    else:
        print(f"All {num_cases} cases completed without recorded failures.")
