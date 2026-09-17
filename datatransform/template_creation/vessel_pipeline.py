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
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy

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
MIN_VOXELS_ACROSS_DIAMETER = 4.5
# Remesh edge <= this * local R so a small tube keeps ~11 triangles around.
CIRCUMFERENTIAL_EDGE_OVER_RADIUS = 0.55
# Fine enough for sub-mm branches. Sphere-stamping only writes voxels near the
# centerline, so this cap is no longer a 4-minute AABB rasterizer.
MAX_GRID_SIZE_HARD_CAP = 420
# Marching-cubes tubes: drop this fraction of the voxel staircase, but never
# collapse below MC_DECIMATE_MIN_POINTS (thin branches need samples for remesh).
MC_DECIMATE_REDUCTION = 0.50
MC_DECIMATE_MIN_POINTS = 20000
# Original STLs are 5–8× denser and carry zero-length edges. Voronoi on that
# tessellation misses thin outlets. Decimate a working copy for VMTK; keep the
# caller's mesh as GT for stretch/raycast. Do NOT key this off point count:
# large area-005 vessels (e.g. p460, 50k pts) look "dense" but already have
# ~0.28 mm edges. Originals have min edge ~0 and median ~0.08 mm.
SANITIZE_INPUT_REDUCTION = 0.85
SANITIZE_INPUT_MIN_POINTS = 20000
SANITIZE_MAX_MEDIAN_EDGE_MM = 0.22
DEGENERATE_EDGE_MM = 1e-5
# Extra triangles on a usage>2 edge are flaps if they are this small vs the two kept faces.
NM_FLAP_AREA_RATIO = 0.35
REMESH_MIN_EDGE_MM = 0.01
# VMTK uses this count twice (split/collapse loop AND final relocation).
# 4 is too few (degenerate tails, jagged rims). 6 matches 10 on tube shape,
# thin-branch diameter, and faceting; the 1% quality tail is only slightly worse.
REMESH_N_ITER = 6
# Extra connectivity flips after each remesh iter. 10 vs 20 produced identical
# meshes on p109/p505/p550; 20 just costs time.
REMESH_CONNECTIVITY_ITER = 10
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
# DecimatePro on originals can leave sub-0.2 mm pinholes that look like ostia.
# Real ICA ostia in this set are ≥ ~0.34 mm.
MIN_SEED_OPENING_RADIUS_MM = 0.20
MIN_OPENING_LOOP_POINTS = 6
# Outboard constant-R polyball + finite cylinder clip (hemoMesh removal endings).
# Interior Voronoi/remesh is unchanged; only the ostium neighbourhood is a pipe section.
OPENING_EXTENSION_LENGTH_FACTOR = 2.0
OPENING_EXTENSION_SPACING_FACTOR = 0.25
OPENING_CLIP_RADIUS_FACTOR = 1.5
OPENING_CLIP_HEIGHT_FACTOR = 5.0
OPENING_CLIP_INWARD_OVERLAP_MM = 0.05
OPENING_CLIP_INSET_STEP_MM = 0.1
OPENING_CLIP_INSET_MAX_MM = 0.5
# Disconnected flow-extension stubs can be >5% of the mesh (thin outlets, 5 mm
# extensions). Drop anything small relative to the largest component.
FRAGMENT_RELATIVE_TO_LARGEST = 0.15
# Match a capped hole to an expected ostium; leftover rims sit farther away.
SPURIOUS_OPENING_MATCH_FACTOR = 4.0
SPURIOUS_OPENING_MATCH_FLOOR_MM = 3.0
MIN_EDGE_LENGTH_MM = 1e-4
SLIVER_Q01_THRESHOLD = 0.3
FILTER_LOCATIONS = ["ICA pcom", "ICA oph", "ICA cav", "ICA bif"]


class TemplateQualityError(RuntimeError):
    """Raised when a case cannot be turned into a usable template."""

    def __init__(self, message, dataset_id=None):
        self.dataset_id = dataset_id
        text = str(message)
        if dataset_id:
            prefix = f"{dataset_id}: "
            if not text.startswith(prefix):
                text = prefix + text
        super().__init__(text)


def with_dataset_id(func):
    """Ensure every failure from a per-case entry point names the vessel."""

    def wrapper(dataset_id, *args, **kwargs):
        try:
            return func(dataset_id, *args, **kwargs)
        except TemplateQualityError as exc:
            if getattr(exc, "dataset_id", None) is None:
                raise TemplateQualityError(str(exc), dataset_id=dataset_id) from exc
            raise
        except Exception as exc:
            raise TemplateQualityError(f"{type(exc).__name__}: {exc}", dataset_id=dataset_id) from exc

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


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


def _poly_points(surface):
    """Float64 vertex array without a DeepCopy when the input is already vtkPolyData."""
    poly = surface if isinstance(surface, vtk.vtkPolyData) else to_vtk_poly(surface)
    if poly.GetPoints() is None or poly.GetNumberOfPoints() == 0:
        return poly, np.zeros((0, 3), dtype=np.float64)
    return poly, np.ascontiguousarray(vtk_to_numpy(poly.GetPoints().GetData()), dtype=np.float64)


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


def mesh_body_point(surface):
    """A point on the mesh, not the AABB center (which can sit in a siphon loop hole)."""
    vtk_poly = to_vtk_poly(surface)
    aabb = mesh_center(vtk_poly)
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(vtk_poly)
    locator.BuildLocator()
    pid = locator.FindClosestPoint(_vec3(aabb))
    return np.array(vtk_poly.GetPoint(pid), dtype=np.float64)


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


def apply_taubin_smoothing(
    surface_mesh,
    pass_band=0.1,
    n_iter=15,
    feature_angle=45.0,
    boundary_smoothing=True,
):
    """Volume-preserving Taubin smoothing (vtkWindowedSincPolyDataFilter).

    PassBand is in [0, 2]: 0 is strongest, 2 is none. VTK's default 0.1 is
    a strong low-pass (template path). Values near 1.5–2.0 barely filter.
    """
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(to_vtk_poly(surface_mesh))
    smoother.SetNumberOfIterations(n_iter)
    smoother.SetPassBand(pass_band)
    smoother.SetFeatureAngle(feature_angle)
    smoother.FeatureEdgeSmoothingOff()
    if boundary_smoothing:
        smoother.BoundarySmoothingOn()
    else:
        smoother.BoundarySmoothingOff()
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    return to_vtk_poly(smoother.GetOutput())


def _feature_edge_boundary_loops(surface):
    """Closed boundary polylines via vtkFeatureEdges + vtkStripper.

    vtkvmtkPolyDataBoundaryExtractor bails when a rim vertex has more than two
    boundary neighbours (a dangling ear glued to an ostium). Stripper still
    walks every simple cycle.
    """
    feat = vtk.vtkFeatureEdges()
    feat.SetInputData(to_vtk_poly(surface))
    feat.BoundaryEdgesOn()
    feat.FeatureEdgesOff()
    feat.NonManifoldEdgesOff()
    feat.ManifoldEdgesOff()
    feat.ColoringOff()
    feat.Update()
    strip = vtk.vtkStripper()
    strip.SetInputConnection(feat.GetOutputPort())
    strip.JoinContiguousSegmentsOn()
    if hasattr(strip, "SetMaximumLength"):
        strip.SetMaximumLength(1000000)
    strip.Update()
    return to_vtk_poly(strip.GetOutput())


def _loop_point_array(cell):
    n = cell.GetNumberOfPoints()
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)
    pts = np.array([cell.GetPoints().GetPoint(j) for j in range(n)], dtype=np.float64)
    if len(pts) >= 2 and float(np.linalg.norm(pts[0] - pts[-1])) < 1e-8:
        pts = pts[:-1]
    return pts


def _n_usable_boundary_loops(loops_poly, min_points=MIN_OPENING_LOOP_POINTS):
    n = 0
    for i in range(loops_poly.GetNumberOfCells()):
        if len(_loop_point_array(loops_poly.GetCell(i))) >= min_points:
            n += 1
    return n


def extract_boundary_loops(surface):
    vtk_poly = to_vtk_poly(surface)
    extractor = vtkvmtk.vtkvmtkPolyDataBoundaryExtractor()
    extractor.SetInputData(vtk_poly)
    extractor.Update()
    vmtk_out = to_vtk_poly(extractor.GetOutput())
    strip_out = _feature_edge_boundary_loops(vtk_poly)
    if _n_usable_boundary_loops(strip_out) > _n_usable_boundary_loops(vmtk_out):
        return strip_out
    return vmtk_out


def _profile_from_loop_points(pts_xyz, body, index):
    if len(pts_xyz) < MIN_OPENING_LOOP_POINTS:
        return None
    bary = pts_xyz.mean(axis=0)
    radius = float(np.mean(np.linalg.norm(pts_xyz - bary, axis=1)))
    normal = np.zeros(3, dtype=np.float64)
    n_loop = len(pts_xyz)
    if n_loop >= 3:
        v1 = pts_xyz[n_loop // 3] - pts_xyz[0]
        v2 = pts_xyz[(2 * n_loop) // 3] - pts_xyz[0]
        normal = _unit(np.cross(v1, v2))
        if float(np.linalg.norm(normal)) < 0.5:
            d = np.linalg.norm(pts_xyz - bary, axis=1)
            i = int(np.argmax(d))
            v1 = pts_xyz[i] - bary
            j = int(np.argmax(np.linalg.norm(np.cross(v1, pts_xyz - bary), axis=1)))
            normal = _unit(np.cross(v1, pts_xyz[j] - bary))
    if float(np.linalg.norm(normal)) >= 0.5 and float(np.dot(bary - body, normal)) < 0.0:
        normal = -normal
    return {
        "index": int(index),
        "barycenter": bary,
        "normal": normal,
        "radius": radius,
    }


def _profiles_from_boundary_loops(surface):
    loops = _feature_edge_boundary_loops(surface)
    body = mesh_body_point(surface)
    profiles = []
    for i in range(loops.GetNumberOfCells()):
        prof = _profile_from_loop_points(_loop_point_array(loops.GetCell(i)), body, i)
        if prof is not None:
            profiles.append(prof)
    return profiles


def _vmtk_boundary_profiles(surface):
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
        return []
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
    return profiles


def _keep_seed_profiles(profiles):
    min_r = MIN_SEED_OPENING_RADIUS_MM
    kept = [
        p
        for p in profiles
        if p["radius"] >= min_r and float(np.linalg.norm(p["normal"])) >= 0.5
    ]
    if len(kept) >= 2:
        n_drop = len(profiles) - len(kept)
        if n_drop:
            print(
                f"  Ignored {n_drop} pinhole/degenerate boundary loops "
                f"(r < {min_r:.3f} mm or missing normal)"
            )
        return kept
    return list(profiles)


def measure_open_profiles(surface):
    """Open-boundary loops with radius, barycenter, and (when available) outward normals."""
    vmtk_profiles = _vmtk_boundary_profiles(surface)
    profiles = _keep_seed_profiles(vmtk_profiles)
    if len(profiles) < 2:
        loop_profiles = _keep_seed_profiles(_profiles_from_boundary_loops(surface))
        if len(loop_profiles) > len(profiles):
            print(
                f"  Boundary extractor found {len(profiles)} opening(s); "
                f"using {len(loop_profiles)} feature-edge loops instead"
            )
            profiles = loop_profiles
    if not profiles:
        raise TemplateQualityError("No open boundary profiles found on vessel mesh.")
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
    centerlines.AppendEndPoints = 0
    centerlines.CapDisplacement = float(DEFAULT_CAP_DISPLACEMENT)
    centerlines.Execute()
    result = to_vtk_poly(centerlines.Centerlines)
    if result.GetNumberOfPoints() < 2 or result.GetNumberOfCells() < 1:
        raise TemplateQualityError("vmtkCenterlines returned an empty centerline.")
    return result


def centerline_looks_valid(centerline, reference_bounds, max_edge_mm=10.0, pad_mm=15.0):
    """Reject Voronoi spikes that leave the vessel (common on looping siphons)."""
    vtk_cl = to_vtk_poly(centerline)
    n_pts = vtk_cl.GetNumberOfPoints()
    if n_pts < 20:
        return False
    b = np.asarray(reference_bounds, dtype=np.float64).reshape(-1)
    lo = np.array([b[0] - pad_mm, b[2] - pad_mm, b[4] - pad_mm])
    hi = np.array([b[1] + pad_mm, b[3] + pad_mm, b[5] + pad_mm])
    n_out = 0
    for i in range(n_pts):
        p = np.asarray(vtk_cl.GetPoint(i), dtype=np.float64)
        if np.any(p < lo) or np.any(p > hi):
            n_out += 1
    if n_out > 0.2 * n_pts:
        return False
    vtk_cl.BuildCells()
    max_edge = 0.0
    max_path = 0.0
    n_ok = 0
    for ci in range(vtk_cl.GetNumberOfCells()):
        cell = vtk_cl.GetCell(ci)
        n = cell.GetNumberOfPoints()
        if n < 2:
            continue
        pts = np.array([vtk_cl.GetPoint(cell.GetPointId(j)) for j in range(n)], dtype=np.float64)
        segs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        if segs.size == 0:
            continue
        max_edge = max(max_edge, float(segs.max()))
        plen = float(segs.sum())
        max_path = max(max_path, plen)
        if plen >= 5.0:
            n_ok += 1
    if max_edge > max_edge_mm or n_ok < 1 or max_path < 5.0:
        return False
    return True


def _centerline_reaches_targets(centerline, n_targets):
    """vmtkCenterlines writes one polyline per inlet→outlet path."""
    n_cells = int(to_vtk_poly(centerline).GetNumberOfCells())
    return n_cells >= int(n_targets)


def extract_centerlines_for_tube(extended_vessel, smoothed_vessel, anatomical_profiles, extended_profiles):
    """Flow extensions help most cases; on looping siphons they can wreck Delaunay. Retry without them."""
    source_ext, target_ext = seed_points_from_profiles(extended_profiles)
    closed_extended = cap_surface(extended_vessel)
    centerline = extract_voronoi_centerlines(closed_extended, source_ext, target_ext)
    ref_bounds = smoothed_vessel.GetBounds()
    n_targets = len(target_ext)
    if centerline_looks_valid(centerline, ref_bounds) and _centerline_reaches_targets(centerline, n_targets):
        return centerline
    print(
        "  WARNING: centerline on the extended surface left the lumen "
        "or missed outlets "
        f"(cells={centerline.GetNumberOfCells()} targets={n_targets}). "
        "Retrying on the capped vessel without flow extensions."
    )
    source_anat, target_anat = seed_points_from_profiles(anatomical_profiles)
    try:
        closed_anat = cap_surface(smoothed_vessel)
    except TemplateQualityError as exc:
        if centerline_looks_valid(centerline, ref_bounds):
            print(f"  WARNING: anatomical cap failed ({exc}); keeping the extended-surface centerline.")
            return centerline
        raise
    retry = extract_voronoi_centerlines(closed_anat, source_anat, target_anat)
    if not centerline_looks_valid(retry, ref_bounds):
        raise TemplateQualityError(
            "Voronoi centerline left the vessel lumen; input openings were detected, "
            "but VMTK could not trace a path inside the tube."
        )
    if not _centerline_reaches_targets(retry, len(target_anat)):
        print(
            f"  WARNING: retry centerline still has {retry.GetNumberOfCells()} cells "
            f"for {len(target_anat)} outlets; thin branches may be missing."
        )
    return retry


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
    r_min = float(min(_misr_values(misr_array)))
    return max_r, r_min


def modeller_sample_grid(extents, grid_spacing, max_grid_size, r_min=None):
    """Isotropic voxel grid. Optionally refine so the thinnest branch has ~4.5 voxels across."""
    extents = [float(e) for e in extents]
    max_ext = max(extents)
    requested = max(32, int(max_grid_size))
    hard = int(MAX_GRID_SIZE_HARD_CAP)
    spacing = float(grid_spacing)
    dims = [max(32, int(np.ceil(e / spacing)) + 1) for e in extents]
    if max(dims) > requested:
        spacing = max_ext / float(requested - 1)

    if r_min is not None:
        r_min = max(float(r_min), MISR_FLOOR_MM)
        needed_spacing = (2.0 * r_min) / float(MIN_VOXELS_ACROSS_DIAMETER)
        if spacing > needed_spacing:
            needed_n = int(np.ceil(max_ext / needed_spacing)) + 1
            n_long = min(hard, max(requested, needed_n))
            spacing = max_ext / float(n_long - 1)

    dims = [max(32, int(round(e / spacing)) + 1) for e in extents]
    dims = [min(d, hard) for d in dims]
    return dims, spacing


def _centerline_xyz_r(vtk_cl):
    n = vtk_cl.GetNumberOfPoints()
    pts = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        pts[i] = vtk_cl.GetPoint(i)
    return pts, _misr_values(_misr_array_or_raise(vtk_cl))


def stamp_polyball_image(pts, radii, model_bounds, dims, spacing):
    """Narrow-band discrete-sphere polyball. Same implicit function as vtkvmtkPolyBall.

    VMTK's modeller evaluates every voxel against every sphere (empty AABB included).
    Stamping only writes the cube around each sphere, which is the tube's actual support.
    """
    nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
    spacing = float(spacing)
    origin = (
        float(model_bounds[0]),
        float(model_bounds[2]),
        float(model_bounds[4]),
    )
    field = np.full((nz, ny, nx), 1.0e6, dtype=np.float32)
    pts = np.asarray(pts, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    n_stamped = 0
    for p, r in zip(pts, radii):
        r = float(r)
        reach = r + 2.0 * spacing
        i0 = max(0, int(np.floor((p[0] - reach - origin[0]) / spacing)))
        i1 = min(nx, int(np.ceil((p[0] + reach - origin[0]) / spacing)) + 1)
        j0 = max(0, int(np.floor((p[1] - reach - origin[1]) / spacing)))
        j1 = min(ny, int(np.ceil((p[1] + reach - origin[1]) / spacing)) + 1)
        k0 = max(0, int(np.floor((p[2] - reach - origin[2]) / spacing)))
        k1 = min(nz, int(np.ceil((p[2] + reach - origin[2]) / spacing)) + 1)
        if i1 <= i0 or j1 <= j0 or k1 <= k0:
            continue
        xs = origin[0] + np.arange(i0, i1, dtype=np.float32) * spacing
        ys = origin[1] + np.arange(j0, j1, dtype=np.float32) * spacing
        zs = origin[2] + np.arange(k0, k1, dtype=np.float32) * spacing
        zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
        d2 = (xx - p[0]) ** 2 + (yy - p[1]) ** 2 + (zz - p[2]) ** 2
        sl = field[k0:k1, j0:j1, i0:i1]
        np.minimum(sl, (d2 - r * r).astype(np.float32, copy=False), out=sl)
        n_stamped += 1
    img = vtk.vtkImageData()
    img.SetDimensions(nx, ny, nz)
    img.SetOrigin(origin)
    img.SetSpacing(spacing, spacing, spacing)
    vtk_arr = numpy_to_vtk(np.ascontiguousarray(field.ravel(order="C")), deep=True)
    vtk_arr.SetName("ImageScalars")
    img.GetPointData().SetScalars(vtk_arr)
    print(
        f"  Narrow-band polyball stamp: {n_stamped} spheres into {nx}x{ny}x{nz} "
        f"({spacing:.4f} mm)"
    )
    return img


def _triangle_points_faces(surface):
    """Point coordinates and triangle vertex ids after a clean triangulate."""
    poly = clean_triangulate(surface)
    pts = np.ascontiguousarray(vtk_to_numpy(poly.GetPoints().GetData()), dtype=np.float64)
    polys = poly.GetPolys()
    offsets = vtk_to_numpy(polys.GetOffsetsArray())
    conn = vtk_to_numpy(polys.GetConnectivityArray())
    sizes = np.diff(offsets)
    if sizes.size == 0:
        faces = np.zeros((0, 3), dtype=np.int64)
    elif np.all(sizes == 3):
        faces = np.ascontiguousarray(conn.reshape(-1, 3), dtype=np.int64)
    else:
        tri = sizes == 3
        starts = offsets[:-1][tri]
        faces = np.ascontiguousarray(
            np.column_stack((conn[starts], conn[starts + 1], conn[starts + 2])),
            dtype=np.int64,
        )
    return poly, pts, faces


def _polydata_from_triangles(pts, faces):
    out = vtk.vtkPolyData()
    vtk_pts = vtk.vtkPoints()
    vtk_pts.SetData(numpy_to_vtk(np.ascontiguousarray(pts, dtype=np.float64), deep=True))
    out.SetPoints(vtk_pts)
    n = int(len(faces))
    if n == 0:
        return out
    offsets = np.arange(0, 3 * n + 1, 3, dtype=np.int64)
    conn = np.ascontiguousarray(faces.reshape(-1), dtype=np.int64)
    cells = vtk.vtkCellArray()
    cells.SetData(
        numpy_to_vtkIdTypeArray(offsets, deep=True),
        numpy_to_vtkIdTypeArray(conn, deep=True),
    )
    out.SetPolys(cells)
    return clean_triangulate(out)


def _triangle_areas(pts, faces):
    if faces.size == 0:
        return np.zeros(0, dtype=np.float64)
    a = pts[faces[:, 1]] - pts[faces[:, 0]]
    b = pts[faces[:, 2]] - pts[faces[:, 0]]
    return 0.5 * np.linalg.norm(np.cross(a, b), axis=1)


def _drop_duplicate_triangles(faces):
    if faces.size == 0:
        return faces, 0
    keys = np.sort(faces, axis=1)
    collapsed = (keys[:, 0] == keys[:, 1]) | (keys[:, 1] == keys[:, 2]) | (keys[:, 0] == keys[:, 2])
    _, first = np.unique(keys, axis=0, return_index=True)
    keep = np.zeros(len(faces), dtype=bool)
    keep[first] = True
    keep &= ~collapsed
    n_drop = int(len(faces) - keep.sum())
    return faces[keep], n_drop


def _drop_nonmanifold_flaps(pts, faces, area_ratio=NM_FLAP_AREA_RATIO):
    """On edges used by >2 triangles, drop extras that are much smaller than the two kept faces.

    Does not delete manifold slivers (usage==2); those must be remeshed, not punched out.
    """
    if faces.size == 0:
        return faces, 0
    n_dropped = 0
    for _ in range(8):
        areas = _triangle_areas(pts, faces)
        edges = {}
        for fi, (a, b, c) in enumerate(faces):
            for e in (
                (int(a), int(b)) if a <= b else (int(b), int(a)),
                (int(b), int(c)) if b <= c else (int(c), int(b)),
                (int(c), int(a)) if c <= a else (int(a), int(c)),
            ):
                edges.setdefault(e, []).append(fi)
        drop = set()
        for fis in edges.values():
            if len(fis) <= 2:
                continue
            ranked = sorted(fis, key=lambda fi: areas[fi], reverse=True)
            keep_min = min(areas[ranked[0]], areas[ranked[1]])
            for fi in ranked[2:]:
                if areas[fi] <= area_ratio * keep_min + 1e-18:
                    drop.add(fi)
        if not drop:
            break
        mask = np.ones(len(faces), dtype=bool)
        mask[list(drop)] = False
        faces = faces[mask]
        n_dropped += len(drop)
    return faces, n_dropped


def _nonmanifold_edge_count_from_faces(faces):
    if faces.size == 0:
        return 0
    counts = {}
    for a, b, c in faces:
        for e in (
            (int(a), int(b)) if a <= b else (int(b), int(a)),
            (int(b), int(c)) if b <= c else (int(c), int(b)),
            (int(c), int(a)) if c <= a else (int(a), int(c)),
        ):
            counts[e] = counts.get(e, 0) + 1
    return sum(1 for usage in counts.values() if usage > 2)


def repair_nonmanifold_triangles(surface):
    """Remove duplicate triangles and small non-manifold flaps. Leaves manifold slivers alone."""
    poly = clean_triangulate(surface)
    feat = vtk.vtkFeatureEdges()
    feat.SetInputData(poly)
    feat.BoundaryEdgesOff()
    feat.FeatureEdgesOff()
    feat.ManifoldEdgesOff()
    feat.NonManifoldEdgesOn()
    feat.ColoringOff()
    feat.Update()
    if feat.GetOutput().GetNumberOfCells() == 0:
        return poly, 0

    _, pts, faces = _triangle_points_faces(poly)
    n0 = len(faces)
    faces, n_dup = _drop_duplicate_triangles(faces)
    faces, n_flap = _drop_nonmanifold_flaps(pts, faces)
    n_nm = _nonmanifold_edge_count_from_faces(faces)
    if n_dup == 0 and n_flap == 0:
        return poly, n_nm
    out = _polydata_from_triangles(pts, faces)
    print(
        f"  Topology repair: dropped {n_dup} duplicate and {n_flap} flap triangles "
        f"({n0} -> {len(faces)}); remaining non-manifold edges={n_nm}"
    )
    return out, n_nm


def _triangle_edge_lengths(pts, faces):
    if faces.size == 0:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(
        (
            np.linalg.norm(pts[faces[:, 1]] - pts[faces[:, 0]], axis=1),
            np.linalg.norm(pts[faces[:, 2]] - pts[faces[:, 1]], axis=1),
            np.linalg.norm(pts[faces[:, 0]] - pts[faces[:, 2]], axis=1),
        )
    )


def tessellation_looks_original(surface):
    """True for raw STLs (tiny/short edges), false for area-005 (~0.28 mm)."""
    _poly, pts, faces = _triangle_points_faces(surface)
    edges = _triangle_edge_lengths(pts, faces)
    if edges.size == 0:
        return False
    return float(np.median(edges)) < SANITIZE_MAX_MEDIAN_EDGE_MM


def drop_degenerate_triangles(surface, min_edge=DEGENERATE_EDGE_MM):
    """Remove zero-length / zero-area triangles that wreck vtkDelaunay3D."""
    poly, pts, faces = _triangle_points_faces(surface)
    if faces.size == 0:
        return poly
    e01 = np.linalg.norm(pts[faces[:, 1]] - pts[faces[:, 0]], axis=1)
    e12 = np.linalg.norm(pts[faces[:, 2]] - pts[faces[:, 1]], axis=1)
    e20 = np.linalg.norm(pts[faces[:, 0]] - pts[faces[:, 2]], axis=1)
    keep = (e01 >= min_edge) & (e12 >= min_edge) & (e20 >= min_edge)
    n_drop = int((~keep).sum())
    if n_drop == 0:
        return poly
    out = _polydata_from_triangles(pts, faces[keep])
    print(f"  Dropped {n_drop} degenerate triangles (min edge < {min_edge:g} mm)")
    return out


def drop_boundary_ear_triangles(surface, max_passes=16):
    """Drop triangles with 2+ boundary edges (fins glued onto an ostium rim).

    Those ears give rim vertices four boundary neighbours, and VMTK's
    vtkvmtkPolyDataBoundaryExtractor then reports only one opening.
    """
    poly, pts, faces = _triangle_points_faces(surface)
    if faces.size == 0:
        return poly
    n_dropped = 0
    for _ in range(int(max_passes)):
        edges = np.concatenate(
            (
                np.sort(faces[:, [0, 1]], axis=1),
                np.sort(faces[:, [1, 2]], axis=1),
                np.sort(faces[:, [2, 0]], axis=1),
            ),
            axis=0,
        )
        uniq, counts = np.unique(edges, axis=0, return_counts=True)
        usage = {(int(a), int(b)): int(c) for (a, b), c in zip(uniq, counts)}
        n_boundary = np.zeros(len(faces), dtype=np.int32)
        for fi, (a, b, c) in enumerate(faces):
            for u, v in ((a, b), (b, c), (c, a)):
                key = (int(u), int(v)) if u <= v else (int(v), int(u))
                if usage.get(key, 0) == 1:
                    n_boundary[fi] += 1
        keep = n_boundary < 2
        n_drop = int((~keep).sum())
        if n_drop == 0:
            break
        faces = faces[keep]
        n_dropped += n_drop
    if n_dropped == 0:
        return poly
    out = _polydata_from_triangles(pts, faces)
    print(f"  Dropped {n_dropped} boundary-ear triangles (2+ free edges)")
    return out


def sanitize_vessel_for_vmtk(
    surface,
    target_reduction=SANITIZE_INPUT_REDUCTION,
    min_points=SANITIZE_INPUT_MIN_POINTS,
):
    """Working copy for centerlines/extensions. Originals stay as raycast GT.

    Original STLs are 5–8× denser than area-005 and include zero-length edges.
    Voronoi on that tessellation misses thin outlets. DecimatePro can punch
    pinholes, so those are filled before opening detection.
    """
    poly = clean_triangulate(surface)
    n0 = poly.GetNumberOfPoints()
    poly = drop_degenerate_triangles(poly)
    poly = drop_boundary_ear_triangles(poly)
    n_mid = poly.GetNumberOfPoints()
    if tessellation_looks_original(poly):
        poly = decimate_dense_mc(
            poly, target_reduction=target_reduction, min_points=min_points
        )
    if poly.GetNumberOfPoints() < n_mid:
        # DecimatePro can punch sub-mm pinholes; fill those but not real ostia (~0.3 mm+).
        poly = fill_pinholes(poly, hole_size=0.25)
        poly, _n_reg = drop_tiny_islands(poly)
    n1 = poly.GetNumberOfPoints()
    if n1 != n0:
        print(f"  Sanitized vessel for VMTK: {n0} -> {n1} points")
    return poly


def decimate_dense_mc(
    surface,
    target_reduction=None,
    min_points=None,
):
    """Collapse over-tessellated surfaces with vtkDecimatePro. Topology stays.

    Reduction is a fraction of points to remove. Remaining count is never
    forced below min_points, so small tubes are not flattened.
    """
    poly = to_vtk_poly(surface)
    n = poly.GetNumberOfPoints()
    if target_reduction is None:
        target_reduction = MC_DECIMATE_REDUCTION
        min_points = MC_DECIMATE_MIN_POINTS if min_points is None else min_points
    reduction = float(target_reduction)
    floor = int(min_points) if min_points is not None else 0
    if n < 4 or reduction <= 0.0:
        return poly
    if floor > 0 and n <= floor:
        return poly
    target_n = int(round(n * (1.0 - reduction)))
    if floor > 0:
        target_n = max(floor, target_n)
    if target_n >= n:
        return poly
    reduction = 1.0 - float(target_n) / float(n)
    if reduction < 0.05:
        return poly
    dec = vtk.vtkDecimatePro()
    dec.SetInputData(poly)
    dec.SetTargetReduction(reduction)
    dec.PreserveTopologyOn()
    dec.SplittingOff()
    if hasattr(dec, "BoundaryVertexDeletionOff"):
        dec.BoundaryVertexDeletionOff()
    # High feature angle: MC staircasing is not anatomy. 45 deg leaves fins that
    # share edges with the wall (non-manifold) even with PreserveTopology on.
    dec.SetFeatureAngle(90.0)
    dec.Update()
    out = clean_triangulate(dec.GetOutput())
    print(
        f"  Decimated surface {n} -> {out.GetNumberOfPoints()} points "
        f"(target reduction {100.0 * reduction:.0f}%, PreserveTopology, openings kept)"
    )
    return out


def generate_base_surface(
    branched_centerline,
    grid_spacing=DEFAULT_GRID_SPACING,
    max_grid_size=DEFAULT_MAX_GRID_SIZE,
    reference_bounds=None,
    profiles=None,
    extension_length=DEFAULT_EXTENSION_LENGTH,
    extra_spheres=None,
):
    """Parent tube via a narrow-band polyball image + marching cubes."""
    if not isinstance(branched_centerline, vtk.vtkPolyData):
        branched_centerline = to_vtk_poly(branched_centerline)
    misr_array = _misr_array_or_raise(branched_centerline)
    if reference_bounds is None:
        reference_bounds = pv.wrap(branched_centerline).bounds
    r_cap = parent_tube_misr_cap(profiles, reference_bounds)
    max_r, r_min = clamp_misr_for_parent_tube(misr_array, r_cap)
    vtk_cl = to_vtk_poly(branched_centerline)
    pts, radii = _centerline_xyz_r(vtk_cl)
    if extra_spheres is not None:
        extra_pts, extra_r = extra_spheres
        extra_pts = np.asarray(extra_pts, dtype=np.float64).reshape(-1, 3)
        extra_r = np.asarray(extra_r, dtype=np.float64).reshape(-1)
        if extra_pts.size and extra_r.size:
            pts = np.vstack((pts, extra_pts))
            radii = np.concatenate((radii, extra_r))
            print(f"  Added {len(extra_r)} constant-R opening spheres for pipe-section cuts")

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
    dims, spacing = modeller_sample_grid(extents, grid_spacing, max_grid_size, r_min=r_min)
    print(
        f"  CenterlineModeller grid dimensions: {dims} "
        f"(isotropic spacing ~{spacing:.4f} mm, r_min={r_min:.3f} mm, "
        f"{(2.0 * r_min) / max(spacing, 1e-9):.1f} voxels across smallest diameter)"
    )
    image = stamp_polyball_image(pts, radii, model_bounds, dims, spacing)
    mc = vmtkscripts.vmtkMarchingCubes()
    mc.Image = image
    mc.Level = 0.0
    mc.Connectivity = 1
    mc.Execute()
    raw = to_vtk_poly(mc.Surface)
    kept = keep_largest_region(raw)
    pre_decimate = kept
    kept = decimate_dense_mc(kept)
    kept, n_nm = repair_nonmanifold_triangles(kept)
    if n_nm > 0 and kept.GetNumberOfPoints() < pre_decimate.GetNumberOfPoints():
        print("  Decimate left non-manifold edges; keeping the full marching-cubes surface")
        kept, n_nm = repair_nonmanifold_triangles(pre_decimate)
    if n_nm > 0:
        raise TemplateQualityError(
            f"Parent-tube surface has {n_nm} non-manifold edges after marching cubes"
        )
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


def _plane_has_sign_change(surface, origin, normal, pts=None):
    """True if the (triangle) surface meets the plane.

    A planar triangle meets a plane iff its vertices are not all strictly on one
    side, so the vertex-sign range is exact. False positives on disjoint
    components only waste a cheap vtkCutter call.
    """
    origin = np.asarray(origin, dtype=np.float64)
    normal = _unit(normal)
    if pts is None:
        _poly, pts = _poly_points(surface)
    if pts.size == 0:
        return False
    signs = (pts - origin) @ normal
    return bool(signs.min() <= 0.0 <= signs.max())


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


def _cut_loops_near_profile(surface, origin, normal, bary):
    """vtkCutter loops on a plane — used to score origins without TopologicalSeamFilter."""
    plane = vtk.vtkPlane()
    _set_vec3(plane.SetOrigin, origin)
    _set_vec3(plane.SetNormal, normal)
    cutter = vtk.vtkCutter()
    cutter.SetInputData(surface)
    cutter.SetCutFunction(plane)
    cutter.Update()
    cut = cutter.GetOutput()
    if cut.GetNumberOfPoints() < 6:
        return None
    strips = vtk.vtkStripper()
    strips.SetInputData(cut)
    strips.JoinContiguousSegmentsOn()
    strips.Update()
    loops = strips.GetOutput()
    best = None
    best_dist = None
    for i in range(loops.GetNumberOfCells()):
        cell = loops.GetCell(i)
        n = cell.GetNumberOfPoints()
        if n < MIN_OPENING_LOOP_POINTS:
            continue
        pts = np.array([cell.GetPoints().GetPoint(j) for j in range(n)], dtype=np.float64)
        center = pts.mean(axis=0)
        radius = float(np.mean(np.linalg.norm(pts - center, axis=1)))
        dist = float(np.linalg.norm(center - bary))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = {"radius": radius, "center": center, "n_points": n, "dist": dist}
    return best


def clip_one_profile(surface, profile, body_point, search_mm):
    """Local seam clip at one opening. Score planes with vtkCutter, then seam-clip once.

    On a looping siphon the anatomical plane can cut the tube twice. The first
    intersection is often a thin mid-vessel hole; a later fill then seals the real end.
    """
    outward = _unit(profile["normal"])
    plane_normal = -outward
    origins, _inward = _clip_origin_candidates(surface, profile, search_mm)
    if not origins:
        return surface, False

    r_gt = float(profile["radius"])
    min_r = max(MIN_OPENING_RADIUS_MM, 0.35 * r_gt)
    max_r = max(4.0 * r_gt, r_gt + 2.5)
    bary = np.asarray(profile["barycenter"], dtype=np.float64)

    unique = []
    for cand in origins:
        arr = np.asarray(cand, dtype=np.float64)
        if any(np.linalg.norm(arr - kept) < 0.15 for kept in unique):
            continue
        unique.append(arr)

    _poly, pts = _poly_points(surface)
    scored = []
    for origin in unique:
        if not _plane_has_sign_change(_poly, origin, plane_normal, pts=pts):
            continue
        loop = _cut_loops_near_profile(surface, origin, plane_normal, bary)
        if loop is None:
            continue
        if loop["radius"] < min_r or loop["radius"] > max_r:
            continue
        if loop["n_points"] < MIN_OPENING_LOOP_POINTS:
            continue
        score = abs(loop["radius"] - r_gt) + 0.15 * loop["dist"]
        scored.append((score, origin, loop))
    if not scored:
        print(
            f"  [Uncap] Profile {profile['index']} found no loop matching r={r_gt:.3f} mm "
            f"(accepted band {min_r:.3f}–{max_r:.3f} mm). Skipping this end."
        )
        return surface, False
    scored.sort(key=lambda t: t[0])
    origin = scored[0][1]
    loop = scored[0][2]

    plane = vtk.vtkPlane()
    _set_vec3(plane.SetOrigin, origin)
    _set_vec3(plane.SetNormal, plane_normal)

    seam_filter = vtkvmtk.vtkvmtkTopologicalSeamFilter()
    seam_filter.SetInputData(to_vtk_poly(surface))
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
    openings = inspect_openings(candidate)
    if not openings:
        return surface, False
    matched = min(
        openings,
        key=lambda op: float(np.linalg.norm(np.asarray(op["center"]) - bary)),
    )
    r_open = float(matched.get("radius", 0.0))
    if r_open < min_r or r_open > max_r or matched.get("n_points", 0) < MIN_OPENING_LOOP_POINTS:
        print(
            f"  [Uncap] Profile {profile['index']} seam clip loop r={r_open:.3f} mm "
            f"outside {min_r:.3f}–{max_r:.3f} mm. Skipping this end."
        )
        return surface, False
    print(
        f"  [Uncap] Profile {profile['index']} opened r={r_open:.3f} mm "
        f"(GT r={r_gt:.3f} mm, cutter r={loop['radius']:.3f} mm)"
    )
    return candidate, True


def _n_boundary_loops(surface):
    return int(extract_boundary_loops(surface).GetNumberOfCells())


def _centerline_tangent_at_id(vtk_cl, pid):
    vtk_cl.BuildLinks()
    cell_ids = vtk.vtkIdList()
    vtk_cl.GetPointCells(int(pid), cell_ids)
    if cell_ids.GetNumberOfIds() == 0:
        return None
    cell = vtk_cl.GetCell(cell_ids.GetId(0))
    n = cell.GetNumberOfPoints()
    idx = None
    for j in range(n):
        if int(cell.GetPointId(j)) == int(pid):
            idx = j
            break
    if idx is None or n < 2:
        return None
    p = np.asarray(vtk_cl.GetPoint(int(pid)), dtype=np.float64)
    if idx == 0:
        q = np.asarray(vtk_cl.GetPoint(cell.GetPointId(1)), dtype=np.float64)
        tangent = p - q
    elif idx == n - 1:
        q = np.asarray(vtk_cl.GetPoint(cell.GetPointId(n - 2)), dtype=np.float64)
        tangent = p - q
    else:
        a = np.asarray(vtk_cl.GetPoint(cell.GetPointId(idx - 1)), dtype=np.float64)
        b = np.asarray(vtk_cl.GetPoint(cell.GetPointId(idx + 1)), dtype=np.float64)
        tangent = b - a
    if float(np.linalg.norm(tangent)) < 1e-12:
        return None
    return _unit(tangent)


def opening_clip_frames(centerline, profiles):
    """Anatomical ostium origin, outward centerline tangent, local MISR."""
    vtk_cl = to_vtk_poly(centerline)
    if vtk_cl.GetNumberOfPoints() < 2:
        raise TemplateQualityError("Centerline too short to build opening clip frames.")
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(vtk_cl)
    locator.BuildLocator()
    misr = vtk_cl.GetPointData().GetArray("MaximumInscribedSphereRadius")
    frames = []
    for profile in profiles:
        origin = np.asarray(profile["barycenter"], dtype=np.float64)
        profile_n = _unit(profile["normal"])
        profile_r = max(float(profile["radius"]), MIN_OPENING_RADIUS_MM)
        pid = locator.FindClosestPoint(_vec3(origin))
        closest = np.asarray(vtk_cl.GetPoint(pid), dtype=np.float64)
        if np.linalg.norm(closest - origin) > 8.0 * max(profile_r, 0.5):
            tangent = profile_n
            radius = profile_r
        else:
            tangent = _centerline_tangent_at_id(vtk_cl, pid)
            if tangent is None or float(np.linalg.norm(tangent)) < 0.5:
                tangent = profile_n
            if float(np.dot(tangent, profile_n)) < 0.0:
                tangent = -tangent
            radius = profile_r
            if misr is not None:
                radius = max(float(misr.GetComponent(pid, 0)), 0.5 * profile_r, MIN_OPENING_RADIUS_MM)
        frames.append((origin, _unit(tangent), float(radius)))
    return frames


def extra_opening_spheres(centerline, profiles):
    """Constant-R balls along the outward tangent, starting just past each ostium."""
    extra_pts = []
    extra_r = []
    for origin, outward, radius in opening_clip_frames(centerline, profiles):
        radius = max(float(radius), 1e-3)
        length = OPENING_EXTENSION_LENGTH_FACTOR * radius
        spacing = max(OPENING_EXTENSION_SPACING_FACTOR * radius, 0.1)
        n_steps = max(int(np.ceil(length / spacing)), 2)
        for i in range(1, n_steps + 1):
            extra_pts.append(origin + i * spacing * outward)
            extra_r.append(radius)
    if not extra_pts:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    return np.asarray(extra_pts, dtype=np.float64), np.asarray(extra_r, dtype=np.float64)


def _opening_clip_radius(radius):
    return max(float(radius) * OPENING_CLIP_RADIUS_FACTOR, float(radius) + 0.2)


def _opening_clip_height(radius, extension_length=None):
    """Finite cylinder length along the outward tangent.

    ``7R`` is enough for the short polyball stubs on the parent tube. GT remesh
    adds a fixed-length flow extension; on thin outlets ``7R`` is shorter than
    that extension and the far stub survives as a leftover fragment.
    """
    r = float(radius)
    by_radius = max(r * OPENING_CLIP_HEIGHT_FACTOR, 2.0) + r * OPENING_EXTENSION_LENGTH_FACTOR
    if extension_length is None:
        return by_radius
    return max(by_radius, float(extension_length) + 2.0 * r)


def _outboard_cap_implicit(origin, outward, radius, extension_length=None):
    origin = np.asarray(origin, dtype=np.float64)
    outward = _unit(outward)
    clip_radius = _opening_clip_radius(radius)
    height = _opening_clip_height(radius, extension_length=extension_length)
    p_in = origin - OPENING_CLIP_INWARD_OVERLAP_MM * outward
    p_far = origin + height * outward

    cylinder = vtk.vtkCylinder()
    _set_vec3(cylinder.SetCenter, origin)
    _set_vec3(cylinder.SetAxis, outward)
    cylinder.SetRadius(clip_radius)

    plane_near = vtk.vtkPlane()
    _set_vec3(plane_near.SetOrigin, p_in)
    _set_vec3(plane_near.SetNormal, -outward)

    plane_far = vtk.vtkPlane()
    _set_vec3(plane_far.SetOrigin, p_far)
    _set_vec3(plane_far.SetNormal, outward)

    region = vtk.vtkImplicitBoolean()
    region.SetOperationTypeToIntersection()
    region.AddFunction(cylinder)
    region.AddFunction(plane_near)
    region.AddFunction(plane_far)
    return region


def _drop_small_fragments(surface, min_fraction=0.05):
    """Keep the main vessel; drop cut-off caps and leftover extension stubs."""
    mesh = pv.wrap(to_vtk_poly(surface))
    if mesh.n_points == 0:
        return to_vtk_poly(surface)
    connected = mesh.connectivity(extraction_mode="all")
    if "RegionId" not in connected.point_data:
        return to_vtk_poly(surface)
    region_ids, counts = np.unique(connected.point_data["RegionId"], return_counts=True)
    largest = int(counts.max())
    threshold = max(
        int(min_fraction * connected.n_points),
        int(FRAGMENT_RELATIVE_TO_LARGEST * largest),
        3,
    )
    keep_ids = region_ids[counts >= threshold]
    if keep_ids.size == 0:
        return keep_largest_region(surface)
    mask = np.isin(connected.point_data["RegionId"], keep_ids)
    kept = connected.extract_points(mask, adjacent_cells=True)
    if not isinstance(kept, pv.PolyData):
        kept = pv.wrap(kept).extract_surface(algorithm="dataset_surface")
    return to_vtk_poly(kept.triangulate().clean())


def _delete_outboard_leftover(surface, origin, outward, radius, outboard_mm=0.0):
    poly, pts, faces = _triangle_points_faces(surface)
    if faces.size == 0:
        return poly
    origin = np.asarray(origin, dtype=np.float64)
    outward = _unit(outward)
    centroids = pts[faces].mean(axis=1)
    rel = centroids - origin
    proj = rel @ outward
    radial = np.linalg.norm(rel - np.outer(proj, outward), axis=1)
    bad = (proj > outboard_mm) & (radial < _opening_clip_radius(radius))
    if not np.any(bad):
        return poly
    return _polydata_from_triangles(pts, faces[~bad])


def _clip_opening_cap_locally(surface, origin, outward, radius, extension_length=None):
    """Delete the outboard stub of one opening with a bounded cylinder."""
    region = _outboard_cap_implicit(
        origin, outward, radius, extension_length=extension_length
    )
    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(to_vtk_poly(surface))
    clipper.SetClipFunction(region)
    clipper.InsideOutOff()
    clipper.GenerateClippedOutputOff()
    clipper.Update()
    clipped = clean_triangulate(clipper.GetOutput())
    if clipped.GetNumberOfPoints() == 0:
        return to_vtk_poly(surface)
    return _delete_outboard_leftover(clipped, origin, outward, radius)


def clip_one_opening_pipe_section(
    surface, origin, outward, radius, _body_point, extension_length=None
):
    """Open one ostium with a pipe-section cut; inset slightly if the cutter misses."""
    origin0 = np.asarray(origin, dtype=np.float64)
    outward = _unit(outward)
    radius = max(float(radius), 1e-3)
    before = _n_boundary_loops(surface)
    n_prev = surface.GetNumberOfPoints()
    inset = 0.0
    while inset <= OPENING_CLIP_INSET_MAX_MM + 1e-12:
        origin_i = origin0 - inset * outward
        clipped = _clip_opening_cap_locally(
            surface, origin_i, outward, radius, extension_length=extension_length
        )
        clipped = _drop_small_fragments(clipped)
        n_cand = clipped.GetNumberOfPoints()
        if n_cand < 50 or n_cand < 0.45 * n_prev:
            inset += OPENING_CLIP_INSET_STEP_MM
            continue
        loops_after = _n_boundary_loops(clipped)
        near = any(
            float(np.linalg.norm(np.asarray(op["center"]) - origin_i)) < 2.0 * radius
            for op in inspect_openings(clipped)
        )
        if loops_after > before or (near and n_cand < n_prev):
            if inset > 0:
                print(f"  [Uncap] Pipe-section clip inset {inset:.1f} mm to create a hole")
            return clipped, True
        inset += OPENING_CLIP_INSET_STEP_MM
    return surface, False


def _cap_surface_with_entity_ids(surface, displacement):
    capper = vtkvmtk.vtkvmtkCapPolyData()
    capper.SetInputData(to_vtk_poly(surface))
    capper.SetDisplacement(float(displacement))
    capper.SetInPlaneDisplacement(0.0)
    capper.SetCellEntityIdsArrayName("CellEntityIds")
    capper.Update()
    capped = vtk.vtkPolyData()
    capped.DeepCopy(capper.GetOutput())
    return capper, capped


def _polydata_from_kept_cells(poly, keep_cell_ids):
    keep = vtk.vtkIdList()
    for cid in keep_cell_ids:
        keep.InsertNextId(int(cid))
    extractor = vtk.vtkExtractCells()
    extractor.SetInputData(poly)
    extractor.SetCellList(keep)
    extractor.Update()
    geom = vtk.vtkGeometryFilter()
    geom.SetInputConnection(extractor.GetOutputPort())
    geom.Update()
    return clean_triangulate(geom.GetOutput())


def remove_spurious_openings(surface, profiles):
    """Fill leftover rims / wall tears that are not the expected ostia.

    Caps every boundary loop, then deletes the cap that matches each anatomical
    profile so true ostia stay open. Extra loops (clip leftovers, nicked walls)
    remain capped. Does nothing when there are no extra openings.
    """
    poly = clean_triangulate(surface)
    openings = inspect_openings(poly)
    n_expected = len(profiles)
    if n_expected < 1 or len(openings) <= n_expected:
        return poly, 0

    capper = None
    capped = None
    for displacement in (0.0, DEFAULT_CAP_DISPLACEMENT):
        capper, capped = _cap_surface_with_entity_ids(poly, displacement)
        n_left = extract_boundary_loops(capped).GetNumberOfCells()
        if n_left == 0:
            break
    if capped is None or extract_boundary_loops(capped).GetNumberOfCells() != 0:
        print("  WARNING: could not cap leftover openings; leaving them in place")
        return poly, 0

    ids = capped.GetCellData().GetArray("CellEntityIds")
    center_ids = capper.GetCapCenterIds()
    if ids is None or center_ids is None or center_ids.GetNumberOfIds() == 0:
        print("  WARNING: capper produced no CellEntityIds; skipping leftover fill")
        return poly, 0

    offset = int(capper.GetCellEntityIdOffset())
    n_caps = int(center_ids.GetNumberOfIds())
    cap_centers = []
    cap_eids = []
    for i in range(n_caps):
        cap_centers.append(np.array(capped.GetPoint(center_ids.GetId(i)), dtype=np.float64))
        cap_eids.append(offset + 1 + i)

    remaining = list(range(n_caps))
    reopen = []
    for profile in profiles:
        bary = np.asarray(profile["barycenter"], dtype=np.float64)
        r_p = max(float(profile["radius"]), MIN_OPENING_RADIUS_MM)
        best = None
        best_d = None
        for i in remaining:
            dist = float(np.linalg.norm(cap_centers[i] - bary))
            if best_d is None or dist < best_d:
                best_d = dist
                best = i
        max_d = max(SPURIOUS_OPENING_MATCH_FACTOR * r_p, SPURIOUS_OPENING_MATCH_FLOOR_MM)
        if best is None or best_d > max_d:
            print(
                f"  WARNING: no cap within {max_d:.2f} mm of profile {profile['index']} "
                f"(best d={best_d if best_d is not None else float('inf'):.2f} mm); "
                "skipping leftover fill"
            )
            return poly, 0
        reopen.append(cap_eids[best])
        remaining.remove(best)

    n_filled = len(remaining)
    if n_filled == 0:
        return poly, 0

    reopen_set = set(reopen)
    keep_cells = [
        ci
        for ci in range(capped.GetNumberOfCells())
        if int(ids.GetComponent(ci, 0)) not in reopen_set
    ]
    if not keep_cells:
        return poly, 0
    filled = _polydata_from_kept_cells(capped, keep_cells)
    filled, _n_nm = repair_nonmanifold_triangles(filled)
    filled = drop_degenerate_triangles(filled, min_edge=MIN_EDGE_LENGTH_MM)
    filled, _n_reg = drop_tiny_islands(filled)
    n_after = len(inspect_openings(filled))
    print(
        f"  Filled {n_filled} leftover opening(s) "
        f"({len(openings)} -> {n_after} loops; expected {n_expected})"
    )
    if n_after < n_expected or n_after < 2:
        print("  WARNING: leftover fill closed a true ostium; keeping pre-fill surface")
        return poly, 0
    filled = strip_all_arrays(filled)
    filled = recompute_point_normals(filled, auto_orient=False)
    return filled, n_filled


def clip_flow_extensions_and_uncap(
    base_surface,
    profiles,
    extension_length=DEFAULT_EXTENSION_LENGTH,
    centerline=None,
):
    current = to_vtk_poly(base_surface)
    body_pt = mesh_body_point(current)
    n_clipped = 0
    frames = opening_clip_frames(centerline, profiles) if centerline is not None else None
    for i, profile in enumerate(profiles):
        search_mm = float(extension_length) + 3.0 * max(float(profile["radius"]), 0.5)
        ok = False
        if frames is not None:
            origin, outward, radius = frames[i]
            current, ok = clip_one_opening_pipe_section(
                current,
                origin,
                outward,
                radius,
                body_pt,
                extension_length=extension_length,
            )
            if ok:
                print(
                    f"  [Uncap] Profile {profile['index']} pipe-section cut "
                    f"r={radius:.3f} mm at {np.round(origin, 2)}"
                )
        if not ok:
            current, ok = clip_one_profile(current, profile, body_pt, search_mm)
            if ok:
                print(
                    f"  [Uncap] Profile {profile['index']} fell back to anatomical plane clip"
                )
        if ok:
            n_clipped += 1
            body_pt = mesh_body_point(current)
    print(f"  Uncap: clipped {n_clipped}/{len(profiles)} openings")
    current = clean_triangulate(current)
    current, n_nm = repair_nonmanifold_triangles(current)
    if n_nm > 0:
        raise TemplateQualityError(
            f"Uncapped parent tube has {n_nm} non-manifold edges; remesh would amplify them"
        )
    current, _n_regions = drop_tiny_islands(current)
    current, n_filled = remove_spurious_openings(current, profiles)
    post = inspect_openings(current)
    print(
        "  Openings after uncap/pinhole-fill: "
        + ", ".join(f"r={op['radius']:.3f}mm n={op['n_points']}" for op in post)
        + (f" (filled {n_filled} leftover)" if n_filled else "")
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
    """R_template from the closest point on the centerline polyline, not the nearest vertex."""
    vtk_template = to_vtk_poly(template_mesh)
    vtk_cl = to_vtk_poly(branched_centerline)
    n_pts = vtk_template.GetNumberOfPoints()
    r_template = np.full(n_pts, R_TEMPLATE_FLOOR_MM, dtype=np.float64)
    if vtk_cl.GetNumberOfPoints() == 0 or n_pts == 0:
        return r_template

    misr_arr = vtk_cl.GetPointData().GetArray("MaximumInscribedSphereRadius")
    if misr_arr is None:
        return r_template
    misr = np.ascontiguousarray(vtk_to_numpy(misr_arr), dtype=np.float64)
    query = np.ascontiguousarray(vtk_to_numpy(vtk_template.GetPoints().GetData()), dtype=np.float64)

    vtk_cl.BuildCells()
    locator = vtk.vtkCellLocator()
    locator.SetDataSet(vtk_cl)
    locator.BuildLocator()
    closest = [0.0, 0.0, 0.0]
    cell_id = vtk.mutable(0)
    sub_id = vtk.mutable(0)
    dist2 = vtk.mutable(0.0)
    for i, p in enumerate(query):
        locator.FindClosestPoint(_vec3(p), closest, cell_id, sub_id, dist2)
        cell = vtk_cl.GetCell(int(cell_id.get()))
        n = cell.GetNumberOfPoints()
        if n < 2:
            pid = cell.GetPointId(0) if n == 1 else 0
            r_val = float(misr[pid]) if pid < misr.size else R_TEMPLATE_FLOOR_MM
            r_template[i] = max(R_TEMPLATE_FLOOR_MM, r_val)
            continue
        sid = int(sub_id.get())
        sid = max(0, min(sid, n - 2))
        i0 = cell.GetPointId(sid)
        i1 = cell.GetPointId(sid + 1)
        p0 = np.asarray(vtk_cl.GetPoint(i0), dtype=np.float64)
        p1 = np.asarray(vtk_cl.GetPoint(i1), dtype=np.float64)
        seg = p1 - p0
        denom = float(np.dot(seg, seg))
        if denom < 1e-18:
            t = 0.0
        else:
            t = float(np.clip(np.dot(np.asarray(closest, dtype=np.float64) - p0, seg) / denom, 0.0, 1.0))
        r_val = (1.0 - t) * float(misr[i0]) + t * float(misr[i1])
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

    gt_normals_filter = vtk.vtkPolyDataNormals()
    gt_normals_filter.SetInputData(to_vtk_poly(ground_truth_mesh))
    gt_normals_filter.ComputeCellNormalsOn()
    gt_normals_filter.ComputePointNormalsOff()
    gt_normals_filter.ConsistencyOn()
    gt_normals_filter.SplittingOff()
    gt_normals_filter.Update()
    gt_mesh_with_normals = to_vtk_poly(gt_normals_filter.GetOutput())
    gt_cell_normals_vtk = gt_mesh_with_normals.GetCellData().GetNormals()
    gt_cell_normals = (
        np.ascontiguousarray(vtk_to_numpy(gt_cell_normals_vtk), dtype=np.float64)
        if gt_cell_normals_vtk is not None
        else None
    )

    locator = vtk.vtkCellLocator()
    locator.SetDataSet(gt_mesh_with_normals)
    locator.BuildLocator()

    template_pts = np.ascontiguousarray(
        vtk_to_numpy(template_mesh_with_normals.GetPoints().GetData()), dtype=np.float64
    )
    nrm_vtk = template_mesh_with_normals.GetPointData().GetNormals()
    if nrm_vtk is None:
        return np.zeros(template_pts.shape[0], dtype=np.float64)
    template_normals = np.ascontiguousarray(vtk_to_numpy(nrm_vtk), dtype=np.float64)
    lens = np.linalg.norm(template_normals, axis=1, keepdims=True)
    lens = np.maximum(lens, 1e-12)
    outward = -template_normals / lens

    n_pts = template_pts.shape[0]
    distances = np.zeros(n_pts, dtype=np.float64)
    t = vtk.mutable(0.0)
    x = [0.0, 0.0, 0.0]
    pcoords = [0.0, 0.0, 0.0]
    sub_id = vtk.mutable(0)
    cell_id = vtk.mutable(0)
    r_arr = None if r_template is None else np.asarray(r_template, dtype=np.float64)

    for i in range(n_pts):
        p = template_pts[i]
        n = outward[i]
        r_local = 1.0 if r_arr is None else float(r_arr[i])
        p0 = (float(p[0]), float(p[1]), float(p[2]))
        p_inward = (float(p[0] - n[0] * 1.5), float(p[1] - n[1] * 1.5), float(p[2] - n[2] * 1.5))
        hit_inward = locator.IntersectWithLine(p0, p_inward, tol, t, x, pcoords, sub_id, cell_id)
        if hit_inward:
            d_inward = float(np.sqrt((x[0] - p[0]) ** 2 + (x[1] - p[1]) ** 2 + (x[2] - p[2]) ** 2))
            if d_inward < 0.4:
                distances[i] = 0.0
                continue

        p_end = (
            float(p[0] + n[0] * max_ray_length),
            float(p[1] + n[1] * max_ray_length),
            float(p[2] + n[2] * max_ray_length),
        )
        hit = locator.IntersectWithLine(p0, p_end, tol, t, x, pcoords, sub_id, cell_id)
        if hit:
            d = float(np.sqrt((x[0] - p[0]) ** 2 + (x[1] - p[1]) ** 2 + (x[2] - p[2]) ** 2))
            if 0.10 < d <= (3.5 * r_local):
                cid = int(cell_id.get())
                if gt_cell_normals is not None and 0 <= cid < len(gt_cell_normals):
                    if float(np.dot(n, gt_cell_normals[cid])) > 0.2:
                        distances[i] = d
                else:
                    distances[i] = d
    return distances


def build_target_edge_array(template_mesh, distances, r_template, base_edge=0.50, min_edge=0.01):
    """Stretch densifies aneurysms; local radius caps edge length so thin tubes stay round."""
    vtk_poly = to_vtk_poly(template_mesh)
    n_pts = vtk_poly.GetNumberOfPoints()
    r = np.maximum(R_TEMPLATE_FLOOR_MM, np.asarray(r_template, dtype=np.float64))
    stretch_factors = 1.0 + (distances / r)
    radius_limited = np.minimum(base_edge, np.maximum(min_edge, CIRCUMFERENTIAL_EDGE_OVER_RADIUS * r))
    target_edge_lengths = np.maximum(min_edge, radius_limited / (stretch_factors ** 1.5))

    vtk_target_array = vtk.vtkDoubleArray()
    vtk_target_array.SetName("TargetEdgeLength")
    vtk_target_array.SetNumberOfTuples(n_pts)
    for i in range(n_pts):
        vtk_target_array.SetValue(i, float(target_edge_lengths[i]))
    vtk_poly.GetPointData().AddArray(vtk_target_array)
    return vtk_poly, target_edge_lengths, stretch_factors


def remesh_surface_adaptively(
    open_surface_with_array,
    edge_array_name="TargetEdgeLength",
    n_iter=REMESH_N_ITER,
    connectivity_iter=REMESH_CONNECTIVITY_ITER,
):
    remesher = vmtkscripts.vmtkSurfaceRemeshing()
    remesher.Surface = to_vtk_poly(open_surface_with_array)
    remesher.ElementSizeMode = "edgelengtharray"
    remesher.TargetEdgeLengthArrayName = edge_array_name
    remesher.PreserveBoundaryEdges = 1
    remesher.NumberOfIterations = int(n_iter)
    remesher.NumberOfConnectivityOptimizationIterations = int(connectivity_iter)
    remesher.MinEdgeLength = float(REMESH_MIN_EDGE_MM)
    remesher.Execute()
    return to_vtk_poly(remesher.Surface)


def remesh_surface_isotropically(
    open_surface,
    target_edge_length=0.5,
    n_iter=REMESH_N_ITER,
    connectivity_iter=REMESH_CONNECTIVITY_ITER,
):
    remesher = vmtkscripts.vmtkSurfaceRemeshing()
    remesher.Surface = to_vtk_poly(open_surface)
    remesher.ElementSizeMode = "edgelength"
    remesher.TargetEdgeLength = float(target_edge_length)
    remesher.PreserveBoundaryEdges = 1
    remesher.NumberOfIterations = int(n_iter)
    remesher.NumberOfConnectivityOptimizationIterations = int(connectivity_iter)
    remesher.MinEdgeLength = float(REMESH_MIN_EDGE_MM)
    remesher.Execute()
    return to_vtk_poly(remesher.Surface)


def uniform_edge_length_for_profiles(profiles, target_edge_length):
    """Keep enough triangles around the smallest opening so it cannot flatten."""
    if not profiles:
        return float(target_edge_length)
    r_min = min(float(p["radius"]) for p in profiles)
    return float(min(target_edge_length, max(0.15, CIRCUMFERENTIAL_EDGE_OVER_RADIUS * r_min)))


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
    poly, pts, faces = _triangle_points_faces(surface)
    n_tri = int(len(faces))
    if n_tri == 0 or pts.size == 0:
        return {
            "n_triangles": 0,
            "n_boundary_edges": 0,
            "n_nonmanifold": 0,
            "min_edge": 0.0,
            "median_q01": None,
            "frac_sliver": None,
        }
    e01 = np.linalg.norm(pts[faces[:, 1]] - pts[faces[:, 0]], axis=1)
    e12 = np.linalg.norm(pts[faces[:, 2]] - pts[faces[:, 1]], axis=1)
    e20 = np.linalg.norm(pts[faces[:, 0]] - pts[faces[:, 2]], axis=1)
    min_edge = float(np.min(np.minimum(np.minimum(e01, e12), e20)))
    edges = np.concatenate(
        (np.sort(faces[:, [0, 1]], axis=1), np.sort(faces[:, [1, 2]], axis=1), np.sort(faces[:, [2, 0]], axis=1)),
        axis=0,
    )
    _uniq, counts = np.unique(edges, axis=0, return_counts=True)
    n_boundary = int(np.sum(counts == 1))
    n_nonmanifold = int(np.sum(counts > 2))

    med_q01 = None
    frac_sliver = None
    try:
        quality = vtk.vtkMeshQuality()
        quality.SetInputData(poly)
        quality.SetTriangleQualityMeasureToRadiusRatio()
        quality.Update()
        arr = quality.GetOutput().GetCellData().GetArray("Quality")
        if arr is not None and arr.GetNumberOfTuples() > 0:
            rr = np.ascontiguousarray(vtk_to_numpy(arr), dtype=np.float64)
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
    cleaned, _n_nm = repair_nonmanifold_triangles(clean_triangulate(surface))
    cleaned = fill_pinholes(cleaned)
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


def assert_template_quality(surface, context="template"):
    vtk_poly = to_vtk_poly(surface)
    n_regions = count_connected_regions(vtk_poly)
    openings = inspect_openings(vtk_poly)
    issues = []
    if n_regions != 1:
        issues.append(f"{n_regions} connected components")
    n_open = len(openings)
    if n_open < 2:
        issues.append(f"{n_open} openings (need at least inlet and one outlet)")
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
    dataset_id=None,
):
    """Shared path: smooth -> extend -> cap -> centerline -> polyball tube -> uncap at anatomy."""
    print("Step 1: Applying Taubin surface smoothing...")
    work_vessel = sanitize_vessel_for_vmtk(vessel_mesh)
    smoothed_vessel = apply_taubin_smoothing(work_vessel)

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

    print("Step 3: Extracting Voronoi centerline and MISR...")
    centerline = extract_centerlines_for_tube(
        extended_vessel, smoothed_vessel, anatomical_profiles, extended_profiles
    )

    print(f"Step 4: Spline resampling ({sample_spacing} mm) and trajectory smoothing...")
    resampled = resample_centerline(centerline, sample_spacing=sample_spacing)
    smooth_centerline = smooth_centerline_preserve_misr(resampled)

    print("Step 5: Extracting branches...")
    branched_centerline = extract_branches(smooth_centerline)

    print("Step 5b: Constant-radius polyball stubs past anatomical openings...")
    extra_pts, extra_r = extra_opening_spheres(branched_centerline, anatomical_profiles)

    print("Step 6: Generating multi-branch base surface (vmtkCenterlineModeller)...")
    base_surface = generate_base_surface(
        branched_centerline,
        grid_spacing=grid_spacing,
        max_grid_size=max_grid_size,
        reference_bounds=smoothed_vessel.GetBounds(),
        profiles=anatomical_profiles,
        extension_length=extension_length,
        extra_spheres=(extra_pts, extra_r),
    )

    print("Step 7: Uncapping open boundaries with pipe-section cuts...")
    open_base_surface, n_clipped = clip_flow_extensions_and_uncap(
        base_surface,
        anatomical_profiles,
        extension_length=extension_length,
        centerline=branched_centerline,
    )
    print(f"  -> Open base surface points: {open_base_surface.GetNumberOfPoints()}")
    n_in = len(anatomical_profiles)
    if n_clipped < n_in:
        print(
            f"  WARNING: uncap opened {n_clipped}/{n_in} anatomical ends; "
            "a branch may be missing from the parent tube."
        )
    if n_clipped < 2:
        raise TemplateQualityError(
            f"Parent-tube uncap opened {n_clipped}/{n_in} ends "
            f"(input vessel has {n_in} openings; this is a reconstructed-tube miss, not a sealed input).",
            dataset_id=dataset_id,
        )
    return {
        "smoothed_vessel": smoothed_vessel,
        "anatomical_profiles": anatomical_profiles,
        "branched_centerline": branched_centerline,
        "open_base_surface": open_base_surface,
        "n_clipped": n_clipped,
    }


@with_dataset_id
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
        dataset_id=dataset_id,
    )
    open_base_surface = built["open_base_surface"]
    branched_centerline = built["branched_centerline"]
    anatomical_profiles = built["anatomical_profiles"]

    print("Step 8a: Computing local tube radius and raycasting stretch vs ground truth...")
    r_template = compute_template_local_radii(open_base_surface, branched_centerline)
    stretch_distances = compute_raycast_stretch_distances(
        open_base_surface, vessel_mesh, r_template=r_template
    )
    min_edge = REMESH_MIN_EDGE_MM
    print(
        f"Step 8b: Building stretch metric k = 1 + d / R_template "
        f"(Base={target_edge_length} mm, Min={min_edge:.2f} mm)..."
    )
    surface_with_array, edge_lengths, stretch_factors = build_target_edge_array(
        open_base_surface,
        stretch_distances,
        r_template,
        base_edge=target_edge_length,
        min_edge=min_edge,
    )
    thin = np.asarray(r_template) < 0.7
    if np.any(thin):
        print(
            f"  Thin-branch target edges (R<0.7 mm): "
            f"min/median/max={edge_lengths[thin].min():.3f}/"
            f"{np.median(edge_lengths[thin]):.3f}/{edge_lengths[thin].max():.3f} mm "
            f"n={int(thin.sum())}"
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
    openings = assert_template_quality(final_surface, context=dataset_id)

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
        f"(pipe-section clipped {int(built['n_clipped'])}; "
        f"{len(anatomical_profiles)} anatomical profiles)"
    )
    return out_file


@with_dataset_id
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
        dataset_id=dataset_id,
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
    openings = assert_template_quality(final_surface, context=dataset_id)

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
        f"(pipe-section clipped {int(built['n_clipped'])}; "
        f"{len(anatomical_profiles)} anatomical profiles)"
    )
    return out_file


def compute_centerline_from_mesh(
    vessel_mesh,
    extension_length=DEFAULT_EXTENSION_LENGTH,
    sample_spacing=DEFAULT_SAMPLE_SPACING,
):
    """Voronoi centerline for an in-memory surface. Same steps as centerline_creation.py.

    Does not read or write files. Callers that need a ``.vtp`` should use
    ``process_centerline_dataset``.
    """
    from batch_run_log import set_step

    print("Step 1: Preparing surface (drop degenerates/ears, then Taubin)...")
    set_step("1_prepare_surface")
    work = drop_boundary_ear_triangles(drop_degenerate_triangles(clean_triangulate(vessel_mesh)))
    smoothed_vessel = apply_taubin_smoothing(work)
    print("Step 1b: Detecting anatomical inlet/outlet boundaries...")
    set_step("1b_anatomical_openings")
    anatomical_profiles = measure_open_profiles(smoothed_vessel)
    log_profiles(anatomical_profiles, label="Anatomical")
    seed_points_from_profiles(anatomical_profiles)

    print("Step 2: Adding flow extensions on the open surface...")
    set_step("2_flow_extensions")
    extended_vessel = add_flow_extensions(smoothed_vessel, extension_length=extension_length)
    extended_profiles = measure_open_profiles(extended_vessel)
    log_profiles(extended_profiles, label="Extended")

    print("Step 3: Extracting Voronoi centerline and MISR...")
    set_step("3_voronoi_centerline")
    centerline = extract_centerlines_for_tube(
        extended_vessel, smoothed_vessel, anatomical_profiles, extended_profiles
    )
    print(f"Step 4: Spline resampling ({sample_spacing} mm) and trajectory smoothing...")
    set_step("4_resample_smooth")
    resampled = resample_centerline(centerline, sample_spacing=sample_spacing)
    smooth_centerline = smooth_centerline_preserve_misr(resampled)
    print("Step 5: Extracting branches...")
    set_step("5_extract_branches")
    branched_centerline = extract_branches(smooth_centerline)
    print("Step 6: Trimming flow-extension ends at anatomical profile planes...")
    set_step("6_clip_extensions")
    final_centerline = clip_centerline_at_profiles(
        branched_centerline, anatomical_profiles, extension_length=extension_length
    )
    if final_centerline.GetNumberOfCells() < 1:
        raise TemplateQualityError("Clipped centerline has no cells.")
    return final_centerline


@with_dataset_id
def process_centerline_dataset(
    dataset_id,
    v_file,
    output_dir,
    extension_length=DEFAULT_EXTENSION_LENGTH,
    sample_spacing=DEFAULT_SAMPLE_SPACING,
):
    print(f"\n=========================================\nProcessing Centerline Case: {dataset_id}")
    vessel_mesh = pv.read(v_file)
    final_centerline = compute_centerline_from_mesh(
        vessel_mesh,
        extension_length=extension_length,
        sample_spacing=sample_spacing,
    )
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{dataset_id}.vtp")
    save_polydata(final_centerline, out_file)
    print(f"Successfully saved centerline to: {out_file}")
    return out_file


VESSEL_FILE_EXTENSIONS = (".vtp", ".stl", ".vtk", ".ply")


def resolve_vessel_file(vessel_dir, dataset_id, explicit=None):
    """Find a vessel mesh. Originals are STL; remeshed copies are VTP."""
    if explicit:
        return explicit
    for ext in VESSEL_FILE_EXTENSIONS:
        path = os.path.join(vessel_dir, f"{dataset_id}{ext}")
        if os.path.exists(path):
            return path
    return os.path.join(vessel_dir, f"{dataset_id}.vtp")


def load_meshes_from_folder(vessel_dir, limit=None, case_ids=None):
    """Every mesh in ``vessel_dir`` is a case; the filename stem is the id."""
    if not os.path.isdir(vessel_dir):
        raise FileNotFoundError(f"Vessel directory not found: {vessel_dir}")
    wanted = {str(x) for x in case_ids} if case_ids else None
    ext_rank = {ext: i for i, ext in enumerate(VESSEL_FILE_EXTENSIONS)}
    by_id = {}
    for name in os.listdir(vessel_dir):
        stem, ext = os.path.splitext(name)
        ext = ext.lower()
        if ext not in ext_rank:
            continue
        if wanted is not None and stem not in wanted:
            continue
        path = os.path.join(vessel_dir, name)
        if not os.path.isfile(path):
            continue
        prev = by_id.get(stem)
        if prev is None:
            by_id[stem] = path
            continue
        prev_ext = os.path.splitext(prev)[1].lower()
        if ext_rank[ext] < ext_rank[prev_ext]:
            by_id[stem] = path
    valid = [(dataset_id, by_id[dataset_id]) for dataset_id in sorted(by_id)]
    if limit is not None:
        valid = valid[: int(limit)]
    return valid


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
        v_file = resolve_vessel_file(vessel_dir, dataset_id)
        if os.path.exists(v_file):
            valid.append((dataset_id, v_file))
    if limit is not None:
        valid = valid[: int(limit)]
    return valid


def add_shared_cli_args(parser, default_output_dir, default_workers, include_remesh_grid=True):
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV_PATH, help="Path to clinical.csv")
    parser.add_argument("--vessel-dir", type=str, default=DEFAULT_VESSEL_DIR, help="Directory of input vessel meshes (.vtp or .stl)")
    parser.add_argument(
        "--from-folder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Process every mesh in --vessel-dir instead of filtering by clinical.csv",
    )
    parser.add_argument("--output-dir", type=str, default=default_output_dir, help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of dataset meshes to process")
    parser.add_argument("--workers", type=int, default=default_workers, help="Number of parallel worker processes")
    parser.add_argument("--case", type=str, default=None, help="Process a single dataset id")
    parser.add_argument("--vessel-file", type=str, default=None, help="Explicit input mesh for --case (.vtp or .stl)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip cases whose output .vtp already exists")
    parser.add_argument("--cases", type=str, nargs="*", default=None, help="Optional subset of dataset ids")
    parser.add_argument("--extension-length", type=float, default=DEFAULT_EXTENSION_LENGTH, help="Flow extension length in mm")
    parser.add_argument("--sample-spacing", type=float, default=DEFAULT_SAMPLE_SPACING, help="Centerline resampling spacing in mm")
    if include_remesh_grid:
        parser.add_argument("--target-edge-length", type=float, default=DEFAULT_TARGET_EDGE_LENGTH, help="Base / uniform target edge length in mm")
        parser.add_argument("--grid-spacing", type=float, default=DEFAULT_GRID_SPACING, help="Requested modeller voxel size in mm")
        parser.add_argument("--max-grid-size", type=int, default=DEFAULT_MAX_GRID_SIZE, help="Max voxels along the longest axis (spacing stays isotropic)")
    return parser


def run_batch(script_path, process_one, args, extra_cli_flags, on_worker_result=None):
    os.makedirs(args.output_dir, exist_ok=True)
    if args.case:
        v_file = resolve_vessel_file(args.vessel_dir, args.case, explicit=args.vessel_file)
        if not os.path.exists(v_file):
            raise FileNotFoundError(f"Vessel file not found: {v_file}")
        print(f"Running single case: {args.case}")
        print(f"Vessel file: {v_file}")
        process_one(args.case, v_file, args)
        return

    if getattr(args, "from_folder", False):
        valid_datasets = load_meshes_from_folder(
            args.vessel_dir, limit=args.limit, case_ids=args.cases
        )
    else:
        valid_datasets = load_valid_datasets(
            args.csv, args.vessel_dir, limit=args.limit, case_ids=args.cases
        )
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
    if getattr(args, "from_folder", False):
        print(f"Input folder (all meshes): {args.vessel_dir}")
    else:
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
                if on_worker_result is not None:
                    try:
                        on_worker_result(dataset_id, proc.returncode, out)
                    except Exception as log_exc:
                        print(f"  WARNING: on_worker_result failed for {dataset_id}: {log_exc}")
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
