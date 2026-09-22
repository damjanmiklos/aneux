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
# VTK_NUMBER_OF_THREADS only caps the old vtkMultiThreader. VTK 9 runs its
# filters on vtkSMPTools, which is built here against TBB and sizes itself
# from the machine (32 threads), so with 20 workers the partitioning varies
# with load and the arithmetic comes out slightly differently each time. That
# is enough to move a cutter radius and flip a verdict: 20 identical runs of
# p398 split 3 passed / 17 failed. This is the variable that pins the pool.
os.environ.setdefault("VTK_SMP_MAX_THREADS", "1")

import argparse
import json
import shutil
import subprocess
import tempfile
import threading
import time
from collections import Counter
from queue import Empty, Queue

import numpy as np
import pandas as pd
import pyvista as pv
import vtk
from tqdm import tqdm
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy

# The env var above only lands if nothing imported vtk first, and a worker that
# reaches this module through another import does exactly that. Saying it again
# here costs nothing and makes the pool size independent of import order.
vtk.vtkSMPTools.Initialize(1)

try:
    from vmtk import vmtkscripts
    from vmtk import vtkvmtk
except ImportError as exc:
    raise ImportError(
        "Required package 'vmtk' is not installed. "
        "Install VMTK Python bindings so that `from vmtk import vmtkscripts` succeeds "
        "(e.g. conda install -c vmtk vmtk)."
    ) from exc

try:
    from stretch_raycast import compute as _stretch_raycast_c
except ImportError:
    try:
        from .stretch_raycast import compute as _stretch_raycast_c
    except ImportError:
        _stretch_raycast_c = None

if _stretch_raycast_c is None:
    print("WARNING: stretch_raycast C extension not loaded; variable remesh uses the Python raycast loop")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from aneux_paths import (
    CLEANDATA_ORIGINAL_CENTERLINE,
    CSV_PATH as DEFAULT_CSV_PATH,
    VESSELS_AREA005 as DEFAULT_VESSEL_DIR,
)

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
# Variable remesh only (not remeshing.py, not sanitize). After uncap, collapse
# the MC staircase toward this count so VMTK is not chewing a 20k wall. 8000
# is below the median shipped template (~12.6k points) and above the p5 (~6.7k),
# and still leaves ~11 tris around a R<0.7 mm branch via CIRCUMFERENTIAL_EDGE_OVER_RADIUS.
VAR_MC_DECIMATE_MIN_POINTS = 8000
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
# Once trim_extension_patches has cut the tubes back to the ostium plane there is
# only a short collar left to remove, so the cutter no longer needs to be as long
# as the flow extension. A long cutter is what swallowed mid-vessel segments of
# tortuous siphons and lost a branch (area ratio < 0.88).
OPENING_CLIP_TRIMMED_HEIGHT_FACTOR = 2.0
OPENING_CLIP_TRIMMED_HEIGHT_MIN_MM = 0.75
# A pipe-section cut on a pre-trimmed surface removes a collar, never a branch.
CLIP_MAX_AREA_LOSS_FRACTION = 0.12
# Disconnected flow-extension stubs can be >5% of the mesh (thin outlets, 5 mm
# extensions). Drop anything small relative to the largest component.
FRAGMENT_RELATIVE_TO_LARGEST = 0.15
# Match a capped hole to an expected ostium; leftover rims sit farther away.
SPURIOUS_OPENING_MATCH_FACTOR = 4.0
SPURIOUS_OPENING_MATCH_FLOOR_MM = 3.0
MIN_EDGE_LENGTH_MM = 1e-4
# Weld vertices closer than this. 1e-3 mm is 0.7% of the 0.15 mm GT target edge,
# i.e. far below any anatomical feature, but 10x above the MIN_EDGE_LENGTH_MM gate
# so a single pass cannot leave an edge that still trips it.
WELD_TOLERANCE_MM = 1e-3
WELD_MAX_PASSES = 4
# Sweeps of the tiny-edge collapse. One sweep folds every candidate edge whose
# endpoints are not already spoken for, so it does most of the work; the rest
# are for edges a neighbour blocked. It returns the moment a sweep finds
# nothing left to fold, so the ceiling is rarely reached.
WELD_MAX_SWEEPS = 10
# Boundary loops smaller than this on an input surface are wall punctures, not
# ostia. Flow extensions grow tubes out of them and wreck the uncap, so they are
# patched before anything else runs.
WALL_PINHOLE_RADIUS_MM = MIN_SEED_OPENING_RADIUS_MM
# An audit of the 629 outputs of the 2026-09-17 run put the smallest genuine
# ostium at r=0.205 mm while the leftover rims measured 0.128-0.199 mm. Radius
# alone therefore separates anatomy from debris by 2.5%, which is no margin at
# all, so any loop sitting at an anatomical profile is protected by name instead.
PROFILE_PROTECT_RADIUS_FACTOR = 1.5
PROFILE_PROTECT_MIN_MM = 0.5
# Proximity alone over-protects: clip debris sits right next to the ostium it
# was cut from, and a 4-point, 4-micron rim 0.3 mm from a real opening was being
# shielded as if it were anatomy. A loop only counts as the ostium if it is also
# the right size for it.
PROFILE_PROTECT_MIN_RADIUS_FRACTION = 0.5
# An extension cell this far outboard of its ostium plane is leftover tube.
EXTENSION_PLANE_TOL_MM = 1e-3
# VMTK writes float32 point coordinates, so a vessel a few tens of mm across comes
# back displaced by ~1e-7 mm. Still 100x under WELD_TOLERANCE_MM, so no two
# distinct vertices can be confused.
ORIGINAL_MATCH_TOL_MM = 1e-5
# A flat end cap is planar to machine precision; an anatomical wall is not.
CAP_PLANARITY_MM = 0.02
CAP_PATCH_ANGLE_DEG = 5.0
CAP_MAX_AREA_FRACTION = 0.15
CAP_MAX_COUNT = 32
CAP_MIN_DISC_FILL = 0.4
CAP_MIN_TRIANGLES = 6
# A cap meets the wall at a rim; a flat piece of wall flows smoothly into its
# neighbours. Calibrated on USFD_0052, where the caps sit at 6.8-7.5 deg and
# every equally planar wall patch is under 3.2 deg -- the rim is shallow because
# the AneuX originals were remeshed and smoothed after they were capped.
CAP_MIN_RIM_ANGLE_DEG = 5.0
# Below this a "cap" is a wall artefact; leaving a sub-millimetre branch capped
# is much safer than tearing a hole in the wall.
CAP_MIN_RADIUS_MM = 0.5
# vmtkFlowExtensions advances a boundary by ~its mean rim edge per layer. A
# degenerate rim asks for millions of layers and hundreds of millions of cells.
MAX_FLOW_EXTENSION_LAYERS = 4000
MAX_EXTENSION_CELL_GROWTH = 20.0
SLIVER_Q01_THRESHOLD = 0.3
# A healthy GT remesh finishes well inside this; anything longer is a runaway
# VMTK filter, and the 2026-09-17 run lost ten hours to nine such cases.
DEFAULT_CASE_TIMEOUT_S = 5400.0
# Peak resident set of one worker on a large vessel. 25 workers x this exceeded
# 32 GB and produced the vtkGenericDataArray allocation failures.
# Measured, not guessed: a worker's peak working set on this dataset came in at
# 0.4 GB, and the multi-GB cases that motivated the original 2.5 GB figure were
# the flow-extension blow-ups, which now get refused before they allocate.
DEFAULT_WORKER_MEMORY_GB = 1.25
# What to leave the desktop and this session. The cap is taken from memory that
# is actually free, because the installed total says nothing when a browser and
# an editor are already holding 28 GB of it.
HOST_RESERVE_GB = 3.0
# Accept some paging rather than collapsing to a handful of workers when the
# machine is momentarily busy: measured peak per worker is 0.16-0.43 GB, so this
# floor is roughly a dozen of them, and Windows reclaims the difference from
# idle applications. Sized from the peak_memory_gb column of a real run.
MIN_POOL_MEMORY_GB = 16.0
# Leave threads for the desktop and this session.
HOST_RESERVE_THREADS = 4
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


def _as_poly(mesh):
    """vtkPolyData view without a DeepCopy when the input is already polydata."""
    if mesh is None:
        return vtk.vtkPolyData()
    if isinstance(mesh, vtk.vtkPolyData):
        return mesh
    return to_vtk_poly(mesh)


def _vtk_c_address(obj, class_name):
    text = obj.GetAddressAsString(class_name)
    return int(str(text).split("=")[-1], 16)


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


def count_boundary_regions(surface):
    """How many openings the surface has, by connectivity on its boundary edges.

    This is topology and nothing else: a connected rim is one region however
    ragged it is, and two rims are never one. Both loop extractors can disagree
    with it -- they walk the rim and can break the walk part way -- so it is the
    referee rather than a third opinion.
    """
    feat = vtk.vtkFeatureEdges()
    feat.SetInputData(to_vtk_poly(surface))
    feat.BoundaryEdgesOn()
    feat.FeatureEdgesOff()
    feat.NonManifoldEdgesOff()
    feat.ManifoldEdgesOff()
    feat.ColoringOff()
    feat.Update()
    if feat.GetOutput().GetNumberOfCells() == 0:
        return 0
    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(feat.GetOutput())
    conn.SetExtractionModeToAllRegions()
    conn.Update()
    return int(conn.GetNumberOfExtractedRegions())


def extract_boundary_loops(surface):
    """Ordered boundary polylines, one per opening.

    Two extractors, because each fails where the other holds: the VMTK one
    bails when a rim vertex has more than two boundary neighbours, and the
    stripper walks only simple cycles. The old rule was to take whichever
    reported more loops, which is wrong in the one direction that matters --
    a rim reported twice is an opening that does not exist. On p489 the VMTK
    extractor split five rims into nine and won on count, and the case shipped
    claiming nine openings against five anatomical profiles; p129 went out at
    six against four the same way.

    So connectivity decides. Whichever extractor agrees with the number of
    boundary regions is the one telling the truth about how many openings there
    are; if neither does, the one that over-counts least is kept, since an
    invented opening is worse than a missed rim walk.
    """
    vtk_poly = to_vtk_poly(surface)
    extractor = vtkvmtk.vtkvmtkPolyDataBoundaryExtractor()
    extractor.SetInputData(vtk_poly)
    extractor.Update()
    vmtk_out = to_vtk_poly(extractor.GetOutput())
    strip_out = _feature_edge_boundary_loops(vtk_poly)

    expected = count_boundary_regions(vtk_poly)
    n_vmtk = _n_usable_boundary_loops(vmtk_out)
    n_strip = _n_usable_boundary_loops(strip_out)
    if n_vmtk == expected:
        return vmtk_out
    if n_strip == expected:
        return strip_out
    return vmtk_out if abs(n_vmtk - expected) <= abs(n_strip - expected) else strip_out


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
    """Profiles built from the arbitrated loop walk rather than the stripper alone.

    extract_boundary_loops runs both extractors and keeps the one that agrees
    with count_boundary_regions, so it is right where either one alone is not.
    """
    loops = extract_boundary_loops(surface)
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
    """Open-boundary loops with radius, barycenter, and (when available) outward normals.

    vtkvmtkBoundaryReferenceSystems is preferred when it works, because it
    gives a real boundary normal per rim instead of one guessed from three
    loop points. It walks the rim with the extractor that bails on a vertex
    carrying more than two boundary neighbours -- and when it bails it does not
    fail, it returns the rims it managed. p376 keep 1 has six clean openings on
    its 234k-point original, r=2.32 down to 0.83 mm with 117 to 330 rim points
    each, and this reported two of them. Four ostia were gone before a single
    centerline seed was placed and the case died in the Voronoi trace with
    "the centerline left the lumen", which named the wrong step entirely.

    count_boundary_regions is the referee. It is connectivity on the boundary
    edges and nothing else, so a walk that breaks part way cannot fool it. When
    the reference systems report fewer rims than the surface has, the profiles
    are rebuilt from extract_boundary_loops -- which arbitrates both extractors
    against that same count -- and kept only if they come closer to it.
    """
    vtk_poly = to_vtk_poly(surface)
    expected = count_boundary_regions(vtk_poly)
    # Compared before the seed filter: that filter drops pinholes on purpose,
    # so the count it leaves is legitimately below the number of rims.
    vmtk_profiles = _vmtk_boundary_profiles(vtk_poly)
    chosen = vmtk_profiles
    if expected > 0 and len(vmtk_profiles) < expected:
        loop_profiles = _profiles_from_boundary_loops(vtk_poly)
        if abs(len(loop_profiles) - expected) < abs(len(vmtk_profiles) - expected):
            print(
                f"  Boundary reference systems walked {len(vmtk_profiles)} of "
                f"{expected} rim(s); using {len(loop_profiles)} extracted loops instead"
            )
            chosen = loop_profiles
    profiles = _keep_seed_profiles(chosen)
    if len(profiles) < 2 and chosen is vmtk_profiles:
        loop_profiles = _keep_seed_profiles(_profiles_from_boundary_loops(vtk_poly))
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


def _as_ostium_frame(frame):
    """Canonical in-memory ostium frame: origin (3,), unit normal (3,), radius float."""
    origin = np.ascontiguousarray(frame["origin"], dtype=np.float64).reshape(3)
    normal = np.ascontiguousarray(_unit(frame["normal"]), dtype=np.float64)
    return {
        "origin": origin,
        "normal": normal,
        "radius": float(frame["radius"]),
    }


def save_ostium_frames(path, frames):
    """Write ostium cut frames as ``{stem}.ostium_frames.npz``.

    In-memory contract: ``list[dict]`` with ``origin`` (3,) float64, unit
    ``normal`` (3,), and ``radius`` float. On disk: keys ``origin`` (K, 3),
    ``normal`` (K, 3), ``radius`` (K,) float64.
    """
    parsed = [_as_ostium_frame(fr) for fr in list(frames)]
    if parsed:
        origin = np.ascontiguousarray([fr["origin"] for fr in parsed], dtype=np.float64)
        normal = np.ascontiguousarray([fr["normal"] for fr in parsed], dtype=np.float64)
        radius = np.ascontiguousarray([fr["radius"] for fr in parsed], dtype=np.float64)
    else:
        origin = np.zeros((0, 3), dtype=np.float64)
        normal = np.zeros((0, 3), dtype=np.float64)
        radius = np.zeros((0,), dtype=np.float64)
    np.savez(path, origin=origin, normal=normal, radius=radius)
    return path


def load_ostium_frames(path):
    """Load ostium frames saved by :func:`save_ostium_frames`."""
    with np.load(path) as data:
        origin = np.ascontiguousarray(data["origin"], dtype=np.float64)
        normal = np.ascontiguousarray(data["normal"], dtype=np.float64)
        radius = np.ascontiguousarray(data["radius"], dtype=np.float64)
    if origin.ndim != 2 or origin.shape[1] != 3:
        raise ValueError(f"ostium frames origin must be (K, 3), got {origin.shape}")
    if normal.ndim != 2 or normal.shape[1] != 3:
        raise ValueError(f"ostium frames normal must be (K, 3), got {normal.shape}")
    radius = np.atleast_1d(np.ascontiguousarray(radius, dtype=np.float64))
    k = int(origin.shape[0])
    if int(normal.shape[0]) != k or int(radius.shape[0]) != k:
        raise ValueError(
            f"ostium frames length mismatch: origin={origin.shape[0]}, "
            f"normal={normal.shape[0]}, radius={radius.shape[0]}"
        )
    frames = []
    for i in range(k):
        frames.append(
            {
                "origin": origin[i].copy(),
                "normal": np.ascontiguousarray(_unit(normal[i]), dtype=np.float64),
                "radius": float(radius[i]),
            }
        )
    return frames


def _profiles_from_ostium_frames(frames):
    """Profile dicts so pinhole/spurious-opening match can use GT ostia."""
    profiles = []
    for i, fr in enumerate(frames):
        parsed = _as_ostium_frame(fr)
        profiles.append(
            {
                "index": int(fr["index"]) if "index" in fr else i,
                "barycenter": parsed["origin"],
                "normal": parsed["normal"],
                "radius": parsed["radius"],
            }
        )
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


def _vmtk_boundary_count(surface):
    """How many rims vtkvmtkPolyDataFlowExtensionsFilter will index.

    It extracts boundaries with vtkvmtkPolyDataBoundaryExtractor, which walks
    rims and skips the ones it cannot, so its numbering is its own -- taking the
    count from extract_boundary_loops would hand the filter ids it never made.
    """
    extractor = vtkvmtk.vtkvmtkPolyDataBoundaryExtractor()
    extractor.SetInputData(to_vtk_poly(surface))
    extractor.Update()
    return to_vtk_poly(extractor.GetOutput()).GetNumberOfCells()


def _extrude_boundaries(poly, extension_length, boundary_ids=None):
    """Run the extrusion filter directly, optionally on a subset of rims.

    vmtkFlowExtensions only forwards BoundaryIds when it is interactive, and
    choosing which rims to extrude is the whole point here. Settings mirror what
    the script would apply for boundarynormal/linear.
    """
    extender = vtkvmtk.vtkvmtkPolyDataFlowExtensionsFilter()
    extender.SetInputData(poly)
    extender.SetSigma(1.0)
    extender.SetAdaptiveExtensionLength(0)
    extender.SetAdaptiveExtensionRadius(1)
    extender.SetAdaptiveNumberOfBoundaryPoints(0)
    extender.SetExtensionLength(float(extension_length))
    extender.SetExtensionRatio(10.0)
    extender.SetExtensionRadius(1.0)
    extender.SetTransitionRatio(0.25)
    extender.SetCenterlineNormalEstimationDistanceRatio(1.0)
    extender.SetNumberOfBoundaryPoints(50)
    extender.SetExtensionModeToUseNormalToBoundary()
    extender.SetInterpolationModeToLinear()
    if boundary_ids is not None:
        ids = vtk.vtkIdList()
        for i in boundary_ids:
            ids.InsertNextId(int(i))
        extender.SetBoundaryIds(ids)
    extender.Update()
    return to_vtk_poly(extender.GetOutput())


def _extension_attempt(poly, extension_length, boundary_ids, max_cells):
    """(cleaned surface, opening count) for one extrusion, or (None, None)."""
    try:
        raw = _extrude_boundaries(poly, extension_length, boundary_ids)
    except Exception:
        return None, None
    if raw is None or raw.GetNumberOfCells() == 0:
        return None, None
    if raw.GetNumberOfCells() > max_cells:
        return None, None
    cleaned = clean_triangulate(raw)
    return cleaned, extract_boundary_loops(cleaned).GetNumberOfCells()


def add_flow_extensions(open_surface, extension_length=DEFAULT_EXTENSION_LENGTH):
    """Extrude cylinders on an OPEN surface. Must not be capped first or this is a no-op."""
    vtk_poly = clean_triangulate(open_surface)
    n_open = extract_boundary_loops(vtk_poly).GetNumberOfCells()
    if n_open == 0:
        raise TemplateQualityError(
            "Flow extensions require open boundaries; input surface is already closed."
        )

    n_layers = _flow_extension_layer_estimate(vtk_poly, extension_length)
    if n_layers > MAX_FLOW_EXTENSION_LAYERS:
        raise TemplateQualityError(
            f"a boundary loop would need {n_layers} flow-extension layers "
            f"(cap {MAX_FLOW_EXTENSION_LAYERS}); its rim edges are degenerate."
        )

    max_cells = MAX_EXTENSION_CELL_GROWTH * max(vtk_poly.GetNumberOfCells(), 1)
    extended, n_after = _extension_attempt(vtk_poly, extension_length, None, max_cells)
    if extended is None:
        raise TemplateQualityError(
            f"flow extensions blew past {max_cells} cells from "
            f"{vtk_poly.GetNumberOfCells()}; refusing to clean a runaway surface."
        )

    if n_after > n_open:
        # An extrusion should move a rim, not multiply it. More openings out than
        # in means the extruder tore the wall, and every later step inherits the
        # damage: the capper cannot close the tears, and the Voronoi diagram of
        # whatever does close them leaves the lumen. Extruding one rim at a time
        # shows which rim the filter cannot handle, and a vessel with one end
        # left flat still traces a good centerline -- a torn one never does.
        n_ids = _vmtk_boundary_count(vtk_poly)
        good = []
        for i in range(n_ids):
            _probe, n_probe = _extension_attempt(
                vtk_poly, extension_length, [i], max_cells
            )
            if n_probe == n_open:
                good.append(i)
        subset, n_subset = (None, None)
        if good:
            subset, n_subset = _extension_attempt(
                vtk_poly, extension_length, good, max_cells
            )
        if subset is not None and n_subset == n_open:
            print(
                f"  Flow extensions tore {n_after - n_open} opening(s); extended "
                f"{len(good)}/{n_ids} rim(s) and left the rest flat"
            )
            extended, n_after = subset, n_subset
        else:
            print(
                f"  Flow extensions tear this surface ({n_open} openings in, "
                f"{n_after} out); tracing the centerline without them"
            )
            return vtk_poly

    if n_after == 0:
        raise TemplateQualityError("Flow extensions produced a closed surface (unexpected).")
    if extended.GetNumberOfPoints() <= vtk_poly.GetNumberOfPoints():
        print("  WARNING: flow-extension point count did not increase; VMTK may have skipped ends.")
    print(
        f"  Flow extensions: {n_open} openings in, {n_after} openings out, "
        f"{vtk_poly.GetNumberOfPoints()} -> {extended.GetNumberOfPoints()} points"
    )
    return extended


def _flow_extension_layer_estimate(surface, extension_length):
    """Worst-case number of extrusion layers vmtkFlowExtensions would build.

    The filter advances each boundary by roughly its mean rim edge length, so a
    rim with micron-scale edges asks for millions of layers. Estimating this is
    cheap and turns a machine-killing allocation into a normal case failure.
    """
    worst = 0
    for radius, n_points, _bary in boundary_loop_radii(surface):
        perimeter = 2.0 * np.pi * max(float(radius), 1e-9)
        step = perimeter / max(int(n_points), 1)
        if step <= 0.0:
            return MAX_FLOW_EXTENSION_LAYERS + 1
        worst = max(worst, int(float(extension_length) / step))
    return worst


def _loop_apex(coords, centroid, displacement):
    """Where to put the tip of a lid over one rim.

    Not in the rim's own plane. ``cap_surface`` displaces its caps precisely so
    the Delaunay tetrahedralisation behind the centerlines does not have to work
    with coplanar points, and a flat lid reintroduces the degeneracy that
    displacement exists to avoid -- on the damaged meshes that reach this
    fallback in the first place, which are exactly the ones that hang.

    Pushed along the rim's Newell normal, away from the surface centroid so the
    lid domes outwards like a real cap rather than into the lumen. Unlike
    vtkvmtkCapPolyData's displacement, which is an absolute distance, this one
    scales with the rim's own radius so it breaks coplanarity by the same
    proportion on a 0.5 mm branch and on a 4 mm parent.
    """
    center = coords.mean(axis=0)
    if displacement <= 0.0:
        return center
    rolled = np.roll(coords, -1, axis=0)
    normal = np.cross(coords, rolled).sum(axis=0)
    norm = float(np.linalg.norm(normal))
    radius = float(np.mean(np.linalg.norm(coords - center, axis=1)))
    if norm <= 0.0 or radius <= 0.0:
        return center
    normal = normal / norm
    if float(np.dot(normal, center - centroid)) < 0.0:
        normal = -normal
    return center + normal * (displacement * radius)


def _close_loops_once(surface, weld, displacement=DEFAULT_CAP_DISPLACEMENT):
    """Close every boundary loop, either by fanning it or by welding it shut.

    Fanning keeps the rim geometry and adds a lid. Welding pulls the whole rim to
    one point, which closes a hole whatever shape it is -- including rims that
    repeat a vertex, where a fan only produces degenerate triangles that get
    cleaned away again, leaving the hole open.
    """
    poly, pts, faces = _triangle_points_faces(surface)
    if faces.size == 0:
        return poly, 0
    loops = extract_boundary_loops(poly)
    if loops.GetNumberOfCells() == 0:
        return poly, 0
    locator = vtk.vtkStaticPointLocator()
    locator.SetDataSet(poly)
    locator.BuildLocator()

    centroid = pts.mean(axis=0)
    pts_list = pts.tolist()
    new_faces = faces.tolist()
    remap = {}
    n_closed = 0
    for i in range(loops.GetNumberOfCells()):
        cell = loops.GetCell(i)
        n = cell.GetNumberOfPoints()
        if n < 3:
            continue
        coords = np.array(
            [cell.GetPoints().GetPoint(j) for j in range(n)], dtype=np.float64
        )
        ids = [int(locator.FindClosestPoint(xyz)) for xyz in coords]
        pts_list.append(_loop_apex(coords, centroid, displacement).tolist())
        target = len(pts_list) - 1
        if weld:
            for pid in set(ids):
                remap[pid] = target
        else:
            seen = set()
            ring = []
            for pid in ids:
                if pid not in seen:
                    seen.add(pid)
                    ring.append(pid)
            if len(ring) < 3:
                continue
            for k in range(len(ring)):
                new_faces.append([ring[k], ring[(k + 1) % len(ring)], target])
        n_closed += 1
    if n_closed == 0:
        return poly, 0
    if remap:
        rebuilt = []
        for a, b, c in new_faces:
            a, b, c = remap.get(a, a), remap.get(b, b), remap.get(c, c)
            if a == b or b == c or c == a:
                continue
            rebuilt.append([a, b, c])
        new_faces = rebuilt
    closed = _polydata_from_triangles(
        np.asarray(pts_list, dtype=np.float64),
        np.asarray(new_faces, dtype=np.int64).reshape(-1, 3),
    )
    return closed, n_closed


def fan_fill_every_loop(surface, max_passes=3):
    """Make a surface watertight for the Delaunay/Voronoi step.

    Used only on the capped copy that the centerline is traced from, and that
    copy is discarded afterwards, so a flat lid over each opening costs nothing.
    It needs no rim walk, which is the point: vtkvmtkCapPolyData gives up on rims
    it cannot traverse.

    Fans first, because they keep the rim where it is. Rims that survive a fan
    repeat a vertex, so they get welded shut instead -- forcing manifoldness
    between passes only re-cut what the fan had just closed.
    """
    current = clean_triangulate(surface)
    for weld in (False, True):
        for _ in range(int(max_passes)):
            if extract_boundary_loops(current).GetNumberOfCells() == 0:
                return current
            current, n_closed = _close_loops_once(current, weld=weld)
            if n_closed == 0:
                break
    current, _n = force_manifold_triangles(current)
    if extract_boundary_loops(current).GetNumberOfCells() != 0:
        current, _n2 = _close_loops_once(current, weld=True)
    return current


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
        # The capper walks rims; a fragmented vessel has rims it cannot walk, and
        # refusing here ended the case before a centerline was ever attempted.
        print(f"  Capper left {n_open} opening(s); fanning them shut for the centerline copy")
        capped = fan_fill_every_loop(capped)
        n_open = extract_boundary_loops(capped).GetNumberOfCells()
        if n_open != 0:
            raise TemplateQualityError(
                f"Capping left {n_open} openings; cannot run centerlines."
            )
    return recompute_point_normals(capped, auto_orient=True)


# vtkDelaunay3D places each tetrahedron's circumsphere by factoring a 4x4
# system, and a flow extension is the one thing on these surfaces that makes
# that system singular: an extension patch is a boundary ring swept along a
# single straight axis, so its points lie in exact concentric circles on exact
# parallel planes, and four of them are cospherical far more often than
# anything organic is. A failed factorisation is not an error the tessellator
# raises, it is one it grinds on -- SNF00000261 logs 375 "Unable to factor
# linear system" warnings and is still inside vmtkDelaunayVoronoi a quarter of
# an hour later, while the same vessel without its extensions traces in 7 s.
# In the 747-case run, where nothing yet put a clock on the trace, that one
# case ran 57,019 s: 15.8 hours against a median of 1,024, and the same failure
# as the six vessels that never came back at all.
#
# Raising or removing DelaunayTolerance does not touch it -- 0.0 and 0.01 both
# hang with the same 375 warnings -- because the points are not coincident,
# they are cospherical. Moving every point by a fraction of a micron destroys
# that without moving the geometry: the sigma below is about 1/1900 of the mean
# edge on that surface and 1/680 of the distance at which the tessellator
# already calls two points the same one, so the only tetrahedra it can change
# are the degenerate ones that had no well-defined circumsphere to start with.
# It takes SNF00000261 from 375 warnings to none and from hanging to 16.5 s.
#
# The seed is fixed, so the same case traces the same line on every run.
DELAUNAY_JITTER_MM = 1e-4
DELAUNAY_JITTER_SEED = 0


def _break_tessellation_degeneracy(closed_surface, sigma_mm=DELAUNAY_JITTER_MM):
    """A sub-micron nudge, so no four points sit exactly on one sphere."""
    poly = vtk.vtkPolyData()
    poly.DeepCopy(to_vtk_poly(closed_surface))
    points = poly.GetPoints()
    if points is None or points.GetNumberOfPoints() == 0 or sigma_mm <= 0.0:
        return poly
    coords = vtk_to_numpy(points.GetData()).astype(float)
    rng = np.random.default_rng(DELAUNAY_JITTER_SEED)
    coords += rng.normal(0.0, float(sigma_mm), coords.shape)
    points.SetData(numpy_to_vtk(coords, deep=1))
    points.Modified()
    return poly


def extract_voronoi_centerlines(closed_surface, source_points, target_points):
    if not target_points:
        raise TemplateQualityError("No outlet seed points for vmtkCenterlines.")
    vtk_poly = _break_tessellation_degeneracy(closed_surface)
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


def drop_degenerate_tracts(centerline, max_edge_mm=10.0, min_length_mm=1.0):
    """Delete the Voronoi stubs so one bad tract cannot condemn a whole trace.

    vmtkCenterlines writes one polyline per inlet-to-outlet path, and when the
    descent fails for one outlet it still writes a polyline -- just not a
    centerline. On p463 the extended trace came back with six tracts of which
    four are immaculate (450 to 876 points, median step 0.09 to 0.12 mm, no
    step over 0.69 mm, 64 to 99 mm long) and two are debris: a four-point stub
    whose last step jumps 96 mm to y=121, far outside the head, and a two-point
    tract of zero length sitting on the inlet. centerline_looks_valid then
    rejects the entire trace on its worst step, both candidates are discarded,
    and the case dies with "Voronoi centerline left the vessel lumen" while four
    perfectly good tracts are sitting in the object.

    A tract with a step of several centimetres is not a path down a vessel whose
    own resampling is 0.1 mm, and a tract of zero length is not a path at all.
    Removing them is a repair of the trace rather than a relaxation of the test:
    what is left still has to pass centerline_looks_valid on its own merits, and
    the arrival test still decides which openings were really reached.

    The point arrays come along -- MaximumInscribedSphereRadius lives there and
    the whole tube is built from it.
    """
    poly = to_vtk_poly(centerline)
    poly.BuildCells()
    n_cells = int(poly.GetNumberOfCells())
    keep = []
    for ci in range(n_cells):
        cell = poly.GetCell(ci)
        n = cell.GetNumberOfPoints()
        if n < 2:
            continue
        ids = [int(cell.GetPointId(j)) for j in range(n)]
        pts = np.array([poly.GetPoint(i) for i in ids], dtype=np.float64)
        segs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        if segs.size == 0:
            continue
        if float(segs.max()) > float(max_edge_mm):
            continue
        if float(segs.sum()) < float(min_length_mm):
            continue
        keep.append(ids)
    if not keep or len(keep) == n_cells:
        return centerline, 0

    used = sorted({i for ids in keep for i in ids})
    remap = {old: new for new, old in enumerate(used)}
    new_pts = vtk.vtkPoints()
    for i in used:
        new_pts.InsertNextPoint(poly.GetPoint(i))
    lines = vtk.vtkCellArray()
    for ids in keep:
        lines.InsertNextCell(len(ids))
        for i in ids:
            lines.InsertCellPoint(remap[i])
    out = vtk.vtkPolyData()
    out.SetPoints(new_pts)
    out.SetLines(lines)
    idx = np.asarray(used, dtype=np.int64)
    pd_in = poly.GetPointData()
    for a in range(pd_in.GetNumberOfArrays()):
        arr = pd_in.GetArray(a)
        if arr is None:
            continue
        vals = vtk_to_numpy(arr)
        sub = numpy_to_vtk(np.ascontiguousarray(vals[idx]), deep=True)
        sub.SetName(arr.GetName())
        out.GetPointData().AddArray(sub)
    return out, n_cells - len(keep)


def centerline_invalid_reason(centerline, reference_bounds, max_edge_mm=10.0, pad_mm=15.0):
    """Why this trace is not usable, or ``None`` when it is.

    Same test as ``centerline_looks_valid``, which is now a thin wrapper. A
    bare False tells whoever asked nothing at all, and that is how six
    variable-remesh cases came to report only "the re-trace did not produce a
    usable centerline" for six different underlying faults.
    """
    vtk_cl = to_vtk_poly(centerline)
    n_pts = vtk_cl.GetNumberOfPoints()
    if n_pts < 20:
        return f"only {n_pts} points"
    b = np.asarray(reference_bounds, dtype=np.float64).reshape(-1)
    lo = np.array([b[0] - pad_mm, b[2] - pad_mm, b[4] - pad_mm])
    hi = np.array([b[1] + pad_mm, b[3] + pad_mm, b[5] + pad_mm])
    n_out = 0
    for i in range(n_pts):
        p = np.asarray(vtk_cl.GetPoint(i), dtype=np.float64)
        if np.any(p < lo) or np.any(p > hi):
            n_out += 1
    if n_out > 0.2 * n_pts:
        return (f"{n_out} of {n_pts} points sit more than {pad_mm:.0f} mm "
                "outside the vessel bounds")
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
    if max_edge > max_edge_mm:
        return f"a {max_edge:.2f} mm jump between consecutive points"
    if n_ok < 1 or max_path < 5.0:
        return (f"its longest tract is {max_path:.2f} mm, under the 5 mm a real "
                "vessel path has")
    return None


def centerline_looks_valid(centerline, reference_bounds, max_edge_mm=10.0, pad_mm=15.0):
    """Reject Voronoi spikes that leave the vessel (common on looping siphons)."""
    return centerline_invalid_reason(
        centerline, reference_bounds, max_edge_mm=max_edge_mm, pad_mm=pad_mm
    ) is None


def _rescue_trace(centerline, ref_bounds, label):
    """Last resort for a trace about to be thrown away: drop its debris tracts.

    Only ever reached by a trace that has already failed centerline_looks_valid,
    and only kept when what is left passes the test on its own merits.

    Pruning unconditionally is not free, and it cost six cases to find out:
    applied to every trace it took SNF00000208, both SNF00000259 keeps,
    UPF_P0171, UPF_P0194 and SNF00000228 -- all of which had produced a surface
    -- under _centerline_reaches_targets, because a tract that looks like debris
    by length alone was the only tract reaching some outlet. So the prune runs
    where the alternative is losing the trace altogether, and nowhere else.
    """
    repaired, n_junk = drop_degenerate_tracts(centerline)
    if not n_junk or not centerline_looks_valid(repaired, ref_bounds):
        return centerline
    print(
        f"  Dropped {n_junk} degenerate tract(s) from the {label} trace, "
        "which is otherwise discarded entirely"
    )
    return repaired


# Each subdivision quadruples the points and the Delaunay pass that follows
# costs roughly five times as much: on SNF00000426_03, 20k points trace in 4 s,
# 81k in 24 s and 323k in 240 s. That is affordable for a case that is
# otherwise lost and not affordable as a habit, which is why nothing reaches
# here until the ordinary re-trace has already failed. Two levels are worth
# trying because one is not always enough -- SNF00000143_01_2's 0.431 mm ostium
# is found at 81k, and SNF00000426_03's 0.349 mm ostium dead-ends at 81k and is
# found at 323k -- and the ceiling stops a large surface turning this into a
# Delaunay that outlives the case timeout.
DENSE_RETRACE_MAX_POINTS = 400_000
DENSE_RETRACE_LEVELS = (1, 2)


def _densify_for_tracing(surface, n_subdivisions=1):
    """Linear subdivision, or ``None`` when that will not help.

    vtkvmtkSteepestDescentLineTracer walks the Voronoi diagram of the surface
    points, and in a branch a third of a millimetre wide there are too few of
    them for a descending path to exist -- the tracer says "Cannot find a
    steepest descent edge" and vmtkCenterlines returns a three-point stub
    sitting on the seed. Nothing about the vessel is wrong; the tessellation is
    too coarse, and every other handle makes no difference at all. On
    SNF00000426_03 profile 10 the source, the seed depth and the direction of
    travel were all varied at the original density and all twelve attempts
    dead-ended within a third of a millimetre of the seed; the only thing that
    changed the answer was points.

    The subdivider refuses a non-manifold input outright, and capping makes
    those: SNF00000426_03's cap leaves seven bad edges, which is the whole
    reason its first densified attempt came back with zero points.
    """
    base = clean_triangulate(surface)
    base, _n_nm = repair_nonmanifold_triangles(base)
    base, _n_forced = force_manifold_triangles(base)
    base, n_left = repair_nonmanifold_triangles(base)
    poly = to_vtk_poly(base)
    n_pts = poly.GetNumberOfPoints()
    grown = n_pts * (4 ** int(n_subdivisions))
    if n_pts == 0 or grown > DENSE_RETRACE_MAX_POINTS:
        print(
            f"  Not subdividing {n_subdivisions}x for the re-trace: {n_pts} points "
            f"would become {grown}, past the {DENSE_RETRACE_MAX_POINTS} the "
            "Delaunay is worth here."
        )
        return None
    subdivider = vtk.vtkLinearSubdivisionFilter()
    subdivider.SetInputData(poly)
    subdivider.SetNumberOfSubdivisions(int(n_subdivisions))
    subdivider.Update()
    dense = clean_triangulate(subdivider.GetOutput())
    if to_vtk_poly(dense).GetNumberOfPoints() <= n_pts:
        print(
            "  The surface could not be subdivided for the re-trace "
            f"({n_left} non-manifold edge(s) left); keeping the original."
        )
        return None
    return dense


def _retrace_outlets(surface, source_anat, seeds, profiles, missing, ref_bounds, what):
    """One re-trace attempt. Returns ``(centerline, gained)``; gained may be empty."""
    extra = _centerlines_in_child(
        surface, source_anat, seeds, CENTERLINE_TIMEOUT_S, label="missing outlets",
    )
    if extra is None:
        print(f"  The {what} re-trace returned nothing at all.")
        return None, []
    bad = centerline_invalid_reason(extra, ref_bounds)
    if bad is not None:
        print(f"  The {what} re-trace is not a usable centerline ({bad}).")
        return None, []
    extra_arrived, _gaps = centerline_arrivals(extra, profiles)
    gained = [i for i in missing if bool(extra_arrived[i])]
    if not gained:
        print(f"  The {what} re-trace reached none of them either.")
    return extra, gained


def _complete_missing_outlets(chosen, label, closed_anat, source_anat,
                              profiles, arrived, ref_bounds):
    """Ask again for the outlets this trace never reached, and add them to it.

    vmtkCenterlines writes one polyline per inlet-outlet path and, when it
    cannot descend to one of them, it does not say so: it returns fewer tracts.
    p463 keep 1 has seven clean rims on a 44k-point original with no
    non-manifold edges and nothing wrong with its profiles, and both traces came
    back with four tracts for six outlets, missing the 0.905 mm and 0.850 mm
    ostia of a tight trifurcation. The case was refused for branches the vessel
    plainly has.

    Asking for those outlets on their own is a different problem for the same
    solver -- descend from the inlet to two targets rather than pick a steepest
    path per target across the whole Voronoi diagram -- and it is asked on the
    capped anatomical surface, where the seed for an ostium is that ostium's own
    barycentre and the mapping cannot be mistaken. The raw trace is already one
    polyline per outlet, each re-walking the shared trunk, so appending another
    such polyline gives back exactly that kind of object; everything downstream
    resamples and re-branches it anyway.

    The retry is kept only if it is a valid trace on its own and reaches
    openings the original did not.
    """
    if closed_anat is None:
        return chosen, label
    missing = [int(i) for i in np.flatnonzero(~arrived)]
    if not missing:
        return chosen, label
    names = ", ".join(
        f"profile {profiles[i]['index']} (r={float(profiles[i]['radius']):.3f} mm)"
        for i in missing
    )
    print(f"  Re-tracing for the {len(missing)} opening(s) the {label} trace missed: {names}")
    found = []
    outstanding = list(missing)
    attempts = [(closed_anat, "plain")]
    # Each denser attempt asks only for what is still outstanding, because one
    # level is enough for some ostia and not for others: on SNF00000426_03 the
    # 0.374 mm ostium is reached at 81k points and the 0.349 mm one only at
    # 323k, and stopping at the first success left the second shut.
    for n_sub in DENSE_RETRACE_LEVELS:
        attempts.append((None, n_sub))
    for surface, how in attempts:
        if not outstanding:
            break
        if surface is None:
            surface = _densify_for_tracing(closed_anat, n_subdivisions=how)
            if surface is None:
                break
            how = f"{how}x subdivided"
            print(
                f"  Re-tracing on a {how} surface "
                f"({to_vtk_poly(surface).GetNumberOfPoints()} points) for "
                f"{len(outstanding)} opening(s) still missing."
            )
        seeds = [np.asarray(profiles[i]["barycenter"], dtype=np.float64)
                 for i in outstanding]
        extra, gained = _retrace_outlets(
            surface, source_anat, seeds, profiles, outstanding, ref_bounds, how
        )
        if gained:
            found.append(extra)
            outstanding = [i for i in outstanding if i not in set(gained)]
    if not found:
        print("  Keeping the original trace.")
        return chosen, label
    append = vtk.vtkAppendPolyData()
    append.AddInputData(to_vtk_poly(chosen))
    for extra in found:
        append.AddInputData(to_vtk_poly(extra))
    append.Update()
    merged = to_vtk_poly(append.GetOutput())
    if merged.GetPointData().GetArray("MaximumInscribedSphereRadius") is None:
        print("  The merged trace lost its MISR array; keeping the original.")
        return chosen, label
    if not centerline_looks_valid(merged, ref_bounds):
        print("  The merged trace failed the lumen check; keeping the original.")
        return chosen, label
    n_tracts = sum(int(to_vtk_poly(e).GetNumberOfCells()) for e in found)
    print(
        f"  Added {n_tracts} tract(s) from {len(found)} re-trace(s); together they "
        f"reach {len(missing) - len(outstanding)} of the {len(missing)} missed opening(s)"
    )
    return merged, f"{label} plus a re-trace"


def _centerline_reaches_targets(centerline, n_targets):
    """vmtkCenterlines writes one polyline per inlet→outlet path."""
    n_cells = int(to_vtk_poly(centerline).GetNumberOfCells())
    return n_cells >= int(n_targets)


# How far the nearest centerline point may sit from an ostium and still count as
# having arrived there, in units of that ostium's own radius.
#
# The trace is seeded on the ostium barycentres, so on a branch it reaches it
# comes within a rounding error of them -- and where it does not, it is not
# merely further away, it is in a different vessel. Measured over 391 openings
# in 72 cases the two populations do not overlap or even approach each other:
#
#     arrived   p50 0.08, p90 0.79, largest 1.01
#     missed    9.34, 9.37, 10.37, 11.54, 11.58, 13.17, 13.43
#
# Nothing at all falls between 1.01 and 9.34. Two radii sits in that gap with a
# factor of two below it and nearly five above, and it selects exactly the seven
# openings that were genuinely never visited, in two cases. The floor is three
# times the 0.1 mm centerline resampling step, so an ostium small enough that
# the resampling alone keeps the trace off its barycentre is not failed for it.
CENTERLINE_ARRIVAL_RADII = 2.0
CENTERLINE_ARRIVAL_FLOOR_MM = 0.3


def _traced_points(centerline, max_edge_mm=10.0, min_length_mm=1.0):
    """Every point on a tract that is actually a path, and none that is not.

    vmtkCenterlines writes a polyline for an outlet it could not descend to as
    readily as for one it reached, and that stub sits on the seed. So the stub
    lies within a rounding error of the ostium it failed to reach, and any
    question asked of the trace's raw points answers that the trace is right
    there. On p463 keep 1 two 2- and 3-point stubs of 0.01 and 0.03 mm put the
    arrival gaps for two ostia at 0.67 and 0.64 mm -- well inside their own
    radii -- while the nearest real tract was 6.69 and 5.09 mm away, and that
    is the difference between an opening this trace visits and one it does not.
    The same rule that drop_degenerate_tracts uses decides which is which.
    """
    poly = to_vtk_poly(centerline)
    poly.BuildCells()
    kept = []
    for ci in range(int(poly.GetNumberOfCells())):
        cell = poly.GetCell(ci)
        n = cell.GetNumberOfPoints()
        if n < 2:
            continue
        pts = np.array(
            [poly.GetPoint(int(cell.GetPointId(j))) for j in range(n)], dtype=np.float64
        )
        segs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        if segs.size == 0 or float(segs.max()) > float(max_edge_mm):
            continue
        if float(segs.sum()) < float(min_length_mm):
            continue
        kept.append(pts)
    if kept:
        return np.vstack(kept)
    return np.asarray(pv.wrap(poly).points, dtype=np.float64)


def centerline_arrival_gaps(centerline, profiles):
    """Distance from each ostium barycentre to the nearest point on the trace."""
    from scipy.spatial import cKDTree

    points = _traced_points(centerline)
    if points.size == 0:
        return np.full(len(profiles), np.inf)
    tree = cKDTree(points)
    return np.array(
        [float(tree.query(np.asarray(p["barycenter"], dtype=float))[0]) for p in profiles],
        dtype=float,
    )


def centerline_arrivals(centerline, profiles):
    """Which openings this trace actually reaches, and how far it missed the rest."""
    gaps = centerline_arrival_gaps(centerline, profiles)
    limits = np.array(
        [
            max(
                CENTERLINE_ARRIVAL_RADII * float(p["radius"]),
                CENTERLINE_ARRIVAL_FLOOR_MM,
            )
            for p in profiles
        ],
        dtype=float,
    )
    return gaps <= limits, gaps


CENTERLINE_TIMEOUT_S = 600.0


def _centerlines_in_child(closed_surface, source_points, target_points, timeout_s,
                          label="extended surface"):
    """extract_voronoi_centerlines, but able to give up. None when it does not finish.

    vmtkCenterlines tetrahedralises the whole capped surface before it traces
    anything, and on six of the SNF vessels that step never returns. That is a
    hang inside native VTK, not an exception, so no amount of try/except reaches
    it -- the retry below sat unreachable while each of those cases burned its
    entire 40-minute case budget. Run it where it can be killed and the retry
    gets its turn: all six then trace cleanly off the un-extended surface in
    4-15 s, reaching every target.
    """
    workdir = tempfile.mkdtemp(prefix="vmtk_centerline_")
    try:
        surface_path = os.path.join(workdir, "surface.vtp")
        seeds_path = os.path.join(workdir, "seeds.json")
        out_path = os.path.join(workdir, "centerline.vtp")
        save_polydata(closed_surface, surface_path)
        with open(seeds_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "source": [[float(c) for c in pt] for pt in source_points],
                    "target": [[float(c) for c in pt] for pt in target_points],
                },
                fh,
            )
        worker = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "centerline_worker.py")
        try:
            done = subprocess.run(
                [sys.executable, worker, surface_path, seeds_path, out_path],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                timeout=float(timeout_s),
                capture_output=True,
                text=True,
            )
        except subprocess.TimeoutExpired:
            print(
                f"  WARNING: centerline on the {label} did not finish in "
                f"{timeout_s:.0f}s; it takes no further part in the trace."
            )
            return None
        if done.returncode != 0 or not os.path.isfile(out_path):
            tail = (done.stderr or "").strip().splitlines()[-1:] or [""]
            print(
                f"  WARNING: centerline on the {label} failed "
                f"({tail[0][:120]}); it takes no further part in the trace."
            )
            return None
        return to_vtk_poly(pv.read(out_path))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


CENTERLINE_COVERAGE_VOXEL_MM = 0.5


def centerline_anatomical_coverage(centerline, anatomical_profiles,
                                   extension_length=DEFAULT_EXTENSION_LENGTH,
                                   voxel_mm=CENTERLINE_COVERAGE_VOXEL_MM):
    """How much of the vessel this trace actually visits, in 0.5 mm voxels.

    Arc length is the obvious measure and the wrong one twice over: the raw
    trace is one polyline per outlet, so every path re-walks the shared trunk
    and the total counts that trunk once per outlet, and the extended trace
    carries a flow extension on each end that is not anatomy at all. Clipping
    to the anatomical planes removes the second problem and counting occupied
    voxels removes the first, since a stretch walked six times occupies the
    same voxels as a stretch walked once.
    """
    try:
        clipped = clip_centerline_at_profiles(
            centerline, anatomical_profiles, extension_length=extension_length
        )
    except TemplateQualityError:
        return 0
    points = np.asarray(pv.wrap(to_vtk_poly(clipped)).points, dtype=float)
    if points.size == 0:
        return 0
    voxels = np.unique(np.round(points / float(voxel_mm)).astype(np.int64), axis=0)
    return int(len(voxels))


# Voxel size for the lumen-compartment test. It only has to be fine enough to
# keep a vessel open, and the smallest ostium in the corpus is r=0.205 mm.
LUMEN_VOXEL_MM = 0.3


def ostium_lumen_compartments(surface, profiles, voxel_mm=LUMEN_VOXEL_MM):
    """Which enclosed volume each ostium opens into. ``None`` when undecidable.

    A surface can be one connected sheet and still hold two separate lumens: a
    branch welded onto the trunk from the outside shares a wall with it and
    opens nowhere into it. No centerline can cross that wall, because there is
    nothing to cross into, and vmtkCenterlines says so only as "Cannot find a
    steepest descent edge" on the outlets in the far compartment. Asking the
    volume directly says it plainly.
    """
    from scipy import ndimage

    if not profiles:
        return None
    try:
        capped = to_vtk_poly(cap_surface(surface))
    except Exception:
        return None
    b = np.asarray(capped.GetBounds(), dtype=np.float64)
    lo = b[[0, 2, 4]] - 2.0 * voxel_mm
    dims = np.ceil((b[[1, 3, 5]] - b[[0, 2, 4]] + 4.0 * voxel_mm) / voxel_mm).astype(int)
    if np.any(dims < 4) or np.any(dims > 600):
        return None
    img = vtk.vtkImageData()
    img.SetSpacing(voxel_mm, voxel_mm, voxel_mm)
    img.SetOrigin(*[float(v) for v in lo])
    img.SetDimensions(*[int(v) for v in dims])
    img.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    img.GetPointData().GetScalars().Fill(1)
    stencil = vtk.vtkPolyDataToImageStencil()
    stencil.SetInputData(capped)
    stencil.SetOutputOrigin(*[float(v) for v in lo])
    stencil.SetOutputSpacing(voxel_mm, voxel_mm, voxel_mm)
    stencil.SetOutputWholeExtent(img.GetExtent())
    stencil.Update()
    cut = vtk.vtkImageStencil()
    cut.SetInputData(img)
    cut.SetStencilConnection(stencil.GetOutputPort())
    cut.ReverseStencilOff()
    cut.SetBackgroundValue(0)
    cut.Update()
    inside = vtk_to_numpy(cut.GetOutput().GetPointData().GetScalars()).reshape(dims[::-1])
    labels, n_lumens = ndimage.label(inside > 0)
    if n_lumens <= 1:
        return [1] * len(profiles)
    out = []
    reach = int(np.ceil(1.5 / voxel_mm))
    for prof in profiles:
        bary = np.asarray(prof["barycenter"], dtype=np.float64)
        normal = _unit(prof["normal"])
        seed = bary - normal * max(float(prof["radius"]), 0.5)
        idx = np.round((seed - lo) / voxel_mm).astype(int)
        best = None
        for dk in range(-reach, reach + 1):
            for dj in range(-reach, reach + 1):
                for di in range(-reach, reach + 1):
                    k, j, i = idx[2] + dk, idx[1] + dj, idx[0] + di
                    if not (0 <= k < dims[2] and 0 <= j < dims[1] and 0 <= i < dims[0]):
                        continue
                    lab = int(labels[k, j, i])
                    if lab == 0:
                        continue
                    d2 = dk * dk + dj * dj + di * di
                    if best is None or d2 < best[0]:
                        best = (d2, lab)
        out.append(best[1] if best is not None else 0)
    return out


def extract_centerlines_for_tube(extended_vessel, smoothed_vessel, anatomical_profiles, extended_profiles):
    """Trace on both the extended and the bare vessel, and keep the fuller one.

    Flow extensions exist to stop the Voronoi trace curling at the openings,
    and they are clipped off again before anything ships. They usually help and
    sometimes cost a great deal: on SNF00000415 the extended trace came back
    with every outlet reached and 6 cells for 6 targets -- passing every check
    this function used to make -- while quietly missing 186 mm of vessel. All
    5234 points of the bare-surface trace sit inside the lumen, and the 367 the
    extended one never visits sit 0.881 mm from the wall carrying an inscribed
    radius of 0.895 mm, which is what a centerline point in a thin branch looks
    like. VMTK had said so in passing: "Cannot find a steepest descent edge.
    Target not reached."

    Counting cells cannot catch that, so it is no longer asked to. Both traces
    are computed and the one covering more anatomy wins. On a healthy vessel
    that costs a second trace and changes nothing -- on C0010 the two agree to a
    median of 0.015 mm, covering 99.5% and 98.7% of each other.
    """
    ref_bounds = smoothed_vessel.GetBounds()
    source_ext, target_ext = seed_points_from_profiles(extended_profiles)
    n_targets = len(target_ext)

    candidates = []
    extended_cl = _centerlines_in_child(
        cap_surface(extended_vessel), source_ext, target_ext, CENTERLINE_TIMEOUT_S,
        label="extended surface",
    )
    if extended_cl is not None and not centerline_looks_valid(extended_cl, ref_bounds):
        extended_cl = _rescue_trace(extended_cl, ref_bounds, "extended-surface")
    if extended_cl is not None and centerline_looks_valid(extended_cl, ref_bounds):
        candidates.append(("extended surface", extended_cl, n_targets))
    elif extended_cl is not None:
        print("  WARNING: the extended-surface centerline left the lumen; discarding it.")

    source_anat, target_anat = seed_points_from_profiles(anatomical_profiles)
    closed_anat = None
    try:
        closed_anat = cap_surface(smoothed_vessel)
    except TemplateQualityError as exc:
        if not candidates:
            raise
        print(f"  WARNING: anatomical cap failed ({exc}); only the extended trace is available.")
    if closed_anat is not None:
        # On the same clock as the extended trace. This one is the fallback and
        # it is usually the fast one, but it runs the same tessellator on the
        # same kind of surface, so leaving it in-process would leave exactly
        # the hole the watchdog was written to close: a case that hangs here
        # burns its entire budget with nothing to fall back to.
        bare_cl = _centerlines_in_child(
            closed_anat, source_anat, target_anat, CENTERLINE_TIMEOUT_S,
            label="un-extended vessel",
        )
        if bare_cl is not None and not centerline_looks_valid(bare_cl, ref_bounds):
            bare_cl = _rescue_trace(bare_cl, ref_bounds, "bare-vessel")
        if bare_cl is not None and centerline_looks_valid(bare_cl, ref_bounds):
            candidates.append(("bare vessel", bare_cl, len(target_anat)))
        elif bare_cl is not None:
            print("  WARNING: the bare-vessel centerline left the lumen; discarding it.")

    if not candidates:
        raise TemplateQualityError(
            "Voronoi centerline left the vessel lumen; input openings were detected, "
            "but VMTK could not trace a path inside the tube."
        )

    # Coverage alone is too blunt to choose between these. On SNF00000607_01 the
    # extended trace never goes near three of the eight ostia -- 3.56, 3.84 and
    # 4.63 mm from openings of radius 0.38, 0.29 and 0.40 -- while the bare trace
    # arrives at all eight, and yet the extended one wins on voxels by 2.8% (330
    # to 322), because the branches it drops are short and the trunk it re-walks
    # is not. Picking it there cost the 0.296 mm ostium its frame: with no
    # centerline at the opening, opening_clip_frames read MISR off a vessel
    # 3.9 mm away and cut that ostium at 0.874 mm, three times too wide.
    #
    # An opening the trace never visits is not a smaller amount of the same good;
    # it is a branch this vessel has and this centerline does not. So arrivals
    # are counted first, and coverage only separates traces that arrive at the
    # same openings.
    scored = []
    for name, cl, want in candidates:
        arrived, gaps = centerline_arrivals(cl, anatomical_profiles)
        scored.append(
            (
                int(arrived.sum()),
                centerline_anatomical_coverage(cl, anatomical_profiles),
                name,
                cl,
                want,
                arrived,
                gaps,
            )
        )
    scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
    n_arrived, coverage, name, chosen, want, arrived, gaps = scored[0]
    if len(scored) > 1:
        other = scored[1]
        if n_arrived != other[0]:
            print(
                f"  NOTE: centerline taken from the {name}: it reaches "
                f"{n_arrived}/{len(anatomical_profiles)} openings against "
                f"{other[0]} for the {other[2]}"
            )
        else:
            other_cov = other[1]
            drift = (coverage - other_cov) / other_cov if other_cov else float("inf")
            if abs(drift) > 0.02:
                print(
                    f"  NOTE: centerline taken from the {name}: it covers {coverage} "
                    f"voxels against {other_cov} for the {other[2]} ({drift:+.1%})"
                )
    # A rescue, not a repair: it runs only where the chosen trace has already
    # come up short. Two things can be short, and the tract count is the weaker
    # of them -- vmtkCenterlines writes a polyline for an outlet it could not
    # descend to, so the count can be right while the trace stops eight or
    # twelve millimetres from the opening. The parent tube is built from this
    # trace, so an ostium the trace never visits gets no polyball and no wall,
    # and the uncap then reports "no nearby tube wall (closest 8.96 mm)" and
    # the case dies for a branch the vessel plainly has: SNF00000364_01_2 lost
    # three openings that way, SNF00000426_03 two, and five more cases one
    # each. Arrival is the test that sees it, so arrival gates the rescue too.
    if not _centerline_reaches_targets(chosen, want) or not bool(arrived.all()):
        chosen, name = _complete_missing_outlets(
            chosen, name, closed_anat, source_anat, anatomical_profiles,
            arrived, ref_bounds,
        )
        arrived, gaps = centerline_arrivals(chosen, anatomical_profiles)

    if not bool(arrived.all()):
        missed = ", ".join(
            f"profile {anatomical_profiles[i]['index']} "
            f"(r={anatomical_profiles[i]['radius']:.3f} mm, {gaps[i]:.2f} mm away)"
            for i in np.flatnonzero(~arrived)
        )
        print(
            f"  WARNING: the {name} centerline never reaches {missed}; those "
            f"openings are cut on their own measured plane instead."
        )

    if not _centerline_reaches_targets(chosen, want):
        # Before blaming the trace, ask whether there was anywhere for it to go.
        lumens = ostium_lumen_compartments(smoothed_vessel, anatomical_profiles)
        if lumens and len(set(lumens)) > 1:
            groups = {}
            for prof, lumen in zip(anatomical_profiles, lumens):
                groups.setdefault(lumen, []).append(
                    f"profile {prof['index']} (r={float(prof['radius']):.3f} mm)"
                )
            described = "; ".join(
                f"lumen {k}: " + ", ".join(v) for k, v in sorted(groups.items())
            )
            raise TemplateQualityError(
                f"this vessel encloses {len(set(lumens))} separate lumens and its "
                f"openings are split between them -- {described}. The wall between "
                "them is a double wall with no way through, so no centerline can "
                "reach the far openings and no single parent tube can carry them. "
                "The input mesh is what has to be fixed."
            )
        raise TemplateQualityError(
            f"the best centerline ({name}) has {chosen.GetNumberOfCells()} tracts for "
            f"{want} outlets, so at least one branch was never reached. A centerline "
            f"missing a branch produces a tube missing that branch."
        )
    return chosen


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


def _triangle_points_faces(surface, clean=True):
    """Point coordinates and triangle vertex ids.

    ``clean=True`` (default) welds and triangulates, matching every existing
    caller. ``clean=False`` still runs TriangleFilter when the mesh is stored
    as strips, but skips vtkCleanPolyData.
    """
    if clean:
        poly = clean_triangulate(surface)
    else:
        poly = _as_poly(surface)
        strips = poly.GetStrips()
        n_strips = 0 if strips is None else int(strips.GetNumberOfCells())
        n_polys = 0 if poly.GetPolys() is None else int(poly.GetPolys().GetNumberOfCells())
        if n_strips > 0 or n_polys == 0:
            tri = vtk.vtkTriangleFilter()
            tri.SetInputData(poly)
            tri.PassLinesOff()
            tri.PassVertsOff()
            tri.Update()
            poly = to_vtk_poly(tri.GetOutput())
    if poly.GetPoints() is None or poly.GetNumberOfPoints() == 0:
        return poly, np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64)
    pts = np.ascontiguousarray(vtk_to_numpy(poly.GetPoints().GetData()), dtype=np.float64)
    polys = poly.GetPolys()
    if polys is None or polys.GetNumberOfCells() == 0:
        return poly, pts, np.zeros((0, 3), dtype=np.int64)
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


def _mesh_area_fast(surface):
    _poly, pts, faces = _triangle_points_faces(surface, clean=False)
    if faces.size == 0:
        return 0.0
    return float(_triangle_areas(pts, faces).sum())


def _n_boundary_points_near(pts, faces, origin, radius):
    """How many boundary vertices sit inside 2R of origin. No VMTK extract."""
    if faces.size == 0 or pts.size == 0:
        return 0
    edges = np.concatenate(
        (
            np.sort(faces[:, [0, 1]], axis=1),
            np.sort(faces[:, [1, 2]], axis=1),
            np.sort(faces[:, [2, 0]], axis=1),
        ),
        axis=0,
    )
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    bdry = uniq[counts == 1]
    if bdry.size == 0:
        return 0
    vids = np.unique(bdry.ravel())
    origin = np.asarray(origin, dtype=np.float64).reshape(1, 3)
    d2 = np.sum((pts[vids] - origin) ** 2, axis=1)
    r2 = (2.0 * float(radius)) ** 2
    return int(np.count_nonzero(d2 < r2))


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


def _nonmanifold_edges_from_faces(faces):
    """Sorted (a, b) vertex pairs used by more than two triangles."""
    if faces.size == 0:
        return set()
    edges = np.concatenate(
        (
            np.sort(faces[:, [0, 1]], axis=1),
            np.sort(faces[:, [1, 2]], axis=1),
            np.sort(faces[:, [2, 0]], axis=1),
        ),
        axis=0,
    )
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    return {(int(a), int(b)) for (a, b), c in zip(uniq, counts) if c > 2}


def force_manifold_triangles(surface, max_passes=6):
    """Delete the triangles that keep an edge non-manifold.

    ``repair_nonmanifold_triangles`` only removes duplicates and small flaps, so
    two full-size sheets sharing an edge survive and every later VMTK filter
    inherits them. Here the smallest triangle on each offending edge is dropped
    until the edge is manifold. The holes this opens are sub-triangle sized and
    are closed by the pinhole fill that follows.
    """
    poly, pts, faces = _triangle_points_faces(surface)
    if faces.size == 0:
        return poly, 0
    n_dropped = 0
    for _ in range(int(max_passes)):
        bad_edges = _nonmanifold_edges_from_faces(faces)
        if not bad_edges:
            break
        areas = _triangle_areas(pts, faces)
        per_edge = {}
        for fi, (a, b, c) in enumerate(faces):
            for u, v in ((a, b), (b, c), (c, a)):
                key = (int(u), int(v)) if u <= v else (int(v), int(u))
                if key in bad_edges:
                    per_edge.setdefault(key, []).append(fi)
        drop = set()
        for fis in per_edge.values():
            if len(fis) <= 2:
                continue
            ranked = sorted(fis, key=lambda fi: areas[fi], reverse=True)
            drop.update(ranked[2:])
        if not drop:
            break
        mask = np.ones(len(faces), dtype=bool)
        mask[list(drop)] = False
        faces = faces[mask]
        n_dropped += len(drop)
    if n_dropped == 0:
        return poly, 0
    out = _polydata_from_triangles(pts, faces)
    print(f"  Forced manifold: dropped {n_dropped} triangles on non-manifold edges")
    return out, n_dropped


def _unpinch_split(points, faces, cand, nbr, apex, floor):
    """Split one edge per pinch so the fold becomes legal on the next sweep.

    A short edge whose endpoints share a neighbour that is not one of its own
    apexes cannot be folded -- the fold would weld two sheets together along
    that neighbour -- so collapse_tiny_edges refuses it and the edge survives to
    the quality gate. That is what is left of p305 keep 3: welding on the way
    into finalize_surface takes it from 3.3e-05 to 7.6e-05 mm, but 16 of its 28
    short edges arrive from the remesher already pinched, and no amount of
    welding moves them.

    Splitting the edge that runs from an endpoint to the offending neighbour
    inserts a vertex at the midpoint of a segment that is already part of the
    surface, so the geometry does not move at all, and it removes the adjacency
    that made the fold illegal. The next sweep then collapses the short edge the
    ordinary way, link condition satisfied on its merits.

    Only edges comfortably longer than the floor are split, so a split can never
    manufacture a new degenerate edge, and no triangle is given two splits,
    which keeps the retriangulation to the one-in-two-out case. Whatever is left
    over is picked up on a later sweep.
    """
    edge_faces = {}
    for fi, (a, b, c) in enumerate(faces):
        for x, y in ((a, b), (b, c), (c, a)):
            k = (int(x), int(y)) if x <= y else (int(y), int(x))
            edge_faces.setdefault(k, []).append(fi)

    used_faces = set()
    targets = []
    seen = set()
    for e in cand:
        u, v = int(e[0]), int(e[1])
        key = (min(u, v), max(u, v))
        if key in seen:
            continue
        seen.add(key)
        extra = (nbr[u] & nbr[v]) - apex.get(key, set())
        for w in sorted(int(x) for x in extra):
            opts = []
            for a in (u, v):
                k = (min(a, w), max(a, w))
                opts.append((float(np.linalg.norm(points[a] - points[w])), k))
            opts.sort(reverse=True)
            d, k = opts[0]
            if d <= 4.0 * float(floor):
                continue
            fs = edge_faces.get(k, [])
            if not fs or any(f in used_faces for f in fs):
                continue
            used_faces.update(fs)
            targets.append(k)
    if not targets:
        return None

    pts = [points[i] for i in range(len(points))]
    mid = {}
    for k in targets:
        mid[k] = len(pts)
        pts.append(0.5 * (points[k[0]] + points[k[1]]))

    out = []
    for a, b, c in faces:
        tri = (int(a), int(b), int(c))
        hit = None
        for i in range(3):
            x, y = tri[i], tri[(i + 1) % 3]
            k = (min(x, y), max(x, y))
            if k in mid:
                hit = (i, k)
                break
        if hit is None:
            out.append(tri)
            continue
        i, k = hit
        x, y, z = tri[i], tri[(i + 1) % 3], tri[(i + 2) % 3]
        m = mid[k]
        out.append((x, m, z))
        out.append((m, y, z))
    return np.asarray(pts, dtype=np.float64), np.asarray(out, dtype=np.int64)


def _boundary_loop_count(faces):
    """How many separate rims the triangle soup has."""
    usage = {}
    for a, b, c in faces:
        for x, y in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            k = (x, y) if x <= y else (y, x)
            usage[k] = usage.get(k, 0) + 1
    rim = [k for k, n in usage.items() if n == 1]
    if not rim:
        return 0
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in rim:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return len({find(x) for x in parent})


def _faces_are_clean(faces):
    """No triangle repeated and no edge carrying more than two triangles."""
    seen = set()
    usage = {}
    for a, b, c in faces:
        tri = (int(a), int(b), int(c))
        key = tuple(sorted(tri))
        if key in seen:
            return False
        seen.add(key)
        for x, y in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            k = (x, y) if x <= y else (y, x)
            usage[k] = usage.get(k, 0) + 1
            if usage[k] > 2:
                return False
    return True


def _collapse_micro_clusters(points, faces, floor):
    """Merge a blob of near-coincident vertices in one go.

    The pairwise link condition cannot clear these, and neither can a split.
    p305 keep 3 comes to rest on six vertices -- 2150, 2151, 2152, 7256, 8645
    and 8646 -- inside a blob about a micron across, where every pair is joined
    by an edge under the floor and every pair shares the rest of the blob as
    neighbours. Every fold is therefore a pinch by the pairwise test, and every
    edge that might be split to break the pinch is itself under the floor, so
    there is nothing to split either. That blob is not two sheets touching at a
    point; it is one vertex the remesher wrote six times.

    So the whole component of the short-edge graph is merged as a unit, onto its
    own centroid, which moves the surface by less than the blob is wide. The
    result is checked rather than predicted: a merge that produces a repeated
    triangle or an edge with three faces is reverted and that blob is left
    alone, so the sheet-fusion the link condition exists to prevent still cannot
    happen. Components are tried one at a time, so one bad blob does not cost
    the good ones.
    """
    ev = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    length = np.linalg.norm(points[ev[:, 0]] - points[ev[:, 1]], axis=1)
    short = ev[length < float(floor)]
    if not len(short):
        return None

    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in short:
        u, v = int(u), int(v)
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv

    groups = {}
    for x in list(parent):
        groups.setdefault(find(x), []).append(x)
    comps = [sorted(g) for g in groups.values() if len(g) > 1]
    if not comps:
        return None
    comps.sort(key=len, reverse=True)

    pts = np.array(points, dtype=np.float64, copy=True)
    remap = np.arange(len(points))
    rims_before = _boundary_loop_count(faces)
    merged = 0
    for comp in comps:
        trial = remap.copy()
        rep = comp[0]
        for x in comp:
            trial[x] = rep
        kept = trial[faces]
        alive = (
            (kept[:, 0] != kept[:, 1])
            & (kept[:, 1] != kept[:, 2])
            & (kept[:, 2] != kept[:, 0])
        )
        if not alive.any():
            continue
        if not _faces_are_clean(kept[alive]):
            continue
        # An opening must survive the merge. _faces_are_clean does not see this:
        # pulling a narrow rim into a single point leaves no repeated triangle
        # and no three-faced edge, it just quietly drops the degenerate faces
        # and seals the hole. p398 keep 2 lost its 0.53 mm opening exactly that
        # way, coming out with 7 against 8 anatomical profiles. Openings are the
        # point of the dataset, so a merge that changes the number of rims is
        # refused however tidy the triangles look afterwards.
        if _boundary_loop_count(kept[alive]) != rims_before:
            continue
        remap = trial
        pts[rep] = points[comp].mean(axis=0)
        merged += 1

    if not merged:
        return None
    kept = remap[faces]
    alive = (
        (kept[:, 0] != kept[:, 1])
        & (kept[:, 1] != kept[:, 2])
        & (kept[:, 2] != kept[:, 0])
    )
    kept = kept[alive]
    if not len(kept):
        return None
    used, compact = np.unique(kept, return_inverse=True)
    return pts[used], compact.reshape(kept.shape)


def collapse_tiny_edges(
    surface, floor=WELD_TOLERANCE_MM, max_sweeps=WELD_MAX_SWEEPS, allow_cuts=True
):
    """Collapse edges shorter than ``floor`` without ever fusing two sheets.

    A sub-micron edge has to go. VMTK's boundary-preserving remesh keeps every
    rim edge it is given, so the edge survives to the final quality gate and the
    case is rejected; worse, vtkvmtkPolyDataFlowExtensionsFilter reads its layer
    thickness off a boundary's mean edge length, so a rim with micron edges
    extrudes millions of layers. That is what the ten-hour hangs and the
    out-of-memory crashes are, and 7 of the 9 hangs in the 2026-09-17 run
    arrived on a surface already carrying such an edge.

    Collapsing the edge is the repair. A naive collapse is not: fold an edge
    whose endpoints share neighbours beyond the triangles on that edge and two
    sheets that only touched at a point are welded into a seam. The link
    condition rules it out -- collapsing (u, v) is safe exactly when the
    vertices adjacent to both endpoints are the apexes of the triangles on that
    edge, two for an interior edge and one on a boundary. A pinch fails the test
    and is left alone, which is the right answer rather than a miss.

    Measured over the 116 surfaces of the keep-one run this came from: 39
    carried edges under the floor, 38 came out clear of it, no surface lost an
    opening -- it collapses boundary edges too, so that was the thing to be sure
    of -- and the worst area drift was 0.999999, which is why the aneurysm
    texture survives it.
    """
    poly = to_vtk_poly(surface)
    for _sweep in range(int(max_sweeps)):
        _p, points, faces = _triangle_points_faces(poly)
        if faces.size == 0:
            return poly
        ev = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        length = np.linalg.norm(points[ev[:, 0]] - points[ev[:, 1]], axis=1)
        cand = ev[length < float(floor)]
        if not len(cand):
            return poly

        nbr = [set() for _ in range(len(points))]
        apex = {}
        for a, b, c in faces:
            nbr[a].update((b, c)); nbr[b].update((a, c)); nbr[c].update((a, b))
            for u, v, w in ((a, b, c), (b, c, a), (c, a, b)):
                apex.setdefault((min(u, v), max(u, v)), set()).add(w)

        order = np.argsort(
            np.linalg.norm(points[cand[:, 0]] - points[cand[:, 1]], axis=1)
        )
        chosen, blocked = [], set()
        for idx in order:
            u, v = int(cand[idx][0]), int(cand[idx][1])
            if u == v or u in blocked or v in blocked:
                continue
            if nbr[u] & nbr[v] != apex.get((min(u, v), max(u, v)), set()):
                continue
            chosen.append((u, v))
            blocked.update({u, v})
            blocked |= nbr[u] | nbr[v]
        if not chosen:
            # Every short edge left is pinched. Break the pinch and retry --
            # unless the caller forbade it. Both fallbacks below cut geometry
            # rather than fold it, which is the right trade at the sub-micron
            # floor this function was written for and the wrong one at a floor
            # a hundred times larger: swept at 0.01 mm over the 26 GT surfaces
            # that carry a sub-floor edge, the cuts tore UPF_P0171.00_ID1 open
            # from 6 rims to 7 and p469 from 6 to 11, taking their worst
            # circularity from 0.93 and 0.98 to 0.02. A fold can only ever
            # shorten the mesh; a cut can open it.
            if not allow_cuts:
                return poly
            split = _unpinch_split(points, faces, cand, nbr, apex, floor)
            if split is None:
                split = _collapse_micro_clusters(points, faces, floor)
            if split is None:
                return poly
            poly = _polydata_from_triangles(split[0], split[1])
            continue

        remap = np.arange(len(points))
        for u, v in chosen:
            remap[v] = u            # v folds onto u, and u does not move
        kept = remap[faces]
        alive = (
            (kept[:, 0] != kept[:, 1])
            & (kept[:, 1] != kept[:, 2])
            & (kept[:, 2] != kept[:, 0])
        )
        if not alive.any():
            return poly
        kept = kept[alive]
        used, compact = np.unique(kept, return_inverse=True)
        poly = _polydata_from_triangles(points[used], compact.reshape(kept.shape))
    return poly


def weld_degenerate_vertices(
    surface, tolerance=WELD_TOLERANCE_MM, min_edge=MIN_EDGE_LENGTH_MM, max_passes=None
):
    """Collapse micron-scale edges so no edge is shorter than ``min_edge``.

    This used to be vtkCleanPolyData with an absolute tolerance, doubled on
    every pass until the shortest edge cleared the floor. That is not an edge
    collapse -- it is a global point merge, and it welds every pair of vertices
    anywhere on the surface that falls inside the radius. Four passes from
    1e-3 mm reach 8e-3, a twentieth of the 0.15 mm target edge, the same band as
    the pre-remesh weld in the keep-one pipeline: that one took SNF00000228
    keep 2 from 8 non-manifold edges to 742, with 129 duplicate triangles that
    were not there before. The escalation guaranteed the widest radius would be
    reached on exactly the surfaces that needed the most care.

    ``collapse_tiny_edges`` touches only the two faces on the edge it folds, so
    there is no radius to get wrong and nothing to escalate. Where it refuses,
    the rim is pinched and merging it would fuse two sheets; the warning below
    is then an honest report rather than a failure to try hard enough.

    ``max_passes`` is accepted and ignored so existing call sites keep working.
    """
    poly = collapse_tiny_edges(surface, floor=float(tolerance))
    _p, pts, faces = _triangle_points_faces(poly)
    if faces.size == 0:
        return poly, 0.0
    edges = _triangle_edge_lengths(pts, faces)
    shortest = float(edges.min()) if edges.size else 0.0
    if shortest < float(min_edge):
        # Retry at the floor the quality gate actually enforces. The sweep above
        # runs at the weld tolerance, ten times coarser, and _unpinch_split only
        # splits an escape edge comfortably longer than the floor it is given --
        # four times it -- so at 1e-3 mm it refuses escapes of 2.2e-3 mm and the
        # pinch survives. p551 reaches the gate on exactly one such edge,
        # 3.98e-05 mm, pinched on one extra shared neighbour whose escapes are
        # 2.18e-03 mm: illegal to split against a 1e-3 floor, legal against 1e-4.
        poly = collapse_tiny_edges(poly, floor=float(min_edge))
        _p, pts, faces = _triangle_points_faces(poly)
        edges = _triangle_edge_lengths(pts, faces)
        shortest = float(edges.min()) if edges.size else 0.0
    if shortest < float(min_edge):
        print(f"  WARNING: shortest edge still {shortest:.3e} mm after welding")
    return poly, shortest


def _triangle_adjacency(faces):
    """Neighbour lists over manifold (usage == 2) edges."""
    edge_faces = {}
    for fi, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            key = (int(u), int(v)) if u <= v else (int(v), int(u))
            edge_faces.setdefault(key, []).append(fi)
    adj = {}
    for fis in edge_faces.values():
        if len(fis) != 2:
            continue
        adj.setdefault(fis[0], []).append(fis[1])
        adj.setdefault(fis[1], []).append(fis[0])
    return adj


def _patch_rim_angles(labels, label, normals, adj):
    """Dihedral angles, in degrees, across the boundary of one patch.

    This is what separates an end cap from an equally planar piece of wall: the
    vessel meets a cap at a sharp edge all the way round, while a wall patch
    continues smoothly into its neighbours.
    """
    angles = []
    inside = np.flatnonzero(labels == label)
    inside_set = set(inside.tolist())
    for fi in inside.tolist():
        for gi in adj.get(fi, ()):
            if gi in inside_set:
                continue
            dot = float(np.clip(np.dot(normals[fi], normals[gi]), -1.0, 1.0))
            angles.append(np.degrees(np.arccos(dot)))
    return np.asarray(angles, dtype=np.float64)


def _coplanar_patches(pts, faces, angle_deg=CAP_PATCH_ANGLE_DEG):
    """Label triangles by region-growing while the normal stays near the seed's."""
    a = pts[faces[:, 1]] - pts[faces[:, 0]]
    b = pts[faces[:, 2]] - pts[faces[:, 0]]
    cross = np.cross(a, b)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), 1e-30)
    adj = _triangle_adjacency(faces)
    cos_tol = float(np.cos(np.deg2rad(angle_deg)))
    labels = np.full(len(faces), -1, dtype=np.int64)
    n_labels = 0
    for seed in range(len(faces)):
        if labels[seed] >= 0:
            continue
        seed_n = normals[seed]
        labels[seed] = n_labels
        stack = [seed]
        while stack:
            f = stack.pop()
            for g in adj.get(f, ()):
                if labels[g] < 0 and float(np.dot(normals[g], seed_n)) >= cos_tol:
                    labels[g] = n_labels
                    stack.append(g)
        n_labels += 1
    return labels, n_labels, areas, normals, adj


def _is_disc(faces_subset):
    """Euler characteristic 1 -> the patch is a triangulated disc (one rim, no handle)."""
    if faces_subset.size == 0:
        return False
    verts = np.unique(faces_subset)
    edges = np.unique(
        np.concatenate(
            (
                np.sort(faces_subset[:, [0, 1]], axis=1),
                np.sort(faces_subset[:, [1, 2]], axis=1),
                np.sort(faces_subset[:, [2, 0]], axis=1),
            ),
            axis=0,
        ),
        axis=0,
    )
    return int(len(verts) - len(edges) + len(faces_subset)) == 1


def uncap_closed_surface(surface, max_caps=CAP_MAX_COUNT):
    """Remove flat end caps from a vessel that arrives fully closed.

    A handful of AneuX originals were never decapped, so ``measure_open_profiles``
    finds no ostium and the case dies at step 1b. A cap is a machine-planar,
    disc-shaped patch (it was made by triangulating a planar cross-section), which
    no anatomical wall is. Surfaces that already have open boundaries are returned
    untouched, so this can run on every case.
    """
    poly = clean_triangulate(surface)
    if extract_boundary_loops(poly).GetNumberOfCells() > 0:
        return poly
    _p, pts, faces = _triangle_points_faces(poly)
    if faces.size == 0:
        return poly
    labels, n_labels, areas, normals, adj = _coplanar_patches(pts, faces)
    total_area = float(areas.sum())
    caps = []
    for label in range(n_labels):
        sel = labels == label
        area = float(areas[sel].sum())
        if area <= 0.0 or area > CAP_MAX_AREA_FRACTION * total_area:
            continue
        sub = faces[sel]
        # One triangle is trivially planar, disc-shaped and Euler-1; a cap is
        # a fan over a real cross-section.
        if len(sub) < CAP_MIN_TRIANGLES:
            continue
        corners = pts[sub].reshape(-1, 3)
        center = corners.mean(axis=0)
        normal = _unit((normals[sel] * areas[sel, None]).sum(axis=0))
        if float(np.abs((corners - center) @ normal).max()) > CAP_PLANARITY_MM:
            continue
        r_max = float(np.linalg.norm(corners - center, axis=1).max())
        if r_max < CAP_MIN_RADIUS_MM:
            continue
        # A cap fills most of its own circumcircle. A flat strip of cylinder
        # facets is just as planar but nothing like a disc, and removing one
        # would tear a hole in the vessel wall.
        fill = area / (np.pi * r_max * r_max)
        if fill < CAP_MIN_DISC_FILL or fill > 1.0:
            continue
        if not _is_disc(sub):
            continue
        rim_angles = _patch_rim_angles(labels, label, normals, adj)
        if rim_angles.size == 0 or float(np.median(rim_angles)) < CAP_MIN_RIM_ANGLE_DEG:
            continue
        caps.append((area, label))
    if not caps:
        print("  WARNING: surface is closed but no flat cap was recognised")
        return poly
    caps.sort(reverse=True)
    keep_labels = {label for _area, label in caps[: int(max_caps)]}
    keep = ~np.isin(labels, list(keep_labels))
    opened = _polydata_from_triangles(pts, faces[keep])
    opened, _n_regions = drop_tiny_islands(opened)
    n_loops = extract_boundary_loops(opened).GetNumberOfCells()
    if n_loops < 2:
        print(
            f"  WARNING: removing {len(keep_labels)} cap(s) left {n_loops} opening(s); "
            "keeping the closed surface"
        )
        return poly
    removed = float(sum(area for area, _label in caps[: int(max_caps)]))
    print(
        f"  Removed {len(keep_labels)} flat cap(s) ({removed:.1f} mm^2, "
        f"{100.0 * removed / total_area:.1f}% of area) -> {n_loops} openings"
    )
    return opened


def boundary_loop_radii(surface):
    """(radius, n_points, barycenter) for every closed boundary loop."""
    loops = extract_boundary_loops(surface)
    out = []
    for i in range(loops.GetNumberOfCells()):
        cell = loops.GetCell(i)
        n = cell.GetNumberOfPoints()
        if n == 0:
            continue
        pts = np.array([cell.GetPoints().GetPoint(j) for j in range(n)], dtype=np.float64)
        center = pts.mean(axis=0)
        out.append((float(np.mean(np.linalg.norm(pts - center, axis=1))), int(n), center))
    return out


def _loop_at_a_profile(bary, profiles, radius=None, n_points=None):
    """True when a boundary loop really is one of the anatomical ostia.

    The geometric pinhole tests know nothing about anatomy, and on this dataset
    a real ostium can be smaller than a leftover rim, so when the profiles are
    known they decide what may be closed. Position is not enough on its own:
    clip debris lies next to the ostium it came from. The loop also has to be
    the right size for that ostium and carry a real rim's worth of points -- the
    smallest genuine opening measured here had 10.
    """
    if not profiles:
        return False
    if n_points is not None and int(n_points) < MIN_OPENING_LOOP_POINTS:
        return False
    b = np.asarray(bary, dtype=np.float64)
    for profile in profiles:
        center = np.asarray(profile["barycenter"], dtype=np.float64)
        r_p = max(float(profile["radius"]), MIN_OPENING_RADIUS_MM)
        tol = max(PROFILE_PROTECT_RADIUS_FACTOR * r_p, PROFILE_PROTECT_MIN_MM)
        if float(np.linalg.norm(b - center)) > tol:
            continue
        if radius is not None and float(radius) < PROFILE_PROTECT_MIN_RADIUS_FRACTION * r_p:
            continue
        return True
    return False


def _rim_graph(surface):
    """Boundary vertices, their rim edges, and the degree of each, by point id."""
    poly = clean_triangulate(surface)
    _p, pts, faces = _triangle_points_faces(poly)
    if faces.size == 0:
        return poly, pts, faces, np.empty((0, 2), dtype=np.int64)
    ev = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    key = np.sort(ev, axis=1)
    uniq, counts = np.unique(key, axis=0, return_counts=True)
    return poly, pts, faces, uniq[counts == 1]


def branching_rims(surface, report=False):
    """Rim components that are not a simple cycle, i.e. a slit joined to a rim.

    A boundary vertex with three or more rim edges on it is the seam of a tear
    that reaches an opening. Every edge on it still has exactly one face, so the
    surface is manifold and ``assert_template_quality`` sees nothing; the rim,
    though, is no longer a loop. On p398 four such vertices took an ostium that
    left the remesher at circularity 0.997 down to 0.181, its perimeter from
    6.08 mm to 14.26 mm, because ``extract_boundary_loops`` measures a rim
    component and the component had grown five extra cycles.
    """
    _poly, _pts, _faces, rim = _rim_graph(surface)
    if rim.size == 0:
        return 0
    deg = np.bincount(rim.ravel())
    branch = np.flatnonzero(deg > 2)
    if branch.size == 0:
        return 0
    comp = _rim_components(rim, deg.size)
    n = len(set(comp[b] for b in branch))
    if report:
        print(f"  {n} rim component(s) branch at {branch.size} vertex/vertices")
    return n


def _rim_components(rim, n_points):
    """Component label per point id over the rim graph (-1 off the rim)."""
    parent = np.arange(n_points, dtype=np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in rim:
        ru, rv = find(int(u)), find(int(v))
        if ru != rv:
            parent[ru] = rv
    label = np.full(n_points, -1, dtype=np.int64)
    for pid in np.unique(rim):
        label[pid] = find(int(pid))
    return label


def _shortest_rim_ear(rim, pts, b):
    """The shortest cycle through rim vertex ``b``, as a list of point ids.

    Straight Dijkstra on the rim graph with ``b`` removed, between each pair of
    its neighbours: the cheapest pair plus the two edges back to ``b`` is the
    smallest loop the branch closes, which is the tear rather than the ostium.
    """
    import heapq

    adj = {}
    for u, v in rim:
        adj.setdefault(int(u), []).append(int(v))
        adj.setdefault(int(v), []).append(int(u))
    nbrs = adj.get(int(b), [])
    if len(nbrs) < 2:
        return None
    best = None
    for i, src in enumerate(nbrs):
        dist = {src: 0.0}
        prev = {}
        heap = [(0.0, src)]
        targets = set(nbrs[i + 1:])
        seen = set()
        while heap and targets - seen:
            d, u = heapq.heappop(heap)
            if u in seen:
                continue
            seen.add(u)
            for w in adj.get(u, []):
                if w == int(b) or w in seen:
                    continue
                nd = d + float(np.linalg.norm(pts[u] - pts[w]))
                if nd < dist.get(w, np.inf):
                    dist[w] = nd
                    prev[w] = u
                    heapq.heappush(heap, (nd, w))
        for dst in nbrs[i + 1:]:
            if dst not in dist:
                continue
            total = (dist[dst]
                     + float(np.linalg.norm(pts[int(b)] - pts[src]))
                     + float(np.linalg.norm(pts[int(b)] - pts[dst])))
            if best is None or total < best[0]:
                chain, node = [dst], dst
                while node != src:
                    node = prev[node]
                    chain.append(node)
                best = (total, [int(b)] + chain)
    return None if best is None else best[1]


def unbranch_rims(
    surface, label="surface", profiles=None, min_radius=WALL_PINHOLE_RADIUS_MM,
    max_ears=32,
):
    """Close the tears that hang off an opening's rim.

    ``collapse_pinhole_loops`` reaches a puncture that shares a rim component
    with an ostium only when the boundary extractor still splits the component
    into loops, and it does not: loops are the components now, which is what
    stopped rims being thrown away mid-walk. So a tear welded onto an ostium is
    measured together with it -- the component is enormous by pinhole standards
    -- and no pass closes it. This one finds the smallest cycle through each
    branch vertex instead, which is the tear and not the opening.

    The tear is fanned shut, not welded to a point. Welding fuses two rim
    vertices, which is a non-manifold edge, and the manifold repair then cuts
    triangles off and opens the next tear: on p398 that chased itself through
    sixteen welds and left the rim branching anyway. A fan only adds triangles,
    so nothing downstream has to cut.
    """
    current = clean_triangulate(surface)
    closed = 0
    for _ in range(int(max_ears)):
        poly, pts, faces, rim = _rim_graph(current)
        if rim.size == 0:
            break
        deg = np.bincount(rim.ravel(), minlength=len(pts))
        branch = np.flatnonzero(deg > 2)
        if branch.size == 0:
            break
        # How big a tear may be is set by the rim it hangs off, not by a
        # constant: an ear is worth closing when it is small beside its own
        # opening. p398's is 0.426 mm against an ostium of 0.97 mm, which no
        # fixed pinhole radius would reach without also reaching real anatomy
        # elsewhere in the dataset.
        comp = _rim_components(rim, len(pts))
        comp_limit = {}
        for c in np.unique(comp[comp >= 0]):
            member = pts[comp == c]
            radius = float(np.mean(np.linalg.norm(member - member.mean(axis=0), axis=1)))
            comp_limit[int(c)] = max(float(min_radius), 0.5 * radius)
        ear = None
        for b in branch:
            cycle = _shortest_rim_ear(rim, pts, int(b))
            if cycle is None or len(cycle) < 3:
                continue
            ring = []
            for pid in cycle:
                if pid not in ring:
                    ring.append(int(pid))
            if len(ring) < 3:
                continue
            coords = pts[np.asarray(ring, dtype=np.int64)]
            centre = coords.mean(axis=0)
            radius = float(np.mean(np.linalg.norm(coords - centre, axis=1)))
            # Radius only, never the point count. ``_is_wall_pinhole`` also
            # calls a loop of under six points a pinhole, which is right for a
            # free-standing rim and wrong here: an ear is a cycle through one
            # branch vertex, so three and four point cycles are the normal case
            # whatever their size. Judged that way p469 fanned twenty-four of
            # them and took a 0.977 rim to 0.001.
            if radius >= comp_limit.get(int(comp[int(b)]), float(min_radius)):
                continue
            if _loop_at_a_profile(centre, profiles, radius=radius, n_points=len(ring)):
                continue
            if ear is None or radius < ear[2]:
                ear = (ring, centre, radius)
        if ear is None:
            break
        ring, centre, radius = ear
        pts_list = pts.tolist()
        pts_list.append(centre.tolist())
        apex = len(pts_list) - 1
        new_faces = faces.tolist()
        for k in range(len(ring)):
            new_faces.append([ring[k], ring[(k + 1) % len(ring)], apex])
        current = clean_triangulate(_polydata_from_triangles(
            np.asarray(pts_list, dtype=np.float64),
            np.asarray(new_faces, dtype=np.int64).reshape(-1, 3),
        ))
        closed += 1
        print(f"  Fanned a {len(ring)}-point tear (r={radius:.4f} mm) off a rim "
              f"on the {label}")
    if closed == 0:
        return clean_triangulate(surface), 0
    # Checked, not trusted: the same rule the rest of these repairs now follow.
    # A fan that leaves fewer openings, a rounder rim somewhere else at the cost
    # of this one, or a new non-manifold edge is not a repair.
    before = clean_triangulate(surface)
    before_rims, before_circ = _rim_shape(before)
    after_rims, after_circ = _rim_shape(current)
    harm = None
    if after_rims != before_rims:
        harm = f"openings {before_rims} -> {after_rims}"
    elif after_circ < before_circ - 1e-6:
        harm = f"worst rim circularity {before_circ:.3f} -> {after_circ:.3f}"
    elif int(inspect_surface_topology(current)["n_nonmanifold"]) > int(
        inspect_surface_topology(before)["n_nonmanifold"]
    ):
        harm = "a new non-manifold edge"
    if harm is not None:
        print(f"  NOTE: fanning {closed} rim tear(s) on the {label} would cost {harm}; "
              "left the rims as they were")
        return before, 0
    return current, closed


def _is_wall_pinhole(loop, min_radius, profiles=None):
    """A loop too small, or with too few points, to be an anatomical ostium.

    The point-count test is not cosmetic: vtkvmtkPolyDataFlowExtensionsFilter
    derives its layer thickness from a boundary's mean edge length, so a 4-point
    rim whose edges are microns long makes it extrude millions of layers. That is
    what turned a 70k-point vessel into a 123M-cell surface and produced both the
    out-of-memory worker crashes and the multi-hour hangs.
    """
    radius, n_points, bary = loop
    if _loop_at_a_profile(bary, profiles, radius=radius, n_points=n_points):
        return False
    return radius < float(min_radius) or n_points < MIN_OPENING_LOOP_POINTS


def _boundary_loop_vertex_rings(surface):
    """Boundary loops as lists of point ids, walking the rim edges of ``surface``.

    ``extract_boundary_loops`` returns coordinates from a stripper, which is fine
    for measuring but useless for stitching. Walking the mesh's own rim edges
    gives ids that can be triangulated directly.
    """
    poly = clean_triangulate(surface)
    _p, pts, faces = _triangle_points_faces(poly)
    if faces.size == 0:
        return poly, pts, faces, []
    usage = {}
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            key = (int(u), int(v)) if u <= v else (int(v), int(u))
            usage[key] = usage.get(key, 0) + 1
    neighbours = {}
    for (u, v), count in usage.items():
        if count != 1:
            continue
        neighbours.setdefault(u, []).append(v)
        neighbours.setdefault(v, []).append(u)
    rings = []
    visited = set()
    for start in neighbours:
        if start in visited or len(neighbours[start]) != 2:
            continue
        ring = [start]
        visited.add(start)
        cur, prev = start, None
        while True:
            options = [w for w in neighbours.get(cur, ()) if w != prev]
            nxt = next((w for w in options if w not in visited), None)
            if nxt is None:
                break
            ring.append(nxt)
            visited.add(nxt)
            prev, cur = cur, nxt
        if len(ring) >= 3 and start in neighbours.get(ring[-1], ()):
            rings.append(ring)
    return poly, pts, faces, rings


def fan_fill_small_loops(surface, min_radius=WALL_PINHOLE_RADIUS_MM, profiles=None):
    """Stitch every sub-``min_radius`` boundary loop shut with a triangle fan.

    vtkvmtkCapPolyData refuses surfaces whose rims it cannot walk, and
    vtkFillHolesFilter silently gives up above its hole size, so neither can be
    the only way to close a puncture. A fan over the loop's barycentre always
    can, and a puncture is small enough that the flat patch is invisible.
    """
    # A rim vertex with four boundary neighbours is not part of any simple cycle,
    # so the walk below would skip its hole entirely. Those ears are debris from
    # the clip, never anatomy.
    surface = drop_boundary_ear_triangles(surface, profiles=profiles)
    poly, pts, faces, rings = _boundary_loop_vertex_rings(surface)
    if not rings:
        return poly, 0
    pts = pts.tolist()
    faces = faces.tolist()
    n_filled = 0
    for ring in rings:
        coords = np.asarray([pts[i] for i in ring], dtype=np.float64)
        center = coords.mean(axis=0)
        radius = float(np.mean(np.linalg.norm(coords - center, axis=1)))
        if not _is_wall_pinhole((radius, len(ring), center), min_radius, profiles):
            continue
        pts.append(center.tolist())
        apex = len(pts) - 1
        for k in range(len(ring)):
            faces.append([ring[k], ring[(k + 1) % len(ring)], apex])
        n_filled += 1
    if n_filled == 0:
        return poly, 0
    filled = _polydata_from_triangles(
        np.asarray(pts, dtype=np.float64), np.asarray(faces, dtype=np.int64)
    )
    filled = recompute_point_normals(filled, auto_orient=False)
    print(f"  Fan-filled {n_filled} small boundary loop(s)")
    return filled, n_filled


def _free_edge_components(pts, faces):
    """Connected components of the free-edge graph, as lists of point ids.

    Unlike a loop walk this makes no assumption that a rim is a simple cycle, so
    it still describes a boundary whose vertices branch.
    """
    if faces.size == 0:
        return []
    edges = np.concatenate(
        (
            np.sort(faces[:, [0, 1]], axis=1),
            np.sort(faces[:, [1, 2]], axis=1),
            np.sort(faces[:, [2, 0]], axis=1),
        ),
        axis=0,
    )
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    free = uniq[counts == 1]
    if free.size == 0:
        return []
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in free:
        union(int(a), int(b))
    groups = {}
    for node in list(parent):
        groups.setdefault(find(node), []).append(node)
    return list(groups.values())


def _boundary_component_extent(pts, ids):
    coords = pts[np.asarray(ids, dtype=np.int64)]
    center = coords.mean(axis=0)
    radius = float(np.mean(np.linalg.norm(coords - center, axis=1)))
    return center, radius


def collapse_small_boundary_components(
    surface, min_radius=WALL_PINHOLE_RADIUS_MM, label="surface", profiles=None
):
    """Weld every sub-``min_radius`` boundary component down to a single point.

    A rim whose vertices branch -- more than two free edges meeting at one point
    -- is not a cycle, so neither vtkvmtkCapPolyData nor a triangle fan can walk
    it; VMTK reports "Can't find adjacent point" and bails, and the puncture then
    reaches assert_template_quality unrepaired. Collapsing the whole component to
    its barycentre needs no traversal order at all, so it closes the hole
    whatever shape the rim is. It only ever runs on punctures far below the
    target edge length, so it moves less geometry than a single triangle.
    """
    poly, pts, faces = _triangle_points_faces(surface)
    comps = _free_edge_components(pts, faces)
    if not comps:
        return poly, 0
    small, keep = [], []
    for ids in comps:
        center, radius = _boundary_component_extent(pts, ids)
        if _is_wall_pinhole((radius, len(ids), center), min_radius, profiles):
            small.append((ids, center, radius))
        else:
            keep.append(ids)
    if not small:
        return poly, 0
    if len(keep) < 2:
        print(
            f"  WARNING: collapsing {len(small)} pinhole(s) on the {label} would "
            f"leave {len(keep)} opening(s); leaving them in place"
        )
        return poly, 0
    pts_list = pts.tolist()
    remap = {}
    for ids, center, _radius in small:
        pts_list.append(center.tolist())
        target = len(pts_list) - 1
        for pid in ids:
            remap[int(pid)] = target
    new_faces = []
    for a, b, c in faces.tolist():
        a, b, c = remap.get(a, a), remap.get(b, b), remap.get(c, c)
        if a == b or b == c or c == a:
            continue
        new_faces.append([a, b, c])
    collapsed = _polydata_from_triangles(
        np.asarray(pts_list, dtype=np.float64),
        np.asarray(new_faces, dtype=np.int64).reshape(-1, 3),
    )
    radii = ", ".join(f"{r:.4f}" for _i, _c, r in small)
    print(f"  Collapsed {len(small)} branching pinhole(s) on the {label} (r={radii} mm)")
    return collapsed, len(small)


def collapse_pinhole_loops(
    surface, min_radius=WALL_PINHOLE_RADIUS_MM, label="surface", profiles=None
):
    """Weld each sub-ostium boundary loop down to a single point.

    This works per loop where collapse_small_boundary_components works per
    free-edge component, and that difference is the whole point: a puncture
    whose rim touches an ostium's rim shares a component with it, so the merged
    component measures far too large to look like a pinhole and nothing gets
    closed at all. The boundary extractor still separates the two as loops, so
    collapsing per loop reaches punctures the component pass cannot.
    """
    poly, pts, faces = _triangle_points_faces(surface)
    if faces.size == 0:
        return poly, 0
    loops = extract_boundary_loops(poly)
    n_loops = loops.GetNumberOfCells()
    if n_loops == 0:
        return poly, 0

    locator = vtk.vtkStaticPointLocator()
    locator.SetDataSet(poly)
    locator.BuildLocator()

    small, n_keep = [], 0
    for i in range(n_loops):
        cell = loops.GetCell(i)
        n = cell.GetNumberOfPoints()
        if n == 0:
            continue
        coords = np.array(
            [cell.GetPoints().GetPoint(j) for j in range(n)], dtype=np.float64
        )
        center = coords.mean(axis=0)
        radius = float(np.mean(np.linalg.norm(coords - center, axis=1)))
        if not _is_wall_pinhole((radius, int(n), center), min_radius, profiles):
            n_keep += 1
            continue
        ids = {int(locator.FindClosestPoint(xyz)) for xyz in coords}
        small.append((ids, center, radius))
    if not small:
        return poly, 0
    if n_keep < 2:
        print(
            f"  WARNING: collapsing {len(small)} pinhole loop(s) on the {label} would "
            f"leave {n_keep} opening(s); leaving them in place"
        )
        return poly, 0

    pts_list = pts.tolist()
    remap = {}
    for ids, center, _radius in small:
        pts_list.append(center.tolist())
        target = len(pts_list) - 1
        for pid in ids:
            remap[pid] = target
    new_faces = []
    for a, b, c in faces.tolist():
        a, b, c = remap.get(a, a), remap.get(b, b), remap.get(c, c)
        if a == b or b == c or c == a:
            continue
        new_faces.append([a, b, c])
    collapsed = _polydata_from_triangles(
        np.asarray(pts_list, dtype=np.float64),
        np.asarray(new_faces, dtype=np.int64).reshape(-1, 3),
    )
    radii = ", ".join(f"{r:.4f}" for _i, _c, r in small)
    print(f"  Collapsed {len(small)} pinhole loop(s) on the {label} (r={radii} mm)")
    return collapsed, len(small)


def close_wall_pinholes(surface, min_radius=WALL_PINHOLE_RADIUS_MM, label="surface", max_passes=4, profiles=None):
    """Patch pinholes until none are left; each fan can expose the next one."""
    total = 0
    current = clean_triangulate(surface)
    for _ in range(int(max_passes)):
        current, n_filled = patch_wall_pinholes(
            current, min_radius=min_radius, label=label, profiles=profiles
        )
        total += n_filled
        if not any(
            _is_wall_pinhole(lp, min_radius, profiles) for lp in boundary_loop_radii(current)
        ):
            return current, total
        if n_filled == 0:
            break
    # Neither the capper nor the fan could walk what is left: those rims branch.
    for _ in range(int(max_passes)):
        current, n_collapsed = collapse_small_boundary_components(
            current, min_radius=min_radius, label=label, profiles=profiles
        )
        total += n_collapsed
        if n_collapsed == 0:
            break
        current, _n_forced = force_manifold_triangles(current)
        if not any(
            _is_wall_pinhole(lp, min_radius, profiles) for lp in boundary_loop_radii(current)
        ):
            return current, total
    # A rim that touches an ostium's rim hides inside its free-edge component.
    for _ in range(int(max_passes)):
        current, n_loops = collapse_pinhole_loops(
            current, min_radius=min_radius, label=label, profiles=profiles
        )
        total += n_loops
        if n_loops == 0:
            break
        current, _n_forced = force_manifold_triangles(current)
        if not any(
            _is_wall_pinhole(lp, min_radius, profiles) for lp in boundary_loop_radii(current)
        ):
            break
    return current, total


def patch_wall_pinholes(surface, min_radius=WALL_PINHOLE_RADIUS_MM, label="surface", profiles=None):
    """Close boundary loops smaller than ``min_radius``; keep the real ostia open.

    Caps every loop with vtkvmtkCapPolyData (which triangulates a hole of any
    shape), then re-opens the caps that belong to loops at or above
    ``min_radius``. Unlike vtkFillHolesFilter this is not limited by hole size
    and never leaves a partially stitched rim.
    """
    poly = clean_triangulate(surface)
    loops = boundary_loop_radii(poly)
    small = [lp for lp in loops if _is_wall_pinhole(lp, min_radius, profiles)]
    if not small:
        return poly, 0
    keep = [lp for lp in loops if not _is_wall_pinhole(lp, min_radius, profiles)]
    if len(keep) < 2:
        print(
            f"  WARNING: patching {len(small)} pinhole(s) on the {label} would leave "
            f"{len(keep)} opening(s); leaving them in place"
        )
        return poly, 0
    capper = None
    capped = None
    for displacement in (0.0, DEFAULT_CAP_DISPLACEMENT):
        capper, capped = _cap_surface_with_entity_ids(poly, displacement)
        if extract_boundary_loops(capped).GetNumberOfCells() == 0:
            break
    if capped is None or extract_boundary_loops(capped).GetNumberOfCells() != 0:
        print(f"  Capper could not close the {label}; fan-filling {len(small)} pinhole(s)")
        return fan_fill_small_loops(poly, min_radius=min_radius, profiles=profiles)
    ids = capped.GetCellData().GetArray("CellEntityIds")
    center_ids = capper.GetCapCenterIds()
    if ids is None or center_ids is None or center_ids.GetNumberOfIds() == 0:
        print(f"  Capper produced no CellEntityIds for the {label}; fan-filling instead")
        return fan_fill_small_loops(poly, min_radius=min_radius, profiles=profiles)
    offset = int(capper.GetCellEntityIdOffset())
    cap_centers = [
        np.array(capped.GetPoint(center_ids.GetId(i)), dtype=np.float64)
        for i in range(center_ids.GetNumberOfIds())
    ]
    cap_eids = [offset + 1 + i for i in range(len(cap_centers))]
    remaining = list(range(len(cap_centers)))
    reopen = set()
    for _radius, _n, bary in keep:
        if not remaining:
            break
        best = min(remaining, key=lambda i: float(np.linalg.norm(cap_centers[i] - bary)))
        reopen.add(cap_eids[best])
        remaining.remove(best)
    keep_cells = [
        ci
        for ci in range(capped.GetNumberOfCells())
        if int(ids.GetComponent(ci, 0)) not in reopen
    ]
    if not keep_cells:
        return poly, 0
    filled = _polydata_from_kept_cells(capped, keep_cells)
    filled = strip_all_arrays(filled)
    n_after = len(boundary_loop_radii(filled))
    if n_after != len(keep):
        print(
            f"  Pinhole patch on the {label} left {n_after} loops (expected "
            f"{len(keep)}); fan-filling instead"
        )
        return fan_fill_small_loops(poly, min_radius=min_radius, profiles=profiles)
    print(f"  Patched {len(small)} wall pinhole(s) on the {label} (r < {min_radius:.2f} mm)")
    return filled, len(small)


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


def _anatomical_rim_points(surface, profiles):
    """Point ids on the boundary loops that are anatomical openings."""
    poly, pts, faces = _triangle_points_faces(surface)
    if faces.size == 0 or not profiles:
        return np.empty(0, dtype=np.int64)
    ev = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    key = np.sort(ev, axis=1)
    uniq, counts = np.unique(key, axis=0, return_counts=True)
    rim = uniq[counts == 1]
    if rim.size == 0:
        return np.empty(0, dtype=np.int64)
    comp = _rim_components(rim, len(pts))
    keep = []
    for c in np.unique(comp[comp >= 0]):
        ids = np.flatnonzero(comp == c)
        member = pts[ids]
        bary = member.mean(axis=0)
        radius = float(np.mean(np.linalg.norm(member - bary, axis=1)))
        if _loop_at_a_profile(bary, profiles, radius=radius, n_points=len(ids)):
            keep.append(ids)
    return np.concatenate(keep) if keep else np.empty(0, dtype=np.int64)


def drop_boundary_ear_triangles(surface, max_passes=16, profiles=None):
    """Drop triangles with 2+ boundary edges (fins glued onto an ostium rim).

    Those ears give rim vertices four boundary neighbours, and VMTK's
    vtkvmtkPolyDataBoundaryExtractor then reports only one opening.

    Dropping one turns its third edge into a boundary edge, which can make a
    neighbour an ear, so this runs until the surface stops changing -- and on a
    finished surface that cascade is a tear. p398 arrived at ``finalize_surface``
    with all eight rims round (worst circularity 0.980) and lost 41 triangles
    here, which split it into five shells, put 40 non-manifold edges into the
    weld that put them back together and cost 54 more triangles to the manifold
    repair; the ostium came out at 0.181. So when the anatomy is known, no
    triangle on one of its rims is dropped: an ear at an ostium is only in the
    way of walking that ostium's loop, and an ostium's loop is never one this
    fills.
    """
    poly, pts, faces = _triangle_points_faces(surface)
    if faces.size == 0:
        return poly
    protected = _anatomical_rim_points(poly, profiles) if profiles else None
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
        if protected is not None and protected.size:
            keep |= np.isin(faces, protected).any(axis=1)
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
    min_points=None,
    skip_decimate=False,
):
    """Parent tube via a narrow-band polyball image + marching cubes.

    ``min_points`` / ``skip_decimate`` are for the variable-remesh path.
    Defaults keep remeshing.py and ``sanitize_vessel_for_vmtk`` on the 20k floor.
    """
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
    if skip_decimate:
        kept, n_nm = repair_nonmanifold_triangles(kept)
    else:
        pre_decimate = kept
        kept = decimate_dense_mc(kept, min_points=min_points)
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


def decimate_variable_parent_tube(surface, min_points=VAR_MC_DECIMATE_MIN_POINTS):
    """Post-uncap collapse of the MC staircase. Variable remesh only.

    Uncap already cut the ostia on the fine MC surface. BoundaryVertexDeletionOff
    keeps those rims; the interior is allowed down to ``min_points``. Non-manifold
    output is thrown away and the pre-decimate surface is kept.
    """
    poly = _as_poly(surface)
    pre = poly
    n = poly.GetNumberOfPoints()
    floor = int(min_points)
    if n <= floor:
        return poly
    reduction = 1.0 - float(floor) / float(n)
    out = decimate_dense_mc(poly, target_reduction=reduction, min_points=floor)
    out, n_nm = repair_nonmanifold_triangles(out)
    if n_nm > 0 and out.GetNumberOfPoints() < pre.GetNumberOfPoints():
        print("  Decimate left non-manifold edges; keeping the pre-decimate surface")
        return pre
    return out


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


# A shell this small, floating free of the vessel, is debris. Measured over all
# 740 inputs, twelve are disconnected and the largest stray shell in any of them
# is 61 points out of 13071 -- 0.47% -- with every other one at 21 points or
# fewer. Nothing in the corpus sits anywhere near this line, so it separates
# debris from a branch without having to guess.
INPUT_DEBRIS_MAX_FRACTION = 0.02


def drop_disconnected_debris(surface, label="original", max_fraction=INPUT_DEBRIS_MAX_FRACTION):
    """Remove free-floating shells before the openings are counted.

    A shell that touches nothing has a rim, and a rim measured on the detailed
    original becomes an anatomical ostium that the whole run then tries to keep.
    UPF_P0194 carries a 61-point scrap whose 0.629 mm loop was counted as its
    seventh profile; the scrap is dropped by the first clip that runs, because
    every clip keeps one connected region, and the case failed six against
    seven every time. It is not an ostium and it cannot be opened, welded or
    traced -- the only honest thing to do with it is to leave it out of the
    count.

    A large disconnected shell is a different matter: it could be a second
    vessel, and dropping it silently would throw away anatomy. That raises.
    """
    poly = to_vtk_poly(surface)
    n_regions = count_connected_regions(poly)
    if n_regions <= 1:
        return poly, 0
    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(poly)
    conn.SetExtractionModeToAllRegions()
    conn.ColorRegionsOn()
    conn.Update()
    from vtk.util.numpy_support import vtk_to_numpy

    region_ids = vtk_to_numpy(conn.GetOutput().GetPointData().GetArray("RegionId"))
    sizes = np.bincount(np.asarray(region_ids, dtype=np.int64), minlength=n_regions)
    n_total = int(sizes.sum())
    main = int(np.argmax(sizes))
    biggest_other = int(np.max(np.delete(sizes, main))) if n_total else 0
    if n_total > 0 and biggest_other > max_fraction * n_total:
        raise TemplateQualityError(
            f"the {label} is in {n_regions} disconnected pieces, the second of them "
            f"{biggest_other} of {n_total} points ({100.0 * biggest_other / n_total:.1f}%). "
            "That is too large to treat as debris and nothing downstream can trace or "
            "clip a second vessel, so it is not dropped quietly."
        )
    kept = keep_largest_region(poly)
    dropped = n_regions - 1
    print(
        f"  Dropped {dropped} free-floating shell(s) from the {label} "
        f"({n_total - kept.GetNumberOfPoints()} of {n_total} points); "
        "they carry rims but no anatomy"
    )
    return kept, dropped


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


# The seam clip opens the loop the cutter scored, on the cutter's own plane, so
# the rim it leaves behind is that loop. Measured over 16 fallback clips, 14
# come out within 2% of it. The two that do not are the ones that merged a
# neighbouring ostium into the hole, and they are far outside: 1.63x and 3.38x.
CLIP_OPENING_OVER_CUTTER = 1.30


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
    # This cleaner runs at the default tolerance, so it merges only points that
    # are exactly equal and leaves everything the seam clip shaved off a vertex.
    # Same cut, same slivers, same consequences as the pipe-section one.
    candidate = weld_clip_slivers(cleaner.GetOutput())

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
        print(
            f"  [Uncap] Profile {profile['index']} seam clip left the surface with no "
            "opening at all. Skipping this end."
        )
        return surface, False
    matched = min(
        openings,
        key=lambda op: float(np.linalg.norm(np.asarray(op["center"]) - bary)),
    )
    r_open = float(matched.get("radius", 0.0))
    n_rim = int(matched.get("n_points", 0))
    d_match = float(np.linalg.norm(np.asarray(matched["center"]) - bary))
    # Three clauses reject a seam clip, and reporting only the radius range sent
    # SNF00000228 profile 6 to the wrong place: it was rejected at r=0.393 mm
    # inside a 0.124-2.853 mm band, for the rim point count. Say which one.
    why = []
    if r_open < min_r or r_open > max_r:
        why.append(f"r={r_open:.3f} mm outside {min_r:.3f}-{max_r:.3f} mm")
    if n_rim < MIN_OPENING_LOOP_POINTS:
        why.append(f"{n_rim} rim points, under the {MIN_OPENING_LOOP_POINTS} needed")
    # The band around r_gt cannot see a clip that merged two ostia: it allows
    # r_gt + 2.5 mm, which for UPF_P0258's 0.658 mm profile 1 is a 3.16 mm
    # crater. That clip opened 2.734 mm against a cutter loop of 0.810 mm,
    # swallowed the 1.412 mm ostium beside it, and the case shipped six
    # openings against seven profiles. The cutter loop is the prediction this
    # clip is meant to fulfil, so measure the outcome against that.
    r_cut = float(loop["radius"])
    if r_cut > 0.0 and r_open > CLIP_OPENING_OVER_CUTTER * r_cut:
        why.append(
            f"r={r_open:.3f} mm is {r_open / r_cut:.2f}x the cutter loop's "
            f"{r_cut:.3f} mm, so the clip took in more than that loop"
        )
    # Where the tube has no branch at this ostium -- because the trace never
    # got there -- the clip plane slices straight across the trunk instead, and
    # every test above passes: the loop it finds is a real loop and the clip
    # fulfils it exactly. What it does not do is open THIS end. SNF00000426_03
    # profile 10 was reported opened at r=2.157 mm against a 0.349 mm ostium,
    # the hole was the inlet's, and the case shipped ten openings for eleven
    # profiles. The distance was measured all along and only printed.
    reach = max(1.0, 2.0 * r_gt)
    if d_match > reach:
        why.append(
            f"the opening it made sits {d_match:.3f} mm away, past the "
            f"{reach:.3f} mm that still counts as this ostium"
        )
    if why:
        print(
            f"  [Uncap] Profile {profile['index']} seam clip rejected: "
            f"{'; '.join(why)} (nearest loop sits {d_match:.3f} mm from the ostium). "
            "Skipping this end."
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
        # The old bar here was eight radii or 4 mm, whichever was larger, which
        # for a small ostium is no bar at all: SNF00000607_01's 0.296 mm opening
        # had no centerline point within 3.9 mm and still took its frame from
        # one, inheriting a 0.874 mm inscribed radius off a vessel it has
        # nothing to do with. The pipe-section cutter widens that again, to
        # max(1.5 r, r + 0.2) = 1.31 mm, and bites 4.4 radii of wall out around
        # a rim it was only supposed to trim. Where the trace is not actually at
        # the opening, the profile measured on the surface is the better
        # authority, and it is what this falls back to.
        if np.linalg.norm(closest - origin) > max(
            CENTERLINE_ARRIVAL_RADII * profile_r, CENTERLINE_ARRIVAL_FLOOR_MM
        ):
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


def _opening_clip_radius(radius, limit_mm=None):
    """Cutter radius, narrowed so it cannot reach the ostium next door.

    The default is 1.5 R, which on p153 keep LICA is 2.126 mm around a 1.417 mm
    ostium -- and the 0.317 mm ostium beside it sits 2.14 mm off that axis, so
    the cutter swallowed it and the neighbour guard threw the cut away. The cut
    was not wrong, only wider than it had any need to be: an ostium is opened
    by taking the stub off its own end, and a cylinder that reaches the next
    branch along is doing something else.
    """
    default = max(float(radius) * OPENING_CLIP_RADIUS_FACTOR, float(radius) + 0.2)
    if limit_mm is None:
        return default
    # Still wide enough to cut a hole this ostium's own size; below that there
    # is nothing to be gained by narrowing further and the guard can judge it.
    return max(min(default, float(limit_mm)), float(radius) + 0.05)


def clip_radius_limit_for(origin, outward, radius, ostia, mine,
                          extension_length=None, trimmed=False):
    """How wide the cutter at ``mine`` may be before it reaches another ostium.

    Only ostia that actually fall inside the cut region constrain it: one
    behind the plane, or past the far end of the cylinder, is never touched
    however wide the cylinder is. ``None`` means nothing is in the way.
    """
    origin = np.asarray(origin, dtype=np.float64)
    outward = _unit(outward)
    height = _opening_clip_height(radius, extension_length=extension_length, trimmed=trimmed)
    limit = None
    for j, (other, reach) in enumerate(ostia):
        if j == mine:
            continue
        rel = np.asarray(other, dtype=np.float64) - origin
        along = float(rel @ outward)
        if along <= -OPENING_CLIP_INWARD_OVERLAP_MM or along >= height:
            continue
        lateral = float(np.linalg.norm(rel - along * outward))
        # ``reach`` is that ostium's own neighbourhood radius, so keeping the
        # cutter outside it keeps the branch it belongs to.
        clear = lateral - 0.5 * float(reach)
        limit = clear if limit is None else min(limit, clear)
    return limit


def _opening_clip_height(radius, extension_length=None, trimmed=False):
    """Finite cylinder length along the outward tangent.

    ``7R`` is enough for the short polyball stubs on the parent tube. GT remesh
    adds a fixed-length flow extension; on thin outlets ``7R`` is shorter than
    that extension and the far stub survives as a leftover fragment.

    ``trimmed`` says the flow extensions were already cut back to the ostium
    plane, so only a collar remains and the cutter can be short. That matters on
    a tortuous siphon, where a 12 mm cylinder reaches a different part of the
    same vessel and deletes it.
    """
    r = float(radius)
    if trimmed:
        return max(r * OPENING_CLIP_TRIMMED_HEIGHT_FACTOR, OPENING_CLIP_TRIMMED_HEIGHT_MIN_MM)
    by_radius = max(r * OPENING_CLIP_HEIGHT_FACTOR, 2.0) + r * OPENING_EXTENSION_LENGTH_FACTOR
    if extension_length is None:
        return by_radius
    return max(by_radius, float(extension_length) + 2.0 * r)


def _outboard_cap_implicit(origin, outward, radius, extension_length=None, trimmed=False,
                           clip_radius_limit=None):
    origin = np.asarray(origin, dtype=np.float64)
    outward = _unit(outward)
    clip_radius = _opening_clip_radius(radius, limit_mm=clip_radius_limit)
    height = _opening_clip_height(radius, extension_length=extension_length, trimmed=trimmed)
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


def _drop_small_fragments_fast(surface, min_fraction=0.05):
    """Same keep/drop rule as ``_drop_small_fragments``, VTK-only (variable uncap)."""
    poly = _as_poly(surface)
    n_pts = poly.GetNumberOfPoints()
    if n_pts == 0:
        return poly
    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(poly)
    conn.SetExtractionModeToAllRegions()
    conn.ColorRegionsOn()
    conn.Update()
    n_regions = int(conn.GetNumberOfExtractedRegions())
    if n_regions <= 1:
        return poly
    labelled = conn.GetOutput()
    region = labelled.GetPointData().GetArray("RegionId")
    if region is None:
        return poly
    ids = np.ascontiguousarray(vtk_to_numpy(region), dtype=np.int64)
    uniq, counts = np.unique(ids, return_counts=True)
    largest = int(counts.max())
    threshold = max(
        int(min_fraction * n_pts),
        int(FRAGMENT_RELATIVE_TO_LARGEST * largest),
        3,
    )
    keep_ids = uniq[counts >= threshold]
    if keep_ids.size == 0:
        return keep_largest_region(poly)
    if keep_ids.size == uniq.size:
        return poly
    keep_set = np.asarray(keep_ids, dtype=np.int64)
    _p, pts, faces = _triangle_points_faces(labelled, clean=False)
    if faces.size == 0:
        return poly
    keep_face = np.isin(ids[faces[:, 0]], keep_set)
    if int(keep_face.sum()) == len(faces):
        return poly
    return _polydata_from_triangles(pts, faces[keep_face])


def _keep_region_with_point(surface, point):
    """Keep only the connected component that contains ``point``.

    ``_drop_small_fragments`` keeps every component above 15% of the largest,
    which is exactly wrong right after a cut: a severed branch is retained as a
    floating shell, and on a bad cut the vessel body can end up as the fragment
    that is thrown away.
    """
    poly = to_vtk_poly(surface)
    if count_connected_regions(poly) <= 1:
        return clean_triangulate(poly)
    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(poly)
    conn.SetExtractionModeToClosestPointRegion()
    _set_vec3(conn.SetClosestPoint, np.asarray(point, dtype=np.float64))
    conn.Update()
    return clean_triangulate(conn.GetOutput())


def _delete_outboard_leftover(surface, origin, outward, radius, outboard_mm=0.0, clean=True,
                              clip_radius_limit=None):
    poly, pts, faces = _triangle_points_faces(surface, clean=clean)
    if faces.size == 0:
        return poly
    origin = np.asarray(origin, dtype=np.float64)
    outward = _unit(outward)
    centroids = pts[faces].mean(axis=1)
    rel = centroids - origin
    proj = rel @ outward
    radial = np.linalg.norm(rel - np.outer(proj, outward), axis=1)
    bad = (proj > outboard_mm) & (radial < _opening_clip_radius(radius, limit_mm=clip_radius_limit))
    if not np.any(bad):
        return poly
    return _polydata_from_triangles(pts, faces[~bad])


def _clip_opening_cap_locally(
    surface, origin, outward, radius, extension_length=None, trimmed=False, fast=False,
    clip_radius_limit=None,
):
    """Delete the outboard stub of one opening with a bounded cylinder."""
    region = _outboard_cap_implicit(
        origin, outward, radius, extension_length=extension_length, trimmed=trimmed,
        clip_radius_limit=clip_radius_limit,
    )
    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(_as_poly(surface) if fast else to_vtk_poly(surface))
    clipper.SetClipFunction(region)
    clipper.InsideOutOff()
    clipper.GenerateClippedOutputOff()
    clipper.Update()
    clipped = weld_clip_slivers(clean_triangulate(clipper.GetOutput()))
    if clipped.GetNumberOfPoints() == 0:
        return _as_poly(surface) if fast else to_vtk_poly(surface)
    return _delete_outboard_leftover(
        clipped, origin, outward, radius, clean=not fast,
        clip_radius_limit=clip_radius_limit,
    )


def clip_one_opening_pipe_section(
    surface, origin, outward, radius, body_point, extension_length=None, trimmed=False, fast=True,
    clip_radius_limit=None,
):
    """Open one ostium with a pipe-section cut; inset slightly if the cutter misses.

    On a pre-trimmed surface the cut may only take a collar off. Anything larger
    means the bounded cylinder reached a different part of a tortuous vessel, so
    the candidate is rejected and the cutter is moved inward instead of silently
    deleting a branch.

    ``fast=True`` (default) keeps the same cylinder clip. It skips the dual
    full-mesh loop extract on every inset and asks a local boundary question
    instead. ``fast=False`` is the old full-mesh extract, kept for comparison.
    """
    origin0 = np.asarray(origin, dtype=np.float64)
    outward = _unit(outward)
    radius = max(float(radius), 1e-3)
    n_prev = surface.GetNumberOfPoints()
    if fast:
        before = None
        area_prev = _mesh_area_fast(surface) if trimmed else 0.0
        _p0, pts0, faces0 = _triangle_points_faces(surface, clean=False)
        n_local_before = _n_boundary_points_near(pts0, faces0, origin0, radius)
    else:
        before = _n_boundary_loops(surface)
        area_prev = float(pv.wrap(to_vtk_poly(surface)).area)
        n_local_before = 0
    inset = 0.0
    # Every inset that fails does so for one of a handful of reasons, and the
    # last one is the honest account of why this ostium was left shut. Without
    # it the loop returns False in silence and the case dies several steps
    # later saying only that some end was not opened -- which is what four of
    # the fourteen variable-remesh failures looked like.
    why = "the cutter never reached the surface"
    while inset <= OPENING_CLIP_INSET_MAX_MM + 1e-12:
        origin_i = origin0 - inset * outward
        clipped = _clip_opening_cap_locally(
            surface, origin_i, outward, radius, extension_length=extension_length,
            trimmed=trimmed, fast=fast, clip_radius_limit=clip_radius_limit,
        )
        if fast:
            n_cand = clipped.GetNumberOfPoints()
            if n_cand < 50 or n_cand < 0.45 * n_prev or n_cand >= n_prev:
                why = (
                    f"the cut left {n_cand}/{n_prev} points"
                    if n_cand < n_prev
                    else "the cutter removed nothing"
                )
                inset += OPENING_CLIP_INSET_STEP_MM
                continue
        clipped = (
            _keep_region_with_point(clipped, body_point)
            if trimmed
            else (_drop_small_fragments_fast(clipped) if fast else _drop_small_fragments(clipped))
        )
        n_cand = clipped.GetNumberOfPoints()
        if n_cand < 50 or n_cand < 0.45 * n_prev:
            why = f"keeping one region left {n_cand}/{n_prev} points"
            inset += OPENING_CLIP_INSET_STEP_MM
            continue
        if trimmed and area_prev > 1e-9:
            cand_area = _mesh_area_fast(clipped) if fast else float(pv.wrap(to_vtk_poly(clipped)).area)
            lost = 1.0 - cand_area / area_prev
            if lost > CLIP_MAX_AREA_LOSS_FRACTION:
                why = f"it would take {100.0 * lost:.0f}% of the surface area"
                inset += OPENING_CLIP_INSET_STEP_MM
                continue
        if fast:
            _pc, pts_c, faces_c = _triangle_points_faces(clipped, clean=False)
            n_local = _n_boundary_points_near(pts_c, faces_c, origin_i, radius)
            opened = n_local >= MIN_OPENING_LOOP_POINTS and n_cand < n_prev
            if n_local_before >= MIN_OPENING_LOOP_POINTS:
                opened = opened and n_local > n_local_before
        else:
            loops_after = _n_boundary_loops(clipped)
            near = any(
                float(np.linalg.norm(np.asarray(op["center"]) - origin_i)) < 2.0 * radius
                for op in inspect_openings(clipped)
            )
            opened = loops_after > before or (near and n_cand < n_prev)
        if opened:
            if inset > 0:
                print(f"  [Uncap] Pipe-section clip inset {inset:.1f} mm to create a hole")
            return clipped, True
        why = "the cut went through but left no rim at the ostium"
        inset += OPENING_CLIP_INSET_STEP_MM
    print(
        f"  [Uncap] Pipe-section cut at {np.round(origin0, 2)} (r={radius:.3f} mm) "
        f"gave up after {OPENING_CLIP_INSET_MAX_MM:.1f} mm of inset: {why}"
    )
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


def _loop_geometry(surface):
    """(point ids, barycentre, radius, n_points) for every boundary loop."""
    poly, pts, faces = _triangle_points_faces(surface)
    if faces.size == 0:
        return poly, pts, faces, []
    loops = extract_boundary_loops(poly)
    locator = vtk.vtkStaticPointLocator()
    locator.SetDataSet(poly)
    locator.BuildLocator()
    out = []
    for i in range(loops.GetNumberOfCells()):
        cell = loops.GetCell(i)
        n = cell.GetNumberOfPoints()
        if n == 0:
            continue
        coords = np.array(
            [cell.GetPoints().GetPoint(j) for j in range(n)], dtype=np.float64
        )
        center = coords.mean(axis=0)
        radius = float(np.mean(np.linalg.norm(coords - center, axis=1)))
        ids = [int(locator.FindClosestPoint(xyz)) for xyz in coords]
        out.append((ids, center, radius, int(n)))
    return poly, pts, faces, out


def boundary_point_cloud(surface):
    """Every boundary point as an (N, 3) array, for provenance tests."""
    poly = to_vtk_poly(surface)
    loops = extract_boundary_loops(poly)
    out = []
    for i in range(loops.GetNumberOfCells()):
        cell = loops.GetCell(i)
        pts = cell.GetPoints()
        for j in range(cell.GetNumberOfPoints()):
            out.append(pts.GetPoint(j))
    if not out:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(out, dtype=np.float64)


def find_repair_tears(surface, before_pts, tol_mm=1e-3, min_old_fraction=0.5):
    """The openings the preparation tore, told apart from the real ostia.

    Preparing the original removes triangles -- the manifold cut, the degenerate
    and ear drops -- and removing a triangle opens a hole wherever it cuts. The
    anatomical profiles are measured afterwards, so such a hole is promoted to
    an ostium and the rest of the run is spent trying to preserve damage.
    SNF00000228 tore a 0.427 mm hole 0.57 mm from a real 0.410 mm ostium,
    counted seven profiles against the six loops its input actually has, and
    failed 6 against 7 -- while the mesh it produced was right, with all six
    openings round and in place. The remesher zips such a tear shut by itself,
    so nothing here needs repairing; the count simply must not include it.

    Position cannot separate the two at that distance: the protect tolerance
    around that ostium is 0.615 mm and the tear falls inside it. Provenance can.
    Removing triangles never invents a point, so a real ostium's rim points were
    already on a rim beforehand, while a torn rim is made of points that were
    interior. ``before_pts`` is the boundary cloud from before the preparation;
    the match runs at ``tol_mm`` rather than on exact coordinates because the
    weld nudges surviving points by well under a micron.
    """
    before = np.asarray(before_pts, dtype=np.float64)
    if before.size == 0:
        # Nothing was open to begin with, so every loop here was opened on
        # purpose -- uncap_closed_surface does that -- and none is damage.
        return []
    poly, pts, _faces, loops = _loop_geometry(surface)
    if not loops:
        return []
    from scipy.spatial import cKDTree

    tree = cKDTree(before)
    torn = []
    kept = 0
    for ids, center, radius, n in loops:
        if n <= 0:
            continue
        coords = pts[np.asarray(ids, dtype=np.int64)]
        d, _idx = tree.query(coords, k=1)
        old = int(np.count_nonzero(np.asarray(d, dtype=np.float64) <= float(tol_mm)))
        if float(old) / float(n) < float(min_old_fraction):
            torn.append({"barycenter": np.asarray(center, dtype=np.float64),
                         "radius": float(radius), "n_points": int(n)})
        else:
            kept += 1
    if kept == 0:
        # Every loop looks new, so the provenance cloud is wrong rather than the
        # surface being all tears. Claiming them all would drop every ostium.
        return []
    return torn


def reconcile_profiles_with_loops(surface, profiles, tears=(), label="preparation"):
    """Profiles cut down to the openings the surface actually has.

    Two different things put a profile in the list that no hole backs.

    VMTK's boundary reference systems walk each boundary polyline, and a rim
    pinched into a figure eight is two polylines meeting at one shared vertex,
    so a single hole is reported as two profiles. p491 keep 2 measures seven
    profiles on a surface carrying six loops: one 10-point torn rim is read as
    a 1.064 mm lobe and a 0.234 mm one.

    And preparing the original tears holes -- see ``find_repair_tears`` -- which
    are openings on the surface but not anatomy.

    Both are settled here against the loops the surface really has. Every
    profile is assigned to the loop it sits on, and where several claim the same
    loop only the best fit keeps it, because one hole is one opening. A profile
    left on a loop ``find_repair_tears`` called torn is dropped as damage.

    Chasing a phantom is not free. p491's spare 0.234 mm lobe sent the
    pipe-section uncap at a place with no hole; the fallback plane clip it then
    took amputated the real 0.583 mm ostium 2.2 mm away, and the case finished
    with four openings against six profiles.
    """
    profiles = list(profiles)
    if not profiles:
        return profiles, 0
    _poly, _pts, _faces, loops = _loop_geometry(surface)
    if not loops:
        return profiles, 0
    centers = np.asarray([c for _ids, c, _r, _n in loops], dtype=np.float64)
    radii = np.asarray([r for _ids, _c, r, _n in loops], dtype=np.float64)

    # The tears were measured by _loop_geometry on this same surface, so their
    # barycentres are the loop barycentres to the bit. Matching on position
    # rather than on an index means a reordered walk cannot mislabel an ostium.
    torn_loop = np.zeros(len(loops), dtype=bool)
    for tear in tears or ():
        d = np.linalg.norm(centers - np.asarray(tear["barycenter"], np.float64), axis=1)
        j = int(np.argmin(d))
        if float(d[j]) <= 1e-3:
            torn_loop[j] = True
        else:
            print(f"  Torn hole r={float(tear['radius']):.3f} mm no longer has a "
                  f"loop on this surface; not dropping anything for it")

    assigned, best = [], {}
    for i, profile in enumerate(profiles):
        bary = np.asarray(profile["barycenter"], dtype=np.float64)
        d = np.linalg.norm(centers - bary, axis=1)
        j = int(np.argmin(d))
        # Nearest centre plus radius mismatch, both in mm: the lobe of a pinched
        # rim is off on the radius even when its barycentre lands close.
        score = float(d[j]) + abs(float(profile["radius"]) - float(radii[j]))
        assigned.append((j, score))
        if j not in best or score < best[j][0]:
            best[j] = (score, i)

    kept, shared, torn = [], [], []
    for i, profile in enumerate(profiles):
        j, _score = assigned[i]
        if best[j][1] != i:
            shared.append(profile)
        elif torn_loop[j]:
            torn.append(profile)
        else:
            kept.append(profile)
    if not shared and not torn:
        return profiles, 0
    if len(kept) < 2:
        # A vessel needs an inlet and an outlet. If this test would leave fewer,
        # it is the test that is wrong here, so keep every profile.
        print(f"  Not dropping {len(shared) + len(torn)} profile(s) without a "
              f"loop of their own: only {len(kept)} would be left")
        return profiles, 0
    if shared:
        radii_txt = ", ".join(f"{p['radius']:.3f}" for p in shared)
        print(f"  Ignoring {len(shared)} profile(s) sharing another opening's rim "
              f"(r={radii_txt} mm)")
    if torn:
        radii_txt = ", ".join(f"{p['radius']:.3f}" for p in torn)
        print(f"  Ignoring {len(torn)} opening(s) the {label} tore "
              f"(r={radii_txt} mm)")
    print(f"  {len(kept)} anatomical ostia remain")
    return kept, len(shared) + len(torn)


def cap_unmatched_loops(surface, profiles, label="surface"):
    """Fan-fill every boundary loop that is not one of the anatomical ostia.

    remove_spurious_openings closes tears by capping the whole surface and
    re-opening the ostia, which needs vtkvmtkCapPolyData to walk every rim. When
    it cannot -- "Can't find adjacent point" -- the tears were left in place and
    the mesh shipped with more openings than it has ostia, which is exactly what
    a homogeneous training set must not contain. A triangle fan per unmatched
    loop needs no such walk.

    It only runs once every profile already has a loop of its own. Without that
    precondition an ostium the matcher failed to recognise would be capped, and
    losing a real opening is far worse than keeping a tear.
    """
    if not profiles:
        return to_vtk_poly(surface), 0
    poly, pts, faces, loops = _loop_geometry(surface)
    if not loops:
        return poly, 0

    matched_profiles = set()
    unmatched = []
    for ids, center, radius, n in loops:
        hit = None
        for k, profile in enumerate(profiles):
            if _loop_at_a_profile(center, [profile], radius=radius, n_points=n):
                hit = k
                break
        if hit is None:
            unmatched.append((ids, center, radius, n))
        else:
            matched_profiles.add(hit)

    if not unmatched:
        return poly, 0
    if len(matched_profiles) < len(profiles):
        # An ostium the matcher failed to recognise has to keep a loop, but it
        # needs only one. Refusing outright meant a single unmatched profile
        # left every tear in place -- on p551 one unmatched ostium kept 12 extra
        # openings on a mesh with 6. Each unmatched ostium reserves the loop
        # nearest to it; the rest are still closed.
        reserved = set()
        for k, profile in enumerate(profiles):
            if k in matched_profiles:
                continue
            center = np.asarray(profile["barycenter"], dtype=np.float64)
            free = [i for i in range(len(unmatched)) if i not in reserved]
            if not free:
                break
            reserved.add(
                min(free, key=lambda i: float(np.linalg.norm(unmatched[i][1] - center)))
            )
        kept = [u for i, u in enumerate(unmatched) if i not in reserved]
        print(
            f"  Reserved {len(reserved)} opening(s) on the {label} for ostia the "
            f"matcher did not recognise; capping the other {len(kept)}"
        )
        unmatched = kept
        if not unmatched:
            return poly, 0

    pts_list = pts.tolist()
    new_faces = faces.tolist()
    for ids, center, _radius, _n in unmatched:
        pts_list.append(center.tolist())
        apex = len(pts_list) - 1
        for k in range(len(ids)):
            a, b = ids[k], ids[(k + 1) % len(ids)]
            if a != b:
                new_faces.append([a, b, apex])
    capped = _polydata_from_triangles(
        np.asarray(pts_list, dtype=np.float64),
        np.asarray(new_faces, dtype=np.int64).reshape(-1, 3),
    )
    radii = ", ".join(f"{r:.3f}" for _i, _c, r, _n in unmatched)
    print(
        f"  Capped {len(unmatched)} opening(s) on the {label} that match no ostium "
        f"(r={radii} mm)"
    )
    return capped, len(unmatched)


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
        print("  Capper could not close the leftover openings; fanning them instead")
        return cap_unmatched_loops(poly, profiles, label="remeshed surface")

    ids = capped.GetCellData().GetArray("CellEntityIds")
    center_ids = capper.GetCapCenterIds()
    if ids is None or center_ids is None or center_ids.GetNumberOfIds() == 0:
        print("  Capper produced no CellEntityIds; fanning the leftovers instead")
        return cap_unmatched_loops(poly, profiles, label="remeshed surface")

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


def original_cell_mask(extended_surface, original_surface, tol=ORIGINAL_MATCH_TOL_MM):
    """Per-cell mask of ``extended_surface``: True where the cell exists on the original.

    vmtkFlowExtensions grows tubes out of every boundary loop and returns them
    fused with the input, untagged. Matching triangles by vertex identity is
    exact and does not depend on the filter's cell ordering, so the extension can
    be isolated and trimmed geometrically later on. ``tol`` is not zero because
    VMTK stores points as float32, so every coordinate comes back rounded.
    """
    ext = clean_triangulate(extended_surface)
    orig = clean_triangulate(original_surface)
    _o, opts, ofaces = _triangle_points_faces(orig)
    _e, epts, efaces = _triangle_points_faces(ext)
    if efaces.size == 0:
        return ext, np.zeros(0, dtype=bool)
    if ofaces.size == 0:
        return ext, np.zeros(len(efaces), dtype=bool)

    locator = vtk.vtkStaticPointLocator()
    locator.SetDataSet(orig)
    locator.BuildLocator()
    mapped = np.full(len(epts), -1, dtype=np.int64)
    for i, xyz in enumerate(epts):
        pid = locator.FindClosestPoint(_vec3(xyz))
        if pid >= 0 and float(np.linalg.norm(opts[pid] - xyz)) <= tol:
            mapped[i] = pid
    orig_keys = set(map(tuple, np.sort(ofaces, axis=1).tolist()))
    mapped_faces = mapped[efaces]
    ok = np.all(mapped_faces >= 0, axis=1)
    keys = np.sort(np.where(ok[:, None], mapped_faces, 0), axis=1)
    mask = np.zeros(len(efaces), dtype=bool)
    for i, good in enumerate(ok):
        if good and tuple(keys[i].tolist()) in orig_keys:
            mask[i] = True
    return ext, mask


def _clip_patch_at_plane(patch, origin, outward):
    """Keep the part of ``patch`` on the inboard side of the ostium plane."""
    plane = vtk.vtkPlane()
    _set_vec3(plane.SetOrigin, np.asarray(origin, dtype=np.float64) + EXTENSION_PLANE_TOL_MM * _unit(outward))
    _set_vec3(plane.SetNormal, _unit(outward))
    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(to_vtk_poly(patch))
    clipper.SetClipFunction(plane)
    clipper.InsideOutOn()
    clipper.GenerateClippedOutputOff()
    clipper.Update()
    return weld_clip_slivers(clean_triangulate(clipper.GetOutput()))


# How far past its own rim a flow-extension collar may reach. The collar only
# has to bridge the original rim to the ostium plane, so its reach is set by
# that rim and not by a constant; the half again is slack for an oblique cut.
EXTENSION_COLLAR_REACH_FACTOR = 1.5


def _ostium_rim_reach(original_surface, origins):
    """How far each ostium's own rim gets from its frame origin."""
    poly, _pts, _faces, loops = _loop_geometry(original_surface)
    reach = np.zeros(len(origins), dtype=np.float64)
    if not loops:
        return reach
    centers = np.asarray([c for _ids, c, _r, _n in loops], dtype=np.float64)
    for j, origin in enumerate(origins):
        k = int(np.argmin(np.linalg.norm(centers - origin, axis=1)))
        ids = loops[k][0]
        coords = np.asarray([poly.GetPoint(int(i)) for i in ids], dtype=np.float64)
        reach[j] = float(np.max(np.linalg.norm(coords - origin, axis=1)))
    return reach


def _keep_cells_within(patch, origin, radius):
    """Drop the cells of ``patch`` whose centre is further than ``radius`` out."""
    poly, pts, faces = _triangle_points_faces(patch)
    if faces.size == 0:
        return poly, 0
    centres = pts[faces].mean(axis=1)
    keep = np.linalg.norm(centres - np.asarray(origin, dtype=np.float64), axis=1) <= radius
    n_cut = int(np.count_nonzero(~keep))
    if n_cut == 0:
        return poly, 0
    return _polydata_from_triangles(pts, faces[keep]), n_cut


def trim_extension_patches(extended_surface, original_surface, frames):
    """Cut every flow-extension tube back to its own ostium plane.

    The bounded pipe-section cutter only reaches a tube that stays inside a
    1.5R cylinder around the centerline tangent. vmtkFlowExtensions extrudes
    along the *boundary normal*, so on an oblique ostium a 5 mm tube walks out
    of that cylinder and survives the uncap, which is what leaves a remeshed
    surface 1.3-1.6x the original area. Clipping each extension patch with its
    own ostium plane removes the tube whatever direction it took, and leaves
    only the collar that fills an oblique rim.
    """
    ext, mask = original_cell_mask(extended_surface, original_surface)
    if not np.any(~mask):
        return ext, 0
    _p, pts, faces = _triangle_points_faces(ext)
    body = _polydata_from_triangles(pts, faces[mask])
    ext_only = _polydata_from_triangles(pts, faces[~mask])
    if ext_only.GetNumberOfCells() == 0:
        return ext, 0

    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(ext_only)
    conn.SetExtractionModeToAllRegions()
    conn.ColorRegionsOn()
    conn.Update()
    n_regions = int(conn.GetNumberOfExtractedRegions())
    labelled = conn.GetOutput()
    region = labelled.GetPointData().GetArray("RegionId")
    if region is None or n_regions == 0:
        return ext, 0
    region_ids = vtk_to_numpy(region)

    origins = np.asarray([np.asarray(f[0], dtype=np.float64) for f in frames], dtype=np.float64)
    outwards = np.asarray([_unit(f[1]) for f in frames], dtype=np.float64)
    radii = np.asarray([float(f[2]) for f in frames], dtype=np.float64)
    # A tube starts at the rim it grew from, so the frame it belongs to is the
    # one it touches -- not the one its centroid happens to be nearest. The
    # centroid sits halfway up a 5 mm tube and on p447 that put a tube from a
    # torn hole onto an ostium 8.7 mm away, to be clipped on a plane that has
    # nothing to do with it.
    reach = _ostium_rim_reach(original_surface, origins)

    pieces = [body]
    n_trimmed = 0
    n_dropped = 0
    n_collared = 0
    for rid in range(n_regions):
        sel = vtk.vtkThreshold()
        sel.SetInputData(labelled)
        sel.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "RegionId")
        sel.SetLowerThreshold(rid - 0.5)
        sel.SetUpperThreshold(rid + 0.5)
        sel.Update()
        geom = vtk.vtkGeometryFilter()
        geom.SetInputConnection(sel.GetOutputPort())
        geom.Update()
        patch = clean_triangulate(geom.GetOutput())
        if patch.GetNumberOfCells() == 0:
            continue
        _pp, ppts = _poly_points(patch)
        touch = np.asarray(
            [float(np.min(np.linalg.norm(ppts - o, axis=1))) for o in origins],
            dtype=np.float64,
        )
        j = int(np.argmin(touch))
        if touch[j] > max(2.0 * radii[j], 1.0):
            # This tube grew out of a hole that is not an ostium -- a tear the
            # preparation left, which vmtkFlowExtensions extrudes just the same.
            # It has no plane of its own to be cut back to, and clipping it on
            # somebody else's leaves whatever happens to fall inboard attached
            # to the vessel.
            n_dropped += 1
            continue
        kept = _clip_patch_at_plane(patch, origins[j], outwards[j])
        before = patch.GetNumberOfCells()
        after = kept.GetNumberOfCells()
        if after > 0 and reach[j] > 0.0:
            # A plane can only cut a tube across if it is square to it. Where
            # the centerline tangent and the boundary normal disagree it slices
            # the tube lengthwise instead and half of it survives: on p447
            # 2123 of 4494 cells, leaving a 1.83 mm rim two and a half
            # millimetres out, which the uncap then cut a second hole beside.
            # The collar only exists to bridge this ostium's own rim to its
            # plane, so it cannot legitimately reach further out than that rim
            # does.
            kept, n_cut = _keep_cells_within(
                kept, origins[j], EXTENSION_COLLAR_REACH_FACTOR * reach[j]
            )
            if n_cut:
                n_collared += 1
            after = kept.GetNumberOfCells()
        if after < before:
            n_trimmed += 1
        if after > 0:
            pieces.append(kept)
    del region_ids

    append = vtk.vtkAppendPolyData()
    for piece in pieces:
        append.AddInputData(to_vtk_poly(piece))
    append.Update()
    merged = clean_triangulate(append.GetOutput())
    if n_trimmed:
        print(
            f"  Trimmed {n_trimmed}/{n_regions} flow-extension patch(es) back to the ostium plane"
        )
    if n_collared:
        print(
            f"  {n_collared} extension patch(es) were cut lengthwise by their own ostium "
            "plane; kept only the collar inside the rim"
        )
    if n_dropped:
        print(
            f"  Dropped {n_dropped} flow extension(s) grown out of a hole that is not an "
            "ostium; they have no plane to be cut back to"
        )
    return merged, n_trimmed


# How much of an ostium's own neighbourhood a clip somewhere else is allowed to
# take. A clip keeps one connected region, so a cut that severs a branch throws
# the whole branch away with it -- measured, the ostium on the severed branch
# keeps 0% of its points while every legitimate clip elsewhere leaves 86-95%.
# Half is well clear of both.
CLIP_NEIGHBOUR_KEEP_FRACTION = 0.5


def _ostium_neighbourhoods(surface, ostia):
    """How many mesh points sit inside each ostium's own neighbourhood."""
    if not ostia:
        return np.zeros(0, dtype=np.int64)
    _poly, pts = _poly_points(surface)
    if pts.size == 0:
        return np.zeros(len(ostia), dtype=np.int64)
    from scipy.spatial import cKDTree

    tree = cKDTree(pts)
    return np.asarray(
        [len(tree.query_ball_point(origin, radius)) for origin, radius in ostia],
        dtype=np.int64,
    )


def _clip_took_another_ostium(before, after, mine, ostia):
    """Which other ostium this clip destroyed, if any.

    Each clip keeps the one connected region holding the body point, so a cut
    that happens to sever a branch deletes that branch entire -- with its
    ostium. UPF_P0258's inlet cut took the whole stub carrying the 0.658 mm
    ostium nine millimetres away and left one 2.76 mm crater where two openings
    had been, and SNF00000143 keep 1 lost a 0.713 mm ostium the same way. The
    old radius band could not see it: it allows r_gt + 2.5 mm, which is wider
    than the crater. What it cannot allow is another ostium's surface going
    missing, and that is what this asks.
    """
    for j in range(len(ostia)):
        if j == mine or before[j] == 0:
            continue
        if after[j] < CLIP_NEIGHBOUR_KEEP_FRACTION * before[j]:
            return j, float(after[j]) / float(before[j])
    return None, 0.0


def unopened_profiles(surface, profiles, reach_factor=2.0, min_reach_mm=1.0):
    """Which ostia the finished surface still has no opening near.

    The uncap's own counter says how many clips succeeded; it cannot say which
    end is missing, and "opened 14/15" is not something anyone can act on. This
    reads the answer off the result instead: an ostium is open if some boundary
    loop's centre sits within ``max(min_reach_mm, reach_factor * r)`` of it.
    """
    loops = inspect_openings(surface)
    centres = np.asarray([np.asarray(op["center"], dtype=np.float64) for op in loops])         if loops else np.zeros((0, 3))
    missing = []
    for profile in profiles:
        bary = np.asarray(profile["barycenter"], dtype=np.float64)
        reach = max(float(min_reach_mm), reach_factor * float(profile["radius"]))
        if centres.shape[0] and float(np.min(np.linalg.norm(centres - bary, axis=1))) <= reach:
            continue
        missing.append(profile)
    return missing


def describe_unopened(profiles):
    return "; ".join(
        f"profile {p['index']} r={float(p['radius']):.3f} mm at "
        f"{np.round(np.asarray(p['barycenter'], dtype=np.float64), 2).tolist()}"
        for p in profiles
    )


def _split_lumen_note(surface, profiles, suspect):
    """Say so when the openings that stayed shut are in a lumen of their own.

    ``suspect`` are the profiles with no rim near them. If the volume test puts
    them in a different compartment from the rest, the input encloses more than
    one lumen and the wall between them is a double wall with nothing through
    it -- the same defect as p463 keep 1. Nothing downstream can repair that,
    so the message has to point upstream.
    """
    if not suspect or not profiles:
        return ""
    try:
        lumens = ostium_lumen_compartments(surface, profiles)
    except Exception:
        return ""
    if not lumens or len(set(lumens)) <= 1:
        return ""
    by_index = {int(p["index"]): lumen for p, lumen in zip(profiles, lumens)}
    shut = {int(p["index"]) for p in suspect}
    main = Counter(l for i, l in by_index.items() if i not in shut).most_common(1)
    if not main:
        return ""
    main_lumen = main[0][0]
    stranded = [i for i in sorted(shut) if by_index.get(i) != main_lumen]
    if not stranded:
        return ""
    return (
        f" This vessel encloses {len(set(lumens))} separate lumens and "
        f"profile(s) {', '.join(str(i) for i in stranded)} open into a compartment "
        "of their own, walled off from the one the rest share. No centerline can "
        "cross a double wall, so the tube has no branch there to cut. The input "
        "mesh is what has to be fixed."
    )


OSTIUM_ROUND_ENOUGH = 0.80


def rim_circularity_at(surface, origin, reach_mm=6.0):
    """How round the boundary nearest ``origin`` is, 1.0 for a circle.

    A pipe-section cut that meets the wall at a glancing angle, or that runs
    into a second rim a millimetre away, opens a long ragged hole instead of an
    ostium and nothing downstream can undo it: on p531 the uncap turned a
    0.24 mm opening into a 199-point rim 21.2 mm around at circularity 0.199,
    and the remesh and the repairs carried it through to the shipped surface at
    0.361. Measuring the rim the clip just made is what lets the fallback clip
    be tried on its merits rather than only when the first one fails outright.
    """
    ring = extract_boundary_loops(to_vtk_poly(surface))
    best = None
    origin = np.asarray(origin, dtype=np.float64)
    for k in range(ring.GetNumberOfCells()):
        cell = ring.GetCell(k)
        co = np.array(
            [ring.GetPoint(cell.GetPointId(j)) for j in range(cell.GetNumberOfPoints())],
            dtype=np.float64,
        )
        if len(co) < 3:
            continue
        d = float(np.linalg.norm(co.mean(axis=0) - origin))
        if d > float(reach_mm):
            continue
        perimeter = float(np.sum(np.linalg.norm(co - np.roll(co, -1, axis=0), axis=1)))
        if perimeter <= 0:
            continue
        rel = co - co.mean(axis=0)
        area = 0.5 * float(np.linalg.norm(np.cross(rel, np.roll(rel, -1, axis=0)).sum(axis=0)))
        circ = 4.0 * np.pi * area / (perimeter * perimeter)
        if best is None or d < best[0]:
            best = (d, circ)
    return 1.0 if best is None else float(best[1])


def clip_flow_extensions_and_uncap(
    base_surface,
    profiles,
    extension_length=DEFAULT_EXTENSION_LENGTH,
    centerline=None,
    unextended_surface=None,
    fast_uncap=True,
    cut_frames=None,
):
    """Pipe-section uncap at each ostium.

    ``cut_frames``: optional non-empty ``list[dict]`` of GT ostium frames
    (``origin``, unit ``normal``, ``radius``). When given, those planes are
    used instead of ``opening_clip_frames(centerline, profiles)``. Empty or
    ``None`` keeps the previous profile/centerline behaviour.
    """
    current = to_vtk_poly(base_surface)
    if cut_frames:
        work_profiles = _profiles_from_ostium_frames(cut_frames)
        frames = [
            (p["barycenter"], p["normal"], float(p["radius"])) for p in work_profiles
        ]
        print(f"  Uncap: using {len(frames)} supplied ostium cut frame(s)")
    else:
        work_profiles = list(profiles)
        frames = opening_clip_frames(centerline, work_profiles) if centerline is not None else None
    trimmed = False
    if unextended_surface is not None and frames is not None:
        current, _n_trimmed = trim_extension_patches(current, unextended_surface, frames)
        trimmed = True
    body_pt = mesh_body_point(current)
    n_clipped = 0
    # Each ostium's own patch of surface, so a clip can be asked whether it took
    # one of the others with it. The radius is generous on purpose: a clip is
    # rejected only when a neighbour's surface is gone, not when it is nicked.
    if frames is not None:
        ostia = [(np.asarray(o, dtype=np.float64), max(1.0, 2.0 * float(r)))
                 for o, _n, r in frames]
    else:
        ostia = [(np.asarray(p["barycenter"], dtype=np.float64),
                  max(1.0, 2.0 * float(p["radius"]))) for p in work_profiles]
    near_before = _ostium_neighbourhoods(current, ostia)

    def _accept(candidate, i, what):
        """Take the clip unless it took another ostium's surface away."""
        nonlocal near_before
        near_after = _ostium_neighbourhoods(candidate, ostia)
        j, frac = _clip_took_another_ostium(near_before, near_after, i, ostia)
        if j is not None:
            print(
                f"  [Uncap] Profile {work_profiles[i]['index']} {what} rejected: it "
                f"takes away {100.0 * (1.0 - frac):.0f}% of the surface around the "
                f"r={0.5 * ostia[j][1]:.3f} mm ostium at {np.round(ostia[j][0], 2)}, "
                "so it would sever that branch rather than open this end."
            )
            return False
        near_before = near_after
        return True

    for i, profile in enumerate(work_profiles):
        search_mm = float(extension_length) + 3.0 * max(float(profile["radius"]), 0.5)
        where = (np.asarray(frames[i][0], dtype=np.float64) if frames is not None
                 else np.asarray(profile["barycenter"], dtype=np.float64))
        before = current
        near_at_entry = near_before
        ok = False
        if frames is not None:
            origin, outward, radius = frames[i]
            # Narrow the cutter rather than let it reach the branch next door
            # and be thrown away for it.
            limit = clip_radius_limit_for(
                origin, outward, radius, ostia, i,
                extension_length=extension_length, trimmed=trimmed,
            )
            candidate, ok = clip_one_opening_pipe_section(
                current,
                origin,
                outward,
                radius,
                body_pt,
                extension_length=extension_length,
                trimmed=trimmed,
                fast=fast_uncap,
                clip_radius_limit=limit,
            )
            if ok:
                ok = _accept(candidate, i, "pipe-section cut")
            if ok:
                current = candidate
                print(
                    f"  [Uncap] Profile {profile['index']} pipe-section cut "
                    f"r={radius:.3f} mm at {np.round(origin, 2)}"
                )
        ragged = rim_circularity_at(current, where) if ok else 1.0
        if ragged < OSTIUM_ROUND_ENOUGH:
            # The cut was taken, but the hole it made is not an opening shape.
            # Try the plane clip from the surface as it stood and keep whichever
            # rim is rounder; on p531 that is the difference between a 0.199 rim
            # 21.2 mm around and the ostium the input actually has.
            alternative, alt_ok = clip_one_profile(before, profile, body_pt, search_mm)
            if alt_ok:
                rounder = rim_circularity_at(alternative, where)
                # _accept measures against the surface as it was before this
                # ostium was cut, and the pipe-section cut has already moved
                # that baseline forward; the alternative starts from the same
                # place the pipe cut did, so the baseline goes back with it.
                near_before = near_at_entry
                if rounder > ragged and _accept(alternative, i, "anatomical plane clip"):
                    current = alternative
                    print(
                        f"  [Uncap] Profile {profile['index']} pipe-section cut left a "
                        f"rim at circularity {ragged:.3f}; the plane clip gives "
                        f"{rounder:.3f} and was taken instead"
                    )
                else:
                    near_before = _ostium_neighbourhoods(current, ostia)
                    print(
                        f"  [Uncap] Profile {profile['index']} rim is ragged "
                        f"(circularity {ragged:.3f}); the plane clip is no better "
                        f"({rounder:.3f}), so the pipe-section cut stands"
                    )
        if not ok:
            candidate, ok = clip_one_profile(current, profile, body_pt, search_mm)
            if ok:
                ok = _accept(candidate, i, "anatomical plane clip")
            if ok:
                current = candidate
                print(
                    f"  [Uncap] Profile {profile['index']} fell back to anatomical plane clip"
                )
        if ok:
            n_clipped += 1
            body_pt = mesh_body_point(current)
    print(f"  Uncap: clipped {n_clipped}/{len(work_profiles)} openings")
    if n_clipped < len(work_profiles):
        still_shut = unopened_profiles(current, work_profiles)
        if still_shut:
            print(f"  Uncap: still shut -> {describe_unopened(still_shut)}")
    current = clean_triangulate(current)
    current, n_nm = repair_nonmanifold_triangles(current)
    if n_nm > 0:
        # A vtkClipPolyData seam can leave two full-size sheets on one edge, which
        # the flap heuristic will not touch. Cut them apart instead of failing the
        # case: the remesher only needs a manifold input, and the holes this opens
        # are sub-triangle sized.
        current, _n_forced = force_manifold_triangles(current)
        current, n_nm = repair_nonmanifold_triangles(current)
        if n_nm > 0:
            raise TemplateQualityError(
                f"Uncapped parent tube has {n_nm} non-manifold edges; remesh would amplify them"
            )
    current, _n_regions = drop_tiny_islands(current)
    current, _n_pin = close_wall_pinholes(
        current, label="uncapped surface", profiles=work_profiles
    )
    current, n_filled = remove_spurious_openings(current, work_profiles)
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
    cell_ids = []
    for ci in range(vtk_cl.GetNumberOfCells()):
        cell = vtk_cl.GetCell(ci)
        if cell.GetCellType() not in (vtk.VTK_LINE, vtk.VTK_POLY_LINE):
            continue
        n = cell.GetNumberOfPoints()
        if n < 2:
            continue
        pts = np.array([vtk_cl.GetPoint(cell.GetPointId(j)) for j in range(n)], dtype=np.float64)
        cells.append(pts)
        cell_ids.append(ci)
    return cells, cell_ids, vtk_cl


def _copy_cell_arrays_for_kept_cells(src_poly, dst_poly, kept_cell_ids):
    """Copy VMTK branch cell arrays (GroupIds, Blanking, ...) onto rebuilt cells."""
    cd = src_poly.GetCellData()
    n_arr = int(cd.GetNumberOfArrays())
    if n_arr == 0 or dst_poly.GetNumberOfCells() == 0:
        return
    ids = np.asarray(kept_cell_ids, dtype=np.int64)
    if ids.size != int(dst_poly.GetNumberOfCells()):
        raise TemplateQualityError(
            "Centerline clip cell-array copy length mismatch "
            f"(kept {ids.size} ids, {dst_poly.GetNumberOfCells()} output cells)."
        )
    for ai in range(n_arr):
        src = cd.GetArray(ai)
        if src is None:
            continue
        name = src.GetName()
        if not name:
            continue
        try:
            values = vtk_to_numpy(src)
        except (ValueError, TypeError, AttributeError):
            dst = src.NewInstance()
            dst.SetName(name)
            dst.SetNumberOfComponents(src.GetNumberOfComponents())
            dst.SetNumberOfTuples(int(ids.size))
            for j, ci in enumerate(ids.tolist()):
                dst.SetTuple(int(j), src.GetTuple(int(ci)))
            dst_poly.GetCellData().AddArray(dst)
            continue
        kept = np.ascontiguousarray(values[ids])
        arr = numpy_to_vtk(kept, deep=True)
        arr.SetName(name)
        dst_poly.GetCellData().AddArray(arr)


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
    """Trim only polyline ends that belong to a nearby opening (never a global AABB).

    Each kept cell retains its VMTK cell arrays (GroupIds, Blanking,
    CenterlineIds, TractIds). Point arrays are copied from the nearest original
    vertex as before.
    """
    cells, cell_ids, vtk_cl = _polyline_cells(centerline)
    kept = []
    kept_cell_ids = []
    for pts, ci in zip(cells, cell_ids):
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
            kept_cell_ids.append(ci)
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
    _copy_cell_arrays_for_kept_cells(vtk_cl, out, kept_cell_ids)
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


def _cell_centroids(poly):
    n = int(poly.GetNumberOfCells())
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)
    centers = vtk.vtkCellCenters()
    centers.SetInputData(poly)
    centers.VertexCellsOff()
    centers.Update()
    out = centers.GetOutput()
    if out.GetNumberOfPoints() == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return np.ascontiguousarray(vtk_to_numpy(out.GetPoints().GetData()), dtype=np.float64)


# How much of a surface has to agree with the radial test before its sign is
# taken from it. vtkPolyDataNormals ran with ConsistencyOn, so the array is
# already coherent and only its global sign is unknown; the vote only has to
# read that one bit. On C0033 the GT votes 97.7% and the parent tube 99.5%.
RADIAL_ORIENTATION_MAJORITY = 0.80


def _orient_along_radial(points, vectors, cl_points):
    """Give ``vectors`` the global sign that points away from the centerline.

    The sign is decided once for the whole array, not once per item. That is
    the only thing the radial test can be trusted with: vtkPolyDataNormals ran
    with ConsistencyOn, so neighbouring normals already agree with each other
    and only the global sense is in doubt, while (x - nearest centerline
    vertex) is a weak proxy that goes the wrong way wherever a sac overhangs
    the vessel it grew from or a rim faces along the trace. Flipping item by
    item took the coherent GT normals on C0033 and turned 7914 of 348557 cells
    against their own neighbours, some of them on a dot product of 8e-05 -- and
    such a cell then passes or fails the hit-agreement test on a coin toss.

    Returns ``(vectors, False)`` when the vote is not decisive, which leaves
    every caller on its orientation-free path rather than acting on a guess.
    """
    from scipy.spatial import cKDTree

    n = int(points.shape[0])
    if n == 0 or cl_points is None or int(cl_points.shape[0]) == 0:
        return vectors, False
    tree = cKDTree(cl_points)
    _, nn = tree.query(points)
    nn = np.asarray(nn, dtype=np.int64).reshape(-1)
    radial = points - cl_points[nn]
    dots = np.einsum("ij,ij->i", vectors, radial)
    n_out = int(np.count_nonzero(dots > 0.0))
    if max(n_out, n - n_out) < RADIAL_ORIENTATION_MAJORITY * n:
        return vectors, False
    oriented = np.ascontiguousarray(vectors, dtype=np.float64)
    if n_out * 2 < n:
        oriented = np.ascontiguousarray(-oriented, dtype=np.float64)
    return oriented, True


# A ray that finds nothing writes the same 0.0 as a template already sitting on
# the GT, and 0.0 travels downstream as "no stretch here, do not refine" --
# build_target_edge_array turns it into k = 1, the coarsest edge the local
# radius allows. So the one place the template deviates most from the GT is the
# one place guaranteed to get the fewest triangles. It is not a rare corner:
# measured over 35 cases, every single one carried points like this, a median of
# 18 per case more than 0.3 mm off the GT, and on p131 one sits 6.5 mm off and
# is recorded as zero.
#
# What makes an outward ray find nothing is the tube poking OUTSIDE the GT --
# the ray then travels away from the surface it is looking for and runs to the
# 25 mm cap. The other silent zeroes are the hit landing past 3.5 R or the GT
# normal disagreeing.
#
# A zero that is honest is always backed by a hit within 0.10 mm outward or
# 0.40 mm inward, so a true distance past 0.40 mm can only mean the cast gave
# up, and the true distance is the answer it should have had. Points are pruned
# against the nearest GT *vertex* first, which is never nearer than the GT
# surface, so the exact query only runs for the handful that survive.
STRETCH_HONEST_ZERO_MM = 0.40


def _rescue_missed_rays(distances, locator, template_pts, r_arr,
                        gt_cell_normals=None, gt_oriented=False, report=True):
    """Give the points whose ray missed their true distance instead of zero.

    Only points the GT encloses are recovered. StretchDistance is signed in
    everything downstream -- train_pipeline builds r* = r_local +
    StretchDistance -- so writing the unsigned gap for a template point that
    sits OUTSIDE the GT puts r* wrong by twice its depth and teaches the
    decoder to push further out exactly where the tube already bulges through.
    Those points keep the honest 0.0: the GT is not outward of them at all.

    Measured, the outside patches are a thin skin and not the blow-out they
    could have been -- median depth 0.06-0.08 mm, deepest 0.51 mm across four
    cases -- so most sit under the rescue threshold regardless: 11 of the 47
    rescued points on ANSYS_UNIGE_09, none at all on p131, p414 or SNF00000267.

    The side test needs normals that really do point outward, which is what
    _orient_along_radial returns; without them the side is unknowable, so the
    rescue stands down rather than guess.
    """
    sign_ok = gt_oriented and gt_cell_normals is not None
    zero = np.flatnonzero(distances <= 0.0)
    if zero.size == 0:
        return distances
    gt = locator.GetDataSet()
    gt_pts = np.ascontiguousarray(vtk_to_numpy(gt.GetPoints().GetData()), dtype=np.float64)
    if gt_pts.shape[0] == 0:
        return distances
    from scipy.spatial import cKDTree

    near_vertex = cKDTree(gt_pts).query(template_pts[zero])[0]
    suspect = zero[near_vertex > STRETCH_HONEST_ZERO_MM]
    if suspect.size == 0:
        return distances

    closest = [0.0, 0.0, 0.0]
    cell_id = vtk.mutable(0)
    sub_id = vtk.mutable(0)
    d2 = vtk.mutable(0.0)
    true_d = np.empty(suspect.size, dtype=np.float64)
    inside = np.ones(suspect.size, dtype=bool)
    for k, i in enumerate(suspect):
        p = template_pts[i]
        locator.FindClosestPoint((float(p[0]), float(p[1]), float(p[2])),
                                 closest, cell_id, sub_id, d2)
        true_d[k] = float(np.sqrt(max(float(d2.get()), 0.0)))
        if sign_ok:
            c = int(cell_id.get())
            if 0 <= c < gt_cell_normals.shape[0]:
                nz = gt_cell_normals[c]
                inside[k] = (
                    (float(p[0]) - closest[0]) * nz[0]
                    + (float(p[1]) - closest[1]) * nz[1]
                    + (float(p[2]) - closest[2]) * nz[2]
                ) <= 0.0
    outside_left = int(np.sum((true_d > STRETCH_HONEST_ZERO_MM) & ~inside))
    keep = (true_d > STRETCH_HONEST_ZERO_MM) & inside
    if not np.any(keep):
        return distances
    idx = suspect[keep]
    # Clamped to the same 3.5 R ceiling the ray path uses, so a stray patch of
    # tube cannot demand a twentyfold refinement of a region that is mostly the
    # tube's own fault.
    ceiling = 3.5 * (np.ones(idx.size) if r_arr is None else r_arr[idx])
    distances[idx] = np.minimum(true_d[keep], ceiling)
    if report:
        print(
            f"  Raycast: {idx.size} point(s) whose ray found no GT within "
            f"{STRETCH_HONEST_ZERO_MM:.2f} mm recovered by closest-point "
            f"(max {float(true_d[keep].max()):.3f} mm, "
            f"{int(np.sum(true_d[keep] > ceiling))} clamped at 3.5 R, "
            f"{outside_left} left at zero as outside the GT)"
        )
    return distances


# How far from an open rim of the ground truth a positive stretch is suspect.
#
# Beyond the rim the ground truth has simply stopped, and a template point out
# there has no GT wall outward of it to measure -- but its ray can still graze
# the outside of the wall near the rim and come back with a distance, and the
# side test cannot catch it because there is no sheet behind the point to
# probe. StretchDistance is signed downstream (r* = r_local + StretchDistance),
# so every one of those is an instruction to push the decoder further out
# exactly where the template already sits outside the vessel.
#
# Measured on C0033 with vtkSelectEnclosedPoints against the capped GT as the
# truth: 108 of 125458 points, 0.086%, carried a positive distance while
# outside, worst 3.394 mm; 107 of them sat within 2 mm of an open rim and the
# furthest was 3.394 mm out. Five millimetres is a comfortable margin over
# that and still leaves the test on a small subset -- about 1500 points on
# these meshes, 0.1 s -- where testing every point costs 2.3-4.5 s.
#
# Nothing is guessed here: the points inside the net are settled by enclosure,
# which is the question the signed distance actually asks. Points outside the
# net keep whatever the ray found, because away from a rim a template point
# that is outside the GT is a real tube blow-out, and the inward side test
# already recognises those -- it has a wall to probe.
STRETCH_RIM_GUARD_REACH_MM = 5.0


def _open_boundary_points(surface):
    """The points on this surface's open rims, if it has any."""
    fe = vtk.vtkFeatureEdges()
    fe.SetInputData(to_vtk_poly(surface))
    fe.BoundaryEdgesOn()
    fe.FeatureEdgesOff()
    fe.NonManifoldEdgesOff()
    fe.ManifoldEdgesOff()
    fe.Update()
    _poly, pts = _poly_points(fe.GetOutput())
    return pts


def _enclosed_by(closed_surface, points):
    """Which of ``points`` the closed surface contains."""
    probe = vtk.vtkPolyData()
    vtk_pts = vtk.vtkPoints()
    vtk_pts.SetData(numpy_to_vtk(np.ascontiguousarray(points, dtype=np.float64), deep=True))
    probe.SetPoints(vtk_pts)
    sel = vtk.vtkSelectEnclosedPoints()
    sel.SetInputData(probe)
    sel.SetSurfaceData(to_vtk_poly(closed_surface))
    sel.SetTolerance(0.0)
    sel.CheckSurfaceOff()
    sel.Update()
    return np.asarray([bool(sel.IsInside(i)) for i in range(points.shape[0])], dtype=bool)


def _zero_rays_that_left_the_gt(distances, ground_truth, template_pts, report=True):
    """Zero the positive distances at template points the GT does not enclose.

    See STRETCH_RIM_GUARD_REACH_MM for what this catches and why the net is
    drawn around the rims rather than the whole surface.

    It is a rescue in the same sense as the others: it only ever takes a
    distance away, it only looks at points that already have one, and it stands
    down rather than guess. If the ground truth is closed there is no rim and
    nothing to do; if the cap does not close it -- this pipeline's capper is
    known to leave rims on some inputs -- the enclosure answer is meaningless
    and the distances are left exactly as the ray found them, with a line in
    the log saying so.
    """
    pos = np.flatnonzero(np.asarray(distances) > 0.0)
    if pos.size == 0:
        return distances
    rim_pts = _open_boundary_points(ground_truth)
    if rim_pts.shape[0] == 0:
        return distances
    from scipy.spatial import cKDTree

    rim_d = cKDTree(rim_pts).query(template_pts[pos])[0]
    near = pos[rim_d <= STRETCH_RIM_GUARD_REACH_MM]
    if near.size == 0:
        return distances
    try:
        capped = to_vtk_poly(cap_surface(ground_truth))
    except Exception as exc:
        print(f"  WARNING: could not cap the ground truth to check which rays left it "
              f"({type(exc).__name__}); {near.size} near-rim distances left as measured")
        return distances
    if _open_boundary_points(capped).shape[0] > 0:
        print(f"  WARNING: the capped ground truth is still open, so enclosure cannot "
              f"be trusted; {near.size} near-rim distances left as measured")
        return distances
    outside = near[~_enclosed_by(capped, template_pts[near])]
    if outside.size == 0:
        return distances
    worst = float(np.max(distances[outside]))
    distances[outside] = 0.0
    if report:
        print(f"  Raycast: {outside.size} point(s) within "
              f"{STRETCH_RIM_GUARD_REACH_MM:.1f} mm of an open rim measured a stretch "
              f"while sitting outside the GT (worst {worst:.3f} mm); zeroed")
    return distances


def compute_raycast_stretch_distances(
    template_mesh,
    ground_truth_mesh,
    r_template=None,
    max_ray_length=25.0,
    tol=1e-4,
    centerline=None,
):
    """Outward MISR-tube stretch vs GT.

    Ray direction and GT cell normals are oriented by the sign of
    ``n · (x − cl_nearest)`` when a centerline is given, so AneuX files wound
    inward still count as legitimate outward hits. Without a centerline the
    hit test falls back to ``abs(dot) > 0.2``.
    """
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
    unit_n = template_normals / lens

    cl_pts = None
    if centerline is not None:
        _cl_poly, cl_pts = _poly_points(centerline)
        if cl_pts.shape[0] == 0:
            cl_pts = None

    radially_oriented = False
    gt_radially_oriented = False
    if cl_pts is not None:
        outward, radially_oriented = _orient_along_radial(template_pts, unit_n, cl_pts)
        if gt_cell_normals is not None:
            centroids = _cell_centroids(gt_mesh_with_normals)
            if centroids.shape[0] == gt_cell_normals.shape[0]:
                gt_cell_normals, gt_radially_oriented = _orient_along_radial(
                    centroids, gt_cell_normals, cl_pts
                )
    else:
        # Historical MC surfaces are inward-wound; abs(dot) below covers GT winding.
        outward = -unit_n

    n_pts = template_pts.shape[0]
    r_arr = None if r_template is None else np.ascontiguousarray(r_template, dtype=np.float64)
    use_c = (
        radially_oriented
        and (gt_cell_normals is None or gt_radially_oriented)
        and _stretch_raycast_c is not None
    )
    distances = None
    if use_c:
        try:
            distances = np.asarray(
                _stretch_raycast_c(
                    _vtk_c_address(locator, "vtkCellLocator"),
                    template_pts,
                    outward,
                    r_arr if r_arr is not None else None,
                    gt_cell_normals,
                    float(max_ray_length),
                    float(tol),
                ),
                dtype=np.float64,
            )
        except Exception as exc:
            print(f"  WARNING: compiled raycast failed ({exc}); using Python loop")

    if distances is not None:
        distances = _rescue_missed_rays(
            distances, locator, template_pts, r_arr, gt_cell_normals, gt_radially_oriented
        )
        return _zero_rays_that_left_the_gt(distances, gt_mesh_with_normals, template_pts)

    distances = np.zeros(n_pts, dtype=np.float64)
    t = vtk.mutable(0.0)
    x = [0.0, 0.0, 0.0]
    pcoords = [0.0, 0.0, 0.0]
    sub_id = vtk.mutable(0)
    cell_id = vtk.mutable(0)

    use_side_test = gt_cell_normals is not None and gt_radially_oriented
    for i in range(n_pts):
        p = template_pts[i]
        n = outward[i]
        r_local = 1.0 if r_arr is None else float(r_arr[i])
        p0 = (float(p[0]), float(p[1]), float(p[2]))
        # Reaches as far inward as an outward hit is allowed to be accepted, so
        # a tube poking through the GT is recognised at any depth at which the
        # outward ray could still find something.
        probe = max(1.5, 3.5 * r_local)
        p_inward = (
            float(p[0] - n[0] * probe),
            float(p[1] - n[1] * probe),
            float(p[2] - n[2] * probe),
        )
        hit_inward = locator.IntersectWithLine(p0, p_inward, tol, t, x, pcoords, sub_id, cell_id)
        if hit_inward:
            d_inward = float(np.sqrt((x[0] - p[0]) ** 2 + (x[1] - p[1]) ** 2 + (x[2] - p[2]) ** 2))
            cid_in = int(cell_id.get())
            if use_side_test and 0 <= cid_in < len(gt_cell_normals):
                # Which sheet the inward ray met settles the side; distance
                # cannot. A wall facing back at us is this point's own near
                # wall, so the tube is outside the GT here and there is no
                # outward stretch to find -- writing one would be wrong by
                # twice the depth, since r* = r_local + StretchDistance
                # downstream. A wall facing away is the far side of the lumen,
                # so the point is inside however close that far side happens to
                # be; the old test read 0.4 mm there and zeroed a real stretch
                # in any vessel narrower than that.
                if float(np.dot(n, gt_cell_normals[cid_in])) > 0.2:
                    distances[i] = 0.0
                    continue
            elif d_inward < 0.4:
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
                if gt_cell_normals is None:
                    distances[i] = d
                elif 0 <= cid < len(gt_cell_normals):
                    agree = float(np.dot(n, gt_cell_normals[cid]))
                    if (agree > 0.2) if gt_radially_oriented else (abs(agree) > 0.2):
                        distances[i] = d
                # A hit whose cell the normals array does not cover cannot be
                # checked for orientation, so it is refused rather than trusted.
                # This used to fall into the branch that accepts any distance,
                # which is the one place the two raycast implementations gave
                # different answers; stretch_raycast.cpp has always refused it.
    distances = _rescue_missed_rays(
        distances, locator, template_pts, r_arr, gt_cell_normals, gt_radially_oriented
    )
    return _zero_rays_that_left_the_gt(distances, gt_mesh_with_normals, template_pts)


def build_target_edge_array(template_mesh, distances, r_template, base_edge=0.50, min_edge=0.01):
    """Stretch densifies aneurysms; local radius caps edge length so thin tubes stay round."""
    vtk_poly = to_vtk_poly(template_mesh)
    r = np.maximum(R_TEMPLATE_FLOOR_MM, np.asarray(r_template, dtype=np.float64))
    stretch_factors = 1.0 + (distances / r)
    radius_limited = np.minimum(base_edge, np.maximum(min_edge, CIRCUMFERENTIAL_EDGE_OVER_RADIUS * r))
    target_edge_lengths = np.maximum(min_edge, radius_limited / (stretch_factors ** 1.5))

    vtk_target_array = numpy_to_vtk(np.ascontiguousarray(target_edge_lengths, dtype=np.float64), deep=True)
    vtk_target_array.SetName("TargetEdgeLength")
    vtk_poly.GetPointData().AddArray(vtk_target_array)
    return vtk_poly, target_edge_lengths, stretch_factors


# Angle-based edge collapse is what tears these surfaces, so it is switched off.
#
# vtkvmtkPolyDataSurfaceRemeshing collapses a triangle whose smallest angle
# falls under this threshold, in radians; the VTK class defaults to 0.5 and the
# vmtk wrapper lowers it to 0.2, which is what this pipeline has been running.
# What it does on a vessel is open slits -- the collapse takes a sliver out and
# leaves a hole -- which the remesher then stitches shut with a fan of enormous
# triangles hung off one distant vertex. That fan is precisely what the hub gate
# rejects, and it is also where essentially all of the area drift lives: on p305
# the output came out 974.04 mm^2 against an input of 848.88, and 95.7% of that
# 125 mm^2 excess sat in the 383 triangles above five times the median area.
#
# Setting it to 0 stops the angle collapse and leaves the length-driven collapse
# and split alone. Measured against the default on the two captured failures,
# same surfaces, same iterations:
#
#                        fan reach     area      edge CV   oversized   non-manifold
#   p305       0.2          35.14     1.1474      0.935        383          10
#              0.0           0.00     0.9961      0.155          0           0
#   SNF208     0.2          29.53     0.9964      0.798        108          12
#              0.0           0.87     0.9714      0.275          0           0
#
# Every one of the four knobs the pipeline had never set was swept across five
# values each on p305, and nothing else came within 16 of the gate; this is the
# only setting that produced a mesh worth shipping, and it runs faster (44 s
# against 72 s on SNF208) because the remesher stops undoing its own work.
#
# It is still not free, so it is a rescue and not the default. On UPF_P0048,
# which converges perfectly well as things stand, turning the angle collapse off
# moves the edge-length CV from 0.129 to 0.144 -- a small loss of exactly the
# uniformity this dataset exists to provide. So the sweep below keeps vmtk's
# 0.2 for the first attempt, and a surface that already converges is remeshed
# byte for byte as before.
REMESH_COLLAPSE_ANGLE = 0.2
REMESH_COLLAPSE_ANGLE_OFF = 0.0


def remesh_surface_adaptively(
    open_surface_with_array,
    edge_array_name="TargetEdgeLength",
    n_iter=REMESH_N_ITER,
    connectivity_iter=REMESH_CONNECTIVITY_ITER,
    collapse_angle=REMESH_COLLAPSE_ANGLE,
):
    remesher = vmtkscripts.vmtkSurfaceRemeshing()
    remesher.Surface = to_vtk_poly(open_surface_with_array)
    remesher.ElementSizeMode = "edgelengtharray"
    remesher.TargetEdgeLengthArrayName = edge_array_name
    remesher.PreserveBoundaryEdges = 1
    remesher.NumberOfIterations = int(n_iter)
    remesher.NumberOfConnectivityOptimizationIterations = int(connectivity_iter)
    remesher.MinEdgeLength = float(REMESH_MIN_EDGE_MM)
    # The default is vmtk's own 0.2, so a template that already converges is
    # remeshed exactly as before; the variable path lowers it only as a rescue.
    remesher.CollapseAngleThreshold = float(collapse_angle)
    remesher.Execute()
    return to_vtk_poly(remesher.Surface)


def remesh_surface_isotropically(
    open_surface,
    target_edge_length=0.5,
    n_iter=REMESH_N_ITER,
    connectivity_iter=REMESH_CONNECTIVITY_ITER,
    collapse_angle=REMESH_COLLAPSE_ANGLE,
):
    remesher = vmtkscripts.vmtkSurfaceRemeshing()
    remesher.Surface = to_vtk_poly(open_surface)
    remesher.ElementSizeMode = "edgelength"
    remesher.TargetEdgeLength = float(target_edge_length)
    remesher.PreserveBoundaryEdges = 1
    remesher.NumberOfIterations = int(n_iter)
    remesher.NumberOfConnectivityOptimizationIterations = int(connectivity_iter)
    # No MinEdgeLength here, because setting it would do nothing. vmtk turns
    # that number into the MinArea the remesher actually reads, and it only
    # does so in the `edgelengtharray` branch (vmtksurfaceremeshing.py:110-111);
    # in `edgelength` mode MinArea keeps its 0.0 default whatever we assign. So
    # this path -- the one the ground-truth remesh takes -- has never had the
    # 0.01 mm floor it looked like it had, which is worth knowing next to a run
    # that logged 46 degenerate min edges with a median of 1e-06 mm. It is not
    # the cause: those edges come in with the surface, the clippers make them
    # and the pre-remesh weld multiplied them, and with that fixed the same
    # cases come out at 3.1e-02 mm. If a floor is ever wanted here it has to be
    # `remesher.MinArea = 0.25 * 3.0**0.5 * REMESH_MIN_EDGE_MM**2`, and it
    # should be measured before it is trusted -- a nonzero MinArea gives the
    # remesher licence to collapse triangles, which is not free.
    remesher.CollapseAngleThreshold = float(collapse_angle)
    remesher.Execute()
    return to_vtk_poly(remesher.Surface)


REMESH_MAX_AREA_DRIFT = 1.15

# Holding the area is not enough to call a remesh sound. A fan tent -- one
# vertex left carrying a spray of triangles instead of a patch of proper ones --
# barely moves the area, so the drift test waves it through: ten of the 111
# meshes the validation run shipped carry one.
#
# The count of triangles at that vertex is the obvious measure and it does not
# work. Measured across those ten, valence says nothing: C0010 carries 61 at an
# edge-length CV of 0.172 and ANSYS_UNIGE_30_614 carries 94 at 0.190, both fine,
# while p129 carries 21 at a CV of 0.437 and p363 22 at 0.540, both not.
#
# What separates them is how far the fan reaches, in units of the mesh's own
# edge. A vertex whose ring sits about two edges away is a crowded vertex and
# nothing more; one whose ring sits ten or twenty edges away is a patch thrown
# across ground that should be carrying a hundred properly sized triangles,
# which is precisely the uniformity this dataset exists to provide. The ten
# split cleanly on that and on nothing else:
#
#     C0010 8.0, ANSYS_UNIGE_30_614 5.8, USFD_0035 4.7, SNF00000607_01_2 3.0
#     p129 20.7, p551 21.4, p391 23.2, p363 32.5, SNF00000538_01 37.3, p399 38.4
#
# 12 sits in the gap with room on both sides. It is not a harsh gate either: the
# weld sweep clears these outright rather than parking them -- p399 goes from
# 38.4 edges to no hub at all, p363 and SNF00000538_01 likewise, and p391 from
# 23.2 to 2.0 -- so reaching this test at all is rare.
REMESH_HUB_VALENCE = 20
REMESH_MAX_HUB_RING_EDGES = 12.0


def worst_hub_ring_edges(surface, hub_valence=REMESH_HUB_VALENCE):
    """How far the widest triangle fan reaches, in mean edge lengths.

    Zero when no vertex carries enough triangles to be a fan at all.
    """
    _p, pts, faces = _triangle_points_faces(to_vtk_poly(surface))
    if faces.size == 0:
        return 0.0
    edges = _triangle_edge_lengths(pts, faces)
    mean_edge = float(edges.mean())
    if not np.isfinite(mean_edge) or mean_edge <= 0.0:
        return 0.0
    valence = np.bincount(faces.ravel(), minlength=len(pts))
    worst = 0.0
    for hub in np.flatnonzero(valence >= int(hub_valence)):
        ring = np.unique(faces[(faces == hub).any(axis=1)])
        ring = ring[ring != hub]
        if ring.size == 0:
            continue
        reach = float(np.linalg.norm(pts[ring] - pts[hub], axis=1).mean()) / mean_edge
        if reach > worst:
            worst = reach
    return worst

# What actually defeats vmtkSurfaceRemeshing on these surfaces is edges far
# shorter than the mesh they sit in, and the tolerance that clears them has to
# be measured in that mesh's own units. WELD_TOLERANCE_MM is a fixed 1e-3 mm,
# which is the right instrument for exact duplicates and far too small for
# this: p097 came out of the uncap with its shortest edge at 0.000777 mm inside
# a mesh averaging 0.1337, welded at 1e-3 mm as before, and the remesh still ran
# away to 1.476x. Welding the same surface at 0.05 of its mean edge -- 0.0067
# mm, under seven times more -- lands it at 0.999x with an edge-length CV of
# 0.119. The GT reconstruction of p379 needs 0.25 to do the same, 0.995x at CV
# 0.133, and is still diverging at 2.157x at 0.05.
#
# So the tolerance is swept rather than chosen, gentlest first: 0.0 is exactly
# what this pipeline did before, and a surface that converges there is remeshed
# untouched. The sweep stops at 0.25 because it does not improve past there and
# does get worse -- 0.5 of the mean edge took p379 to 10.056x.
#
# The weld is measured, not trusted: it moves no point off the surface (max
# deviation 0.0000 mm on p379) and holds the area to within 0.08%, but it does
# create non-manifold edges as it goes, 0 to 79 by 0.25 on that surface. That is
# why every attempt is scored against the *unwelded* input, so a weld that ate
# geometry cannot pass by flattering itself.
REMESH_WELD_FRACTIONS = (0.0, 0.05, 0.10, 0.15, 0.25)

# Last resort only, and it is a real concession: fewer iterations leave
# degenerate tails and jagged rims, so a mesh rescued here is worse than one the
# weld sweep rescued, and the log says which happened.
REMESH_ITER_FALLBACKS = ((4, 6),)


# Every cut this pipeline makes is a plane or a cylinder through a triangle
# mesh, and wherever the cutter passes close to a vertex it keeps that vertex
# and adds another one microns away. vtkCleanPolyData then merges nothing,
# because its default tolerance is zero and the two points are not equal, only
# indistinguishable.
#
# On p551 that is the whole failure. Trimming the extension patches takes the
# shortest edge from 2.20e-03 mm to 1.92e-05 and leaves 19 sub-micron edges on
# one ostium rim; the pipe-section cut at that same ostium turns them into 130
# vertex pairs within 5 microns of each other; close_wall_pinholes patches the
# mess and adds 22 more; and the light Taubin pass, where a vertex whose
# neighbours are all microns away is pulled in with them, finally draws 27
# vertices into a ball a tenth of a millimetre across -- still carrying their
# original long connections, mean edge 0.83 mm against the mesh's 0.196. The
# remesher is handed that and builds a tent on it: 368 triangles on one vertex,
# spokes out to 9.3 mm.
#
# Welding it at the end does not help, and that is the point. By then the ball
# exists, and merging it leaves one vertex holding every long edge that ran
# into it -- the same hub, arrived at by a different route. The slivers have to
# go when they are made, before the next step builds on them, which is why this
# runs at each cut rather than once before the remesh.
#
# A twentieth of the mean edge is far below anything real: on p551 it is
# 0.0098 mm against a 1st-percentile legitimate edge of 0.109, an eleven-fold
# margin, and it catches every one of the slivers.
CLIP_SLIVER_WELD_FRACTION = 0.05


def weld_clip_slivers(surface):
    """Collapse the sliver edges a cut leaves behind."""
    return weld_to_edge_fraction(surface, CLIP_SLIVER_WELD_FRACTION)


def weld_to_edge_fraction(surface, fraction):
    """Collapse edges shorter than ``fraction`` of this surface's mean edge.

    The radius is the one that was calibrated on p551 and is kept exactly; what
    changed is the operation. This was vtkCleanPolyData with that radius as an
    absolute tolerance, which merges every pair of points anywhere on the
    surface that falls inside it -- including two walls that happen to pass
    within a twentieth of an edge of each other, which is not rare in a vessel
    tree and is not something a cut created.

    Measured over 140 cases, running both at one and the same radius: the merge
    gained non-manifold edges on 44 of them and changed the opening count on 6,
    the collapse on none of either, and the collapse held area slightly better
    besides (worst drift 0.999999 against 0.999997). Six openings altered by a
    repair that was supposed to be invisible is the whole argument -- well made
    openings are the point of this pipeline.
    """
    poly = to_vtk_poly(surface)
    if fraction <= 0:
        return poly
    _p, pts, faces = _triangle_points_faces(poly)
    if faces.size == 0:
        return poly
    mean_edge = float(_triangle_edge_lengths(pts, faces).mean())
    if not np.isfinite(mean_edge) or mean_edge <= 0.0:
        return poly
    return collapse_tiny_edges(poly, floor=mean_edge * float(fraction))


def _surface_area(surface):
    return float(pv.wrap(to_vtk_poly(surface)).area)


def remesh_surface_verified(
    open_surface,
    target_edge_length,
    n_iter,
    connectivity_iter,
    label="surface",
):
    """Remesh, and reject a pass that diverged instead of shipping it.

    vmtkSurfaceRemeshing does not always converge, and it reports no error when
    it fails to. On p129 the configured 20 iterations inflated the area from
    1986 to 10825 mm^2, shattered one connected region into seven, and drove the
    edge-length CV to 2.32 -- the opposite of the uniformity the iterations are
    there to produce. Six iterations on the same surface land within 2% of the
    input area. More passes are not monotonically better: past the point where
    the optimiser starts fighting itself, they fold the surface.

    Because the failure is silent, the damage used to surface only at the final
    area gate, whose message blames flow extensions -- measured per step, the
    extensions were added and clipped back to within 0.2% of the original, and
    every bit of the growth was here. So each attempt is now measured against
    the surface handed in, and the first one that holds its area is taken.

    What rescues a diverging case is welding out the edges that are far shorter
    than the mesh around them, at a tolerance measured in that mesh's own units;
    see REMESH_WELD_FRACTIONS. The sweep starts at no weld at all, so a case that
    already converges is remeshed exactly as before, untouched and at full
    iterations. Iterations are only given up when no tolerance worked, because a
    mesh rescued that way is a worse mesh, and the log says which happened.
    """
    before = _surface_area(open_surface)
    n_iter = int(n_iter)
    connectivity_iter = int(connectivity_iter)

    # Weld first and keep the iterations, because that is what produces a mesh
    # worth having: the sweep that rescues p097 and p379 lands them at CV 0.119
    # and 0.133, where dropping to 4 iterations gives a coarse one. Iterations
    # come off only when no tolerance in the sweep worked.
    #
    # Turning the angle collapse off comes second, before any welding, because
    # welding removes geometry to buy convergence and switching off an operation
    # that was tearing the surface does not: see REMESH_COLLAPSE_ANGLE for what
    # it does to the two failures measured. The weld sweep then runs with the
    # collapse off, since that is the better-behaved remesher, and only after
    # that does it go back to vmtk's 0.2.
    def _ladder(iters, conn):
        """Unwelded with the angle collapse on, then off, then the weld fractions.

        The rescue sits second on purpose. Putting it last instead -- running
        every collapse-on weld fraction to exhaustion first -- was tried on the
        56-case set and is worse: p551 keep 2 then takes a collapse-on attempt
        that passes the gates with a far poorer surface, CV 0.469 against 0.179,
        fan reach 8.52 against 0.75, 76 oversized triangles against none, and a
        spurious seventh opening that fails the count. The ladder returns the
        first attempt that clears the gates, so an attempt that merely clears
        them preempts one that clears them comfortably.

        That reordering was made to protect UPF_P0194, on the belief that the
        rescue had cost it an opening. It had not: UPF_P0194 comes out 6 against
        7 with the rescue early, with it late, and in the run before the rescue
        existed. Its 7-opening surface in cleandata predates all of this and the
        cause is elsewhere.
        """
        off, on = REMESH_COLLAPSE_ANGLE_OFF, REMESH_COLLAPSE_ANGLE
        first, rest = REMESH_WELD_FRACTIONS[0], REMESH_WELD_FRACTIONS[1:]
        return (
            [(first, iters, conn, on), (first, iters, conn, off)]
            + [(f, iters, conn, off) for f in rest]
            + [(f, iters, conn, on) for f in rest]
        )

    attempts = _ladder(n_iter, connectivity_iter)
    for iters, conn in REMESH_ITER_FALLBACKS:
        if iters < n_iter:
            attempts += _ladder(iters, conn)

    best = None
    usable = None
    for fraction, iters, conn, collapse in attempts:
        welded = weld_to_edge_fraction(open_surface, fraction)
        out = remesh_surface_isotropically(
            welded,
            target_edge_length=target_edge_length,
            n_iter=iters,
            connectivity_iter=conn,
            collapse_angle=collapse,
        )
        # Scored against the surface handed in, never against the welded one, so
        # a tolerance that ate geometry cannot hide the loss.
        after = _surface_area(out)
        drift = after / before if before > 0 else float("inf")
        hub = worst_hub_ring_edges(out)
        # Rank attempts by area first and fan reach second, so "closest" in the
        # failure message means the one that came nearest to being usable.
        score = (abs(drift - 1.0), hub)
        if best is None or score < best[0]:
            best = (score, out, drift, hub, fraction, iters, conn, collapse)
        held = drift <= REMESH_MAX_AREA_DRIFT and hub <= REMESH_MAX_HUB_RING_EDGES
        if held:
            # Holding the area and the fan is not the same as handing on a
            # surface the rest of the pipeline can work with. p398 cleared both
            # on the first attempt and still came out with 34 non-manifold edges
            # and an edge of 1e-06 mm, from an input that had none of either --
            # and repairing that is what tore one of its ostia from circularity
            # 0.980 to 0.181. So a torn attempt no longer ends the ladder: it is
            # kept as the answer if nothing better turns up, and the remaining
            # rungs are given the chance to produce a clean one.
            topo = inspect_surface_topology(out)
            n_nm = int(topo["n_nonmanifold"])
            min_edge = float(topo["min_edge"])
            intact = n_nm == 0 and min_edge >= MIN_EDGE_LENGTH_MM
            if usable is None:
                usable = (out, drift, hub, fraction, iters, conn, collapse, n_nm, min_edge)
            if not intact:
                torn = []
                if n_nm:
                    torn.append(f"{n_nm} non-manifold edge(s)")
                if min_edge < MIN_EDGE_LENGTH_MM:
                    torn.append(f"an edge of {min_edge:.2e} mm")
                print(
                    f"  NOTE: remesh at weld {fraction:.2f}, {iters} iterations, "
                    f"collapse angle {collapse:.2f} held the {label} "
                    f"({drift:.3f}x, fan {hub:.1f}) but left "
                    + " and ".join(torn)
                    + "; trying the next rung for a clean one"
                )
                continue
        if held:
            if fraction > 0 or iters != n_iter or collapse != REMESH_COLLAPSE_ANGLE:
                how = []
                if collapse != REMESH_COLLAPSE_ANGLE:
                    how.append("switching the angle collapse off")
                if fraction > 0:
                    how.append(f"welding at {fraction:.2f} of the mean edge")
                if iters != n_iter:
                    how.append(f"backing off to {iters} iterations")
                print(
                    f"  NOTE: remesh diverged on the {label} as configured; "
                    + " and ".join(how)
                    + f" held it ({drift:.3f}x, widest fan {hub:.1f} edges)"
                )
            return out
        why = []
        if drift > REMESH_MAX_AREA_DRIFT:
            why.append(f"area {drift:.2f}x ({before:.1f} -> {after:.1f} mm^2)")
        if hub > REMESH_MAX_HUB_RING_EDGES:
            why.append(f"a triangle fan reaching {hub:.1f} mean edges")
        print(
            f"  WARNING: remesh at weld {fraction:.2f}, {iters} iterations, "
            f"collapse angle {collapse:.2f} left the {label} with "
            + " and ".join(why)
        )
    if usable is not None:
        out, drift, hub, fraction, iters, conn, collapse, n_nm, min_edge = usable
        torn = []
        if n_nm:
            torn.append(f"{n_nm} non-manifold edge(s)")
        if min_edge < MIN_EDGE_LENGTH_MM:
            torn.append(f"an edge of {min_edge:.2e} mm")
        print(
            f"  WARNING: no rung of the ladder remeshed the {label} without "
            "tearing it; keeping the first that held the area and the fan "
            f"(weld {fraction:.2f}, {iters} iterations, collapse angle "
            f"{collapse:.2f}, {drift:.3f}x, fan {hub:.1f}) with "
            + " and ".join(torn)
        )
        return out
    _score, out, drift, hub, fraction, iters, conn, collapse = best
    raise TemplateQualityError(
        f"isotropic remesh did not converge on the {label}: the closest attempt "
        f"(weld {fraction:.2f} of the mean edge, {iters} iterations, collapse "
        f"angle {collapse:.2f}) still changed the area {drift:.2f}x and left a "
        f"triangle fan reaching {hub:.1f} mean edges. This is a remesher "
        f"failure, not a clipping one."
    )


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


def _mesh_components(faces, n_points):
    """Component label per face, from a union-find over shared triangle edges."""
    parent = np.arange(int(n_points), dtype=np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in ((a, b), (b, c), (c, a)):
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv
    roots = np.array([find(int(t[0])) for t in faces], dtype=np.int64)
    return roots


# A crack the remesher leaves is measured against the mesh it left it in. On
# p398 vmtkSurfaceRemeshing retriangulated a branch without sharing the seam
# vertices and left a split whose two sides stand 0.0006-0.05 mm apart in a
# surface of 0.15 mm edges: a third of one triangle at its widest. Half an edge
# separates that from any gap that is really there, and the ceiling keeps the
# same rule from reaching across a coarse input.
REJOIN_SEAM_EDGE_FRACTION = 0.5
REJOIN_SEAM_MAX_MM = 0.1


def _seam_weld_tolerance(pts, faces):
    """How wide a crack may be and still be one the remesher opened."""
    if faces.size == 0:
        return WELD_TOLERANCE_MM
    mean_edge = float(_triangle_edge_lengths(pts, faces).mean())
    if not np.isfinite(mean_edge) or mean_edge <= 0.0:
        return WELD_TOLERANCE_MM
    return float(min(REJOIN_SEAM_MAX_MM,
                     max(WELD_TOLERANCE_MM, REJOIN_SEAM_EDGE_FRACTION * mean_edge)))


def rejoin_ostium_islands(surface, profiles, tol_mm=None):
    """Weld a severed shell that carries a real ostium back onto the body.

    The manifold repairs remove triangles, and where they remove the last
    triangles joining a neck they cut a whole shell loose. On p469 that shell
    was 55.35 mm^2 against the body's 1149 -- 5119 points -- and it carried a
    real, round 0.328 mm ostium, so `keep_largest_region` threw the opening
    away and the case finished 5 against 6 with every remaining rim correct.

    The shell is not floating: its nearest point to the body is 113 nanometres
    away, because cutting duplicates the seam vertices rather than moving them.
    So the repair is to merge those duplicates and nothing else. A global
    `vtkCleanPolyData` merge would do it and must not be used -- it welds every
    pair anywhere inside the radius, which is what once took SNF00000228 from 8
    non-manifold edges to 742 -- so only island points that coincide with a body
    point are remapped.
    """
    poly, pts, faces = _triangle_points_faces(surface)
    if faces.size == 0 or not profiles:
        return poly, 0
    labels = _mesh_components(faces, len(pts))
    uniq = np.unique(labels)
    if uniq.size < 2:
        return poly, 0
    if tol_mm is None:
        tol_mm = _seam_weld_tolerance(pts, faces)
    sizes = {int(u): int(np.count_nonzero(labels == u)) for u in uniq}
    body = max(sizes, key=lambda u: sizes[u])
    body_pt_ids = np.unique(faces[labels == body].reshape(-1))
    from scipy.spatial import cKDTree

    tree = cKDTree(pts[body_pt_ids])
    remap = np.arange(len(pts), dtype=np.int64)
    rejoined = 0
    for u in uniq:
        if int(u) == body:
            continue
        mask = labels == u
        shell = _polydata_from_triangles(pts, faces[mask])
        carries = any(
            _loop_at_a_profile(center, profiles, radius=radius, n_points=n)
            for _ids, center, radius, n in _loop_geometry(shell)[3]
        )
        if not carries:
            continue
        shell_pt_ids = np.unique(faces[mask].reshape(-1))
        d, idx = tree.query(pts[shell_pt_ids], k=1)
        seam = np.asarray(d, dtype=np.float64) <= float(tol_mm)
        if int(np.count_nonzero(seam)) < 3:
            print(
                f"  An ostium-carrying shell of {int(np.count_nonzero(mask))} "
                f"triangles sits {float(np.min(d)):.6f} mm off the body, too far "
                f"to weld back (seam tolerance {float(tol_mm):.4f} mm)"
            )
            continue
        print(
            f"  An ostium-carrying shell of {int(np.count_nonzero(mask))} triangles "
            f"meets the body along {int(np.count_nonzero(seam))} point(s) within "
            f"{float(tol_mm):.4f} mm; welding it back"
        )
        remap[shell_pt_ids[seam]] = body_pt_ids[np.asarray(idx)[seam]]
        rejoined += 1
    if not rejoined:
        return poly, 0
    kept = remap[faces]
    alive = (kept[:, 0] != kept[:, 1]) & (kept[:, 1] != kept[:, 2]) & (kept[:, 2] != kept[:, 0])
    merged = _polydata_from_triangles(pts, kept[alive])
    n_after = count_connected_regions(merged)
    if n_after >= count_connected_regions(poly):
        # The weld did not actually rejoin anything; leave the surface alone.
        return poly, 0
    print(
        f"  Welded {rejoined} shell(s) carrying an ostium back onto the body "
        f"({count_connected_regions(poly)} shells -> {n_after})"
    )
    return merged, rejoined


def drop_tiny_islands(surface, profiles=None):
    """Keep the vessel body -- but never throw away a real opening with a shell.

    ``profiles``, when given, let an ostium-carrying shell be welded back on
    first; without them this is the old keep-the-largest behaviour.
    """
    poly = to_vtk_poly(surface)
    if profiles and count_connected_regions(poly) > 1:
        poly, _n = rejoin_ostium_islands(poly, profiles)
    main = keep_largest_region(poly)
    return main, count_connected_regions(main)


def _finalize_defects(surface, profiles, n_regions):
    """How far a candidate is from being acceptable, worst defect first.

    Ordered so that a straight tuple comparison picks the better surface:
    non-manifold edges first (the remesher amplifies them), then holes that are
    not anatomy, then rims a tear has branched, then extra shells, then
    degenerate edges.

    A branching rim ranks above an extra shell because it is an opening that has
    stopped being one. Every edge on it still has a single face, so the
    non-manifold count is zero and the loop count is right; it is only when the
    rim is measured as a shape that the damage shows. On p398 an ostium left the
    remesher at circularity 0.997 and the repairs below handed it on at 0.181,
    and this tuple -- which reached (0, 0, 0, 0) on that surface -- called it
    finished.
    """
    topo = inspect_surface_topology(surface)
    extra = [
        lp
        for lp in boundary_loop_radii(surface)
        if _is_wall_pinhole(lp, MIN_OPENING_RADIUS_MM, profiles)
    ]
    return (
        int(topo["n_nonmanifold"]),
        len(extra),
        branching_rims(surface),
        max(int(n_regions) - 1, 0),
        0 if topo["min_edge"] >= MIN_EDGE_LENGTH_MM else 1,
    )


def finalize_surface(surface, profiles=None, max_passes=6):
    """Clean, weld, force manifoldness and close every non-ostium hole.

    The remesher is free to leave micron-scale rim edges (PreserveBoundaryEdges
    keeps them), non-manifold sheets a flap heuristic will not touch, and
    pinholes wider than vtkFillHolesFilter's hole size. Each of those used to
    reach ``assert_template_quality`` unrepaired and fail the case, so they are
    repaired here instead. ``profiles`` are the anatomical openings; when given,
    boundary loops that are not one of them are closed and the ostia themselves
    are protected.

    The repairs interact -- welding can fuse two rim vertices into a non-manifold
    edge, and cutting a non-manifold edge opens a new pinhole -- so the sequence
    runs until the surface stops changing rather than exactly once. Two of those
    interactions can cycle instead of settling, so the loop also stops when a
    state repeats and returns the best surface it saw rather than the last one.
    """
    # Fold the sub-micron edges before anything stitches around them. The weld
    # already ran, but at the end of every pass, which is too late: p469 arrived
    # here with 19 edges under the floor and every one of them satisfied the
    # link condition, yet two survived to the quality gate at 3.6e-07 mm. They
    # survived because the manifold repair and the pinhole closer run first and
    # hang new vertices off those edges -- the two that were left shared 93671
    # and 93672, created during the repair -- and an edge whose endpoints share
    # a neighbour that is not its apex is a pinch, which collapse_tiny_edges
    # must refuse or it would fuse two sheets. Welding on the way in means the
    # repairs never meet the degenerate edge at all.
    cleaned = collapse_tiny_edges(clean_triangulate(surface))
    n_regions = count_connected_regions(cleaned)
    best = cleaned
    best_regions = n_regions
    best_defects = _finalize_defects(cleaned, profiles, n_regions)
    seen = set()
    for _ in range(int(max_passes)):
        if not any(best_defects):
            break
        signature = (
            cleaned.GetNumberOfPoints(),
            cleaned.GetNumberOfCells(),
            _finalize_defects(cleaned, profiles, n_regions),
        )
        if signature in seen:
            print("  Repair loop reached a fixed point; keeping the best surface so far")
            break
        seen.add(signature)

        cleaned, _n_nm = repair_nonmanifold_triangles(cleaned)
        cleaned, _n_forced = force_manifold_triangles(cleaned)
        # No vtkFillHolesFilter here: it is neither ostium-aware nor
        # manifold-safe, and it kept stitching back exactly the triangles the
        # manifold repair had just removed, which is how this loop used to spin.
        cleaned, _n_pin = close_wall_pinholes(
            cleaned, label="remeshed surface", profiles=profiles
        )
        cleaned, _n_ears = unbranch_rims(
            cleaned, label="remeshed surface", profiles=profiles
        )
        if profiles:
            cleaned, _n_left = remove_spurious_openings(cleaned, profiles)
        cleaned, _min_edge = weld_degenerate_vertices(cleaned)
        cleaned = strip_all_arrays(cleaned)
        cleaned, n_regions = drop_tiny_islands(cleaned, profiles=profiles)

        defects = _finalize_defects(cleaned, profiles, n_regions)
        if defects < best_defects:
            best, best_regions, best_defects = cleaned, n_regions, defects
        if not any(defects):
            break
    best = strip_all_arrays(best)
    best = recompute_point_normals(best, auto_orient=False)
    return best, best_regions


def assert_template_scale(surface, reference_mesh, context="template", max_area_ratio=2.5):
    tpl_area = float(pv.wrap(to_vtk_poly(surface)).area)
    ref_area = float(pv.wrap(to_vtk_poly(reference_mesh)).area)
    if ref_area <= 1e-6:
        return tpl_area, ref_area
    ratio = tpl_area / ref_area
    print(f"  Template area={tpl_area:.1f} mm^2 vs vessel {ref_area:.1f} mm^2 (ratio {ratio:.2f})")
    if ratio > max_area_ratio:
        # This gate sees the finished template, which is several steps past the
        # polyball, so it cannot name the step that grew. It used to say "MISR
        # blob or modeller overflow" anyway, and on C0088b that was wrong twice
        # over: the parent tube came out at 1454.6 mm^2 against a 1465 mm^2
        # vessel, and the adaptive remesh then inflated it 8.9x. Where the
        # remesh is the culprit the ladder above this call reports it by name,
        # so the honest thing here is to say what was measured and stop.
        raise TemplateQualityError(
            f"{context} template area is {ratio:.2f}x the vessel "
            f"({tpl_area:.1f} vs {ref_area:.1f} mm^2). Check the parent-tube "
            f"area printed above: if it is close to the vessel the overflow is "
            f"downstream of the polyball, otherwise the modeller overflowed."
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


def _try_reuse_centerline(reuse_centerline, reference_bounds, profiles=None):
    """Load a precomputed original_centerline. None means extract Voronoi as today.

    ``profiles`` are the anatomical openings the tube has to end at. A stored
    trace that does not reach one of them is worse than no reuse at all: the
    polyball tube grows no stub there, the pipe-section uncap finds no end to
    cut, and the template ships without an opening the GT has. The arrival test
    is the same one that picks between the two live traces, and it separates the
    two populations cleanly -- measured over 391 openings, arrivals reach p90
    0.79 radii and misses start at 9.34, with nothing in between.
    """
    if reuse_centerline is None:
        return None
    if isinstance(reuse_centerline, str):
        if not os.path.isfile(reuse_centerline):
            return None
        cl = to_vtk_poly(pv.read(reuse_centerline))
        source = reuse_centerline
    else:
        cl = to_vtk_poly(reuse_centerline)
        source = "in-memory centerline"
    if cl.GetPointData().GetArray("MaximumInscribedSphereRadius") is None:
        print(f"  WARNING: reused centerline has no MISR ({source}); extracting Voronoi")
        return None
    if not centerline_looks_valid(cl, reference_bounds):
        print(f"  WARNING: reused centerline failed the lumen check ({source}); extracting Voronoi")
        return None
    if profiles:
        arrived, gaps = centerline_arrivals(cl, profiles)
        if not bool(arrived.all()):
            missed = ", ".join(
                f"r={float(profiles[i]['radius']):.3f} mm, {float(gaps[i]):.2f} mm away"
                for i in np.flatnonzero(~arrived)
            )
            print(
                f"  WARNING: reused centerline reaches {int(arrived.sum())}/"
                f"{len(profiles)} openings ({missed}); extracting Voronoi"
            )
            return None
    print(f"  Reusing original centerline ({cl.GetNumberOfPoints()} points, skip dual Voronoi)")
    return cl


def build_parent_tube(
    vessel_mesh,
    extension_length=DEFAULT_EXTENSION_LENGTH,
    sample_spacing=DEFAULT_SAMPLE_SPACING,
    grid_spacing=DEFAULT_GRID_SPACING,
    max_grid_size=DEFAULT_MAX_GRID_SIZE,
    dataset_id=None,
    reuse_centerline=None,
    skip_mc_decimate=False,
    fast_uncap=True,
    cut_frames=None,
):
    """Shared path: smooth -> extend -> cap -> centerline -> polyball tube -> uncap at anatomy.

    Variable remesh may pass ``reuse_centerline`` and ``skip_mc_decimate``.
    Uncap accounting defaults to the fast local-boundary test; the cylinder
    clip is unchanged. Pass ``fast_uncap=False`` for the old full-mesh extract.

    ``cut_frames``: optional non-empty GT ostium frames (``origin``, unit
    ``normal``, ``radius``). Forwarded to ``clip_flow_extensions_and_uncap`` so
    the tube is cut in the GT planes instead of ``opening_clip_frames``.
    ``None`` / empty keeps the measured-profile uncap. Centerline seeding still
    uses ``measure_open_profiles`` on the vessel; that is not a second cut pass.
    """
    print("Step 1: Applying Taubin surface smoothing...")
    work_vessel = sanitize_vessel_for_vmtk(vessel_mesh)
    smoothed_vessel = apply_taubin_smoothing(work_vessel)

    print("Step 1b: Detecting anatomical inlet/outlet boundaries...")
    anatomical_profiles = measure_open_profiles(smoothed_vessel)
    # One opening per hole. VMTK reports a rim pinched into a figure eight as
    # two profiles, and each one then demands its own polyball stub and its own
    # pipe-section cut at a place where there is only one hole.
    anatomical_profiles, _n_phantom = reconcile_profiles_with_loops(
        smoothed_vessel, anatomical_profiles, label="smoothing"
    )
    log_profiles(anatomical_profiles, label="Anatomical")
    _inlet, _outlets = seed_points_from_profiles(anatomical_profiles)

    branched_centerline = _try_reuse_centerline(
        reuse_centerline, smoothed_vessel.GetBounds(), profiles=anatomical_profiles
    )
    if branched_centerline is None:
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
        t_cl = time.perf_counter()
        centerline = extract_centerlines_for_tube(
            extended_vessel, smoothed_vessel, anatomical_profiles, extended_profiles
        )
        print(f"  [t] Voronoi extract {time.perf_counter() - t_cl:.2f}s")

        print(f"Step 4: Spline resampling ({sample_spacing} mm) and trajectory smoothing...")
        resampled = resample_centerline(centerline, sample_spacing=sample_spacing)
        smooth_centerline = smooth_centerline_preserve_misr(resampled)

        print("Step 5: Extracting branches...")
        branched_centerline = extract_branches(smooth_centerline)
    else:
        print("Step 2-5: skipped flow extensions and Voronoi (original_centerline reused)")

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
        skip_decimate=skip_mc_decimate,
    )

    print("Step 7: Uncapping open boundaries with pipe-section cuts...")
    t_un = time.perf_counter()
    uncap_cut_frames = cut_frames if cut_frames else None
    if uncap_cut_frames:
        print(
            f"  Uncap: forwarding {len(uncap_cut_frames)} GT ostium cut frame(s) "
            "(skip opening_clip_frames)"
        )
    open_base_surface, n_clipped = clip_flow_extensions_and_uncap(
        base_surface,
        anatomical_profiles,
        extension_length=extension_length,
        centerline=branched_centerline,
        fast_uncap=fast_uncap,
        cut_frames=uncap_cut_frames,
    )
    print(f"  [t] uncap {time.perf_counter() - t_un:.2f}s")
    print(f"  -> Open base surface points: {open_base_surface.GetNumberOfPoints()}")
    n_in = len(uncap_cut_frames) if uncap_cut_frames else len(anatomical_profiles)
    if n_clipped != n_in:
        check_against = (
            _profiles_from_ostium_frames(uncap_cut_frames)
            if uncap_cut_frames
            else anatomical_profiles
        )
        still_shut = unopened_profiles(open_base_surface, check_against)
        detail = (
            f" Still shut: {describe_unopened(still_shut)}."
            if still_shut
            else " Every ostium has a rim near it, so the shortfall is in the count, not the mesh."
        )
        # An ostium in its own lumen is not a bug in the uncap and no amount of
        # cutting will open it: the trace cannot get there, so the tube has no
        # wall there to cut. Measured over the fourteen template-less cases,
        # seven died this way -- and the message they gave was "opened 4/7",
        # which sends the reader to the cutter. Name the real fault instead.
        detail += _split_lumen_note(smoothed_vessel, check_against, still_shut)
        raise TemplateQualityError(
            f"Parent-tube uncap opened {n_clipped}/{n_in} anatomical ends; "
            f"a template missing an opening the GT has is unusable.{detail}",
            dataset_id=dataset_id,
        )
    return {
        "smoothed_vessel": smoothed_vessel,
        "anatomical_profiles": anatomical_profiles,
        "branched_centerline": branched_centerline,
        "open_base_surface": open_base_surface,
        "n_clipped": n_clipped,
    }


# ---------------------------------------------------------------------------
# Steps 8a-8c, as one retryable attempt
# ---------------------------------------------------------------------------

def _rim_shape(poly):
    """(number of rim loops, worst circularity) -- what a bad collapse ruins."""
    ring = extract_boundary_loops(poly)
    worst = 1.0
    n = ring.GetNumberOfCells()
    for k in range(n):
        cell = ring.GetCell(k)
        co = np.array(
            [ring.GetPoint(cell.GetPointId(j)) for j in range(cell.GetNumberOfPoints())],
            dtype=float,
        )
        if len(co) < 3:
            return n, 0.0
        perimeter = float(np.sum(np.linalg.norm(co - np.roll(co, -1, axis=0), axis=1)))
        if perimeter <= 0:
            return n, 0.0
        rel = co - co.mean(axis=0)
        area = 0.5 * float(np.linalg.norm(np.cross(rel, np.roll(rel, -1, axis=0)).sum(axis=0)))
        worst = min(worst, 4.0 * np.pi * area / (perimeter * perimeter))
    return n, worst


def enforce_min_edge(surface, floor=REMESH_MIN_EDGE_MM, label="template", report=True):
    """Fold the edges the remesher was told not to make, and keep the fold only
    if it cost nothing.

    ``remesh_surface_adaptively`` passes MinEdgeLength to vmtk, which turns it
    into the MinArea the remesher reads, and the target-edge array is clamped
    at the same floor -- so nothing in the configuration asks for an edge below
    it. The remesher makes a few anyway. On p414 the shipped template carried
    24 edges under 0.01 mm out of 104538, the shortest 0.002066 mm, in 22
    triangles whose median area is a hundredth of the mesh's; none of them was
    near a rim, so they are not a clipping artefact but slivers left behind in
    a dense stretch zone. Across the 55 GT surfaces of the validation run, 26
    carry one.

    Two things make this safe to run on a surface that already passed. The fold
    is pure -- ``allow_cuts=False`` bars the two fallbacks that cut geometry
    when every remaining short edge is pinched, which is what tore
    UPF_P0171.00_ID1 from 6 rims to 7 and p469 from 6 to 11 in the first sweep
    at this floor. And the result is checked rather than trusted: a fold that
    changes the rim count, worsens the worst rim's circularity, adds a
    non-manifold edge or moves the area by more than a part in ten thousand is
    dropped and the input returned untouched.

    What survives the check is free. Over the 24 GT surfaces where the pure
    fold applies it merges a median of 24 points, moves the area by at most
    5.4e-05 relative, leaves every rim count and circularity exactly as it
    found them, and takes the mean sliver fraction from 0.891% to 0.857%. On
    p414 it merges 11 points and lifts the shortest edge from 0.002066 mm to
    0.0105 mm with the six openings, the fan reach and the non-manifold count
    unchanged. A mesh with no sub-floor edge is skipped outright and keeps its
    interpolated point arrays.
    """
    poly = to_vtk_poly(surface)
    # Measured on the merged mesh, because the remesher hands back a surface
    # whose points are split per triangle corner -- p414 comes out of it with
    # 70447 points that clean to 17534 -- and counting a collapse against that
    # number would read as a catastrophe rather than the eleven points it is.
    merged = to_vtk_poly(clean_triangulate(poly))
    _p, pts, faces = _triangle_points_faces(merged)
    if faces.size == 0:
        return poly
    edges = _triangle_edge_lengths(pts, faces)
    n_short = int(np.count_nonzero(edges < float(floor)))
    if n_short == 0:
        return poly
    shortest = float(edges.min())
    out = to_vtk_poly(
        collapse_tiny_edges(merged, floor=float(floor), allow_cuts=False)
    )
    if out.GetNumberOfPoints() == merged.GetNumberOfPoints():
        if report:
            print(f"  NOTE: {n_short} edge(s) under the {floor} mm floor on the {label} "
                  f"(shortest {shortest:.6f} mm) are all pinched; left alone")
        return poly

    # Cheapest check first and stop at the first complaint: this runs once per
    # rung of the variable path's remesh ladder, where the time matters.
    def _harm():
        before_area = _surface_area(merged)
        after_area = _surface_area(out)
        drift = abs(after_area / before_area - 1.0) if before_area > 0 else float("inf")
        if drift > 1e-4:
            return f"area x{after_area / before_area:.6f}"
        before_rims, before_circ = _rim_shape(merged)
        after_rims, after_circ = _rim_shape(out)
        if after_rims != before_rims:
            return f"rims {before_rims} -> {after_rims}"
        if after_circ < before_circ - 1e-6:
            return f"worst rim circularity {before_circ:.3f} -> {after_circ:.3f}"
        fan_before = worst_hub_ring_edges(merged)
        fan_after = worst_hub_ring_edges(out)
        if fan_after > REMESH_MAX_HUB_RING_EDGES >= fan_before:
            # A fold moves every triangle on the collapsed vertex onto its
            # partner, so it can in principle raise the fan reach. Measured it
            # never does by much -- the worst of the 24 GT surfaces moved 1.14
            # to 1.17 against a gate of 12 -- but a fold that pushed a mesh over
            # the gate would be trading a sliver for a tent, the worse of the
            # two.
            return f"fan reach {fan_before:.1f} -> {fan_after:.1f}"
        n_nm_before = int(inspect_surface_topology(merged)["n_nonmanifold"])
        n_nm = int(inspect_surface_topology(out)["n_nonmanifold"])
        if n_nm > n_nm_before:
            return f"non-manifold edges {n_nm_before} -> {n_nm}"
        return None

    harm = _harm()
    if harm:
        if report:
            print(f"  NOTE: folding the {n_short} sub-floor edge(s) on the {label} would "
                  f"cost {harm}; kept the surface as it was")
        return poly

    left = int(np.count_nonzero(
        _triangle_edge_lengths(*_triangle_points_faces(out)[1:]) < float(floor)))
    if report:
        tail = "" if left == 0 else f"; {left} still pinched under it"
        print(f"  Folded {n_short - left} edge(s) under the {floor} mm floor on the "
              f"{label} (shortest was {shortest:.6f} mm): "
              f"{merged.GetNumberOfPoints()} -> {out.GetNumberOfPoints()} points{tail}")
    return out


def _adaptive_remesh_ladder():
    """Weld fraction and collapse angle to try, in order, gentlest first.

    Same shape as the ground-truth path's ladder in vessel_pipeline, and for
    the same reasons: the first rung is exactly what this pipeline did before,
    so a template that already converges is remeshed byte for byte as it was
    and pays nothing for this machinery. Switching the angle collapse off comes
    next because it removes no geometry, and only then does the weld sweep run.

    The difference here is what a rung costs. On the isotropic path a rung is
    one remesh; here the weld merges points and VTK drops every point array
    when it does, so a welded rung has to raycast the ground truth again before
    it can remesh. That is the price of a rescue on a case that would otherwise
    ship nothing, and the cache in the caller keeps rungs that share a weld
    fraction from paying it twice.
    """
    off, on = REMESH_COLLAPSE_ANGLE_OFF, REMESH_COLLAPSE_ANGLE
    first, rest = REMESH_WELD_FRACTIONS[0], REMESH_WELD_FRACTIONS[1:]
    return ([(first, on), (first, off)]
            + [(f, off) for f in rest]
            + [(f, on) for f in rest])


def build_template_supervision(base_surface, vessel_mesh, centerline,
                               target_edge_length, min_edge, weld_fraction,
                               prepare=None, announce=True):
    """Weld the tube, then measure R_template, StretchDistance and the edge array on it.

    The weld has to happen before the arrays and not after: weld_to_edge_fraction
    merges points, and at 0.10 and above the surface comes back with no point
    arrays at all. Everything the remesh is steered by therefore belongs to the
    welded surface, which is also the honest thing -- the raycast should measure
    the surface that is actually going to be remeshed.

    ``prepare`` is for a caller that wants more than the target-edge array on
    the surface before it is remeshed: it is handed the surface and the three
    measurements and returns the surface to remesh. variable_remeshing uses it
    to attach the supervision arrays, so VMTK interpolates them onto the new
    vertices instead of the caller filling them in afterwards by nearest
    neighbour.
    """
    surface = (to_vtk_poly(base_surface) if weld_fraction <= 0
               else to_vtk_poly(weld_to_edge_fraction(base_surface, weld_fraction)))
    if weld_fraction > 0:
        print(f"  Welded the tube at {weld_fraction:.2f} of the mean edge: "
              f"{to_vtk_poly(base_surface).GetNumberOfPoints()} -> "
              f"{surface.GetNumberOfPoints()} points")
    t_ray = time.perf_counter()
    r_template = compute_template_local_radii(surface, centerline)
    # The centerline has to go in. Without it the ray direction falls back to
    # -unit_n, and vtkPolyDataNormals with AutoOrientNormalsOff cannot promise
    # which way that points on an open surface; with it, every direction is
    # oriented by the sign of n . (x - nearest centerline vertex), which is
    # outward by construction. It also gates the compiled loop, which refuses
    # to run unless both the template and the GT normals were radially
    # oriented. vessel_pipeline's own variable path has always passed it -- this
    # call was the one that did not.
    stretch_distances = compute_raycast_stretch_distances(
        surface, vessel_mesh, r_template=r_template, centerline=centerline,
    )
    print(f"  [t] radii+raycast {time.perf_counter() - t_ray:.2f}s")
    if announce:
        print(
            f"Step 8b: Building stretch metric k = 1 + d / R_template "
            f"(Base={target_edge_length} mm, Min={min_edge:.2f} mm)..."
        )
    surface_with_array, edge_lengths, stretch_factors = build_target_edge_array(
        surface, stretch_distances, r_template,
        base_edge=target_edge_length, min_edge=min_edge,
    )
    t_attach = time.perf_counter()
    if prepare is not None:
        surface_with_array = prepare(
            surface_with_array, r_template, stretch_distances, edge_lengths
        )
    t_attach_s = time.perf_counter() - t_attach
    pre_pts = np.ascontiguousarray(
        vtk_to_numpy(to_vtk_poly(surface_with_array).GetPoints().GetData()),
        dtype=np.float64,
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
    return {
        "surface": surface_with_array,
        "pre_pts": pre_pts,
        "r_template": np.ascontiguousarray(r_template, dtype=np.float64).reshape(-1),
        "stretch_distances": np.ascontiguousarray(
            stretch_distances, dtype=np.float64).reshape(-1),
        "edge_lengths": np.ascontiguousarray(edge_lengths, dtype=np.float64).reshape(-1),
        "t_attach_s": t_attach_s,
    }


def supervise_and_remesh_verified(base_surface, vessel_mesh, centerline,
                                  target_edge_length, min_edge, prepare=None,
                                  dataset_id="template"):
    """Adaptively remesh the parent tube, and reject a pass that diverged.

    The array-driven remesher fails the same way the isotropic one does and was
    never guarded for it. On C0088b the parent tube is sound -- 1454.6 mm^2
    against a 1465 mm^2 vessel, one region, five clean loops, no non-manifold
    edges -- and the adaptive remesh returned 12980.5 mm^2, 8.92x, with
    triangles up to 111.5 mm^2 and a fan of 46 mm spokes. Every giant triangle
    hung off one vertex on the r=1.7 mm opening, a vertex carrying two
    triangles and a 0.025 mm sliver edge. Nothing downstream could tell what
    had happened: the area gate blamed the polyball, which was innocent.

    Welding that sliver out before the arrays are built fixes it completely --
    0.99x, edge-length CV 0.305, no fan, all five openings -- so the cure is
    the one the ground-truth path already uses, and it is applied the same way:
    as a rescue that runs only after the configured settings have failed.
    """
    base_area = _surface_area(base_surface)
    supervision = {}
    best = None
    for rung, (fraction, collapse) in enumerate(_adaptive_remesh_ladder()):
        if fraction not in supervision:
            if rung:
                print(f"  Rebuilding the supervision arrays for a weld of "
                      f"{fraction:.2f} of the mean edge...")
            supervision[fraction] = build_template_supervision(
                base_surface, vessel_mesh, centerline, target_edge_length,
                min_edge, fraction, prepare=prepare, announce=(rung == 0),
            )
        sup = supervision[fraction]
        if rung == 0:
            print("Step 8c: Adaptively remeshing surface (ElementSizeMode='edgelengtharray')...")
        t_rm = time.perf_counter()
        remeshed = remesh_surface_adaptively(
            sup["surface"], edge_array_name="TargetEdgeLength", collapse_angle=collapse,
        )
        remeshed = enforce_min_edge(remeshed, label="remeshed template")
        t_rm_s = time.perf_counter() - t_rm
        # Scored against the surface handed in, never against the welded one,
        # so a weld that ate geometry cannot pass by flattering itself.
        after = _surface_area(remeshed)
        drift = after / base_area if base_area > 0 else float("inf")
        fan = worst_hub_ring_edges(remeshed)
        score = (abs(drift - 1.0), fan)
        if best is None or score < best[0]:
            best = (score, remeshed, sup, t_rm_s, drift, fan, fraction, collapse)
        if drift <= REMESH_MAX_AREA_DRIFT and fan <= REMESH_MAX_HUB_RING_EDGES:
            if rung:
                how = []
                if collapse != REMESH_COLLAPSE_ANGLE:
                    how.append("switching the angle collapse off")
                if fraction > 0:
                    how.append(f"welding at {fraction:.2f} of the mean edge")
                print("  NOTE: the adaptive remesh diverged as configured; "
                      + " and ".join(how)
                      + f" held it ({drift:.3f}x, widest fan {fan:.1f} edges)")
            return remeshed, sup, t_rm_s
        why = []
        if drift > REMESH_MAX_AREA_DRIFT:
            why.append(f"area {drift:.2f}x ({base_area:.1f} -> {after:.1f} mm^2)")
        if fan > REMESH_MAX_HUB_RING_EDGES:
            why.append(f"a triangle fan reaching {fan:.1f} mean edges")
        print(f"  WARNING: adaptive remesh at weld {fraction:.2f}, collapse angle "
              f"{collapse:.2f} left the template with " + " and ".join(why))
    _score, _remeshed, _sup, _t, drift, fan, fraction, collapse = best
    raise TemplateQualityError(
        f"adaptive remesh did not converge on the parent tube: the closest "
        f"attempt (weld {fraction:.2f} of the mean edge, collapse angle "
        f"{collapse:.2f}) still changed the area {drift:.2f}x and left a "
        f"triangle fan reaching {fan:.1f} mean edges. The tube itself is not "
        f"what failed here; the remesher is.",
        dataset_id=dataset_id,
    )


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
    speedups=False,
):
    print(f"\n=========================================\nProcessing Adaptive Variable Remeshing Case: {dataset_id}")
    t_all = time.perf_counter()
    vessel_mesh = pv.read(v_file)
    reuse = None
    if speedups:
        cl_path = os.path.join(CLEANDATA_ORIGINAL_CENTERLINE, f"{dataset_id}.vtp")
        if os.path.isfile(cl_path):
            reuse = cl_path
        else:
            print("  original_centerline missing; will extract Voronoi")
    built = build_parent_tube(
        vessel_mesh,
        extension_length=extension_length,
        sample_spacing=sample_spacing,
        grid_spacing=grid_spacing,
        max_grid_size=max_grid_size,
        dataset_id=dataset_id,
        reuse_centerline=reuse,
        skip_mc_decimate=bool(speedups),
    )
    open_base_surface = built["open_base_surface"]
    branched_centerline = built["branched_centerline"]
    anatomical_profiles = built["anatomical_profiles"]
    if speedups:
        n_pre = open_base_surface.GetNumberOfPoints()
        t_dec = time.perf_counter()
        open_base_surface = decimate_variable_parent_tube(open_base_surface)
        print(
            f"  Post-uncap decimate {n_pre} -> {open_base_surface.GetNumberOfPoints()} "
            f"points in {time.perf_counter() - t_dec:.2f}s "
            f"(floor {VAR_MC_DECIMATE_MIN_POINTS})"
        )

    print("Step 8a: Computing local tube radius and raycasting stretch vs ground truth...")
    min_edge = REMESH_MIN_EDGE_MM
    remeshed_surface, _supervision, t_rm_s = supervise_and_remesh_verified(
        open_base_surface,
        vessel_mesh,
        branched_centerline,
        target_edge_length,
        min_edge,
        dataset_id=dataset_id,
    )
    print(f"  [t] remesh {t_rm_s:.2f}s")
    print(f"  -> Adaptive remeshed surface points: {remeshed_surface.GetNumberOfPoints()}")
    remesh_openings = inspect_openings(remeshed_surface)
    print(
        "  Openings after remesh: "
        + ", ".join(f"r={op['radius']:.3f}mm n={op['n_points']}" for op in remesh_openings)
    )

    # The profiles have to go in. Without them _loop_at_a_profile answers
    # False for every loop and finalize_surface falls back to pure geometry,
    # which on this dataset is the wrong judge: a real ostium here can be
    # smaller than a leftover rim, so an opening narrower than
    # MIN_OPENING_RADIUS_MM gets sealed as if it were a pinhole. The GT path
    # in remeshing.py has always passed them; these calls had not.
    final_surface, _n_regions = finalize_surface(
        remeshed_surface, profiles=anatomical_profiles
    )
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
    print(f"  [t] case total {time.perf_counter() - t_all:.2f}s")
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

    # The profiles have to go in. Without them _loop_at_a_profile answers
    # False for every loop and finalize_surface falls back to pure geometry,
    # which on this dataset is the wrong judge: a real ostium here can be
    # smaller than a leftover rim, so an opening narrower than
    # MIN_OPENING_RADIUS_MM gets sealed as if it were a pinhole. The GT path
    # in remeshing.py has always passed them; these calls had not.
    final_surface, _n_regions = finalize_surface(
        remeshed_surface, profiles=anatomical_profiles
    )
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
    parser.add_argument(
        "--case-timeout",
        type=float,
        default=DEFAULT_CASE_TIMEOUT_S,
        help=(
            "Kill a worker that runs longer than this many seconds and log it as a "
            "failure (0 disables). Default 5400."
        ),
    )
    parser.add_argument(
        "--worker-memory-gb",
        type=float,
        default=DEFAULT_WORKER_MEMORY_GB,
        help=(
            "Memory to reserve per worker when capping --workers against installed RAM "
            "(0 disables the cap). Default 2.5."
        ),
    )
    parser.add_argument(
        "--retry-failed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After the parallel pass, retry crashed/timed-out cases one at a time.",
    )
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


# 0xC0000409 (STATUS_STACK_BUFFER_OVERRUN) is what Windows reports when VTK
# aborts on a failed allocation; -9/-6/137 are the POSIX equivalents.
RESOURCE_FAILURE_CODES = frozenset({3221226505, 3221225725, -9, -6, 137})
TIMEOUT_RETURNCODE = -1000


def _is_resource_failure(returncode):
    return returncode in RESOURCE_FAILURE_CODES or returncode == TIMEOUT_RETURNCODE


def _run_one_worker(script_path, dataset_id, v_file, args, extra_cli_flags, case_timeout):
    """Run one case in a subprocess, killing it if it exceeds ``case_timeout``."""
    cmd = [
        sys.executable,
        "-u",
        script_path,
        "--case",
        str(dataset_id),
        "--vessel-file",
        v_file,
        "--output-dir",
        args.output_dir,
    ] + extra_cli_flags
    env = dict(os.environ)
    # Without this a crashed worker loses every "Step N" line to the pipe buffer,
    # which is why every crash in the 2026-09-17 run logged an empty step.
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
    )
    try:
        out, _ = proc.communicate(timeout=case_timeout if case_timeout > 0 else None)
        return proc.returncode, out or ""
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            out, _ = proc.communicate(timeout=60)
        except Exception:
            out = ""
        msg = f"case timed out after {case_timeout:.0f} s and was killed"
        print(f"[TIMEOUT] {dataset_id}: {msg}")
        return TIMEOUT_RETURNCODE, (out or "") + "\n" + msg + "\n"


def _installed_memory_gb():
    try:
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in getattr(os, "sysconf_names", {}):
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except (OSError, ValueError):
        pass
    try:
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullTotalPhys / 1024**3
    except Exception:
        pass
    return None


def _available_memory_gb():
    """Memory that can be handed out right now, or None if it cannot be read."""
    try:
        import psutil
    except ImportError:
        return None
    try:
        return float(psutil.virtual_memory().available) / 1024**3
    except Exception:
        return None


def memory_capped_workers(requested, per_worker_gb=DEFAULT_WORKER_MEMORY_GB):
    """Lower ``requested`` so the pool cannot exhaust RAM or starve the desktop.

    Every worker holds the extended surface, the Voronoi diagram and a remeshed
    surface of several hundred thousand triangles at once. 25 of them on a 32 GB
    machine is what made VTK fail to allocate and took thirteen workers down.

    The budget comes from free memory rather than installed memory: this machine
    has 32 GB but routinely only 3 GB of it going spare, and sizing the pool off
    the nameplate figure is how a batch run makes the desktop unusable.
    """
    requested = max(1, int(requested))
    if per_worker_gb <= 0:
        return requested

    available_gb = _available_memory_gb()
    total_gb = _installed_memory_gb()
    if available_gb is not None:
        budget_gb = max(available_gb - HOST_RESERVE_GB, MIN_POOL_MEMORY_GB)
        basis = f"{available_gb:.1f} GB free"
    elif total_gb:
        budget_gb = max(total_gb - 4.0, MIN_POOL_MEMORY_GB)
        basis = f"{total_gb:.0f} GB installed"
    else:
        return requested

    allowed = max(1, int(budget_gb // float(per_worker_gb)))
    n_cpu = os.cpu_count() or allowed
    cpu_allowed = max(1, n_cpu - HOST_RESERVE_THREADS)
    allowed = min(allowed, cpu_allowed)
    if allowed < requested:
        print(
            f"Capping workers {requested} -> {allowed} "
            f"({basis}, {per_worker_gb:.2f} GB per worker, "
            f"{n_cpu} threads less {HOST_RESERVE_THREADS} for the desktop)"
        )
    return min(requested, allowed)


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
    num_workers = memory_capped_workers(
        args.workers, getattr(args, "worker_memory_gb", DEFAULT_WORKER_MEMORY_GB)
    )
    num_workers = max(1, min(num_workers, num_cases))
    if getattr(args, "from_folder", False):
        print(f"Input folder (all meshes): {args.vessel_dir}")
    else:
        print(f"CSV Path: {args.csv}")
        print(f"Vessel Dir: {args.vessel_dir}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Processing limit: {args.limit} samples | Valid dataset cases: {num_cases}")
    print(f"Parallel Workers: {num_workers} (Requested={args.workers}, Active Workers={num_workers})")

    case_timeout = float(getattr(args, "case_timeout", DEFAULT_CASE_TIMEOUT_S) or 0.0)
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
        if case_timeout > 0:
            print(f"Per-case timeout: {case_timeout:.0f} s")
        task_queue = Queue()
        for item in valid_datasets:
            task_queue.put(item)
        pbar = tqdm(total=num_cases, desc="Processing Parallel")
        lock = threading.Lock()
        retry_queue = []

        def report(dataset_id, returncode, out):
            if on_worker_result is None:
                return
            try:
                on_worker_result(dataset_id, returncode, out)
            except Exception as log_exc:
                print(f"  WARNING: on_worker_result failed for {dataset_id}: {log_exc}")

        def worker_thread():
            while True:
                try:
                    dataset_id, v_file = task_queue.get_nowait()
                except Empty:
                    break
                returncode, out = _run_one_worker(
                    script_path, dataset_id, v_file, args, extra_cli_flags, case_timeout
                )
                report(dataset_id, returncode, out)
                if returncode != 0:
                    print(f"\n[ERROR] Case {dataset_id} failed (code {returncode}):\n{out[-2000:]}")
                    with lock:
                        failures.append((dataset_id, out[-2000:] if out else f"exit {returncode}"))
                        if _is_resource_failure(returncode):
                            retry_queue.append((dataset_id, v_file))
                pbar.update(1)

        threads = [threading.Thread(target=worker_thread) for _ in range(num_workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        pbar.close()
        print("Parallel execution complete. All worker processes finished.")

        if retry_queue and getattr(args, "retry_failed", True):
            print(
                f"Retrying {len(retry_queue)} case(s) one at a time "
                "(a crash under a full worker pool is usually memory pressure)."
            )
            for dataset_id, v_file in retry_queue:
                returncode, out = _run_one_worker(
                    script_path, dataset_id, v_file, args, extra_cli_flags, case_timeout
                )
                report(dataset_id, returncode, out)
                if returncode == 0:
                    print(f"  Retry succeeded: {dataset_id}")
                    failures = [f for f in failures if f[0] != dataset_id]
                else:
                    print(f"  Retry failed: {dataset_id} (code {returncode})")

    if failures:
        fail_path = os.path.join(args.output_dir, "failures.txt")
        with open(fail_path, "w", encoding="utf-8") as handle:
            for dataset_id, msg in failures:
                handle.write(f"{dataset_id}\n{msg}\n\n")
        print(f"Failed {len(failures)}/{num_cases} cases. See {fail_path}")
    else:
        print(f"All {num_cases} cases completed without recorded failures.")
