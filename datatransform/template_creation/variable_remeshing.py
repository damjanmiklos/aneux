"""AneuX variable (MISR-stretch) surface remeshing.

Appendix B item 12 (consume half): optional GT ostium frames drive
``clip_flow_extensions_and_uncap`` instead of the template's own measured
clip planes. Item 13: ``R_template``, ``StretchDistance`` and
``TargetEdgeLength`` are attached before adaptive remesh (so VMTK can
interpolate them) and restored on the *final* vertices after
``finalize_surface`` (which strips point arrays).
"""
import os
import sys

os.environ["VTK_OFFSCREEN"] = "1"
os.environ["EGL_PLATFORM"] = "surfaceless"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VTK_NUMBER_OF_THREADS"] = "1"
# VTK_NUMBER_OF_THREADS only caps the old vtkMultiThreader. VTK 9 runs its
# filters on vtkSMPTools, which is built here against TBB and sizes itself
# from the machine (32 threads), so with 20 workers the partitioning varies
# with load and the arithmetic comes out slightly differently each time. That
# is enough to move a cutter radius and flip a verdict: 20 identical runs of
# p398 split 3 passed / 17 failed. This is the variable that pins the pool.
os.environ["VTK_SMP_MAX_THREADS"] = "1"

import argparse
import hashlib
import inspect
import pickle
import shutil
import time
from contextlib import contextmanager

import numpy as np
import pyvista as pv
import vtk
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from aneux_paths import (
    CLEANDATA_TEMPLATE_MESH as DEFAULT_OUTPUT_DIR,
    CLEANDATA_UNIFORM,
)

from batch_run_log import (
    add_run_log_args,
    configure_batch_logging,
    finalize_run_logs,
    run_logged_case,
)

import vessel_pipeline
from vessel_pipeline import (
    DEFAULT_EXTENSION_LENGTH,
    DEFAULT_GRID_SPACING,
    DEFAULT_MAX_GRID_SIZE,
    DEFAULT_SAMPLE_SPACING,
    DEFAULT_TARGET_EDGE_LENGTH,
    REMESH_MIN_EDGE_MM,
    TemplateQualityError,
    _poly_points,
    add_shared_cli_args,
    assert_template_quality,
    assert_template_scale,
    build_parent_tube,
    clip_flow_extensions_and_uncap,
    decimate_variable_parent_tube,
    enforce_min_edge,
    finalize_surface,
    inspect_openings,
    measure_open_profiles,
    recompute_point_normals,
    run_batch,
    save_polydata,
    supervise_and_remesh_verified,
    to_vtk_poly,
    with_dataset_id,
)


# Its own folder: the GT path writes "gt_remesh_logs" beside the same output
# root, and a template run that shared it would overwrite the transcripts of
# the run its inputs came from.
LOG_FOLDER = "template_remesh_logs"

TEMPLATE_POINT_ARRAYS = ("R_template", "StretchDistance", "TargetEdgeLength")
OSTIUM_FRAMES_SUFFIX = ".ostium_frames.npz"


# ---------------------------------------------------------------------------
# Item 12 — ostium-frame contract (in-memory list[dict] / on-disk npz)
# ---------------------------------------------------------------------------

def _unit_normal(normal):
    arr = np.asarray(normal, dtype=np.float64).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"ostium normal must have shape (3,), got {arr.shape}")
    nrm = float(np.linalg.norm(arr))
    if nrm < 1e-12:
        raise ValueError("ostium normal has zero length")
    return arr / nrm


def normalize_cut_frames(cut_frames):
    """Coerce frames to ``list[dict]`` with origin (3,), unit normal, radius."""
    if cut_frames is None:
        return None
    if isinstance(cut_frames, dict) and "origin" in cut_frames:
        origins = np.asarray(cut_frames["origin"], dtype=np.float64)
        normals = np.asarray(cut_frames["normal"], dtype=np.float64)
        radii = np.asarray(cut_frames["radius"], dtype=np.float64).reshape(-1)
        if origins.ndim != 2 or origins.shape[1] != 3:
            raise ValueError(f"origin must be (K, 3), got {origins.shape}")
        if normals.shape != origins.shape:
            raise ValueError(f"normal must match origin {origins.shape}, got {normals.shape}")
        if radii.shape[0] != origins.shape[0]:
            raise ValueError(f"radius length {radii.shape[0]} != K={origins.shape[0]}")
        cut_frames = [
            {"origin": origins[i], "normal": normals[i], "radius": float(radii[i])}
            for i in range(origins.shape[0])
        ]
    if not cut_frames:
        return None
    out = []
    for i, frame in enumerate(cut_frames):
        if isinstance(frame, dict):
            origin = np.asarray(frame["origin"], dtype=np.float64).reshape(3)
            normal = _unit_normal(frame["normal"])
            radius = float(frame["radius"])
        else:
            origin = np.asarray(frame[0], dtype=np.float64).reshape(3)
            normal = _unit_normal(frame[1])
            radius = float(frame[2])
        out.append({"origin": origin, "normal": normal, "radius": radius})
    return out


def cut_frames_to_profiles(cut_frames):
    """Profile dicts ``clip_flow_extensions_and_uncap`` already understands."""
    frames = normalize_cut_frames(cut_frames)
    if not frames:
        return []
    return [
        {
            "index": i,
            "barycenter": f["origin"],
            "normal": f["normal"],
            "radius": f["radius"],
        }
        for i, f in enumerate(frames)
    ]


def cut_frames_to_tuples(cut_frames):
    frames = normalize_cut_frames(cut_frames)
    if not frames:
        return []
    return [(f["origin"], f["normal"], float(f["radius"])) for f in frames]


def ostium_frames_path(stem, directory):
    return os.path.join(directory, f"{stem}{OSTIUM_FRAMES_SUFFIX}")


def resolve_cut_frames(
    dataset_id,
    v_file,
    cut_frames=None,
    ostium_frames=None,
    cut_frames_path=None,
    cut_frames_dir=None,
):
    """In-memory frames win; else load ``{stem}.ostium_frames.npz`` (read-only)."""
    frames = normalize_cut_frames(cut_frames if cut_frames is not None else ostium_frames)
    if frames:
        return frames, "argument"
    stem = str(dataset_id) if dataset_id else os.path.splitext(os.path.basename(v_file))[0]
    candidates = []
    if cut_frames_path:
        candidates.append(cut_frames_path)
    if v_file:
        candidates.append(ostium_frames_path(stem, os.path.dirname(os.path.abspath(v_file))))
    if cut_frames_dir:
        candidates.append(ostium_frames_path(stem, cut_frames_dir))
    candidates.append(ostium_frames_path(stem, CLEANDATA_UNIFORM))
    seen = set()
    for path in candidates:
        path = os.path.abspath(path)
        if path in seen:
            continue
        seen.add(path)
        if os.path.isfile(path):
            with np.load(path) as data:
                loaded = {
                    "origin": np.ascontiguousarray(data["origin"], dtype=np.float64),
                    "normal": np.ascontiguousarray(data["normal"], dtype=np.float64),
                    "radius": np.ascontiguousarray(data["radius"], dtype=np.float64),
                }
            return normalize_cut_frames(loaded), path
    return None, None


# How far a GT frame may sit from the template opening it is matched to before
# the match stops being believable. The Hungarian assignment always returns a
# pairing -- it has no notion of "no match" -- so without a distance test a
# frame belonging to a different ostium is accepted in silence and the template
# is clipped on the wrong plane. The two are the same opening seen on two
# surfaces, so they should sit within a radius of each other; this allows three,
# and says so when it has to stretch that far.
CUT_FRAME_MATCH_RADII = 3.0
CUT_FRAME_MATCH_FLOOR_MM = 1.0


def align_cut_frames_to_profiles(cut_frames, profiles):
    """One frame per profile (Hungarian on origin distance). K ostia, not vertices."""
    frames = normalize_cut_frames(cut_frames)
    if not frames:
        return []
    if not profiles:
        return frames
    bary = np.stack(
        [np.asarray(p["barycenter"], dtype=np.float64).reshape(3) for p in profiles]
    )
    origins = np.stack([f["origin"] for f in frames])
    dist = np.linalg.norm(bary[:, None, :] - origins[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(dist)
    assigned = {int(r): frames[int(c)] for r, c in zip(rows, cols)}
    picked = {int(r): float(dist[int(r), int(c)]) for r, c in zip(rows, cols)}
    ordered = []
    reused = 0
    for i in range(len(profiles)):
        if i in assigned:
            ordered.append(assigned[i])
        else:
            j = int(np.argmin(dist[i]))
            ordered.append(frames[j])
            picked[i] = float(dist[i, j])
            reused += 1
    if reused:
        print(
            f"  WARNING: {len(profiles)} template profiles vs {len(frames)} GT frames; "
            f"reused nearest frame on {reused} opening(s)"
        )
    leftover = len(frames) - min(len(frames), len(profiles))
    if leftover > 0:
        print(f"  WARNING: {leftover} GT ostium frame(s) unmatched to template profiles")

    far = []
    for i, d in sorted(picked.items()):
        r = float(profiles[i].get("radius", 0.0) or 0.0)
        limit = max(CUT_FRAME_MATCH_RADII * r, CUT_FRAME_MATCH_FLOOR_MM)
        if d > limit:
            far.append((i, d, r, limit))
    if far:
        worst = max(far, key=lambda t: t[1] / t[3])
        print(
            f"  WARNING: {len(far)} of {len(profiles)} ostium frame(s) matched to a "
            f"template opening further away than {CUT_FRAME_MATCH_RADII:.0f} opening "
            f"radii; worst is opening {worst[0]} at {worst[1]:.3f} mm against a "
            f"radius of {worst[2]:.3f} mm. The template will be clipped on those "
            "planes, so check the ostium frames before trusting this case."
        )
    return ordered


# ---------------------------------------------------------------------------
# Item 12 — thread cut_frames into clip (kwarg now, HEAD fallback)
# ---------------------------------------------------------------------------

@contextmanager
def _patch_opening_clip_frames(cut_frames):
    """HEAD: ``opening_clip_frames`` returns the GT planes instead of measuring."""
    import vessel_pipeline as vp

    frames = normalize_cut_frames(cut_frames)
    orig = vp.opening_clip_frames

    def _forced(centerline, profiles):
        aligned = align_cut_frames_to_profiles(frames, profiles)
        return cut_frames_to_tuples(aligned)

    vp.opening_clip_frames = _forced
    try:
        yield
    finally:
        vp.opening_clip_frames = orig


def clip_flow_extensions_and_uncap_with_frames(
    base_surface,
    profiles=None,
    *,
    cut_frames=None,
    **kwargs,
):
    """Call ``clip_flow_extensions_and_uncap`` with GT frames when available.

    Writes ``cut_frames=`` whenever the sibling signature has that kwarg.
    On current HEAD, the same planes are forced through ``opening_clip_frames``
    so pipe-section cuts still run — no extra VMTK centerline.
    ``measure_open_profiles`` is only used when neither frames nor profiles
    were supplied (clip still needs a keep-set for pinholes).
    """
    frames = normalize_cut_frames(cut_frames)
    if frames:
        clip_profiles = cut_frames_to_profiles(frames)
        params = inspect.signature(clip_flow_extensions_and_uncap).parameters
        if "cut_frames" in params:
            print(
                f"  clip_flow_extensions_and_uncap(cut_frames={len(frames)}) "
                "(GT ostium planes; skip template measure_open_profiles)"
            )
            return clip_flow_extensions_and_uncap(
                base_surface, clip_profiles, cut_frames=frames, **kwargs
            )
        print(
            f"  clip_flow_extensions_and_uncap HEAD fallback: {len(frames)} GT "
            "frames via opening_clip_frames patch (no second VMTK pass)"
        )
        with _patch_opening_clip_frames(frames):
            return clip_flow_extensions_and_uncap(base_surface, clip_profiles, **kwargs)
    if not profiles:
        profiles = measure_open_profiles(base_surface)
    return clip_flow_extensions_and_uncap(base_surface, profiles, **kwargs)


@contextmanager
def inject_cut_frames(cut_frames):
    """Patch ``vessel_pipeline.clip_flow_extensions_and_uncap`` for ``build_parent_tube``.

    Also patches ``opening_clip_frames`` so ``extra_opening_spheres`` uses the
    same GT ostia. No second Voronoi / remesh.
    """
    frames = normalize_cut_frames(cut_frames)
    if not frames:
        yield
        return
    import vessel_pipeline as vp

    orig_clip = vp.clip_flow_extensions_and_uncap
    orig_ocf = vp.opening_clip_frames

    def _clip(base_surface, profiles, **kwargs):
        kwargs.pop("cut_frames", None)
        return clip_flow_extensions_and_uncap_with_frames(
            base_surface, profiles, cut_frames=frames, **kwargs
        )

    def _ocf(centerline, profiles):
        return cut_frames_to_tuples(align_cut_frames_to_profiles(frames, profiles))

    vp.clip_flow_extensions_and_uncap = _clip
    vp.opening_clip_frames = _ocf
    try:
        yield
    finally:
        vp.clip_flow_extensions_and_uncap = orig_clip
        vp.opening_clip_frames = orig_ocf


def _build_parent_tube(vessel_mesh, cut_frames=None, **kwargs):
    frames = normalize_cut_frames(cut_frames)
    params = inspect.signature(build_parent_tube).parameters
    if frames is not None and "cut_frames" in params:
        kwargs["cut_frames"] = frames
    with inject_cut_frames(frames):
        return build_parent_tube(vessel_mesh, **kwargs)


# ---------------------------------------------------------------------------
# Parent-tube cache
# ---------------------------------------------------------------------------
# The tube (flow extensions, Voronoi trace, polyball, uncap) is a pure function
# of the GT mesh, its frames, four grid parameters and the code, and it is most
# of a case: the Voronoi trace alone is 17-50 s typically and 299 s on
# SNF00000426_03, against 20-50 s for the remesh that follows. This path is
# rerun many times on the same GT set, so the tube is kept on disk and a rerun
# pays only for the raycast and the remesh. The key hashes the input file's
# bytes, the frames, the parameters and the full source of both modules, so
# any edit to the code rebuilds rather than reusing a tube it would not make.
DEFAULT_TUBE_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".variable_tube_cache"
)
TUBE_CACHE_FORMAT = 1


def _file_digest(path, h):
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)


def tube_cache_key(v_file, frames, params, reuse_centerline=None):
    """Hex digest naming the tube this input, these frames and this code build."""
    h = hashlib.sha256()
    h.update(f"format={TUBE_CACHE_FORMAT}".encode())
    for path in (v_file, reuse_centerline, vessel_pipeline.__file__, os.path.abspath(__file__)):
        h.update(b"|file|")
        if path:
            _file_digest(path, h)
    h.update(repr(sorted(params.items())).encode())
    for frame in frames or []:
        h.update(np.asarray(frame["origin"], dtype=np.float64).tobytes())
        h.update(np.asarray(frame["normal"], dtype=np.float64).tobytes())
        h.update(np.float64(frame["radius"]).tobytes())
    return h.hexdigest()


def _write_vtp(poly, path):
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(to_vtk_poly(poly))
    writer.SetDataModeToBinary()
    if not writer.Write():
        raise OSError(f"could not write {path}")


def _read_vtp(path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(reader.GetOutput())
    return out


def load_cached_tube(cache_dir, key, vessel_mesh):
    """The ``built`` dict stored under ``key``, or None on a miss."""
    entry = os.path.join(cache_dir, key)
    meta_path = os.path.join(entry, "meta.pkl")
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, "rb") as fh:
            meta = pickle.load(fh)
        built = {
            "anatomical_profiles": meta["anatomical_profiles"],
            "n_clipped": meta["n_clipped"],
            "open_base_surface": _read_vtp(os.path.join(entry, "open_base_surface.vtp")),
            "branched_centerline": _read_vtp(os.path.join(entry, "branched_centerline.vtp")),
        }
        if meta["vessel_repaired"]:
            built["vessel_mesh"] = pv.wrap(_read_vtp(os.path.join(entry, "vessel_mesh.vtp")))
        else:
            built["vessel_mesh"] = vessel_mesh
    except Exception as exc:  # a torn entry is a miss, never a wrong tube
        print(f"  Tube cache entry {key[:12]} unreadable ({exc}); rebuilding")
        return None
    if built["open_base_surface"].GetNumberOfPoints() == 0:
        return None
    return built


def store_cached_tube(cache_dir, key, built, vessel_mesh):
    """Write ``built`` under ``key``; the entry appears whole or not at all."""
    entry = os.path.join(cache_dir, key)
    if os.path.isdir(entry):
        return
    tmp = f"{entry}.tmp{os.getpid()}"
    try:
        os.makedirs(tmp, exist_ok=True)
        repaired = built.get("vessel_mesh", vessel_mesh) is not vessel_mesh
        _write_vtp(built["open_base_surface"], os.path.join(tmp, "open_base_surface.vtp"))
        _write_vtp(built["branched_centerline"], os.path.join(tmp, "branched_centerline.vtp"))
        if repaired:
            _write_vtp(built["vessel_mesh"], os.path.join(tmp, "vessel_mesh.vtp"))
        with open(os.path.join(tmp, "meta.pkl"), "wb") as fh:
            pickle.dump(
                {
                    "anatomical_profiles": built["anatomical_profiles"],
                    "n_clipped": int(built["n_clipped"]),
                    "vessel_repaired": bool(repaired),
                },
                fh,
            )
        os.replace(tmp, entry)
    except Exception as exc:
        print(f"  Tube cache: could not store {key[:12]} ({exc})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Item 13 — supervision arrays on FINAL vertices
# ---------------------------------------------------------------------------

def _mutable_poly(surface):
    """Mutate this object: ``to_vtk_poly`` DeepCopies and would drop writes."""
    if isinstance(surface, vtk.vtkPolyData):
        return surface
    return to_vtk_poly(surface)


def _named_point_array(surface, name):
    poly = _mutable_poly(surface)
    arr = poly.GetPointData().GetArray(name)
    n = int(poly.GetNumberOfPoints())
    if arr is None or int(arr.GetNumberOfTuples()) != n:
        return None
    vals = np.ascontiguousarray(vtk_to_numpy(arr), dtype=np.float64).reshape(-1)
    if vals.size != n:
        return None
    return vals


def _set_named_point_array(surface, name, values):
    poly = _mutable_poly(surface)
    n = int(poly.GetNumberOfPoints())
    vals = np.ascontiguousarray(values, dtype=np.float64).reshape(-1)
    if vals.size != n:
        raise TemplateQualityError(f"{name} has {vals.size} values for {n} points")
    vtk_arr = numpy_to_vtk(vals, deep=True)
    vtk_arr.SetName(name)
    pd = poly.GetPointData()
    if pd.GetArray(name) is not None:
        pd.RemoveArray(name)
    pd.AddArray(vtk_arr)
    return poly


def nearest_neighbour_index(src_pts, dst_pts):
    """Nearest source vertex for every destination vertex, or None if either is empty.

    The three supervision arrays are carried between the same two point sets, so
    the tree and the query are shared: doing it per array built the same tree
    three times over meshes of a few hundred thousand vertices.
    """
    src_pts = np.ascontiguousarray(src_pts, dtype=np.float64)
    dst_pts = np.ascontiguousarray(dst_pts, dtype=np.float64)
    if int(dst_pts.shape[0]) == 0 or int(src_pts.shape[0]) == 0:
        return None
    _, idx = cKDTree(src_pts).query(dst_pts, k=1, workers=1)
    return np.asarray(idx, dtype=np.int64)


def nearest_neighbour_scalars(src_pts, src_vals, dst_pts, idx=None):
    """Vectorised NN transfer (cKDTree). Not a Python vertex loop.

    ``idx`` is the pairing from ``nearest_neighbour_index`` when the caller has
    already built it for these same two point sets.
    """
    src_vals = np.ascontiguousarray(src_vals, dtype=np.float64).reshape(-1)
    n_dst = int(np.asarray(dst_pts).shape[0])
    if idx is None:
        idx = nearest_neighbour_index(src_pts, dst_pts)
    if idx is None:
        return np.zeros(n_dst, dtype=np.float64)
    return src_vals[idx]


def attach_template_supervision_arrays(surface, r_template, stretch_distances, target_edge=None):
    """Write the three named point arrays on ``surface`` (pre-remesh fast path)."""
    poly = _mutable_poly(surface)
    n = int(poly.GetNumberOfPoints())
    r = np.ascontiguousarray(r_template, dtype=np.float64).reshape(-1)
    stretch = np.ascontiguousarray(stretch_distances, dtype=np.float64).reshape(-1)
    if r.size != n or stretch.size != n:
        raise TemplateQualityError(
            f"supervision length mismatch: n={n} R={r.size} stretch={stretch.size}"
        )
    if target_edge is None:
        existing = _named_point_array(poly, "TargetEdgeLength")
        if existing is None:
            raise TemplateQualityError("TargetEdgeLength missing; build it before attach")
        target_edge = existing
    else:
        target_edge = np.ascontiguousarray(target_edge, dtype=np.float64).reshape(-1)
        if target_edge.size != n:
            raise TemplateQualityError(f"TargetEdgeLength length {target_edge.size} != {n}")
    _set_named_point_array(poly, "R_template", r)
    _set_named_point_array(poly, "StretchDistance", stretch)
    _set_named_point_array(poly, "TargetEdgeLength", target_edge)
    return poly


def snapshot_template_supervision(surface, fallback_pts=None, fallback_arrays=None):
    """Harvest interpolated arrays; NN-fill from pre-remesh where missing/non-finite."""
    poly = _mutable_poly(surface)
    _, pts = _poly_points(poly)
    arrays = {}
    n_nn = 0
    nn_idx = None
    for name in TEMPLATE_POINT_ARRAYS:
        vals = _named_point_array(poly, name)
        need_nn = vals is None or not np.all(np.isfinite(vals))
        if need_nn:
            if fallback_pts is None or fallback_arrays is None or name not in fallback_arrays:
                raise TemplateQualityError(
                    f"{name} missing after remesh and no pre-remesh fallback was supplied"
                )
            if nn_idx is None:
                nn_idx = nearest_neighbour_index(fallback_pts, pts)
            filled = nearest_neighbour_scalars(
                fallback_pts, fallback_arrays[name], pts, idx=nn_idx
            )
            if vals is None:
                vals = filled
                n_nn += int(pts.shape[0])
            else:
                bad = ~np.isfinite(vals)
                n_nn += int(bad.sum())
                vals = vals.copy()
                vals[bad] = filled[bad]
            _set_named_point_array(poly, name, vals)
        arrays[name] = np.ascontiguousarray(vals, dtype=np.float64)
    return poly, pts.copy(), arrays, n_nn


def restore_template_supervision(dst_surface, src_pts, src_arrays):
    """NN from a snapshot onto ``dst`` (covers ``finalize_surface`` stripping)."""
    poly = _mutable_poly(dst_surface)
    _, dst_pts = _poly_points(poly)
    nn_idx = None
    for name in TEMPLATE_POINT_ARRAYS:
        existing = _named_point_array(poly, name)
        # _named_point_array already refuses an array whose length is not the
        # point count, so an array that is here and finite everywhere needs no
        # transfer at all -- and the nearest-neighbour search it used to run
        # anyway was over every vertex of the final mesh, three times a case.
        if existing is not None and np.all(np.isfinite(existing)):
            _set_named_point_array(poly, name, existing)
            continue
        if nn_idx is None:
            nn_idx = nearest_neighbour_index(src_pts, dst_pts)
        filled = nearest_neighbour_scalars(
            src_pts, src_arrays[name], dst_pts, idx=nn_idx
        )
        if existing is not None and np.any(np.isfinite(existing)):
            filled = np.where(np.isfinite(existing), existing, filled)
        _set_named_point_array(poly, name, filled)
    return poly


# Slack on the supervision contract below. Every bound there holds exactly at
# the raycast's own vertices, and linear interpolation and nearest-neighbour
# transfer both keep it (S <= 3.5 R is linear in the pair), so this only has to
# absorb float32 storage. Over the 40 templates of the 2026-09-23 ladder run
# the closest any vertex came to 3.5 R was 0.0056 mm under it.
SUPERVISION_CONTRACT_TOL = 1e-5


def assert_template_supervision_arrays(surface, context="template",
                                       base_edge=DEFAULT_TARGET_EDGE_LENGTH,
                                       min_edge=REMESH_MIN_EDGE_MM):
    """Every array present and finite, and inside the bounds the raycast promises.

    Presence is not enough: a stretch the raycast could never have written is
    a template that would teach the decoder a wall that is not there. The ray
    only accepts a hit at most 3.5 R_template out, and a point outside the GT
    gets 0, never a negative distance -- r* = R_template + StretchDistance
    downstream, so a negative one would shrink the tube twice over (see
    _zero_rays_that_left_the_gt). The target edge is clamped to
    [min_edge, base_edge] where it is built.
    """
    poly = to_vtk_poly(surface)
    n = int(poly.GetNumberOfPoints())
    issues = []
    stats = {}
    vals_by_name = {}
    for name in TEMPLATE_POINT_ARRAYS:
        vals = _named_point_array(poly, name)
        if vals is None:
            issues.append(f"{name} missing or length != {n} points")
            continue
        if not np.all(np.isfinite(vals)):
            issues.append(f"{name} has non-finite values")
            continue
        vals_by_name[name] = vals
        stats[name] = (float(np.min(vals)), float(np.max(vals)))
    tol = SUPERVISION_CONTRACT_TOL
    r = vals_by_name.get("R_template")
    s = vals_by_name.get("StretchDistance")
    t = vals_by_name.get("TargetEdgeLength")
    if r is not None and np.any(r <= 0):
        issues.append(f"R_template is not positive at {int(np.sum(r <= 0))} vertices")
    if s is not None and np.any(s < -tol):
        issues.append(
            f"StretchDistance is negative at {int(np.sum(s < -tol))} vertices "
            f"(min {s.min():.4f} mm)"
        )
    if r is not None and s is not None:
        over = s > 3.5 * r + tol
        if np.any(over):
            k = int(np.argmax(s - 3.5 * r))
            issues.append(
                f"StretchDistance exceeds the raycast's 3.5 R ceiling at "
                f"{int(over.sum())} vertices (worst {s[k]:.3f} mm at R={r[k]:.3f})"
            )
    if t is not None:
        bad = (t < min_edge - tol) | (t > base_edge + tol)
        if np.any(bad):
            issues.append(
                f"TargetEdgeLength outside [{min_edge}, {base_edge}] mm at "
                f"{int(bad.sum())} vertices (range {t.min():.4f}-{t.max():.4f})"
            )
    if issues:
        raise TemplateQualityError(f"{context} supervision arrays failed: " + "; ".join(issues))
    return stats


def _attach_supervision(surface, r_template, stretch_distances, edge_lengths):
    """Put the supervision arrays on the tube before it is remeshed.

    They have to be there before, not after: VMTK interpolates the point
    arrays it is handed onto the new vertices, which is a far better answer
    than the nearest-neighbour fill that snapshot_template_supervision falls
    back to when they are missing.
    """
    return attach_template_supervision_arrays(
        surface, r_template, stretch_distances, target_edge=edge_lengths
    )


# ---------------------------------------------------------------------------
# Per-case entry
# ---------------------------------------------------------------------------

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
    cut_frames=None,
    ostium_frames=None,
    cut_frames_dir=None,
    cut_frames_path=None,
    tube_cache_dir=None,
):
    print(f"\n=========================================\nProcessing Adaptive Variable Remeshing Case: {dataset_id}")
    t_all = time.perf_counter()
    vessel_mesh = pv.read(v_file)

    frames, frames_src = resolve_cut_frames(
        dataset_id,
        v_file,
        cut_frames=cut_frames,
        ostium_frames=ostium_frames,
        cut_frames_path=cut_frames_path,
        cut_frames_dir=cut_frames_dir,
    )
    if frames:
        print(
            f"  Item 12: consuming {len(frames)} GT ostium frames from {frames_src} "
            "(template uncap uses these planes, not a second VMTK centerline)"
        )
    else:
        print("  Item 12: no GT ostium frames on disk; uncap uses measured template profiles")

    reuse = None
    if speedups:
        from aneux_paths import CLEANDATA_ORIGINAL_CENTERLINE

        cl_path = os.path.join(CLEANDATA_ORIGINAL_CENTERLINE, f"{dataset_id}.vtp")
        if os.path.isfile(cl_path):
            reuse = cl_path
        else:
            print("  original_centerline missing; will extract Voronoi")

    tube_params = dict(
        extension_length=extension_length,
        sample_spacing=sample_spacing,
        grid_spacing=grid_spacing,
        max_grid_size=max_grid_size,
        skip_mc_decimate=bool(speedups),
    )
    built = key = None
    if tube_cache_dir:
        t_key = time.perf_counter()
        key = tube_cache_key(v_file, frames, tube_params, reuse_centerline=reuse)
        built = load_cached_tube(tube_cache_dir, key, vessel_mesh)
        if built is not None:
            print(
                f"  Parent tube from cache {key[:12]} "
                f"({time.perf_counter() - t_key:.2f}s; steps 1-7 skipped)"
            )
    if built is None:
        built = _build_parent_tube(
            vessel_mesh,
            cut_frames=frames,
            dataset_id=dataset_id,
            reuse_centerline=reuse,
            **tube_params,
        )
        if key is not None:
            store_cached_tube(tube_cache_dir, key, built, vessel_mesh)
    open_base_surface = built["open_base_surface"]
    branched_centerline = built["branched_centerline"]
    anatomical_profiles = built["anatomical_profiles"]
    # If the tube builder had to reopen a sealed-off branch, the ground truth
    # the raycast measures against has to be the mesh that has it.
    vessel_mesh = built.get("vessel_mesh", vessel_mesh)
    if speedups:
        n_pre = open_base_surface.GetNumberOfPoints()
        t_dec = time.perf_counter()
        open_base_surface = decimate_variable_parent_tube(open_base_surface)
        print(
            f"  Post-uncap decimate {n_pre} -> {open_base_surface.GetNumberOfPoints()} "
            f"points in {time.perf_counter() - t_dec:.2f}s"
        )

    print("Step 8a: Computing local tube radius and raycasting stretch vs ground truth...")
    min_edge = REMESH_MIN_EDGE_MM
    remeshed_surface, supervision, t_rm_s = supervise_and_remesh_verified(
        open_base_surface,
        vessel_mesh,
        branched_centerline,
        target_edge_length,
        min_edge,
        prepare=_attach_supervision,
        dataset_id=dataset_id,
    )
    pre_pts = supervision["pre_pts"]
    pre_arrays = {name: supervision[key] for name, key in (
        ("R_template", "r_template"),
        ("StretchDistance", "stretch_distances"),
        ("TargetEdgeLength", "edge_lengths"),
    )}
    t_attach_s = supervision["t_attach_s"]
    print(f"  [t] remesh {t_rm_s:.2f}s")
    print(f"  -> Adaptive remeshed surface points: {remeshed_surface.GetNumberOfPoints()}")
    remesh_openings = inspect_openings(remeshed_surface)
    print(
        "  Openings after remesh: "
        + ", ".join(f"r={op['radius']:.3f}mm n={op['n_points']}" for op in remesh_openings)
    )

    t_xfer = time.perf_counter()
    remeshed_surface, snap_pts, snap_arrays, n_nn = snapshot_template_supervision(
        remeshed_surface, fallback_pts=pre_pts, fallback_arrays=pre_arrays
    )
    if n_nn:
        print(
            f"  Item 13: remesh dropped supervision arrays; "
            f"NN-filled {n_nn} values from pre-remesh "
            f"({remeshed_surface.GetNumberOfPoints()} vertices)"
        )
    else:
        print("  Item 13: VMTK interpolated R_template / StretchDistance / TargetEdgeLength")

    # The profiles have to go in. Without them _loop_at_a_profile answers
    # False for every loop and finalize_surface falls back to pure geometry,
    # which on this dataset is the wrong judge: a real ostium here can be
    # smaller than a leftover rim, so an opening narrower than
    # MIN_OPENING_RADIUS_MM gets sealed as if it were a pinhole. The GT path
    # in remeshing.py has always passed them; these calls had not.
    final_surface, _n_regions = finalize_surface(
        remeshed_surface, profiles=anatomical_profiles
    )
    # finalize_surface welds at WELD_TOLERANCE_MM, a hundredth of the target
    # edge, so an edge it leaves can still be orders of magnitude under the
    # quality floor: p402 shipped nothing because its final mesh carried a
    # 0.000001 mm edge and p414 a 0.000034 mm one, both made after the remesh
    # step that already ran enforce_min_edge. The GT path has folded them here
    # since it was written; this path never did. The fold is checked before it
    # is kept, so a surface it cannot improve comes back untouched -- and it
    # runs before the supervision transfer so the arrays land on the vertices
    # that actually ship.
    folded = enforce_min_edge(final_surface, label="template")
    if folded is not final_surface:
        final_surface = recompute_point_normals(folded, auto_orient=False)
    final_surface = restore_template_supervision(final_surface, snap_pts, snap_arrays)
    t_xfer_s = time.perf_counter() - t_xfer
    print(
        f"  [t] item13 attach {t_attach_s:.4f}s + transfer {t_xfer_s:.4f}s "
        f"(remesh {t_rm_s:.2f}s)"
    )

    assert_template_scale(final_surface, vessel_mesh, context=dataset_id)
    openings = assert_template_quality(final_surface, context=dataset_id)
    # build_parent_tube checks the uncap opened every end, but finalize_surface
    # runs afterwards and it closes holes: close_wall_pinholes and
    # remove_spurious_openings can each take an ostium the remesh has narrowed.
    # Nothing between there and disk looked again, so a template could ship with
    # fewer ostia than the GT it is supposed to be the parent of, and the
    # decoder would be trained to reproduce a vessel with a branch missing.
    n_want = len(frames) if frames else len(anatomical_profiles)
    if len(openings) != n_want:
        raise TemplateQualityError(
            f"{dataset_id}: template finished with {len(openings)} openings against "
            f"{n_want} " + ("GT ostium frames" if frames else "anatomical profiles")
            + "; a template whose ostia do not match the ground truth cannot supervise."
        )
    stats = assert_template_supervision_arrays(
        final_surface, context=dataset_id, base_edge=target_edge_length, min_edge=min_edge
    )
    print(
        "  Item 13 final arrays: "
        + ", ".join(f"{n}=[{lo:.4g}, {hi:.4g}]" for n, (lo, hi) in stats.items())
    )

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{dataset_id}.vtp")
    save_polydata(final_surface, out_file)
    read_back = pv.read(out_file)
    n_pts = int(read_back.n_points)
    for name in TEMPLATE_POINT_ARRAYS:
        if name not in read_back.point_data:
            raise TemplateQualityError(f"{dataset_id}: saved mesh missing {name}")
        if np.asarray(read_back.point_data[name]).reshape(-1).size != n_pts:
            raise TemplateQualityError(f"{dataset_id}: saved {name} length != n_points")
    print(f"Successfully saved variable remeshed surface to: {out_file}")
    print(
        f"  -> Verified Saved Mesh: {read_back.n_points} points, {read_back.n_cells} cells, "
        f"disk size={os.path.getsize(out_file)} bytes"
    )
    print(
        f"  -> Verified Open Boundaries Count: {len(openings)} "
        f"(pipe-section clipped {int(built['n_clipped'])}; "
        f"{len(anatomical_profiles)} anatomical profiles"
        + (f"; {len(frames)} GT cut frames" if frames else "")
        + ")"
    )
    print(f"  [t] case total {time.perf_counter() - t_all:.2f}s")
    return out_file


def _process_one(dataset_id, v_file, args):
    def work():
        return process_variable_dataset(
            dataset_id=dataset_id,
            v_file=v_file,
            output_dir=args.output_dir,
            target_edge_length=args.target_edge_length,
            extension_length=args.extension_length,
            sample_spacing=args.sample_spacing,
            grid_spacing=args.grid_spacing,
            max_grid_size=args.max_grid_size,
            speedups=args.speedups,
            cut_frames_dir=getattr(args, "cut_frames_dir", None),
            cut_frames_path=getattr(args, "cut_frames_file", None),
            tube_cache_dir=getattr(args, "tube_cache_dir", None) or None,
        )

    # Stage 2 records every case this way; without it only a crashed worker
    # left a trace, and a finished run reported "0 success" however many
    # templates it had delivered.
    return run_logged_case(
        dataset_id,
        v_file,
        args,
        work,
        log_folder_name=LOG_FOLDER,
        default_output_dir=DEFAULT_OUTPUT_DIR,
    )


def main():
    parser = argparse.ArgumentParser(description="AneuX variable (MISR-stretch) surface remeshing pipeline")
    add_shared_cli_args(parser, DEFAULT_OUTPUT_DIR, default_workers=20, include_remesh_grid=True)
    parser.add_argument(
        "--speedups",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Variable-only experimental path (reuse original_centerline, post-uncap 8k "
            "decimate). Off by default: it drops ostia when a reused centerline is "
            "fragmented. Fast uncap is the shared default; compiled raycast is on."
        ),
    )
    parser.add_argument(
        "--cut-frames-dir",
        type=str,
        default=None,
        help="Directory of {stem}.ostium_frames.npz (GT ostium planes for template uncap).",
    )
    parser.add_argument(
        "--cut-frames-file",
        type=str,
        default=None,
        help="Explicit {stem}.ostium_frames.npz for --case (overrides directory search).",
    )
    parser.add_argument(
        "--tube-cache-dir",
        type=str,
        default=DEFAULT_TUBE_CACHE_DIR,
        help=(
            "Where built parent tubes are kept between runs (keyed on input, frames, "
            "parameters and code). Pass an empty string to build every tube afresh."
        ),
    )
    add_run_log_args(parser, LOG_FOLDER)
    parser.set_defaults(vessel_dir=CLEANDATA_UNIFORM, from_folder=True)
    args = parser.parse_args()
    # Keeps every worker's stdout, which is where the per-case raycast
    # diagnostics are printed. Without it only a failing case leaves a trace
    # and the passing 700 cannot be audited at all.
    extra_log, on_worker_result = configure_batch_logging(
        args, LOG_FOLDER, DEFAULT_OUTPUT_DIR
    )
    extra = [
        "--target-edge-length", str(args.target_edge_length),
        "--extension-length", str(args.extension_length),
        "--sample-spacing", str(args.sample_spacing),
        "--grid-spacing", str(args.grid_spacing),
        "--max-grid-size", str(args.max_grid_size),
        "--speedups" if args.speedups else "--no-speedups",
    ] + extra_log
    if args.cut_frames_dir:
        extra.extend(["--cut-frames-dir", str(args.cut_frames_dir)])
    if args.cut_frames_file:
        extra.extend(["--cut-frames-file", str(args.cut_frames_file)])
    extra.extend(["--tube-cache-dir", str(args.tube_cache_dir or "")])
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
