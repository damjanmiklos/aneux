"""One process per case: shared centerline, GT remesh, template, ostium frames.

``process_case(v_file, *, out_gt_dir, out_cl_dir, out_tpl_dir, ...)`` is the
Stage-2 generator entry point (§2.9, Appendix B item 14). Inputs are already
open vessels (cleandata / cleaned meshes); there is no cap-exclusion list.

1. Run GT remesh (working copy + centerline computed there once when the
   remesh API cannot take an injected centerline; injected when it can).
2. Receive ostium cut frames from the GT step and persist
   ``{stem}.ostium_frames.npz`` next to the GT mesh.
3. Write ``original_centerline`` from the shared centerline when returned,
   otherwise from the remeshed GT.
4. Build the template with those same frames (``cut_frames=``).
5. Hard-gate ``n_clipped == n_profiles`` on both products when the counts
   are available; also compare opening counts to ``len(frames)`` when cheap.

Output directories have no live-``cleandata/`` default. Tests must pass
scratch paths explicitly. This module does not import VMTK at load time.
"""
from __future__ import annotations

import inspect
import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_TEMPLATE_DIR = os.path.abspath(os.path.dirname(__file__))
if _TEMPLATE_DIR not in sys.path:
    sys.path.insert(0, _TEMPLATE_DIR)

DEFAULT_GT_EDGE_LENGTH_MM = 0.15
DEFAULT_TEMPLATE_EDGE_LENGTH_MM = 0.5
DEFAULT_EXTENSION_LENGTH = 5.0
DEFAULT_SAMPLE_SPACING = 0.1
TEMPLATE_CENTERLINE_FOLDER = "template_centerline"
VESSEL_EXTENSIONS = (".vtp", ".stl", ".vtk", ".ply")

__all__ = [
    "DEFAULT_EXTENSION_LENGTH",
    "DEFAULT_GT_EDGE_LENGTH_MM",
    "DEFAULT_SAMPLE_SPACING",
    "DEFAULT_TEMPLATE_EDGE_LENGTH_MM",
    "TEMPLATE_CENTERLINE_FOLDER",
    "as_ostium_frame_dicts",
    "case_id_from_vessel",
    "is_template_centerline_dir",
    "load_ostium_frames",
    "normalize_dataset_id",
    "ostium_frames_npz_path",
    "process_case",
    "save_ostium_frames",
]


class ProcessCaseError(RuntimeError):
    """Quality / orchestration failure when VMTK's TemplateQualityError is unavailable."""


def normalize_dataset_id(value):
    """Basename stem of a vessel path or CSV filename cell."""
    if value is None:
        return ""
    text = str(value).strip().strip('"').strip("'")
    if not text:
        return ""
    base = os.path.basename(text.replace("\\", "/"))
    stem, ext = os.path.splitext(base)
    if ext.lower() in VESSEL_EXTENSIONS:
        return stem
    return base


def case_id_from_vessel(v_file, dataset_id=None):
    if dataset_id:
        return normalize_dataset_id(dataset_id)
    return normalize_dataset_id(v_file)


def is_template_centerline_dir(path):
    """True when ``path`` is the dropped ``template_centerline`` folder."""
    if not path:
        return False
    base = os.path.basename(os.path.normpath(path))
    if base.lower() == TEMPLATE_CENTERLINE_FOLDER:
        return True
    try:
        from aneux_paths import CLEANDATA_TEMPLATE_CENTERLINE
    except ImportError:
        return False
    if not CLEANDATA_TEMPLATE_CENTERLINE:
        return False
    return os.path.normcase(os.path.abspath(path)) == os.path.normcase(
        os.path.abspath(CLEANDATA_TEMPLATE_CENTERLINE)
    )


def _ostium_frame_parts(frame):
    if isinstance(frame, dict):
        origin = np.asarray(frame["origin"], dtype=np.float64)
        normal = np.asarray(
            frame.get("normal", frame.get("outward")), dtype=np.float64
        )
        radius = float(frame["radius"])
        return origin, normal, radius
    origin, outward, radius = frame[0], frame[1], frame[2]
    return (
        np.asarray(origin, dtype=np.float64),
        np.asarray(outward, dtype=np.float64),
        float(radius),
    )


def as_ostium_frame_dicts(frames):
    """Shared ostium-frame contract: list of origin / unit normal / radius dicts."""
    if frames is None:
        return []
    if isinstance(frames, dict) and "origin" in frames:
        origin = np.asarray(frames["origin"], dtype=np.float64).reshape(-1, 3)
        normal = np.asarray(frames["normal"], dtype=np.float64).reshape(-1, 3)
        radius = np.asarray(frames["radius"], dtype=np.float64).reshape(-1)
        frames = [
            {"origin": origin[i], "normal": normal[i], "radius": float(radius[i])}
            for i in range(origin.shape[0])
        ]
    out = []
    for frame in frames:
        origin, normal, radius = _ostium_frame_parts(frame)
        origin = np.array(origin, dtype=np.float64, copy=True).reshape(3)
        normal = np.array(normal, dtype=np.float64, copy=True).reshape(3)
        nrm = float(np.linalg.norm(normal))
        if nrm < 1e-12:
            raise ValueError("ostium frame normal has zero length")
        out.append(
            {
                "origin": origin,
                "normal": normal / nrm,
                "radius": float(radius),
            }
        )
    return out


def ostium_frames_npz_path(output_dir, dataset_id):
    return os.path.join(output_dir, f"{dataset_id}.ostium_frames.npz")


def save_ostium_frames(path, frames):
    contract = as_ostium_frame_dicts(frames)
    k = len(contract)
    origin = np.empty((k, 3), dtype=np.float64)
    normal = np.empty((k, 3), dtype=np.float64)
    radius = np.empty((k,), dtype=np.float64)
    for i, frame in enumerate(contract):
        origin[i] = frame["origin"]
        normal[i] = frame["normal"]
        radius[i] = frame["radius"]
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    np.savez(path, origin=origin, normal=normal, radius=radius)
    return path, contract


def load_ostium_frames(path):
    with np.load(path) as data:
        origin = np.array(data["origin"], dtype=np.float64, copy=True).reshape(-1, 3)
        normal = np.array(data["normal"], dtype=np.float64, copy=True).reshape(-1, 3)
        radius = np.array(data["radius"], dtype=np.float64, copy=True).reshape(-1)
    if origin.shape[0] != normal.shape[0] or origin.shape[0] != radius.shape[0]:
        raise ValueError(
            f"ostium_frames npz length mismatch: origin={origin.shape} "
            f"normal={normal.shape} radius={radius.shape}"
        )
    return as_ostium_frame_dicts(
        [
            {"origin": origin[i], "normal": normal[i], "radius": float(radius[i])}
            for i in range(origin.shape[0])
        ]
    )


def persist_ostium_frames(output_dir, dataset_id, frames):
    path = ostium_frames_npz_path(output_dir, dataset_id)
    return save_ostium_frames(path, frames)


def _is_under_rawdata(path):
    if not path:
        return False
    norm = os.path.normpath(os.path.abspath(path)).replace("/", os.sep).lower()
    marker = (os.sep + "rawdata" + os.sep).lower()
    return marker in (norm + os.sep)


def _require_output_dirs(out_gt_dir, out_cl_dir, out_tpl_dir):
    dirs = {
        "out_gt_dir": out_gt_dir,
        "out_cl_dir": out_cl_dir,
        "out_tpl_dir": out_tpl_dir,
    }
    missing = [name for name, value in dirs.items() if not value]
    if missing:
        raise ValueError(
            "process_case requires explicit output directories "
            f"({', '.join(missing)} missing); pass scratch paths in tests, "
            "never rely on a silent live cleandata/ default."
        )
    for name, value in dirs.items():
        if _is_under_rawdata(value):
            raise ValueError(f"Refusing {name} under rawdata/: {value}")
        if is_template_centerline_dir(value):
            raise ValueError(
                f"{name}={value!r} is template_centerline, which is dropped "
                "(§15 item 3). Write original_centerline only."
            )


def _unwrap_callable(fn):
    seen = set()
    current = fn
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        wrapped = getattr(current, "__wrapped__", None)
        if wrapped is not None:
            current = wrapped
            continue
        closure = getattr(current, "__closure__", None) or ()
        inner = None
        for cell in closure:
            contents = cell.cell_contents
            if callable(contents) and getattr(contents, "__code__", None) is not getattr(
                current, "__code__", None
            ):
                inner = contents
                break
        if inner is None:
            break
        current = inner
    return current


def _accepted_param_names(fn):
    inner = _unwrap_callable(fn)
    try:
        signature = inspect.signature(inner)
    except (TypeError, ValueError):
        return None
    params = list(signature.parameters.values())
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params):
        return None
    return set(signature.parameters)


def _filter_kwargs(fn, kwargs):
    names = _accepted_param_names(fn)
    cleaned = {key: value for key, value in kwargs.items() if value is not None}
    if names is None:
        return cleaned
    return {key: value for key, value in cleaned.items() if key in names}


def _invoke(fn, **kwargs):
    return fn(**_filter_kwargs(fn, kwargs))


def _quality_error(message, dataset_id=None):
    try:
        from vessel_pipeline import TemplateQualityError
    except ImportError:
        return ProcessCaseError(
            f"{dataset_id}: {message}" if dataset_id else message
        )
    try:
        return TemplateQualityError(message, dataset_id=dataset_id)
    except TypeError:
        return TemplateQualityError(message)


def _unpack_processor_result(result):
    empty = {
        "path": None,
        "frames": None,
        "n_clipped": None,
        "n_profiles": None,
        "n_openings": None,
        "centerline": None,
    }
    if result is None:
        return empty
    if isinstance(result, (str, os.PathLike)):
        empty["path"] = str(result)
        return empty
    if isinstance(result, dict):
        path = (
            result.get("path")
            or result.get("out_file")
            or result.get("gt_path")
            or result.get("tpl_path")
            or result.get("cl_path")
        )
        frames = (
            result.get("frames")
            or result.get("cut_frames")
            or result.get("ostium_frames")
        )
        return {
            "path": str(path) if path else None,
            "frames": frames,
            "n_clipped": result.get("n_clipped"),
            "n_profiles": result.get("n_profiles"),
            "n_openings": result.get("n_openings"),
            "centerline": result.get("centerline") or result.get("branched_centerline"),
        }
    if isinstance(result, (tuple, list)):
        if not result:
            return empty
        first = result[0]
        if isinstance(first, dict):
            return _unpack_processor_result(first)
        empty["path"] = str(first) if first is not None else None
        if len(result) > 1:
            empty["frames"] = result[1]
        if len(result) > 2 and isinstance(result[2], dict):
            extra = _unpack_processor_result(result[2])
            for key, value in extra.items():
                if extra[key] is not None and (key == "path" or empty[key] is None):
                    if key != "path" or extra[key]:
                        empty[key] = extra[key]
        return empty
    empty["path"] = str(result)
    return empty


def _gate_n_clipped(n_clipped, n_profiles, dataset_id, label):
    if n_clipped is None or n_profiles is None:
        return
    if int(n_clipped) != int(n_profiles):
        raise _quality_error(
            f"{label}: n_clipped ({int(n_clipped)}) != n_profiles ({int(n_profiles)})",
            dataset_id=dataset_id,
        )


def _count_openings(path):
    if not path or not os.path.isfile(path):
        return None
    try:
        import pyvista as pv
        from vessel_pipeline import inspect_openings
    except ImportError:
        return None
    try:
        return len(inspect_openings(pv.read(path)))
    except Exception:
        return None


def _gate_openings(path, n_profiles, dataset_id, label):
    if n_profiles is None:
        return
    n_open = _count_openings(path)
    if n_open is None:
        return
    if int(n_open) != int(n_profiles):
        raise _quality_error(
            f"{label}: {int(n_open)} openings against {int(n_profiles)} profiles",
            dataset_id=dataset_id,
        )


def _lazy_gt_processor():
    import remeshing

    return remeshing.process_gt_remesh_dataset


def _lazy_template_processor():
    from vessel_pipeline import process_variable_dataset

    return process_variable_dataset


def _lazy_centerline_processor():
    from centerline_creation import process_centerline_dataset

    return process_centerline_dataset


def _write_centerline_mesh(mesh, out_file):
    from vessel_pipeline import save_polydata

    parent = os.path.dirname(out_file)
    if parent:
        os.makedirs(parent, exist_ok=True)
    save_polydata(mesh, out_file)
    return out_file


def _maybe_shared_centerline(
    v_file,
    *,
    extension_length,
    sample_spacing,
    gt_processor,
):
    """Compute working copy + centerline only when GT remesh can consume it."""
    names = _accepted_param_names(gt_processor)
    if names is None or not (
        "centerline" in names or "reuse_centerline" in names
    ):
        return None, None
    try:
        import pyvista as pv
        import remeshing
        import vessel_pipeline as vp
    except ImportError:
        return None, None
    original = pv.read(v_file)
    gt_surface = remeshing.prepare_gt_surface(original)
    gt_profiles = vp.measure_open_profiles(gt_surface)
    if not hasattr(remeshing, "_gt_centerline"):
        return None, None
    _work, _work_profiles, branched = remeshing._gt_centerline(
        gt_surface,
        extension_length,
        sample_spacing,
        gt_profiles=gt_profiles,
    )
    frames = as_ostium_frame_dicts(vp.opening_clip_frames(branched, gt_profiles))
    return branched, frames


def process_case(
    v_file,
    *,
    out_gt_dir,
    out_cl_dir,
    out_tpl_dir,
    dataset_id=None,
    target_edge_length=DEFAULT_GT_EDGE_LENGTH_MM,
    template_edge_length=DEFAULT_TEMPLATE_EDGE_LENGTH_MM,
    extension_length=DEFAULT_EXTENSION_LENGTH,
    sample_spacing=DEFAULT_SAMPLE_SPACING,
    grid_spacing=None,
    max_grid_size=None,
    speedups=False,
    gt_processor=None,
    template_processor=None,
    centerline_processor=None,
):
    """Run GT remesh, original centerline, and template for one vessel.

    Parameters
    ----------
    v_file:
        Original (or already-cleaned) vessel mesh.
    out_gt_dir, out_cl_dir, out_tpl_dir:
        Required. No default to live ``cleandata/``.
    gt_processor, template_processor, centerline_processor:
        Optional callables for tests. Real runs lazy-import remeshing /
        ``process_variable_dataset`` / ``centerline_creation``. Extra kwargs
        such as ``cut_frames`` are passed only when the callee accepts them.
    """
    _require_output_dirs(out_gt_dir, out_cl_dir, out_tpl_dir)
    if _is_under_rawdata(v_file):
        raise ValueError(f"Refusing input under rawdata/: {v_file}")

    dataset_id = case_id_from_vessel(v_file, dataset_id)
    if not dataset_id:
        raise ValueError("Could not determine dataset_id from v_file")

    os.makedirs(out_gt_dir, exist_ok=True)
    os.makedirs(out_cl_dir, exist_ok=True)
    os.makedirs(out_tpl_dir, exist_ok=True)

    gt_fn = gt_processor or _lazy_gt_processor()
    shared_centerline, precomputed_frames = _maybe_shared_centerline(
        v_file,
        extension_length=extension_length,
        sample_spacing=sample_spacing,
        gt_processor=gt_fn,
    )

    gt_result = _invoke(
        gt_fn,
        dataset_id=dataset_id,
        v_file=v_file,
        output_dir=out_gt_dir,
        target_edge_length=target_edge_length,
        extension_length=extension_length,
        sample_spacing=sample_spacing,
        centerline=shared_centerline,
        reuse_centerline=shared_centerline,
        return_frames=True,
        persist_frames=True,
    )
    gt_unpacked = _unpack_processor_result(gt_result)
    gt_path = gt_unpacked["path"] or os.path.join(out_gt_dir, f"{dataset_id}.vtp")
    frames = as_ostium_frame_dicts(
        gt_unpacked["frames"] if gt_unpacked["frames"] is not None else precomputed_frames
    )
    if not frames:
        sidecar = ostium_frames_npz_path(out_gt_dir, dataset_id)
        if os.path.isfile(sidecar):
            frames = load_ostium_frames(sidecar)

    frames_path = None
    if frames:
        frames_path, frames = persist_ostium_frames(out_gt_dir, dataset_id, frames)

    n_profiles = gt_unpacked["n_profiles"]
    if n_profiles is None and frames:
        n_profiles = len(frames)
    n_clipped_gt = gt_unpacked["n_clipped"]
    if n_clipped_gt is None:
        n_clipped_gt = gt_unpacked["n_openings"]
    _gate_n_clipped(n_clipped_gt, n_profiles, dataset_id, "GT")
    _gate_openings(gt_path, n_profiles, dataset_id, "GT")

    cl_path = os.path.join(out_cl_dir, f"{dataset_id}.vtp")
    centerline_mesh = gt_unpacked["centerline"] or shared_centerline
    if centerline_mesh is not None and not isinstance(centerline_mesh, (str, os.PathLike)):
        _write_centerline_mesh(centerline_mesh, cl_path)
    elif isinstance(centerline_mesh, (str, os.PathLike)) and os.path.isfile(centerline_mesh):
        cl_path = str(centerline_mesh)
    else:
        cl_fn = centerline_processor or _lazy_centerline_processor()
        cl_source = gt_path if gt_path and os.path.isfile(gt_path) else v_file
        cl_result = _invoke(
            cl_fn,
            dataset_id=dataset_id,
            v_file=cl_source,
            output_dir=out_cl_dir,
            extension_length=extension_length,
            sample_spacing=sample_spacing,
        )
        unpacked_cl = _unpack_processor_result(cl_result)
        if unpacked_cl["path"]:
            cl_path = unpacked_cl["path"]

    tpl_fn = template_processor or _lazy_template_processor()
    tpl_source = gt_path if gt_path and os.path.isfile(gt_path) else v_file
    reuse_cl = cl_path if cl_path and os.path.isfile(cl_path) else centerline_mesh
    tpl_result = _invoke(
        tpl_fn,
        dataset_id=dataset_id,
        v_file=tpl_source,
        output_dir=out_tpl_dir,
        target_edge_length=template_edge_length,
        extension_length=extension_length,
        sample_spacing=sample_spacing,
        grid_spacing=grid_spacing,
        max_grid_size=max_grid_size,
        speedups=speedups,
        cut_frames=frames or None,
        ostium_frames=frames or None,
        reuse_centerline=reuse_cl,
        centerline=reuse_cl,
    )
    tpl_unpacked = _unpack_processor_result(tpl_result)
    tpl_path = tpl_unpacked["path"] or os.path.join(out_tpl_dir, f"{dataset_id}.vtp")
    n_clipped_tpl = tpl_unpacked["n_clipped"]
    if n_clipped_tpl is None:
        n_clipped_tpl = tpl_unpacked["n_openings"]
    n_profiles_tpl = tpl_unpacked["n_profiles"] or n_profiles
    _gate_n_clipped(n_clipped_tpl, n_profiles_tpl, dataset_id, "template")
    _gate_openings(tpl_path, n_profiles_tpl, dataset_id, "template")

    return {
        "status": "ok",
        "dataset_id": dataset_id,
        "gt_path": gt_path,
        "cl_path": cl_path,
        "tpl_path": tpl_path,
        "frames_path": frames_path,
        "frames": frames,
        "n_profiles": n_profiles,
        "n_clipped_gt": n_clipped_gt,
        "n_clipped_tpl": n_clipped_tpl,
    }
