"""Stage-2 training data from `cleandata/` only.

The five folders are filled offline with:

- uniformly_remeshed: `uniform_remeshing.py` on the original vessels (GT surface)
- coarse_remeshed: coarser uniform remesh of those GT surfaces
- template_mesh: `variable_remeshing.py` parent templates
- original_centerline: `centerline_creation.py` on uniformly_remeshed
- template_centerline: `centerline_creation.py` on template_mesh

Each folder contains `{dataset_id}.vtp` files. Training never reads `rawdata/`.

If a derived centerline or coarse remesh is missing at cache time, this module
calls the same VMTK helpers as `datatransform/template_creation/` (needs
`vmtk_env`). Loading already-written VTPs does not import VMTK.
"""
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from aneux_paths import (
    CLEANDATA,
    CLEANDATA_COARSE,
    CLEANDATA_FOLDER_NAMES,
    CLEANDATA_ORIGINAL_CENTERLINE,
    CLEANDATA_TEMPLATE_CENTERLINE,
    CLEANDATA_TEMPLATE_MESH,
    CLEANDATA_UNIFORM,
    TEMPLATE_DIR,
    ensure_cleandata_layout,
)

# Coarser isotropic remesh of uniformly_remeshed (mm). Matches the Stage-2
# review's ~1.5 mm coarse hierarchy, not a second pass of build_parent_tube.
COARSE_TARGET_EDGE_MM = 1.5
SAMPLE_KEYS = (
    "dataset_id",
    "vessel_file",
    "centerline_file",
    "coarse_file",
    "template_mesh_file",
    "template_centerline_file",
)


def _is_under_rawdata(path):
    if not path:
        return False
    norm = os.path.normpath(os.path.abspath(path)).replace("/", os.sep).lower()
    marker = (os.sep + "rawdata" + os.sep).lower()
    return marker in (norm + os.sep)


def vtp_stem(filename):
    name = os.path.basename(filename)
    if name.lower().endswith(".vtp"):
        return name[:-4]
    return os.path.splitext(name)[0]


def list_vtp_ids(folder):
    """Dataset ids for `{id}.vtp` files in `folder` (missing folder → empty set)."""
    if not folder or not os.path.isdir(folder):
        return set()
    ids = set()
    for name in os.listdir(folder):
        if name.startswith(".") or not name.lower().endswith(".vtp"):
            continue
        ids.add(vtp_stem(name))
    return ids


def vtp_path(folder, dataset_id):
    return os.path.join(folder, f"{dataset_id}.vtp")


def cleandata_layout(root=None):
    """Absolute paths for the five training folders under `root`."""
    base = CLEANDATA if root is None else os.path.abspath(root)
    return {
        "root": base,
        "uniform": os.path.join(base, "uniformly_remeshed"),
        "coarse": os.path.join(base, "coarse_remeshed"),
        "template_mesh": os.path.join(base, "template_mesh"),
        "original_centerline": os.path.join(base, "original_centerline"),
        "template_centerline": os.path.join(base, "template_centerline"),
    }


def default_layout():
    return {
        "root": CLEANDATA,
        "uniform": CLEANDATA_UNIFORM,
        "coarse": CLEANDATA_COARSE,
        "template_mesh": CLEANDATA_TEMPLATE_MESH,
        "original_centerline": CLEANDATA_ORIGINAL_CENTERLINE,
        "template_centerline": CLEANDATA_TEMPLATE_CENTERLINE,
    }


def sample_record(dataset_id, layout):
    return {
        "dataset_id": str(dataset_id),
        "vessel_file": vtp_path(layout["uniform"], dataset_id),
        "centerline_file": vtp_path(layout["original_centerline"], dataset_id),
        "coarse_file": vtp_path(layout["coarse"], dataset_id),
        "template_mesh_file": vtp_path(layout["template_mesh"], dataset_id),
        "template_centerline_file": vtp_path(layout["template_centerline"], dataset_id),
    }


def _has_file(path):
    return bool(path) and os.path.isfile(path)


def sample_is_complete(sample, require_templates=True):
    needed = ["vessel_file", "centerline_file", "coarse_file"]
    if require_templates:
        needed.extend(["template_mesh_file", "template_centerline_file"])
    return all(_has_file(sample.get(key)) for key in needed)


def sample_has_sources(sample, require_templates=True):
    if not _has_file(sample.get("vessel_file")):
        return False
    if require_templates and not _has_file(sample.get("template_mesh_file")):
        return False
    return True


def reject_rawdata_paths(sample):
    for key, path in sample.items():
        if not isinstance(path, str):
            continue
        if _is_under_rawdata(path):
            raise ValueError(
                f"Training data path {key}={path!r} is under rawdata/; "
                "Stage 2 may only read cleandata/."
            )


def _vessel_pipeline():
    """Lazy import: centerline/remesh helpers require the vmtk conda env."""
    template_dir = os.path.abspath(TEMPLATE_DIR)
    if template_dir not in sys.path:
        sys.path.insert(0, template_dir)
    try:
        import vessel_pipeline as vp
    except ImportError as exc:
        raise RuntimeError(
            "Creating derived cleandata files needs VMTK. "
            "Activate the vmtk_env conda environment and rerun, or write the "
            ".vtp files with datatransform/template_creation/"
            "centerline_creation.py and uniform_remeshing.py first."
        ) from exc
    return vp


def write_centerline(dataset_id, vessel_vtp, output_dir):
    """Run the template-creation Voronoi centerline pipeline on an existing mesh."""
    vp = _vessel_pipeline()
    os.makedirs(output_dir, exist_ok=True)
    return vp.process_centerline_dataset(
        dataset_id=str(dataset_id),
        v_file=vessel_vtp,
        output_dir=output_dir,
    )


def write_coarse_from_uniform(dataset_id, uniform_vtp, output_dir, target_edge_length=COARSE_TARGET_EDGE_MM):
    """Coarser isotropic remesh of a uniformly_remeshed surface (not from rawdata)."""
    import pyvista as pv

    vp = _vessel_pipeline()
    os.makedirs(output_dir, exist_ok=True)
    mesh = pv.read(uniform_vtp)
    remeshed = vp.remesh_surface_isotropically(
        mesh, target_edge_length=float(target_edge_length)
    )
    final_surface, _n_regions = vp.finalize_surface(remeshed)
    out_file = vtp_path(output_dir, dataset_id)
    vp.save_polydata(final_surface, out_file)
    return out_file


def ensure_sample_derived(sample, overwrite=False):
    """Fill missing original/template centerlines and coarse remesh in place.

    Sources must already live in cleandata (uniform GT, optional template mesh).
    Never reads rawdata.
    """
    reject_rawdata_paths(sample)
    sample = dict(sample)
    dataset_id = sample["dataset_id"]

    if not _has_file(sample["vessel_file"]):
        raise FileNotFoundError(
            f"{dataset_id}: uniformly_remeshed file missing ({sample['vessel_file']})"
        )

    cl_path = sample["centerline_file"]
    if overwrite or not _has_file(cl_path):
        write_centerline(dataset_id, sample["vessel_file"], os.path.dirname(cl_path))

    coarse_path = sample.get("coarse_file")
    if coarse_path and (overwrite or not _has_file(coarse_path)):
        write_coarse_from_uniform(
            dataset_id, sample["vessel_file"], os.path.dirname(coarse_path)
        )

    tpl = sample.get("template_mesh_file")
    tpl_cl = sample.get("template_centerline_file")
    if _has_file(tpl) and tpl_cl and (overwrite or not _has_file(tpl_cl)):
        write_centerline(dataset_id, tpl, os.path.dirname(tpl_cl))

    return sample


def list_cleandata_samples(
    root=None,
    require_templates=True,
    include_incomplete=False,
):
    """Every `{id}.vtp` already present under `cleandata`.

    The folder is pre-curated; this does not filter by clinical location or CSV.
    A sample is loadable when its required meshes exist (all five folders by
    default). Incomplete ids are reported, not silently discarded as 'unwanted'.
    """
    layout = cleandata_layout(root)
    ensure_cleandata_layout(layout["root"])

    candidate_ids = list_vtp_ids(layout["uniform"])
    if require_templates:
        candidate_ids |= list_vtp_ids(layout["template_mesh"])
    candidate_ids |= list_vtp_ids(layout["original_centerline"])
    candidate_ids |= list_vtp_ids(layout["template_centerline"])
    candidate_ids |= list_vtp_ids(layout["coarse"])

    samples = []
    incomplete = []
    for dataset_id in sorted(candidate_ids):
        rec = sample_record(dataset_id, layout)
        reject_rawdata_paths(rec)
        if sample_is_complete(rec, require_templates=require_templates):
            samples.append(rec)
        else:
            incomplete.append(rec)

    if include_incomplete:
        return samples, incomplete
    return samples


def summarize_cleandata(root=None):
    layout = cleandata_layout(root)
    counts = {name: len(list_vtp_ids(layout[key])) for key, name in (
        ("uniform", "uniformly_remeshed"),
        ("coarse", "coarse_remeshed"),
        ("template_mesh", "template_mesh"),
        ("original_centerline", "original_centerline"),
        ("template_centerline", "template_centerline"),
    )}
    return layout, counts


def assert_not_rawdata_dir(*paths):
    for path in paths:
        if _is_under_rawdata(path):
            raise ValueError(
                f"Refusing training path under rawdata/: {path}"
            )


# Re-export folder names so callers do not hard-code the typo-prone layout.
FOLDER_NAMES = CLEANDATA_FOLDER_NAMES
__all__ = [
    "COARSE_TARGET_EDGE_MM",
    "FOLDER_NAMES",
    "SAMPLE_KEYS",
    "assert_not_rawdata_dir",
    "cleandata_layout",
    "default_layout",
    "ensure_sample_derived",
    "list_cleandata_samples",
    "list_vtp_ids",
    "reject_rawdata_paths",
    "sample_has_sources",
    "sample_is_complete",
    "sample_record",
    "summarize_cleandata",
    "vtp_path",
    "vtp_stem",
    "write_centerline",
    "write_coarse_from_uniform",
]
