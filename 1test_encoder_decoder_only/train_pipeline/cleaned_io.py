"""Stage-2 training data from `cleandata/` only.

Folders (each `{dataset_id}.vtp`):

- uniformly_remeshed: original vessels remeshed finely (GT surface)
- template_mesh: `variable_remeshing.py` parent templates (decoder baseline)
- original_centerline: `centerline_creation.py` on uniformly_remeshed

`template_centerline` is dropped (§15 item 3). Completeness gating does not
require it; parametrisation uses `original_centerline`. Sample records still
expose `template_centerline_file` as an alias of that path so older
`dataset.py` loaders keep working.

Training never reads `rawdata/`. Missing centerlines can be filled with the
VMTK helpers in `datatransform/template_creation/` (`vmtk_env`).
"""
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from aneux_paths import (
    CLEANDATA,
    CLEANDATA_FOLDER_NAMES,
    CLEANDATA_ORIGINAL_CENTERLINE,
    CLEANDATA_TEMPLATE_MESH,
    CLEANDATA_UNIFORM,
    TEMPLATE_DIR,
    ensure_cleandata_layout,
)

SAMPLE_KEYS = (
    "dataset_id",
    "vessel_file",
    "centerline_file",
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
    """Absolute paths for the three training folders under `root`."""
    base = CLEANDATA if root is None else os.path.abspath(root)
    original_cl = os.path.join(base, "original_centerline")
    return {
        "root": base,
        "uniform": os.path.join(base, "uniformly_remeshed"),
        "template_mesh": os.path.join(base, "template_mesh"),
        "original_centerline": original_cl,
        # Alias only: folder dropped; dataset.py still reads this key.
        "template_centerline": original_cl,
    }


def default_layout():
    return {
        "root": CLEANDATA,
        "uniform": CLEANDATA_UNIFORM,
        "template_mesh": CLEANDATA_TEMPLATE_MESH,
        "original_centerline": CLEANDATA_ORIGINAL_CENTERLINE,
        "template_centerline": CLEANDATA_ORIGINAL_CENTERLINE,
    }


def sample_record(dataset_id, layout):
    centerline = vtp_path(layout["original_centerline"], dataset_id)
    return {
        "dataset_id": str(dataset_id),
        "vessel_file": vtp_path(layout["uniform"], dataset_id),
        "centerline_file": centerline,
        "template_mesh_file": vtp_path(layout["template_mesh"], dataset_id),
        "template_centerline_file": centerline,
    }


def _has_file(path):
    return bool(path) and os.path.isfile(path)


def sample_is_complete(sample, require_templates=True):
    needed = ["vessel_file", "centerline_file"]
    if require_templates:
        needed.append("template_mesh_file")
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


def write_centerline(dataset_id, vessel_vtp, output_dir):
    """Run the template-creation Voronoi centerline pipeline on an existing mesh.

    Refuses ``template_centerline`` output dirs (dropped, §15 item 3).
    """
    template_dir = os.path.abspath(TEMPLATE_DIR)
    if template_dir not in sys.path:
        sys.path.insert(0, template_dir)
    try:
        from centerline_creation import process_centerline_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Creating derived cleandata files needs VMTK. "
            "Activate the vmtk_env conda environment and rerun, or write the "
            ".vtp files with datatransform/template_creation/"
            "centerline_creation.py first."
        ) from exc

    os.makedirs(output_dir, exist_ok=True)
    return process_centerline_dataset(
        dataset_id=str(dataset_id),
        v_file=vessel_vtp,
        output_dir=output_dir,
    )


def ensure_sample_derived(sample, overwrite=False):
    """Fill a missing original_centerline in place.

    Sources must already live in cleandata (uniform GT). Never reads rawdata.
    Does not write ``template_centerline``.
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

    return sample


def list_cleandata_samples(
    root=None,
    require_templates=True,
    include_incomplete=False,
):
    """Every `{id}.vtp` already present under `cleandata`.

    The folder is pre-curated; this does not filter by clinical location or CSV.
    A sample is loadable when GT, original centerline, and (when
    ``require_templates``) template mesh exist. ``template_centerline`` is not
    required. Incomplete ids are reported, not dropped as unwanted.
    """
    layout = cleandata_layout(root)
    ensure_cleandata_layout(layout["root"])

    candidate_ids = list_vtp_ids(layout["uniform"])
    if require_templates:
        candidate_ids |= list_vtp_ids(layout["template_mesh"])
    candidate_ids |= list_vtp_ids(layout["original_centerline"])

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
        ("template_mesh", "template_mesh"),
        ("original_centerline", "original_centerline"),
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
]
