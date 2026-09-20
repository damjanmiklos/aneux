"""Canonical paths for the aneux repo on this machine.

Resolves against this file's location so scripts do not depend on cwd,
OneDrive copies, or the Linux nested `rawdata/rawdata` layout.
`rawdata/` is read-only input. Stage-2 training reads only `cleandata/`.
"""
import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _rawdata_root():
    local = os.path.join(REPO_ROOT, "rawdata")
    nested = os.path.join(local, "rawdata")
    if os.path.isdir(os.path.join(local, "data-v1.0")):
        return local
    if os.path.isdir(os.path.join(nested, "data-v1.0")):
        return nested
    return local


RAWDATA = _rawdata_root()

CSV_PATH = os.path.join(RAWDATA, "data-v1.0", "data", "clinical.csv")
VESSELS_ORIGINAL = os.path.join(RAWDATA, "models-v1.0", "models", "vessels", "original")
VESSELS_AREA001 = os.path.join(RAWDATA, "models-v1.0", "models", "vessels", "remeshed", "area-001")
VESSELS_AREA005 = os.path.join(RAWDATA, "models-v1.0", "models", "vessels", "remeshed", "area-005")
# AneuX-provided centerlines. Read-only archive; do not use in pipelines.
CENTERLINES = os.path.join(RAWDATA, "models-v1.0", "models", "centerlines")

DATATRANSFORM = os.path.join(REPO_ROOT, "datatransform")
LABEL_DIR = os.path.join(DATATRANSFORM, "label")
LABEL_OUTPUT_DIR = os.path.join(LABEL_DIR, "output")
HASCAP_CSV = os.path.join(LABEL_DIR, "hascap.csv")
HASEXTENSION_CSV = os.path.join(LABEL_DIR, "hasextension.csv")
HASCAPOREXTENSION_CSV = os.path.join(LABEL_DIR, "hascaporextension.csv")
KEPT_VESSELS_ZIP = os.path.join(LABEL_DIR, "kept_vessels.zip")
CLEANED_DATA = os.path.join(DATATRANSFORM, "cleaned_data")
CLEANED_VESSELS = os.path.join(CLEANED_DATA, "vessels_cleaned_and_decapped")
TOTAL_CLEAN_ORIGINAL_MESH = os.path.join(CLEANED_DATA, "total_clean_original_mesh")
CLEAN_UNIFORM_MESH = os.path.join(CLEANED_DATA, "clean_uniform_mesh")
CLEAN_CENTERLINE = os.path.join(CLEANED_DATA, "clean_centerline")
UNCAPPED_VESSELS = os.path.join(DATATRANSFORM, "uncapped")

TEMPLATE_DIR = os.path.join(DATATRANSFORM, "template_creation")
TEMPLATE_OUTPUT = os.path.join(TEMPLATE_DIR, "output")
TEMPLATE_OUTPUT_REMESHED = os.path.join(TEMPLATE_DIR, "output_remeshed")
TEMPLATE_OUTPUT_VARIABLE = os.path.join(TEMPLATE_DIR, "output_variable_remeshed")

EXPERIMENT_DIR = os.path.join(REPO_ROOT, "1test_encoder_decoder_only")
EXPERIMENT_OUTPUT = os.path.join(EXPERIMENT_DIR, "output")
EXPERIMENT_CACHE = os.path.join(EXPERIMENT_DIR, "tube_cache")
EXTRA_CENTERLINES = os.path.join(EXPERIMENT_DIR, "centerlines")

# Training-ready meshes. Each subfolder holds only `{dataset_id}.vtp` files.
# uniformly_remeshed: original vessels remeshed finely (GT surface).
# template_mesh: parent templates from variable_remeshing.py (decoder baseline).
# original_centerline: centerline_creation.py on uniformly_remeshed.
# template_centerline was dropped (§15 item 3 / STAGE2 §2.5): same curve as
# original_centerline; training parametrises from original_centerline.
CLEANDATA = os.path.join(REPO_ROOT, "cleandata")
CLEANDATA_UNIFORM = os.path.join(CLEANDATA, "uniformly_remeshed")
CLEANDATA_TEMPLATE_MESH = os.path.join(CLEANDATA, "template_mesh")
CLEANDATA_ORIGINAL_CENTERLINE = os.path.join(CLEANDATA, "original_centerline")
# Deprecated alias only. Not a training folder; not created by ensure_cleandata_layout.
CLEANDATA_TEMPLATE_CENTERLINE = os.path.join(CLEANDATA, "template_centerline")
CLEANDATA_SUBDIRS = (
    CLEANDATA_UNIFORM,
    CLEANDATA_TEMPLATE_MESH,
    CLEANDATA_ORIGINAL_CENTERLINE,
)
CLEANDATA_FOLDER_NAMES = (
    "uniformly_remeshed",
    "template_mesh",
    "original_centerline",
)


def ensure_cleandata_layout(root=None):
    """Create the three training data folders if they are missing.

    Does not create ``template_centerline`` (dropped, §15 item 3). Existing
    files in a live copy of that folder are left untouched.
    """
    base = CLEANDATA if root is None else os.path.abspath(root)
    os.makedirs(base, exist_ok=True)
    folders = [os.path.join(base, name) for name in CLEANDATA_FOLDER_NAMES]
    for path in folders:
        os.makedirs(path, exist_ok=True)
    return base
