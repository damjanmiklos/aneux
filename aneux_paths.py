"""Canonical paths for the aneux repo on this machine.

Resolves against this file's location so scripts do not depend on cwd,
OneDrive copies, or the Linux nested `rawdata/rawdata` layout.
`rawdata/` is read-only input.
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
CENTERLINES = os.path.join(RAWDATA, "models-v1.0", "models", "centerlines")

DATATRANSFORM = os.path.join(REPO_ROOT, "datatransform")
LABEL_DIR = os.path.join(DATATRANSFORM, "label")
LABEL_OUTPUT_DIR = os.path.join(LABEL_DIR, "output")
HASCAP_CSV = os.path.join(LABEL_DIR, "hascap.csv")
HASEXTENSION_CSV = os.path.join(LABEL_DIR, "hasextension.csv")
KEPT_VESSELS_ZIP = os.path.join(LABEL_DIR, "kept_vessels.zip")
CLEANED_VESSELS = os.path.join(DATATRANSFORM, "cleaned_data", "vessels_cleaned_and_decapped")

TEMPLATE_DIR = os.path.join(DATATRANSFORM, "template_creation")
TEMPLATE_OUTPUT = os.path.join(TEMPLATE_DIR, "output")
TEMPLATE_OUTPUT_REMESHED = os.path.join(TEMPLATE_DIR, "output_remeshed")
TEMPLATE_OUTPUT_VARIABLE = os.path.join(TEMPLATE_DIR, "output_variable_remeshed")

EXPERIMENT_DIR = os.path.join(REPO_ROOT, "1test_encoder_decoder_only")
EXPERIMENT_OUTPUT = os.path.join(EXPERIMENT_DIR, "output")
EXPERIMENT_CACHE = os.path.join(EXPERIMENT_DIR, "tube_cache")
EXTRA_CENTERLINES = os.path.join(EXPERIMENT_DIR, "centerlines")
