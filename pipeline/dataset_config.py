"""Single source of truth for which Data/Dataset/* version the batch
pipeline scripts (build_features.py, build_splits.py, agreement_report.py)
target, so bumping to a new dataset version means editing one constant
here instead of each script's own copy.

tracking/dataset_logging.py's dataset_version_tag() intentionally stays a
separate, auto-discovering implementation -- it fingerprints whichever
folder is present *after* the fact (for MLflow tagging), whereas this
constant is what the pipeline scripts need *before* any parquet/run exists.
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACTIVE_DATASET_VERSION = "hri-multimodal-intent-v2.0.0"
DATASET_ROOT = os.path.join(REPO_ROOT, "Data", "Dataset", ACTIVE_DATASET_VERSION)
