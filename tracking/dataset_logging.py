"""Dataset version fingerprinting + MLflow dataset logging.

Called from fusion/rule_based.py and fusion/gbt.py (and future retraining
scripts) inside an active MLflow run, so every run records exactly which
dataset state it trained/evaluated on: the `Data/Dataset/*` version folder
name, a content hash of clip_features.parquet (the actual fusion input),
and a content hash + git commit of the annotation CSVs it was built from.
The hashes exist specifically so a run is fingerprinted correctly even if
someone edits files in place and forgets to bump the version folder name.

Assumes a single active dataset version folder under Data/Dataset/ at a
time (renamed in place when bumped, e.g. v1.0.0 -> v1.1.0), matching this
repo's existing `hri-multimodal-intent-v1.0.0` naming convention. If you
instead want to keep multiple version folders side by side, this needs a
small change (accept an explicit version param instead of auto-discovering
the one folder) -- flag it and we'll adjust before Step 3.
"""
import glob
import os
import subprocess
import warnings

import mlflow
import mlflow.data
import pandas as pd

from tracking.hashing import sha256_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT_GLOB = os.path.join(REPO_ROOT, "Data", "Dataset", "*")
FEATURES_PATH = os.path.join(REPO_ROOT, "data", "features", "clip_features.parquet")
ANNOTATION_FILES = ["clips.csv", "scenarios.csv", "splits.csv"]
TARGET_COLUMN = "intent"


def dataset_version_tag() -> str:
    """The dataset folder name under Data/Dataset/, e.g.
    'hri-multimodal-intent-v1.0.0'. Bump this folder name when the dataset
    changes -- this function just reads whatever's currently there."""
    candidates = sorted(d for d in glob.glob(DATASET_ROOT_GLOB) if os.path.isdir(d))
    if not candidates:
        raise FileNotFoundError(f"No dataset folder found under {DATASET_ROOT_GLOB}")
    if len(candidates) > 1:
        raise RuntimeError(
            f"Expected exactly one dataset version folder under Data/Dataset/, found "
            f"{len(candidates)}: {[os.path.basename(c) for c in candidates]}. "
            "Remove stale versions or point dataset_version_tag() at the one in use."
        )
    return os.path.basename(candidates[0])


def _annotations_dir(dataset_version: str) -> str:
    return os.path.join(REPO_ROOT, "Data", "Dataset", dataset_version, "annotations")


def annotation_fingerprint(dataset_version: str) -> dict:
    """sha256 of each annotation CSV, keyed by filename."""
    ann_dir = _annotations_dir(dataset_version)
    return {name: sha256_file(os.path.join(ann_dir, name)) for name in ANNOTATION_FILES}


def _git(*args):
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def annotations_git_state(dataset_version: str) -> dict:
    """Last commit that touched the annotation files, and whether they
    currently have uncommitted changes relative to that commit."""
    ann_dir = os.path.relpath(_annotations_dir(dataset_version), REPO_ROOT)
    commit = _git("log", "-1", "--format=%H", "--", ann_dir)
    dirty_output = _git("status", "--porcelain", "--", ann_dir)
    return {
        "commit": commit or "unknown",
        "dirty": bool(dirty_output),
    }


def build_dataset_and_tags():
    """Loads clip_features.parquet and builds everything needed to log it:
    the DataFrame itself, an mlflow.data Dataset wrapper (for lineage in
    the UI), and a flat dict of fingerprint tags.

    Returns (df, mlflow_dataset, tags).
    """
    version = dataset_version_tag()
    df = pd.read_parquet(FEATURES_PATH)

    mlflow_dataset = mlflow.data.from_pandas(
        df, source=FEATURES_PATH, targets=TARGET_COLUMN, name=version
    )

    ann_hashes = annotation_fingerprint(version)
    git_state = annotations_git_state(version)

    tags = {
        "dataset_version": version,
        "dataset_features_path": os.path.relpath(FEATURES_PATH, REPO_ROOT),
        "dataset_features_sha256": sha256_file(FEATURES_PATH),
        "dataset_features_digest": mlflow_dataset.digest,
        "dataset_n_clips": str(len(df)),
        "dataset_annotations_commit": git_state["commit"],
        "dataset_annotations_dirty": str(git_state["dirty"]),
    }
    for name, digest in ann_hashes.items():
        key = name.replace(".csv", "")
        tags[f"dataset_annotation_sha256_{key}"] = digest

    return df, mlflow_dataset, tags


class DatasetVersionMismatchError(RuntimeError):
    """Raised when dataset_version and the clip_features.parquet content
    hash are inconsistent with this experiment's run history -- i.e. the
    version folder was swapped without rebuilding the parquet, or the same
    version label was reused for different parquet content."""


def _prior_dataset_tags(experiment_id: str) -> list:
    """[{'dataset_version', 'dataset_features_sha256', 'run_id'}, ...] for
    every past (non-deleted) run in this experiment that has both tags,
    most recent first."""
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["start_time DESC"],
        max_results=1000,
    )
    out = []
    for r in runs:
        version = r.data.tags.get("dataset_version")
        sha256 = r.data.tags.get("dataset_features_sha256")
        if version and sha256:
            out.append({"dataset_version": version, "dataset_features_sha256": sha256, "run_id": r.info.run_id})
    return out


def check_dataset_consistency(dataset_version: str, features_sha256: str, experiment_id: str,
                               allow_override: bool = False) -> None:
    """Guards two failure modes using this experiment's run history:

    1. Stale rebuild: dataset_version changed but clip_features.parquet's
       content hash matches a run logged under a DIFFERENT prior version --
       the Data/Dataset/ folder was swapped but pipeline/build_features.py
       (and the cue runners feeding it) were never rerun.
    2. Version reuse: this SAME dataset_version label was already logged
       with a DIFFERENT parquet content hash -- the version name no longer
       maps to one fixed dataset, breaking reproducibility.

    Raises DatasetVersionMismatchError on either, unless allow_override=True
    (prints a warning instead and lets the run proceed).
    """
    prior = _prior_dataset_tags(experiment_id)
    if not prior:
        return  # nothing logged yet -- e.g. this is the very first baseline run

    stale_match = next(
        (p for p in prior if p["dataset_features_sha256"] == features_sha256
         and p["dataset_version"] != dataset_version),
        None,
    )
    reuse_conflict = next(
        (p for p in prior if p["dataset_version"] == dataset_version
         and p["dataset_features_sha256"] != features_sha256),
        None,
    )
    if stale_match is None and reuse_conflict is None:
        return

    messages = []
    if stale_match is not None:
        messages.append(
            f"dataset_version is now '{dataset_version}', but clip_features.parquet's content hash "
            f"is IDENTICAL to run {stale_match['run_id']} logged under dataset_version="
            f"'{stale_match['dataset_version']}'. This almost always means the Data/Dataset/ folder "
            f"was swapped but pipeline/build_features.py (and the cue runners feeding it) were not "
            f"rerun -- you are about to log this run as '{dataset_version}' while training on old data."
        )
    if reuse_conflict is not None:
        messages.append(
            f"dataset_version '{dataset_version}' was already used in run {reuse_conflict['run_id']} "
            f"with a DIFFERENT clip_features.parquet content hash. Reusing a version label for "
            f"different data breaks reproducibility -- use a new version folder name instead."
        )
    full_message = "\n".join(messages)

    if allow_override:
        warnings.warn(full_message, stacklevel=2)
    else:
        raise DatasetVersionMismatchError(
            full_message + "\n\nRebuild clip_features.parquet from the new dataset before retraining, "
            "or if this is genuinely intentional, call log_dataset(allow_stale_override=True)."
        )


def log_dataset(context: str = "training", allow_stale_override: bool = False) -> pd.DataFrame:
    """Call inside an active mlflow.start_run(). Logs clip_features.parquet
    as an MLflow Dataset input plus fingerprint tags on the current run,
    checks it for staleness/reuse against this experiment's run history,
    and returns the loaded DataFrame for training/eval.
    """
    df, mlflow_dataset, tags = build_dataset_and_tags()
    active_run = mlflow.active_run()
    if active_run is None:
        raise RuntimeError("log_dataset() must be called inside an active mlflow.start_run() block.")

    check_dataset_consistency(
        dataset_version=tags["dataset_version"],
        features_sha256=tags["dataset_features_sha256"],
        experiment_id=active_run.info.experiment_id,
        allow_override=allow_stale_override,
    )

    mlflow.log_input(mlflow_dataset, context=context)
    mlflow.set_tags(tags)
    return df
