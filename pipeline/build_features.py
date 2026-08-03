"""
Phase 2 orchestration: runs pipeline/aggregate.py's per-clip feature builder
over every clip in splits.csv (the dataset's own pre-curated, already
usable==TRUE-filtered clip list -- see pipeline/dataset_config.py for which
Data/Dataset/* version this targets), joins in the target label and the
train/test split assignment splits.csv already ships, carves an additional
dev split out of the train rows, and writes
data/features/clip_features.parquet.

v2.0.0 schema notes (this script no longer matches v1.0.0's):
  - intent is already a column on clips.csv/splits.csv directly -- no more
    join through scenarios.csv via a scenario_id.split("_")[0] hack.
  - splits.csv IS clips.csv filtered to usable=='TRUE' (verified: exact set
    match) plus the intent/split_design columns -- it's the authoritative,
    already-deduplicated clip list to iterate, not clips.csv itself.
  - splits.csv only ships a 2-way split_design (train/test). See
    assign_dev_split() below for the dev split this pipeline still needs.
  - Real (not synthetic) per-clip cue masking: clips.csv's context_masked/
    emotion_masked/gesture_masked flags (motion is never masked in this
    dataset) mark clips where the ground-truth intent was authored assuming
    the fusion model can't see that cue -- but the raw video still shows a
    real face/background/gesture, so the cue runner emits a real (leaky)
    measurement unless overridden here. See apply_cue_mask().

PROVISIONAL, per explicit instruction -- this feature layout and the
resulting parquet are a moving baseline, not yet frozen.

Needs pandas + pyarrow (see .venvs/pipeline) -- unlike Phase 0's aggregation
scripts, Phase 2+ has no stdlib-only constraint (that hard contract was
specific to pipeline/aggregate_clip_cues.py / agreement_report.py).
"""
import csv
import os
import random
import sys
from collections import defaultdict

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline"))
from aggregate import build_clip_feature_row, load_frame_cues_by_clip  # noqa: E402
from dataset_config import DATASET_ROOT  # noqa: E402

CLIPS_CSV = os.path.join(DATASET_ROOT, "annotations", "clips.csv")
SPLITS_CSV = os.path.join(DATASET_ROOT, "annotations", "splits.csv")
FEATURES_DIR = os.path.join(REPO_ROOT, "data", "features")
OUT_PATH = os.path.join(FEATURES_DIR, "clip_features.parquet")

# Cues clips.csv can flag as deliberately masked for a given clip. Motion has
# no *_masked column in this dataset (verified: always FALSE) -- never masked.
MASKABLE_CUES = ["context", "emotion", "gesture"]

DEV_FRACTION = 0.20  # docx: "hold out 10 of the 50 videos per TRAIN scenario"
                     # (~20%); actual per-scenario video counts range 18-70,
                     # not a fixed 50, so this uses the ratio, not a fixed 10.
RANDOM_SEED = 42     # matches pipeline/build_splits.py's existing convention


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def apply_cue_mask(feat_row: dict, clip_row: dict) -> None:
    """Overrides a cue's feature block to the standard "missing" shape
    (all value columns zeroed, missing_{cue}=1) whenever clips.csv flags that
    cue as masked for this clip -- discarding whatever the cue runner actually
    measured from the still-visible video. Same zeroing shape as
    fusion/gbt.py's apply_modality_dropout(), applied here deterministically
    from real annotation instead of a random draw, so both fusion/rule_based.py
    and fusion/gbt.py see the corrected vector automatically.
    """
    for cue in MASKABLE_CUES:
        if clip_row.get(f"{cue}_masked") == "TRUE":
            for col in list(feat_row.keys()):
                if col.startswith(f"{cue}_"):
                    feat_row[col] = 0.0
            feat_row[f"missing_{cue}"] = 1.0


def assign_dev_split(split_rows: list, clips_by_id: dict, seed: int = RANDOM_SEED) -> dict:
    """Carves a dev split out of split_design=='train' rows, grouped by
    (scenario_dir, person_id, take_index) -- the "one video" unit. take_index
    lives on clips.csv, not splits.csv, hence the clips_by_id lookup. A single
    video can yield multiple clips (verified: up to 3 for this dataset), so
    grouping -- not row-level splitting -- is required to keep near-duplicate
    clips from the same video off opposite sides of train/dev. ~20% of each
    scenario's video groups become dev, via a seeded RNG for reproducibility.

    Returns {clip_id: "train"/"dev"} for train rows only (test rows aren't
    touched -- callers should fall back to the shipped split_design for those).
    """
    rng = random.Random(seed)
    train_rows = [r for r in split_rows if r["split_design"] == "train"]

    groups_by_scenario = defaultdict(lambda: defaultdict(list))
    for r in train_rows:
        take_index = clips_by_id[r["clip_id"]]["take_index"]
        groups_by_scenario[r["scenario_dir"]][(r["person_id"], take_index)].append(r["clip_id"])

    assignment = {}
    for scenario_dir, groups in groups_by_scenario.items():
        group_keys = list(groups.keys())
        rng.shuffle(group_keys)
        n_dev = round(DEV_FRACTION * len(group_keys))
        dev_keys = set(group_keys[:n_dev])
        for key, clip_ids in groups.items():
            split = "dev" if key in dev_keys else "train"
            for clip_id in clip_ids:
                assignment[clip_id] = split

    # Leakage check, mirroring pipeline/build_splits.py's verify_no_leakage().
    for scenario_dir, groups in groups_by_scenario.items():
        for key, clip_ids in groups.items():
            splits_seen = {assignment[c] for c in clip_ids}
            assert len(splits_seen) == 1, f"LEAKAGE: group {scenario_dir}/{key} spans {splits_seen}"

    return assignment


def main():
    clips_by_id = {r["clip_id"]: r for r in read_csv(CLIPS_CSV)}
    split_rows = read_csv(SPLITS_CSV)
    print(f"[build_features] {len(split_rows)} clips in splits.csv (usable==TRUE already)")

    dev_assignment = assign_dev_split(split_rows, clips_by_id)
    n_train = sum(1 for v in dev_assignment.values() if v == "train")
    n_dev = sum(1 for v in dev_assignment.values() if v == "dev")
    print(f"[build_features] dev split carved from train: {n_train} train, {n_dev} dev "
          f"({n_dev / (n_train + n_dev):.1%})")

    print("[build_features] loading per-frame cues (this reads all 4 *_frame_cues.jsonl files)...")
    frames_by_cue = {cue: load_frame_cues_by_clip(cue) for cue in ["emotion", "gesture", "motion", "context"]}

    rows = []
    skipped_no_clip_row = 0
    n_masked_clips = 0
    for split_row in split_rows:
        clip_id = split_row["clip_id"]
        clip_row = clips_by_id.get(clip_id)
        if clip_row is None:
            skipped_no_clip_row += 1
            continue

        feat_row = build_clip_feature_row(
            clip_id,
            frames_by_cue["emotion"].get(clip_id, []),
            frames_by_cue["gesture"].get(clip_id, []),
            frames_by_cue["motion"].get(clip_id, []),
            frames_by_cue["context"].get(clip_id, []),
        )
        was_masked = any(clip_row.get(f"{cue}_masked") == "TRUE" for cue in MASKABLE_CUES)
        apply_cue_mask(feat_row, clip_row)
        n_masked_clips += was_masked

        feat_row["scenario_dir"] = split_row["scenario_dir"]
        feat_row["person_id"] = split_row["person_id"]
        feat_row["intent"] = split_row["intent"]
        feat_row["split_design"] = split_row["split_design"]  # as shipped: train/test
        feat_row["split_design_v2"] = dev_assignment.get(clip_id, split_row["split_design"])  # train/dev/test
        feat_row["scoreable"] = clip_row["scoreable"]
        rows.append(feat_row)

    if skipped_no_clip_row:
        print(f"[build_features] WARNING: {skipped_no_clip_row} clips in splits.csv had no clips.csv row")
    print(f"[build_features] {n_masked_clips} clips had >=1 cue masked (context/emotion/gesture only)")

    df = pd.DataFrame(rows)
    assert df.isna().sum().sum() == 0, "NaNs present in feature matrix -- aggregation bug, see handover Phase 2 failure points"

    os.makedirs(FEATURES_DIR, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"[build_features] wrote {len(df)} clips x {len(df.columns)} columns -> {OUT_PATH}")
    print(f"[build_features] intent label distribution:\n{df['intent'].value_counts()}")
    print(f"[build_features] split_design_v2 distribution:\n{df['split_design_v2'].value_counts()}")


if __name__ == "__main__":
    main()
