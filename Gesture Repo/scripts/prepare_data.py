"""
Stage 0.5 - Build split index CSVs + a data report from extracted landmarks.

Scans LANDMARK_DIR, maps dataset-native classes to the 8 final labels
(src/data.py tables), assigns splits per the guide (§6):

  jester   official train split -> train; validation split halved
           deterministically into val / test (official test is unlabeled)
  ntu      subject-wise 70/15/15 (a subject appears in exactly one split)
  custom   subject-wise; live_test/ folder -> the untouchable live_test split

Writes data/index/{train,val,test,live_test}.csv and reports/data/DATA_REPORT.md.

    python scripts/prepare_data.py
"""
import glob
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import GESTURE_LABELS, INDEX_DIR, LABEL_TO_ID, LANDMARK_DIR, REPORT_DIR
from src.data import resolve_label

OUT_DIR = os.path.join(REPORT_DIR, "data")


def scan_landmarks():
    files = sorted(glob.glob(os.path.join(LANDMARK_DIR, "**", "*.npz"), recursive=True))
    if not files:
        raise SystemExit(f"No .npz landmarks under {LANDMARK_DIR} — "
                         "run scripts/extract_landmarks.py first.")
    rows = []
    for path in tqdm(files, desc="scanning"):
        try:
            with np.load(path) as d:
                rows.append({
                    "path": os.path.relpath(path, LANDMARK_DIR).replace(os.sep, "/"),
                    "dataset": str(d["dataset"]),
                    "source_class": str(d["source_class"]),
                    "subject": str(d["subject"]),
                    "split_hint": str(d["split_hint"]),
                    "n_frames": int(d["n_frames"]),
                })
        except Exception as e:
            print(f"unreadable, skipping: {path} ({e})")
    return pd.DataFrame(rows)


def subject_split(subjects, train=0.70, val=0.15):
    """Deterministic subject-wise assignment by sorted position."""
    subjects = sorted(set(subjects))
    n = len(subjects)
    assign = {}
    for i, s in enumerate(subjects):
        frac = (i + 0.5) / n
        assign[s] = "train" if frac < train else ("val" if frac < train + val else "test")
    return assign


def assign_splits(df):
    split = pd.Series("", index=df.index)

    # jester: official hints; validation halved into val/test by clip-id parity
    jm = df["dataset"] == "jester"
    split[jm & (df["split_hint"] == "train")] = "train"
    val_ids = df.loc[jm & (df["split_hint"] == "val"), "path"].str.extract(r"(\d+)")[0]
    split[val_ids.index] = np.where(val_ids.astype(int) % 2 == 0, "val", "test")

    # ntu: subject-wise
    nm = df["dataset"] == "ntu"
    if nm.any():
        assign = subject_split(df.loc[nm, "subject"])
        split[nm] = df.loc[nm, "subject"].map(assign)

    # custom: live_test folder wins; the rest subject-wise PER LABEL, so a
    # class recorded by few people still gets subject-pure val/test data
    cm = df["dataset"] == "custom"
    live = cm & (df["split_hint"] == "live_test")
    split[live] = "live_test"
    pool = cm & ~live
    for label, part in df.loc[pool].groupby("label"):
        subs = sorted(part["subject"].unique())
        if len(subs) >= 5:
            assign = subject_split(part["subject"])
            split[part.index] = part["subject"].map(assign)
        elif len(subs) >= 2:
            # too few subjects for separate val AND test subjects: hold out
            # the last subject entirely, alternating its clips val/test —
            # eval stays subject-pure w.r.t. training
            print(f"NOTE: custom '{label}': {len(subs)} subjects — "
                  f"'{subs[-1]}' held out for val/test, rest train.")
            eval_idx = part[part["subject"] == subs[-1]].sort_values("path").index
            split[part.index.difference(eval_idx)] = "train"
            split[eval_idx] = ["val" if i % 2 == 0 else "test"
                               for i in range(len(eval_idx))]
        else:
            print(f"WARNING: custom '{label}': single subject — clip-level "
                  "70/15/15 split; metrics will be optimistic. Record more people.")
            idx = part.sort_values("path").index
            frac = (np.arange(len(idx)) + 0.5) / len(idx)
            split[idx] = np.where(frac < 0.70, "train",
                                  np.where(frac < 0.85, "val", "test"))

    return split


def write_report(df, path):
    pivot = df.pivot_table(index="label", columns="split", values="path",
                           aggfunc="count", fill_value=0).reindex(GESTURE_LABELS)
    by_ds = df.pivot_table(index="label", columns="dataset", values="path",
                           aggfunc="count", fill_value=0).reindex(GESTURE_LABELS)
    lines = [
        "# Gesture Model - Data Preparation Report\n",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M} · {len(df)} sequences from "
        f"{df['dataset'].nunique()} datasets\n",
        "## Sequences per class and split\n",
        pivot.to_markdown(), "",
        "## Sequences per class and source dataset\n",
        by_ds.to_markdown(), "",
        "## Notes\n",
        "- Splits: jester = official (validation halved into val/test); "
        "ntu & custom = subject-wise 70/15/15; custom/live_test is never trained on.",
        "- Watch for empty cells in `raise_hand`/`beckoning` rows — those classes "
        "depend on custom recordings (guide §4.3).",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    df = scan_landmarks()
    df["label"] = [resolve_label(d, c) for d, c in zip(df["dataset"], df["source_class"])]
    dropped = df["label"].isna().sum()
    if dropped:
        print(f"Dropping {dropped} sequences with unmapped classes")
        df = df.dropna(subset=["label"]).reset_index(drop=True)
    df["label_id"] = df["label"].map(LABEL_TO_ID)
    df["split"] = assign_splits(df)

    os.makedirs(INDEX_DIR, exist_ok=True)
    cols = ["path", "dataset", "subject", "source_class", "label", "label_id", "n_frames"]
    for split in ("train", "val", "test", "live_test"):
        part = df[df["split"] == split][cols]
        part.to_csv(os.path.join(INDEX_DIR, f"{split}.csv"), index=False)
        print(f"{split:9s}: {len(part):6d} sequences")

    write_report(df, os.path.join(OUT_DIR, "DATA_REPORT.md"))

    missing = [l for l in GESTURE_LABELS
               if (df.loc[df["split"] == "train", "label"] == l).sum() == 0]
    if missing:
        print(f"\nWARNING: no TRAINING data for: {', '.join(missing)} — "
              "training will ignore these classes until data is added.")
    print(f"\nIndex CSVs: {INDEX_DIR}")
    print(f"Report:     {OUT_DIR}/DATA_REPORT.md")
    print("Next: python scripts/compare_models.py")


if __name__ == "__main__":
    main()
