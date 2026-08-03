"""
Stage 0 - Offline MediaPipe landmark extraction over the raw datasets.

Runs MediaPipe Holistic over every needed clip in Jester / NTU RGB+D /
custom recordings and stores raw landmarks as one .npz per clip under
LANDMARK_DIR. This is the expensive, CPU-bound step; it shards trivially:

    python scripts/extract_landmarks.py --dataset all
    python scripts/extract_landmarks.py --dataset jester --shard 3/16   # HPC array job

Existing outputs are skipped (safe to re-run / resume). Dataset selection
(which classes, negative caps) follows the mapping tables in src/data.py.

Output .npz per clip:
    pose [T,33,4] (x,y,z,vis) · left_hand [T,21,3] · right_hand [T,21,3]
    (NaN where absent) + meta: dataset, source_class, subject, split_hint,
    fps, source_file, n_frames
"""
import argparse
import csv
import glob
import os
import re
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2 as cv
import numpy as np
from tqdm import tqdm

from config import CUSTOM_DIR, JESTER_DIR, LANDMARK_DIR, NTU_DIR
from src.data import (
    JESTER_MAP,
    JESTER_NEGATIVE_CAP,
    NTU_MAP,
    NTU_NEGATIVE_CAP,
    NTU_NEGATIVES,
)

_NTU_NAME = re.compile(r"S(\d+)C(\d+)P(\d+)R(\d+)A(\d+)", re.IGNORECASE)


# ── clip listing ─────────────────────────────────────────────────────────
def _jester_frames_dir():
    for cand in ("20bn-jester-v1", "20bn-jester-v1-frames", "frames", "."):
        d = os.path.join(JESTER_DIR, cand)
        if os.path.isdir(d) and any(os.scandir(d)):
            return d
    raise FileNotFoundError(f"No Jester frame folders found under {JESTER_DIR}")


def _jester_annotation_files():
    hits = sorted(
        glob.glob(os.path.join(JESTER_DIR, "**", "*train*.csv"), recursive=True)
        + glob.glob(os.path.join(JESTER_DIR, "**", "*validation*.csv"), recursive=True))
    if not hits:
        raise FileNotFoundError(f"No Jester train/validation CSVs under {JESTER_DIR}")
    return hits


def list_jester():
    """All clips of mapped classes + a deterministic cap of negatives per class."""
    frames_dir = _jester_frames_dir()
    rows = []
    for csv_path in _jester_annotation_files():
        split_hint = "val" if "validation" in os.path.basename(csv_path).lower() else "train"
        with open(csv_path, newline="", encoding="utf-8") as f:
            for line in csv.reader(f, delimiter=";"):
                if len(line) < 2:
                    continue
                rows.append((line[0].strip(), line[1].strip(), split_hint))

    by_class = {}
    for clip_id, label, split_hint in rows:
        by_class.setdefault(label, []).append((clip_id, split_hint))

    clips = []
    for label, items in by_class.items():
        if label not in JESTER_MAP:  # hard negatives -> idle, capped deterministically
            items = sorted(items, key=lambda t: int(t[0]))[:JESTER_NEGATIVE_CAP]
        for clip_id, split_hint in items:
            clips.append({
                "dataset": "jester", "clip_id": f"jester_{clip_id}",
                "path": os.path.join(frames_dir, clip_id), "kind": "frames",
                "source_class": label, "subject": f"jester_{clip_id}",
                "split_hint": split_hint, "fps": 12.0,
            })
    return clips


def list_ntu():
    """Mapped actions in full + capped daily-action negatives (guide §4.2)."""
    videos = sorted(glob.glob(os.path.join(NTU_DIR, "**", "*_rgb.avi"), recursive=True))
    if not videos:
        raise FileNotFoundError(f"No *_rgb.avi files under {NTU_DIR}")
    clips, neg_counts = [], {}
    for path in videos:
        m = _NTU_NAME.search(os.path.basename(path))
        if not m:
            continue
        subject, action = int(m.group(3)), int(m.group(5))
        if action in NTU_MAP:
            pass
        elif action in NTU_NEGATIVES:
            if neg_counts.get(action, 0) >= NTU_NEGATIVE_CAP:
                continue
            neg_counts[action] = neg_counts.get(action, 0) + 1
        else:
            continue
        clips.append({
            "dataset": "ntu", "clip_id": os.path.splitext(os.path.basename(path))[0],
            "path": path, "kind": "video", "source_class": str(action),
            "subject": f"P{subject:03d}", "split_hint": "", "fps": None,
        })
    return clips


def list_custom():
    """custom/train/<label>/ and custom/live_test/<label>/; subject = name
    before '__', falling back to the first '_' token (S01_F04_c001 -> S01)."""
    clips = []
    for sub, split_hint in (("train", ""), ("live_test", "live_test")):
        base = os.path.join(CUSTOM_DIR, sub)
        if not os.path.isdir(base):
            continue
        for label in sorted(os.listdir(base)):
            label_dir = os.path.join(base, label)
            if not os.path.isdir(label_dir):
                continue
            for fname in sorted(os.listdir(label_dir)):
                if os.path.splitext(fname)[1].lower() not in (".mp4", ".avi", ".mov", ".mkv"):
                    continue
                stem = os.path.splitext(fname)[0]
                subject = stem.split("__")[0] if "__" in stem else stem.split("_")[0]
                clips.append({
                    "dataset": "custom", "clip_id": f"{sub}_{label}_{stem}",
                    "path": os.path.join(label_dir, fname), "kind": "video",
                    "source_class": label, "subject": subject,
                    "split_hint": split_hint, "fps": None,
                })
    return clips


# ── frame readers ────────────────────────────────────────────────────────
def iter_frames(clip):
    if clip["kind"] == "frames":
        for fpath in sorted(glob.glob(os.path.join(clip["path"], "*.jpg"))):
            frame = cv.imread(fpath)
            if frame is not None:
                yield frame
    else:
        cap = cv.VideoCapture(clip["path"])
        if clip["fps"] is None:
            fps = cap.get(cv.CAP_PROP_FPS)
            clip["fps"] = float(fps) if fps and fps > 0 else 30.0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
        finally:
            cap.release()


# ── extraction ───────────────────────────────────────────────────────────
def extract_clip(clip, holistic_cls, out_path):
    poses, lhs, rhs = [], [], []
    # fresh Holistic per clip: its temporal tracking state must not leak
    # between unrelated videos
    with holistic_cls(model_complexity=1, min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as holistic:
        for frame in iter_frames(clip):
            res = holistic.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

            pose = np.full((33, 4), np.nan, dtype=np.float32)
            if res.pose_landmarks is not None:
                for i, lm in enumerate(res.pose_landmarks.landmark):
                    pose[i] = [lm.x, lm.y, lm.z, lm.visibility]
            poses.append(pose)

            for out, lms in ((lhs, res.left_hand_landmarks),
                             (rhs, res.right_hand_landmarks)):
                hand = np.full((21, 3), np.nan, dtype=np.float32)
                if lms is not None:
                    for i, lm in enumerate(lms.landmark):
                        hand[i] = [lm.x, lm.y, lm.z]
                out.append(hand)

    if len(poses) < 4:
        raise ValueError(f"only {len(poses)} readable frames")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        pose=np.stack(poses), left_hand=np.stack(lhs), right_hand=np.stack(rhs),
        dataset=np.str_(clip["dataset"]), source_class=np.str_(clip["source_class"]),
        subject=np.str_(clip["subject"]), split_hint=np.str_(clip["split_hint"]),
        fps=np.float32(clip["fps"] or 30.0), source_file=np.str_(clip["path"]),
        n_frames=np.int32(len(poses)),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="all",
                    choices=["all", "jester", "ntu", "custom"])
    ap.add_argument("--shard", default="0/1",
                    help="i/N — process every Nth clip starting at i (HPC array jobs)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    shard_i, shard_n = (int(x) for x in args.shard.split("/"))

    listers = {"jester": list_jester, "ntu": list_ntu, "custom": list_custom}
    names = list(listers) if args.dataset == "all" else [args.dataset]
    clips = []
    for name in names:
        try:
            found = listers[name]()
            print(f"{name}: {len(found)} clips selected")
            clips.extend(found)
        except FileNotFoundError as e:
            print(f"{name}: skipped ({e})")
    clips.sort(key=lambda c: c["clip_id"])
    clips = [c for i, c in enumerate(clips) if i % shard_n == shard_i]
    print(f"Shard {shard_i}/{shard_n}: {len(clips)} clips")

    import mediapipe as mp  # deferred: listing shouldn't need mediapipe
    holistic_cls = mp.solutions.holistic.Holistic

    done = skipped = failed = 0
    for clip in tqdm(clips, desc="extracting"):
        out_path = os.path.join(LANDMARK_DIR, clip["dataset"], clip["clip_id"] + ".npz")
        if not args.overwrite and os.path.exists(out_path):
            skipped += 1
            continue
        try:
            extract_clip(clip, holistic_cls, out_path)
            done += 1
        except Exception as e:
            failed += 1
            print(f"\nFAILED {clip['clip_id']}: {e}")
            traceback.print_exc(limit=1)

    print(f"\nExtracted {done} | skipped (existing) {skipped} | failed {failed}")
    print(f"Landmarks under: {LANDMARK_DIR}")
    print("Next: python scripts/prepare_data.py")


if __name__ == "__main__":
    main()
