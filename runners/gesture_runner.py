"""
Standalone Gesture runner. Wraps the Gesture Repo's v2 model (MediaPipe
Holistic landmarks -> 185-dim per-frame features -> 32-frame window -> TCN
temporal classifier, native 8-class vocabulary INCLUDING "idle" as a real
trained class -- see Gesture Repo/reports/GESTURE_V2_DESIGN_AND_HPC_GUIDE.md)
behind the same NormalisedFrameCue interface as the other three cue runners.

This replaces the previous per-frame KeyPointClassifier/PointHistoryClassifier
heuristic -- that repo layout (Gesture Repo/model/...) no longer exists, the
whole recognition approach was replaced (same kind of swap Motion Repo already
went through; see motion_runner.py's own docstring for the precedent this
follows). GestureEngine (Gesture Repo/src/engine.py) mirrors MotionInference's
API almost exactly:
  - stateful (rolling ~2s frame buffer, EMA-smoothed softmax, debounced label
    switching); loaded ONCE per batch run, engine.reset() at the start of
    every clip (matches motion_runner.py's convention).
  - MediaPipe Holistic itself IS recreated fresh per clip (its own internal
    tracking state must not leak across unrelated clips, same reasoning as
    the other three runners).
  - The first WINDOW-1 (31) frames of every clip, and any frame with no
    person detected at all, come back at confidence=0.0 -- the engine's own
    honest "not enough signal yet" state (see GestureEngine.process(): it
    returns ("idle", 0.0) verbatim in both cases). No special-casing needed:
    those frames are naturally invalid via the confidence floor below, the
    same treatment motion_runner.py already gives its own buffering frames.
  - "idle" (GESTURE_LABELS[0]) is a genuine trained class -- a person/hand
    present but not making a recognized gesture -- not a fallback label. It
    maps 1:1 to this pipeline's canonical "Unknown" class (the 8th slot in
    pipeline/aggregate.py's GESTURE_CLASSES); the other 7 native labels
    already match the canonical vocabulary's names exactly. This is also why
    a real idle observation and a genuinely-absent one are NOT confused here
    the way the old per-frame classifier conflated them: idle has a real,
    often-high confidence (a confident temporal read of "nothing active
    happening"), while absence reports exactly 0.0 -- the confidence floor
    below separates them without any extra bookkeeping.

Run inside .venvs/gesture (torch, mediapipe, numpy -- see Gesture Repo's own
requirements for the exact pins).

Usage:
    # single clip
    .venvs/gesture/bin/python runners/gesture_runner.py --clip <path> --out <out.jsonl>

    # batch mode: loads the model ONCE, loops every clip in clips.csv
    .venvs/gesture/bin/python runners/gesture_runner.py \
        --manifest Data/Dataset/hri-multimodal-intent-v2.0.0/annotations/clips.csv \
        --clips-root Data/Dataset/hri-multimodal-intent-v2.0.0/raw/clips \
        --out pipeline/measured/gesture_frame_cues.jsonl
"""
import argparse
import os
import sys
import time

RUNNERS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RUNNERS_DIR)
GESTURE_REPO = os.path.join(os.path.dirname(RUNNERS_DIR), "Gesture Repo")
sys.path.insert(0, GESTURE_REPO)

from common.schema import NormalisedFrameCue, write_jsonl, append_batch, read_manifest  # noqa: E402
from common.constants import CONFIDENCE_FLOOR  # noqa: E402

import cv2  # noqa: E402
import mediapipe as mp  # noqa: E402
# Module-level import only -- GestureEngine's __init__ does the (one-time,
# batch-wide) checkpoint load; nothing here triggers webcam/GUI code.
from src.engine import GestureEngine  # noqa: E402

CUE = "gesture"
FLOOR = CONFIDENCE_FLOOR[CUE]

# Native engine label -> this pipeline's canonical GESTURE_CLASSES (see
# pipeline/aggregate.py). Only "idle" differs in name; the other 7 already
# match exactly.
NATIVE_TO_CANONICAL = {"idle": "Unknown"}


def process_clip(clip_path: str, engine: GestureEngine):
    """Pure per-clip logic. Creates a fresh MediaPipe Holistic tracker for
    this clip and resets the (reused, already-loaded) GestureEngine's
    rolling buffer -- see module docstring."""
    mp_holistic = mp.solutions.holistic
    engine.reset()

    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open clip: {clip_path}")

    records = []
    frame_idx = -1
    with mp_holistic.Holistic(model_complexity=1, min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = holistic.process(rgb)
            rgb.flags.writeable = True

            native_label, confidence = engine.process_holistic(res)
            label = NATIVE_TO_CANONICAL.get(native_label, native_label)
            has_person = res.pose_landmarks is not None

            records.append(NormalisedFrameCue(
                cue=CUE, frame_idx=frame_idx, label=label, confidence=float(confidence),
                probs={}, valid=(confidence >= FLOOR),
                extra={"has_person": has_person}))

    cap.release()
    return records


def run_single(clip_path: str, out_path: str):
    engine = GestureEngine()
    records = process_clip(clip_path, engine)
    write_jsonl(records, out_path)
    n_valid = sum(1 for r in records if r.valid)
    print(f"[gesture_runner] {len(records)} frames -> {out_path} "
          f"({n_valid} valid, {len(records)-n_valid} invalid)")


def run_batch(manifest_csv: str, clips_root: str, out_path: str, limit=None, resume=False):
    rows = read_manifest(manifest_csv)
    if limit:
        rows = rows[:limit]

    done_ids = set()
    mode = "a"
    if resume and os.path.isfile(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(line.split('"clip_id": "', 1)[1].split('"', 1)[0])
                except IndexError:
                    pass
        print(f"[gesture_runner] resuming: {len(done_ids)} clips already done")
    else:
        mode = "w"

    engine = GestureEngine()

    t0 = time.time()
    n_done = 0
    with open(out_path, mode, encoding="utf-8") as f:
        for i, row in enumerate(rows):
            clip_id = row["clip_id"]
            if clip_id in done_ids:
                continue
            clip_path = os.path.join(clips_root, row["filepath"])
            try:
                records = process_clip(clip_path, engine)
            except Exception as e:
                print(f"[gesture_runner] ERROR on {clip_id} ({clip_path}): {e}")
                continue
            append_batch(f, clip_id, records)
            f.flush()
            n_done += 1
            if n_done % 25 == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed
                remaining = (len(rows) - len(done_ids) - n_done) / rate if rate > 0 else float("inf")
                print(f"[gesture_runner] {i+1}/{len(rows)} clips ({n_done} this run, "
                      f"{rate:.2f} clips/s, ~{remaining/60:.1f} min remaining)")

    print(f"[gesture_runner] batch done: {n_done} clips processed -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", help="single-clip mode: path to one clip")
    ap.add_argument("--out", required=True, help="output JSONL path")
    ap.add_argument("--manifest", help="batch mode: path to clips.csv")
    ap.add_argument("--clips-root", help="batch mode: dataset root (filepath column is relative to this)")
    ap.add_argument("--limit", type=int, default=None, help="batch mode: only process first N rows (testing)")
    ap.add_argument("--resume", action="store_true", help="batch mode: skip clip_ids already present in --out")
    args = ap.parse_args()

    if args.manifest:
        if not args.clips_root:
            raise SystemExit("--clips-root is required with --manifest")
        run_batch(args.manifest, args.clips_root, args.out, limit=args.limit, resume=args.resume)
    else:
        if not args.clip:
            raise SystemExit("either --clip or --manifest is required")
        run_single(args.clip, args.out)
