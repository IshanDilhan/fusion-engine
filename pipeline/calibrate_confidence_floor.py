"""
Empirical CONFIDENCE_FLOOR calibration, per cue, against scenarios.csv's/
clips.csv's authored ground truth -- formalizes the ad hoc sweep that found
gesture's floor was badly miscalibrated (0.80, tuned for the old per-frame
keypoint classifier; the new GestureEngine's real ceiling turned out to be
~0.65-0.70 even for confidently-correct "idle" reads, discarding 98.6% of
real detections until recalibrated to 0.20 -- see runners/common/constants.py).

For each candidate floor, recomputes valid=(confidence>=floor) per frame,
re-aggregates per clip using the SAME function pipeline/aggregate.py
actually uses for that cue (mean-probability-vector-then-argmax for
emotion/context/motion, matching _prob_mean_features -- NOT majority-vote,
which is gesture-only), and compares the resulting clip-level dominant
class against ground truth:
  - emotion: scenarios.csv's emotion_v3, excluding emotion_masked=='TRUE'
    clips (those are specifically flagged as designed-ambiguous/unreliable
    -- see clips.csv's caveat column).
  - context: clips.csv's OWN context column directly (which room a clip was
    filmed in is ground truth by construction, not a designed/actable
    quantity -- no masking exclusion needed, context_masked is about what
    reaches the fusion model, not whether the room is really identifiable).

Reports clip-level accuracy vs floor -- the metric that matters, since
that's what actually reaches the fusion models, not a raw frame-level
confidence pass-rate. Motion is intentionally NOT covered here: it was
separately retrained/validated via the Motion Repo replacement, outside
this session's audit scope.
"""
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline"))
from aggregate import EMOTION_CLASSES, CONTEXT_CLASSES, MEASURED_DIR, CLIP_MISSING_THRESHOLD  # noqa: E402
from canonical_map import map_intended  # noqa: E402
from dataset_config import DATASET_ROOT  # noqa: E402

CLIPS_CSV = os.path.join(DATASET_ROOT, "annotations", "clips.csv")
SCENARIOS_CSV = os.path.join(DATASET_ROOT, "annotations", "scenarios.csv")

FLOOR_CANDIDATES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def load_frames(cue):
    frames_by_clip = defaultdict(list)
    path = os.path.join(MEASURED_DIR, f"{cue}_frame_cues.jsonl")
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            frames_by_clip[d["clip_id"]].append(d)
    return frames_by_clip


def ground_truth_emotion():
    with open(SCENARIOS_CSV, newline="", encoding="utf-8") as f:
        scenarios = {r["v3_row"]: r for r in csv.DictReader(f)}
    with open(CLIPS_CSV, newline="", encoding="utf-8") as f:
        clips = list(csv.DictReader(f))
    intended = {}
    for c in clips:
        if c.get("emotion_masked") == "TRUE":
            continue
        scen = scenarios.get(c["v3_row"])
        if scen is None:
            continue
        val = map_intended("emotion", scen["emotion_v3"])
        if val is None:
            continue
        intended[c["clip_id"]] = val
    return intended


def ground_truth_context():
    with open(CLIPS_CSV, newline="", encoding="utf-8") as f:
        clips = list(csv.DictReader(f))
    return {c["clip_id"]: map_intended("context", c["context"]) for c in clips
            if map_intended("context", c["context"]) is not None}


def eval_floor(frames_by_clip, intended_by_clip, classes, floor):
    n_correct = 0
    n_total = 0
    n_missing = 0
    for cid, intended in intended_by_clip.items():
        frames = frames_by_clip.get(cid)
        if not frames:
            continue
        valid = [f for f in frames if f["confidence"] >= floor]
        valid_fraction = len(valid) / len(frames)
        if valid_fraction < CLIP_MISSING_THRESHOLD:
            pred = None
            n_missing += 1
        else:
            mat = np.array([[f["probs"].get(c, 0.0) for c in classes] for f in valid])
            pred = classes[int(np.argmax(mat.mean(axis=0)))]
        n_correct += (pred == intended)
        n_total += 1
    return n_correct / n_total, n_missing, n_total


def sweep(cue, classes, ground_truth_fn):
    intended_by_clip = ground_truth_fn()
    frames_by_clip = load_frames(cue)
    print(f"\n=== {cue}: {len(intended_by_clip)} clips with checkable ground truth ===")
    print(f"{'floor':>6} {'accuracy':>10} {'n_missing':>10}")
    best_floor, best_acc = None, -1.0
    for floor in FLOOR_CANDIDATES:
        acc, n_missing, n_total = eval_floor(frames_by_clip, intended_by_clip, classes, floor)
        print(f"{floor:6.2f} {acc:10.3f} {n_missing:10d}")
        if acc > best_acc:
            best_floor, best_acc = floor, acc
    print(f"[calibrate] {cue}: best floor={best_floor:.2f} (accuracy={best_acc:.3f})")
    return best_floor, best_acc


def main():
    sweep("emotion", EMOTION_CLASSES, ground_truth_emotion)
    sweep("context", CONTEXT_CLASSES, ground_truth_context)


if __name__ == "__main__":
    main()
