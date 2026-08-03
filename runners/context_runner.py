"""
Standalone Context runner. Uses the context modality's own scene-classification
sub-model (zero-shot CLIP image-text matching, replacing the earlier trained
CNN -- see modalities/context/scene_classification/src/zero_shot.py's own
docstring: 99.5% vs 82.2% on the captured clips) for model construction and
per-frame inference, then emits NormalisedFrameCue records.

The scene modality now ships a 5-class deployed vocabulary (classroom/
kitchen/hospital/cloth_store/museum, for the broader jetson deployment
project) -- restricted here to just classroom/kitchen via create_scene_
classifier(classes=...), matching this dataset's actual 2 environments and
pipeline/aggregate.py's CONTEXT_CLASSES. The classifier does its own internal
temporal smoothing (a probability-history deque) -- reset() per clip so it
never leaks state across videos, same as the other runners' stateful trackers.

Correctness fixes applied here (see Integration_API.md #2.4):
  - native "uncertain" label -> canonical "Unknown"
  - activity/engaged/n_objects are structurally absent from this model
    (no object detection, activity recognition, or engagement logic exists
    anywhere in the repo) -> hardcoded documented placeholders, not fabricated
    values.

Run inside .venvs/context (torch, torchvision, opencv-python, pillow, numpy,
open_clip_torch -- see Integration_API.md #4). Needs HF_HOME pointed at the
modality's pre-downloaded CLIP weights (see JETSON_SETUP_GUIDE.md on the
external drive) -- set below, before importing open_clip, so this script
works without the caller remembering to export it.

Usage:
    # single clip
    .venvs/context/Scripts/python.exe runners/context_runner.py --clip <path> --out <out.jsonl>

    # batch mode: loads the model ONCE, loops every clip in clips.csv
    .venvs/context/Scripts/python.exe runners/context_runner.py \
        --manifest Data/Dataset/hri-multimodal-intent-v2.0.0/annotations/clips.csv \
        --clips-root Data/Dataset/hri-multimodal-intent-v2.0.0/raw/clips \
        --out data/measured/context_frame_cues.jsonl
"""
import argparse
import os
import sys
import time

# Pre-downloaded CLIP/SmolVLM2 weights live here (external drive) -- must be
# set before open_clip is imported (by scene_classification/src/zero_shot.py)
# so it never tries to hit the network. See JETSON_SETUP_GUIDE.md.
MODALITIES_ROOT = "/media/hri_multimodal/KINGSTON_KG/hri-jetson"
os.environ.setdefault("HF_HOME", os.path.join(MODALITIES_ROOT, "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

RUNNERS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RUNNERS_DIR)
# scene_classification/ has no __init__.py (a sys.path root, not a proper
# subpackage) -- classifier.py bootstraps its own local `config`/`src`
# imports once this directory is on sys.path, same shape as the old
# "Context Repo/scene classification" + `import video` pattern this replaces.
sys.path.insert(0, os.path.join(MODALITIES_ROOT, "modalities", "context", "scene_classification"))

from common.schema import NormalisedFrameCue, write_jsonl, append_batch, read_manifest  # noqa: E402
from common.constants import CONFIDENCE_FLOOR  # noqa: E402

import cv2  # noqa: E402
from src.classifier import create_scene_classifier  # noqa: E402

CUE = "context"
FLOOR = CONFIDENCE_FLOOR[CUE]
SCENE_CLASSES = ["classroom", "kitchen"]  # this dataset's only 2 environments

# Structurally absent from this model -- see Integration_API.md #2.4.
NOT_MEASURED_EXTRA = {"activity": None, "engaged": None, "n_objects": 0}


def load_model():
    return create_scene_classifier(backend="clip", classes=SCENE_CLASSES)


def process_clip(clip_path: str, classifier):
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open clip: {clip_path}")

    classifier.reset()
    records = []
    frame_idx = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        result = classifier.predict(frame)

        native_label = result["label"]
        label = "Unknown" if native_label == "uncertain" else native_label
        conf = result["confidence"]

        records.append(NormalisedFrameCue(
            cue=CUE, frame_idx=frame_idx, label=label, confidence=conf,
            probs=result["probs"], valid=(conf >= FLOOR and label != "Unknown"),
            extra=dict(NOT_MEASURED_EXTRA)))

    cap.release()
    return records


def run_single(clip_path: str, out_path: str):
    classifier = load_model()
    records = process_clip(clip_path, classifier)
    write_jsonl(records, out_path)
    print(f"[context_runner] {len(records)} frames -> {out_path}")


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
        print(f"[context_runner] resuming: {len(done_ids)} clips already done")
    else:
        mode = "w"

    classifier = load_model()

    t0 = time.time()
    n_done = 0
    with open(out_path, mode, encoding="utf-8") as f:
        for i, row in enumerate(rows):
            clip_id = row["clip_id"]
            if clip_id in done_ids:
                continue
            clip_path = os.path.join(clips_root, row["filepath"])
            try:
                records = process_clip(clip_path, classifier)
            except Exception as e:
                print(f"[context_runner] ERROR on {clip_id} ({clip_path}): {e}")
                continue
            append_batch(f, clip_id, records)
            f.flush()
            n_done += 1
            if n_done % 25 == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed
                remaining = (len(rows) - len(done_ids) - n_done) / rate if rate > 0 else float("inf")
                print(f"[context_runner] {i+1}/{len(rows)} clips ({n_done} this run, "
                      f"{rate:.2f} clips/s, ~{remaining/60:.1f} min remaining)")

    print(f"[context_runner] batch done: {n_done} clips processed -> {out_path}")


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
