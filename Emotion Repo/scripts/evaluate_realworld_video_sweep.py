"""
Full real-world video sweep: runs the ACTUAL deployed detection pipeline
(inference/video.py's robust full->CLAHE->tiled face detector) over every
clip still present under data/raw/clips (i.e. after the manual review that
removed clips with a wrong recorded emotion — every clip iterated here is by
construction "cleaned", since removed clips simply no longer exist on disk).

Scores four models against the trusted labels (from data/realworld crops,
matched by clip id):
  - CNN mean-softmax fusion: best_MobileNetV2, finetuned_MobileNetV2
  - LSTM sequence classifier: best_MobileNetV2_LSTM, finetuned_MobileNetV2_LSTM
    (see src/models_lstm.py)

For each model reports: overall accuracy/balanced-accuracy/macro-F1, full
per-class precision/recall/F1, and a confusion matrix (which classes get
misclassified as which). Also reports face-detection coverage.

Run (from modalities/emotion/):
    python scripts/evaluate_realworld_video_sweep.py
    python scripts/evaluate_realworld_video_sweep.py --limit 50   # smoke test
"""
import argparse
import os
import re
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from config import CHECKPOINT_DIR, DATA_DIR, EMOTION_LABELS, REPORT_DIR, ROOT
from src.engine import get_device
from src.models import ALL_MODELS
from src.models_lstm import MobileNetV2LSTM
from src.transforms import get_test_transforms

PROJECT_ROOT = os.path.normpath(os.path.join(ROOT, "..", ".."))
CLIPS_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "clips")
REALWORLD_DIR = os.path.join(DATA_DIR, "realworld")
OUT_DIR = os.path.join(REPORT_DIR, "evaluation_realworld_video_sweep")
CLIP_RE = re.compile(r"(S\d+_F\d+_c\d+)_f\d+\.jpg$", re.IGNORECASE)

FRAMES_PER_CLIP = 24     # sampled for CNN mean-softmax + face coverage
LSTM_WINDOW = 12         # subsequence length fed to the LSTM (matches its
                          # training-time window; see src/models_lstm.py note)
MAX_W = 640
_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


# ── Robust face detection (identical to inference/video.py) ────────────────
def apply_clahe(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    return cv2.cvtColor(cv2.merge((_clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)


def box_to_px(box, w, h):
    x, y = max(0, int(box.xmin * w)), max(0, int(box.ymin * h))
    return x, y, int(box.width * w), int(box.height * h)


def detect_face_box(detector, frame, small):
    h, w = frame.shape[:2]
    r = detector.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    if r.detections:
        return box_to_px(r.detections[0].location_data.relative_bounding_box, w, h)
    small_cl = apply_clahe(small)
    r = detector.process(cv2.cvtColor(small_cl, cv2.COLOR_BGR2RGB))
    if r.detections:
        return box_to_px(r.detections[0].location_data.relative_bounding_box, w, h)
    frame_cl = apply_clahe(frame)
    ov = 0.15
    tw, th = int(w * (0.5 + ov)), int(h * (0.5 + ov))
    for tx, ty in [(0, 0), (w - tw, 0), (0, h - th), (w - tw, h - th)]:
        crop = frame_cl[ty:ty + th, tx:tx + tw]
        r = detector.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        if r.detections:
            cx, cy, cbw, cbh = box_to_px(r.detections[0].location_data.relative_bounding_box, tw, th)
            return (tx + cx, ty + cy, cbw, cbh)
    return None


def clip_label_map():
    """clip_id -> (label_idx, split), from the trusted (currently-existing-
    clip-backed) crops. Because build happens by scanning crop filenames and
    is only ever CONSULTED for clips still present on disk, results are
    automatically restricted to the cleaned label set."""
    mapping = {}
    for split in ("train", "val"):
        for cls in os.listdir(os.path.join(REALWORLD_DIR, split)):
            for f in os.listdir(os.path.join(REALWORLD_DIR, split, cls)):
                m = CLIP_RE.search(f)
                if m:
                    mapping[m.group(1)] = (int(cls) - 1, split)
    return mapping


def evenly_spaced(seq, k):
    if len(seq) <= k:
        return seq
    idx = np.linspace(0, len(seq) - 1, k).astype(int)
    return [seq[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Score only the first N clips (smoke test).")
    args = ap.parse_args()

    device = get_device()
    tf = get_test_transforms()

    cnn_ckpts = {"best_MobileNetV2": "best_MobileNetV2.pth",
                 "finetuned_MobileNetV2": "finetuned_MobileNetV2.pth"}
    cnn_models = {}
    for name, fname in cnn_ckpts.items():
        m = ALL_MODELS["MobileNetV2"]()
        m.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, fname),
                                     map_location=device, weights_only=True))
        cnn_models[name] = m.to(device).eval()

    lstm_ckpts = {"best_MobileNetV2_LSTM": "best_MobileNetV2_LSTM.pth",
                  "finetuned_MobileNetV2_LSTM": "finetuned_MobileNetV2_LSTM.pth"}
    lstm_models = {}
    for name, fname in lstm_ckpts.items():
        path = os.path.join(CHECKPOINT_DIR, fname)
        if os.path.exists(path):
            m = MobileNetV2LSTM()
            m.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
            lstm_models[name] = m.to(device).eval()

    detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    labels_map = clip_label_map()

    videos = []
    for dirpath, _, files in os.walk(CLIPS_DIR):
        for f in sorted(files):
            if f.lower().endswith(".mp4"):
                videos.append(os.path.join(dirpath, f))
    if args.limit:
        videos = videos[:args.limit]
    print(f"{len(videos)} clips on disk (post-cleanup) | {len(labels_map)} labeled | device={device}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    t0 = time.time()
    for i, vp in enumerate(videos, 1):
        cid = os.path.splitext(os.path.basename(vp))[0]
        scene = "kitchen" if f"{os.sep}kitchen{os.sep}" in vp else "classroom"
        lab_split = labels_map.get(cid)

        cap = cv2.VideoCapture(vp)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        idxs = set(np.linspace(0, max(0, n - 1), min(FRAMES_PER_CLIP, max(1, n))).astype(int).tolist())
        face_tensors, fi, sampled = [], -1, 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            fi += 1
            if fi not in idxs:
                continue
            sampled += 1
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (MAX_W, int(h * MAX_W / w))) if w > MAX_W else frame
            box = detect_face_box(detector, frame, small)
            if box is None:
                continue
            x, y, bw, bh = box
            face = frame[y:y + bh, x:x + bw]
            if face.size == 0:
                continue
            face_tensors.append(tf(Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))))
        cap.release()

        row = {"clip": cid, "scene": scene, "frames_sampled": sampled, "faces": len(face_tensors),
               "face_coverage": round(len(face_tensors) / sampled, 3) if sampled else 0.0,
               "true": EMOTION_LABELS[lab_split[0]] if lab_split else "",
               "split": lab_split[1] if lab_split else "unlabeled"}

        if face_tensors:
            batch = torch.stack(face_tensors).to(device)
            with torch.no_grad():
                for name, m in cnn_models.items():
                    mean = F.softmax(m(batch), dim=1).mean(dim=0).cpu().numpy()
                    row[f"pred_{name}"] = EMOTION_LABELS[int(mean.argmax())]
                    row[f"conf_{name}"] = round(float(mean.max()), 4)

                seq = evenly_spaced(face_tensors, LSTM_WINDOW)
                seq_batch = torch.stack(seq).unsqueeze(0).to(device)  # (1, T, 3, 224, 224)
                for name, m in lstm_models.items():
                    b, t_, c, hh, ww = seq_batch.shape
                    feat = m.features(seq_batch.view(b * t_, c, hh, ww))
                    feat = m.pool(feat).flatten(1).view(b, t_, -1)
                    out, _ = m.lstm(feat)
                    for agg in ("last", "mean"):
                        pooled = out[:, -1, :] if agg == "last" else out.mean(dim=1)
                        probs = F.softmax(m.classifier(pooled), dim=1)[0].cpu().numpy()
                        row[f"pred_{name}_{agg}"] = EMOTION_LABELS[int(probs.argmax())]
                        row[f"conf_{name}_{agg}"] = round(float(probs.max()), 4)
        else:
            for name in cnn_models:
                row[f"pred_{name}"], row[f"conf_{name}"] = "none", 0.0
            for name in lstm_models:
                for agg in ("last", "mean"):
                    row[f"pred_{name}_{agg}"], row[f"conf_{name}_{agg}"] = "none", 0.0
        rows.append(row)

        if i % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {i}/{len(videos)} clips ({elapsed:.0f}s, ~{elapsed/i*len(videos):.0f}s total est.)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "clip_predictions.csv"), index=False)

    model_pred_cols = {**{n: f"pred_{n}" for n in cnn_models},
                       **{f"{n}_{a}": f"pred_{n}_{a}" for n in lstm_models for a in ("last", "mean")}}

    total_frames, total_faces = df["frames_sampled"].sum(), df["faces"].sum()
    lines = ["# Real-world video sweep - cleaned labels, fixed robust detector\n",
             f"- Clips on disk (post manual cleanup): {len(df)}",
             f"- Labeled clips: {(df['split'] != 'unlabeled').sum()}",
             f"- Frame-level face coverage: {total_faces}/{total_frames} = {100*total_faces/total_frames:.1f}%",
             f"- Clips with a face in every sampled frame: {(df['face_coverage']==1.0).sum()}/{len(df)}",
             f"- Clips with zero face detected: {(df['faces']==0).sum()}/{len(df)}\n"]

    summary_rows = []
    for model_key, col in model_pred_cols.items():
        lines.append(f"\n## {model_key}\n")
        for split in ("val", "train"):
            sub = df[(df["split"] == split) & (df["faces"] > 0) & (df[col] != "none")]
            if not len(sub):
                continue
            y = [EMOTION_LABELS.index(v) for v in sub["true"]]
            yh = [EMOTION_LABELS.index(v) for v in sub[col]]
            acc = accuracy_score(y, yh)
            bacc = balanced_accuracy_score(y, yh)
            mf1 = f1_score(y, yh, average="macro")
            note = " (seen in fine-tuning - memorized, not a generalization estimate)" \
                   if (split == "train" and ("finetuned" in model_key)) else ""
            lines.append(f"### split={split}{note}: n={len(sub)} clips  "
                         f"acc={acc*100:.1f}%  bAcc={bacc*100:.1f}%  macroF1={mf1*100:.1f}%\n")
            report = classification_report(y, yh, labels=range(7), target_names=EMOTION_LABELS,
                                          output_dict=True, zero_division=0)
            lines.append("| Emotion | Precision | Recall | F1 | Support |")
            lines.append("|---|---|---|---|---|")
            for e in EMOTION_LABELS:
                m = report[e]
                lines.append(f"| {e} | {m['precision']*100:.0f}% | {m['recall']*100:.0f}% | "
                             f"{m['f1-score']*100:.0f}% | {int(m['support'])} |")
            cm = confusion_matrix(y, yh, labels=range(7))
            cm_df = pd.DataFrame(cm, index=EMOTION_LABELS, columns=EMOTION_LABELS)
            cm_df.to_csv(os.path.join(OUT_DIR, f"confusion_{model_key}_{split}.csv"))
            lines.append(f"\nConfusion matrix (rows=true, cols=predicted):\n")
            lines.append(cm_df.to_markdown())
            lines.append("")
            if split == "val":
                summary_rows.append({"model": model_key, "n_clips": len(sub),
                                    "accuracy": round(acc*100, 1), "balanced_acc": round(bacc*100, 1),
                                    "macro_f1": round(mf1*100, 1)})

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False)
        lines_summary = ["\n## Summary (val split, sorted by macro-F1)\n", summary_df.to_markdown(index=False)]
    else:
        lines_summary = ["\n## Summary\n", "(no val-split clips scored - sample too small)"]
    full_report = lines[:6] + lines_summary + lines[6:]

    with open(os.path.join(OUT_DIR, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(full_report))
    print("\n".join(lines_summary))
    print(f"\nFull report saved to {OUT_DIR}/RESULTS.md")


if __name__ == "__main__":
    main()
