"""
Real-world evaluation on the CLEANED label set.

data/realworld/{train,val} crops were extracted from data/raw/clips before a
manual review removed clips whose recorded emotion didn't match the label
(wrong acted emotion). The crop folders were never regenerated, so some crops
still reference clips that no longer exist in data/raw/clips — those crops
carry a label the manual review already rejected. This script excludes them
(matching crop filename `<clip_id>_f###.jpg` against clip files still present
under data/raw/clips) before scoring, so results reflect trusted labels only.

Evaluates three things on the same cleaned val clips:
  1. CNN frame-level      (best_MobileNetV2, finetuned_MobileNetV2)
  2. CNN clip-level, mean-softmax fusion  (the window_probs.py-style baseline)
  3. LSTM clip-level      (best_MobileNetV2_LSTM, finetuned_MobileNetV2_LSTM;
     see src/models_lstm.py for the reverse-engineered architecture note)

Run (from modalities/emotion/):
    python scripts/evaluate_realworld_cleaned.py
"""
import glob
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from config import CHECKPOINT_DIR, DATA_DIR, EMOTION_LABELS, REPORT_DIR, ROOT
from src.engine import get_device
from src.models import ALL_MODELS
from src.models_lstm import MobileNetV2LSTM
from src.transforms import get_test_transforms

PROJECT_ROOT = os.path.normpath(os.path.join(ROOT, "..", ".."))
RAW_CLIPS_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "clips")
REALWORLD_DIR = os.path.join(DATA_DIR, "realworld")
OUT_DIR = os.path.join(REPORT_DIR, "evaluation_realworld_cleaned")
CLIP_RE = re.compile(r"(S\d+_F\d+_c\d+)_f(\d+)\.jpg$", re.IGNORECASE)


def raw_clip_ids():
    ids = set()
    for p in glob.glob(os.path.join(RAW_CLIPS_DIR, "**", "*.mp4"), recursive=True):
        ids.add(os.path.splitext(os.path.basename(p))[0])
    return ids


def build_clean_index(split, valid_clip_ids):
    """clip_id -> (label_idx, [frame paths sorted by frame index]), excluding
    clips whose source video was removed (crop label no longer trusted)."""
    by_clip = defaultdict(list)
    for cls in sorted(os.listdir(os.path.join(REALWORLD_DIR, split))):
        cls_dir = os.path.join(REALWORLD_DIR, split, cls)
        for f in os.listdir(cls_dir):
            m = CLIP_RE.search(f)
            if not m or m.group(1) not in valid_clip_ids:
                continue
            by_clip[m.group(1)].append((int(m.group(2)), os.path.join(cls_dir, f), int(cls) - 1))
    result = {}
    for cid, items in by_clip.items():
        items.sort(key=lambda x: x[0])
        result[cid] = (items[0][2], [p for _, p, _ in items])
    return result


def main():
    device = get_device()
    tf = get_test_transforms()
    valid = raw_clip_ids()
    print(f"Valid (non-removed) raw clip ids: {len(valid)}")

    clean_val = build_clean_index("val", valid)
    print(f"Cleaned val clips: {len(clean_val)}")
    val_counts = np.bincount([lab for lab, _ in clean_val.values()], minlength=7)
    print("Cleaned val per-class clip counts:",
          {EMOTION_LABELS[i]: int(val_counts[i]) for i in range(7)})
    os.makedirs(OUT_DIR, exist_ok=True)

    lines = ["# Real-world evaluation - cleaned labels\n",
             f"- Raw clips remaining after manual cleanup: {len(valid)}",
             f"- Cleaned val clips (labels trusted): {len(clean_val)}",
             "- Cleaned val per-class: " +
             ", ".join(f"{EMOTION_LABELS[i]}={int(val_counts[i])}" for i in range(7)) + "\n"]

    # ── CNN frame-level + clip-level (mean-softmax) ─────────────────────
    cnn_ckpts = {"best_MobileNetV2": "best_MobileNetV2.pth",
                 "finetuned_MobileNetV2": "finetuned_MobileNetV2.pth"}
    frame_paths, frame_labels, frame_clip = [], [], []
    for cid, (label, paths) in clean_val.items():
        for p in paths:
            frame_paths.append(p); frame_labels.append(label); frame_clip.append(cid)
    frame_labels = np.array(frame_labels)
    clip_ids_order = list(clean_val.keys())
    clip_true = np.array([clean_val[c][0] for c in clip_ids_order])

    frame_probs = {}
    for name, fname in cnn_ckpts.items():
        m = ALL_MODELS["MobileNetV2"]()
        m.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, fname),
                                     map_location=device, weights_only=True))
        m.to(device).eval()
        probs = []
        with torch.no_grad():
            for i in range(0, len(frame_paths), 128):
                batch = frame_paths[i:i + 128]
                x = torch.stack([tf(Image.open(p).convert("RGB")) for p in batch]).to(device)
                probs.append(F.softmax(m(x), dim=1).cpu().numpy())
        frame_probs[name] = np.concatenate(probs) if probs else np.zeros((0, 7))
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    lines.append("## 1) CNN frame-level\n")
    for name, probs in frame_probs.items():
        preds = probs.argmax(1)
        pc = f1_score(frame_labels, preds, average=None, labels=range(7), zero_division=0)
        lines.append(f"### {name}: n={len(frame_labels)} frames  "
                     f"acc={accuracy_score(frame_labels,preds)*100:.1f}%  "
                     f"bAcc={balanced_accuracy_score(frame_labels,preds)*100:.1f}%  "
                     f"macroF1={f1_score(frame_labels,preds,average='macro')*100:.1f}%\n")
        lines.append("| | " + " | ".join(EMOTION_LABELS) + " |")
        lines.append("|---|" + "---|" * 7)
        lines.append("| F1 | " + " | ".join(f"{v*100:.0f}" for v in pc) + " |\n")
        cm = confusion_matrix(frame_labels, preds, labels=range(7))
        pd.DataFrame(cm, index=EMOTION_LABELS, columns=EMOTION_LABELS).to_csv(
            os.path.join(OUT_DIR, f"confusion_frame_{name}.csv"))

    lines.append("\n## 2) CNN clip-level via mean-softmax (fusion baseline)\n")
    for name, probs in frame_probs.items():
        by_clip = defaultdict(list)
        for p, cid in zip(probs, frame_clip):
            by_clip[cid].append(p)
        preds = np.array([np.mean(by_clip[c], axis=0).argmax() for c in clip_ids_order])
        pc = f1_score(clip_true, preds, average=None, labels=range(7), zero_division=0)
        lines.append(f"### {name}: n={len(clip_true)} clips  "
                     f"acc={accuracy_score(clip_true,preds)*100:.1f}%  "
                     f"bAcc={balanced_accuracy_score(clip_true,preds)*100:.1f}%  "
                     f"macroF1={f1_score(clip_true,preds,average='macro')*100:.1f}%\n")
        lines.append("| | " + " | ".join(EMOTION_LABELS) + " |")
        lines.append("|---|" + "---|" * 7)
        lines.append("| F1 | " + " | ".join(f"{v*100:.0f}" for v in pc) + " |\n")

    # ── LSTM clip-level ──────────────────────────────────────────────────
    lines.append("\n## 3) LSTM clip-level\n")
    lstm_ckpts = {"best_MobileNetV2_LSTM": "best_MobileNetV2_LSTM.pth",
                  "finetuned_MobileNetV2_LSTM": "finetuned_MobileNetV2_LSTM.pth"}
    clip_tensors = {cid: torch.stack([tf(Image.open(p).convert("RGB")) for p in paths])
                    for cid, (_, paths) in clean_val.items()}

    for name, fname in lstm_ckpts.items():
        path = os.path.join(CHECKPOINT_DIR, fname)
        if not os.path.exists(path):
            lines.append(f"### {name}: checkpoint not found ({path})\n")
            continue
        model = MobileNetV2LSTM()
        sd = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(sd, strict=True)  # raises if architecture mismatches
        model.to(device).eval()

        for agg in ("last", "mean"):
            preds = []
            with torch.no_grad():
                for cid in clip_ids_order:
                    x = clip_tensors[cid].unsqueeze(0).to(device)
                    preds.append(int(model(x, agg=agg).argmax(1).item()))
            preds = np.array(preds)
            pc = f1_score(clip_true, preds, average=None, labels=range(7), zero_division=0)
            lines.append(f"### {name} (agg={agg}): n={len(clip_true)} clips  "
                         f"acc={accuracy_score(clip_true,preds)*100:.1f}%  "
                         f"bAcc={balanced_accuracy_score(clip_true,preds)*100:.1f}%  "
                         f"macroF1={f1_score(clip_true,preds,average='macro')*100:.1f}%\n")
            lines.append("| | " + " | ".join(EMOTION_LABELS) + " |")
            lines.append("|---|" + "---|" * 7)
            lines.append("| F1 | " + " | ".join(f"{v*100:.0f}" for v in pc) + " |\n")
            cm = confusion_matrix(clip_true, preds, labels=range(7))
            pd.DataFrame(cm, index=EMOTION_LABELS, columns=EMOTION_LABELS).to_csv(
                os.path.join(OUT_DIR, f"confusion_lstm_{name}_{agg}.csv"))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with open(os.path.join(OUT_DIR, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
