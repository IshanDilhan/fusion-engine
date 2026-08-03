"""
Stage 4 - Full evaluation of the trained model (guide §10).

Reports, in increasing order of honesty:
  1. held-out test split (subject-wise, mixed datasets)
  2. per-source-dataset breakdown on that split (cross-dataset generalization)
  3. the live_test split (RealSense clips from the real room, never trained on)

Plus per-class precision/recall/F1, confusion matrices, and CPU latency of a
single window (the Jetson deployment budget check).

    python scripts/evaluate.py
    python scripts/evaluate.py --model BiGRU
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DEFAULT_MODEL_CONFIG,
    FEATURE_DIM,
    GESTURE_LABELS,
    NUM_WORKERS,
    REPORT_DIR,
    WINDOW,
)
from src.data import GestureSequenceDataset, load_index
from src.models import build_model, safe_name
from src.training import evaluate, get_device


def predict_split(model, df, device, batch_size, num_workers):
    ds = GestureSequenceDataset(df, train=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers,
                        pin_memory=(device.type == "cuda"))
    metrics, preds, targets = evaluate(model, loader, device, return_preds=True)
    return metrics, preds, targets


def confusion_png(targets, preds, title, path):
    cm = confusion_matrix(targets, preds, labels=range(len(GESTURE_LABELS)),
                          normalize="true")
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=GESTURE_LABELS, yticklabels=GESTURE_LABELS,
                vmin=0, vmax=1, cbar=False)
    plt.xlabel("predicted"); plt.ylabel("true"); plt.title(title)
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def metrics_table(metrics):
    return "\n".join([
        "| Metric | Value |", "|---|---|",
        f"| Accuracy | {metrics['accuracy']*100:.2f}% |",
        f"| Balanced accuracy | {metrics['balanced_accuracy']*100:.2f}% |",
        f"| Macro-F1 | {metrics['f1_macro']*100:.2f}% |",
        f"| Weighted-F1 | {metrics['f1_weighted']*100:.2f}% |",
    ])


def per_class_table(targets, preds):
    rep = classification_report(targets, preds, labels=range(len(GESTURE_LABELS)),
                                target_names=GESTURE_LABELS, output_dict=True,
                                zero_division=0)
    rows = ["| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for name in GESTURE_LABELS:
        r = rep[name]
        rows.append(f"| {name} | {r['precision']:.2f} | {r['recall']:.2f} | "
                    f"{r['f1-score']:.2f} | {int(r['support'])} |")
    return "\n".join(rows)


@torch.no_grad()
def cpu_latency_ms(model_cfg, state, runs=100):
    model = build_model(model_cfg["model"], **model_cfg.get("model_kwargs", {}))
    model.load_state_dict(state)
    model.eval()
    x = torch.randn(1, WINDOW, FEATURE_DIM)
    for _ in range(10):
        model(x)
    t0 = time.perf_counter()
    for _ in range(runs):
        model(x)
    return (time.perf_counter() - t0) / runs * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="defaults to the model named in checkpoints/model_config.json")
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    args = ap.parse_args()

    with open(DEFAULT_MODEL_CONFIG, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model_name = args.model or cfg["model"]
    sname = safe_name(model_name)
    ckpt = os.path.join(CHECKPOINT_DIR, f"best_{sname}.pth")

    device = get_device()
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model = build_model(model_name, **cfg.get("model_kwargs", {})).to(device)
    model.load_state_dict(state)
    print(f"Device: {device} | Model: {model_name} | Checkpoint: {ckpt}")

    out_dir = os.path.join(REPORT_DIR, "evaluation", sname)
    os.makedirs(out_dir, exist_ok=True)

    sections = [
        f"# Gesture Model - Stage 4: Evaluation Report ({model_name})\n",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M} · checkpoint `{os.path.basename(ckpt)}`\n",
    ]

    # 1) held-out test split
    test_df = load_index("test")
    metrics, preds, targets = predict_split(model, test_df, device,
                                            args.batch_size, args.num_workers)
    confusion_png(targets, preds, f"{model_name} - test confusion (row-normalized)",
                  os.path.join(out_dir, "confusion_test.png"))
    sections += [
        f"## 1. Held-out test split ({len(test_df)} sequences, subject-wise)\n",
        metrics_table(metrics), "",
        "### Per-class\n", per_class_table(targets, preds), "",
        "Confusion matrix: `confusion_test.png`\n",
    ]
    print(f"test      macro-F1 {metrics['f1_macro']*100:.2f}%  "
          f"acc {metrics['accuracy']*100:.2f}%")

    # 2) per-source-dataset breakdown (cross-dataset generalization)
    rows = ["| Source dataset | Sequences | Accuracy | Macro-F1 |", "|---|---|---|---|"]
    for ds_name, part in test_df.groupby("dataset"):
        m, _, _ = predict_split(model, part, device, args.batch_size, args.num_workers)
        rows.append(f"| {ds_name} | {len(part)} | {m['accuracy']*100:.2f}% | "
                    f"{m['f1_macro']*100:.2f}% |")
    sections += ["## 2. Test breakdown by source dataset\n", "\n".join(rows), "",
                 "Macro-F1 here spans only the classes each dataset contributes.\n"]

    # 3) live test set — the deployment truth
    try:
        live_df = load_index("live_test")
    except FileNotFoundError:
        live_df = pd.DataFrame()
    if len(live_df):
        metrics, preds, targets = predict_split(model, live_df, device,
                                                args.batch_size, args.num_workers)
        confusion_png(targets, preds, f"{model_name} - live test confusion",
                      os.path.join(out_dir, "confusion_live_test.png"))
        verdict = ("**meets** the ≥ 0.80 target" if metrics["f1_macro"] >= 0.80
                   else "**below** the ≥ 0.80 target — record more custom clips "
                        "for the weakest classes (guide §10)")
        sections += [
            f"## 3. Live test set ({len(live_df)} RealSense sequences, never trained on)\n",
            metrics_table(metrics), "",
            "### Per-class\n", per_class_table(targets, preds), "",
            f"Live macro-F1 {verdict}.\n",
            "Confusion matrix: `confusion_live_test.png`\n",
        ]
        print(f"live_test macro-F1 {metrics['f1_macro']*100:.2f}%  "
              f"acc {metrics['accuracy']*100:.2f}%")
    else:
        sections += ["## 3. Live test set\n",
                     "_No live_test sequences found — record them "
                     "(custom/live_test/<label>/) and re-run extraction + "
                     "prepare_data. This is the deployment metric (guide §10)._\n"]
        print("live_test: no data found — section skipped")

    # 4) deployment latency
    lat = cpu_latency_ms(cfg, state)
    sections += ["## 4. Deployment budget\n",
                 f"- Single-window forward pass (CPU, this machine): **{lat:.2f} ms** "
                 "— the temporal net is negligible next to MediaPipe landmark "
                 "extraction, as designed for the Jetson Orin Nano.\n"]

    report_path = os.path.join(out_dir, "EVALUATION_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sections))
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
