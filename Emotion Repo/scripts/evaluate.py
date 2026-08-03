"""
Stage 4 - Full evaluation of a trained checkpoint on the RAF-DB test set.

Reports overall metrics (accuracy, balanced accuracy, weighted & macro
precision/recall/F1), per-class metrics, a confusion matrix, confidence
analysis, and the highest-confidence misclassifications. Saves CSVs, plots and
a markdown report under reports/evaluation/<model>/.

Run:
    python scripts/evaluate.py --model EfficientNet-B0
    python scripts/evaluate.py --model MobileNetV2 --checkpoint checkpoints/best_MobileNetV2.pth
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import BATCH_SIZE, CHECKPOINT_DIR, DEFAULT_MODEL, EMOTION_LABELS, REPORT_DIR, TEST_DIR
from src.engine import get_device
from src.models import ALL_MODELS, safe_name
from src.transforms import get_test_transforms


def collect_predictions(model, loader, device):
    preds, labels, confs = [], [], []
    with torch.no_grad():
        for imgs, y in loader:
            imgs = imgs.to(device)
            probs = F.softmax(model(imgs), dim=1)
            conf, pred = torch.max(probs, 1)
            preds.extend(pred.cpu().numpy())
            labels.extend(y.numpy())
            confs.extend(conf.cpu().numpy())
    return np.array(labels), np.array(preds), np.array(confs)


def plot_confusion(cm, path):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS)
    plt.title("Confusion matrix (cell color = row-normalized, text = counts)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_per_class(per_class_df, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col in zip(axes, ["F1-Score", "Recall"]):
        vals = per_class_df[col]
        colors = ["#5aa469" if v > 0.85 else "#e0a93b" if v > 0.70 else "#d8563b" for v in vals]
        ax.bar(per_class_df["Emotion"], vals, color=colors)
        ax.axhline(0.85, ls="--", color="red", label="0.85 target")
        ax.set_ylim(0, 1)
        ax.set_ylabel(col)
        ax.set_title(f"Per-class {col}")
        ax.tick_params(axis="x", rotation=30)
        ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confidence(confs, correct_mask, path):
    plt.figure(figsize=(9, 5))
    plt.hist(confs[correct_mask], bins=30, alpha=0.7, color="green",
             label=f"Correct (n={correct_mask.sum()})")
    if (~correct_mask).sum() > 0:
        plt.hist(confs[~correct_mask], bins=30, alpha=0.7, color="red",
                 label=f"Incorrect (n={(~correct_mask).sum()})")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.title("Confidence distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_report(model_name, ckpt, overall, per_class_df, cm, confs, correct_mask, out_dir):
    top_conf = confs[correct_mask].mean() if correct_mask.any() else 0
    wrong_conf = confs[~correct_mask].mean() if (~correct_mask).any() else 0
    lines = [
        f"# Emotion Model - Stage 4: Evaluation Report ({model_name})\n",
        f"- Checkpoint: `{os.path.relpath(ckpt)}`",
        f"- Test images: {overall['Total']}\n",
        "## Overall metrics\n",
        "| Metric | Value |",
        "|---|---|",
        f"| Accuracy | {overall['Accuracy']*100:.2f}% |",
        f"| Balanced accuracy | {overall['Balanced accuracy']*100:.2f}% |",
        f"| Macro-F1 | {overall['Macro-F1']*100:.2f}% |",
        f"| Weighted-F1 | {overall['Weighted-F1']*100:.2f}% |",
        f"| Macro precision | {overall['Macro precision']*100:.2f}% |",
        f"| Macro recall | {overall['Macro recall']*100:.2f}% |\n",
        "## Per-class metrics\n",
        per_class_df.assign(
            Precision=lambda d: (d["Precision"] * 100).round(2),
            Recall=lambda d: (d["Recall"] * 100).round(2),
            **{"F1-Score": lambda d: (d["F1-Score"] * 100).round(2)},
        ).to_markdown(index=False),
        "",
        "\n## Confidence\n",
        f"- Mean confidence on correct predictions: {top_conf:.3f}",
        f"- Mean confidence on incorrect predictions: {wrong_conf:.3f}",
        f"- A well-calibrated model is more confident when correct.\n",
        "## Figures\n",
        "- `confusion_matrix.png`",
        "- `per_class_metrics.png`",
        "- `confidence_distribution.png`\n",
        "## Notes\n",
        "- Minority emotions (Fear, Disgust, Anger) are the hardest; watch their "
        "recall in the per-class table and the off-diagonal mass in the confusion "
        "matrix (Fear is often confused with Surprise).",
    ]
    with open(os.path.join(out_dir, "EVALUATION_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(ALL_MODELS.keys()))
    ap.add_argument("--checkpoint", default=None,
                    help="Defaults to checkpoints/best_<Model>.pth")
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = ap.parse_args()

    device = get_device()
    sname = safe_name(args.model)
    ckpt = args.checkpoint or os.path.join(CHECKPOINT_DIR, f"best_{sname}.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\nTrain first: "
            f"python scripts/train.py --model \"{args.model}\"")

    model = ALL_MODELS[args.model]()
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.to(device).eval()
    print(f"Loaded {args.model} from {ckpt}")

    test_ds = ImageFolder(TEST_DIR, transform=get_test_transforms())
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_ds)}")

    labels, preds, confs = collect_predictions(model, loader, device)
    correct_mask = preds == labels

    overall = {
        "Accuracy": accuracy_score(labels, preds),
        "Balanced accuracy": balanced_accuracy_score(labels, preds),
        "Macro-F1": f1_score(labels, preds, average="macro"),
        "Weighted-F1": f1_score(labels, preds, average="weighted"),
        "Macro precision": precision_score(labels, preds, average="macro", zero_division=0),
        "Macro recall": recall_score(labels, preds, average="macro", zero_division=0),
        "Total": len(labels),
        "Correct": int(correct_mask.sum()),
    }
    print("\n=== Overall ===")
    for k, v in overall.items():
        print(f"  {k}: {v*100:.2f}%" if isinstance(v, float) else f"  {k}: {v}")

    report = classification_report(labels, preds, target_names=EMOTION_LABELS,
                                   digits=4, output_dict=True, zero_division=0)
    per_class_rows = []
    for name in EMOTION_LABELS:
        m = report[name]
        per_class_rows.append({
            "Emotion": name, "Precision": m["precision"], "Recall": m["recall"],
            "F1-Score": m["f1-score"], "Support": int(m["support"]),
        })
    per_class_df = pd.DataFrame(per_class_rows)
    print("\n=== Per-class ===")
    print(per_class_df.to_string(index=False))

    cm = confusion_matrix(labels, preds)

    # ── Save everything ──────────────────────────────────────────────────
    out_dir = os.path.join(REPORT_DIR, "evaluation", sname)
    os.makedirs(out_dir, exist_ok=True)

    per_class_df.to_csv(os.path.join(out_dir, "per_class_metrics.csv"), index=False)
    pd.DataFrame([{k: v for k, v in overall.items()}]).to_csv(
        os.path.join(out_dir, "overall_metrics.csv"), index=False)
    pd.DataFrame(cm, index=EMOTION_LABELS, columns=EMOTION_LABELS).to_csv(
        os.path.join(out_dir, "confusion_matrix.csv"))

    # misclassifications (highest confidence errors first)
    mis = [{"true": EMOTION_LABELS[t], "predicted": EMOTION_LABELS[p], "confidence": float(c)}
           for t, p, c in zip(labels, preds, confs) if t != p]
    mis.sort(key=lambda d: d["confidence"], reverse=True)
    pd.DataFrame(mis).to_csv(os.path.join(out_dir, "misclassifications.csv"), index=False)

    plot_confusion(cm, os.path.join(out_dir, "confusion_matrix.png"))
    plot_per_class(per_class_df, os.path.join(out_dir, "per_class_metrics.png"))
    plot_confidence(confs, correct_mask, os.path.join(out_dir, "confidence_distribution.png"))
    write_report(args.model, ckpt, overall, per_class_df, cm, confs, correct_mask, out_dir)

    print(f"\nEvaluation complete. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
