"""
Stage 4 - Full evaluation of a trained checkpoint on the scene validation set.

Reports overall metrics, per-class precision/recall/F1, a confusion matrix, and
confidence analysis. Saves CSVs, plots and a markdown report under
reports/evaluation/<model>/.

Run:
    python scripts/evaluate.py --model EfficientNet-B0
    python scripts/evaluate.py --model ResNet18 --checkpoint checkpoints/best_ResNet18.pth
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
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
)
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import BATCH_SIZE, CHECKPOINT_DIR, DEFAULT_MODEL, REPORT_DIR, VAL_DIR
from src.engine import get_device
from src.models import ALL_MODELS, safe_name
from src.transforms import get_test_transforms


def collect_predictions(model, loader, device):
    preds, labels, confs = [], [], []
    with torch.no_grad():
        for imgs, y in loader:
            probs = F.softmax(model(imgs.to(device)), dim=1)
            conf, pred = torch.max(probs, 1)
            preds.extend(pred.cpu().numpy()); labels.extend(y.numpy()); confs.extend(conf.cpu().numpy())
    return np.array(labels), np.array(preds), np.array(confs)


def plot_confusion(cm, class_names, path):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion matrix (color = row-normalized, text = counts)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_confidence(confs, correct_mask, path):
    plt.figure(figsize=(9, 5))
    plt.hist(confs[correct_mask], bins=30, alpha=0.7, color="green",
             label=f"Correct (n={correct_mask.sum()})")
    if (~correct_mask).sum() > 0:
        plt.hist(confs[~correct_mask], bins=30, alpha=0.7, color="red",
                 label=f"Incorrect (n={(~correct_mask).sum()})")
    plt.xlabel("Confidence"); plt.ylabel("Frequency"); plt.title("Confidence distribution")
    plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def write_report(model_name, ckpt, overall, per_class_df, out_dir):
    lines = [
        f"# Scene Model - Stage 4: Evaluation Report ({model_name})\n",
        f"- Checkpoint: `{os.path.relpath(ckpt)}`",
        f"- Validation images: {overall['Total']}\n",
        "## Overall metrics\n", "| Metric | Value |", "|---|---|",
        f"| Accuracy | {overall['Accuracy']*100:.2f}% |",
        f"| Balanced accuracy | {overall['Balanced accuracy']*100:.2f}% |",
        f"| Macro-F1 | {overall['Macro-F1']*100:.2f}% |\n",
        "## Per-class metrics\n",
        per_class_df.assign(
            Precision=lambda d: (d["Precision"] * 100).round(2),
            Recall=lambda d: (d["Recall"] * 100).round(2),
            **{"F1-Score": lambda d: (d["F1-Score"] * 100).round(2)},
        ).to_markdown(index=False),
        "",
        "\n## Figures\n", "- `confusion_matrix.png`", "- `confidence_distribution.png`\n",
        "## Note\n",
        "- Places365 val accuracy can be optimistic vs. captured footage; validate "
        "on real clips with `inference/video.py` or `../inference/video.py --mode scene`.",
    ]
    with open(os.path.join(out_dir, "EVALUATION_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(ALL_MODELS.keys()))
    ap.add_argument("--checkpoint", default=None, help="Defaults to checkpoints/best_<Model>.pth")
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = ap.parse_args()

    device = get_device()
    sname = safe_name(args.model)
    ckpt = args.checkpoint or os.path.join(CHECKPOINT_DIR, f"best_{sname}.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\nTrain first: python scripts/train.py --model \"{args.model}\"")

    test_ds = ImageFolder(VAL_DIR, transform=get_test_transforms())
    class_names = test_ds.classes
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Val samples: {len(test_ds)} | Classes: {class_names}")

    model = ALL_MODELS[args.model](num_classes=len(class_names), pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.to(device).eval()
    print(f"Loaded {args.model} from {ckpt}")

    labels, preds, confs = collect_predictions(model, loader, device)
    correct_mask = preds == labels
    overall = {
        "Accuracy": accuracy_score(labels, preds),
        "Balanced accuracy": balanced_accuracy_score(labels, preds),
        "Macro-F1": f1_score(labels, preds, average="macro"),
        "Total": len(labels), "Correct": int(correct_mask.sum()),
    }
    print("\n=== Overall ===")
    for k, v in overall.items():
        print(f"  {k}: {v*100:.2f}%" if isinstance(v, float) else f"  {k}: {v}")

    report = classification_report(labels, preds, target_names=class_names,
                                   digits=4, output_dict=True, zero_division=0)
    per_class_df = pd.DataFrame([
        {"Scene": n, "Precision": report[n]["precision"], "Recall": report[n]["recall"],
         "F1-Score": report[n]["f1-score"], "Support": int(report[n]["support"])}
        for n in class_names
    ])
    print("\n=== Per-class ===")
    print(per_class_df.to_string(index=False))

    cm = confusion_matrix(labels, preds)
    out_dir = os.path.join(REPORT_DIR, "evaluation", sname)
    os.makedirs(out_dir, exist_ok=True)
    per_class_df.to_csv(os.path.join(out_dir, "per_class_metrics.csv"), index=False)
    pd.DataFrame([overall]).to_csv(os.path.join(out_dir, "overall_metrics.csv"), index=False)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        os.path.join(out_dir, "confusion_matrix.csv"))
    plot_confusion(cm, class_names, os.path.join(out_dir, "confusion_matrix.png"))
    plot_confidence(confs, correct_mask, os.path.join(out_dir, "confidence_distribution.png"))
    write_report(args.model, ckpt, overall, per_class_df, out_dir)
    print(f"\nEvaluation complete. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
