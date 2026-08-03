"""
Stage 2 - Full training of the chosen model with the balanced two-stage recipe.

Saves the best checkpoint (by macro-F1), the full epoch history, training
curves, and a short markdown report. Accepts hyper-parameters on the command
line so the best config from Stage 3 (tuning) can be plugged straight in.

Run:
    python scripts/train.py --model EfficientNet-B0
    python scripts/train.py --model MobileNetV2 --batch_size 64 --base_lr 5e-5
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DEFAULT_MODEL,
    HEAD_LR,
    LABEL_SMOOTHING,
    LR,
    MIXUP_ALPHA,
    NUM_WORKERS,
    REPORT_DIR,
    STAGE1_EPOCHS,
    STAGE2_EPOCHS,
    WEIGHT_DECAY,
)
from src.data import compute_class_weights, get_datasets
from src.engine import fit_two_stage, get_device, set_seed
from src.models import ALL_MODELS, model_size_mb, safe_name
from torch.utils.data import DataLoader

OUT_DIR = os.path.join(REPORT_DIR, "training")


def plot_curves(history, model_name, path):
    df = pd.DataFrame(history)
    df["x"] = range(1, len(df) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(df["x"], df["loss"], marker="o", label="train loss")
    axes[0].set_xlabel("epoch (S1 then S2)")
    axes[0].set_ylabel("loss")
    axes[0].set_title(f"{model_name} - training loss")
    axes[0].legend()

    axes[1].plot(df["x"], df["accuracy"] * 100, marker="o", label="val accuracy")
    axes[1].plot(df["x"], df["f1_macro"] * 100, marker="s", label="val macro-F1")
    axes[1].plot(df["x"], df["balanced_accuracy"] * 100, marker="^", label="val balanced acc")
    axes[1].set_xlabel("epoch (S1 then S2)")
    axes[1].set_ylabel("%")
    axes[1].set_title(f"{model_name} - validation metrics")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_report(model_name, best, args, ckpt_path, size, path):
    lines = [
        f"# Emotion Model - Stage 2: Training Report ({model_name})\n",
        "Trained with the balanced two-stage recipe (weighted CrossEntropy + "
        "label smoothing + mixup + cosine LR with warmup + AMP).\n",
        "## Configuration\n",
        f"- Model: {model_name} ({size:.1f} MB)",
        f"- Epochs: {args.stage1_epochs} head-only + {args.stage2_epochs} full fine-tune",
        f"- Batch size: {args.batch_size}",
        f"- Head LR: {args.head_lr} | Base LR: {args.base_lr} | Weight decay: {args.weight_decay}",
        f"- Label smoothing: {args.label_smoothing} | Mixup alpha: {args.mixup_alpha}",
        f"- Optimizer: {args.optimizer}\n",
        "## Best validation metrics\n",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Accuracy | {best['accuracy']*100:.2f}% |",
        f"| Balanced accuracy | {best['balanced_accuracy']*100:.2f}% |",
        f"| Macro-F1 | {best['f1_macro']*100:.2f}% |",
        f"| Weighted-F1 | {best['f1_weighted']*100:.2f}% |\n",
        f"- Checkpoint: `{os.path.relpath(ckpt_path)}`",
        "- Training curves: `training_curves.png`\n",
        "## Next step\n",
        "```",
        f"python scripts/evaluate.py --model \"{model_name}\"",
        "```",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(ALL_MODELS.keys()))
    ap.add_argument("--stage1_epochs", type=int, default=STAGE1_EPOCHS)
    ap.add_argument("--stage2_epochs", type=int, default=STAGE2_EPOCHS)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--head_lr", type=float, default=HEAD_LR)
    ap.add_argument("--base_lr", type=float, default=LR)
    ap.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    ap.add_argument("--label_smoothing", type=float, default=LABEL_SMOOTHING)
    ap.add_argument("--mixup_alpha", type=float, default=MIXUP_ALPHA)
    ap.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])
    ap.add_argument("--no_amp", action="store_true")
    args = ap.parse_args()

    set_seed()
    device = get_device()
    print(f"Device: {device} | Model: {args.model}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    train_ds, val_ds = get_datasets()
    class_weights = compute_class_weights(train_ds)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"Class weights: {class_weights.numpy().round(3).tolist()}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    model = ALL_MODELS[args.model]().to(device)
    size = model_size_mb(model)
    sname = safe_name(args.model)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_{sname}.pth")

    best, history = fit_two_stage(
        model, train_loader, val_loader, device, class_weights,
        stage1_epochs=args.stage1_epochs, stage2_epochs=args.stage2_epochs,
        ckpt_path=ckpt_path, head_lr=args.head_lr, base_lr=args.base_lr,
        weight_decay=args.weight_decay, label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha, optimizer_name=args.optimizer,
        use_amp=not args.no_amp, select_metric="f1_macro",
    )

    # Persist history + report
    hist_path = os.path.join(CHECKPOINT_DIR, f"history_{sname}.json")
    with open(hist_path, "w") as f:
        json.dump({"model": args.model, "best": best,
                   "checkpoint": ckpt_path, "config": vars(args),
                   "history": history}, f, indent=2)
    plot_curves(history, args.model, os.path.join(OUT_DIR, "training_curves.png"))
    write_report(args.model, best, args, ckpt_path, size,
                 os.path.join(OUT_DIR, "TRAINING_REPORT.md"))

    print(f"\nBest macro-F1: {best['f1_macro']*100:.2f}%  acc: {best['accuracy']*100:.2f}%")
    print(f"Checkpoint: {ckpt_path}")
    print(f"History:    {hist_path}")
    print(f"Report:     {OUT_DIR}/TRAINING_REPORT.md")
    print(f"Next: python scripts/evaluate.py --model \"{args.model}\"")


if __name__ == "__main__":
    main()
