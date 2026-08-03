"""
Stage 2 - Full training of the chosen model.

Saves the best checkpoint (by val macro-F1), the epoch history, training
curves, a markdown report, and checkpoints/model_config.json — the contract
inference (GestureEngine) loads so deployment can't drift from training.

    python scripts/train.py --model BiGRU
    python scripts/train.py --model BiGRU --use-tuned      # params from Stage 3
"""
import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    CONF_THRESHOLD,
    DEFAULT_MODEL,
    DROPOUT,
    EPOCHS,
    FEATURE_DIM,
    GESTURE_LABELS,
    HIDDEN_SIZE,
    LABEL_SMOOTHING,
    LR,
    NUM_WORKERS,
    PATIENCE,
    REPORT_DIR,
    SEED,
    WEIGHT_DECAY,
    WINDOW,
)
from src.data import compute_class_weights, get_datasets, load_index
from src.models import ALL_MODELS, count_params, model_size_mb, safe_name
from src.training import fit, get_device, set_seed

OUT_DIR = os.path.join(REPORT_DIR, "training")
TUNED_PARAMS = os.path.join(REPORT_DIR, "tuning", "best_params.json")


def plot_curves(history, model_name, path):
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(df["epoch"], df["loss"], marker="o", markersize=3, label="train loss")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].set_title(f"{model_name} - training loss"); axes[0].legend()
    axes[1].plot(df["epoch"], df["accuracy"] * 100, marker="o", markersize=3, label="val accuracy")
    axes[1].plot(df["epoch"], df["f1_macro"] * 100, marker="s", markersize=3, label="val macro-F1")
    axes[1].plot(df["epoch"], df["balanced_accuracy"] * 100, marker="^", markersize=3, label="val balanced acc")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("%")
    axes[1].set_title(f"{model_name} - validation metrics"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_report(model_name, best, args, ckpt_path, size, path):
    lines = [
        f"# Gesture Model - Stage 2: Training Report ({model_name})\n",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M}. Recipe: class-weighted "
        "CrossEntropy + label smoothing, AdamW, cosine LR with warmup, "
        "early stopping on val macro-F1.\n",
        "## Configuration\n",
        f"- Model: {model_name} ({size:.2f} MB)",
        f"- Epochs: {args.epochs} (patience {args.patience}) | Batch size: {args.batch_size}",
        f"- LR: {args.lr} | Weight decay: {args.weight_decay} | Dropout: {args.dropout}",
        f"- Hidden size: {args.hidden_size} | Label smoothing: {args.label_smoothing}",
        f"- Window: {WINDOW} frames | Feature dim: {FEATURE_DIM}\n",
        "## Best validation metrics\n",
        "| Metric | Value |",
        "|---|---|",
        f"| Accuracy | {best['accuracy']*100:.2f}% |",
        f"| Balanced accuracy | {best['balanced_accuracy']*100:.2f}% |",
        f"| Macro-F1 | {best['f1_macro']*100:.2f}% |",
        f"| Weighted-F1 | {best['f1_weighted']*100:.2f}% |",
        f"| Best epoch | {best['epoch']} |\n",
        f"- Checkpoint: `{os.path.relpath(ckpt_path)}`",
        "- Training curves: `training_curves.png`\n",
        "## Next step\n",
        "```", f"python scripts/evaluate.py --model {model_name}", "```",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(ALL_MODELS))
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    ap.add_argument("--dropout", type=float, default=DROPOUT)
    ap.add_argument("--hidden_size", type=int, default=HIDDEN_SIZE)
    ap.add_argument("--label_smoothing", type=float, default=LABEL_SMOOTHING)
    ap.add_argument("--use-tuned", action="store_true",
                    help=f"load best params from {os.path.relpath(TUNED_PARAMS)}")
    args = ap.parse_args()

    if args.use_tuned:
        with open(TUNED_PARAMS) as f:
            tuned = json.load(f)
        for k, v in tuned.items():
            if hasattr(args, k):
                setattr(args, k, v)
        print(f"Loaded tuned params: {tuned}")

    set_seed(SEED)
    device = get_device()
    print(f"Device: {device} | Model: {args.model}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    train_ds, val_ds = get_datasets(seed=SEED)
    class_weights = compute_class_weights(load_index("train"))
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"Class weights: {class_weights.numpy().round(3).tolist()}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=(device.type == "cuda"))

    model_kwargs = {"hidden_size": args.hidden_size, "dropout": args.dropout}
    model = ALL_MODELS[args.model](**model_kwargs).to(device)
    size = model_size_mb(model)
    sname = safe_name(args.model)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_{sname}.pth")

    best, history = fit(model, train_loader, val_loader, device, class_weights,
                        epochs=args.epochs, ckpt_path=ckpt_path, lr=args.lr,
                        weight_decay=args.weight_decay,
                        label_smoothing=args.label_smoothing,
                        patience=args.patience)

    # model_config.json — everything GestureEngine needs to rebuild the model
    model_config = {
        "model": args.model,
        "model_kwargs": model_kwargs,
        "labels": GESTURE_LABELS,
        "window": WINDOW,
        "feature_dim": FEATURE_DIM,
        "conf_threshold": CONF_THRESHOLD,
        "params": count_params(model),
        "checkpoint": os.path.basename(ckpt_path),
        "trained": f"{datetime.now():%Y-%m-%d}",
        "best_val": {k: round(v, 4) for k, v in best.items()},
        "train_config": vars(args),
    }
    with open(os.path.join(CHECKPOINT_DIR, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=2)

    with open(os.path.join(CHECKPOINT_DIR, f"history_{sname}.json"), "w") as f:
        json.dump({"model": args.model, "best": best, "history": history}, f, indent=2)
    plot_curves(history, args.model, os.path.join(OUT_DIR, "training_curves.png"))
    write_report(args.model, best, args, ckpt_path, size,
                 os.path.join(OUT_DIR, "TRAINING_REPORT.md"))

    print(f"\nBest macro-F1: {best['f1_macro']*100:.2f}%  acc: {best['accuracy']*100:.2f}%")
    print(f"Checkpoint:   {ckpt_path}")
    print(f"Model config: {os.path.join(CHECKPOINT_DIR, 'model_config.json')}")
    print(f"Report:       {OUT_DIR}/TRAINING_REPORT.md")
    print(f"Next: python scripts/evaluate.py --model {args.model}")


if __name__ == "__main__":
    main()
