"""
Stage 3 - Hyper-parameter tuning for the chosen model.

Grid search over base learning rate, batch size, optimizer, weight decay and
label smoothing using a shortened two-stage run per trial. Trials are ranked by
**macro-F1**. Writes all results, the best config, a plot, and a report, and
prints the ready-to-run final training command.

Run:
    python scripts/tune.py --model EfficientNet-B0
    python scripts/tune.py --model EfficientNet-B0 --lrs 1e-4 5e-5 --batch_sizes 32 64
"""
import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CHECKPOINT_DIR, DEFAULT_MODEL, NUM_WORKERS, REPORT_DIR
from src.data import compute_class_weights, get_datasets, subset_per_class
from src.engine import fit_two_stage, get_device, set_seed
from src.models import ALL_MODELS
from torch.utils.data import DataLoader

OUT_DIR = os.path.join(REPORT_DIR, "tuning")


def plot_tuning(df, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col, title in [
        (axes[0], "base_lr", "Base learning rate"),
        (axes[1], "batch_size", "Batch size"),
        (axes[2], "optimizer", "Optimizer"),
    ]:
        grp = df.groupby(col)["macro_f1"].max()
        ax.bar([str(k) for k in grp.index], grp.values, color="#3b7dd8")
        ax.set_title(f"Best macro-F1 by {title.lower()}")
        ax.set_ylabel("macro-F1 (%)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_report(df, best, model_name, args, path):
    top = df.sort_values("macro_f1", ascending=False).head(10)
    lines = [
        f"# Emotion Model - Stage 3: Hyper-parameter Tuning Report ({model_name})\n",
        f"Grid search of **{len(df)} configurations**, each a shortened "
        f"{args.stage1_epochs}+{args.stage2_epochs}-epoch two-stage run, ranked by "
        "**macro-F1**.\n",
        "## Search space\n",
        f"- Base LR: {args.lrs}",
        f"- Batch size: {args.batch_sizes}",
        f"- Optimizer: {args.optimizers}",
        f"- Weight decay: {args.weight_decays}",
        f"- Label smoothing: {args.label_smoothings}\n",
        "## Top 10 configurations\n",
        top.to_markdown(index=False),
        "",
        "\n## Best configuration\n",
        f"- Base LR: **{best['base_lr']}**",
        f"- Batch size: **{best['batch_size']}**",
        f"- Optimizer: **{best['optimizer']}**",
        f"- Weight decay: **{best['weight_decay']}**",
        f"- Label smoothing: **{best['label_smoothing']}**",
        f"- Macro-F1: **{best['macro_f1']}%** | Accuracy: {best['accuracy']}%\n",
        "## Final training command\n",
        "```",
        (f"python scripts/train.py --model \"{model_name}\" "
         f"--batch_size {best['batch_size']} --base_lr {best['base_lr']} "
         f"--optimizer {best['optimizer']} --weight_decay {best['weight_decay']} "
         f"--label_smoothing {best['label_smoothing']}"),
        "```",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(ALL_MODELS.keys()))
    ap.add_argument("--lrs", nargs="+", type=float, default=[1e-4, 5e-5, 2e-4])
    ap.add_argument("--batch_sizes", nargs="+", type=int, default=[32, 64])
    ap.add_argument("--optimizers", nargs="+", default=["adam"], choices=["adam", "sgd"])
    ap.add_argument("--weight_decays", nargs="+", type=float, default=[1e-5])
    ap.add_argument("--label_smoothings", nargs="+", type=float, default=[0.1])
    ap.add_argument("--stage1_epochs", type=int, default=2)
    ap.add_argument("--stage2_epochs", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--max_per_class", type=int, default=0,
                    help="If >0, subsample train/val to N per class (fast trials / smoke).")
    ap.add_argument("--no_amp", action="store_true")
    args = ap.parse_args()

    set_seed()
    device = get_device()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_ds, val_ds = get_datasets()
    class_weights = compute_class_weights(train_ds)
    if args.max_per_class > 0:
        train_ds = subset_per_class(train_ds, args.max_per_class)
        val_ds = subset_per_class(val_ds, args.max_per_class)

    grid = list(itertools.product(
        args.lrs, args.batch_sizes, args.optimizers,
        args.weight_decays, args.label_smoothings,
    ))
    print(f"Device: {device} | Model: {args.model} | {len(grid)} configurations")

    rows = []
    for i, (lr, bs, opt, wd, ls) in enumerate(grid, 1):
        print(f"\n[{i}/{len(grid)}] lr={lr} bs={bs} opt={opt} wd={wd} ls={ls}")
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                                num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
        model = ALL_MODELS[args.model]().to(device)
        best, _ = fit_two_stage(
            model, train_loader, val_loader, device, class_weights,
            stage1_epochs=args.stage1_epochs, stage2_epochs=args.stage2_epochs,
            ckpt_path=None, base_lr=lr, weight_decay=wd, label_smoothing=ls,
            optimizer_name=opt, use_amp=not args.no_amp, select_metric="f1_macro",
            verbose=False,
        )
        rows.append({
            "trial": i, "base_lr": lr, "batch_size": bs, "optimizer": opt,
            "weight_decay": wd, "label_smoothing": ls,
            "accuracy": round(best["accuracy"] * 100, 2),
            "balanced_acc": round(best["balanced_accuracy"] * 100, 2),
            "macro_f1": round(best["f1_macro"] * 100, 2),
            "f1_weighted": round(best["f1_weighted"] * 100, 2),
        })
        print(f"  -> macro-F1 {rows[-1]['macro_f1']}%  acc {rows[-1]['accuracy']}%")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "tuning_results.csv"), index=False)
    best = df.sort_values("macro_f1", ascending=False).iloc[0].to_dict()
    with open(os.path.join(OUT_DIR, "best_config.json"), "w") as f:
        json.dump(best, f, indent=2)
    plot_tuning(df, os.path.join(OUT_DIR, "tuning.png"))
    write_report(df, best, args.model, args, os.path.join(OUT_DIR, "TUNING_REPORT.md"))

    print(f"\n{'='*70}\nBest config (macro-F1 {best['macro_f1']}%):")
    print(f"  lr={best['base_lr']} bs={best['batch_size']} opt={best['optimizer']} "
          f"wd={best['weight_decay']} ls={best['label_smoothing']}")
    print(f"\nFinal training command:")
    print(f"  python scripts/train.py --model \"{args.model}\" "
          f"--batch_size {best['batch_size']} --base_lr {best['base_lr']} "
          f"--optimizer {best['optimizer']} --weight_decay {best['weight_decay']} "
          f"--label_smoothing {best['label_smoothing']}")
    print(f"Reports written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
