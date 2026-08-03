"""
Stage 1 - Compare all candidate backbones with the balanced two-stage recipe
and pick the most suitable one for deployment.

For each model in the zoo it measures: parameter count, size (MB), best
accuracy / macro-F1 / balanced accuracy, CPU+GPU inference latency, and train
time. The winner is the highest **macro-F1** model within the size budget.

Outputs:
  reports/comparison/comparison.csv
  reports/comparison/comparison.json
  reports/comparison/comparison.png
  reports/comparison/COMPARISON_REPORT.md
  checkpoints/compare_<Model>.pth          (best of each model during search)

Run (defaults are a quick but real search; raise epochs for the final report):
    python scripts/compare_models.py --stage1_epochs 3 --stage2_epochs 12
    python scripts/compare_models.py --models EfficientNet-B0 MobileNetV2
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    NUM_WORKERS,
    REPORT_DIR,
    SIZE_BUDGET_MB,
)
from src.data import compute_class_weights, get_dataloaders, get_datasets, subset_per_class
from src.engine import evaluate, fit_two_stage, get_device, set_seed
from src.models import ALL_MODELS, count_params, model_size_mb, safe_name
from torch.utils.data import DataLoader

OUT_DIR = os.path.join(REPORT_DIR, "comparison")


@torch.no_grad()
def measure_latency(model, device, n=60, warmup=10):
    """Mean inference latency (ms) for a single 224x224 image. Moves model to device."""
    model.to(device).eval()
    x = torch.randn(1, 3, 224, 224, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000.0


def plot_comparison(df, path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    order = df.sort_values("macro_f1", ascending=False)

    axes[0, 0].bar(order["model"], order["macro_f1"], color="#3b7dd8")
    axes[0, 0].set_title("Macro-F1 (selection metric)")
    axes[0, 0].set_ylabel("%")
    axes[0, 0].tick_params(axis="x", rotation=25)

    axes[0, 1].scatter(df["size_mb"], df["macro_f1"], s=60, color="#e07a3b")
    for _, r in df.iterrows():
        axes[0, 1].annotate(r["model"], (r["size_mb"], r["macro_f1"]), fontsize=8)
    axes[0, 1].axvline(SIZE_BUDGET_MB, ls="--", color="red", label=f"{SIZE_BUDGET_MB:.0f} MB budget")
    axes[0, 1].set_xlabel("size (MB)")
    axes[0, 1].set_ylabel("macro-F1 (%)")
    axes[0, 1].set_title("Accuracy vs size")
    axes[0, 1].legend()

    axes[1, 0].bar(order["model"], order["gpu_ms"], color="#5aa469")
    axes[1, 0].set_title("GPU inference latency")
    axes[1, 0].set_ylabel("ms / image")
    axes[1, 0].tick_params(axis="x", rotation=25)

    axes[1, 1].bar(order["model"], order["params_m"], color="#9b6dd8")
    axes[1, 1].set_title("Parameters")
    axes[1, 1].set_ylabel("millions")
    axes[1, 1].tick_params(axis="x", rotation=25)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_report(df, winner, args, path):
    order = df.sort_values("macro_f1", ascending=False)
    lines = [
        "# Emotion Model - Stage 1: Model Comparison Report\n",
        "Candidate ImageNet-pretrained backbones trained with the identical "
        "balanced two-stage recipe (weighted CE + label smoothing + mixup + "
        "cosine warmup) and ranked by **macro-F1** within a "
        f"**{SIZE_BUDGET_MB:.0f} MB** deployment budget.\n",
        f"- Search length: {args.stage1_epochs} head-only + {args.stage2_epochs} "
        f"full-finetune epochs/model",
        f"- Batch size: {args.batch_size}",
        "- Selection metric: macro-F1 (every emotion weighted equally)\n",
        "## Results\n",
        order.to_markdown(index=False),
        "",
        f"\n## Recommendation: **{winner}**\n",
        f"- Best macro-F1 among models within the {SIZE_BUDGET_MB:.0f} MB budget.",
        f"- Size {df.set_index('model').loc[winner, 'size_mb']} MB, "
        f"macro-F1 {df.set_index('model').loc[winner, 'macro_f1']}%, "
        f"accuracy {df.set_index('model').loc[winner, 'accuracy']}%.\n",
        "## Next step\n",
        "```",
        f"python scripts/train.py --model \"{winner}\"",
        "```",
        "",
        "> Note: these numbers come from a shortened search. The final reported "
        "accuracy comes from the full training run (Stage 2).",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(ALL_MODELS.keys()),
                    choices=list(ALL_MODELS.keys()))
    ap.add_argument("--stage1_epochs", type=int, default=3)
    ap.add_argument("--stage2_epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--max_per_class", type=int, default=0,
                    help="If >0, subsample train set to N per class (smoke test).")
    ap.add_argument("--no_amp", action="store_true")
    args = ap.parse_args()

    set_seed()
    device = get_device()
    print(f"Device: {device}")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_ds, val_ds = get_datasets()
    class_weights = compute_class_weights(train_ds)
    if args.max_per_class > 0:
        train_ds = subset_per_class(train_ds, args.max_per_class)
        val_ds = subset_per_class(val_ds, args.max_per_class)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    rows = []
    for name in args.models:
        print(f"\n{'='*70}\n  {name}\n{'='*70}")
        model = ALL_MODELS[name]().to(device)
        size = model_size_mb(model)
        params = count_params(model)
        print(f"  Params: {params:,}  Size: {size:.1f} MB"
              f"{'  *** over budget ***' if size > SIZE_BUDGET_MB else ''}")

        ckpt = os.path.join(CHECKPOINT_DIR, f"compare_{safe_name(name)}.pth")
        t0 = time.time()
        best, _ = fit_two_stage(
            model, train_loader, val_loader, device, class_weights,
            stage1_epochs=args.stage1_epochs, stage2_epochs=args.stage2_epochs,
            ckpt_path=ckpt, use_amp=not args.no_amp, select_metric="f1_macro",
        )
        train_time = time.time() - t0
        gpu_ms = measure_latency(model, device)
        cpu_ms = measure_latency(model, torch.device("cpu")) if device.type == "cuda" else gpu_ms

        rows.append({
            "model": name,
            "params_m": round(params / 1e6, 2),
            "size_mb": round(size, 2),
            "within_budget": size <= SIZE_BUDGET_MB,
            "accuracy": round(best["accuracy"] * 100, 2),
            "balanced_acc": round(best["balanced_accuracy"] * 100, 2),
            "f1_weighted": round(best["f1_weighted"] * 100, 2),
            "macro_f1": round(best["f1_macro"] * 100, 2),
            "gpu_ms": round(gpu_ms, 2),
            "cpu_ms": round(cpu_ms, 2),
            "train_time_s": round(train_time, 1),
        })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "comparison.csv"), index=False)
    with open(os.path.join(OUT_DIR, "comparison.json"), "w") as f:
        json.dump(rows, f, indent=2)
    plot_comparison(df, os.path.join(OUT_DIR, "comparison.png"))

    eligible = df[df["within_budget"]]
    pool = eligible if len(eligible) else df
    winner = pool.sort_values("macro_f1", ascending=False).iloc[0]["model"]
    write_report(df, winner, args, os.path.join(OUT_DIR, "COMPARISON_REPORT.md"))

    print(f"\n{'='*70}\nSUMMARY (sorted by macro-F1)\n{'='*70}")
    print(df.sort_values("macro_f1", ascending=False).to_string(index=False))
    print(f"\nRecommended model: {winner}")
    print(f"Next: python scripts/train.py --model \"{winner}\"")
    print(f"Reports written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
