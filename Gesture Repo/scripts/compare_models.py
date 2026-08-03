"""
Stage 1 - Compare the three temporal architectures on identical data.

Trains BiGRU / TCN / TinyTransformer with the shared recipe (shortened
schedule), then writes a comparison table + validation curves so the winner
is picked by numbers, mirroring the emotion pipeline.

    python scripts/compare_models.py
    python scripts/compare_models.py --epochs 40 --models BiGRU TCN
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
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import BATCH_SIZE, NUM_WORKERS, REPORT_DIR, SEED
from src.data import compute_class_weights, get_datasets, load_index
from src.models import ALL_MODELS, count_params, model_size_mb
from src.training import fit, get_device, set_seed

OUT_DIR = os.path.join(REPORT_DIR, "comparison")


def plot_curves(histories, path):
    plt.figure(figsize=(8, 5))
    for name, hist in histories.items():
        plt.plot([h["epoch"] for h in hist], [h["f1_macro"] * 100 for h in hist],
                 marker="o", markersize=3, label=name)
    plt.xlabel("epoch")
    plt.ylabel("val macro-F1 (%)")
    plt.title("Gesture model comparison - validation macro-F1")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(ALL_MODELS),
                    choices=list(ALL_MODELS))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    args = ap.parse_args()

    device = get_device()
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Device: {device}")

    results, histories = [], {}
    for name in args.models:
        set_seed(SEED)  # identical data order & init conditions per candidate
        train_ds, val_ds = get_datasets(seed=SEED)
        class_weights = compute_class_weights(load_index("train"))
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=(device.type == "cuda"))
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers,
                                pin_memory=(device.type == "cuda"))

        model = ALL_MODELS[name]().to(device)
        print(f"\n== {name} ({count_params(model)/1e3:.0f}k params, "
              f"{model_size_mb(model):.2f} MB) ==")
        t0 = time.time()
        best, hist = fit(model, train_loader, val_loader, device, class_weights,
                         epochs=args.epochs, patience=args.patience)
        results.append({
            "model": name, "params_k": round(count_params(model) / 1e3, 1),
            "size_mb": round(model_size_mb(model), 2),
            "train_min": round((time.time() - t0) / 60, 1),
            **{k: round(v, 4) for k, v in best.items()
               if k in ("accuracy", "balanced_accuracy", "f1_macro", "f1_weighted")},
        })
        histories[name] = hist

    df = pd.DataFrame(results).sort_values("f1_macro", ascending=False)
    df.to_csv(os.path.join(OUT_DIR, "comparison.csv"), index=False)
    plot_curves(histories, os.path.join(OUT_DIR, "comparison_curves.png"))
    with open(os.path.join(OUT_DIR, "histories.json"), "w") as f:
        json.dump(histories, f, indent=2)

    winner = df.iloc[0]["model"]
    lines = [
        "# Gesture Model - Stage 1: Architecture Comparison\n",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M} · shortened schedule "
        f"({args.epochs} epochs max, patience {args.patience}), shared recipe "
        "(class-weighted CE + label smoothing, AdamW, cosine warmup).\n",
        df.to_markdown(index=False), "",
        f"**Winner (val macro-F1): {winner}**\n",
        "- Curves: `comparison_curves.png`\n",
        "## Next step\n",
        "```", f"python scripts/train.py --model {winner}", "```",
    ]
    with open(os.path.join(OUT_DIR, "COMPARISON_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n{df.to_string(index=False)}")
    print(f"\nWinner: {winner}")
    print(f"Report: {OUT_DIR}/COMPARISON_REPORT.md")
    print(f"Next: python scripts/train.py --model {winner}")


if __name__ == "__main__":
    main()
