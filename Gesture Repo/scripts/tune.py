"""
Stage 3 - Hyper-parameter tuning with Optuna (objective: val macro-F1).

Searches lr, weight decay, dropout, hidden size, label smoothing and batch
size on a shortened schedule. Writes reports/tuning/best_params.json (which
`train.py --use-tuned` consumes) and a markdown report of the top trials.

    python scripts/tune.py --model BiGRU --trials 40
"""
import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna
import pandas as pd
from torch.utils.data import DataLoader

from config import DEFAULT_MODEL, NUM_WORKERS, REPORT_DIR, SEED
from src.data import compute_class_weights, get_datasets, load_index
from src.models import ALL_MODELS
from src.training import fit, get_device, set_seed

OUT_DIR = os.path.join(REPORT_DIR, "tuning")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(ALL_MODELS))
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=25, help="per-trial max epochs")
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    args = ap.parse_args()

    device = get_device()
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Device: {device} | Model: {args.model} | Trials: {args.trials}")

    class_weights = compute_class_weights(load_index("train"))

    def objective(trial):
        params = {
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 192]),
            "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256]),
        }
        set_seed(SEED)
        train_ds, val_ds = get_datasets(seed=SEED)
        train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=(device.type == "cuda"))
        val_loader = DataLoader(val_ds, batch_size=params["batch_size"], shuffle=False,
                                num_workers=args.num_workers,
                                pin_memory=(device.type == "cuda"))
        model = ALL_MODELS[args.model](
            hidden_size=params["hidden_size"], dropout=params["dropout"]).to(device)
        best, _ = fit(model, train_loader, val_loader, device, class_weights,
                      epochs=args.epochs, lr=params["lr"],
                      weight_decay=params["weight_decay"],
                      label_smoothing=params["label_smoothing"],
                      patience=args.patience, verbose=False)
        return best.get("f1_macro", 0.0)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    best_params = dict(study.best_params)
    with open(os.path.join(OUT_DIR, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)

    rows = [{"trial": t.number, "f1_macro": round(t.value or 0.0, 4), **t.params}
            for t in study.trials if t.value is not None]
    df = pd.DataFrame(rows).sort_values("f1_macro", ascending=False).head(10)
    lines = [
        f"# Gesture Model - Stage 3: Tuning Report ({args.model})\n",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M} · {len(study.trials)} trials, "
        f"{args.epochs} epochs max each (patience {args.patience}), "
        "objective: val macro-F1.\n",
        f"**Best macro-F1: {study.best_value*100:.2f}%**\n",
        "## Best parameters\n",
        "```json", json.dumps(best_params, indent=2), "```\n",
        "## Top 10 trials\n",
        df.to_markdown(index=False), "",
        "## Next step\n",
        "```", f"python scripts/train.py --model {args.model} --use-tuned", "```",
    ]
    with open(os.path.join(OUT_DIR, "TUNING_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nBest macro-F1: {study.best_value*100:.2f}%")
    print(f"Best params:   {best_params}")
    print(f"Report: {OUT_DIR}/TUNING_REPORT.md")
    print(f"Next: python scripts/train.py --model {args.model} --use-tuned")


if __name__ == "__main__":
    main()
