"""
Shared training engine for the gesture pipeline (guide §7): single-stage
recipe with class-weighted CrossEntropy + label smoothing, AdamW, cosine LR
with linear warmup, gradient clipping, and early stopping on val macro-F1
(class counts are heavily imbalanced — Jester thumbs >> custom beckoning).
Used by compare_models.py, train.py and tune.py.
"""
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import LABEL_SMOOTHING, LR, PATIENCE, SEED, WARMUP_EPOCHS, WEIGHT_DECAY


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cosine_warmup(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def train_one_epoch(model, loader, criterion, optimizer, device,
                    scheduler=None, desc="train"):
    model.train()
    total_loss, preds, targets = 0.0, [], []
    pbar = tqdm(loader, desc=desc, leave=False)
    for seqs, labels in pbar:
        seqs = seqs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = model(seqs)
        loss = criterion(out, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        preds.extend(out.argmax(1).detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(1, len(loader)), accuracy_score(targets, preds)


@torch.no_grad()
def evaluate(model, loader, device, return_preds=False):
    """Metrics dict (accuracy, balanced acc, macro/weighted F1); optionally raw preds."""
    model.eval()
    preds, targets = [], []
    for seqs, labels in tqdm(loader, desc="val", leave=False):
        preds.extend(model(seqs.to(device, non_blocking=True)).argmax(1).cpu().tolist())
        targets.extend(labels.tolist())
    metrics = {
        "accuracy": accuracy_score(targets, preds),
        "balanced_accuracy": balanced_accuracy_score(targets, preds),
        "f1_weighted": f1_score(targets, preds, average="weighted", zero_division=0),
        "f1_macro": f1_score(targets, preds, average="macro", zero_division=0),
    }
    if return_preds:
        return metrics, np.array(preds), np.array(targets)
    return metrics


def fit(model, train_loader, val_loader, device, class_weights, epochs,
        ckpt_path=None, lr=LR, weight_decay=WEIGHT_DECAY,
        label_smoothing=LABEL_SMOOTHING, warmup_epochs=WARMUP_EPOCHS,
        patience=PATIENCE, select_metric="f1_macro", verbose=True):
    """
    Full recipe with early stopping. Saves the best checkpoint (by
    ``select_metric``) to ``ckpt_path`` if given. Returns (best_metrics, history).
    """
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device),
                                    label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: _cosine_warmup(s, warmup_steps, total_steps))

    best_score, best_metrics, history, since_best = -1.0, {}, [], 0
    for ep in range(1, epochs + 1):
        loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer,
                                       device, scheduler, desc=f"ep {ep}/{epochs}")
        m = evaluate(model, val_loader, device)
        m.update(epoch=ep, loss=loss, train_acc=tr_acc)
        history.append(m)
        if verbose:
            print(f"  {ep:3d}/{epochs}  loss={loss:.4f}  train_acc={tr_acc*100:.2f}%  "
                  f"val_acc={m['accuracy']*100:.2f}%  macroF1={m['f1_macro']*100:.2f}%  "
                  f"balAcc={m['balanced_accuracy']*100:.2f}%")
        if m[select_metric] > best_score:
            best_score, best_metrics, since_best = m[select_metric], dict(m), 0
            if ckpt_path is not None:
                torch.save(model.state_dict(), ckpt_path)
        else:
            since_best += 1
            if patience and since_best >= patience:
                if verbose:
                    print(f"  early stop at epoch {ep} "
                          f"(no {select_metric} gain for {patience} epochs)")
                break

    if verbose and best_metrics:
        print(f"\nBest {select_metric}: {best_score*100:.2f}%  "
              f"(val_acc={best_metrics['accuracy']*100:.2f}%)")
    return best_metrics, history
