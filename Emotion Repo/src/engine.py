"""
Shared training engine: the balanced two-stage fine-tuning recipe used by both
model comparison and full training, plus metric helpers, mixup, AMP and seeding.

Recipe (optimized for macro-F1 on imbalanced RAF-DB):
  Stage 1 - train classifier head only (frozen backbone), inverse-frequency
            weighted CrossEntropy, Adam @ HEAD_LR.
  Stage 2 - unfreeze, full fine-tune with weighted CE + label smoothing,
            mixup (alpha), gradient clipping, cosine LR with linear warmup,
            and automatic mixed precision (AMP) on CUDA.
Selection metric is macro-F1 (treats every emotion equally).
"""
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from tqdm import tqdm

from config import (
    HEAD_LR,
    LABEL_SMOOTHING,
    LR,
    MIXUP_ALPHA,
    SEED,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── mixup ────────────────────────────────────────────────────────────────
def _mixup(x, y, alpha, device):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0)).to(device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def _mixup_loss(criterion, pred, ya, yb, lam):
    return lam * criterion(pred, ya) + (1 - lam) * criterion(pred, yb)


def _cosine_warmup(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


# ── epoch loops ──────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device,
                    scheduler=None, use_mixup=False, alpha=MIXUP_ALPHA,
                    scaler=None, desc="train"):
    model.train()
    total_loss, preds, targets = 0.0, [], []
    pbar = tqdm(loader, desc=desc, leave=False)
    amp_enabled = scaler is not None and device.type == "cuda"
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=amp_enabled):
            if use_mixup and np.random.rand() < 0.5:
                mixed, ya, yb, lam = _mixup(imgs, labels, alpha, device)
                out = model(mixed)
                loss = _mixup_loss(criterion, out, ya, yb, lam)
                batch_targets = ya
            else:
                out = model(imgs)
                loss = criterion(out, labels)
                batch_targets = labels
        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        preds.extend(out.argmax(1).detach().cpu().tolist())
        targets.extend(batch_targets.detach().cpu().tolist())
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(1, len(loader)), accuracy_score(targets, preds)


@torch.no_grad()
def evaluate(model, loader, device):
    """Return dict of accuracy, balanced_accuracy, weighted-F1, macro-F1."""
    model.eval()
    preds, targets = [], []
    for imgs, labels in tqdm(loader, desc="val", leave=False):
        preds.extend(model(imgs.to(device, non_blocking=True)).argmax(1).cpu().tolist())
        targets.extend(labels.tolist())
    return {
        "accuracy": accuracy_score(targets, preds),
        "balanced_accuracy": balanced_accuracy_score(targets, preds),
        "f1_weighted": f1_score(targets, preds, average="weighted"),
        "f1_macro": f1_score(targets, preds, average="macro"),
    }


# ── two-stage fit ────────────────────────────────────────────────────────
def fit_two_stage(model, train_loader, val_loader, device, class_weights,
                  stage1_epochs, stage2_epochs, ckpt_path=None,
                  head_lr=HEAD_LR, base_lr=LR, weight_decay=WEIGHT_DECAY,
                  label_smoothing=LABEL_SMOOTHING, mixup_alpha=MIXUP_ALPHA,
                  warmup_epochs=WARMUP_EPOCHS, optimizer_name="adam",
                  use_amp=True, select_metric="f1_macro", verbose=True):
    """
    Run the full two-stage recipe. Saves the best checkpoint (by ``select_metric``)
    to ``ckpt_path`` if given. Returns (best_metrics, history).
    """
    class_weights = class_weights.to(device)
    scaler = torch.amp.GradScaler("cuda") if (use_amp and device.type == "cuda") else None
    best_score, best_metrics, history = -1.0, {}, []

    def _make_opt(params, lr):
        if optimizer_name.lower() == "sgd":
            return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)

    def _log(stage, ep, n_ep, loss, tr_acc, m):
        if verbose:
            print(f"  [{stage}] {ep:2d}/{n_ep}  loss={loss:.4f}  "
                  f"train_acc={tr_acc*100:.2f}%  val_acc={m['accuracy']*100:.2f}%  "
                  f"macroF1={m['f1_macro']*100:.2f}%  balAcc={m['balanced_accuracy']*100:.2f}%")

    def _maybe_save(m):
        nonlocal best_score, best_metrics
        if m[select_metric] > best_score:
            best_score = m[select_metric]
            best_metrics = dict(m)
            if ckpt_path is not None:
                torch.save(model.state_dict(), ckpt_path)

    # ── Stage 1: head only ───────────────────────────────────────────────
    if stage1_epochs > 0:
        for name, p in model.named_parameters():
            p.requires_grad = ("classifier" in name) or ("fc" in name)
        head_params = [p for p in model.parameters() if p.requires_grad]
        crit1 = nn.CrossEntropyLoss(weight=class_weights)
        opt1 = _make_opt(head_params, head_lr)
        if verbose:
            print(f"\n-- Stage 1: {stage1_epochs} epochs (head only) --")
        for ep in range(1, stage1_epochs + 1):
            loss, tr_acc = train_one_epoch(model, train_loader, crit1, opt1, device,
                                           scheduler=None, use_mixup=False, scaler=scaler,
                                           desc=f"S1 {ep}/{stage1_epochs}")
            m = evaluate(model, val_loader, device)
            m.update(stage=1, epoch=ep, loss=loss, train_acc=tr_acc)
            history.append(m)
            _log("S1", ep, stage1_epochs, loss, tr_acc, m)
            _maybe_save(m)

    # ── Stage 2: full fine-tune ──────────────────────────────────────────
    if stage2_epochs > 0:
        for p in model.parameters():
            p.requires_grad = True
        crit2 = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        opt2 = _make_opt(model.parameters(), base_lr)
        total_steps = stage2_epochs * len(train_loader)
        warmup_steps = warmup_epochs * len(train_loader)
        scheduler = optim.lr_scheduler.LambdaLR(
            opt2, lambda s: _cosine_warmup(s, warmup_steps, total_steps)
        )
        if verbose:
            print(f"\n-- Stage 2: {stage2_epochs} epochs (full fine-tune + mixup + label smoothing) --")
        for ep in range(1, stage2_epochs + 1):
            loss, tr_acc = train_one_epoch(model, train_loader, crit2, opt2, device,
                                           scheduler=scheduler, use_mixup=True,
                                           alpha=mixup_alpha, scaler=scaler,
                                           desc=f"S2 {ep}/{stage2_epochs}")
            m = evaluate(model, val_loader, device)
            m.update(stage=2, epoch=ep, loss=loss, train_acc=tr_acc)
            history.append(m)
            _log("S2", ep, stage2_epochs, loss, tr_acc, m)
            _maybe_save(m)

    if verbose and best_metrics:
        print(f"\nBest {select_metric}: {best_score*100:.2f}%  "
              f"(val_acc={best_metrics['accuracy']*100:.2f}%)")
    return best_metrics, history
