"""
Stage 0 - Exploratory Data Analysis for RAF-DB.

Produces:
  reports/eda/class_distribution.png   class counts (train vs test)
  reports/eda/sample_grid.png          one sample row per emotion (train)
  reports/eda/image_sizes.png          distribution of source image sizes
  reports/eda/class_counts.csv         per-class counts + class weights
  reports/eda/EDA_REPORT.md            written summary

Run:
    python scripts/eda.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import EMOTION_LABELS, REPORT_DIR, TEST_DIR, TRAIN_DIR
from src.data import class_counts, compute_class_weights, get_datasets

OUT_DIR = os.path.join(REPORT_DIR, "eda")


def _counts_frame(train_ds, val_ds):
    tr = class_counts(train_ds)
    va = class_counts(val_ds)
    weights = compute_class_weights(train_ds).numpy()
    rows = []
    for i, name in enumerate(EMOTION_LABELS):
        rows.append({
            "index": i,
            "emotion": name,
            "train": tr.get(i, 0),
            "test": va.get(i, 0),
            "train_pct": 100 * tr.get(i, 0) / max(1, len(train_ds)),
            "class_weight": round(float(weights[i]), 4),
        })
    return pd.DataFrame(rows)


def plot_distribution(df, path):
    x = np.arange(len(df))
    w = 0.4
    plt.figure(figsize=(11, 6))
    plt.bar(x - w / 2, df["train"], w, label="train", color="#3b7dd8")
    plt.bar(x + w / 2, df["test"], w, label="test", color="#e07a3b")
    for i, (t, v) in enumerate(zip(df["train"], df["test"])):
        plt.text(i - w / 2, t, str(t), ha="center", va="bottom", fontsize=8)
        plt.text(i + w / 2, v, str(v), ha="center", va="bottom", fontsize=8)
    plt.xticks(x, df["emotion"], rotation=30)
    plt.ylabel("Number of images")
    plt.title("RAF-DB class distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_sample_grid(path, n_per_class=5):
    classes = sorted(os.listdir(TRAIN_DIR), key=lambda s: int(s) if s.isdigit() else s)
    fig, axes = plt.subplots(len(classes), n_per_class,
                             figsize=(n_per_class * 1.6, len(classes) * 1.6))
    for r, cls in enumerate(classes):
        cls_dir = os.path.join(TRAIN_DIR, cls)
        files = os.listdir(cls_dir)[:n_per_class]
        label = EMOTION_LABELS[int(cls) - 1] if cls.isdigit() else cls
        for c in range(n_per_class):
            ax = axes[r][c] if len(classes) > 1 else axes[c]
            ax.axis("off")
            if c < len(files):
                ax.imshow(Image.open(os.path.join(cls_dir, files[c])).convert("RGB"))
            if c == 0:
                ax.set_title(label, loc="left", fontsize=9, x=-0.1)
    fig.suptitle("RAF-DB sample images per emotion (train)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_image_sizes(path, sample=400):
    sizes = []
    for root in (TRAIN_DIR, TEST_DIR):
        for cls in os.listdir(root):
            cls_dir = os.path.join(root, cls)
            for f in os.listdir(cls_dir)[: sample // 14]:
                with Image.open(os.path.join(cls_dir, f)) as im:
                    sizes.append(im.size)  # (w, h)
    sizes = np.array(sizes)
    plt.figure(figsize=(7, 6))
    plt.scatter(sizes[:, 0], sizes[:, 1], alpha=0.3, s=12)
    plt.xlabel("width (px)")
    plt.ylabel("height (px)")
    plt.title(f"Source image dimensions (n={len(sizes)} sampled)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return sizes


def write_report(df, sizes, train_ds, val_ds, path):
    imbalance = df["train"].max() / max(1, df["train"].min())
    uniq = {tuple(s) for s in sizes}
    size_note = (f"all sampled images are {sizes[0][0]}x{sizes[0][1]} px"
                 if len(uniq) == 1 else f"{len(uniq)} distinct sizes sampled")
    lines = [
        "# Emotion Model - EDA Report (RAF-DB)\n",
        "## 1. Dataset overview\n",
        f"- Classes: **{len(EMOTION_LABELS)}** ({', '.join(EMOTION_LABELS)})",
        f"- Train images: **{len(train_ds)}**",
        f"- Test images: **{len(val_ds)}**",
        f"- Layout: `ImageFolder` (folders 1..7, alphabetically sorted -> label index 0..6)",
        f"- Source image size: {size_note}\n",
        "## 2. Class distribution\n",
        df.to_markdown(index=False),
        "",
        f"\n- **Imbalance ratio (max/min): {imbalance:.1f}x** "
        f"(largest: {df.loc[df['train'].idxmax(), 'emotion']}, "
        f"smallest: {df.loc[df['train'].idxmin(), 'emotion']}).",
        "- This justifies the balanced training recipe: inverse-frequency "
        "**weighted CrossEntropy**, label smoothing, mixup, and selecting models "
        "by **macro-F1 / balanced accuracy** rather than raw accuracy.\n",
        "## 3. Preprocessing decisions\n",
        "- RAF-DB `_aligned` crops are already face-aligned, so no extra face "
        "detection is needed for training.",
        "- Train augmentation: random crop/flip/rotation/affine, color jitter, "
        "occasional grayscale, CLAHE, RandomErasing (see `src/transforms.py`).",
        "- Eval/inference: resize to 224x224 + ImageNet normalization.",
        "- CLAHE is applied in both training augmentation and live inference so the "
        "model sees a consistent contrast distribution.\n",
        "## 4. Figures\n",
        "- `class_distribution.png` - per-class train/test counts",
        "- `sample_grid.png` - example faces per emotion",
        "- `image_sizes.png` - source resolution scatter\n",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading datasets ...")
    train_ds, val_ds = get_datasets()
    df = _counts_frame(train_ds, val_ds)
    print(df.to_string(index=False))

    df.to_csv(os.path.join(OUT_DIR, "class_counts.csv"), index=False)
    plot_distribution(df, os.path.join(OUT_DIR, "class_distribution.png"))
    plot_sample_grid(os.path.join(OUT_DIR, "sample_grid.png"))
    sizes = plot_image_sizes(os.path.join(OUT_DIR, "image_sizes.png"))
    write_report(df, sizes, train_ds, val_ds, os.path.join(OUT_DIR, "EDA_REPORT.md"))

    print(f"\nEDA complete. Outputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
