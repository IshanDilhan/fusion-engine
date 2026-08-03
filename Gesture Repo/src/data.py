"""
Dataset-mapping tables (guide §4) and the PyTorch sequence dataset.

The mapping tables are consumed by scripts/extract_landmarks.py (to decide
what to extract) and scripts/prepare_data.py (to assign final labels and
splits). The dataset class reads the split index CSVs written by
prepare_data.py and loads landmark .npz files lazily.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import GESTURE_LABELS, INDEX_DIR, LABEL_TO_ID, LANDMARK_DIR, WINDOW
from src.features import augment_features, build_features, mirror_raw, sample_window, uniform_indices

# ── Jester → labels (guide §4.1) ─────────────────────────────────────────
JESTER_MAP = {
    "Thumb Up": "thumbs_up",
    "Thumb Down": "thumbs_down",
    "Shaking Hand": "wave",
    "Pulling Hand In": "beckoning",   # proxy — blended with custom beckoning clips
    "No gesture": "idle",
    "Doing other things": "idle",
}
# Every other Jester class also maps to idle as a hard negative, capped
# per class at extraction time (deterministic: lowest clip ids first).
JESTER_NEGATIVE_CAP = 300

# ── NTU RGB+D 120 → labels (guide §4.2) ──────────────────────────────────
NTU_MAP = {
    23: "wave",            # hand waving
    31: "point",           # pointing to something with finger
    69: "thumbs_up",       # thumb up
    70: "thumbs_down",     # thumb down
    95: "both_hands_up",   # hands up (both hands)
    22: "both_hands_up",   # cheer up — visually verify a sample, drop if unfit
}
# Daily-action confusers -> idle. A38 salute is deliberately NOT raise_hand.
NTU_NEGATIVES = {1: "drink water", 8: "sit down", 10: "clapping",
                 34: "rub hands", 37: "wipe face", 38: "salute"}
NTU_NEGATIVE_CAP = 300


def resolve_label(dataset, source_class):
    """Map a dataset-native class to a final gesture label (or None = skip)."""
    if dataset == "jester":
        return JESTER_MAP.get(source_class, "idle")
    if dataset == "ntu":
        action = int(source_class)
        if action in NTU_MAP:
            return NTU_MAP[action]
        if action in NTU_NEGATIVES:
            return "idle"
        return None
    if dataset == "custom":
        return source_class if source_class in GESTURE_LABELS else None
    return None


# ── index CSVs ───────────────────────────────────────────────────────────
def load_index(split):
    """Load a split index written by prepare_data.py ('train'/'val'/'test'/'live_test')."""
    path = os.path.join(INDEX_DIR, f"{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run scripts/prepare_data.py first.")
    return pd.read_csv(path)


def compute_class_weights(df):
    """Square-root inverse-frequency class weights (mean-normalized).

    sqrt softens the raw inverse-frequency ratio (idle vs raise_hand is
    ~280:1 here — raw weights make the model abandon the majority class,
    which tanks precision everywhere). Classes with no training data get
    weight 0 — otherwise their (huge) weight leaks into every sample's
    loss through label smoothing and the model collapses onto the
    untrained classes.
    """
    counts = df["label_id"].value_counts().reindex(
        range(len(GESTURE_LABELS)), fill_value=0).to_numpy(dtype=np.float64)
    present = counts > 0
    weights = np.zeros_like(counts)
    weights[present] = 1.0 / np.sqrt(counts[present])
    weights[present] *= present.sum() / weights[present].sum()
    return torch.tensor(weights, dtype=torch.float32)


# ── dataset ──────────────────────────────────────────────────────────────
class GestureSequenceDataset(Dataset):
    """
    Yields (features [WINDOW, FEATURE_DIM] float32, label_id int64).

    train=True enables augmentation: mirror (p=0.5, with L/R side swap),
    speed/start-crop temporal sampling, rotation/scale/jitter, hand dropout.
    """

    def __init__(self, index_df, train=False, seed=0):
        self.df = index_df.reset_index(drop=True)
        self.train = train
        self.base_seed = seed

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        with np.load(os.path.join(LANDMARK_DIR, row["path"])) as d:
            pose = d["pose"]
            lh = d["left_hand"]
            rh = d["right_hand"]

        rng = None
        if self.train:
            # fresh randomness per epoch, reproducible per worker via torch seed
            rng = np.random.default_rng(
                (self.base_seed + i + torch.initial_seed()) % (2**32))
            if rng.random() < 0.5:
                pose, lh, rh = mirror_raw(pose, lh, rh)
            idx = sample_window(pose.shape[0], WINDOW, rng)
        else:
            idx = uniform_indices(pose.shape[0], WINDOW)

        feats = build_features(pose[idx], lh[idx], rh[idx])
        if self.train:
            feats = augment_features(feats, rng)

        return torch.from_numpy(feats), int(row["label_id"])


def get_datasets(seed=0):
    """(train_ds, val_ds) from the index CSVs."""
    train_ds = GestureSequenceDataset(load_index("train"), train=True, seed=seed)
    val_ds = GestureSequenceDataset(load_index("val"), train=False)
    return train_ds, val_ds


def label_ids(names):
    return [LABEL_TO_ID[n] for n in names]
