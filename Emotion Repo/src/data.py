"""
Dataset / dataloader helpers shared across all scripts.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from config import (
    BATCH_SIZE,
    NUM_WORKERS,
    TEST_DIR,
    TRAIN_DIR,
)
from src.transforms import get_test_transforms, get_train_transforms


def get_datasets(train_transform=None, test_transform=None):
    """Return (train_ds, val_ds) RAF-DB ImageFolder datasets."""
    train_ds = ImageFolder(
        TRAIN_DIR, transform=train_transform or get_train_transforms()
    )
    val_ds = ImageFolder(
        TEST_DIR, transform=test_transform or get_test_transforms()
    )
    return train_ds, val_ds


def get_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                    train_ds=None, val_ds=None, pin_memory=True):
    """Return (train_loader, val_loader)."""
    if train_ds is None or val_ds is None:
        train_ds, val_ds = get_datasets()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, val_loader


def compute_class_weights(dataset):
    """Inverse-frequency class weights (normalized to sum to num_classes)."""
    counts = np.bincount([label for _, label in dataset.samples])
    w = 1.0 / counts.astype(float)
    w = w / w.sum() * len(counts)
    return torch.FloatTensor(w)


def class_counts(dataset):
    """Return a {class_index: count} dict for an ImageFolder dataset."""
    counts = np.bincount([label for _, label in dataset.samples])
    return {i: int(c) for i, c in enumerate(counts)}


def subset_per_class(dataset, max_per_class):
    """
    Small stratified subset for smoke tests / fast trials.
    Returns a torch Subset preserving the dataset's transform.
    """
    by_class = {}
    for idx, (_, label) in enumerate(dataset.samples):
        by_class.setdefault(label, []).append(idx)
    keep = []
    for label, idxs in by_class.items():
        keep.extend(idxs[:max_per_class])
    return Subset(dataset, keep)
