"""
Image transforms for training and evaluation.

Train-time augmentation is deliberately rich to combat RAF-DB's small size and
class imbalance: random crop/flip/rotate/affine, color jitter, occasional
grayscale, CLAHE (matching the inference-time contrast normalization), and
RandomErasing. Eval-time is a plain resize + normalize.
"""
import random

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from config import IMAGE_SIZE, NORM_MEAN, NORM_STD

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


class RandomCLAHE:
    """Randomly applies CLAHE on the L channel to match the inference pipeline."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        arr = np.array(img)
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = _clahe.apply(l)
        arr = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
        return Image.fromarray(arr)


def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 16, IMAGE_SIZE + 16)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        RandomCLAHE(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])


def get_test_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
