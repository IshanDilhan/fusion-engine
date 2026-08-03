"""
Image transforms for training and evaluation.

Scenes are distinguished by global layout/texture, so augmentation is moderate:
resize, horizontal flip, small rotation and color jitter. Eval-time is a plain
resize + normalize. Both match the inference preprocessing.
"""
from torchvision import transforms

from config import IMAGE_SIZE, NORM_MEAN, NORM_STD


def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])


def get_test_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
