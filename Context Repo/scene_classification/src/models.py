"""
Lightweight ImageNet-pretrained backbones with the classifier head swapped for
``NUM_CLASSES`` scene outputs. All three are candidates for Jetson Orin Nano
deployment and are compared in scripts/compare_models.py.

Uses the modern ``weights="DEFAULT"`` API (torchvision >= 0.13).
"""
import torch.nn as nn
import torchvision.models as tvm

from config import NUM_CLASSES


def get_efficientnet_b0(num_classes=NUM_CLASSES, pretrained=True):
    w = "DEFAULT" if pretrained else None
    model = tvm.efficientnet_b0(weights=w)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def get_mobilenet_v3_small(num_classes=NUM_CLASSES, pretrained=True):
    w = "DEFAULT" if pretrained else None
    model = tvm.mobilenet_v3_small(weights=w)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model


def get_resnet18(num_classes=NUM_CLASSES, pretrained=True):
    w = "DEFAULT" if pretrained else None
    model = tvm.resnet18(weights=w)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# Registry used by every script and the inference code.
ALL_MODELS = {
    "EfficientNet-B0": get_efficientnet_b0,
    "MobileNetV3-Small": get_mobilenet_v3_small,
    "ResNet18": get_resnet18,
}


def build_model(name, num_classes=NUM_CLASSES, pretrained=True):
    """Instantiate a model by registry name."""
    if name not in ALL_MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(ALL_MODELS)}")
    return ALL_MODELS[name](num_classes=num_classes, pretrained=pretrained)


def safe_name(name):
    """Filesystem-safe variant of a model name (for checkpoint / report paths)."""
    return name.replace(" ", "_").replace("-", "_")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model):
    total = sum(p.nelement() * p.element_size() for p in model.parameters())
    total += sum(b.nelement() * b.element_size() for b in model.buffers())
    return total / (1024 * 1024)
