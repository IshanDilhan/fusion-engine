"""
Lightweight ImageNet-pretrained backbones with the classifier head
swapped for ``NUM_CLASSES`` emotion outputs. All four are candidates for
Jetson Orin Nano deployment.

Uses the modern ``weights="DEFAULT"`` API (torchvision >= 0.13) instead of the
deprecated ``pretrained=True`` flag.
"""
import torch.nn as nn
import torchvision.models as tvm

from config import NUM_CLASSES


def get_mobilenet_v2(num_classes=NUM_CLASSES, pretrained=True):
    w = "DEFAULT" if pretrained else None
    model = tvm.mobilenet_v2(weights=w)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def get_mobilenet_v3_large(num_classes=NUM_CLASSES, pretrained=True):
    w = "DEFAULT" if pretrained else None
    model = tvm.mobilenet_v3_large(weights=w)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def get_efficientnet_b0(num_classes=NUM_CLASSES, pretrained=True):
    w = "DEFAULT" if pretrained else None
    model = tvm.efficientnet_b0(weights=w)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def get_mnasnet(num_classes=NUM_CLASSES, pretrained=True):
    w = "DEFAULT" if pretrained else None
    model = tvm.mnasnet1_0(weights=w)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


# Registry used by every script and the inference code.
ALL_MODELS = {
    "MobileNetV2": get_mobilenet_v2,
    "MobileNetV3-Large": get_mobilenet_v3_large,
    "EfficientNet-B0": get_efficientnet_b0,
    "MNASNet1_0": get_mnasnet,
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
