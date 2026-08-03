"""Reusable scene (environment) classifier for the context model.

Wraps the trained backbone behind a simple ``predict(frame)`` interface with
temporal smoothing and confidence gating, so it can be used both by the
inference scripts and by the fused context pipeline.

This module bootstraps the scene-classification root onto sys.path so it imports
cleanly whether run as a script or imported by the pipeline.
"""
import json
import os
import sys
from collections import deque

import cv2
import numpy as np
import torch
from torchvision import transforms

_SCENE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCENE_ROOT not in sys.path:
    sys.path.insert(0, _SCENE_ROOT)

from config import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_MODEL,
    NORM_MEAN,
    NORM_STD,
    IMAGE_SIZE,
    SCENE_BACKEND,
    SCENE_LABELS,
    CLASSES_FILE,
)
from src.models import build_model  # noqa: E402


def create_scene_classifier(backend=None, **kwargs):
    """Build the configured scene classifier.

    backend "clip" -> ZeroShotSceneClassifier (deployed; 99.5% on captured
    clips), "cnn" -> the trained CNN SceneClassifier (evaluated baseline).
    Defaults to config.SCENE_BACKEND.
    """
    backend = backend or SCENE_BACKEND
    if backend == "clip":
        from src.zero_shot import ZeroShotSceneClassifier
        return ZeroShotSceneClassifier(**kwargs)
    if backend == "cnn":
        return SceneClassifier(**kwargs)
    raise ValueError(f"Unknown scene backend '{backend}' (expected 'clip' or 'cnn').")


class SceneClassifier:
    def __init__(
        self,
        checkpoint=None,
        model_name=DEFAULT_MODEL,
        classes=None,
        device=None,
        smooth_window=15,
        conf_threshold=0.5,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = checkpoint or DEFAULT_CHECKPOINT
        self.classes = classes or self._load_classes()
        self.conf_threshold = conf_threshold

        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Scene checkpoint not found: {checkpoint}")

        self.model = build_model(model_name, num_classes=len(self.classes), pretrained=False)
        self.model.load_state_dict(
            torch.load(checkpoint, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device).eval()

        # Must match the training/validation transforms.
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ])

        self._prob_history = deque(maxlen=smooth_window)

    def _load_classes(self):
        """Prefer classes.json next to the weights; else fall back to config."""
        if os.path.exists(CLASSES_FILE):
            try:
                with open(CLASSES_FILE, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return list(SCENE_LABELS)

    def reset(self):
        """Clear temporal smoothing history (e.g. when switching video sources)."""
        self._prob_history.clear()

    @torch.no_grad()
    def predict(self, frame_bgr):
        """Classify a single BGR frame (as read from OpenCV).

        Returns a dict: label (smoothed, or "uncertain"), confidence, raw_label,
        raw_confidence, probs {class: smoothed_probability}.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)

        probs = torch.softmax(self.model(tensor), dim=1)[0].cpu().numpy()
        raw_idx = int(probs.argmax())

        self._prob_history.append(probs)
        avg_probs = np.mean(self._prob_history, axis=0)
        smooth_idx = int(avg_probs.argmax())
        smooth_conf = float(avg_probs[smooth_idx])

        label = self.classes[smooth_idx] if smooth_conf >= self.conf_threshold else "uncertain"
        return {
            "label": label,
            "confidence": smooth_conf,
            "raw_label": self.classes[raw_idx],
            "raw_confidence": float(probs[raw_idx]),
            "probs": {c: float(p) for c, p in zip(self.classes, avg_probs)},
        }
