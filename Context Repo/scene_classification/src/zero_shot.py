"""Zero-shot scene (environment) classifier via CLIP image-text matching.

Drop-in replacement for the CNN ``SceneClassifier``: same ``predict(frame)`` /
``reset()`` interface, temporal smoothing, and confidence gating. Instead of a
trained head, each frame is matched against per-class prompt ensembles — so
adding an environment is a config edit (SCENE_PROMPTS), not a retraining run.

Chosen over the fine-tuned CNN after benchmarking on the captured clips:
99.5% overall vs 82.2% (the CNN's kitchen domain gap). See
reports/zero_shot/ZERO_SHOT_REPORT.md and scripts/zero_shot_benchmark.py.
"""
import os
import sys
from collections import deque

import cv2
import numpy as np
import torch
from PIL import Image

_SCENE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCENE_ROOT not in sys.path:
    sys.path.insert(0, _SCENE_ROOT)

from config import (  # noqa: E402
    ABSTAIN_PROMPTS,
    CLIP_MODEL,
    CLIP_PRETRAINED,
    SCENE_LABELS,
    SCENE_PROMPTS,
)


class ZeroShotSceneClassifier:
    def __init__(
        self,
        model_name=CLIP_MODEL,
        pretrained=CLIP_PRETRAINED,
        classes=None,
        prompts=None,
        abstain_prompts=None,
        device=None,
        smooth_window=15,
        conf_threshold=0.5,
        abstain_threshold=0.5,
    ):
        import open_clip

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = classes or list(SCENE_LABELS)
        prompts = prompts or SCENE_PROMPTS
        abstain_prompts = abstain_prompts if abstain_prompts is not None else ABSTAIN_PROMPTS
        self.conf_threshold = conf_threshold
        self.abstain_threshold = abstain_threshold

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained)
        self.model.to(self.device).eval()
        tokenizer = open_clip.get_tokenizer(model_name)

        # One averaged, normalized text embedding per class (+ abstain probe).
        with torch.no_grad():
            embs = []
            for label in self.classes:
                e = self.model.encode_text(tokenizer(prompts[label]).to(self.device)).float()
                e = e / e.norm(dim=-1, keepdim=True)
                embs.append(e.mean(dim=0, keepdim=True))
            if abstain_prompts:
                e = self.model.encode_text(tokenizer(abstain_prompts).to(self.device)).float()
                e = e / e.norm(dim=-1, keepdim=True)
                embs.append(e.mean(dim=0, keepdim=True))
            text = torch.cat(embs)
            self.text_embs = text / text.norm(dim=-1, keepdim=True)
        self._has_abstain = bool(abstain_prompts)

        self._prob_history = deque(maxlen=smooth_window)
        self._abstain_history = deque(maxlen=smooth_window)

    def reset(self):
        """Clear temporal smoothing history (e.g. when switching video sources)."""
        self._prob_history.clear()
        self._abstain_history.clear()

    @torch.no_grad()
    def predict(self, frame_bgr):
        """Classify a single BGR frame (as read from OpenCV).

        Returns the same dict shape as the CNN SceneClassifier:
            label, confidence, raw_label, raw_confidence, probs
        `label` is "uncertain" below the confidence threshold or when the frame
        looks like a face close-up with no scene content (abstain probe).
        """
        pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        img = self.model.encode_image(
            self.preprocess(pil).unsqueeze(0).to(self.device)).float()
        img = img / img.norm(dim=-1, keepdim=True)

        logits = 100.0 * img @ self.text_embs.T                    # (1, C[+1])
        scene_probs = torch.softmax(logits[:, : len(self.classes)], dim=1)[0].cpu().numpy()
        if self._has_abstain:
            abstain_p = float(torch.softmax(logits, dim=1)[0, -1])
        else:
            abstain_p = 0.0
        raw_idx = int(scene_probs.argmax())

        # Temporal smoothing: average recent probability vectors.
        self._prob_history.append(scene_probs)
        self._abstain_history.append(abstain_p)
        avg_probs = np.mean(self._prob_history, axis=0)
        avg_abstain = float(np.mean(self._abstain_history))
        smooth_idx = int(avg_probs.argmax())
        smooth_conf = float(avg_probs[smooth_idx])

        if smooth_conf < self.conf_threshold or avg_abstain > self.abstain_threshold:
            label = "uncertain"
        else:
            label = self.classes[smooth_idx]

        return {
            "label": label,
            "confidence": smooth_conf,
            "raw_label": self.classes[raw_idx],
            "raw_confidence": float(scene_probs[raw_idx]),
            "probs": {c: float(p) for c, p in zip(self.classes, avg_probs)},
        }
