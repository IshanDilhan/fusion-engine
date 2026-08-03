"""
GestureEngine — real-time gesture resolver (guide §8).

Feeds MediaPipe Holistic landmarks frame-by-frame, keeps a rolling raw-feature
buffer (~2 s, matching the temporal span of training clips), resamples it to
the model window, and stabilizes predictions with EMA softmax smoothing plus
confidence-gated debouncing. API mirrors motion's MotionEngine:

    engine = GestureEngine()
    label, confidence = engine.process(res.pose_landmarks,
                                       res.left_hand_landmarks,
                                       res.right_hand_landmarks)
"""
import json
import os
import sys
import time
from collections import deque

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import (
    CONF_THRESHOLD,
    DEBOUNCE_S,
    DEFAULT_CHECKPOINT,
    DEFAULT_MODEL_CONFIG,
    EMA_ALPHA,
    ENGINE_BUFFER_FRAMES,
    GESTURE_LABELS,
    NUM_CLASSES,
    WINDOW,
)
from src.features import build_features, uniform_indices
from src.models import build_model


def landmarks_to_arrays(pose_landmarks, left_hand_landmarks, right_hand_landmarks):
    """MediaPipe Holistic landmark lists -> raw arrays for one frame (NaN = absent)."""
    pose = np.full((1, 33, 4), np.nan, dtype=np.float32)
    if pose_landmarks is not None:
        for i, lm in enumerate(pose_landmarks.landmark):
            pose[0, i] = [lm.x, lm.y, lm.z, lm.visibility]
    hands = []
    for hl in (left_hand_landmarks, right_hand_landmarks):
        hand = np.full((1, 21, 3), np.nan, dtype=np.float32)
        if hl is not None:
            for i, lm in enumerate(hl.landmark):
                hand[0, i] = [lm.x, lm.y, lm.z]
        hands.append(hand)
    return pose, hands[0], hands[1]


class GestureEngine:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT,
                 model_config=DEFAULT_MODEL_CONFIG, device=None):
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # model_config.json (written by scripts/train.py) pins architecture,
        # labels and window so inference can't silently drift from training.
        with open(model_config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.labels = cfg.get("labels", GESTURE_LABELS)
        self.window = int(cfg.get("window", WINDOW))
        kwargs = cfg.get("model_kwargs", {})
        self.model = build_model(cfg["model"], **kwargs).to(self.device)
        state = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

        self.conf_threshold = float(cfg.get("conf_threshold", CONF_THRESHOLD))
        self.buffer = deque(maxlen=ENGINE_BUFFER_FRAMES)
        self.smooth_probs = np.ones(len(self.labels)) / len(self.labels)
        self.current = "idle"
        self.candidate = "idle"
        self.candidate_since = time.time()

    def reset(self):
        """Call when the person leaves the frame."""
        self.buffer.clear()
        self.smooth_probs = np.ones(len(self.labels)) / len(self.labels)
        self.current = "idle"
        self.candidate = "idle"
        self.candidate_since = time.time()

    @torch.no_grad()
    def process(self, pose_landmarks, left_hand_landmarks=None,
                right_hand_landmarks=None):
        """One frame of Holistic landmarks -> (stable_label, confidence)."""
        if pose_landmarks is None:
            self.reset()
            return "idle", 0.0

        pose, lh, rh = landmarks_to_arrays(
            pose_landmarks, left_hand_landmarks, right_hand_landmarks)
        self.buffer.append(build_features(pose, lh, rh)[0])

        if len(self.buffer) < self.window:
            return self.current, 0.0

        # resample the ~2 s buffer down to the model window, mirroring how
        # training clips are resampled to WINDOW frames
        frames = np.asarray(self.buffer, dtype=np.float32)
        idx = uniform_indices(frames.shape[0], self.window)
        seq = torch.from_numpy(frames[idx][None]).to(self.device)

        probs = torch.softmax(self.model(seq)[0], dim=0).cpu().numpy()
        self.smooth_probs = EMA_ALPHA * probs + (1 - EMA_ALPHA) * self.smooth_probs

        top = int(np.argmax(self.smooth_probs))
        conf = float(self.smooth_probs[top])
        candidate = self.labels[top] if conf >= self.conf_threshold else "idle"

        # debounce: a new label must hold for DEBOUNCE_S before being emitted
        now = time.time()
        if candidate != self.candidate:
            self.candidate = candidate
            self.candidate_since = now
        elif candidate != self.current and now - self.candidate_since >= DEBOUNCE_S:
            self.current = candidate

        return self.current, conf

    def process_holistic(self, results):
        """Convenience wrapper for a MediaPipe Holistic results object."""
        return self.process(results.pose_landmarks,
                            results.left_hand_landmarks,
                            results.right_hand_landmarks)
