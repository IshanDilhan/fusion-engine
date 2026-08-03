"""Context model pipeline: CLIP scene classification + small-VLM situation analysis.

Produces one `ContextState` per frame for the downstream policy / fusion module:
    scene (CLIP zero-shot, throttled to every Nth frame)
  + vlm   (SmolVLM2: people / activity / attention / objects / summary)

The VLM is much slower than a frame interval, so it runs on its own schedule:
  - async mode (realtime camera): a worker thread analyses a frame at most every
    VLM_INTERVAL_SEC; the camera loop never blocks and every ContextState carries
    the most recent finished analysis.
  - sync mode (offline video): the VLM runs inline every VLM_EVERY_FRAMES
    processed frames — deterministic, good for batch evaluation.
"""

from pathlib import Path
import sys
import threading
import time

import cv2

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modalities.context.config import (
    SCENE_EVERY,
    VLM_EVERY_FRAMES,
    VLM_INTERVAL_SEC,
)
from modalities.context.scene_classification.src.classifier import create_scene_classifier
from modalities.context.src.context_state import ContextState, VLMContext
from modalities.context.src.vlm import VLMContextAnalyzer


class ContextPipeline:
    def __init__(
        self,
        scene_every=SCENE_EVERY,
        vlm_async=True,
        vlm_interval_sec=VLM_INTERVAL_SEC,
        vlm_every_frames=VLM_EVERY_FRAMES,
    ):
        self.scene = create_scene_classifier()
        self.vlm = VLMContextAnalyzer()

        self.scene_every = scene_every
        self.vlm_async = vlm_async
        self.vlm_interval_sec = vlm_interval_sec
        self.vlm_every_frames = vlm_every_frames

        self._frame_idx = 0
        self._last_scene = {"label": "unknown", "confidence": 0.0}

        self._vlm_lock = threading.Lock()
        self._vlm_busy = False
        self._latest_vlm = VLMContext()
        self._last_vlm_done = 0.0

    def process_frame(self, frame):
        """Classify the scene, keep the VLM fed, and return the fused state."""
        self._frame_idx += 1

        # --- Scene (throttled) ---
        if self.scene_every <= 1 or self._frame_idx % self.scene_every == 1:
            self._last_scene = self.scene.predict(frame)

        # --- VLM (async worker or inline every N frames) ---
        if self.vlm_async:
            self._maybe_submit_async(frame)
        elif self._frame_idx % self.vlm_every_frames == 1:
            vlm_ctx = self.vlm.analyze(frame, frame_timestamp=time.time())
            with self._vlm_lock:
                self._latest_vlm = vlm_ctx

        with self._vlm_lock:
            vlm_snapshot = self._latest_vlm

        return frame, ContextState(
            scene=self._last_scene["label"],
            scene_confidence=self._last_scene["confidence"],
            vlm=vlm_snapshot,
            timestamp=time.time(),
        )

    def reset(self):
        """Clear temporal state (e.g. when switching video sources)."""
        self._frame_idx = 0
        self.scene.reset()
        with self._vlm_lock:
            self._latest_vlm = VLMContext()

    def close(self):
        pass  # nothing to release; worker threads are daemonic

    # ── async VLM worker ─────────────────────────────────────────────────
    def _maybe_submit_async(self, frame):
        now = time.time()
        with self._vlm_lock:
            if self._vlm_busy or (now - self._last_vlm_done) < self.vlm_interval_sec:
                return
            self._vlm_busy = True
        threading.Thread(
            target=self._vlm_worker, args=(frame.copy(), now), daemon=True
        ).start()

    def _vlm_worker(self, frame, captured_at):
        try:
            ctx = self.vlm.analyze(frame, frame_timestamp=captured_at)
        except Exception as exc:                     # keep the camera loop alive
            ctx = VLMContext(summary=f"vlm error: {exc}")
        with self._vlm_lock:
            self._latest_vlm = ctx
            self._last_vlm_done = time.time()
            self._vlm_busy = False


def _draw_overlay(frame, state, fps):
    """Annotate a frame with the fused context (used by the inference scripts)."""
    h, w = frame.shape[:2]
    v = state.vlm

    cv2.rectangle(frame, (0, 0), (w, 84), (30, 30, 30), -1)
    line1 = (f"scene: {state.scene} ({state.scene_confidence:.2f})   "
             f"people: {v.people}   {fps:.1f} FPS")
    line2 = f"activity: {v.activity}   attention: {v.attention}"
    line3 = "objects: " + (", ".join(v.objects) if v.objects else "-")
    cv2.putText(frame, line1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, line2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv2.putText(frame, line3, (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    if v.summary:
        cv2.rectangle(frame, (0, h - 28), (w, h), (30, 30, 30), -1)
        cv2.putText(frame, v.summary[:110], (10, h - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
    return frame
