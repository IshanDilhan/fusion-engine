"""Live webcam demo of the full context model (CLIP scene + SmolVLM2 situation).

The camera loop runs at full speed; the VLM analyses a frame in a background
thread at most every VLM_INTERVAL_SEC, and the overlay always shows the most
recent finished analysis.

Run from repo root or this folder:
    python inference/realtime.py
"""
from pathlib import Path
import sys
import time

import cv2

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modalities.context.src.pipeline import ContextPipeline, _draw_overlay


def main():
    print("Initializing context pipeline (CLIP scene + SmolVLM2)...")
    pipeline = ContextPipeline(vlm_async=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open webcam.")
        sys.exit(1)
    print("Ready. Press 'q' or ESC to quit.")

    prev, fps = time.time(), 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated, state = pipeline.process_frame(frame)

        now = time.time(); dt = now - prev; prev = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 / dt

        _draw_overlay(annotated, state, fps)
        cv2.imshow("Context Model", annotated)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    pipeline.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
