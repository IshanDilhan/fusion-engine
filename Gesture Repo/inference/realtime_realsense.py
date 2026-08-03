"""
Real-time gesture recognition using MediaPipe Holistic + the trained
temporal network (GestureEngine). Supports Intel RealSense with webcam
fallback, in the same style as the motion modality.

    python inference/realtime_realsense.py
    python inference/realtime_realsense.py --camera 1
"""
import argparse
import os
import sys
import time

import cv2 as cv
import mediapipe as mp
import numpy as np

try:
    import pyrealsense2 as rs
    _RS_AVAILABLE = True
except ImportError:
    _RS_AVAILABLE = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, ".."))
from config import DEFAULT_CHECKPOINT, DEFAULT_MODEL_CONFIG, GESTURE_COLORS
from src import GestureEngine


def draw_dashboard(frame, label, confidence, fps, results, mp_draw, mp_holistic):
    h, w = frame.shape[:2]
    font = cv.FONT_HERSHEY_SIMPLEX

    if results.pose_landmarks:
        mp_draw.draw_landmarks(
            frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0, 255, 100), thickness=2, circle_radius=3),
            mp_draw.DrawingSpec(color=(0, 180, 255), thickness=2))
    for hand_lms in (results.left_hand_landmarks, results.right_hand_landmarks):
        if hand_lms:
            mp_draw.draw_landmarks(
                frame, hand_lms, mp_holistic.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(255, 120, 0), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(255, 200, 120), thickness=1))

    color = GESTURE_COLORS.get(label, (255, 255, 255))
    sidebar_w = 320
    canvas = np.zeros((h, w + sidebar_w, 3), dtype=np.uint8)
    canvas[:, :w] = frame
    cv.rectangle(canvas, (w, 0), (w + sidebar_w, h), (20, 24, 33), -1)
    cv.line(canvas, (w, 0), (w, h), (43, 52, 69), 2)

    cv.putText(canvas, "HRI GESTURE ANALYZER", (w + 20, 35), font, 0.62,
               (230, 235, 245), 2, cv.LINE_AA)
    cv.line(canvas, (w + 20, 48), (w + sidebar_w - 20, 48), (60, 75, 100), 1)
    cv.putText(canvas, f"FPS: {fps:.1f}", (w + 20, 75), font, 0.5,
               (150, 160, 180), 1, cv.LINE_AA)

    cv.putText(canvas, "ACTIVE GESTURE:", (w + 20, 120), font, 0.55,
               (100, 120, 150), 1, cv.LINE_AA)
    cv.rectangle(canvas, (w + 20, 140), (w + sidebar_w - 20, 210), color, -1)
    text_color = (255, 255, 255) if np.mean(color) < 120 else (20, 20, 20)
    cv.putText(canvas, label, (w + 30, 175), font, 0.7, text_color, 2, cv.LINE_AA)
    cv.putText(canvas, f"Conf: {confidence*100:.1f}%", (w + 30, 200), font, 0.55,
               text_color, 1, cv.LINE_AA)
    return canvas


def resize_with_aspect_ratio(image, max_dim=960):
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / float(max(h, w))
    return cv.resize(image, (int(w * scale), int(h * scale)),
                     interpolation=cv.INTER_AREA)


def _try_start_realsense():
    if not _RS_AVAILABLE:
        return None, None
    try:
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(cfg)
        return pipeline, rs.align(rs.stream.color)
    except Exception:
        return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="Webcam fallback index.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG)
    args = parser.parse_args()

    mp_holistic = mp.solutions.holistic
    mp_draw = mp.solutions.drawing_utils
    holistic = mp_holistic.Holistic(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    engine = GestureEngine(checkpoint=args.checkpoint, model_config=args.model_config)
    pipeline, align = _try_start_realsense()

    if pipeline is not None:
        print("RealSense camera connected.")
        window_title, cap = "Gesture Recognition (RealSense)", None
    else:
        print(f"RealSense fallback webcam (index {args.camera}).")
        cap = cv.VideoCapture(args.camera)
        if not cap.isOpened():
            print("Error: camera index invalid.")
            return
        window_title = "Gesture Recognition (Webcam)"

    last_time = time.time()
    fps = 0.0

    try:
        while True:
            if pipeline is not None:
                frames = align.process(pipeline.wait_for_frames())
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                frame = np.asarray(color_frame.get_data())
            else:
                ret, frame = cap.read()
                if not ret:
                    break

            frame = resize_with_aspect_ratio(frame, max_dim=960)
            res = holistic.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            label, confidence = engine.process_holistic(res)

            now = time.time()
            dt = now - last_time
            last_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            canvas = draw_dashboard(frame, label, confidence, fps, res,
                                    mp_draw, mp_holistic)
            cv.imshow(window_title, canvas)
            if cv.waitKey(1) & 0xFF in (27, ord('q')):
                break
    finally:
        if pipeline is not None:
            pipeline.stop()
        else:
            cap.release()
        holistic.close()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
