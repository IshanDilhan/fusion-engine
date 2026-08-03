"""
Gesture recognition over video files -> live preview and/or annotated mp4.

Defaults to scanning the repo-root videos/ folder (same convention as the
other modalities). The engine is reset per video so tracking state never
leaks between files.

    python inference/video.py                          # scan repo videos/
    python inference/video.py --input path/to/clip.mp4
    python inference/video.py --input videos/Classroom --save --no-show
    python inference/video.py --stride 2               # process every 2nd frame

Playback: [SPACE] pause · [N] next video · [Q] quit
"""
import argparse
import glob
import os
import sys
from datetime import datetime

import cv2 as cv
import mediapipe as mp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, ".."))
from config import DEFAULT_CHECKPOINT, DEFAULT_MODEL_CONFIG, GESTURE_COLORS, OUTPUT_DIR, ROOT
from src import GestureEngine

REPO_VIDEOS = os.path.abspath(os.path.join(ROOT, "..", "..", "videos"))
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def collect_videos(input_path):
    if input_path is None:
        input_path = REPO_VIDEOS
    if os.path.isfile(input_path):
        return [input_path]
    hits = []
    for ext in VIDEO_EXTS:
        hits += glob.glob(os.path.join(input_path, "**", f"*{ext}"), recursive=True)
    return sorted(hits)


def annotate(frame, label, confidence):
    color = GESTURE_COLORS.get(label, (255, 255, 255))
    cv.rectangle(frame, (10, 10), (360, 62), (20, 24, 33), -1)
    cv.rectangle(frame, (10, 10), (360, 62), color, 2)
    cv.putText(frame, f"{label}  ({confidence*100:.0f}%)", (22, 45),
               cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv.LINE_AA)
    return frame


def process_video(path, engine, holistic, args):
    cap = cv.VideoCapture(path)
    if not cap.isOpened():
        print(f"cannot open: {path}")
        return "next"
    fps = cap.get(cv.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(
            OUTPUT_DIR, f"{datetime.now():%Y%m%d_%H%M%S}_gesture.mp4")
        writer = cv.VideoWriter(out_path, cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        print(f"  saving -> {out_path}")

    engine.reset()
    verdict, frame_i, label, confidence = "next", 0, "idle", 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_i += 1
        if args.stride > 1 and frame_i % args.stride:
            continue

        res = holistic.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        label, confidence = engine.process_holistic(res)
        annotate(frame, label, confidence)

        if writer is not None:
            writer.write(frame)
        if not args.no_show:
            cv.imshow("Gesture (video)", frame)
            key = cv.waitKey(1) & 0xFF
            if key == ord(' '):
                while (cv.waitKey(30) & 0xFF) != ord(' '):
                    pass
            elif key == ord('n'):
                break
            elif key in (27, ord('q')):
                verdict = "quit"
                break

    cap.release()
    if writer is not None:
        writer.release()
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None,
                    help="video file or folder (default: repo videos/)")
    ap.add_argument("--save", action="store_true", help="write annotated mp4 to outputs/")
    ap.add_argument("--no-show", action="store_true", help="headless (implies --save)")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG)
    args = ap.parse_args()
    if args.no_show:
        args.save = True

    videos = collect_videos(args.input)
    if not videos:
        print(f"No videos found under {args.input or REPO_VIDEOS}")
        return
    print(f"{len(videos)} video(s) queued")

    engine = GestureEngine(checkpoint=args.checkpoint, model_config=args.model_config)
    mp_holistic = mp.solutions.holistic

    for path in videos:
        print(f"\n{os.path.relpath(path)}")
        # fresh Holistic per video: temporal tracking must not leak across files
        with mp_holistic.Holistic(model_complexity=1,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5) as holistic:
            if process_video(path, engine, holistic, args) == "quit":
                break
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
