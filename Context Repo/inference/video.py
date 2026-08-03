"""Run the context model on captured video files.

Discovers videos recursively under a folder (default: repo `videos/`), runs the
full context pipeline (CLIP scene + SmolVLM2 situation analysis) or the scene
classifier alone, shows an annotated preview, and can save annotated clips + a
per-frame JSON log. When a video lives under a folder named after a known scene
(e.g. videos/Classroom/...), that name is used as ground truth to report scene
accuracy.

Examples (run from repo root or this folder):
    python inference/video.py                          # full context, all videos/
    python inference/video.py --mode scene --input ../../videos/Classroom
    python inference/video.py --save --stride 5

Controls: 'n' next video, space pause, 'q'/ESC quit.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modalities.context.scene_classification.config import SCENE_LABELS
from modalities.context.scene_classification.src.classifier import create_scene_classifier
from modalities.context.src.pipeline import ContextPipeline, _draw_overlay

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
KNOWN_SCENES = set(SCENE_LABELS)
MODULE_ROOT = Path(__file__).resolve().parents[1]  # modalities/context


def find_videos(root):
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS)


def ground_truth_scene(path):
    """First path component that matches a known scene name (case-insensitive)."""
    for part in path.parts:
        if part.lower() in KNOWN_SCENES:
            return part.lower()
    return None


def make_processor(mode):
    """Return (process(frame, fps) -> (annotated, info, pred_scene), reset, close)."""
    if mode == "context":
        # Sync VLM: deterministic for offline video (analyses every Nth frame).
        pipeline = ContextPipeline(vlm_async=False)

        def proc(frame, fps):
            annotated, state = pipeline.process_frame(frame)
            _draw_overlay(annotated, state, fps)
            return annotated, state.to_dict(), state.scene

        return proc, pipeline.reset, pipeline.close

    if mode == "scene":
        scene = create_scene_classifier()

        def proc(frame, fps):
            r = scene.predict(frame)
            color = (0, 255, 0) if r["label"] != "uncertain" else (0, 165, 255)
            cv2.putText(frame, f"scene: {r['label']} ({r['confidence']:.2f})  {fps:.1f} FPS",
                        (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            return frame, r, r["label"]

        return proc, scene.reset, lambda: None

    raise ValueError(f"Unknown mode: {mode}")


def _jsonable(info):
    return json.loads(json.dumps(info, default=str))


def run_video(video_path, proc, args, gt_scene, out_dir):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ! could not open {video_path}")
        return None

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer, log = None, []
    if args.save:
        out_dir.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_dir / f"{video_path.stem}_{args.mode}.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (width, height))

    frame_idx, correct, scored = 0, 0, 0
    prev, fps, paused, quit_all = time.time(), 0.0, False, False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if args.stride > 1 and frame_idx % args.stride != 0:
                continue

            annotated, info, pred_scene = proc(frame, fps)
            now = time.time(); dt = now - prev; prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 / dt

            if gt_scene and pred_scene is not None and pred_scene != "uncertain":
                scored += 1
                if pred_scene == gt_scene:
                    correct += 1

            if writer is not None:
                writer.write(annotated)
            if args.save:
                log.append({"frame": frame_idx, **_jsonable(info)})

            if not args.no_show:
                if gt_scene:
                    cv2.putText(annotated, f"GT: {gt_scene}", (15, height - 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.imshow("context", annotated)

        if not args.no_show:
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                quit_all = True; break
            if key == ord("n"):
                break
            if key == ord(" "):
                paused = not paused

    cap.release()
    if writer is not None:
        writer.release()
    if args.save:
        (out_dir / f"{video_path.stem}_{args.mode}.json").write_text(
            json.dumps(log), encoding="utf-8")

    acc = (correct / scored) if scored else None
    return {"frames": frame_idx, "scene_correct": correct, "scene_scored": scored,
            "accuracy": acc, "quit": quit_all}


def main():
    parser = argparse.ArgumentParser(description="Run the context model on videos.")
    parser.add_argument("--input", default=str(REPO_ROOT / "videos"), help="Video file or folder.")
    parser.add_argument("--mode", default="context", choices=["context", "scene"])
    parser.add_argument("--save", action="store_true", help="Write annotated mp4 + JSON log.")
    parser.add_argument("--no-show", action="store_true", help="Don't open a preview window.")
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame.")
    args = parser.parse_args()

    root = Path(args.input)
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()

    videos = find_videos(root)
    if not videos:
        print(f"No videos found under: {root}")
        sys.exit(1)

    print(f"Mode: {args.mode}  |  {len(videos)} video(s) under {root}")
    proc, reset, close = make_processor(args.mode)
    out_root = MODULE_ROOT / "runs" / "context_test" / args.mode

    totals = {"correct": 0, "scored": 0}
    try:
        for i, video in enumerate(videos, 1):
            gt = ground_truth_scene(video)
            print(f"[{i}/{len(videos)}] {video.name}  (GT scene: {gt or '-'})")
            reset()
            rel = video.parent.relative_to(root) if root in video.parents else Path()
            result = run_video(video, proc, args, gt, out_root / rel)
            if result is None:
                continue
            if result["accuracy"] is not None:
                print(f"    scene acc: {result['accuracy']:.1%} "
                      f"({result['scene_correct']}/{result['scene_scored']} confident frames)")
                totals["correct"] += result["scene_correct"]
                totals["scored"] += result["scene_scored"]
            if result["quit"]:
                print("Quit requested.")
                break
    finally:
        close()
        cv2.destroyAllWindows()

    if totals["scored"]:
        print(f"\nOverall scene accuracy: {totals['correct'] / totals['scored']:.1%} "
              f"({totals['correct']}/{totals['scored']} confident frames)")


if __name__ == "__main__":
    main()
