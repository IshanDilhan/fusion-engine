"""
Standalone zero-shot scene (environment) classification on a video file (or a
folder of videos). Defaults to scanning the repo's videos/ folder.

SELF-CONTAINED: no project imports and no trained weights file — CLIP weights
auto-download on first run (~350 MB, cached). Only these pip packages:
    pip install torch torchvision open_clip_torch opencv-python pillow numpy

Run:
    python video.py                                    # batch the repo videos/ folder
    python video.py --video myclip.mp4
    python video.py --videos-dir ./my_videos            # batch a different folder

Method: frame -> CLIP image embedding -> cosine similarity against per-class
prompt ensembles -> scene label + confidence, smoothed over a rolling window.
Adding a scene class = adding an entry to SCENE_PROMPTS below (no retraining).
"""
import argparse
import os
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# ── Configuration (edit prompts to add/change scene classes) ──────────────
CLIP_MODEL = "ViT-B-32-quickgelu"
CLIP_PRETRAINED = "openai"
SCENE_PROMPTS = {
    "classroom": [
        "a photo of a classroom",
        "a photo taken inside a classroom",
        "a classroom with desks and chairs",
        "a lecture room in a university",
        "students sitting in a classroom",
        "a whiteboard at the front of a classroom",
    ],
    "kitchen": [
        "a photo of a kitchen",
        "a photo taken inside a kitchen",
        "a kitchen with cabinets and appliances",
        "a person cooking in a kitchen",
        "a kitchen countertop with utensils",
        "a stove and a sink in a kitchen",
    ],
    "hospital": [
        "a photo of a hospital ward",
        "a photo taken inside a hospital",
        "a hospital corridor with medical equipment",
        "a nurse or doctor beside a hospital bed",
        "a patient room in a hospital",
        "a medical clinic examination room",
    ],
    "cloth_store": [
        "a photo of a clothing store",
        "a photo taken inside a clothes shop",
        "racks of clothes hanging in a store",
        "shelves of folded clothing in a shop",
        "a fitting room in a clothing store",
        "a boutique with clothes on display",
    ],
    "museum": [
        "a photo of a museum",
        "a photo taken inside a museum gallery",
        "exhibits in glass display cases in a museum",
        "paintings on the walls of an art gallery",
        "a museum hall with statues and artifacts",
        "visitors looking at museum exhibits",
    ],
}
SMOOTH_WINDOW = 15
CONF_THRESHOLD = 0.5
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# repo_root/modalities/context/scene_classification/inference -> repo_root
DEFAULT_VIDEOS_DIR = str(Path(SCRIPT_DIR).parents[3] / "videos")

SCENE_LABELS = list(SCENE_PROMPTS.keys())


def load_clip(device):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED)
    model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)

    with torch.no_grad():
        embs = []
        for label in SCENE_LABELS:
            e = model.encode_text(tokenizer(SCENE_PROMPTS[label]).to(device)).float()
            e = e / e.norm(dim=-1, keepdim=True)
            embs.append(e.mean(dim=0, keepdim=True))
        text = torch.cat(embs)
        text_embs = text / text.norm(dim=-1, keepdim=True)
    return model, preprocess, text_embs


@torch.no_grad()
def classify(frame_bgr, model, preprocess, text_embs, device, prob_history):
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    img = model.encode_image(preprocess(pil).unsqueeze(0).to(device)).float()
    img = img / img.norm(dim=-1, keepdim=True)
    probs = torch.softmax(100.0 * img @ text_embs.T, dim=1)[0].cpu().numpy()
    prob_history.append(probs)
    avg = np.mean(prob_history, axis=0)
    idx = int(avg.argmax())
    conf = float(avg[idx])
    label = SCENE_LABELS[idx] if conf >= CONF_THRESHOLD else "uncertain"
    return label, conf


def collect_videos(videos_dir):
    found = []
    for dirpath, _, filenames in os.walk(videos_dir):
        for fname in sorted(filenames):
            if os.path.splitext(fname)[1].lower() in VIDEO_EXTENSIONS:
                found.append(os.path.join(dirpath, fname))
    return sorted(found)


def build_output_path(video_path, videos_dir, out_root):
    rel = os.path.relpath(video_path, videos_dir)
    out_path = os.path.join(out_root, os.path.splitext(rel)[0] + "_scene.mp4")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return out_path


_HINT = "[SPACE] Pause   [N] Next   [P] Prev   [Q] Quit"


def _draw_overlay(frame, paused):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 28), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, _HINT, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (220, 220, 220), 1, cv2.LINE_AA)
    if paused:
        text, font, scale, thick = "PAUSED", cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        cx, cy = (w - tw) // 2, (h + th) // 2
        cv2.putText(frame, text, (cx + 2, cy + 2), font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(frame, text, (cx, cy), font, scale, (0, 220, 255), thick, cv2.LINE_AA)


def process_video(video_path, out_path, model, preprocess, text_embs, device, show):
    """Process one video. Returns: "done" | "next" | "prev" | "quit"."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [skip] Cannot open: {video_path}")
        return "next"

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    prob_history = deque(maxlen=SMOOTH_WINDOW)
    action, paused, display_frame = "done", False, None
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            label, conf = classify(frame, model, preprocess, text_embs, device, prob_history)
            color = (0, 255, 0) if label != "uncertain" else (0, 165, 255)
            cv2.putText(frame, f"{label}: {conf * 100:.1f}%", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            writer.write(frame)
            display_frame = frame.copy()

        if show and display_frame is not None:
            view = display_frame.copy()
            _draw_overlay(view, paused)
            cv2.imshow("Scene Classification (zero-shot)", view)
            key = cv2.waitKey(100 if paused else 30) & 0xFF
            if key == ord("q"):
                action = "quit"; break
            elif key == ord("n"):
                action = "next"; break
            elif key == ord("p"):
                action = "prev"; break
            elif key == ord(" "):
                paused = not paused

    cap.release()
    writer.release()
    return action


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--video", default=None, help="Path to a single input video file.")
    src.add_argument("--videos-dir", default=None,
                     help=f"Folder to scan recursively (default: repo videos/ = {DEFAULT_VIDEOS_DIR}).")
    ap.add_argument("--output", default=None, help="Output path (single-file mode only).")
    ap.add_argument("--out-dir", default="outputs", help="Output root for batch mode.")
    ap.add_argument("--no-show", action="store_true", help="Do not open a preview window.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip videos whose output file already exists.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading CLIP (downloads ~350 MB on first run)...")
    model, preprocess, text_embs = load_clip(device)
    print("=" * 60)
    print(f" model:   {CLIP_MODEL} ({CLIP_PRETRAINED}), zero-shot")
    print(f" device:  {device}")
    print(f" labels:  {SCENE_LABELS}")
    print("=" * 60)

    out_root = os.path.abspath(args.out_dir)

    # ── Single-file mode ─────────────────────────────────────────────────
    if args.video is not None:
        os.makedirs(out_root, exist_ok=True)
        out_path = args.output or os.path.join(
            out_root, os.path.splitext(os.path.basename(args.video))[0] + "_scene.mp4")
        print(f"Writing to {out_path}")
        process_video(args.video, out_path, model, preprocess, text_embs, device,
                      show=not args.no_show)
        cv2.destroyAllWindows()
        print(f"Saved: {out_path}")
        return

    # ── Batch mode (default: repo videos/ folder) ───────────────────────
    videos_dir = os.path.abspath(args.videos_dir or DEFAULT_VIDEOS_DIR)
    videos = collect_videos(videos_dir)
    if not videos:
        print(f"No video files found under: {videos_dir}")
        return

    print(f"Found {len(videos)} video(s) under {videos_dir}")
    done = skipped = idx = 0
    while idx < len(videos):
        video_path = videos[idx]
        out_path = build_output_path(video_path, videos_dir, out_root)
        label = os.path.relpath(video_path, videos_dir)
        if args.skip_existing and os.path.exists(out_path):
            print(f"[{idx+1}/{len(videos)}] skip (exists): {label}")
            skipped += 1; idx += 1
            continue
        print(f"[{idx+1}/{len(videos)}] {label}")
        action = process_video(video_path, out_path, model, preprocess, text_embs, device,
                               show=not args.no_show)
        cv2.destroyAllWindows()
        if action == "quit":
            print("  Quit by user."); break
        elif action == "prev":
            idx = max(0, idx - 1)
        else:
            done += 1; idx += 1

    cv2.destroyAllWindows()
    print(f"\nDone. {done} processed, {skipped} skipped.")


if __name__ == "__main__":
    main()
