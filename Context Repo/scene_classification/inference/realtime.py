"""
Standalone real-time zero-shot scene (environment) classification from a camera.
Uses an Intel RealSense camera if connected, else the default laptop webcam.

SELF-CONTAINED: no project imports and no trained weights file — CLIP weights
auto-download on first run (~350 MB, cached). Only these pip packages:
    pip install torch torchvision open_clip_torch opencv-python pillow numpy
    # pyrealsense2 is optional (only for a RealSense camera)

Run:
    python realtime.py
    python realtime.py --camera 1

Method: camera frame -> CLIP image embedding -> cosine similarity against
per-class prompt ensembles -> scene label + confidence, smoothed over a
rolling window. Adding a scene class = adding an entry to SCENE_PROMPTS below.
"""
import argparse
from collections import deque

import cv2
import numpy as np
import torch
from PIL import Image

try:
    import pyrealsense2 as rs
    _RS_AVAILABLE = True
except ImportError:
    _RS_AVAILABLE = False

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


def _try_start_realsense():
    if not _RS_AVAILABLE:
        return None, None
    try:
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        pipeline.start(cfg)
        return pipeline, rs.align(rs.stream.color)
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="Webcam index (fallback).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading CLIP (downloads ~350 MB on first run)...")
    model, preprocess, text_embs = load_clip(device)
    print("=" * 60)
    print(f" model:   {CLIP_MODEL} ({CLIP_PRETRAINED}), zero-shot")
    print(f" device:  {device}")
    print(f" labels:  {SCENE_LABELS}")
    print("=" * 60)

    pipeline, align = _try_start_realsense()
    if pipeline is not None:
        print("RealSense camera connected.")
        window_title, cap = "Scene Classification (RealSense)", None
    else:
        print(f"RealSense not available — using webcam (index {args.camera}).")
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print("Error: no camera found.")
            return
        window_title = "Scene Classification (Webcam)"

    prob_history = deque(maxlen=SMOOTH_WINDOW)
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
                    print("Failed to read from webcam.")
                    break

            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            with torch.no_grad():
                img = model.encode_image(preprocess(pil).unsqueeze(0).to(device)).float()
                img = img / img.norm(dim=-1, keepdim=True)
                probs = torch.softmax(100.0 * img @ text_embs.T, dim=1)[0].cpu().numpy()
            prob_history.append(probs)
            avg = np.mean(prob_history, axis=0)
            idx = int(avg.argmax())
            conf = float(avg[idx])
            label = SCENE_LABELS[idx] if conf >= CONF_THRESHOLD else "uncertain"
            color = (0, 255, 0) if label != "uncertain" else (0, 165, 255)
            cv2.putText(frame, f"{label}: {conf * 100:.1f}%", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

            cv2.imshow(window_title, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if pipeline is not None:
            pipeline.stop()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
