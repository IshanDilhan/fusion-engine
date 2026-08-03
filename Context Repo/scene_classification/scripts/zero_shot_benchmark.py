"""
Zero-shot scene classification benchmark: CLIP / SigLIP vs the trained CNN.

Samples frames from the captured videos (videos/Classroom, videos/Kitchen) and
compares, on the exact same frames:
  1. EfficientNet-B0 baseline  (trained on Places365, checkpoints/)
  2. CLIP ViT-B/32             (zero-shot, prompt ensemble)
  3. SigLIP ViT-B/16           (zero-shot, prompt ensemble)

Also probes the "no scene content" failure mode: a third prompt set describing
a face close-up lets the zero-shot models ABSTAIN, and we report how accuracy
changes when those frames are excluded.

Outputs: reports/zero_shot/ZERO_SHOT_REPORT.md + results.csv

Run (from scene_classification/):
    python scripts/zero_shot_benchmark.py                     # all videos, 4 frames each
    python scripts/zero_shot_benchmark.py --max-videos 50     # quick pass
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows consoles default to cp1252; force UTF-8 so prints don't crash the run.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from config import (
    ABSTAIN_PROMPTS,
    CHECKPOINT_DIR,
    REPORT_DIR,
    SCENE_LABELS,
    SCENE_PROMPTS as PROMPTS,
)
from src.models import build_model
from src.transforms import get_test_transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = os.path.join(REPORT_DIR, "zero_shot")

# Repo root: scene_classification/scripts -> repo
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "..", "..", "..", ".."))
# Ground-truth video folders: only classes we have captured footage for.
# The zero-shot classifier still scores against ALL of SCENE_LABELS (imported
# from config) — this is a real regression check: adding more classes gives
# CLIP more ways to be confused, so classroom/kitchen accuracy must be
# re-verified whenever the vocabulary grows, not just assumed.
VIDEO_DIRS = {
    "classroom": os.path.join(REPO_ROOT, "videos", "Classroom"),
    "kitchen": os.path.join(REPO_ROOT, "videos", "Kitchen"),
}


# ── Model wrappers: each returns (probs over SCENE_LABELS, abstain_prob) ──
class BaselineCNN:
    name = "EfficientNet-B0 (trained)"

    def __init__(self):
        import json
        ckpt = os.path.join(CHECKPOINT_DIR, "best_EfficientNet_B0.pth")
        # The CNN's classes come from training (classes.json), NOT SCENE_LABELS —
        # the zero-shot vocabulary can grow beyond what the CNN was trained on.
        with open(os.path.join(CHECKPOINT_DIR, "classes.json"), encoding="utf-8") as f:
            self.classes = json.load(f)
        self.model = build_model("EfficientNet-B0", num_classes=len(self.classes),
                                 pretrained=False)
        self.model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        self.model.to(DEVICE).eval()
        self.tf = get_test_transforms()

    @torch.no_grad()
    def predict(self, pil_images):
        batch = torch.stack([self.tf(im) for im in pil_images]).to(DEVICE)
        probs = torch.softmax(self.model(batch), dim=1).cpu().numpy()
        return probs, np.zeros(len(pil_images))  # CNN cannot abstain


class ZeroShot:
    """Shared logic for CLIP / SigLIP: image-text similarity vs prompt ensembles."""

    def __init__(self, name, encode_image, encode_text, preprocess):
        self.name = name
        self._encode_image = encode_image
        self._preprocess = preprocess

        # Pre-compute one averaged, normalized text embedding per class.
        embs = []
        for label in SCENE_LABELS + ["__abstain__"]:
            prompts = ABSTAIN_PROMPTS if label == "__abstain__" else PROMPTS[label]
            e = encode_text(prompts)
            e = e / e.norm(dim=-1, keepdim=True)
            embs.append(e.mean(dim=0, keepdim=True))
        self.text_embs = torch.cat(embs)  # (num_classes + 1, dim)
        self.text_embs = self.text_embs / self.text_embs.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def predict(self, pil_images):
        batch = torch.stack([self._preprocess(im) for im in pil_images]).to(DEVICE)
        img = self._encode_image(batch)
        img = img / img.norm(dim=-1, keepdim=True)
        logits = 100.0 * img @ self.text_embs.T          # (N, classes+1)
        probs_all = torch.softmax(logits, dim=1).cpu().numpy()
        scene_logits = logits[:, : len(SCENE_LABELS)]
        probs = torch.softmax(scene_logits, dim=1).cpu().numpy()  # forced choice
        return probs, probs_all[:, -1]                    # abstain prob = face class


def load_openai_clip():
    import clip
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    model.eval()

    def encode_text(prompts):
        return model.encode_text(clip.tokenize(prompts).to(DEVICE)).float()

    return ZeroShot("CLIP ViT-B/32 (zero-shot)",
                    lambda b: model.encode_image(b).float(), encode_text, preprocess)


def load_siglip():
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP", pretrained="webli")
    tokenizer = open_clip.get_tokenizer("ViT-B-16-SigLIP")
    model.to(DEVICE).eval()

    def encode_text(prompts):
        return model.encode_text(tokenizer(prompts).to(DEVICE)).float()

    return ZeroShot("SigLIP ViT-B/16 (zero-shot)",
                    lambda b: model.encode_image(b).float(), encode_text, preprocess)


# ── Data ─────────────────────────────────────────────────────────────────
def collect_videos(folder):
    exts = {".mp4", ".avi", ".mov", ".mkv"}
    found = []
    for dirpath, _, files in os.walk(folder):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in exts:
                found.append(os.path.join(dirpath, f))
    return sorted(found)


def sample_frames(video_path, n):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total > 0:
        for idx in np.linspace(0, total - 1, n).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


@torch.no_grad()
def measure_latency(predict_fn, n=30, warmup=5):
    dummy = [Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))]
    for _ in range(warmup):
        predict_fn(dummy)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        predict_fn(dummy)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000.0


# ── Benchmark ────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-per-video", type=int, default=4)
    ap.add_argument("--max-videos", type=int, default=0,
                    help="If >0, limit videos per class (quick pass).")
    ap.add_argument("--abstain-threshold", type=float, default=0.5,
                    help="Abstain when the face-closeup prob exceeds this.")
    ap.add_argument("--models", default="baseline,clip,siglip",
                    help="Comma list: baseline, clip, siglip.")
    args = ap.parse_args()

    print(f"Device: {DEVICE}")
    print("Loading models...")
    loaders = {"baseline": BaselineCNN, "clip": load_openai_clip, "siglip": load_siglip}
    models = [loaders[k.strip()]() for k in args.models.split(",") if k.strip()]
    for m in models:
        print(f"  ✓ {m.name}")

    # results[model][gt] = list of (pred_idx, abstain_prob) ; video-level votes too
    frame_records = {m.name: [] for m in models}   # (gt_idx, pred_idx, abstain_p)
    video_records = {m.name: [] for m in models}   # (gt_idx, majority_pred_idx)

    for gt_label, folder in VIDEO_DIRS.items():
        gt_idx = SCENE_LABELS.index(gt_label)
        videos = collect_videos(folder)
        if args.max_videos > 0:
            videos = videos[: args.max_videos]
        print(f"\n{gt_label}: {len(videos)} videos")
        for vi, vp in enumerate(videos, 1):
            frames = sample_frames(vp, args.frames_per_video)
            if not frames:
                continue
            for m in models:
                probs, abstain_p = m.predict(frames)
                preds = probs.argmax(axis=1)
                for p, a in zip(preds, abstain_p):
                    frame_records[m.name].append((gt_idx, int(p), float(a)))
                video_records[m.name].append((gt_idx, int(np.bincount(preds).argmax())))
            if vi % 100 == 0:
                print(f"  [{vi}/{len(videos)}]")

    # ── Metrics ──────────────────────────────────────────────────────────
    rows = []
    for m in models:
        fr = np.array(frame_records[m.name])          # (N, 3)
        vr = np.array(video_records[m.name])          # (V, 2)
        gt, pred, abst = fr[:, 0].astype(int), fr[:, 1].astype(int), fr[:, 2]

        def acc(mask):
            return float((pred[mask] == gt[mask]).mean()) if mask.any() else float("nan")

        row = {"model": m.name}
        for i, lbl in enumerate(SCENE_LABELS):
            row[f"{lbl}_acc"] = round(acc(gt == i) * 100, 1)
        row["overall_acc"] = round(acc(np.ones_like(gt, bool)) * 100, 1)
        row["video_majority_acc"] = round(
            float((vr[:, 1] == vr[:, 0]).mean()) * 100, 1)

        # Abstain-aware: exclude frames the model thinks are face close-ups.
        keep = abst < args.abstain_threshold
        row["abstain_rate"] = round(float((~keep).mean()) * 100, 1)
        row["acc_after_abstain"] = round(acc(keep) * 100, 1)

        row["latency_ms"] = round(measure_latency(m.predict), 1)
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "results.csv"), index=False)

    n_frames = len(frame_records[models[0].name])
    report = [
        "# Zero-shot vs Trained CNN — Scene Classification on Captured Videos\n",
        f"- Frames evaluated: {n_frames} ({args.frames_per_video}/video)",
        f"- Classes: {SCENE_LABELS}",
        f"- Zero-shot prompt ensembles + face-closeup abstain probe "
        f"(threshold {args.abstain_threshold})",
        f"- Latency measured on this machine ({DEVICE}), batch=1 — relative "
        "comparison only, not Jetson numbers.\n",
        "## Results\n",
        df.to_markdown(index=False),
        "",
        "\n## Reading the numbers\n",
        "- `*_acc`: frame-level accuracy per ground-truth folder (forced choice).",
        "- `video_majority_acc`: majority vote over each clip's sampled frames "
        "(approximates deployed temporal smoothing).",
        "- `abstain_rate`: frames the zero-shot model judged to be a face close-up "
        "with no scene content.",
        "- `acc_after_abstain`: accuracy on the remaining frames — how much of the "
        "error is 'scene not visible' vs actual misclassification.",
    ]
    with open(os.path.join(OUT_DIR, "ZERO_SHOT_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("\n" + "=" * 80)
    print(df.to_string(index=False))
    print(f"\nReport: {os.path.join(OUT_DIR, 'ZERO_SHOT_REPORT.md')}")


if __name__ == "__main__":
    main()
