# Context Model

Situational-context perception for the adaptive HRI system. Two components
produce one structured **`ContextState`** for the downstream policy / fusion
module:

1. **Scene classification** — which environment (`classroom` / `kitchen`),
   zero-shot CLIP image-text matching (99.5% on captured clips; the trained
   EfficientNet-B0 CNN is kept as the evaluated baseline).
2. **VLM situation analysis** — a small vision-language model (SmolVLM2-500M)
   reads the whole frame and reports the situation in structured form: how many
   people, what the main person is doing, what they are focused on, the key
   objects, and a one-sentence summary.

```
ContextState {
  scene, scene_confidence,            # CLIP, every 5th frame
  vlm: { people, activity, attention, # SmolVLM2, every ~2 s (async)
         objects[], summary },
  timestamp
}
```

Targeted at **Jetson Orin Nano 8GB**. The camera loop never blocks on the VLM:
in realtime mode the VLM analyses a frame in a background thread at most every
`VLM_INTERVAL_SEC`, and each ContextState carries the latest finished analysis.

> The earlier geometric sub-models (YOLO object detection + MediaPipe gaze
> estimation with rule-based fusion) were replaced by the VLM and removed —
> they remain recoverable from git history if needed.

## Folder structure

```
context/
├── config.py                  # pipeline params: scene throttle, VLM model + intervals
├── README.md
├── src/
│   ├── context_state.py       # the ContextState / VLMContext contract
│   ├── vlm.py                 # SmolVLM2 wrapper: frame -> structured VLMContext
│   └── pipeline.py            # ContextPipeline: CLIP scene + VLM -> ContextState
├── inference/
│   ├── realtime.py            # webcam — full context (async VLM)
│   └── video.py               # video files — --mode context|scene (+ scene accuracy)
│
└── scene_classification/      # scene sub-model (mirrors the emotion layout)
    ├── config.py              # incl. SCENE_BACKEND (clip/cnn) + SCENE_PROMPTS
    ├── src/                   # models, transforms, data, engine, classifier, zero_shot
    ├── scripts/               # compare_models, train, tune, evaluate, zero_shot_benchmark
    ├── inference/             # standalone realtime.py + video.py (self-contained CLIP)
    ├── checkpoints/           # CNN weights (git-ignored) + classes.json
    └── reports/               # comparison / training / evaluation / zero_shot reports
```

## Setup

Use the repo's `.venv` (CUDA-enabled PyTorch + open_clip_torch + transformers).
Model weights auto-download on first run and are cached locally (~350 MB CLIP,
~1 GB SmolVLM2); after that everything runs fully offline (set `HF_HUB_OFFLINE=1`
on a device without internet).

## Run the context model

```bash
# Live webcam (CLIP scene + SmolVLM2, async)
python modalities/context/inference/realtime.py

# Captured videos — full context; VLM analyses every 60th processed frame
python modalities/context/inference/video.py --input videos/Kitchen --save

# Scene classifier only, with accuracy vs the folder name
python modalities/context/inference/video.py --mode scene --input videos/Classroom
```

Flags: `--save` (annotated mp4 + per-frame JSON log), `--no-show` (headless),
`--stride N` (process every Nth frame).

## Scene classification details

- **Deployed backend: zero-shot CLIP** (`ViT-B/32`). Frames are matched against
  per-class prompt ensembles (`SCENE_PROMPTS` in `scene_classification/config.py`).
  Adding an environment = adding a prompt list, no retraining. A face-close-up
  abstain probe reports "uncertain" when no scene content is visible.
- **CNN baseline** (EfficientNet-B0, 2-class) remains for comparison; switch via
  `SCENE_BACKEND = "cnn"`. Benchmark on 1,109 captured clips:
  CLIP 99.5% vs CNN 82.2% (`reports/zero_shot/ZERO_SHOT_REPORT.md`).
  Independently re-confirmed 2026-07-16 on the project's real intent-dataset
  video (`data/`, 4 frames/clip, mean-softmax): **98.0% / 98.7% macro-F1
  across all 1,061 clips** (98.8%/99.3% on strictly held-out test subjects),
  99.5% on an earlier 205-clip held-out subset, 100% on an earlier 168-clip
  subset — consistently near-perfect across every check, no known failure
  modes. This is the strongest of the four modalities.
- The full training pipeline for the CNN (compare → train → tune → evaluate)
  lives in `scene_classification/scripts/`.
- Standalone self-contained inference scripts (no project imports):
  `scene_classification/inference/realtime.py` and `video.py` — the latter
  defaults to scanning the repo `videos/` folder.

## Notes

- **VLM latency**: SmolVLM2-500M takes on the order of a second per analysis —
  hence the async schedule; the scene label stays fresh at frame rate.
- **Jetson deployment**: export the CLIP image encoder to TensorRT and ship
  precomputed prompt embeddings; run SmolVLM2 quantized (INT8/INT4). Both fit
  the Orin Nano 8GB budget alongside the other modality models.
