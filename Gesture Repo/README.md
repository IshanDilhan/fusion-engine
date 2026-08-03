# Gesture Recognition (v2)

Real-time hand/body gesture recognition for the adaptive HRI system.
MediaPipe Holistic landmarks (pose + both hands) are turned into 185-dim
per-frame features, a 32-frame window feeds a small temporal network
(BiGRU / TCN / TinyTransformer), and the engine emits one of **8 classes**:

`idle · wave · point · thumbs_up · thumbs_down · beckoning · raise_hand · both_hands_up`

Unlike the removed v1 (static hand-pose MLP + hand-written motion rules),
static and dynamic gestures are learned **uniformly** from keypoint sequences
— no rules, real metrics for every class. Targeted at **Jetson Orin Nano**.

**Design spec & HPC runbook (source of truth):
[`docs/GESTURE_V2_DESIGN_AND_HPC_GUIDE.md`](../../docs/GESTURE_V2_DESIGN_AND_HPC_GUIDE.md)**

## Results (what to trust)

Two very different benchmarks exist for this model — **do not compare them
directly**:

| Benchmark | What it measures | Current checkpoint (`best_TCN.pth`, trained 2026-07-15) |
|---|---|---|
| `checkpoints/model_config.json` `best_val` / `reports/evaluation/TCN/EVALUATION_REPORT.md` | Held-out split of the **training data itself** (Jester + NTU + custom landmark sequences) | 93.2% acc / 92.8% macro-F1 — but this is stale, see caveat below |
| Real intent-dataset video, 205-clip mixed subset | Held-out subset + some train-subject "coverage" clips | 74.1% acc / 72.0% macro-F1 |
| **Real intent-dataset video, all 1,061 clips, test-subjects only** | Cleanest generalization estimate — actors never in gesture's training data at all | **84.1% acc / 82.9% macro-F1** |

The real-video number is the one that matters for fusion, and it depends a
lot on which subjects you score: on the **strictly held-out test subjects**
(P03/05/07/08/09 — actors never touching this model's training data at all,
since gesture trains on Jester/NTU/custom landmark datasets fully disjoint
from this project's video) it's actually **84.1% accuracy / 82.9% macro-F1**,
better than the mixed 205-clip subset first suggested. Per-class on test
subjects: `thumbs_down` 93% F1, `thumbs_up` 91%, `both_hands_up` 90% (82%
recall), `point` 89% F1 (86% recall!), `idle` 77%, `wave` 57% (small n=3).
`beckoning`/`raise_hand` have zero test-subject support — those two gestures
were only recorded in scenarios that happened to fall entirely in
train/val subjects, so their generalization is currently unverified.

**`point` and `both_hands_up` used to be dead classes (0% recall each)** —
the 2026-07-15 checkpoint fixed this by adding real training data for them
(see the `*_pre_bhu`/`*_pretune`/`*_prev` checkpoint variants, milestones
from that fix). Remaining known weak spots:
- **`point` still fails when there's no motion in the gesture** — a
  static/seated point (e.g. pointing while writing, scenario `S03_F05`)
  reads as `idle` in aggregate, while a point made while walking or standing
  generalizes fine (86% test-subject recall overall). It's specifically the
  *stationary* variant that's weak, not point as a class.
- **`wave` recall is weak everywhere** (both classroom and kitchen scenarios,
  57% F1 on test subjects, small sample) — worth checking whether wave's
  training-data share shrank in the pass that added both_hands_up/point.
- **`both_hands_up` at one kitchen scenario (`S26_F02`) is subject-specific,
  not universal**: train-subject clips there score 100% (25/25) while
  test-subject clips score 29% (mostly falling to `idle`). This is a genuine
  actor-generalization gap for gesture at that scenario — *not* the same
  root cause as the motion model's failure on the same clips (motion fails
  there for train and test subjects alike, i.e. it's an environment/recording
  issue for motion, but an unseen-actor issue for gesture). Don't assume one
  fix addresses both.

> **`reports/evaluation/TCN/EVALUATION_REPORT.md` and `REALWORLD_REPORT.md`
> are stale** — both still say "Generated: 2026-07-12" and describe the
> *previous* checkpoint. The checkpoint on disk (`best_TCN.pth`) was replaced
> 2026-07-15 without regenerating them. Re-run `scripts/evaluate.py` before
> trusting anything in those files.

## Data

Trained on public datasets + a small custom set (guide §4):

| Source | Classes contributed |
|---|---|
| 20BN-Jester | thumbs_up/down, wave, beckoning (proxy), idle negatives |
| NTU RGB+D 120 | wave, point, thumbs_up/down, both_hands_up, idle negatives |
| Custom recordings | beckoning, raise_hand (+ live test set for ALL classes) |

Raw datasets live outside the repo — set `GESTURE_DATA_ROOT` (defaults to
`data/raw/`), laid out per guide §4.

## Pretrained model (use without retraining)

The deployed weights are published as a versioned [GitHub Release](https://github.com/KesharaGunathilaka/adaptive-multimodal-hri/releases)
(kept out of git). Fetches both `best_TCN.pth` and `model_config.json` —
they must ship together, the config pins the exact architecture/labels/window:

```bash
python scripts/download_model.py                       # latest release
python scripts/download_model.py --tag gesture-v2.1     # a specific version
```

### Publishing a new model version (maintainers)

1. Tag the commit: `git tag gesture-vX.Y && git push origin gesture-vX.Y`.
2. On GitHub: **Releases → Draft a new release →** choose that tag, add
   notes (metrics from "Results" above), and **attach both `best_TCN.pth`
   and `model_config.json`** as release assets → Publish.
3. `download_model.py` then serves both automatically (it points at the latest release).

## Folder structure

```
gesture/
├── config.py               # classes, paths, feature/window spec, engine params
├── src/                    # importable library
│   ├── features.py         # landmark -> 185-dim features, normalization, augmentation
│   ├── data.py             # dataset mapping tables, sequence dataset, splits I/O
│   ├── models.py           # BiGRU / TCN / TinyTransformer (< 1M params each)
│   ├── training.py         # shared fit/eval recipe (early stop on macro-F1)
│   └── engine.py           # GestureEngine: rolling window + EMA + debounce
├── scripts/                # runnable pipeline stages
│   ├── extract_landmarks.py  # Stage 0  MediaPipe over datasets -> .npz (CPU, shardable)
│   ├── prepare_data.py       # Stage 0.5 split index CSVs + data report
│   ├── compare_models.py     # Stage 1  compare architectures -> pick winner
│   ├── train.py              # Stage 2  full training (+ model_config.json)
│   ├── tune.py               # Stage 3  optuna hyper-parameter tuning
│   ├── evaluate.py           # Stage 4  test + live-test + latency report
│   └── download_model.py     # fetches best_TCN.pth + model_config.json from a GitHub Release
├── inference/
│   ├── realtime_realsense.py # live RealSense / webcam dashboard
│   └── video.py              # video files -> annotated mp4
├── data/index/             # split CSVs (committed); data/raw & landmarks git-ignored
├── checkpoints/            # weights (git-ignored; published as GitHub Releases)
└── reports/                # generated stage reports
```

## Pipeline (run from this folder)

```bash
python scripts/extract_landmarks.py --dataset all      # shardable: --shard i/N
python scripts/prepare_data.py
python scripts/compare_models.py
python scripts/train.py --model <winner>
python scripts/tune.py --model <winner> --trials 40
python scripts/train.py --model <winner> --use-tuned
python scripts/evaluate.py
```

Heavy stages run on the HPC — SLURM templates and setup in the guide (§9).

## Real-time inference

```bash
python inference/realtime_realsense.py                 # RealSense, webcam fallback
python inference/video.py --input ../../videos --save
```

## Main script integration (how to call)

```python
import cv2 as cv
import mediapipe as mp
from modalities.gesture.src.engine import GestureEngine

holistic = mp.solutions.holistic.Holistic(model_complexity=1)
gesture_detector = GestureEngine()

cap = cv.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    res = holistic.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

    # (stable_label_str, confidence_float) — same call style as MotionEngine
    gesture, confidence = gesture_detector.process_holistic(res)
    print(f"Gesture: {gesture} ({confidence:.2f})")
```

`GestureEngine` buffers ~2 s of frames, resamples to the model window,
smooths the softmax with EMA (α = 0.25) and only emits a non-idle label once
it has won for 300 ms at ≥ 0.60 confidence — so the fusion layer sees stable
intents, not per-frame flicker. Call `engine.reset()` when the tracked person
changes.

## Setup

Use the repo `.venv` (PyTorch + mediapipe): `pip install -r requirements.txt`
from the repo root. Trained weights ship as GitHub Releases (`gesture-v2.x`)
— drop `best_*.pth` + `model_config.json` into `checkpoints/`.
