# Emotion Recognition (RAF-DB + real-world fine-tune)

Facial emotion recognition for the adaptive HRI system. A lightweight
ImageNet-pretrained MobileNetV2 classifies a detected face into 7 emotions
(**Surprise, Fear, Disgust, Happy, Sad, Anger, Neutral**), pretrained on
**RAF-DB** then fine-tuned on real-world footage from this project's own
intent dataset (`data/`), targeted at **Jetson Orin Nano** deployment.

## Results (what to trust)

RAF-DB is close-up, curated portraits; this project's footage has subjects
2–5 m from the camera (40–90 px faces), which collapses a RAF-DB-only model's
predictions to Neutral/Happy. Fine-tuning on real crops fixes this:

| Checkpoint | RAF-DB acc/macro-F1 | **Real test-subject acc/macro-F1** |
|---|---|---|
| `best_MobileNetV2.pth` (RAF-DB only) | 84.3% / 76.7% | 58.8% / **38.9%** |
| **`finetuned_MobileNetV2.pth` (deployed)** | 81.9% / 74.3% | 92.5% / **90.1%** |

"Real test-subject" = the held-out subjects (P03/P05/P07/P08/P09 in
`data/annotations/clips.csv`) that were in **neither** fine-tuning nor
checkpoint selection — the only trustworthy generalization estimate. Per-class
on the deployed checkpoint: Anger 100%, Neutral 100%/95%, Surprise 89%,
Disgust 71%, Fear 100% (n=3, small), Happy 78% — no dead classes.

Confirmed 2026-07-16 across the complete dataset (all 1,061 clips) with the
train/val/test split laid bare, showing exactly why "val" numbers overstate
generalization: train 98.3%/97.9% (near-memorized — literally what
fine-tuning saw), val 98.9%/99.3% (checkpoint-selection-inflated — this is
what the model was early-stopped on), **test 92.5%/90.1% (the only number
to trust)**. Still strong even at its lowest, honest estimate.

> **Known footgun, already hit once**: `config.py`'s `DEFAULT_CHECKPOINT` and
> both inference scripts' `DEFAULT_WEIGHTS` are separate hardcoded defaults.
> A folder copy once silently reset all of them to `best_MobileNetV2.pth`
> (the weak baseline), producing a false "this model is bad" reading for
> days. If results ever look surprisingly bad, **print the resolved
> checkpoint path and confirm it's `finetuned_MobileNetV2.pth`** before
> concluding the model regressed.

### The LSTM variant — do not deploy

`checkpoints/{best,finetuned}_MobileNetV2_LSTM.pth` (MobileNetV2 backbone +
per-clip LSTM temporal head, `src/models_lstm.py`) exist but:
- **Their original training script is lost** — the architecture in
  `models_lstm.py` was reverse-engineered from the checkpoint weights
  (verified with a strict `state_dict` load), so it can't be reproduced or
  retrained from a known recipe.
- On the true held-out test subjects, `finetuned_MobileNetV2_LSTM` scores
  **87.5% acc / 82.9% macro-F1 — worse than the plain CNN** (92.5% / 90.1%).
  The 98–99% numbers in `reports/evaluation_realworld_video_sweep/RESULTS.md`
  are **not** a real generalization estimate: that script (and
  `evaluate_realworld_cleaned.py`) only ever scores the `train`/`val` split,
  and `val` (subject P04) is exactly what the checkpoint was early-stopped /
  selected on — those numbers are checkpoint-selection-inflated, not
  held-out. Keep `finetuned_MobileNetV2.pth` (plain CNN, mean-softmax over
  sampled frames) as the deployed model.

## Folder structure

```
emotion/
├── config.py               # paths, classes, default checkpoint (see footgun note above)
├── README.md
├── src/                    # importable library
│   ├── models.py           # model zoo (4 backbones) + helpers
│   ├── models_lstm.py       # MobileNetV2+LSTM — reverse-engineered, do not deploy (see above)
│   ├── transforms.py       # train/eval image transforms
│   ├── data.py             # datasets, loaders, class weights
│   └── engine.py           # two-stage training recipe, metrics, mixup, AMP
├── scripts/                # runnable pipeline stages
│   ├── eda.py                        # Stage 0  EDA
│   ├── compare_models.py             # Stage 1  compare backbones -> pick winner
│   ├── train.py                      # Stage 2  full training of chosen model (RAF-DB)
│   ├── tune.py                       # Stage 3  hyper-parameter tuning
│   ├── evaluate.py                   # Stage 4  RAF-DB test-set evaluation
│   ├── evaluate_realworld_cleaned.py       # real-world eval — train/val split ONLY, see caveat above
│   ├── evaluate_realworld_video_sweep.py   # real-world eval (CNN+LSTM) — train/val split ONLY, see caveat above
│   └── download_model.py             # fetches finetuned_MobileNetV2.pth from a GitHub Release
├── inference/
│   ├── realtime_realsense.py   # live RealSense camera (self-contained)
│   └── video.py                 # video file -> annotated mp4 (self-contained)
├── data/                   # git-ignored: RAF-DB train/1..7, test/1..7
│                           # + realworld/{train,val}/1..7 (real-world face crops)
├── checkpoints/            # saved weights (git-ignored)
├── reports/                # generated reports, CSVs, plots
└── outputs/                # inference video outputs (git-ignored)
```

**Missing from this copy**: `extract_realworld_faces.py` and
`finetune_realworld.py` (the scripts that built `data/realworld/` and
produced `finetuned_MobileNetV2.pth`) and `evaluate_realworld.py` (the
correct held-out-test-subject evaluator) are not present in this folder —
they existed in an earlier version of the pipeline but were dropped by a
folder copy. The fine-tuned checkpoint and its `data/realworld/` crops still
exist and work; only the scripts to *reproduce* that pipeline are gone. If
you need to retrain or re-verify against test subjects, these need to be
rewritten (a leak-free test-subject evaluator can be adapted from
`evaluate_realworld_video_sweep.py` by changing which `data/realworld/`
split(s) it scores against, or ported from this session's scratch eval
scripts).

## Pretrained model (use without retraining)

The deployed weights are published as a versioned [GitHub Release](https://github.com/KesharaGunathilaka/adaptive-multimodal-hri/releases)
(kept out of git so the repo stays lean). Fetch into `checkpoints/`:

```bash
python scripts/download_model.py                       # latest release
python scripts/download_model.py --tag emotion-v3.0     # a specific version
```

### Publishing a new model version (maintainers)

1. Tag the commit: `git tag emotion-vX.Y && git push origin emotion-vX.Y`.
2. On GitHub: **Releases → Draft a new release →** choose that tag, add
   notes (metrics from the table above), and **attach `finetuned_MobileNetV2.pth`**
   as the release asset → Publish.
3. `download_model.py` then serves it automatically (it points at the latest release).

> Earlier releases (`emotion-v1.0`, `emotion-v2.0`) may have shipped
> `best_MobileNetV2.pth` (the RAF-DB-only baseline) — verify which asset an
> old release actually contains before relying on it; only trust a release
> explicitly confirmed to contain `finetuned_MobileNetV2.pth`.

## Setup

Use the repo's `.venv` (CUDA-enabled PyTorch). All commands run **from this
`emotion/` folder**.

```bash
# from repo root
.venv\Scripts\activate          # Windows PowerShell: .\.venv\Scripts\Activate.ps1
cd modalities/emotion
```

Dataset layout (RAF-DB `_aligned` crops, folders named by label 1–7):
`data/train/<1-7>/*.jpg` and `data/test/<1-7>/*.jpg`.

## Pipeline (RAF-DB stages; run in order)

```bash
# Stage 0 — EDA: class distribution, samples, imbalance  -> reports/eda/
python scripts/eda.py

# Stage 1 — compare all 4 backbones, pick best by macro-F1  -> reports/comparison/
python scripts/compare_models.py --stage1_epochs 3 --stage2_epochs 12

# Stage 2 — full training of the chosen model  -> checkpoints/ + reports/training/
python scripts/train.py --model MobileNetV2

# Stage 3 — hyper-parameter tuning  -> reports/tuning/
python scripts/tune.py --model MobileNetV2

# Stage 4 — RAF-DB test-set evaluation  -> reports/evaluation/
python scripts/evaluate.py --model MobileNetV2
```

Real-world fine-tuning (Stages 5–7 in spirit) needs `extract_realworld_faces.py`
and `finetune_realworld.py` rewritten first — see "Missing from this copy" above.

## Inference

```bash
# Live Intel RealSense camera (falls back to webcam)
python inference/realtime_realsense.py --checkpoint finetuned_MobileNetV2.pth

# Annotate a video file (writes to outputs/)
python inference/video.py --video ../../data/raw/clips/classroom/S01_F04/S01_F04_c001.mp4
```

Both scripts are self-contained (no project imports beyond pip packages) and
use MediaPipe close-range face detection + plain 224×224 ImageNet
normalization, matching the fine-tuning preprocessing. Both default to
`finetuned_MobileNetV2.pth` (fixed 2026-07-16 — see footgun note above);
override with `--checkpoint` if you need the RAF-DB-only baseline.

## Models

| Name | Registry key | Notes |
|---|---|---|
| MobileNetV2 | `MobileNetV2` | **deployed**, smallest, fast |
| MobileNetV3-Large | `MobileNetV3-Large` | strong accuracy/size balance |
| EfficientNet-B0 | `EfficientNet-B0` | best RAF-DB accuracy in budget; also fine-tuned (`finetuned_EfficientNet_B0.pth`) as an alternative |
| MNASNet 1.0 | `MNASNet1_0` | mobile-optimized |
