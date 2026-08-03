# Gesture Model v2 — Design Spec & HPC Training Guide

**Branch:** `gesture-v2` · **Status:** design agreed, implementation pending
**Purpose of this document:** the full handoff for continuing this work on the HPC.
It records every design decision already made, the exact data/feature/training
spec, and the step-by-step workflow. If you continue with Claude Code on the
HPC, point it at this file first — it replaces the original conversation.

---

## 1. Background & decisions already made

- The old gesture modality (MediaPipe Hands → static-pose MLP + hand-written
  rules for wave/raise/beckon) was **removed** in commit `0074d2a`. Its two
  structural flaws: dynamic gestures were unlearned heuristics (no metrics
  possible), and hands-only input cannot see arm/body gestures.
- **Agreed approach:** keypoint-sequence classification. MediaPipe **pose +
  hands** landmarks per frame → normalized feature vector → sliding window of
  32 frames → small temporal network → 8 classes. Static and dynamic gestures
  are handled uniformly; no rules.
- **Agreed data strategy:** public datasets (Jester, NTU RGB+D 120, optionally
  HaGRIDv2) for most classes; **custom recorded clips (~100/class) for
  `beckoning` and `raise_hand`**, where public coverage is missing; a small
  custom **live test set for all classes** recorded with the project RealSense
  (never used in training).
- **Agreed scope:** full research pipeline mirroring `modalities/emotion/`:
  compare-models → train → tune → evaluate stages, each generating markdown
  reports, plus `realtime_realsense.py` / `video.py` inference and a
  `GestureEngine` API matching `MotionEngine` for the fusion layer.
- **Deployment target:** Jetson Orin Nano, running alongside emotion CNN,
  YOLO11n, scene CNN, and gaze — the gesture net must stay tiny (landmarks are
  the only expensive part, and MediaPipe is already running for motion/gaze).

## 2. Classes (8)

| # | Label | Nature | Definition |
|---|---|---|---|
| 0 | `idle` | — | No target gesture: standing/sitting, talking, scratching head, any other movement |
| 1 | `wave` | dynamic | Greeting wave — hand raised, side-to-side oscillation (any height, one hand) |
| 2 | `point` | static/short | Arm extended, index finger pointing (any direction) |
| 3 | `thumbs_up` | static | Fist with thumb extended upward |
| 4 | `thumbs_down` | static | Fist with thumb extended downward |
| 5 | `beckoning` | dynamic | "Come here" — repeated pull of hand/fingers toward the body |
| 6 | `raise_hand` | static/held | One arm raised above shoulder, hand open or fist (classroom style) |
| 7 | `both_hands_up` | static/held | Both arms raised above shoulders |

Assumptions (confirmed): `raise_hand` is one arm and distinct from
`both_hands_up`; single person in frame — multi-person selection is deferred to
the fusion layer.

`idle` is a real trained class, not a threshold fallback. Most frames in
deployment are idle; false positives are the main UX failure.

## 3. Feature specification

Landmarks come from MediaPipe Holistic (`mediapipe==0.10.35`, legacy
`mp.solutions.holistic`, `model_complexity=1`, video mode).

**Extraction stores RAW landmarks** (so features can be re-engineered without
re-running MediaPipe — extraction is the expensive step). One `.npz` per clip:

```
pose      float32 [T, 33, 4]   # x, y, z, visibility (image-normalized coords)
left_hand float32 [T, 21, 3]   # x, y, z  (NaN-filled frames where absent)
right_hand float32 [T, 21, 3]
meta: dataset, source_file, subject_id, label, fps
```

**Per-frame feature vector (dim 185), computed at load time:**

| Block | Dim | Normalization |
|---|---|---|
| Pose: 33 × (x, y, visibility) | 99 | Translate so mid-shoulder = origin; scale by shoulder width (dist L-shoulder ↔ R-shoulder). Mid-hip/hips may be off-frame at close range — never depend on them. |
| Left hand: 21 × (x, y) | 42 | Translate so wrist = origin; scale by wrist ↔ middle-MCP distance |
| Left-hand presence flag | 1 | 1.0 if detected this frame else 0.0 (features zeroed when absent) |
| Right hand: 21 × (x, y) | 42 | same |
| Right-hand presence flag | 1 | same |

Notes:
- Hand blocks are wrist-relative, so they encode *shape* only; hand *position*
  (raised? moving?) is carried by pose wrist landmarks 15/16. This is
  intentional — don't add global hand position to the hand blocks.
- The presence flags are the fix for the dataset framing mismatch: Jester
  clips are close-up (pose partly/fully missing → pose visibility ≈ 0), NTU is
  full-body at distance (hand detail poor → flags often 0). The model learns
  which block decides which class. This also matches deployment, where
  MediaPipe Hands drops out intermittently at 2–4 m.
- If pose is entirely absent in a frame (Jester close-ups), set the pose block
  to zeros and visibilities to 0.

**Window:** 32 frames. Training clips are uniformly resampled in time to 32
frames (plus speed augmentation, §6). Inference uses a rolling 32-frame buffer
at the camera rate with per-frame prediction.

## 4. Datasets & class mapping

Expected HPC layout (adjust `GESTURE_DATA_ROOT` in `config.py` if different):

```
<GESTURE_DATA_ROOT>/
├── jester/                    # 20BN-Jester: numbered clip folders of jpg frames
│   ├── 20bn-jester-v1/<clip_id>/*.jpg
│   └── annotations/ jester-v1-train.csv, jester-v1-validation.csv
├── ntu/                       # NTU RGB+D 120 RGB videos
│   └── nturgb+d_rgb/SsssCcccPpppRrrrAaaa_rgb.avi
├── custom/                    # our recorded clips
│   ├── train/<label>/<subject_id>__<clip>.mp4     # beckoning, raise_hand (+ any extras)
│   └── live_test/<label>/<subject_id>__<clip>.mp4 # ALL 8 classes, RealSense, NEVER trained on
└── hagrid/                    # optional, static images (see §4.4)
```

### 4.1 Jester → labels

| Jester class | → label | Note |
|---|---|---|
| Thumb Up | `thumbs_up` | ~5k clips |
| Thumb Down | `thumbs_down` | |
| Shaking Hand | `wave` | close-range hand wave |
| Pulling Hand In | `beckoning` | **proxy** — palm pull-in, not a true finger beckon; blend with custom data |
| No gesture, Doing other things | `idle` | purpose-built negatives |
| All other 21 classes (swipes, zooms, …) | `idle` | subsample to ≤ 300 clips/class — hard negatives against false positives |

### 4.2 NTU RGB+D 120 → labels

Filename code `SsssCcccPpppRrrrAaaa`: `P` = subject (use for subject-wise
splits!), `A` = action.

| NTU action | → label | Note |
|---|---|---|
| A23 hand waving | `wave` | full-body arm wave — complements Jester's close-up |
| A31 pointing to something with finger | `point` | |
| A69 thumb up | `thumbs_up` | |
| A70 thumb down | `thumbs_down` | |
| A95 hands up (both hands) | `both_hands_up` | |
| A22 cheer up | `both_hands_up` | visually verify a sample first; drop if it's fists-pumping rather than arms-up |
| A38 salute | `idle` (negative) | deliberately NOT `raise_hand` — hand-to-brow, not arm-up; useful confuser |
| A1 drink water, A8 sit down, A10 clapping, A34 rub hands, A37 wipe face | `idle` | daily-action negatives, subsample ≤ 300 each |

`raise_hand` has **no NTU source** — it comes entirely from custom clips.

### 4.3 Custom clips

- `beckoning`, `raise_hand`: ~100 clips each. Variety beats volume: several
  people, 1–4 m, both hands, sitting + standing. **Subject-wise split** — a
  person appearing in training must not appear in test for these classes.
- Live test set: 20–30 clips × all 8 classes on the RealSense in the actual
  room. This is the headline cross-domain evaluation and never enters training.

### 4.4 HaGRIDv2 (optional, phase 2)

Static images (`like`/`dislike`/`point`/`one`) can be turned into pseudo-static
sequences (repeat frame + jitter) to boost the static classes. Skip for v2.0
unless thumbs/point underperform — Jester+NTU should suffice.

## 5. Module layout (to implement — mirrors `emotion/` + `motion/`)

```
modalities/gesture/
├── config.py                  # labels, paths (GESTURE_DATA_ROOT env override), window=32, feature dims
├── README.md
├── src/
│   ├── features.py            # raw landmarks -> 185-dim frames; normalization + presence flags + mirror aug
│   ├── data.py                # npz dataset, class mapping tables (§4), subject-wise splits, samplers
│   ├── models.py              # BiGRU / TCN / TransformerEncoder (see §7)
│   └── engine.py              # GestureEngine: rolling buffer, EMA, debounce (see §8)
├── scripts/
│   ├── extract_landmarks.py   # MediaPipe over jester/ntu/custom -> data/landmarks/*.npz  (--dataset, --shard i/n)
│   ├── prepare_data.py        # apply mapping, build train/val/test index CSVs + class stats report
│   ├── compare_models.py      # Stage 1: 3 architectures -> reports/comparison/
│   ├── train.py               # Stage 2: winner, full recipe -> checkpoints/ + reports/training/
│   ├── tune.py                # Stage 3: optuna -> reports/tuning/
│   └── evaluate.py            # Stage 4: test + live_test + cross-dataset -> reports/evaluation/
└── inference/
    ├── realtime_realsense.py  # RealSense/webcam live overlay
    └── video.py               # video files -> annotated mp4 (repo videos/ conventions)
```

Fusion-facing API, same shape as `MotionEngine`:
`GestureEngine.process(pose_landmarks, left_hand, right_hand) -> (label, confidence)`.

## 6. Splits & augmentation

**Splits** (enforced in `prepare_data.py`, stamped into the index CSVs):
- Jester: keep official train/validation split.
- NTU: split by **subject ID** (e.g., 20 % of subjects held out for test).
- Custom train pool: split by subject **per label** (recorded change,
  2026-07-11): with ≥5 subjects a label splits 70/15/15 subject-wise; with
  2–4 subjects the last (sorted) subject is held out entirely and its clips
  alternate val/test (eval stays subject-pure w.r.t. training); with 1
  subject it falls back to a clip-level 70/15/15 split with a warning.
  Custom filenames without `__` take the first `_` token as subject
  (`S01_F04_c001.mp4` → `S01`).
- `custom/live_test/`: never trained or tuned on; reported separately.

**Augmentation** (feature-space, on-the-fly in `data.py`):
- Horizontal mirror: negate x, swap L/R shoulders/elbows/wrists etc., swap hand
  blocks + flags (doubles every class; critical for left/right-hand symmetry).
- Temporal: speed 0.8–1.2× (resample), random start-crop before resampling to 32.
- Spatial: Gaussian landmark jitter (σ ≈ 0.01 post-normalization), small global
  rotation (±10°), small scale (±10 %).
- Random hand dropout: zero a hand block + flag for a contiguous span —
  simulates MediaPipe dropouts at distance and teaches mask reliance.

## 7. Models & training recipe

Stage-1 candidates (all < 1 M params, input `[32, 185]`):

| Model | Spec |
|---|---|
| **BiGRU** | 2 × 128 bidirectional, mean+max pool over time, FC head |
| **TCN** | 4 residual dilated-conv blocks (k=5, d=1/2/4/8, ch=128), global pool |
| **Tiny Transformer** | linear proj → d=128, 2 encoder layers, 4 heads, learned pos-emb, CLS token |

Starting recipe (Stage 2/3 refine): AdamW lr 1e-3, cosine schedule, wd 1e-4,
batch 256, ≤ 100 epochs early-stopped on **val macro-F1** (class counts are
very imbalanced — Jester thumbs ≫ custom beckoning — so macro-F1 selection +
**sqrt** inverse-frequency class weights (recorded change, 2026-07-12: raw
inverse frequency at ~280:1 idle-vs-raise_hand made models abandon idle and
plateau at ~60% macro-F1; sqrt softening to ~17:1 fixed it. Classes with no
training data get weight 0, or label smoothing leaks their weight into every
sample's loss), label smoothing 0.1, dropout 0.3. Report accuracy, balanced
accuracy, macro-F1, per-class P/R, confusion matrix.

Watch these confusion pairs specifically: `beckoning`↔`wave`,
`raise_hand`↔`both_hands_up`, `point`↔`thumbs_*`, everything↔`idle`.

## 8. Inference engine spec

- Rolling 32-frame feature buffer; predict every frame once full.
- EMA over the softmax vector, α = 0.25 (as in motion).
- Debounce: emit a label only if it wins for ≥ 300 ms AND smoothed confidence
  ≥ 0.6 (tune on live test set); otherwise emit `idle`.
- Buffer resets when the person leaves frame.

## 9. HPC workflow

### 9.1 One-time setup

```bash
git clone https://github.com/KesharaGunathilaka/adaptive-multimodal-hri.git
cd adaptive-multimodal-hri
git checkout gesture-v2

python -m venv .venv && source .venv/bin/activate
# edit requirements.txt: uncomment the "FOR HPC ONLY" cu128 torch pins,
# comment out the plain torch/torchvision lines
pip install -r requirements.txt

# point the pipeline at the datasets (or edit modalities/gesture/config.py)
export GESTURE_DATA_ROOT=/path/to/datasets
```

Arrange datasets as in §4. Sanity-check MediaPipe on a login/compute node
(`python -c "import mediapipe"`) — it is CPU-only here, which is fine:
extraction is CPU-bound by design.

### 9.2 Stage 0 — landmark extraction (the heavy step; CPU array job)

`extract_landmarks.py` supports `--shard i/N` so it parallelizes trivially.
SLURM template (**adapt partition/account/module names to your cluster**):

```bash
#!/bin/bash
#SBATCH --job-name=gesture-extract
#SBATCH --array=0-15
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
source .venv/bin/activate
python modalities/gesture/scripts/extract_landmarks.py \
    --dataset all --shard ${SLURM_ARRAY_TASK_ID}/16
```

Ballpark: Jester subset (~15–20k clips of ~37 frames) + NTU subset (~8k clips,
trimmed) ≈ a few hours across 16 × 8 cores. Output: `data/landmarks/*.npz`
(small — tens of MB total). Re-running skips existing files.

### 9.3 Stages 1–4 (GPU job, light)

```bash
python modalities/gesture/scripts/prepare_data.py        # index CSVs + class-balance report
python modalities/gesture/scripts/compare_models.py      # Stage 1 -> reports/comparison/
python modalities/gesture/scripts/train.py               # Stage 2 -> checkpoints/ + reports/training/
python modalities/gesture/scripts/tune.py                # Stage 3 (optuna) -> reports/tuning/
python modalities/gesture/scripts/train.py --use-tuned   # retrain with best params
python modalities/gesture/scripts/evaluate.py            # Stage 4 -> reports/evaluation/
```

GPU SLURM template:

```bash
#!/bin/bash
#SBATCH --job-name=gesture-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
source .venv/bin/activate
python modalities/gesture/scripts/compare_models.py && \
python modalities/gesture/scripts/train.py
```

Training is minutes-to-an-hour per run (tiny model, sequences of 32 × 185
floats) — the GPU job is short; tuning (~50 optuna trials) is the longest.

### 9.4 Bringing results back

- Commit `reports/**` and the small index CSVs to `gesture-v2` and push.
- Publish `checkpoints/best_*.pth` (+ a `model_config.json` with labels, dims,
  window, normalization constants — like `motion/checkpoints/model_config_v2.json`)
  as a GitHub Release `gesture-v2.0`, same convention as `emotion-v1.0`.
  Raw `.npz` landmarks and datasets stay on the HPC (git-ignored).

## 10. Evaluation & acceptance

`evaluate.py` reports three numbers, in increasing order of honesty:
1. **Held-out test** (subject-wise, mixed datasets) — the standard metric.
2. **Cross-dataset checks** (e.g., wave trained w/o NTU, tested on NTU) — the
   generalization chapter for the research write-up.
3. **Live test set** (`custom/live_test/`, RealSense in the real room) — the
   deployment truth. Target: macro-F1 ≥ 0.80 live; watch `beckoning` (trained
   partly on the Jester proxy) and `raise_hand` (custom-only) — if either
   falls below ~0.7 live, the fix is recording more custom clips for that
   class (the pipeline absorbs new data with zero architecture change).

## 11. Known risks (agreed & accepted)

1. **Beckoning** rests on Jester "Pulling Hand In" (a proxy) + ~100 custom
   clips → weakest class, verify live early.
2. **Raise-hand** is custom-only (~100 clips × mirror/temporal augmentation);
   NTU salute is deliberately a *negative*.
3. **Framing mismatch** between Jester (close-up) and NTU/deployment
   (full-body) is handled by presence flags + hand-dropout augmentation (§3,
   §6) — do not remove those thinking they're optional.
4. Landmark quality bounds everything: motion blur / occlusion at > 4 m
   degrades MediaPipe before it degrades our model.

## 12. Continuing with Claude Code on the HPC

Suggested opening prompt:

> Read `docs/GESTURE_V2_DESIGN_AND_HPC_GUIDE.md` fully — it contains all design
> decisions for the new gesture modality. Also skim `modalities/emotion/`
> (pipeline/report conventions) and `modalities/motion/src/engine.py` (engine
> API to match). Then implement/continue the `modalities/gesture/` module
> exactly per the guide. Datasets are at `$GESTURE_DATA_ROOT` laid out per §4.
> Do not change the feature spec (§3), class mapping (§4), or split rules (§6)
> without recording the change in the guide.

Keep this document updated as decisions change — it is the source of truth.
