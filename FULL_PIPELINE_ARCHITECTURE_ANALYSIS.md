# Full Pipeline Architecture Analysis

> Forensic reconstruction of the HRI Fusion Engine from the actual
> implementation. Every claim below was traced through source code and,
> where possible, verified against the real data artifacts
> (`pipeline/measured/*.jsonl`, `data/features/clip_features.parquet`,
> `Motion Repo/checkpoints/best_model_finetuned.pt`) and by executing
> `fusion/rule_based.py` and `fusion/gbt.py`.
>
> Convention used throughout:
> **[VERIFIED]** = read directly from code or confirmed by running it /
> inspecting the artifact. **[INTERPRETATION]** = reasonable inference from
> verified facts. **[CONCEPTUAL]** = generic ML explanation, not this repo's
> literal object. **NOT VERIFIABLE** = could not be confirmed from the
> present codebase.

---

## 0. The single most important structural fact

**This system is NOT an online, frame-streaming fusion engine. It is a
batch, file-staged pipeline with four completely independent processes and a
hard serialization boundary (JSONL on disk) between the cue models and
fusion.** [VERIFIED]

The four cue models never call each other and never call fusion. They run in
four separate Python virtual environments (`.venvs/emotion`, `.venvs/gesture`,
`.venvs/motion`, `.venvs/context`), each producing one append-only JSONL file
of per-frame records. A separate pipeline stage collapses those per-frame
records into **one 33-dimensional feature vector per clip** and writes a
Parquet file. Only then do the two fusion approaches read that Parquet.

```
video clips (.mp4)
   │
   ├── runners/emotion_runner.py   (.venvs/emotion)  ─► pipeline/measured/emotion_frame_cues.jsonl
   ├── runners/gesture_runner.py   (.venvs/gesture)  ─► pipeline/measured/gesture_frame_cues.jsonl
   ├── runners/motion_runner.py    (.venvs/motion)   ─► pipeline/measured/motion_frame_cues.jsonl
   └── runners/context_runner.py   (.venvs/context)  ─► pipeline/measured/context_frame_cues.jsonl
                                                              │ (4 files, one JSON object per frame per clip)
                                                              ▼
                              pipeline/build_features.py  +  pipeline/aggregate.py
                                                              │ (per-clip aggregation → 33 features)
                                                              ▼
                                          data/features/clip_features.parquet   (1270 rows × 40 cols)
                                                              │
                              ┌───────────────────────────────┴───────────────────────────────┐
                              ▼                                                                 ▼
                     fusion/rule_based.py                                              fusion/gbt.py
                     (priority IF-THEN over                                            (LightGBM over the
                      argmax'd cue blocks)                                              33-col feature vector)
                              │                                                                 │
                              ▼                                                                 ▼
                      predicted intent code                                            predicted intent code
                      (F01..F10) per clip                                              (F01..F10) per clip
```

There is **no `main.py`, no `FusionEngine` class, no `run_cue_models.py`**
(the last is referenced in `runners/emotion_runner.py`'s and
`runners/motion_runner.py`'s docstrings but does **not exist** in the repo —
see §17). The pipeline is run by executing individual scripts by hand in the
documented order. Fusion consumes a **precomputed file**, not live cue
objects.

---

## 1. Executive Pipeline Summary

**Raw input.** Video clips (`.mp4`, e.g. `640×480`, 15 or 30 fps, ~4–5 s,
68–150 frames each), enumerated by
`Data/Dataset/hri-multimodal-intent-v1.0.0/annotations/clips.csv`. 1270 clips
across 22 base scenarios (S01…S29 with gaps), each scenario authored to map to
one of 10 intent codes F01…F10 (`annotations/scenarios.csv`). [VERIFIED]

**Where processing begins.** Each of the four `runners/*_runner.py` scripts is
an independent entry point. In batch mode a runner loads its model **once**,
then loops every clip in `clips.csv`, decoding frames with OpenCV and running
its model per frame. [VERIFIED, `run_batch()` in every runner]

**How the four cue models are invoked.** Independently and offline — one
process per cue, one venv per cue, no cross-talk. They are *not* orchestrated
by a shared driver in the committed code. Each writes a combined JSONL
(`append_batch()` adds a `clip_id` envelope to each `NormalisedFrameCue`).
[VERIFIED]

**What each model returns (per frame).** All four emit the same dataclass,
`runners/common/schema.py::NormalisedFrameCue`:
`{cue, frame_idx, label, confidence, probs, valid, extra}`. The *contents*
differ sharply per cue (emotion/motion/context populate `probs`; gesture
leaves `probs` empty; see §8). [VERIFIED]

**Where outputs are collected.** On disk, as four JSONL files under
`pipeline/measured/`. Each currently holds **141,721 lines** (= total frames
across all clips), and all four line counts are identical. [VERIFIED by
`wc -l`]

**Temporal aggregation.** Yes — this is the crux. `pipeline/aggregate.py`
collapses each clip's per-frame records for each cue into a fixed block:
- **Emotion, Motion, Context** → **mean of per-frame probability vectors over
  valid frames** + a confidence scalar + `valid_fraction`.
- **Gesture** → **majority-vote one-hot** over valid frames + mean confidence
  of the winning label + `valid_fraction`.
A cue whose `valid_fraction < 0.40` gets a `missing_<cue> = 1.0` bit. [VERIFIED]

**How cue outputs become fusion input.** `pipeline/build_features.py` joins
each clip's four aggregated blocks into one row, appends the target `intent`
and the train/val/test split assignment, and writes
`data/features/clip_features.parquet` (1270 × 40: `clip_id` + **33 features** +
6 metadata columns). This Parquet **is the fusion input for both approaches**.
[VERIFIED]

**How rule-based fusion works.** `fusion/rule_based.py` reads each clip's
feature row, reconstructs a single dominant label per cue by `argmax` over
that cue's block (`_dominant()`), then runs a fixed, **priority-ordered**
IF-THEN cascade (emergency F02 first, then most-to-least specific patterns,
then a train-set-mode fallback). First matching rule returns immediately.
[VERIFIED]

**How GBT fusion works.** `fusion/gbt.py` feeds the **raw 33-dim feature
vector** (no argmax, no re-encoding) into a `lightgbm.LGBMClassifier`
(multiclass, 300 trees, depth 5, lr 0.05, `class_weight="balanced"`). Training
adds per-cue "modality dropout" augmentation; inference adds an F02 safety
override (if `P(F02) ≥ 0.15`, force F02). [VERIFIED]

**Final system output.** A predicted intent **string** (`"F01"`…`"F10"`) per
clip. Neither fusion script serializes a model or writes a prediction file —
both **print accuracy metrics to stdout** and exit. There is no deployed
inference endpoint in the committed code. [VERIFIED]

Measured accuracies (executed 2026-07, `split_scenario` grouped split):

| Model | train | val | test |
|---|---|---|---|
| rule_based | 0.335 | 0.255 | **0.179** |
| GBT | (n/a printed) | 0.216 | **0.237** |

Both are low; §17 explains why (the grouped test split contains only 4 of the
10 intent classes, and the label ceiling documented in
`fusion/rule_based.py`). [VERIFIED by running the scripts]

---

## 2. Verified Runtime Entry Points

There are **six** real entry points, in three tiers. None of them is a unified
"process a video end-to-end" driver — that does not exist.

### Tier 1 — Cue runners (raw video → per-frame JSONL)

| # | File | Entry | Input | Output | Why it's a real entry point |
|---|---|---|---|---|---|
| 1 | `runners/emotion_runner.py` | `__main__` → `run_batch()`/`run_single()` | `--manifest clips.csv --clips-root … --out …jsonl` (or `--clip`) | `emotion_frame_cues.jsonl` | Has `argparse` `__main__`; loads MobileNetV2, loops frames, writes JSONL. [VERIFIED] |
| 2 | `runners/gesture_runner.py` | same shape | same | `gesture_frame_cues.jsonl` | Same; loads two TFLite classifiers. [VERIFIED] |
| 3 | `runners/motion_runner.py` | same shape | same | `motion_frame_cues.jsonl` | Same; loads `MotionInference` LSTM. [VERIFIED] |
| 4 | `runners/context_runner.py` | same shape | same | `context_frame_cues.jsonl` | Same; loads EfficientNet-B0. [VERIFIED] |

Each runner is called by a human/shell (documented in its own docstring
Usage). Return value: none (side effect = JSONL file). These are real because
they contain `if __name__ == "__main__":` + `argparse` and do the actual model
loading and frame loop.

### Tier 2 — Feature builder (JSONL → Parquet)

| # | File | Entry | Input | Output |
|---|---|---|---|---|
| 5 | `pipeline/build_features.py` | `__main__` → `main()` | 4× `*_frame_cues.jsonl` + `clips.csv`, `scenarios.csv`, `splits.csv` | `data/features/clip_features.parquet` |

Real because it imports `aggregate.build_clip_feature_row` /
`load_frame_cues_by_clip`, iterates every clip, and writes the Parquet that
both fusion scripts open. [VERIFIED]

Prerequisite entry point (run once before build_features to create splits.csv):
`pipeline/build_splits.py::main()`.

### Tier 3 — Fusion (Parquet → intent predictions + metrics)

| # | File | Entry | Input | Output |
|---|---|---|---|---|
| 6a | `fusion/rule_based.py` | `__main__` → `predict_all()` | `clip_features.parquet` | stdout accuracy; `df["rule_pred"]` in-memory |
| 6b | `fusion/gbt.py` | `__main__` → `main()` | `clip_features.parquet` | stdout accuracy; trained model in-memory (never saved) |

Both are real terminal entry points. Note `fusion/gbt.py` **imports**
`fusion/rule_based.py` (`from rule_based import predict_all, fit_fallback`) and
runs the rule baseline alongside GBT for direct comparison — so running
`gbt.py` exercises *both* fusion paths. [VERIFIED]

### Not entry points (diagnostic side-branch)

`pipeline/aggregate_clip_cues.py` → `pipeline/agreement_report.py` form a
**Phase-0 diagnostic** path (majority-vote dominant label per clip, compared
against the authored `scenarios.csv` intent). Their output
(`pipeline/measured/clip_cues.csv`, `reports/phase0_agreement.*`) **does not
feed fusion**. This is a parallel QA branch, easy to mistake for the feature
pipeline (it is *not* — see §9 and §17). [VERIFIED]

---

## 3. Complete End-to-End Call Graph

```text
# ── STAGE 1: four independent runner processes (one per venv) ──────────────

runners/emotion_runner.py  __main__
└── run_batch(manifest, clips_root, out)
    ├── read_manifest(clips.csv)                       # common/schema.py
    ├── load_model()                                   # → build_model()+load_state_dict (Emotion Repo/video.py)
    │   └── emotion_video.build_model()                # torchvision MobileNetV2, head→Linear(1280,7)
    └── for each clip:  process_clip(clip, model, transform, device, mp_face)
        ├── cv2.VideoCapture(clip).read()  (per frame)
        ├── mp_face.process(rgb_small)                 # MediaPipe FaceDetection(model_selection=1)
        ├── pick_face(detections)                      # largest bbox by area
        ├── transform(pil).unsqueeze(0)                # → (1,3,224,224)
        ├── F.softmax(model(tensor))[0]                # → (7,) probs
        └── NormalisedFrameCue(label, confidence, probs, valid=conf>=0.50, extra={bbox})
    → append_batch(f, clip_id, records)                # writes JSONL lines

runners/gesture_runner.py  __main__
└── run_batch → process_clip(clip, keypoint_classifier, point_history_classifier, ...)
    ├── hands.process(rgb)                             # MediaPipe Hands(max_num_hands=2)
    ├── calc_landmark_list / pre_process_landmark      # 21 landmarks → 42-vec normalised
    ├── keypoint_classifier(vec)  → (sign_id, conf)    # TFLite KeyPointClassifier (6 signs)
    ├── point_history_classifier(hist) → (act_id,conf) # TFLite PointHistoryClassifier (6 actions)
    ├── detect_wave / detect_come_here / check_hand_raised   # heuristic geometry
    ├── <Global Scenario Resolution>                   # 1-hand / 2-hand IF-THEN → scenario text
    └── GESTURE_SCENARIO_TO_CANONICAL[text] → label    # constants.py; probs={} always
    → NormalisedFrameCue(label, confidence, probs={}, valid=(label!=Unknown and conf>=0.80))

runners/motion_runner.py  __main__
└── run_batch → process_clip(clip, engine)             # engine loaded once, engine.reset() per clip
    ├── mp_pose.Pose.process(rgb)  → pose_world_landmarks   # MediaPipe Pose (33 world lm)
    ├── mediapipe_to_ntu25(world_landmarks.landmark)   # skeleton_utils.py → (25,3)
    └── engine.update(joints_25)                       # inference.py MotionInference
        ├── joints_25[JOINT_SUBSET] → (14,3)
        ├── normalize_skeleton → flatten → pos(42); vel = pos - prev_pos(42)
        ├── frame_feat = concat(pos,vel) (84,); push to 30-frame deque
        └── if len(deque)==30: _predict() → MotionLSTM.forward → softmax → MotionResult(label,probs(4,))
    → NormalisedFrameCue(label, confidence, probs{4 classes}, valid=…, extra={buffering,has_landmarks})

runners/context_runner.py  __main__
└── run_batch → process_clip(clip, model, transform, device)
    ├── transform(rgb).unsqueeze(0)                    # → (1,3,224,224)
    ├── torch.softmax(model(tensor))[0]                # EfficientNet-B0 → (2,) probs
    ├── prob_history(deque maxlen=15); avg = mean(history)   # temporal smoothing
    └── NormalisedFrameCue(label, confidence=avg_conf, probs{classroom,kitchen}, valid=…)

# ── STAGE 2: feature builder (single process, .venvs/pipeline) ─────────────

pipeline/build_features.py  main()
├── read_csv(clips.csv / scenarios.csv / splits.csv)
├── load_frame_cues_by_clip(cue) for cue in [emotion,gesture,motion,context]   # aggregate.py
│   └── {clip_id: [frame_record,…]}                    # groups JSONL by clip_id
└── for each clip:  build_clip_feature_row(clip_id, e_recs, g_recs, m_recs, c_recs)   # aggregate.py
    ├── _prob_mean_features(emotion_records, EMOTION_CLASSES)   # mean probs + max_conf + valid_frac
    ├── _majority_onehot_features(gesture_records, GESTURE_CLASSES)  # one-hot + mean_conf + valid_frac
    ├── _prob_mean_features(motion_records, MOTION_CLASSES)
    ├── _prob_mean_features(context_records, CONTEXT_CLASSES)   # + separate mean-confidence calc
    └── missing_<cue> = float(valid_fraction < 0.40)
    → row dict (33 features) + intent + split cols
→ df.to_parquet(clip_features.parquet)

# ── STAGE 3: fusion (single process, .venvs/pipeline) ──────────────────────

fusion/rule_based.py  __main__
├── df = read_parquet(clip_features.parquet)
├── fallback = fit_fallback(train_df)                  # train intent mode = "F04"
└── predict_all(df, fallback) → df.apply(predict_intent, axis=1)
    └── predict_intent(row):
        ├── emotion = _dominant(row,"emotion",EMOTION_CLASSES)   # argmax of block, None if missing
        ├── gesture = _dominant(row,"gesture",GESTURE_CLASSES)
        ├── motion  = _dominant(row,"motion",MOTION_CLASSES)
        ├── context = _dominant(row,"context",CONTEXT_CLASSES)
        └── priority IF-THEN cascade → intent string

fusion/gbt.py  main()
├── df = read_parquet(clip_features.parquet)
├── X_train = apply_modality_dropout(train_df[FEATURE_NAMES], rng)   # per-cue zeroing + missing bit
├── model = LGBMClassifier(multiclass, n_estimators=300, max_depth=5, lr=0.05, class_weight="balanced")
├── model.fit(X_train, train_df["intent"])
├── (also) predict_all(df, fallback)  ← imports rule_based, runs it for comparison
└── predict_with_safety_override(model, split_df[FEATURE_NAMES], f02_idx)
    ├── proba = model.predict_proba(X)                 # (n, 10)
    ├── preds = model.classes_[proba.argmax(1)]
    └── preds = where(proba[:,f02_idx] >= 0.15, "F02", preds)   # safety override
```

---

## 4. Emotion Cue — Full Data Trace

Files: `runners/emotion_runner.py` (loop), `Emotion Repo/video.py` (model,
preprocess, labels). Model weights: `Emotion Repo/best_MobileNetV2.pth`.

### 4.1 Input

- **Source:** one decoded video frame from `cv2.VideoCapture(clip).read()`.
- **Type / shape / dtype:** `numpy.ndarray`, `(H, W, 3)`, `uint8`, **BGR**
  channel order (OpenCV default). For this dataset typically `(480, 640, 3)`.
- **Value range:** 0–255. Example: a `(480,640,3)` uint8 array. [VERIFIED,
  `emotion_runner.py:81-85`]

### 4.2 Preprocessing

| Step | Function | Input | Operation | Output type | Output shape |
|---|---|---|---|---|---|
| 1 | `emotion_runner.process_clip` | `(480,640,3)` uint8 BGR | downscale if `w>640` (`MAX_FRAME_WIDTH`) | ndarray uint8 BGR | ≤`(…,640,3)` |
| 2 | `cv2.cvtColor(...BGR2RGB)` | small BGR | BGR→RGB for detector | ndarray uint8 RGB | same |
| 3 | `mp_face.process` | RGB frame | MediaPipe FaceDetection (`model_selection=1`, full-range) | detections list | — |
| 4 | `pick_face` | detections | choose largest bbox by `w*h` | one detection | — |
| 5 | bbox → pixel crop | full-res **BGR** frame | `face = frame[y:y+bh, x:x+bw]` (crop on original, not the downscaled copy) | ndarray uint8 BGR | `(bh,bw,3)` |
| 6 | `cv2.cvtColor(...BGR2RGB)` + `Image.fromarray` | face BGR | → PIL RGB image | `PIL.Image` | `(bw,bh)` |
| 7 | `transform(pil)` = `Resize((224,224))`+`ToTensor`+`Normalize(ImageNet)` | PIL RGB | resize, /255, standardize | `torch.float32` | `(3,224,224)` |
| 8 | `.unsqueeze(0).to(device)` | tensor | add batch dim | `torch.float32` | `(1,3,224,224)` |

Normalize constants: mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`.
[VERIFIED, `video.py:45-50`]

### 4.3 Model Architecture

`build_model()` (`video.py:38-42`): **torchvision `mobilenet_v2(weights=None)`**
with `model.classifier[1] = nn.Linear(1280, 7)` (`last_channel=1280`, 7 emotion
classes). Weights loaded from `best_MobileNetV2.pth` via
`load_state_dict(..., weights_only=True)`. [VERIFIED] Data flow: `(1,3,224,224)`
→ MobileNetV2 feature extractor → global pool → `(1,1280)` → Linear → `(1,7)`
logits. The `.pth` is a raw `state_dict` (no config); architecture is fully
specified by `build_model()`, so it **is** verifiable (not an opaque blob).

### 4.4 Prediction Generation

`emotion_runner.py:112-117`:
```python
probs_vec = F.softmax(model(tensor), dim=1)[0].cpu().numpy()   # (7,) float
idx  = int(probs_vec.argmax())
conf = float(probs_vec[idx])
label = emotion_video.EMOTION_LABELS[idx]
probs = {lbl: float(p) for lbl,p in zip(EMOTION_LABELS, probs_vec)}
```
- `EMOTION_LABELS = ["Surprise","Fear","Disgust","Happy","Sad","Anger","Neutral"]`
  (index 0→6). [VERIFIED, `video.py:30`]
- Softmax over logits → `argmax` → class index → `conf` = that probability.
- **valid gate:** `valid = (conf >= 0.50)` (`CONFIDENCE_FLOOR["emotion"]`).
- If **no face** detected or crop is empty → emits
  `label="Unknown", confidence=0.0, probs={}, valid=False`. [VERIFIED]

### 4.5 Exact Returned Output (per frame)

`NormalisedFrameCue` serialized as one JSON line. Real record from
`emotion_frame_cues.jsonl` [VERIFIED]:
```json
{"cue":"emotion","frame_idx":0,"label":"Neutral","confidence":0.99996,
 "probs":{"Surprise":4.5e-06,"Fear":4.6e-10,"Disgust":3.9e-07,"Happy":3.3e-05,
          "Sad":2.8e-07,"Anger":3.0e-10,"Neutral":0.99996},
 "valid":true,"extra":{"bbox":[256,140,56,56]},"clip_id":"S01_F04_c001"}
```
Fields: `probs` has all 7 class keys; `confidence` == `probs[label]`;
`extra.bbox` = `[x,y,bw,bh]` pixel box (or `null`).

### 4.6 Downstream Data Trace

`emotion_frame_cues.jsonl` → `aggregate.load_frame_cues_by_clip("emotion")`
→ list per `clip_id` → `_prob_mean_features(records, EMOTION_CLASSES)`:
- `valid = [r for r in records if r["valid"]]`
- `mat = [[r["probs"].get(c,0.0) for c in EMOTION_CLASSES] for r in valid]`
  → `mat.mean(axis=0)` = **7 mean probabilities**
- `max_confidence = max(r["confidence"] for r in valid)`
- `valid_fraction = len(valid)/len(records)`
→ becomes columns `emotion_Surprise…emotion_Neutral`,
`emotion_max_confidence`, `emotion_valid_fraction`, plus
`missing_emotion = float(valid_fraction < 0.40)`. Then straight into the
Parquet, then into both fusion approaches. **No emotion value is dropped;
every one reaches fusion.** [VERIFIED]

---

## 5. Motion Cue — Full Data Trace

Files: `runners/motion_runner.py`, `Motion Repo/inference.py`,
`Motion Repo/model.py`, `Motion Repo/skeleton_utils.py`. Weights:
`Motion Repo/checkpoints/best_model_finetuned.pt`.

### 5.1 Input

- **Source:** decoded frame → MediaPipe Pose. The model does **not** consume
  pixels; it consumes **3D world landmarks**.
- After `mp_pose.Pose.process(rgb)`: `results.pose_world_landmarks.landmark` =
  **33 metric 3D landmarks** (metres), or `None` if no person.
- `mediapipe_to_ntu25()` converts to `numpy (25,3) float32` in NTU layout
  (X,Y negated = 180° about Z; 3 spine joints approximated). If no landmarks,
  `joints_25 = np.zeros((25,3), float32)`. [VERIFIED, `motion_runner.py:140-152`]

### 5.2 Preprocessing (inside `MotionInference.update`, `inference.py:145-184`)

| Step | Function | Input | Operation | Output type | Output shape |
|---|---|---|---|---|---|
| 1 | frame rotate/resize | `(H,W,3)` uint8 | orientation fix, `resize_with_aspect_ratio(max 960)` | uint8 | ≤`(960,…)` |
| 2 | `pose.process` | RGB | MediaPipe Pose world landmarks | 33 lm or None | — |
| 3 | `mediapipe_to_ntu25` | 33 lm | remap+negate → NTU | `(25,3)` float32 | `(25,3)` |
| 4 | `joints_25[JOINT_SUBSET]` | `(25,3)` | select 14 joints | `(14,3)` | `(14,3)` |
| 5 | `normalize_skeleton` | `(14,3)` | hip-center, /shoulder-width (if >0.05) | `(14,3)` float32 | `(14,3)` |
| 6 | `.flatten()` | `(14,3)` | → position vector | `(42,)` | `(42,)` |
| 7 | `vel = pos - prev_pos` | `(42,)` | per-frame velocity (0 on first frame) | `(42,)` | `(42,)` |
| 8 | `concatenate([pos,vel])` | two `(42,)` | frame feature | `(84,)` | `(84,)` |
| 9 | `deque(maxlen=30).append` | `(84,)` | sliding window | deque | up to 30×84 |
| 10 | `np.stack(...).unsqueeze(0)` | 30×`(84,)` | window tensor | `torch.float32` | `(1,30,84)` |

`FEATURE_DIM=84`, `WINDOW_SIZE=30`, `NUM_CLASSES=4`. [VERIFIED]

### 5.3 Model Architecture

`model.py::MotionLSTM` [VERIFIED, and checkpoint config confirmed]:
`LayerNorm(84)` → `LSTM(input=84, hidden=256, num_layers=3, batch_first,
dropout=0.35)` → temporal attention `Linear(256,1)`+softmax over 30 timesteps
→ weighted-sum context `(1,256)` → classifier `Linear(256,64)→ReLU→Dropout→
Linear(64,4)` → `(1,4)` logits.

Checkpoint `best_model_finetuned.pt` is a **dict**
`{epoch, model_state_dict, val_acc, config, warm_started_from}`. Verified:
`config = {hidden_size:256, num_layers:3, dropout:0.35, …}`, `epoch=20`,
`val_acc≈0.5416`, final classifier weight shape **`(4,64)`** → genuinely
4 classes. [VERIFIED by loading the checkpoint]

> **Note — stale "6-class" comments.** `inference.py` docstrings/comments say
> `probs shape (6,)` and `logits (1,6)`; `model.py::forward` docstring says
> `returns (B,6)`. These are **wrong/stale**; the real head is 4-class and the
> real `probs` vector is length 4. Behaviour is correct; the comments lie.
> See §17.

### 5.4 Prediction Generation

`inference.py:194-213`: `logits = model(window)` → `F.softmax(dim=-1)` →
`probs_np (4,)` → `argmax` → `label_idx` → `MOTION_LABELS[idx]`
(`{0:"sitting",1:"standing",2:"walking",3:"stepping_back"}`). While the window
is filling (frames 0–28), `update()` returns `MotionResult(label="buffering",
confidence=0.0, probs=zeros(4))`. In the runner, buffering →
`label="Unknown", probs={}`. `valid = (not buffering) and (conf>=0.50) and
has_landmarks`. [VERIFIED, `motion_runner.py:154-168`]

### 5.5 Exact Returned Output (per frame)

Real record (buffering frame 0) [VERIFIED]:
```json
{"cue":"motion","frame_idx":0,"label":"Unknown","confidence":0.0,"probs":{},
 "valid":false,"extra":{"buffering":true,"has_landmarks":true},"clip_id":"S01_F04_c001"}
```
A settled frame instead carries `label∈{sitting,standing,walking,stepping_back}`,
`confidence∈[0,1]`, `probs={"sitting":…,"standing":…,"walking":…,"stepping_back":…}`.

### 5.6 Downstream Data Trace

Identical mechanism to emotion: `_prob_mean_features(motion_records,
MOTION_CLASSES)` → `motion_sitting…motion_stepping_back` (4 mean probs),
`motion_max_confidence`, `motion_valid_fraction`, `missing_motion`. Reaches
both fusion approaches. The first 29 buffering frames of every clip are
`valid=False`, so they are excluded from the mean (but counted in the
denominator of `valid_fraction`). [VERIFIED] In the built Parquet, only **6**
of 1270 clips are `missing_motion=1`. [VERIFIED]

---

## 6. Gesture Cue — Full Data Trace

Files: `runners/gesture_runner.py`, `Gesture Repo/model/keypoint_classifier/*`,
`Gesture Repo/model/point_history_classifier/*` (+ `.tflite` weights and label
CSVs).

### 6.1 Input

- Decoded frame `(H,W,3)` uint8 BGR → rotate → `resize_with_aspect_ratio(960)`
  → **`cv.flip(image,1)` horizontal mirror** → RGB → `hands.process`.
- MediaPipe Hands (`max_num_hands=2`, det/track conf 0.45) returns up to 2 hands
  of 21 landmarks each. [VERIFIED, `gesture_runner.py:202-220`]

> The mirror flip matters: it swaps left/right handedness and X geometry
> before all downstream keypoint classification and wave/beckon detection.

### 6.2 Preprocessing (per detected hand)

| Step | Function | Input | Operation | Output | Shape |
|---|---|---|---|---|---|
| 1 | `calc_landmark_list` | 21 lm | → pixel `[x,y]` list | list | `21×2` |
| 2 | EMA smoothing (`alpha=0.45`) | raw list | temporal smoothing vs previous frame | list | `21×2` |
| 3 | `pre_process_landmark` | smoothed | subtract wrist origin, flatten, /max\|v\| (÷0 guarded) | list[float] | `42` |
| 4 | `keypoint_classifier(vec)` | `42` | TFLite → `(sign_id, sign_conf)` | (int, float) | — |
| 5 | `pre_process_point_history` | 16-pt history | normalise by image dims, flatten | list[float] | `32` |
| 6 | `point_history_classifier(hist)` | `32` | TFLite → `(action_id, action_conf)` (if `sign==2`) | (int, float) | — |

`KeyPointClassifier` inputs `np.array([landmark_list], float32)` shape
`(1,42)`; `PointHistoryClassifier` inputs `(1,32)`; both `argmax` the output
and return `(index, confidence)`. [VERIFIED, classifier `__call__`]

Label vocabularies [VERIFIED from label CSVs]:
- keypoint signs: `0 Open Palm, 1 Close, 2 Pointer, 3 Thumbs Up, 4 Thumbs Down, 5 Beckoning`
- point-history actions: `0 Stop, 1 Clockwise, 2 Counter Clockwise, 3 Move, 4 Wave, 5 Come Here`

### 6.3 Decision logic (this is where the gesture "class" is actually formed)

Gesture does **not** output the raw classifier label. It runs a hand-crafted
**Global Scenario Resolution** (copied verbatim from `test_video.py`) over 1 or
2 hands, combining sign IDs, action IDs, `check_hand_raised`, `detect_wave`,
`detect_come_here` into a scenario **text** (`"Wave"`, `"Arms up"`,
`"Pointing"`, `"Thumbs up"`, `"One hand raised"`, `"Beckoning"`, …), then maps
via `GESTURE_SCENARIO_TO_CANONICAL` (`constants.py`) to the canonical label.
Key confidence gate: sign IDs 2–5 with `conf<0.80` are demoted to `-1`
(unknown). [VERIFIED, `gesture_runner.py:253-354`]

`GESTURE_SCENARIO_TO_CANONICAL`: `Wave/Brief wave/Arms waving→wave`,
`Pointing→point`, `Thumbs up→thumbs_up`, `Thumbs down→thumbs_down`,
`One hand raised→raise_hand`, `Arms up→both_hands_up`, `Beckoning→beckoning`,
`None→Unknown`. [VERIFIED]

### 6.4 Prediction Generation

```python
label = GESTURE_SCENARIO_TO_CANONICAL.get(global_scenario_text, "Unknown")
confidence = float(global_conf) if global_scenario_text != "None" else 0.0
valid = (label != "Unknown" and confidence >= 0.80)   # CONFIDENCE_FLOOR["gesture"]=0.80
```
**`probs` is always `{}` for gesture** — it emits only a discrete label +
scalar confidence, never a distribution. [VERIFIED]

### 6.5 Exact Returned Output (per frame)

Real record (no gesture) [VERIFIED]:
```json
{"cue":"gesture","frame_idx":0,"label":"Unknown","confidence":0.0,"probs":{},
 "valid":false,
 "extra":{"point_direction":null,"motion_direction":"none","point_target":"unknown"},
 "clip_id":"S01_F04_c001"}
```
`extra.motion_direction` and `extra.point_target` are **hardcoded constants**
for every frame of every clip (`"none"`/`"unknown"`) — no logic populates them.
They are never read downstream (§17). A detected frame carries e.g.
`label="raise_hand", confidence=0.96`.

### 6.6 Downstream Data Trace

Gesture is the **only** cue aggregated by
`_majority_onehot_features(gesture_records, GESTURE_CLASSES)`:
- majority-vote label among valid frames (`Counter(...).most_common(1)`)
- one-hot into `GESTURE_CLASSES = [wave,point,thumbs_up,thumbs_down,raise_hand,
  both_hands_up,beckoning,Unknown]` (8 dims)
- `mean_confidence` = mean confidence **of the winning label's frames only**
- `valid_fraction`
→ `gesture_wave…gesture_Unknown`, `gesture_mean_confidence`,
`gesture_valid_fraction`, `missing_gesture`. [VERIFIED] Confirmed against the
Parquet: gesture block row-sums are exactly `{0, 1}` (true one-hot / all-zero),
unlike the mean-prob cues. Gesture is the most-often-missing cue: **457/1270**
clips have `missing_gesture=1`. [VERIFIED]

---

## 7. Context Cue — Full Data Trace

Files: `runners/context_runner.py`, `Context Repo/scene classification/video.py`.
Weights: `Context Repo/scene classification/best_EfficientNet_B0.pth`.

### 7.1 Input

Decoded frame `(H,W,3)` uint8 BGR → `cv2.cvtColor(BGR2RGB)`. No detection /
crop — the **whole frame** is classified. [VERIFIED, `context_runner.py:74`]

### 7.2 Preprocessing

| Step | Function | Input | Operation | Output | Shape |
|---|---|---|---|---|---|
| 1 | `cv2.cvtColor(BGR2RGB)` | `(H,W,3)` uint8 BGR | → RGB | ndarray | same |
| 2 | `transform` = `ToPILImage`+`Resize((224,224))`+`ToTensor`+`Normalize(ImageNet)` | RGB ndarray | → tensor | `torch.float32` | `(3,224,224)` |
| 3 | `.unsqueeze(0)` | tensor | batch dim | `torch.float32` | `(1,3,224,224)` |

Note context's transform starts with `ToPILImage` (takes an ndarray), whereas
emotion's takes a PIL image directly — different first step, same end result.
[VERIFIED, `video.py:49-55`]

### 7.3 Model Architecture

`build_model()`: torchvision **`efficientnet_b0(weights=None)`** with
`classifier[1] = nn.Linear(in_features, 2)`. `SCENE_LABELS=["classroom","kitchen"]`.
Weights from `best_EfficientNet_B0.pth` (raw `state_dict`). Flow:
`(1,3,224,224)` → EfficientNet-B0 → `(1,2)` logits. [VERIFIED]

### 7.4 Prediction Generation — with temporal smoothing inside the runner

`context_runner.py:66-89`:
```python
probs_vec = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()   # (2,)
prob_history.append(probs_vec)          # deque(maxlen=SMOOTH_WINDOW=15)
avg  = np.mean(prob_history, axis=0)    # rolling-mean over ≤15 frames
idx  = int(avg.argmax()); conf = float(avg[idx])
native = SCENE_LABELS[idx] if conf >= CONF_THRESHOLD(0.5) else "uncertain"
label  = "Unknown" if native == "uncertain" else native
probs  = {lbl: float(p) for lbl,p in zip(SCENE_LABELS, avg)}   # the SMOOTHED avg
valid  = (conf >= 0.50 and label != "Unknown")
```
So context's per-frame `confidence`/`probs` are already **temporally
smoothed** (unlike emotion/motion, which are raw per-frame). [VERIFIED]

### 7.5 Exact Returned Output (per frame)

Real record [VERIFIED]:
```json
{"cue":"context","frame_idx":0,"label":"classroom","confidence":0.99823,
 "probs":{"classroom":0.99823,"kitchen":0.00177},"valid":true,
 "extra":{"activity":null,"engaged":null,"n_objects":0},"clip_id":"S01_F04_c001"}
```
`extra.activity/engaged/n_objects` are **structural placeholders**
(`NOT_MEASURED_EXTRA`) — the model has no object/activity/engagement head;
these never vary and are never read downstream. [VERIFIED]

### 7.6 Downstream Data Trace

Aggregated by `_prob_mean_features(context_records, CONTEXT_CLASSES)` giving 2
**mean probabilities** `context_classroom, context_kitchen`; but the confidence
column is computed **separately** as `context_mean_confidence` = mean of
per-frame `confidence` over valid frames (not `max`, unlike emotion/motion).
Plus `context_valid_fraction`, `missing_context`. [VERIFIED, `aggregate.py:145-153`]
In the Parquet, **0/1270** clips are `missing_context` (context is essentially
always available). [VERIFIED]

> **Documentation bug:** `aggregate.py`'s header docstring (line 39) calls
> the context block "2 one-hot scene". It is **not** one-hot — it is the mean
> probability vector. Verified: all 1270 rows have `context_classroom` /
> `context_kitchen` values strictly inside (0,1), never a clean `{0,1}`. See §17.

---

## 8. Cue Output Comparison

Per-frame `NormalisedFrameCue` contract (all four share the schema; contents
diverge):

| Cue | `label` vocabulary | `confidence` | `probs` | `valid` gate | `extra` |
|---|---|---|---|---|---|
| Emotion | 7 emotions or `Unknown` | raw per-frame softmax max | **7-key dict** | `conf≥0.50` | `{bbox}` |
| Motion | 4 motions or `Unknown`(buffering) | raw per-frame softmax max | **4-key dict** (empty while buffering) | `not buffering ∧ conf≥0.50 ∧ has_landmarks` | `{buffering,has_landmarks}` |
| Gesture | 8-way canonical or `Unknown` | heuristic scenario conf | **always `{}`** | `label≠Unknown ∧ conf≥0.80` | `{point_direction,motion_direction,point_target}` (constant) |
| Context | `classroom`/`kitchen`/`Unknown` | **smoothed** mean over ≤15 frames | **2-key dict** (smoothed) | `conf≥0.50 ∧ label≠Unknown` | `{activity,engaged,n_objects}` (constant) |

**Aggregated (per-clip) contract that fusion actually sees:**

| Cue | Aggregation | Block in feature vector | Example (real, `S01_F04_c001`) | Fusion usage |
|---|---|---|---|---|
| Emotion | **mean probs** over valid frames | 7 probs + `max_confidence` + `valid_fraction` | `Neutral≈0.993`, `Happy≈0.0065`, max_conf≈1.0, valid_frac=1.0 | rule: argmax→label; GBT: raw 7 probs + 2 scalars |
| Gesture | **majority one-hot** | 8 one-hot + `mean_confidence` + `valid_fraction` | `raise_hand=1.0`, mean_conf≈0.958, valid_frac≈0.635 | rule: argmax→label; GBT: raw one-hot + 2 scalars |
| Motion | **mean probs** | 4 probs + `max_confidence` + `valid_fraction` | `sitting≈1.0`, max_conf=1.0, valid_frac≈0.608 | rule: argmax→label; GBT: raw 4 probs + 2 scalars |
| Context | **mean probs** | 2 probs + `mean_confidence` + `valid_fraction` | `classroom≈0.996`, mean_conf≈0.996, valid_frac=1.0 | rule: argmax→label; GBT: raw 2 probs + 2 scalars |

**Key incompatibility, made obvious:** at the per-frame level the four cues are
**not** structurally uniform — gesture carries no probability distribution at
all, and context is pre-smoothed while emotion/motion are not. The aggregation
layer is what forces them into one flat numeric vector, and even there gesture
is one-hot while the other three are mean-prob. [VERIFIED]

---

## 9. Cue Output Collection and Aggregation

### Granularity

- **One JSONL line per frame per clip per cue** (Stage 1 output).
  [VERIFIED: 141,721 lines each = total frames]
- **One feature row per clip** after aggregation (Stage 2).
  [VERIFIED: Parquet is 1270 rows]
- There is **no per-window or per-second** intermediate; the temporal window
  exists only *inside* motion's LSTM (30 frames) and context's smoothing
  deque (15 frames), not in the fusion-facing aggregation.

### Collection structure

`build_features.py` builds, per cue, a dict `{clip_id: [frame_record,…]}` via
`load_frame_cues_by_clip`, then calls `build_clip_feature_row(clip_id,
e_records, g_records, m_records, c_records)`. The collected object is
effectively:
```python
frames_by_cue = {
  "emotion": {clip_id: [rec, rec, …], …},
  "gesture": {clip_id: [...]},
  "motion":  {clip_id: [...]},
  "context": {clip_id: [...]},
}
```
[VERIFIED, `build_features.py:46,64-70`]

### The actual aggregation algorithms

There are **two different** algorithms — not one shared voting scheme:

**A. Mean-probability (emotion, motion, context)** — `_prob_mean_features`:
1. keep only `valid` frames;
2. `valid_fraction = n_valid / n_total`;
3. if no valid frames → zeros block, `max_confidence=0`, and caller sets
   `missing_<cue>=1`;
4. else stack each valid frame's `probs` (missing keys→0.0) → mean over axis 0
   → per-class mean probability;
5. `max_confidence = max(confidence over valid frames)` (context overrides
   this with **mean** confidence).

**B. Majority-vote one-hot (gesture only)** — `_majority_onehot_features`:
1. keep only `valid` frames;
2. `Counter(label).most_common(1)` → winner;
3. one-hot the winner into `GESTURE_CLASSES`;
4. `mean_confidence` = mean confidence of winner-labelled valid frames.

`missing_<cue> = float(valid_fraction < 0.40)` (`CLIP_MISSING_THRESHOLD`).
No weighting, no last-frame selection, no smoothing at this stage, no
cross-cue temporal alignment. [VERIFIED]

### Numerical walkthrough — gesture majority vote (algorithm B)

Suppose a clip's gesture frames (valid only): `raise_hand, raise_hand, Unknown*
(invalid, dropped), raise_hand, wave`. After dropping invalid:
```
Counter: raise_hand=3, wave=1  →  winner = raise_hand
one-hot: gesture_raise_hand=1.0, all others 0.0
gesture_mean_confidence = mean(conf of the 3 raise_hand frames)
gesture_valid_fraction  = 4 valid / N_total
```
[matches `_majority_onehot_features`]

### Numerical walkthrough — motion mean-prob (algorithm A)

Two valid frames with `probs`:
```
frame A: sitting 0.90 standing 0.08 walking 0.01 stepping_back 0.01
frame B: sitting 0.80 standing 0.15 walking 0.03 stepping_back 0.02
mean   : sitting 0.85 standing 0.115 walking 0.02 stepping_back 0.015
motion_max_confidence = max(0.90, 0.80) = 0.90
```
[matches `_prob_mean_features`]

> Note: `pipeline/aggregate_clip_cues.py` implements a **third** aggregation
> (majority-vote dominant label for **all** cues) — but that output feeds only
> the Phase-0 agreement report, **never fusion**. Do not conflate it with the
> Stage-2 aggregation above. [VERIFIED]

---

## 10. Exact Fusion Boundary

The boundary is the Parquet file. Everything left of it is cue-side;
everything right is fusion-side. The 33 feature columns (index within
`FEATURE_NAMES`) are:

```
idx  column                      source / meaning
0    emotion_Surprise            mean prob
1    emotion_Fear                mean prob
2    emotion_Disgust             mean prob
3    emotion_Happy               mean prob
4    emotion_Sad                 mean prob
5    emotion_Anger               mean prob
6    emotion_Neutral             mean prob
7    emotion_max_confidence      max conf over valid frames
8    emotion_valid_fraction      valid/total
9    gesture_wave                one-hot
10   gesture_point               one-hot
11   gesture_thumbs_up           one-hot
12   gesture_thumbs_down         one-hot
13   gesture_raise_hand          one-hot
14   gesture_both_hands_up       one-hot
15   gesture_beckoning           one-hot
16   gesture_Unknown             one-hot
17   gesture_mean_confidence     mean conf of winning label
18   gesture_valid_fraction      valid/total
19   motion_sitting              mean prob
20   motion_standing             mean prob
21   motion_walking              mean prob
22   motion_stepping_back        mean prob
23   motion_max_confidence       max conf over valid frames
24   motion_valid_fraction       valid/total
25   context_classroom           mean prob
26   context_kitchen             mean prob
27   context_mean_confidence     mean conf over valid frames
28   context_valid_fraction      valid/total
29   missing_emotion             1.0 if emotion_valid_fraction < 0.40
30   missing_gesture             1.0 if gesture_valid_fraction < 0.40
31   missing_motion              1.0 if motion_valid_fraction < 0.40
32   missing_context             1.0 if context_valid_fraction < 0.40
```
[VERIFIED against Parquet column order and `aggregate.FEATURE_NAMES`.]

```text
CUE SIDE (per-clip aggregated blocks)
-------------------------------------
emotion → 7 mean probs + max_conf + valid_frac (+ missing bit)
gesture → 8 one-hot     + mean_conf + valid_frac (+ missing bit)
motion  → 4 mean probs  + max_conf + valid_frac (+ missing bit)
context → 2 mean probs  + mean_conf + valid_frac (+ missing bit)

        ↓ (this is the ONLY transformation at the boundary)

RULE-BASED path: _dominant(row, prefix, classes)
   → if missing_<cue> ≥ 1.0: label = None
   → elif max(block values) ≤ 0: label = None
   → else: label = classes[argmax(block values)]
   i.e. each cue block is collapsed back to ONE categorical label (or None).

GBT path: NO transformation.
   → X = df[FEATURE_NAMES]  (all 33 raw floats, in the order above)
   → straight into LGBMClassifier. No argmax, no re-encoding, no scaling.
```

**Label→number mappings at the boundary:**
- Rule-based converts a numeric block back to a label via `argmax`; the class
  order is `EMOTION_CLASSES / GESTURE_CLASSES / MOTION_CLASSES /
  CONTEXT_CLASSES` from `aggregate.py`. No numeric encoding of the *output* —
  the rule output is already an intent **string**.
- GBT does **not** encode cue labels at all (it never sees labels; it sees the
  numeric blocks). Its **target** `y = df["intent"]` is left as strings;
  LightGBM handles the string classes internally (`model.classes_`). [VERIFIED]
- **Missing-cue defaults:** an absent cue is represented as an **all-zero
  block + `missing_<cue>=1.0`**. Rule-based reads that as `label=None`
  (via the `missing_` guard); GBT reads the literal zeros + the missing bit as
  features. [VERIFIED]

---

## 11. Rule-Based Fusion — Full Internal Trace

File: `fusion/rule_based.py`.

### 11.1 Exact Input

- `df` = `read_parquet(clip_features.parquet)`; one `row` = a pandas Series
  with the 33 feature columns above (+ metadata).
- `predict_intent(row, fallback_intent)` first computes four labels via
  `_dominant`:
  `emotion∈EMOTION_CLASSES∪{None}`, `gesture∈GESTURE_CLASSES∪{None}`,
  `motion∈MOTION_CLASSES∪{None}`, `context∈{classroom,kitchen,None}`.
  [VERIFIED, `rule_based.py:68-83`]

### 11.2 Rule Evaluation Order (verbatim execution order)

```text
R1  if gesture == "both_hands_up":
        if context=="kitchen" and emotion=="Anger" and motion=="standing":  return "F07"
        else:                                                                return "F02"
R2  if gesture=="thumbs_down" and motion=="stepping_back" and emotion=="Disgust":  return "F08"
R3  if gesture in ("raise_hand","thumbs_down") and motion=="sitting":  return "F04"
R4  if gesture in ("point","raise_hand") and motion in ("sitting","standing"):  return "F05"
R5  if gesture=="beckoning":  return "F03"
    if gesture=="point" and motion=="walking":  return "F03"
R6  if emotion=="Happy" and gesture in ("wave","thumbs_up"):
        if context=="kitchen" and motion=="walking":  return "F09"
        else:                                          return "F01"
R7  if gesture=="point" and motion=="stepping_back":  return "F06"
R8  return fallback_intent           # = train-set intent mode = "F04"
```
[VERIFIED, `rule_based.py:85-125`]

### 11.3 Priority and Conflict Resolution

- **First match wins, function returns immediately** (no score accumulation).
- Priority is strictly top-to-bottom: emergency **F02/F07** (`both_hands_up`)
  is checked before everything, matching the doc's "any meaningful evidence of
  F02 escalates" asymmetric-cost stance.
- **Fallback** for anything unmatched: `fit_fallback(train_df)` =
  `train_df["intent"].mode().iloc[0]` = **"F04"** (verified at runtime). The
  module constant `DEFAULT_FALLBACK_INTENT="F05"` is overridden by the fitted
  value when driven through `predict_all(..., fallback_intent=fallback)`.
  [VERIFIED]
- The author flags two **irreducible** ambiguities in the source scenario
  table (F02-vs-F07 and F04-vs-F10) that cap achievable accuracy; R3's
  `→F04` tie-break will misclassify every true F10 clip by construction.
  [VERIFIED, docstring + `scenarios.csv` S21/S28]

### 11.4 Manual Example

Real clip `S01_F04_c001` (dominant labels from its feature row: emotion
`Neutral`, gesture `raise_hand`, motion `sitting`, context `classroom`;
intent truth `F04`):
```text
R1 both_hands_up?              gesture=raise_hand → False
R2 thumbs_down∧stepping_back∧Disgust? → False
R3 gesture∈{raise_hand,thumbs_down} ∧ motion==sitting?
     raise_hand ∧ sitting → TRUE → return "F04"
R4..R8 not evaluated (already returned)
Final = "F04"   ✓ (matches truth)
```
[VERIFIED — labels read from the actual Parquet row.]

### 11.5 Exact Output

- Return type: **`str`**, one of `F01…F10` (`predict_intent`).
- `predict_all(df, ...)` → `df.apply(...)` → a **pandas Series of strings**,
  one per clip; assigned to `df["rule_pred"]`.
- Accuracy is then `(df["rule_pred"] == df["intent"]).mean()`. Not serialized.
  [VERIFIED]

---

## 12. GBT Fusion — Full Internal Trace

File: `fusion/gbt.py`.

### 12.1 Raw Cue Inputs

The **same 33-column Parquet**. GBT uses `df[FEATURE_NAMES]` directly — the raw
mean-prob/one-hot/confidence/valid/missing floats. No argmax, no label
reconstruction. [VERIFIED]

### 12.2 Feature Construction

There is essentially **no separate feature-construction step** for GBT — the
Stage-2 aggregation already produced the model-ready vector. The only
train-time manipulation is **modality dropout** (`apply_modality_dropout`,
train split only):
```python
for cue in {emotion,gesture,motion,context}:
    drop_mask = rng.random(n) < 0.15          # DROPOUT_P
    value_cols = [cols of that cue except *_valid_fraction]
    X.loc[drop_mask, value_cols] = 0.0
    X.loc[drop_mask, f"{cue}_valid_fraction"] = 0.0
    X.loc[drop_mask, f"missing_{cue}"] = 1.0
```
i.e. it randomly simulates a missing cue by zeroing its block + valid_fraction
and setting its missing bit — exactly mirroring how real missingness is
encoded. **Inference applies no dropout.** [VERIFIED]

The exact feature **order** is `aggregate.FEATURE_NAMES` (the 34-item list),
enforced because both `fit` and `predict` index `df[FEATURE_NAMES]`. This order
is provable and equals the index table in §10. [VERIFIED]

### 12.3 Encoding

- **Cue features:** none — already numeric. Gesture is pre-one-hot from
  aggregation; the other three are mean probabilities. No label/one-hot
  encoding happens in `gbt.py`.
- **Target:** `y = df["intent"]` kept as **strings**; LightGBM builds its own
  internal class list `model.classes_` (sorted intent codes). Predictions map
  back via `model.classes_[argmax]`. [VERIFIED]
- No category mapping dict, no missing-value imputation beyond the modality
  dropout described above (real missing cues arrive as zeros+bit from Stage 2).

### 12.4 Example Feature Vector

For real clip `S01_F04_c001`, the 33-float vector (order of §10) is:
```python
[0.00063, 4.8e-07, 5.1e-05, 0.00653, 1.7e-05, 6.2e-07, 0.99277,   # emotion 7 probs
 0.999999, 1.0,                                                    # emotion max_conf, valid_frac
 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,                           # gesture one-hot (raise_hand=1)
 0.95761, 0.63514,                                                 # gesture mean_conf, valid_frac
 0.99999992, 8.0e-08, 1.6e-09, 1.1e-09,                            # motion 4 probs (sitting≈1)
 1.0, 0.60811,                                                     # motion max_conf, valid_frac
 0.99645, 0.00355,                                                 # context probs (classroom)
 0.99645, 1.0,                                                     # context mean_conf, valid_frac
 0.0, 0.0, 0.0, 0.0]                                               # missing bits (none)
```
[VERIFIED — copied from the real Parquet row.] Every value is either a mean
probability, a one-hot flag, a confidence scalar, a valid-fraction, or a
missing bit — per §10.

### 12.5 GBT Model Architecture

- Library: **LightGBM** (`from lightgbm import LGBMClassifier`).
- Params (`gbt.py:87-95`): `objective="multiclass"`,
  `class_weight="balanced"`, `n_estimators=300` (trees), `max_depth=5`,
  `learning_rate=0.05`, `random_state=42`, `verbosity=-1`. [VERIFIED]
- Trained on `train_df` (872 clips) with modality dropout; targets = 10 intent
  classes. No calibration (isotonic/Platt) — explicitly deferred per docstring.

**[CONCEPTUAL] What GBT does:** gradient boosting fits an additive ensemble of
shallow decision trees; each new tree is trained to correct the residual error
(gradient of the multiclass log-loss) of the running ensemble, scaled by the
learning rate. For multiclass, LightGBM effectively grows one set of trees per
class and softmaxes their summed leaf scores into class probabilities.
**In this project:** 300 depth-≤5 trees split on the 33 numeric cue features
(e.g. "is `motion_sitting` ≥ 0.5?", "is `gesture_mean_confidence` ≥ 0.8?");
`class_weight="balanced"` up-weights rarer intent classes.

### 12.6 Prediction Trace

```text
X = split_df[FEATURE_NAMES]            # (n, 33) float
   ↓ predict_with_safety_override
proba = model.predict_proba(X)         # (n, 10) softmaxed class probs
argmax_idx = proba.argmax(axis=1)      # (n,)
preds = model.classes_[argmax_idx]     # (n,) intent strings
escalate = proba[:, f02_idx] >= 0.15   # F02_SAFETY_THRESHOLD
preds = np.where(escalate, "F02", preds)   # force F02 whenever its prob ≥ 0.15
return preds, proba
```
[VERIFIED, `gbt.py:68-74`] `f02_idx = list(model.classes_).index("F02")`.

### 12.7 Conceptual Tree Walkthrough

The 300 fitted trees are held only in memory (the model is never saved), so I
did **not** dump literal split thresholds. **[CONCEPTUAL, clearly labelled]** a
representative tree in this model would look like:
```text
if motion_sitting >= 0.5:
    if gesture_raise_hand >= 0.5:  leaf → boosts F04
    else:                          leaf → boosts F05
else:
    if gesture_both_hands_up >= 0.5: leaf → boosts F02
    else:                            leaf → boosts F01
```
This is illustrative only — not the project's real tree. What **is** verified
is the ranked feature importance (gain) the trained model reports:
`gesture_mean_confidence` (2494) > `motion_sitting` (2282) >
`motion_valid_fraction` (2274) > `context_classroom` (2151) >
`motion_standing` (2124) > `gesture_valid_fraction` (2081) >
`emotion_Surprise` (2059) > `emotion_Neutral` (1903) > `context_kitchen` (1828)
> `motion_stepping_back` (1748). [VERIFIED by running `gbt.py`]

### 12.8 Exact GBT Output

- `predict_with_safety_override` returns `(preds, proba)`:
  `preds` = `numpy.ndarray` of dtype `<U3` strings shape `(n,)`
  (values `F01…F10`); `proba` = `numpy.ndarray` `(n,10)` float.
- Accuracy = `(preds == split_df["intent"].values).mean()`. Printed, not saved.
- Measured (test, 190 clips): **GBT acc 0.237**; F02 recall 0.700 (15 FN / 50).
  [VERIFIED by running]

---

## 13. Training-Time vs Inference-Time Pipeline

There is only ONE trained fusion model with a train/infer split — **GBT**
(rule-based has no learned parameters, only the `fit_fallback` mode). The cue
models are pre-trained upstream; the runners are inference-only. The important
comparison is therefore *within the fusion stage*.

| Stage | Training | Inference | Same? |
|---|---|---|---|
| Emotion preprocessing | (upstream, frozen) | identical runner | ✅ same runner code |
| Motion preprocessing | (upstream, frozen) | identical runner | ✅ |
| Gesture preprocessing | (upstream, frozen) | identical runner | ✅ |
| Context preprocessing | (upstream, frozen) | identical runner | ✅ |
| Cue aggregation | `aggregate.py` (same code) | `aggregate.py` (same code) | ✅ single implementation |
| GBT feature creation | `df[FEATURE_NAMES]` **+ modality dropout (p=0.15)** | `df[FEATURE_NAMES]` **no dropout** | ⚠️ **different** (intentional augmentation) |
| Encoding | none (numeric) / target strings | none / `classes_[argmax]` | ✅ |
| Safety override | not applied during `fit` | `P(F02)≥0.15 → F02` at predict | ⚠️ **inference-only** |
| Fallback (rule) | `fit_fallback` = train mode "F04" | same value reused | ✅ |

**Train/inference deltas that matter:**
1. **Modality dropout** is train-only by design — makes GBT robust to missing
   cues it rarely saw in this near-complete dataset. Not a bug. [VERIFIED]
2. **F02 safety override** is inference-only — so the model's *reported*
   `predict_proba` argmax and its *final* prediction can differ, and training
   never optimizes against the override. This is a genuine train/serve
   asymmetry to keep in mind when reading metrics. [VERIFIED]
3. GBT train accuracy is evaluated on **un-augmented** `train_df[FEATURE_NAMES]`
   (not the dropout-augmented `X_train`), so the printed train number is
   optimistic relative to what the model was actually fit on. [VERIFIED,
   `gbt.py:84 vs 109`]

There is **no** classic feature-scaling / normalization mismatch, because the
same `aggregate.py` produces the vector for both, and LightGBM needs no scaling.

---

## 14. Full Worked Example — Rule-Based Pipeline

Real clip `S01_F04_c001` (classroom, subject P01, truth **F04**):

```text
RAW INPUT
↓  raw/clips/classroom/S01_F04/S01_F04_c001.mp4  (640×480, 15fps, 74 frames)

EMOTION MODEL (per frame → jsonl)
↓  frame0: {"label":"Neutral","confidence":0.99996,"probs":{...Neutral:0.99996},"valid":true}
   … 74 frames, essentially all Neutral, all valid

GESTURE MODEL
↓  frame0: {"label":"Unknown","confidence":0.0,"probs":{},"valid":false}
   … valid frames vote raise_hand (winner), valid_fraction≈0.635

MOTION MODEL
↓  frame0..28: {"label":"Unknown","buffering":true,"valid":false}
   frame29+: {"label":"sitting","probs":{sitting≈1.0,...},"valid":true}

CONTEXT MODEL
↓  frame0: {"label":"classroom","confidence":0.998,"probs":{classroom:0.998,kitchen:0.002}}

CUE COLLECTION (build_features.load_frame_cues_by_clip)
↓  {emotion:[74 recs], gesture:[74], motion:[74], context:[74]} for this clip_id

AGGREGATION (aggregate.build_clip_feature_row)
↓  emotion_Neutral≈0.993 (mean prob), emotion_max_confidence≈1.0, emotion_valid_fraction=1.0
   gesture_raise_hand=1.0 (majority one-hot), gesture_mean_confidence≈0.958, gesture_valid_fraction≈0.635
   motion_sitting≈1.0 (mean prob), motion_max_confidence=1.0, motion_valid_fraction≈0.608
   context_classroom≈0.996 (mean prob), context_mean_confidence≈0.996, context_valid_fraction=1.0
   missing_* all 0.0

RULE-BASED FUSION INPUT (_dominant argmax per block)
↓  emotion="Neutral", gesture="raise_hand", motion="sitting", context="classroom"

RULE EXECUTION
↓  R1 both_hands_up? no
   R2 thumbs_down∧stepping_back∧Disgust? no
   R3 gesture∈{raise_hand,thumbs_down} ∧ motion==sitting?  YES → return "F04"

FINAL OUTPUT
↓  "F04"   (string; matches truth F04) ✓
```
[VERIFIED end-to-end against the real JSONL + Parquet + rule code.]

---

## 15. Full Worked Example — GBT Pipeline

Same clip `S01_F04_c001`:

```text
Four cue outputs (aggregated blocks) — identical to §14 aggregation
↓
FEATURE EXTRACTION  (df[FEATURE_NAMES], no argmax)
↓
ENCODING  — none (already numeric); target intent kept as string
↓
EXACT ORDERED FEATURE VECTOR (33 floats, §10 order)
   [0.00063,4.8e-07,5.1e-05,0.00653,1.7e-05,6.2e-07,0.99277,  1.0,1.0,     # emotion
    0,0,0,0,1,0,0,0,  0.95761,0.63514,                                     # gesture (raise_hand=1)
    1.0,8e-08,1.6e-09,1.1e-09,  1.0,0.60811,                               # motion (sitting≈1)
    0.99645,0.00355,  0.99645,1.0,                                         # context (classroom)
    0,0,0,0]                                                               # missing bits
↓
GBT INPUT: X = (1,33) → LGBMClassifier
↓
GBT PREDICTION: proba = predict_proba(X) → (1,10); preds = classes_[argmax]
↓
POST-PROCESSING: if proba[:,f02_idx] ≥ 0.15 → force "F02" (safety override)
↓
FINAL OUTPUT: one intent string (e.g. "F04")
```
[VERIFIED for the feature vector and the predict path; the exact per-clip GBT
class output for this single clip was not individually printed by the script —
the script reports split-level accuracy, not per-clip predictions — so the
final string here is illustrative of the mechanism. The mechanism itself is
VERIFIED.]

---

## 16. Final Architecture Diagram

```text
                                  RAW VIDEO CLIPS (.mp4)
                                  clips.csv enumerates 1270
                                            │
        ┌──────────────────┬────────────────┼────────────────┬──────────────────┐
        │                  │                │                │                  │
        ▼                  ▼                ▼                ▼                  │
  emotion_runner     gesture_runner    motion_runner    context_runner        (4 separate
  (.venvs/emotion)   (.venvs/gesture)  (.venvs/motion)  (.venvs/context)       processes,
  MobileNetV2→7      MediaPipe Hands   MediaPipe Pose   EfficientNet-B0→2      no shared
  softmax probs      +2 TFLite +       →MotionLSTM      softmax probs,          driver)
                     heuristic FSM     (30-frame win)   15-frame smoothing
        │                  │                │                │
        ▼                  ▼                ▼                ▼
  emotion_frame     gesture_frame     motion_frame     context_frame
  _cues.jsonl       _cues.jsonl       _cues.jsonl      _cues.jsonl      ← DISK BOUNDARY (schema)
        │                  │                │                │
        └──────────────────┴───────┬────────┴────────────────┘
                                    ▼
                    build_features.py + aggregate.py
                    per-clip aggregation:
                      emotion/motion/context → MEAN PROBS
                      gesture               → MAJORITY ONE-HOT
                      + confidence + valid_fraction + missing bit
                                    │
                                    ▼
                    data/features/clip_features.parquet
                    1270 rows × (clip_id + 33 features + 6 meta)   ← FUSION BOUNDARY
                                    │
              ┌─────────────────────┴─────────────────────┐
              ▼                                             ▼
   fusion/rule_based.py                            fusion/gbt.py
   _dominant() argmax per block                    df[FEATURE_NAMES] (raw 33)
   → 8 priority IF-THEN rules                       → LGBMClassifier (300×depth5)
   → fallback "F04"                                 → predict_proba → argmax
   │                                                → F02 safety override (≥0.15)
   ▼                                                        │
   intent string F01..F10                                   ▼
   (stdout accuracy)                              intent string F01..F10
                                                  (stdout accuracy; model NOT saved)
```

---

## 17. Critical Findings and Implementation Problems

Ordered roughly by impact.

### F-1  No runtime fusion / no persisted model — this is an offline eval harness
```
Problem:      There is no end-to-end "video → intent" runtime. Both fusion
              scripts read a precomputed Parquet, print accuracy, and exit.
              The GBT model is never serialized (no joblib/pickle anywhere).
Evidence:     grep for joblib|pickle|dump|save_model → NONE FOUND. gbt.py.main()
              trains + prints, returns nothing. No FusionEngine class exists.
File/fn:      fusion/gbt.py::main, fusion/rule_based.py::__main__
Why matters:  To deploy, someone must add serialization + an online path that
              re-runs the 4 runners live and re-implements aggregate.py per
              stream. None of that exists.
Impact:       "Fusion" today = a batch benchmarking script, not a serving path.
```

### F-2  `run_cue_models.py` referenced but does not exist
```
Problem:      emotion_runner.py and motion_runner.py docstrings point users to
              "run_cue_models.py" for batch orchestration. The file is absent.
Evidence:     find . -name run_cue_models.py → not found (outside venvs).
File/fn:      runners/emotion_runner.py docstring; runners/motion_runner.py
Why matters:  There is no committed single command to regenerate all 4 JSONLs;
              each runner must be launched by hand in its own venv.
Impact:       Reproducibility gap; onboarding confusion.
```

### F-3  Stale "6-class" comments in the motion model (code is 4-class)
```
Problem:      inference.py (MotionResult.probs "shape (6,)", "(1,6)") and
              model.py::forward docstring ("returns (B,6)") describe 6 classes.
Evidence:     Checkpoint classifier weight shape = (4,64); MOTION_LABELS has 4
              entries; NUM_CLASSES=4. Verified by loading the .pt.
File/fn:      Motion Repo/inference.py:59,200-201 ; Motion Repo/model.py:66,83
Why matters:  Misleads any reader about the motion output width; the aggregation
              and rule/GBT feature layout correctly assume 4, so behaviour is OK.
Impact:       Documentation-only, but actively misleading. No runtime effect.
```

### F-4  Context block documented as one-hot but is actually mean-prob
```
Problem:      aggregate.py header (line 39) says context = "2 one-hot scene".
              It is the mean probability vector, not one-hot.
Evidence:     All 1270 Parquet rows have context_classroom/context_kitchen
              strictly in (0,1); build uses _prob_mean_features, not
              _majority_onehot_features.
File/fn:      pipeline/aggregate.py:39 vs :145-153
Why matters:  A reader trusting the docstring would mis-model the feature; and
              only gesture is truly one-hot despite the header grouping them.
Impact:       Documentation inconsistency; no runtime bug (rule _dominant argmax
              works on mean-probs too).
```

### F-5  Cue outputs computed but never used (dead payload)
```
Problem:      Several emitted fields never reach fusion:
              - gesture extra.{point_direction, motion_direction, point_target}
                are hardcoded constants for every frame.
              - context extra.{activity, engaged, n_objects} are constant
                placeholders (model has no such heads).
              - motion extra.{buffering, has_landmarks}, emotion extra.bbox,
                per-frame probs dicts — used only for aggregation/validity,
                the raw label strings of emotion/motion/context are discarded
                (only mean-probs survive).
Evidence:     aggregate.py reads only probs/label/confidence/valid; nothing
              reads any extra.* field. gesture_runner.py:360, context_runner.py:48.
File/fn:      runners/*_runner.py extra dicts; pipeline/aggregate.py
Why matters:  These are documented as "explicit, not fabricated" placeholders,
              so it's intentional — but a naive integrator might wire them in
              expecting signal. They carry zero information.
Impact:       Vector size honestly excludes them; no leakage, just dead fields.
```

### F-6  Grouped test split contains only 4 of 10 intent classes
```
Problem:      split_scenario groups whole scenarios into train/val/test. With
              only 22 base scenarios, the test split covers just a handful of
              scenarios → only 4 intents appear (F01,F02,F08,F09).
Evidence:     Running rule_based.py / gbt.py: per-class test recall printed for
              exactly F01,F02,F08,F09. splits are scenario-grouped (build_splits.py).
File/fn:      pipeline/build_splits.py ; observed in fusion run output
Why matters:  Test accuracy (0.18 rule / 0.24 GBT) is over a tiny, non-
              representative label subset; NOT a trustworthy overall metric.
Impact:       Headline accuracy numbers are low-power and class-incomplete.
```

### F-7  subject/scenario splits are the SAME partition (confound)
```
Problem:      split_scenario and split_subject are identical partitions because
              the dataset is 1:1 subject↔scenario (each subject did one scenario).
Evidence:     build_splits.py explicitly detects and prints this (is_confounded,
              identical_partition). 23 subjects, 23 scenarios.
File/fn:      pipeline/build_splits.py:125-141
Why matters:  "Unseen person" and "unseen scenario" generalization cannot be
              measured independently on this dataset version.
Impact:       Evaluation-scope limitation (the code flags it honestly).
```

### F-8  Two irreducible label ambiguities cap all 4-cue fusion accuracy
```
Problem:      F02 vs F07 (S05 vs S24) and F04 vs F10 (S21 vs S28) have identical
              measured (emotion,gesture,motion,context) tuples mapping to
              different intents. F04-vs-F10 has NO distinguishing measured signal.
Evidence:     scenarios.csv rows; rule_based.py docstring ambiguities #1/#2.
File/fn:      fusion/rule_based.py:11-33 ; scenarios.csv S21/S28, S05/S24
Why matters:  Every true F10 clip is misclassified by R3 (→F04) by construction;
              this is a real accuracy ceiling, not fixable with these 4 cues.
Impact:       Upper bound on both fusion approaches. F10 recall ≈ 0 by design.
```

### F-9  GBT train accuracy evaluated on un-augmented data (mildly optimistic)
```
Problem:      Model is fit on X_train (with modality dropout) but the printed
              train accuracy is computed on train_df[FEATURE_NAMES] (no dropout).
Evidence:     gbt.py:84 (X_train dropout) vs :109 (X = split_df[FEATURE_NAMES]).
File/fn:      fusion/gbt.py:84,106-111
Why matters:  Train metric doesn't reflect the distribution the model trained on.
Impact:       Minor; only the train diagnostic, not val/test.
```

### F-10  Inference-only F02 safety override creates train/serve asymmetry
```
Problem:      predict_with_safety_override forces F02 when P(F02)≥0.15, applied
              only at inference; training never optimizes against it.
Evidence:     gbt.py:68-74 vs fit at :96.
File/fn:      fusion/gbt.py
Why matters:  Reported predictions can diverge from the model's own argmax; a
              threshold of 0.15 is aggressive (fires well below argmax), trading
              precision on other classes for F02 recall. Intentional per doc,
              but must be understood when reading confusion metrics.
Impact:       Behavioural (raises F02 recall to 0.70, adds F02 false positives).
```

### F-11  Frame-rate mismatch feeds the motion model out-of-distribution
```
Problem:      MotionLSTM assumes ~30fps (30-frame window ≈ 1s). 827/1270 clips
              are 15fps (window ≈ 2s), degrading accuracy per the model's README.
              No resampling is done (deliberately, per instruction).
Evidence:     motion_runner.py docstring; clips.csv fps column (many 15fps);
              checkpoint val_acc≈0.54.
File/fn:      runners/motion_runner.py:37-44
Why matters:  Motion mean-probs (a large share of GBT importance) are computed
              on out-of-distribution windows for most clips.
Impact:       Systematically weakens the motion cue on 65% of clips.
```

### F-12  Gesture is the dominant missing cue (36% of clips)
```
Problem:      457/1270 clips have missing_gesture=1 (valid_fraction<0.40),
              driven by the strict 0.80 gesture confidence floor + demotion of
              low-confidence signs to Unknown.
Evidence:     Parquet missing_gesture.sum()=457; CONFIDENCE_FLOOR["gesture"]=0.80.
File/fn:      runners/common/constants.py ; pipeline/aggregate.py
Why matters:  Gesture is the single most decisive cue in the rule cascade (most
              rules key on it) and top GBT importance — yet it's absent for a
              third of clips, forcing fallback/other-cue reliance.
Impact:       Large real effect on both fusion approaches.
```

### F-13  Non-obvious: emotion/motion use MAX confidence, context uses MEAN
```
Problem:      emotion_max_confidence & motion_max_confidence are the max over
              valid frames; context_mean_confidence is the mean. Inconsistent
              semantics under the similar "*_confidence" naming.
Evidence:     aggregate.py:102 (max) vs :149 (mean).
File/fn:      pipeline/aggregate.py
Why matters:  A GBT split on "confidence" means different things per cue; easy
              to misread when interpreting feature importance.
Impact:       Interpretability/consistency; intentional per handover spec.
```

---

## 18. Final Data Contract Summary

| Pipeline Boundary | Input Type | Input Shape | Output Type | Output Shape | Example |
|---|---|---|---|---|---|
| Raw → Emotion | ndarray uint8 BGR frame | (480,640,3) | NormalisedFrameCue (JSON line) | 7-key probs + scalars | `{"label":"Neutral","confidence":0.99996,...}` |
| Raw → Gesture | ndarray uint8 BGR frame | (480,640,3) | NormalisedFrameCue | label + conf, `probs={}` | `{"label":"raise_hand","confidence":0.96,"probs":{}}` |
| Raw → Motion | world landmarks → (25,3) → window | (1,30,84) tensor | NormalisedFrameCue | 4-key probs + scalars | `{"label":"sitting","probs":{sitting:1.0,...}}` |
| Raw → Context | ndarray uint8 BGR frame | (H,W,3) | NormalisedFrameCue | 2-key probs (smoothed) | `{"label":"classroom","probs":{classroom:0.998,kitchen:0.002}}` |
| Emotion → Aggregation | list of frame records | N frames | 9 floats | 7 probs + max_conf + valid_frac | `emotion_Neutral=0.993,...` |
| Gesture → Aggregation | list of frame records | N frames | 10 floats | 8 one-hot + mean_conf + valid_frac | `gesture_raise_hand=1.0,...` |
| Motion → Aggregation | list of frame records | N frames | 6 floats | 4 probs + max_conf + valid_frac | `motion_sitting=1.0,...` |
| Context → Aggregation | list of frame records | N frames | 4 floats | 2 probs + mean_conf + valid_frac | `context_classroom=0.996,...` |
| Aggregation → Parquet | 4 blocks + missing bits | 33 features | Parquet row | (clip_id + 33 + 6 meta) | 1270 rows total |
| Aggregation → Rule Fusion | pandas row | 33 features | 4 labels (or None) via `_dominant` | 4 categoricals | `(Neutral, raise_hand, sitting, classroom)` |
| Aggregation → GBT Builder | DataFrame slice | (n,33) | same floats (train: +modality dropout) | (n,33) | raw 33-vector |
| GBT Builder → GBT | ndarray | (n,33) | proba then argmax+override | (n,10)→(n,) | `proba (1,10)` |
| Rule Fusion → Final | 4 labels | — | intent string | scalar | `"F04"` |
| GBT → Final | (n,33) | — | intent string(s) | (n,) | `"F04"` (or forced `"F02"`) |

---

### Appendix — files that DON'T touch the fusion data path (avoid confusion)
- `pipeline/aggregate_clip_cues.py`, `pipeline/agreement_report.py`,
  `pipeline/measured/clip_cues.csv`, `reports/phase0_agreement.*` — Phase-0
  QA/agreement branch only.
- `pipeline/experiments/*` — ablation experiments (no-gate gesture, fps
  normalization); not in the production path.
- `Gesture Repo/{app.py,play_video.py,test_video.py,extract_dataset.py,train/*}`,
  `Emotion Repo/{video.py::main,realtime_realsense.py}`,
  `Motion Repo/example_webcam.py`, `Context Repo/.../{realtime.py,video.py::main}`
  — original per-repo GUI/demo/training scripts; the runners import only the
  model-construction/preprocess helpers from them, never their loops.
