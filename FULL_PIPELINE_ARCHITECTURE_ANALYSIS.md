# Full Pipeline Architecture Analysis

> Forensic reconstruction of the HRI Fusion Engine from the actual
> implementation, **as of 2026-08-03**. Every claim below was traced through
> source code and, where possible, verified against real data artifacts
> (`pipeline/measured/*.jsonl`, `Data/Dataset/hri-multimodal-intent-v2.0.0/`,
> `tracking/*`) and by attempting to execute `fusion/rule_based.py`.
>
> This supersedes the prior version of this document, which described the
> repo before its 2026-07/08 restructure (dataset v1.0.0, an EfficientNet-B0
> context model, a heuristic TFLite gesture FSM, no MLflow). That version of
> the system no longer exists on disk — the old code paths were deleted, not
> kept side by side. This rewrite describes only the current system.
>
> Convention: **[VERIFIED]** = read directly from code or confirmed by
> running it / inspecting the artifact. **[INTERPRETATION]** = reasonable
> inference from verified facts. **NOT VERIFIABLE** = could not be confirmed
> from the present codebase.

---

## 0. The single most important structural fact

**This is still a batch, file-staged pipeline, not an online fusion engine —
and right now it does not run end-to-end.** [VERIFIED]

The architecture shape from before is unchanged: four independent cue-model
processes write per-frame JSONL, a feature builder aggregates per clip into a
Parquet file, and two fusion approaches (rule-based, LightGBM) read that
Parquet. What changed is that **all four cue models were substantially
reworked, the dataset moved from v1.0.0 to v2.0.0, and the fusion layer now
guards itself with MLflow dataset-consistency checks** — and the pipeline is
currently caught mid-transition:

```
pipeline/measured/*_frame_cues.jsonl   ← regenerated 2026-08-03, against
                                           dataset v2.0.0 and the new cue models
data/features/clip_features.parquet    ← still dated 2026-07-13, built from
                                           the OLD dataset v1.0.0 run
```

Running `fusion/rule_based.py` today raises `DatasetVersionMismatchError`
(§17) — MLflow's own lineage guard, added this session, is correctly
detecting that the Parquet fusion would read is stale relative to the
dataset version currently on disk. **No current accuracy numbers exist for
this system.** Regenerating the Parquet (`pipeline/build_features.py` against
the fresh JSONLs) is the next required step before fusion can run at all.

---

## 1. Executive Summary

**Dataset.** `Data/Dataset/hri-multimodal-intent-v2.0.0/`: 2,904 clips
(2,869 `usable`, 2,670 `scoreable`), 62 scenarios (`S01`…`S63`, gap at S30),
10 subjects (`P01`…`P10`), 2 contexts (classroom/kitchen), **9** intent codes
(`F01`–`F08`, `F10` — **F09 no longer exists**). This replaces v1.0.0 (1,270
clips, 22 scenarios, 23 subjects, 10 intents `F01`–`F10`), which has been
deleted from disk. Unlike v1, every scenario in v2 has multiple subjects
(2–10), so the old subject/scenario 1:1 confound is structurally resolved.
[VERIFIED, §5]

**The four cue models, current state:**

| Cue | Old (deleted) | Current | Status |
|---|---|---|---|
| Emotion | MobileNetV2 (RAF-DB only) | Same MobileNetV2 architecture, now loading the **fine-tuned** checkpoint (`finetuned_MobileNetV2.pth`, 92.5% real-test acc vs baseline's 58.8%) | Reworked (checkpoint swap + relocation), same model family |
| Gesture | Heuristic FSM over 2 TFLite classifiers, always empty `probs` | MediaPipe **Holistic** → 185-dim features → 32-frame window → learned **TCN** classifier (683K params), 8 classes incl. `idle` | Fully replaced |
| Motion | LSTM+attention, 4 classes | **Unchanged** — same checkpoint, same code | Stable (confirmed no diff since last commit) |
| Context | EfficientNet-B0 CNN, 2 classes | Zero-shot **CLIP** (ViT-B-32-quickgelu) via cosine similarity to text prompts, 2 classes in use | Fully replaced (a SmolVLM2 VLM also exists in the repo but is **not** used by this pipeline) |

**Fusion.** Both `fusion/rule_based.py` and `fusion/gbt.py` were rewritten:
the rule-based IF-THEN cascade was fully re-derived from v2.0.0's 62-row
`scenarios.csv` (mechanically, via a new `pipeline/derive_rule_table.py`,
not hand-transcribed), and `fusion/gbt.py`'s modality-dropout augmentation
now caps at 2 dropped cues per row and relabels heavily-degraded rows to
`F05`. Both scripts now log every run to a local **MLflow** tracking store
(`tracking/`, new this session) — params, metrics, per-class recall
figures, and a registered `pyfunc` model — and both refuse to run against a
Parquet whose content hash doesn't match the currently-active dataset
version. That guard is what's currently blocking them (§17, §0).

**Net effect for this document:** every per-cue trace below (§6–§9) is
independently verified against real, freshly-regenerated per-frame JSONL
output. The aggregation/fusion machinery (§11–§15) is verified by reading
the code, but **has not been exercised against current data yet** — no
"worked example, verified end to end" section exists in this version of the
document for that reason (contrast with the old doc's §14/§15).

---

## 2. What Changed Since the Last Snapshot (orientation)

For readers who saw the previous version of this document:

1. **Dataset v1.0.0 → v2.0.0.** New clip/scenario/subject counts, F09
   dropped, per-clip QA columns added (`usable`, `scoreable`,
   `exclude_kind`, `<cue>_masked`, `..._v3` relabeled ground truth). §5.
2. **Emotion Repo, Gesture Repo, Context Repo all restructured** into a
   uniform `config.py` / `src/` / `scripts/` / `inference/` / `checkpoints/`
   / `reports/` layout (mirroring each other). The old flat-file layouts
   (`video.py`, `app.py`, `realtime_realsense.py`, `Gesture Repo/model/*`,
   etc.) are deleted from git.
3. **Gesture model fully replaced**: heuristic FSM + 2 TFLite classifiers →
   learned TCN over MediaPipe Holistic features. §7.
4. **Context model backend switched**: trained EfficientNet-B0 CNN →
   zero-shot CLIP. A new SmolVLM2-based VLM situation-analysis path was also
   added to Context Repo but is not wired into the batch pipeline. §9.
5. **Emotion model unchanged architecturally**, but the batch runner now
   loads the fine-tuned checkpoint instead of the RAF-DB-only baseline. §6.
6. **Motion model untouched** — confirmed identical to the prior commit. §8.
7. **`pipeline/build_features.py` rewritten** for v2.0.0's schema: real
   per-clip cue masking from annotated ground truth, a leakage-checked dev
   split carved out of train. §12.
8. **`fusion/rule_based.py`'s rule cascade rewritten** end-to-end against
   the new, larger, machine-derived scenario table; both v1.0.0's documented
   "irreducible" ambiguities are reported resolved. §14.
9. **`fusion/gbt.py`'s modality-dropout logic reworked** with a capped
   per-row cue-drop count and a relabel-to-`F05` step. §15.
10. **MLflow tracking added** (`tracking/`) — the whole system, entirely new
    this session. §16.
11. `pipeline/aggregate.py` (the cue→feature-vector aggregation code) was
    **not touched** in any of this — it still encodes the old 33-dimension,
    8-class-gesture-one-hot schema from before. This mostly still applies
    cleanly (§11), but it's worth knowing this file is the one piece of the
    chain that wasn't revisited alongside everything around it.

---

## 3. Verified Runtime Entry Points

Still no unified "video → intent" driver. Six real entry points plus one new
supporting tool:

### Tier 1 — Cue runners (raw video → per-frame JSONL)

| # | File | Model it drives | Output |
|---|---|---|---|
| 1 | `runners/emotion_runner.py` | MobileNetV2 (fine-tuned ckpt), from `Emotion Repo/inference/video.py` | `emotion_frame_cues.jsonl` |
| 2 | `runners/gesture_runner.py` | `GestureEngine` (TCN), from `Gesture Repo/src/engine.py` | `gesture_frame_cues.jsonl` |
| 3 | `runners/motion_runner.py` | `MotionInference` (LSTM+attention), from `Motion Repo/inference.py` — **unchanged** | `motion_frame_cues.jsonl` |
| 4 | `runners/context_runner.py` | `create_scene_classifier(backend="clip")`, zero-shot CLIP | `context_frame_cues.jsonl` |

### Tier 2 — Feature builder

| # | File | Input | Output |
|---|---|---|---|
| 5 | `pipeline/build_features.py` | 4× `*_frame_cues.jsonl` + v2.0.0's `clips.csv`/`splits.csv` | `data/features/clip_features.parquet` |

`pipeline/build_splits.py` (previously a prerequisite) is now **largely
inert**: v2.0.0 ships its own pre-curated `splits.csv`, and the script's own
docstring says so (`pipeline/build_splits.py:4-8`). It's kept only for a
future dataset version that, like v1, ships with no split assignment.

New supporting tool: **`pipeline/derive_rule_table.py`** — not a pipeline
stage, an audit tool. Mechanically derives the `(context, emotion, gesture,
motion) → intent` table from `scenarios.csv` and asserts it's unambiguous;
run before hand-editing `fusion/rule_based.py`'s cascade. §13.

### Tier 3 — Fusion (Parquet → intent predictions + MLflow run)

| # | File | Entry | Currently runs? |
|---|---|---|---|
| 6a | `fusion/rule_based.py` | `__main__`, wrapped in `mlflow.start_run()` | **No** — `DatasetVersionMismatchError` (§17) |
| 6b | `fusion/gbt.py` | `main()`, wrapped in `mlflow.start_run()` | **No** — same guard, not independently confirmed but same code path |

Both still import each other (`gbt.py` imports `predict_all`/`fit_fallback`/
`predict_intent` from `rule_based.py`) so a GBT run also re-runs the rule
baseline for direct comparison — unchanged behavior from before. Both now
also log an `mlflow.pyfunc` model (`RuleBasedFusionModel`, `GBTFusionModel`)
to the local Model Registry on every successful run.

### Not entry points (diagnostic side-branch, unchanged in role)

`pipeline/aggregate_clip_cues.py` → `pipeline/agreement_report.py` remain a
Phase-0 diagnostic path (majority-vote label vs. authored scenario intent),
now repointed at v2.0.0 via `pipeline/dataset_config.py` but otherwise
untouched. Still does not feed fusion.

---

## 4. Complete End-to-End Call Graph

```text
# ── STAGE 1: four independent runner processes (one per venv) ──────────────

runners/emotion_runner.py  __main__
└── run_batch → process_clip(clip, model, transform, device, mp_face)
    ├── emotion_video.resolve_weights(DEFAULT_WEIGHTS="finetuned_MobileNetV2.pth")
    ├── MediaPipe FaceDetection(model_selection=1) — single pass, no CLAHE/tiling
    ├── pick_face (largest bbox) → crop full-res frame → PIL → transform(224×224)
    └── softmax(MobileNetV2(x)) → 7-way probs → NormalisedFrameCue

runners/gesture_runner.py  __main__
└── process_clip(clip, engine)                    # engine loaded once, reset() per clip
    ├── mp.solutions.holistic.Holistic (recreated per clip)
    ├── engine.process_holistic(res) → GestureEngine (Gesture Repo/src/engine.py)
    │   ├── landmarks_to_arrays → build_features() → 185-dim vector       (src/features.py)
    │   ├── deque(maxlen=64); once ≥32 buffered, resample to 32, run TCN
    │   ├── softmax → EMA(α=0.25) smooth → argmax → 0.60 conf-floor → idle
    │   └── 0.30s debounce before switching self.current
    │   returns (native_label, confidence) ONLY — smoothed probs computed but discarded
    ├── NATIVE_TO_CANONICAL = {"idle": "Unknown"}  (other 7 labels pass through)
    └── NormalisedFrameCue(label, confidence, probs={}, valid=conf≥0.80)

runners/motion_runner.py  __main__                # UNCHANGED — see §8
└── process_clip(clip, engine) → mediapipe_to_ntu25 → MotionInference.update
    → 30-frame window → MotionLSTM → softmax(4) → NormalisedFrameCue

runners/context_runner.py  __main__
└── process_clip(clip, model, transform, device)
    ├── create_scene_classifier(backend="clip", classes=["classroom","kitchen"])
    │   → ZeroShotSceneClassifier (CLIP ViT-B-32-quickgelu)
    ├── BGR→RGB → CLIP encode_image → cosine-sim ×100 vs. text-prompt embeddings
    ├── softmax over 2 active classes; internal deque(maxlen=15) smoothing
    └── NormalisedFrameCue(label, confidence=avg_conf, probs{classroom,kitchen})

# ── STAGE 2: feature builder (single process, .venvs/pipeline) ─────────────

pipeline/build_features.py  main()
├── clips_by_id = {clip_id: row} from v2.0.0 clips.csv
├── split_rows  = v2.0.0 splits.csv (already usable==TRUE-filtered, 2,869 rows)
├── dev_assignment = assign_dev_split(split_rows, clips_by_id)   # ~20% of train, grouped
│                                                                  by (scenario_dir, person_id, take_index)
├── frames_by_cue = {cue: load_frame_cues_by_clip(cue) for 4 cues}   # aggregate.py, UNCHANGED
└── for each split_row:
    ├── feat_row = build_clip_feature_row(...)      # aggregate.py, UNCHANGED — 33 dims
    ├── apply_cue_mask(feat_row, clip_row)           # zero+missing-bit per real *_masked flag
    ├── feat_row["scenario_dir"/"person_id"/"intent"/"split_design"/"split_design_v2"/"scoreable"] = ...
    └── rows.append(feat_row)
→ df.to_parquet(clip_features.parquet)     # NOT YET RUN against current JSONLs — see §17

# ── STAGE 3: fusion (single process, .venvs/pipeline, wrapped in mlflow run) ─

fusion/rule_based.py  __main__
├── init_tracking() → mlflow SQLite store + "fusion-engine-intent-classification" experiment
├── df = log_dataset(context="training")     # tracking/dataset_logging.py
│   └── check_dataset_consistency(...)       # RAISES DatasetVersionMismatchError today
├── fallback = fit_fallback(train_df)        # train-split intent mode
├── preds = predict_all(df, fallback)        # rewritten 8-branch cascade, §14
└── logs overall/scoreable/per-split/per-class-recall metrics + a pyfunc model

fusion/gbt.py  main()
├── same init_tracking()/log_dataset() gate
├── X_train, y_train = apply_modality_dropout(...)   # capped at 2 cues, relabels to F05
├── LGBMClassifier(...).fit(X_train, y_train)         # same hyperparams as before
├── rule_preds_all = predict_all(df, fallback)         # runs rule_based for comparison
├── predict_with_safety_override(...)                  # F02 ≥ 0.15 → force F02, unchanged
└── logs metrics/figures + a pyfunc model
```

---

## 5. Dataset — `hri-multimodal-intent-v2.0.0`

`Data/Dataset/hri-multimodal-intent-v1.0.0/` has been deleted from disk
(`git status` shows its 3 annotation CSVs as `deleted`; `raw/` was already
gitignored). It is replaced by `Data/Dataset/hri-multimodal-intent-v2.0.0/`
(untracked — this dataset version has not been committed yet). [VERIFIED]

| | v1.0.0 (deleted) | v2.0.0 (current) |
|---|---|---|
| Clips | 1,270 | 2,904 total (2,869 `usable`, 2,670 `scoreable`) |
| Scenarios | 22 (`S01`…`S29`, gaps) | 62 (`S01`…`S63`, gap at S30) |
| Subjects | 23 | 10 (`P01`…`P10`) |
| Intents | 10 (`F01`–`F10`) | **9** (`F01`–`F08`, `F10` — **F09 removed**) |
| Subject:scenario | 1:1 (confounded) | Many:many — every scenario has 2–10 subjects |
| Splits | none shipped (`build_splits.py` derives them) | `splits.csv` ships `split_design` (train/test: 1,866/1,003) pre-curated |

**`clips.csv` (39 columns)** adds real provenance/QA fields absent from v1:
`usable`, `exclude_kind`, `exclude_reason`, `scoreable`, `caveat`,
`derived_from_row`, `derived_from_clip_id`, `dup_of`, and per-cue relabeled
ground truth (`emotion_v3`, `gesture_v3`, `motion_v3`, `missing_v3`,
`gt_emotion`) plus real masking flags (`context_masked`, `emotion_masked`,
`gesture_masked`, `motion_masked` — motion is verified never masked in this
dataset). These indicate v2 is a **relabeled/reconciled merge** of prior
recordings ("v3 row"), not a fresh capture — `old_clip_id`/`old_filepath`/
`old_scenario_id` columns point back at the source rows. [VERIFIED]

**`scenarios.csv` (62 rows, 21 columns)** now carries the
`(context, emotion_v3, gesture_v3, motion_v3) → intent` mapping directly,
machine-readable, which is what makes `pipeline/derive_rule_table.py` (§13)
possible — v1's ~22-row table had to be read by eye.

`pipeline/dataset_config.py` is the new single source of truth for which
version folder the batch scripts target:
```python
ACTIVE_DATASET_VERSION = "hri-multimodal-intent-v2.0.0"
DATASET_ROOT = REPO_ROOT/Data/Dataset/{ACTIVE_DATASET_VERSION}
```
`pipeline/build_features.py`, `pipeline/build_splits.py`, and
`pipeline/agreement_report.py` were all repointed at this constant (each a
small, mechanical diff — no logic changes beyond the path source).
[VERIFIED]

---

## 6. Emotion Cue — Full Trace

Files: `runners/emotion_runner.py`, `Emotion Repo/inference/video.py`.
Checkpoint: `Emotion Repo/checkpoints/finetuned_MobileNetV2.pth`.

### 6.1 What changed vs. before

Architecturally, **nothing** — same MobileNetV2 backbone, same 7-class head,
same preprocessing shape. What changed: (a) the file moved from
`Emotion Repo/video.py` to `Emotion Repo/inference/video.py` as part of the
repo-wide restructure, and (b) **the runner now resolves and loads the
fine-tuned checkpoint by default**, not the RAF-DB-only baseline.
`emotion_video.DEFAULT_WEIGHTS = "finetuned_MobileNetV2.pth"`
(`video.py:36`); `resolve_weights()` (`video.py:59-72`) searches cwd →
script-dir → script-dir/checkpoints → script-dir/../checkpoints, resolving
to `Emotion Repo/checkpoints/finetuned_MobileNetV2.pth` (confirmed present,
9.18 MB). Per `Emotion Repo/README.md`, this checkpoint scores **92.5% acc /
90.1% macro-F1** on the true held-out test subjects, vs. 58.8%/38.9% for the
old baseline checkpoint the previous version of this pipeline effectively
used. [VERIFIED]

> **Footgun already documented in the repo, worth repeating here:**
> `Emotion Repo/config.py` also defines a `DEFAULT_CHECKPOINT` constant that
> currently agrees with `video.py`'s default — but it is a second,
> independent hardcoded value that `emotion_runner.py` never reads. The two
> "defaults" are only coincidentally in sync. `Emotion Repo/README.md`
> documents a real past incident where they drifted and produced a silent
> false "this model is bad" regression.

### 6.2 Preprocessing (as actually run by the batch runner)

Downscale to `MAX_FRAME_WIDTH=640` → BGR→RGB → single-pass MediaPipe
`FaceDetection(model_selection=1, min_detection_confidence=0.5)` → largest
bbox by area (`pick_face`) → crop from the **full-resolution** frame → PIL →
`Resize(224,224)` + `ToTensor` + ImageNet normalize.

`video.py` also defines a more robust `detect_face_box()` (plain pass + CLAHE
pass + 4 tiled quadrants, claimed ~100% vs. ~68% face-detection recall on
far-field footage) — but this is used only by `video.py`'s own interactive
demo loop, **not** by `emotion_runner.py`, which reimplements a plain
single-pass detector. The batch pipeline does not currently benefit from
that robustness improvement. [VERIFIED — flagged as a gap in §17]

### 6.3 Model, labels, confidence

MobileNetV2 (`weights=None`) with `classifier[1] = nn.Linear(1280, 7)`.
Labels: `["Surprise","Fear","Disgust","Happy","Sad","Anger","Neutral"]`.
`CONFIDENCE_FLOOR["emotion"] = 0.50` (`runners/common/constants.py:8`).

### 6.4 Real output (`pipeline/measured/emotion_frame_cues.jsonl`, line 1)

```json
{"cue":"emotion","frame_idx":0,"label":"Happy","confidence":0.598,
 "probs":{"Surprise":...,"Fear":...,"Disgust":...,"Happy":0.598,"Sad":...,"Anger":...,"Neutral":...},
 "valid":true,"extra":{"bbox":[345,131,30,30]},"clip_id":"S01_F01_c001"}
```
Schema is byte-for-byte the same `NormalisedFrameCue` shape as before —
`aggregate.py`'s emotion aggregation code needs no changes for this cue.

---

## 7. Gesture Cue — Full Trace

Files: `runners/gesture_runner.py`, `Gesture Repo/src/engine.py`,
`Gesture Repo/src/features.py`, `Gesture Repo/src/models.py`. Checkpoint:
`Gesture Repo/checkpoints/best_TCN.pth`. **Fully replaced this session.**

### 7.1 Input and feature construction

Per frame: MediaPipe **Holistic** (`model_complexity=1`, recreated fresh per
clip, `gesture_runner.py:91-92`) → pose (33 landmarks) + both hands (21
landmarks each) → `build_features()` (`Gesture Repo/src/features.py`)
produces a **185-dim** vector: pose ×(x,y,visibility)=99, dims,
mid-shoulder-origin / shoulder-width-scaled; left hand 21×(x,y)=42 + 1
presence flag = 43; right hand same = 43. `99+43+43=185`, wrist-relative /
wrist↔middle-MCP-scaled for each hand. [VERIFIED]

### 7.2 Temporal model — TCN, confirmed deployed

`Gesture Repo/config.py`: `DEFAULT_MODEL="TCN"`,
`DEFAULT_CHECKPOINT=checkpoints/best_TCN.pth`; confirmed by
`checkpoints/model_config.json` (`"model":"TCN"`, `hidden_size=128`,
`dropout=0.3`, `683,272` params). `TCNClassifier`
(`Gesture Repo/src/models.py`): `Conv1d` input projection (185→128) → 4
residual dilated blocks (kernel=5, dilations 1/2/4/8, each `Conv1d+BatchNorm
+ReLU+Dropout` ×2 with residual add) → global mean-pool over time →
`Dropout→Linear(128→8)`. Other checkpoints on disk
(`best_TCN_prev/pretune/pre_bhu.pth`) are superseded TCN variants of the
same architecture — no BiGRU/TinyTransformer checkpoint exists despite those
being mentioned as design alternatives in the README. [VERIFIED]

### 7.3 Windowing, smoothing, and the "idle" class

A `deque(maxlen=64)` buffers raw per-frame features (`engine.py`,
`ENGINE_BUFFER_FRAMES=64`). While fewer than 32 frames are buffered — the
first 31 frames of every clip — `process()` returns confidence `0.0`
verbatim (a buffering/warm-up state directly analogous to the Motion
model's own 29-frame warm-up, §8). Once ≥32 frames are available, the
buffer is uniformly resampled to exactly 32 frames and run through the TCN
in one forward pass. Output: softmax(8) → EMA smoothing (`α=0.25`) → argmax
→ if confidence `< 0.60` the candidate is forced to `"idle"` → a new label
must hold for `≥0.30s` (debounce) before it becomes the reported label.
`"idle"` is a genuine trained class (person/hand present, not gesturing) —
not a stand-in for "nothing detected"; no person at all (`pose_landmarks is
None`) also reports `("idle", 0.0)`, but is distinguished downstream purely
by the confidence-floor gate (§7.4), same mechanism as before. [VERIFIED]

### 7.4 Label mapping and confidence floor

`NATIVE_TO_CANONICAL = {"idle": "Unknown"}` (`gesture_runner.py:75`); the
other 7 native labels (`wave, point, thumbs_up, thumbs_down, beckoning,
raise_hand, both_hands_up`) already match `pipeline/aggregate.py`'s
`GESTURE_CLASSES` vocabulary exactly, so no other remapping happens.
`CONFIDENCE_FLOOR["gesture"] = 0.80` — **unchanged** from the old system's
strict floor. Empirically confirmed: a `"wave"` at confidence 0.746 is
`valid:false`; the same clip's frame at confidence 0.801 is `valid:true`.
Across the full 359,363-line file, ~27.3% of rows are `valid:true`.
[VERIFIED]

> **Note:** `runners/common/constants.py`'s `GESTURE_SCENARIO_TO_CANONICAL`
> table and its `"Idle"`/`"Wave"`/`"Brief wave"` string keys are **leftovers
> from the deleted heuristic FSM**. The current runner does not import or
> use it — it defines its own two-entry `NATIVE_TO_CANONICAL` inline. The
> old table is now consumed only by `pipeline/experiments/
> gesture_no_gate_experiment.py`, an ablation script, not the production
> path. `pipeline/canonical_map.py`'s comments referencing "gesture_runner.py's
> keypoint_classifier" are similarly stale prose, though the map itself
> (`GESTURE_MAP`) is current and includes `"idle": "Unknown"`.

### 7.5 The probs gap — a real, reportable finding

The TCN **does** compute a full 8-class softmax internally
(`engine.py`'s `smooth_probs`) — architecturally, this cue is now capable of
emitting a real probability distribution, unlike the old FSM which had no
distribution to emit at all. **However**, `GestureEngine.process()` /
`process_holistic()` only ever return `(label, confidence)` — the
probability vector is computed and then discarded, never exposed through the
public API. `gesture_runner.py:110` correspondingly hardcodes `probs={}` on
every record. Verified directly against the regenerated JSONL: every one of
359,363 lines has `"probs": {}`. **So despite the model upgrade, the pipeline
still cannot use gesture's probability distribution for anything** — the
plumbing to carry it through from `engine.py` to the JSONL was never built.
This has no effect on current aggregation (`_majority_onehot_features` only
reads `label`/`confidence`, never `probs`), but it blocks any future switch
of gesture's aggregation from majority-vote-one-hot to mean-probability. §17.

### 7.6 Real output (`pipeline/measured/gesture_frame_cues.jsonl`)

```json
{"cue":"gesture","frame_idx":0,"label":"Unknown","confidence":0.0,"probs":{},"valid":false,"extra":{"has_person":true},"clip_id":"S01_F01_c001"}
{"cue":"gesture","frame_idx":39,"label":"wave","confidence":0.7226,"probs":{},"valid":false,"extra":{"has_person":true},"clip_id":"S01_F01_c001"}
{"cue":"gesture","frame_idx":47,"label":"wave","confidence":0.8011,"probs":{},"valid":true,"extra":{"has_person":true},"clip_id":"S01_F01_c001"}
```

### 7.7 Old system fully removed

`git status` confirms `Gesture Repo/model/{keypoint_classifier,
point_history_classifier}/*` (`.tflite`, `.hdf5`, label CSVs), `app.py`,
`train/*.ipynb`, `utils/cvfpscalc.py` are all deleted. No `.tflite` or
`KeyPointClassifier`/`PointHistoryClassifier` reference remains anywhere in
`Gesture Repo/src/` or `runners/gesture_runner.py` outside one historical
mention in the runner's own docstring. [VERIFIED]

---

## 8. Motion Cue — Full Trace (unchanged)

Files: `runners/motion_runner.py`, `Motion Repo/inference.py`,
`Motion Repo/model.py`, `Motion Repo/skeleton_utils.py`. Checkpoint:
`Motion Repo/checkpoints/best_model_finetuned.pt`.

**Confirmed identical to the prior commit** — `git status`/`git diff` show
zero changes to `Motion Repo/` or `runners/motion_runner.py` beyond a
4-line path tweak. This section is carried forward unchanged.

MediaPipe `Pose` → 33 world landmarks → `mediapipe_to_ntu25()` → 25-joint
NTU layout → subset to 14 joints → hip-center/shoulder-width normalize →
flatten to a 42-dim position vector, paired with a 42-dim frame-to-frame
velocity vector → 84-dim frame feature → 30-frame sliding window →
`MotionLSTM`: `LayerNorm(84)` → `LSTM(84→256, 3 layers, dropout 0.35)` →
temporal attention over the 30 timesteps → `Linear(256,64)→ReLU→Dropout→
Linear(64,4)`. **Confirmed 4 classes** (`sitting, standing, walking,
stepping_back`) via checkpoint's classifier weight shape `(4,64)` and
`Motion Repo/model.py`'s `num_classes=4` / `inference.py`'s `NUM_CLASSES=4`
— stale "(6,)"/"shape (6,)" comments remain in `inference.py:59,203`
(cosmetic only, no runtime effect). The first 29 frames of every clip are a
`"buffering"` state (`valid=False`). `CONFIDENCE_FLOOR["motion"] = 0.50`.
[VERIFIED]

Output schema unchanged: `{"cue":"motion","label":"sitting"|...,"confidence":
float,"probs":{4 keys},"valid":bool,"extra":{"buffering","has_landmarks"}}`.

---

## 9. Context Cue — Full Trace

Files: `runners/context_runner.py`, `Context Repo/scene_classification/
src/classifier.py`, `.../src/zero_shot.py`. **Backend fully replaced this
session** (EfficientNet-B0 CNN → zero-shot CLIP).

### 9.1 Backend selection

`context_runner.py` calls `create_scene_classifier(backend="clip",
classes=["classroom","kitchen"])`. `create_scene_classifier()`
(`classifier.py`) defaults to `SCENE_BACKEND="clip"`
(`scene_classification/config.py`), and the runner also hardcodes
`backend="clip"` explicitly — the CNN path (`SceneClassifier`) exists in the
same module but is never instantiated by the production runner. This
always builds a `ZeroShotSceneClassifier`. [VERIFIED]

> **Sys.path oddity worth flagging:** `context_runner.py` inserts a path
> under `/media/.../KINGSTON_KG/hri-jetson/modalities/context/
> scene_classification` — an **externally mounted drive**, not the in-repo
> `Context Repo/scene_classification/`. The two copies were byte-diffed and
> are currently identical, so behavior matches what's described here, but
> the production runner does not actually import from this repo's own
> tracked source tree. If the two copies ever drift, or the drive isn't
> mounted, the runner will either silently run stale code or fail to import.

### 9.2 Preprocessing and prediction

BGR→RGB → PIL → CLIP `ViT-B-32-quickgelu` (openai pretrained weights)
`preprocess`/`encode_image`, L2-normalized → cosine similarity ×100 against
pre-embedded text prompts (6 prompts/class, averaged) → softmax over the 2
active classes. An `ABSTAIN_PROMPTS` probe can flag "uncertain" if abstain
probability crosses a threshold. `config.py` actually defines 5 scene
classes (`+hospital, cloth_store, museum`) for broader deployment, but this
pipeline restricts to the 2 it needs. [VERIFIED]

### 9.3 Temporal smoothing moved into the classifier

The 15-frame rolling-mean smoothing that previously lived in the runner now
lives **inside** `ZeroShotSceneClassifier` (`deque(maxlen=15)`,
`smooth_window` param). The runner's role is reduced to calling
`classifier.reset()` once per clip. Net behavior is unchanged from before —
context's per-frame `confidence`/`probs` are still temporally smoothed,
unlike emotion/motion. [VERIFIED]

### 9.4 The VLM path exists but is not used here

`Context Repo/README.md` describes a second component — SmolVLM2-500M-based
"VLM situation analysis" producing people-count/activity/attention/objects/
summary — as part of a combined `ContextState`. Grepping
`runners/context_runner.py` and the entire `scene_classification/src/` tree
for any reference to the VLM or `Context Repo/src/vlm.py`/`pipeline.py`
returns nothing. **The fusion-facing batch pipeline uses only the CLIP scene
classifier; the VLM is real, working code elsewhere in the repo, but dead
code from this pipeline's perspective.** [VERIFIED]

### 9.5 Confidence floor and real output

`CONFIDENCE_FLOOR["context"] = 0.50`, combined with `label != "Unknown"`.

```json
{"cue":"context","frame_idx":0,"label":"classroom","confidence":0.9995,
 "probs":{"classroom":0.9995,"kitchen":0.0005},"valid":true,
 "extra":{"activity":null,"engaged":null,"n_objects":0},"clip_id":"S01_F01_c001"}
```
Schema unchanged from before — still a 2-key `probs` dict, still constant
`extra` placeholders that carry no signal.

---

## 10. Cue Output Comparison (current)

| Cue | Model | `label` vocab | `probs` | `valid` gate | Changed this session? |
|---|---|---|---|---|---|
| Emotion | MobileNetV2 (fine-tuned) | 7 emotions or `Unknown` | 7-key dict, real | `conf≥0.50` | Checkpoint swap only |
| Gesture | TCN (185-dim in, 32-frame window) | 8-way canonical incl. `Unknown`(idle) | **always `{}`** despite a real distribution existing internally | `label≠Unknown ∧ conf≥0.80` | Fully replaced |
| Motion | LSTM+attention (unchanged) | 4 motions or `Unknown` | 4-key dict, real | `conf≥0.50 ∧ has_landmarks` | Unchanged |
| Context | Zero-shot CLIP | `classroom`/`kitchen`/`Unknown` | 2-key dict, **smoothed inside classifier** | `conf≥0.50 ∧ label≠Unknown` | Backend fully replaced |

The structural asymmetry noted in the prior version of this document still
holds and is, if anything, sharper now: gesture is the only cue whose model
*could* emit a real distribution but doesn't, because the plumbing (§7.5)
was never finished.

---

## 11. Aggregation Layer — `pipeline/aggregate.py` (unchanged)

**Not touched in this restructure.** `git diff`/`git log` confirm no changes
since the "Implement Rule-based & LightGBM models" commit. It still assumes
exactly the schema every cue runner still produces (§6–§9 confirm this holds
today), so it did not need to change — but it also means this file has not
been revisited to account for anything new (e.g. it has no awareness that
gesture *could* now supply real probs, §7.5).

`FEATURE_NAMES` — **33 columns, unchanged**:

```
emotion (9):  7 mean-probs + max_confidence + valid_fraction
gesture (10): 8 one-hot majority label + mean_confidence + valid_fraction
motion  (6):  4 mean-probs + max_confidence + valid_fraction
context (4):  2 mean-probs + mean_confidence + valid_fraction   [see doc-bug note below]
missing (4):  missing_emotion, missing_gesture, missing_motion, missing_context
```

`GESTURE_CLASSES = ["wave","point","thumbs_up","thumbs_down","raise_hand",
"both_hands_up","beckoning","Unknown"]` — this still matches what
`gesture_runner.py` actually emits after the `idle→Unknown` remap (§7.4), so
there is **no live schema mismatch** despite the model swap.

Two aggregation algorithms, unchanged: **mean-probability** for
emotion/motion/context (`_prob_mean_features`), **majority-vote one-hot**
for gesture only (`_majority_onehot_features`, reads `label`/`confidence`,
never `probs` — which is exactly why §7.5's gap hasn't broken anything yet).
`missing_<cue> = valid_fraction < 0.40`.

> **Documentation bug, still present, unchanged from the prior version of
> this doc:** the module header (line ~39) describes context as "2 one-hot
> scene." It is the mean-probability vector, same as emotion/motion/context's
> other three blocks — only gesture is truly one-hot. Cosmetic, no runtime
> effect (rule-based fusion's `argmax` works on mean-probs identically).

---

## 12. Feature Builder — `pipeline/build_features.py` (rewritten)

Rewritten (~160-line diff) for v2.0.0's schema. Key differences from the
version this file replaces:

- **`intent` is a direct column** on `clips.csv`/`splits.csv` now — no more
  join through `scenarios.csv` via a `scenario_id.split("_")[0]` hack.
- **`splits.csv` is the authoritative clip list to iterate**, already
  filtered to `usable==TRUE` (verified: exact set match against
  `clips.csv`), not `clips.csv` itself.
- **Real per-clip cue masking** (`apply_cue_mask()`): `clips.csv`'s
  `context_masked`/`emotion_masked`/`gesture_masked` flags (motion is never
  masked, verified) mark clips where the *authored* ground truth assumed the
  fusion model couldn't see that cue — even though the raw video still shows
  a real face/background/gesture and the runner would otherwise emit a real
  (leaky) measurement. This function zeroes that cue's block and sets its
  missing bit, deterministically, from real annotation — the same shape as
  `fusion/gbt.py`'s *simulated* dropout (§15), but grounded in fact rather
  than a random draw.
- **A dev split is carved from train** (`assign_dev_split()`): `splits.csv`
  only ships a 2-way `split_design` (train/test). ~20% of each scenario's
  *video groups* — grouped by `(scenario_dir, person_id, take_index)`, since
  one video can yield up to 3 clips — become `dev`, via a seeded RNG, with
  an explicit leakage assertion that no group straddles train/dev.
- **New output columns**: `scenario_dir`, `person_id`, `intent`,
  `split_design` (as shipped), `split_design_v2` (train/dev/test, with the
  carved-out dev), `scoreable`.
- Reads `DATASET_ROOT` from `pipeline/dataset_config.py` (§5) instead of a
  hardcoded path.

This script has **not been rerun** against the freshly-regenerated JSONLs
yet (§0, §17) — everything above describes what it will do the next time it
runs, not what's currently in `data/features/clip_features.parquet`.

---

## 13. Rule Table Derivation — `pipeline/derive_rule_table.py` (new)

Not part of the runtime pipeline — an **audit tool**. Reads v2.0.0's
`scenarios.csv` (62 rows), normalizes each row's `(context, emotion_v3,
gesture_v3, motion_v3)` via `pipeline/canonical_map.py`'s `map_intended()`
(the same normalization `agreement_report.py` already used), groups by that
4-tuple, and **asserts** every group maps to exactly one `intent` — raising
loudly if the dataset is ever revised into genuine ambiguity, rather than
letting `fusion/rule_based.py`'s hand-maintained cascade go silently stale.
`main()` prints the table grouped by gesture as a human-readable reference
for updating `rule_based.py` by hand. This is the tool that produced the
rewritten cascade in §14. [VERIFIED]

---

## 14. Rule-Based Fusion — `fusion/rule_based.py` (rewritten)

The IF-THEN cascade itself was rewritten, not just its surrounding
plumbing — re-derived from v2.0.0's 62-row table via §13's tool. Per the
module's own docstring, **both of v1's documented "irreducible" label
ambiguities are now reported resolved**:

- **F02 vs. F07** (`both_hands_up`) is now emotion-dependent, not
  scene-dependent: Anger → F07 (frustration, not danger), Neutral → F05,
  everything else (Fear/Surprise/unobserved) → F02 (emergency, asymmetric
  cost).
- **F04 vs. F10** is resolved by idle vs. active gesture: Sad + `idle`
  (`gesture=="Unknown"`, a person present but not gesturing) → F10
  (discouraged, no directed signal); Sad + an active gesture
  (`thumbs_down`/`beckoning`) → F04.
- **F09 no longer exists** as an intent class in v2 — folded into F01;
  `wave` is now purely emotion-dispatched (`Anger`→F06, else→F01).

The cascade (`predict_intent()`, verbatim, `fusion/rule_based.py:62-148`) is
grouped by gesture, most-emergency-relevant first:

```text
1. gesture == both_hands_up:
     Anger→F07, Neutral→F05, else→F02
2. gesture == Unknown (idle):
     Sad→F10, Fear→F02, Neutral→F05
3. gesture in {thumbs_down, thumbs_up}:
     Sad→F04, Anger→F07, Disgust→F08, Happy→F01
4. gesture == raise_hand:
     Happy→(F01 if context==kitchen else F05); Neutral/missing→F04
5. gesture == beckoning:
     Neutral→F03, Sad→F04
6. gesture == point:
     Anger→(F06 if walking else F07 if standing); Disgust→F06; Happy/Neutral→F03
7. gesture == wave:
     Anger→F06, else→F01
8. gesture is None (genuinely missing, not idle):
     Fear→F02; Sad+sitting→F04; Neutral/missing+walking→F06
→ else: fallback_intent (train-split intent mode)
```
First match wins per branch group; anything the 62-row table never showed
falls through to the fallback, unchanged in spirit from before.

**Other changes**: split-column references renamed `split_scenario`→
`split_design_v2`, `val`→`dev` throughout; new `scoreable`-only accuracy
metrics logged alongside all-clips/per-split accuracy; a new
`RuleBasedFusionModel(mlflow.pyfunc.PythonModel)` wraps `predict_intent()`
so the rule baseline is directly comparable/servable via the same interface
as the GBT model. Everything is now logged to MLflow (§16) rather than just
printed to stdout — params, per-split/per-class-recall metrics, a bar-chart
figure, and a registered model (`fusion-rule-based`).

---

## 15. GBT Fusion — `fusion/gbt.py` (reworked dropout + mlflow)

Feature input, encoding, and core hyperparameters are **unchanged**:
`FEATURE_NAMES` (33-dim, §11) straight into `LGBMClassifier(objective=
"multiclass", class_weight="balanced", n_estimators=300, max_depth=5,
learning_rate=0.05, random_state=42)`, with the same inference-only F02
safety override (`P(F02)≥0.15 → force F02`).

**What changed is `apply_modality_dropout()`:**

- Previously: independent, vectorized per-cue Bernoulli(0.15) drop across
  the whole training matrix.
- Now: **per-row loop**, drawing each of the 4 cues independently at
  `DROPOUT_P=0.15`, but **capped at `MAX_DROPPED_CUES=2`** per row (matching
  the dataset's own design doc: "mask each cue with p~0.15, max 1–2 cues per
  sample"). When ≥2 cues are dropped from a row **and** running
  `rule_based.predict_intent()` on the now-degraded row no longer recovers
  the row's true label, the row is **relabeled** to `RELABEL_INTENT="F05"` —
  teaching the model to fall back safely rather than confidently guess wrong
  on heavily-degraded input, instead of preserving a label that isn't
  actually recoverable from what remains. A sentinel fallback string
  (`"__NO_RULE_MATCH__"`) is used during this check so "no rule matched"
  can't spuriously register as a correct match.
- Explicitly **not** implemented: the dataset docx's carve-out that
  direction/point_target cues could sometimes still identify F02/F06 even
  under dropout — `pipeline/aggregate.py` never included those as features
  in the first place (they're hardcoded constants upstream, §11's
  inheritance from the prior schema), so there's no such signal to exempt.
  Documented as a known simplification in the code, not a silent omission.

Everything else — `GBTFusionModel` pyfunc wrapper, per-split/per-class-recall
MLflow logging, feature-importance bar charts, comparison against
`rule_based`'s predictions on the same clips — is new plumbing around the
same modeling core.

---

## 16. MLflow Tracking Layer — `tracking/` (new)

Entirely new this session; not present in the prior version of this repo.

- **`tracking/mlflow_setup.py`**: `init_tracking()` points every training/
  eval script at one shared local store: `TRACKING_URI =
  sqlite:///{REPO_ROOT}/mlflow.db`, `ARTIFACT_ROOT = {REPO_ROOT}/mlartifacts`,
  experiment `"fusion-engine-intent-classification"`. Both files/dirs exist
  on disk (`mlflow.db` 884 KB; `mlartifacts/` has 2 run-artifact dirs + a
  `models/` registry with 2 registered model versions) — confirming this has
  already been exercised, not just wired up (from an earlier pass, before
  the dataset-version guard below existed).
- **`tracking/hashing.py`**: one helper, `sha256_file()`.
- **`tracking/dataset_logging.py`** — the interesting one:
  - `dataset_version_tag()` auto-discovers the single folder under
    `Data/Dataset/*` (raises if 0 or >1 — now safe now that v1.0.0 is
    deleted, would have raised otherwise).
  - `log_dataset(context="training")` reads `clip_features.parquet`, logs it
    as an `mlflow.data` Dataset input, and tags the run with dataset
    version, a content sha256 of the Parquet, sha256s of the 3 annotation
    CSVs, and the git commit/dirty-state of the annotations directory.
  - `check_dataset_consistency()` — the guard — compares the current
    Parquet's content hash against **every prior run's tags** in this
    experiment and raises `DatasetVersionMismatchError` if either: (1) the
    same hash was previously logged under a *different* `dataset_version`
    tag (a **stale rebuild** — the dataset folder was swapped but
    `build_features.py` was never rerun), or (2) the same version label was
    previously logged with a *different* hash (**version reuse** — the
    label no longer maps to one fixed dataset). This is exactly the
    situation described in §0/§17. An `allow_stale_override=True` escape
    hatch exists (warns instead of raising) but neither fusion script
    currently uses it.

---

## 17. Current Pipeline Status — Why It Doesn't Run Right Now

This is the most important "current status" fact in this document.

**Running `.venvs/pipeline/bin/python fusion/rule_based.py` today fails**,
by design:

```
tracking.dataset_logging.DatasetVersionMismatchError: dataset_version is now 'hri-multimodal-intent-v2.0.0',
but clip_features.parquet's content hash is IDENTICAL to run fecb7dc3222949908f064761a3052c34 logged under
dataset_version='hri-multimodal-intent-v1.0.0'.
```

This is exactly case (1) from §16: `Data/Dataset/` now points at v2.0.0, but
`data/features/clip_features.parquet` (dated 2026-07-13) was never rebuilt
from it — it's still the file produced against the old v1.0.0 run, and
MLflow's own history proves it byte-for-byte. **Even bypassing this guard**
would fail immediately after: the stale Parquet's actual schema (checked
directly — 1,270 rows × 40 cols) has `split_scenario`, `split_subject`,
`scenario_id`, `subject_id` and **no** `split_design_v2` or `scoreable`
column, both of which `rule_based.py`/`gbt.py` now require unconditionally —
a `KeyError` would follow.

**What's needed to unblock the pipeline:**
1. Confirm all four `pipeline/measured/*_frame_cues.jsonl` are complete for
   the full 2,869-clip v2.0.0 manifest (they were last regenerated
   2026-08-03; run logs are at `pipeline/measured/*_run.log`).
2. Run `pipeline/build_features.py` to rebuild `clip_features.parquet`
   against those JSONLs and v2.0.0's `clips.csv`/`splits.csv` (§12).
3. Then `fusion/rule_based.py` and `fusion/gbt.py` can run — MLflow will log
   the first post-v2.0.0 run for this experiment (no prior v2.0.0-tagged
   run exists yet, so `check_dataset_consistency()` will pass cleanly).

No accuracy numbers exist for this system's current state, and none are
reported in this document — reporting the stale v1.0.0-era numbers here
would be actively misleading given the dataset, all three non-motion cue
models, and the rule cascade have all changed since they were measured.

---

## 18. Final Data Contract Summary

The 33 cue-derived feature columns (`pipeline/aggregate.py::FEATURE_NAMES`,
unchanged, §11) plus the new join/metadata columns
`pipeline/build_features.py` will attach once rerun (§12):

```
idx  column                      source / meaning
0-6  emotion_{7 classes}         mean prob (unchanged)
7    emotion_max_confidence      max conf over valid frames
8    emotion_valid_fraction      valid/total
9-16 gesture_{8 classes}         majority one-hot (unchanged; "idle"→"Unknown")
17   gesture_mean_confidence     mean conf of winning label
18   gesture_valid_fraction      valid/total
19-22 motion_{4 classes}         mean prob (unchanged, model unchanged)
23   motion_max_confidence       max conf over valid frames
24   motion_valid_fraction       valid/total
25-26 context_{classroom,kitchen} mean prob (now from CLIP, not CNN)
27   context_mean_confidence     mean conf over valid frames
28   context_valid_fraction      valid/total
29-32 missing_{emotion,gesture,motion,context}   valid_fraction < 0.40

—— join/metadata columns (build_features.py, v2.0.0 schema) ——
scenario_dir       v2.0.0 scenario id, e.g. "S01_F01"
person_id          P01…P10
intent             target label, F01–F08/F10 (F09 no longer exists)
split_design       as shipped by splits.csv: train/test
split_design_v2    train/dev/test (dev carved from train, leakage-checked)
scoreable          clips.csv's scoreable flag (whether this clip's ground
                   truth is trustworthy enough to count toward eval metrics)
```

Rule-based fusion still collapses each cue block back to one categorical
label via `argmax` (`_dominant()`, unchanged logic); GBT still consumes the
33 raw floats with no re-encoding. Both mechanisms are unchanged from the
prior version of this document — only the cascade evaluated on top of the
rule-based labels changed (§14).

---

## 19. Critical Findings and Known Gaps (current)

Ordered roughly by impact.

### G-1  Pipeline cannot currently run end-to-end
Stale `clip_features.parquet` (v1.0.0-era) vs. fresh per-cue JSONLs and an
active v2.0.0 dataset. MLflow's own consistency guard catches this and
raises `DatasetVersionMismatchError`; bypassing it would then `KeyError` on
missing `split_design_v2`/`scoreable` columns. See §17 for the fix. **This
is not a bug — the guard is working as designed** — but it means no accuracy
claim about this system is currently possible.

### G-2  Gesture's TCN computes real probs; the pipeline still can't see them
`GestureEngine` computes a full 8-class softmax internally but only returns
`(label, confidence)`; `gesture_runner.py` hardcodes `probs={}`. Verified
against all 359,363 lines of the regenerated JSONL. Blocks any future switch
of gesture's clip-level aggregation from majority-vote-one-hot to
mean-probability, and means GBT can never see gesture's actual confidence
distribution, only its winning label + scalar confidence + one-hot. §7.5.

### G-3  Context runner imports from an external mounted drive, not this repo
`context_runner.py`'s `sys.path` points at `/media/.../KINGSTON_KG/hri-jetson/
modalities/context/scene_classification`, not `Context Repo/
scene_classification/` in this working tree. Currently byte-identical
(verified), but this is a two-copies-can-drift risk, and the pipeline
silently depends on that drive being mounted. §9.1.

### G-4  Emotion's batch runner doesn't use the more robust face detector
`video.py` defines a CLAHE+tiled-quadrant `detect_face_box()` claimed to
substantially improve far-field detection recall, but only the interactive
demo loop uses it — `emotion_runner.py` reimplements a plain single-pass
detector. §6.2.

### G-5  Context aggregation still documented as "one-hot" (cosmetic, unchanged)
`pipeline/aggregate.py`'s header docstring still says context is "2 one-hot
scene." It's the mean-probability vector, same as before this session — this
bug predates the restructure and was never touched. §11.

### G-6  SmolVLM2 VLM path is real but unused by this pipeline
`Context Repo`'s `src/vlm.py`/`pipeline.py` implement a working VLM
situation-analysis path (per its README) that the fusion-facing batch
pipeline never calls. Not a bug, but worth knowing it exists separately from
what actually feeds fusion. §9.4.

### G-7  Top-level docs predate this restructure and should not be trusted for paths/numbers
`HRI_Fusion_Engine_Handover.md`, `MODEL_ANALYSIS.md`, `Integration_API.md`,
and `GPU_MACHINE_SETUP.md` were all last touched before the dataset-v2 /
mlflow / cue-model rework. `GPU_MACHINE_SETUP.md` in particular hardcodes
`Data/Dataset/hri-multimodal-intent-v1.0.0` paths that no longer exist on
disk — its own verification steps would fail outright if followed today.
`MODEL_ANALYSIS.md`/`Integration_API.md` describe the deleted flat-file repo
layouts. Treat all four as historical background, not current reference.

### G-8  F09 intent class removed between dataset versions
v1.0.0 had 10 intents (`F01`–`F10`); v2.0.0 has 9 (`F09` absent, folded into
`F01` per `rule_based.py`'s docstring). Anything downstream that assumes 10
classes (dashboards, prior analysis, external docs) needs updating. §5, §14.

---

## Appendix — files that don't touch the fusion data path

- `pipeline/aggregate_clip_cues.py`, `pipeline/agreement_report.py`,
  `pipeline/measured/clip_cues.csv`, `reports/phase0_agreement.*` — Phase-0
  QA/agreement branch only, now repointed at v2.0.0 but otherwise unchanged
  in role.
- `pipeline/experiments/*` — ablation experiments (no-gate gesture, fps
  normalization, MediaPipe version comparison); not in the production path.
  `gesture_no_gate_experiment.py` is the only current consumer of the
  now-otherwise-dead `GESTURE_SCENARIO_TO_CANONICAL` table (§7.4).
- `Context Repo/src/vlm.py`, `Context Repo/src/pipeline.py`,
  `Context Repo/inference/realtime.py` — the VLM (SmolVLM2) path and
  interactive demo entry points; not called by `runners/context_runner.py`.
- `Emotion Repo/scripts/*`, `Emotion Repo/src/models_lstm.py` — training/
  eval scripts and an explicitly do-not-deploy LSTM variant (per
  `Emotion Repo/README.md`); the batch runner imports only
  `inference/video.py`.
- `Gesture Repo/scripts/*`, `Gesture Repo/checkpoints/best_TCN_{prev,
  pretune,pre_bhu}.pth` — training pipeline and superseded checkpoints.
- `*/reports/*` (per-repo evaluation reports) — several are explicitly
  flagged stale by their own repos' READMEs (e.g. `Gesture Repo/reports/
  evaluation/TCN/EVALUATION_REPORT.md` predates the currently-deployed
  checkpoint) — re-run each repo's own `scripts/evaluate.py` before trusting
  numbers in these files rather than relying on what's already written.
- `pipeline/measured/*.old_v1`, `*.old_8class` — prior runs' JSONL/log/CSV
  output, kept for comparison, not read by any current script.
