# Gesture Model - Stage 4: Evaluation Report (TCN)

Generated: 2026-07-12 03:53 · checkpoint `best_TCN.pth`

## 1. Held-out test split (2415 sequences, subject-wise)

| Metric | Value |
|---|---|
| Accuracy | 91.39% |
| Balanced accuracy | 92.04% |
| Macro-F1 | 90.90% |
| Weighted-F1 | 91.40% |

### Per-class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| idle | 0.94 | 0.92 | 0.93 | 1293 |
| wave | 0.90 | 0.91 | 0.90 | 269 |
| point | 0.00 | 0.00 | 0.00 | 0 |
| thumbs_up | 0.87 | 0.88 | 0.87 | 266 |
| thumbs_down | 0.92 | 0.91 | 0.92 | 271 |
| beckoning | 0.87 | 0.90 | 0.88 | 289 |
| raise_hand | 0.90 | 1.00 | 0.95 | 27 |
| both_hands_up | 0.00 | 0.00 | 0.00 | 0 |

Confusion matrix: `confusion_test.png`

## 2. Test breakdown by source dataset

| Source dataset | Sequences | Accuracy | Macro-F1 |
|---|---|---|---|
| custom | 58 | 84.48% | 35.89% |
| jester | 2357 | 91.56% | 75.28% |

Macro-F1 here spans only the classes each dataset contributes.

## 3. Live test set

_No live_test sequences found — record them (custom/live_test/<label>/) and re-run extraction + prepare_data. This is the deployment metric (guide §10)._

## 4. Deployment budget

- Single-window forward pass (CPU, this machine): **0.95 ms** — the temporal net is negligible next to MediaPipe landmark extraction, as designed for the Jetson Orin Nano.
