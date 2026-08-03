# Emotion Model - Stage 4: Evaluation Report (MobileNetV2)

- Checkpoint: `checkpoints\best_MobileNetV2.pth`
- Test images: 3068

## Overall metrics

| Metric | Value |
|---|---|
| Accuracy | 84.32% |
| Balanced accuracy | 75.43% |
| Macro-F1 | 76.72% |
| Weighted-F1 | 84.06% |
| Macro precision | 78.68% |
| Macro recall | 75.43% |

## Per-class metrics

| Emotion   |   Precision |   Recall |   F1-Score |   Support |
|:----------|------------:|---------:|-----------:|----------:|
| Surprise  |       84.55 |    84.8  |      84.67 |       329 |
| Fear      |       65.75 |    64.86 |      65.31 |        74 |
| Disgust   |       68.47 |    47.5  |      56.09 |       160 |
| Happy     |       92.17 |    94.35 |      93.24 |      1185 |
| Sad       |       77.43 |    83.26 |      80.24 |       478 |
| Anger     |       81.82 |    72.22 |      76.72 |       162 |
| Neutral   |       80.56 |    81.03 |      80.79 |       680 |


## Confidence

- Mean confidence on correct predictions: 0.962
- Mean confidence on incorrect predictions: 0.816
- A well-calibrated model is more confident when correct.

## Figures

- `confusion_matrix.png`
- `per_class_metrics.png`
- `confidence_distribution.png`

## Notes

- Minority emotions (Fear, Disgust, Anger) are the hardest; watch their recall in the per-class table and the off-diagonal mass in the confusion matrix (Fear is often confused with Surprise).