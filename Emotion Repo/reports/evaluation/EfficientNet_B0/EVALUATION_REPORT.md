# Emotion Model - Stage 4: Evaluation Report (EfficientNet-B0)

- Checkpoint: `checkpoints/best_EfficientNet_B0.pth`
- Test images: 3068

## Overall metrics

| Metric | Value |
|---|---|
| Accuracy | 83.93% |
| Balanced accuracy | 79.10% |
| Macro-F1 | 76.41% |
| Weighted-F1 | 84.28% |
| Macro precision | 74.34% |
| Macro recall | 79.10% |

## Per-class metrics

| Emotion   |   Precision |   Recall |   F1-Score |   Support |
|:----------|------------:|---------:|-----------:|----------:|
| Surprise  |       79.67 |    89.36 |      84.24 |       329 |
| Fear      |       51.04 |    66.22 |      57.65 |        74 |
| Disgust   |       56.4  |    60.62 |      58.43 |       160 |
| Happy     |       98.08 |    86.41 |      91.88 |      1185 |
| Sad       |       78.36 |    87.87 |      82.84 |       478 |
| Anger     |       73.6  |    80.86 |      77.06 |       162 |
| Neutral   |       83.21 |    82.35 |      82.78 |       680 |


## Confidence

- Mean confidence on correct predictions: 0.676
- Mean confidence on incorrect predictions: 0.536
- A well-calibrated model is more confident when correct.

## Figures

- `confusion_matrix.png`
- `per_class_metrics.png`
- `confidence_distribution.png`

## Notes

- Minority emotions (Fear, Disgust, Anger) are the hardest; watch their recall in the per-class table and the off-diagonal mass in the confusion matrix (Fear is often confused with Surprise).