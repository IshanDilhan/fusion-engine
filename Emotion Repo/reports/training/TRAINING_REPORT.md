# Emotion Model - Stage 2: Training Report (EfficientNet-B0)

Trained with the balanced two-stage recipe (weighted CrossEntropy + label smoothing + mixup + cosine LR with warmup + AMP).

## Configuration

- Model: EfficientNet-B0 (15.5 MB)
- Epochs: 5 head-only + 25 full fine-tune
- Batch size: 32
- Head LR: 0.001 | Base LR: 0.0002 | Weight decay: 1e-05
- Label smoothing: 0.1 | Mixup alpha: 0.2
- Optimizer: adam

## Best validation metrics

| Metric | Value |
|---|---|
| Accuracy | 83.87% |
| Balanced accuracy | 78.88% |
| Macro-F1 | 76.19% |
| Weighted-F1 | 84.23% |

- Checkpoint: `checkpoints/best_EfficientNet_B0.pth`
- Training curves: `training_curves.png`

## Next step

```
python scripts/evaluate.py --model "EfficientNet-B0"
```