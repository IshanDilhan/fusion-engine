# Gesture Model - Stage 2: Training Report (TCN)

Generated: 2026-07-12 03:52. Recipe: class-weighted CrossEntropy + label smoothing, AdamW, cosine LR with warmup, early stopping on val macro-F1.

## Configuration

- Model: TCN (2.61 MB)
- Epochs: 100 (patience 15) | Batch size: 256
- LR: 0.001 | Weight decay: 0.0001 | Dropout: 0.3
- Hidden size: 128 | Label smoothing: 0.1
- Window: 32 frames | Feature dim: 185

## Best validation metrics

| Metric | Value |
|---|---|
| Accuracy | 92.78% |
| Balanced accuracy | 92.93% |
| Macro-F1 | 92.38% |
| Weighted-F1 | 92.78% |
| Best epoch | 53 |

- Checkpoint: `checkpoints/best_TCN.pth`
- Training curves: `training_curves.png`

## Next step

```
python scripts/evaluate.py --model TCN
```