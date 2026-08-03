# Gesture Model - Stage 1: Architecture Comparison

Generated: 2026-07-12 03:48 · shortened schedule (40 epochs max, patience 8), shared recipe (class-weighted CE + label smoothing, AdamW, cosine warmup).

| model           |   params_k |   size_mb |   train_min |   accuracy |   balanced_accuracy |   f1_weighted |   f1_macro |
|:----------------|-----------:|----------:|------------:|-----------:|--------------------:|--------------:|-----------:|
| TCN             |      683.3 |      2.61 |         2.3 |     0.9278 |              0.9338 |        0.928  |     0.9272 |
| BiGRU           |      606.1 |      2.31 |         2.4 |     0.9226 |              0.9307 |        0.9228 |     0.9189 |
| TinyTransformer |      294.4 |      1.12 |         1.7 |     0.9027 |              0.925  |        0.9036 |     0.9087 |

**Winner (val macro-F1): TCN**

- Curves: `comparison_curves.png`

## Next step

```
python scripts/train.py --model TCN
```