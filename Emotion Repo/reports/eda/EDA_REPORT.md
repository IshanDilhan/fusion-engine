# Emotion Model - EDA Report (RAF-DB)

## 1. Dataset overview

- Classes: **7** (Surprise, Fear, Disgust, Happy, Sad, Anger, Neutral)
- Train images: **12271**
- Test images: **3068**
- Layout: `ImageFolder` (folders 1..7, alphabetically sorted -> label index 0..6)
- Source image size: all sampled images are 100x100 px

## 2. Class distribution

|   index | emotion   |   train |   test |   train_pct |   class_weight |
|--------:|:----------|--------:|-------:|------------:|---------------:|
|       0 | Surprise  |    1290 |    329 |    10.5126  |         0.6572 |
|       1 | Fear      |     281 |     74 |     2.28995 |         3.0168 |
|       2 | Disgust   |     717 |    160 |     5.84304 |         1.1823 |
|       3 | Happy     |    4772 |   1185 |    38.8884  |         0.1776 |
|       4 | Sad       |    1982 |    478 |    16.1519  |         0.4277 |
|       5 | Anger     |     705 |    162 |     5.74525 |         1.2025 |
|       6 | Neutral   |    2524 |    680 |    20.5688  |         0.3359 |


- **Imbalance ratio (max/min): 17.0x** (largest: Happy, smallest: Fear).
- This justifies the balanced training recipe: inverse-frequency **weighted CrossEntropy**, label smoothing, mixup, and selecting models by **macro-F1 / balanced accuracy** rather than raw accuracy.

## 3. Preprocessing decisions

- RAF-DB `_aligned` crops are already face-aligned, so no extra face detection is needed for training.
- Train augmentation: random crop/flip/rotation/affine, color jitter, occasional grayscale, CLAHE, RandomErasing (see `src/transforms.py`).
- Eval/inference: resize to 224x224 + ImageNet normalization.
- CLAHE is applied in both training augmentation and live inference so the model sees a consistent contrast distribution.

## 4. Figures

- `class_distribution.png` - per-class train/test counts
- `sample_grid.png` - example faces per emotion
- `image_sizes.png` - source resolution scatter
