# Emotion Model - Stage 3: Hyper-parameter Tuning Report (EfficientNet-B0)

Grid search of **6 configurations**, each a shortened 2+8-epoch two-stage run, ranked by **macro-F1**.

## Search space

- Base LR: [0.0001, 5e-05, 0.0002]
- Batch size: [32, 64]
- Optimizer: ['adam']
- Weight decay: [1e-05]
- Label smoothing: [0.1]

## Top 10 configurations

|   trial |   base_lr |   batch_size | optimizer   |   weight_decay |   label_smoothing |   accuracy |   balanced_acc |   macro_f1 |   f1_weighted |
|--------:|----------:|-------------:|:------------|---------------:|------------------:|-----------:|---------------:|-----------:|--------------:|
|       5 |    0.0002 |           32 | adam        |          1e-05 |               0.1 |      77.09 |          75.51 |      69.67 |         78.08 |
|       6 |    0.0002 |           64 | adam        |          1e-05 |               0.1 |      74.35 |          72.94 |      66.11 |         75.89 |
|       1 |    0.0001 |           32 | adam        |          1e-05 |               0.1 |      72.29 |          73.5  |      65.14 |         73.82 |
|       2 |    0.0001 |           64 | adam        |          1e-05 |               0.1 |      68.97 |          71.3  |      61.7  |         71.28 |
|       3 |    5e-05  |           32 | adam        |          1e-05 |               0.1 |      68.84 |          69.47 |      61.28 |         70.97 |
|       4 |    5e-05  |           64 | adam        |          1e-05 |               0.1 |      63.14 |          67.44 |      56.66 |         65.84 |


## Best configuration

- Base LR: **0.0002**
- Batch size: **32**
- Optimizer: **adam**
- Weight decay: **1e-05**
- Label smoothing: **0.1**
- Macro-F1: **69.67%** | Accuracy: 77.09%

## Final training command

```
python scripts/train.py --model "EfficientNet-B0" --batch_size 32 --base_lr 0.0002 --optimizer adam --weight_decay 1e-05 --label_smoothing 0.1
```