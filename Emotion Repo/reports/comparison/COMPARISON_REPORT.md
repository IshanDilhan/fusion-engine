# Emotion Model - Stage 1: Model Comparison Report

Candidate ImageNet-pretrained backbones trained with the identical balanced two-stage recipe (weighted CE + label smoothing + mixup + cosine warmup) and ranked by **macro-F1** within a **20 MB** deployment budget.

- Search length: 5 head-only + 20 full-finetune epochs/model
- Batch size: 64
- Selection metric: macro-F1 (every emotion weighted equally)

## Results

| model             |   params_m |   size_mb | within_budget   |   accuracy |   balanced_acc |   f1_weighted |   macro_f1 |   gpu_ms |   cpu_ms |   train_time_s |
|:------------------|-----------:|----------:|:----------------|-----------:|---------------:|--------------:|-----------:|---------:|---------:|---------------:|
| EfficientNet-B0   |       4.02 |     15.48 | True            |      77.05 |          75.61 |         78.13 |      69.76 |     1.75 |     9.59 |          298.3 |
| MobileNetV3-Large |       4.21 |     16.16 | True            |      71.97 |          71.36 |         73.88 |      63.75 |     1.33 |     4.39 |          295.5 |
| MobileNetV2       |       2.23 |      8.65 | True            |      69.56 |          70.02 |         71.94 |      61.34 |     1.09 |    11.47 |          292.6 |
| MNASNet1_0        |       3.11 |     12.01 | True            |      66.98 |          65.55 |         69.5  |      58.28 |     1.03 |    91.19 |          288.4 |


## Recommendation: **EfficientNet-B0**

- Best macro-F1 among models within the 20 MB budget.
- Size 15.48 MB, macro-F1 69.76%, accuracy 77.05%.

## Next step

```
python scripts/train.py --model "EfficientNet-B0"
```

> Note: these numbers come from a shortened search. The final reported accuracy comes from the full training run (Stage 2).