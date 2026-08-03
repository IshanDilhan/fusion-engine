# Real-world results v2 (fixed robust face detector)

- Clips: 1180 | labeled: 1032
- Frame-level face coverage: 28087/28320 = 99.2%
- Clips with a face in EVERY sampled frame: 1097/1180
- Clips with ZERO face detected: 0/1180
- Clips with <50% frame coverage (excl. zero): 0/1180


## finetuned_MobileNetV2

### split=val: n=139 clips  acc=74.8%  bAcc=67.2%  macroF1=57.5%

| Surprise | Fear | Disgust | Happy | Sad | Anger | Neutral |
|---|---|---|---|---|---|---|
| 0 | 92 | 31 | 76 | 71 | 33 | 99 |
### split=train (SEEN in fine-tuning): n=893 clips  acc=98.1%  bAcc=97.9%  macroF1=97.4%

| Surprise | Fear | Disgust | Happy | Sad | Anger | Neutral |
|---|---|---|---|---|---|---|
| 98 | 96 | 98 | 100 | 95 | 96 | 99 |

Prediction distribution (all clips): Neutral 418, Happy 266, Sad 163, Anger 126, Disgust 76, Surprise 74, Fear 57


## best_MobileNetV2

### split=val: n=139 clips  acc=65.5%  bAcc=46.9%  macroF1=41.4%

| Surprise | Fear | Disgust | Happy | Sad | Anger | Neutral |
|---|---|---|---|---|---|---|
| 17 | 0 | 23 | 69 | 62 | 23 | 95 |
### split=train: n=893 clips  acc=64.7%  bAcc=42.9%  macroF1=40.6%

| Surprise | Fear | Disgust | Happy | Sad | Anger | Neutral |
|---|---|---|---|---|---|---|
| 53 | 0 | 25 | 80 | 29 | 20 | 77 |

Prediction distribution (all clips): Neutral 608, Happy 344, Surprise 110, Sad 48, Anger 36, Disgust 25, Fear 9
