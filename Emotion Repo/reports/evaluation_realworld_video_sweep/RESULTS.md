# Real-world video sweep - cleaned labels, fixed robust detector

- Clips on disk (post manual cleanup): 1061
- Labeled clips: 964
- Frame-level face coverage: 25276/25464 = 99.3%
- Clips with a face in every sampled frame: 990/1061
- Clips with zero face detected: 0/1061


## Summary (val split, sorted by macro-F1)

| model                           |   n_clips |   accuracy |   balanced_acc |   macro_f1 |
|:--------------------------------|----------:|-----------:|---------------:|-----------:|
| finetuned_MobileNetV2           |        91 |       98.9 |           99   |       99.3 |
| finetuned_MobileNetV2_LSTM_last |        91 |       87.9 |           91.9 |       84.8 |
| finetuned_MobileNetV2_LSTM_mean |        91 |       89   |           92.4 |       77.6 |
| best_MobileNetV2_LSTM_last      |        91 |       71.4 |           73.1 |       54.8 |
| best_MobileNetV2_LSTM_mean      |        91 |       72.5 |           68   |       50.5 |
| best_MobileNetV2                |        91 |       84.6 |           55.3 |       48.6 |

## best_MobileNetV2

### split=val: n=91 clips  acc=84.6%  bAcc=55.3%  macroF1=48.6%

| Emotion | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Surprise | 0% | 0% | 0% | 0 |
| Fear | 0% | 0% | 0% | 6 |
| Disgust | 0% | 0% | 0% | 1 |
| Happy | 82% | 88% | 85% | 16 |
| Sad | 100% | 87% | 93% | 15 |
| Anger | 75% | 60% | 67% | 5 |
| Neutral | 94% | 98% | 96% | 48 |

Confusion matrix (rows=true, cols=predicted):

|          |   Surprise |   Fear |   Disgust |   Happy |   Sad |   Anger |   Neutral |
|:---------|-----------:|-------:|----------:|--------:|------:|--------:|----------:|
| Surprise |          0 |      0 |         0 |       0 |     0 |       0 |         0 |
| Fear     |          6 |      0 |         0 |       0 |     0 |       0 |         0 |
| Disgust  |          0 |      0 |         0 |       0 |     0 |       1 |         0 |
| Happy    |          0 |      0 |         0 |      14 |     0 |       0 |         2 |
| Sad      |          0 |      0 |         1 |       0 |    13 |       0 |         1 |
| Anger    |          0 |      0 |         0 |       2 |     0 |       3 |         0 |
| Neutral  |          0 |      0 |         0 |       1 |     0 |       0 |        47 |

### split=train: n=873 clips  acc=65.2%  bAcc=43.0%  macroF1=40.8%

| Emotion | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Surprise | 43% | 66% | 52% | 53 |
| Fear | 0% | 0% | 0% | 37 |
| Disgust | 56% | 17% | 26% | 60 |
| Happy | 70% | 92% | 80% | 225 |
| Sad | 83% | 19% | 31% | 130 |
| Anger | 100% | 11% | 20% | 74 |
| Neutral | 65% | 97% | 78% | 294 |

Confusion matrix (rows=true, cols=predicted):

|          |   Surprise |   Fear |   Disgust |   Happy |   Sad |   Anger |   Neutral |
|:---------|-----------:|-------:|----------:|--------:|------:|--------:|----------:|
| Surprise |         35 |      0 |         0 |       9 |     0 |       0 |         9 |
| Fear     |         32 |      0 |         0 |       0 |     0 |       0 |         5 |
| Disgust  |          0 |      0 |        10 |      30 |     1 |       0 |        19 |
| Happy    |          0 |      0 |         0 |     206 |     1 |       0 |        18 |
| Sad      |          2 |      0 |         8 |       4 |    25 |       0 |        91 |
| Anger    |         10 |      5 |         0 |      40 |     1 |       8 |        10 |
| Neutral  |          3 |      0 |         0 |       4 |     2 |       0 |       285 |


## finetuned_MobileNetV2

### split=val: n=91 clips  acc=98.9%  bAcc=99.0%  macroF1=99.3%

| Emotion | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Surprise | 0% | 0% | 0% | 0 |
| Fear | 100% | 100% | 100% | 6 |
| Disgust | 100% | 100% | 100% | 1 |
| Happy | 100% | 94% | 97% | 16 |
| Sad | 100% | 100% | 100% | 15 |
| Anger | 100% | 100% | 100% | 5 |
| Neutral | 98% | 100% | 99% | 48 |

Confusion matrix (rows=true, cols=predicted):

|          |   Surprise |   Fear |   Disgust |   Happy |   Sad |   Anger |   Neutral |
|:---------|-----------:|-------:|----------:|--------:|------:|--------:|----------:|
| Surprise |          0 |      0 |         0 |       0 |     0 |       0 |         0 |
| Fear     |          0 |      6 |         0 |       0 |     0 |       0 |         0 |
| Disgust  |          0 |      0 |         1 |       0 |     0 |       0 |         0 |
| Happy    |          0 |      0 |         0 |      15 |     0 |       0 |         1 |
| Sad      |          0 |      0 |         0 |       0 |    15 |       0 |         0 |
| Anger    |          0 |      0 |         0 |       0 |     0 |       5 |         0 |
| Neutral  |          0 |      0 |         0 |       0 |     0 |       0 |        48 |

### split=train (seen in fine-tuning - memorized, not a generalization estimate): n=873 clips  acc=99.0%  bAcc=98.6%  macroF1=98.5%

| Emotion | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Surprise | 100% | 98% | 99% | 53 |
| Fear | 97% | 100% | 99% | 37 |
| Disgust | 98% | 98% | 98% | 60 |
| Happy | 100% | 100% | 100% | 225 |
| Sad | 98% | 97% | 98% | 130 |
| Anger | 95% | 97% | 96% | 74 |
| Neutral | 100% | 100% | 100% | 294 |

Confusion matrix (rows=true, cols=predicted):

|          |   Surprise |   Fear |   Disgust |   Happy |   Sad |   Anger |   Neutral |
|:---------|-----------:|-------:|----------:|--------:|------:|--------:|----------:|
| Surprise |         52 |      1 |         0 |       0 |     0 |       0 |         0 |
| Fear     |          0 |     37 |         0 |       0 |     0 |       0 |         0 |
| Disgust  |          0 |      0 |        59 |       0 |     1 |       0 |         0 |
| Happy    |          0 |      0 |         0 |     225 |     0 |       0 |         0 |
| Sad      |          0 |      0 |         0 |       0 |   126 |       4 |         0 |
| Anger    |          0 |      0 |         1 |       1 |     0 |      72 |         0 |
| Neutral  |          0 |      0 |         0 |       0 |     1 |       0 |       293 |


## best_MobileNetV2_LSTM_last

### split=val: n=91 clips  acc=71.4%  bAcc=73.1%  macroF1=54.8%

| Emotion | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Surprise | 0% | 0% | 0% | 0 |
| Fear | 60% | 50% | 55% | 6 |
| Disgust | 25% | 100% | 40% | 1 |
| Happy | 80% | 25% | 38% | 16 |
| Sad | 60% | 80% | 69% | 15 |
| Anger | 100% | 100% | 100% | 5 |
| Neutral | 82% | 83% | 82% | 48 |

Confusion matrix (rows=true, cols=predicted):

|          |   Surprise |   Fear |   Disgust |   Happy |   Sad |   Anger |   Neutral |
|:---------|-----------:|-------:|----------:|--------:|------:|--------:|----------:|
| Surprise |          0 |      0 |         0 |       0 |     0 |       0 |         0 |
| Fear     |          3 |      3 |         0 |       0 |     0 |       0 |         0 |
| Disgust  |          0 |      0 |         1 |       0 |     0 |       0 |         0 |
| Happy    |          0 |      2 |         0 |       4 |     1 |       0 |         9 |
| Sad      |          0 |      0 |         3 |       0 |    12 |       0 |         0 |
| Anger    |          0 |      0 |         0 |       0 |     0 |       5 |         0 |
| Neutral  |          0 |      0 |         0 |       1 |     7 |       0 |        40 |

### split=train: n=873 clips  acc=57.5%  bAcc=42.6%  macroF1=42.2%

| Emotion | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Surprise | 49% | 68% | 57% | 53 |
| Fear | 7% | 3% | 4% | 37 |
| Disgust | 29% | 18% | 22% | 60 |
| Happy | 84% | 59% | 69% | 225 |
| Sad | 50% | 18% | 26% | 130 |
| Anger | 57% | 42% | 48% | 74 |
| Neutral | 55% | 91% | 68% | 294 |

Confusion matrix (rows=true, cols=predicted):

|          |   Surprise |   Fear |   Disgust |   Happy |   Sad |   Anger |   Neutral |
|:---------|-----------:|-------:|----------:|--------:|------:|--------:|----------:|
| Surprise |         36 |      0 |         1 |       2 |     0 |       6 |         8 |
| Fear     |         28 |      1 |         1 |       0 |     0 |       2 |         5 |
| Disgust  |          1 |      3 |        11 |       7 |     8 |       1 |        29 |
| Happy    |          1 |      2 |         4 |     133 |     3 |       0 |        82 |
| Sad      |          0 |      0 |        20 |       1 |    23 |       3 |        83 |
| Anger    |          3 |      8 |         1 |      16 |     1 |      31 |        14 |
| Neutral  |          4 |      1 |         0 |       0 |    11 |      11 |       267 |


## best_MobileNetV2_LSTM_mean

### split=val: n=91 clips  acc=72.5%  bAcc=68.0%  macroF1=50.5%

| Emotion | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Surprise | 0% | 0% | 0% | 0 |
| Fear | 17% | 17% | 17% | 6 |
| Disgust | 33% | 100% | 50% | 1 |
| Happy | 50% | 6% | 11% | 16 |
| Sad | 82% | 93% | 88% | 15 |
| Anger | 100% | 100% | 100% | 5 |
| Neutral | 85% | 92% | 88% | 48 |

Confusion matrix (rows=true, cols=predicted):

|          |   Surprise |   Fear |   Disgust |   Happy |   Sad |   Anger |   Neutral |
|:---------|-----------:|-------:|----------:|--------:|------:|--------:|----------:|
| Surprise |          0 |      0 |         0 |       0 |     0 |       0 |         0 |
| Fear     |          5 |      1 |         0 |       0 |     0 |       0 |         0 |
| Disgust  |          0 |      0 |         1 |       0 |     0 |       0 |         0 |
| Happy    |          1 |      5 |         0 |       1 |     1 |       0 |         8 |
| Sad      |          0 |      0 |         1 |       0 |    14 |       0 |         0 |
| Anger    |          0 |      0 |         0 |       0 |     0 |       5 |         0 |
| Neutral  |          0 |      0 |         1 |       1 |     2 |       0 |        44 |

### split=train: n=873 clips  acc=59.5%  bAcc=45.9%  macroF1=45.0%

| Emotion | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Surprise | 54% | 81% | 65% | 53 |
| Fear | 0% | 0% | 0% | 37 |
| Disgust | 42% | 30% | 35% | 60 |
| Happy | 87% | 60% | 71% | 225 |
| Sad | 50% | 16% | 24% | 130 |
| Anger | 60% | 42% | 49% | 74 |
| Neutral | 57% | 92% | 70% | 294 |

Confusion matrix (rows=true, cols=predicted):

|          |   Surprise |   Fear |   Disgust |   Happy |   Sad |   Anger |   Neutral |
|:---------|-----------:|-------:|----------:|--------:|------:|--------:|----------:|
| Surprise |         43 |      2 |         0 |       2 |     0 |       2 |         4 |
| Fear     |         32 |      0 |         0 |       0 |     0 |       2 |         3 |
| Disgust  |          0 |      8 |        18 |       5 |     1 |       0 |        28 |
| Happy    |          0 |      7 |         4 |     136 |     4 |       6 |        68 |
| Sad      |          0 |      0 |        18 |       1 |    21 |       2 |        88 |
| Anger    |          3 |      9 |         3 |      12 |     3 |      31 |        13 |
| Neutral  |          2 |      0 |         0 |       0 |    13 |       9 |       270 |


## finetuned_MobileNetV2_LSTM_last

### split=val: n=91 clips  acc=87.9%  bAcc=91.9%  macroF1=84.8%

| Emotion | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Surprise | 0% | 0% | 0% | 0 |
| Fear | 86% | 100% | 92% | 6 |
| Disgust | 50% | 100% | 67% | 1 |
| Happy | 92% | 69% | 79% | 16 |
| Sad | 67% | 93% | 78% | 15 |
| Anger | 100% | 100% | 100% | 5 |
| Neutral | 98% | 90% | 93% | 48 |

Confusion matrix (rows=true, cols=predicted):

|          |   Surprise |   Fear |   Disgust |   Happy |   Sad |   Anger |   Neutral |
|:---------|-----------:|-------:|----------:|--------:|------:|--------:|----------:|
| Surprise |          0 |      0 |         0 |       0 |     0 |       0 |         0 |
| Fear     |          0 |      6 |         0 |       0 |     0 |       0 |         0 |
| Disgust  |          0 |      0 |         1 |       0 |     0 |       0 |         0 |
| Happy    |          0 |      1 |         0 |      11 |     3 |       0 |         1 |
| Sad      |          0 |      0 |         1 |       0 |    14 |       0 |         0 |
| Anger    |          0 |      0 |         0 |       0 |     0 |       5 |         0 |
| Neutral  |          0 |      0 |         0 |       1 |     4 |       0 |        43 |

### split=train (seen in fine-tuning - memorized, not a generalization estimate): n=873 clips  acc=96.3%  bAcc=95.4%  macroF1=95.2%

| Emotion | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Surprise | 93% | 100% | 96% | 53 |
| Fear | 97% | 95% | 96% | 37 |
| Disgust | 96% | 90% | 93% | 60 |
| Happy | 100% | 96% | 98% | 225 |
| Sad | 99% | 94% | 96% | 130 |
| Anger | 84% | 95% | 89% | 74 |
| Neutral | 96% | 99% | 98% | 294 |

Confusion matrix (rows=true, cols=predicted):

|          |   Surprise |   Fear |   Disgust |   Happy |   Sad |   Anger |   Neutral |
|:---------|-----------:|-------:|----------:|--------:|------:|--------:|----------:|
| Surprise |         53 |      0 |         0 |       0 |     0 |       0 |         0 |
| Fear     |          0 |     35 |         1 |       0 |     0 |       0 |         1 |
| Disgust  |          0 |      0 |        54 |       0 |     0 |       6 |         0 |
| Happy    |          1 |      1 |         0 |     215 |     1 |       1 |         6 |
| Sad      |          0 |      0 |         0 |       0 |   122 |       5 |         3 |
| Anger    |          2 |      0 |         1 |       0 |     0 |      70 |         1 |
| Neutral  |          1 |      0 |         0 |       0 |     0 |       1 |       292 |


## finetuned_MobileNetV2_LSTM_mean

### split=val: n=91 clips  acc=89.0%  bAcc=92.4%  macroF1=77.6%

| Emotion | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Surprise | 0% | 0% | 0% | 0 |
| Fear | 86% | 100% | 92% | 6 |
| Disgust | 100% | 100% | 100% | 1 |
| Happy | 91% | 62% | 74% | 16 |
| Sad | 68% | 100% | 81% | 15 |
| Anger | 100% | 100% | 100% | 5 |
| Neutral | 100% | 92% | 96% | 48 |

Confusion matrix (rows=true, cols=predicted):

|          |   Surprise |   Fear |   Disgust |   Happy |   Sad |   Anger |   Neutral |
|:---------|-----------:|-------:|----------:|--------:|------:|--------:|----------:|
| Surprise |          0 |      0 |         0 |       0 |     0 |       0 |         0 |
| Fear     |          0 |      6 |         0 |       0 |     0 |       0 |         0 |
| Disgust  |          0 |      0 |         1 |       0 |     0 |       0 |         0 |
| Happy    |          1 |      1 |         0 |      10 |     4 |       0 |         0 |
| Sad      |          0 |      0 |         0 |       0 |    15 |       0 |         0 |
| Anger    |          0 |      0 |         0 |       0 |     0 |       5 |         0 |
| Neutral  |          0 |      0 |         0 |       1 |     3 |       0 |        44 |

### split=train (seen in fine-tuning - memorized, not a generalization estimate): n=873 clips  acc=98.4%  bAcc=97.9%  macroF1=97.7%

| Emotion | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Surprise | 96% | 100% | 98% | 53 |
| Fear | 100% | 100% | 100% | 37 |
| Disgust | 100% | 90% | 95% | 60 |
| Happy | 100% | 100% | 100% | 225 |
| Sad | 99% | 97% | 98% | 130 |
| Anger | 88% | 100% | 94% | 74 |
| Neutral | 100% | 99% | 99% | 294 |

Confusion matrix (rows=true, cols=predicted):

|          |   Surprise |   Fear |   Disgust |   Happy |   Sad |   Anger |   Neutral |
|:---------|-----------:|-------:|----------:|--------:|------:|--------:|----------:|
| Surprise |         53 |      0 |         0 |       0 |     0 |       0 |         0 |
| Fear     |          0 |     37 |         0 |       0 |     0 |       0 |         0 |
| Disgust  |          0 |      0 |        54 |       0 |     0 |       6 |         0 |
| Happy    |          0 |      0 |         0 |     224 |     0 |       0 |         1 |
| Sad      |          0 |      0 |         0 |       0 |   126 |       4 |         0 |
| Anger    |          0 |      0 |         0 |       0 |     0 |      74 |         0 |
| Neutral  |          2 |      0 |         0 |       0 |     1 |       0 |       291 |
