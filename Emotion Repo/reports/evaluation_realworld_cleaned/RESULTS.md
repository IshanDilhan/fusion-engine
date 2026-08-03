# Real-world evaluation - cleaned labels

- Raw clips remaining after manual cleanup: 1061
- Cleaned val clips (labels trusted): 91
- Cleaned val per-class: Surprise=0, Fear=6, Disgust=1, Happy=16, Sad=15, Anger=5, Neutral=48

## 1) CNN frame-level

### best_MobileNetV2: n=799 frames  acc=68.8%  bAcc=34.5%  macroF1=31.6%

| | Surprise | Fear | Disgust | Happy | Sad | Anger | Neutral |
|---|---|---|---|---|---|---|---|
| F1 | 0 | 16 | 0 | 18 | 74 | 29 | 85 |

### finetuned_MobileNetV2: n=799 frames  acc=72.5%  bAcc=43.7%  macroF1=38.9%

| | Surprise | Fear | Disgust | Happy | Sad | Anger | Neutral |
|---|---|---|---|---|---|---|---|
| F1 | 0 | 35 | 0 | 38 | 74 | 39 | 86 |


## 2) CNN clip-level via mean-softmax (fusion baseline)

### best_MobileNetV2: n=91 clips  acc=61.5%  bAcc=35.3%  macroF1=28.2%

| | Surprise | Fear | Disgust | Happy | Sad | Anger | Neutral |
|---|---|---|---|---|---|---|---|
| F1 | 0 | 0 | 0 | 19 | 67 | 33 | 78 |

### finetuned_MobileNetV2: n=91 clips  acc=64.8%  bAcc=47.8%  macroF1=37.4%

| | Surprise | Fear | Disgust | Happy | Sad | Anger | Neutral |
|---|---|---|---|---|---|---|---|
| F1 | 0 | 40 | 0 | 29 | 70 | 43 | 81 |


## 3) LSTM clip-level

### best_MobileNetV2_LSTM (agg=last): n=91 clips  acc=52.7%  bAcc=35.5%  macroF1=25.6%

| | Surprise | Fear | Disgust | Happy | Sad | Anger | Neutral |
|---|---|---|---|---|---|---|---|
| F1 | 0 | 36 | 0 | 0 | 42 | 22 | 79 |

### best_MobileNetV2_LSTM (agg=mean): n=91 clips  acc=54.9%  bAcc=32.3%  macroF1=22.8%

| | Surprise | Fear | Disgust | Happy | Sad | Anger | Neutral |
|---|---|---|---|---|---|---|---|
| F1 | 0 | 29 | 0 | 0 | 48 | 0 | 83 |

### finetuned_MobileNetV2_LSTM (agg=last): n=91 clips  acc=58.2%  bAcc=39.7%  macroF1=32.6%

| | Surprise | Fear | Disgust | Happy | Sad | Anger | Neutral |
|---|---|---|---|---|---|---|---|
| F1 | 0 | 25 | 0 | 18 | 64 | 44 | 77 |

### finetuned_MobileNetV2_LSTM (agg=mean): n=91 clips  acc=57.1%  bAcc=41.7%  macroF1=33.8%

| | Surprise | Fear | Disgust | Happy | Sad | Anger | Neutral |
|---|---|---|---|---|---|---|---|
| F1 | 0 | 50 | 0 | 20 | 57 | 36 | 74 |
