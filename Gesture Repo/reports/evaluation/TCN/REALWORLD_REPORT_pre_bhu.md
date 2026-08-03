# Real-world dataset evaluation (videos/dataset/hri-multimodal-intent-v1.0.0)

1270 clips, model TCN `best_TCN.pth`. offline = whole-clip resample; engine = GestureEngine simulation (EMA + conf gate + 300 ms debounce, video time).

## All trained-class clips (825 clips)

| Mode | Accuracy | Macro-F1 (6 trained classes) |
|---|---|---|
| offline | 69.9% | 66.8% |
| engine | 71.6% | 69.6% |

### Per-class (engine mode)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| idle | 0.28 | 0.98 | 0.44 | 56 |
| wave | 0.98 | 0.34 | 0.51 | 117 |
| thumbs_up | 1.00 | 0.65 | 0.79 | 127 |
| thumbs_down | 0.97 | 0.65 | 0.78 | 279 |
| beckoning | 0.96 | 0.88 | 0.91 | 122 |
| raise_hand | 0.59 | 1.00 | 0.74 | 124 |

### Confusion (engine mode)

| true \ pred | idle | wave | point | thumbs_up | thumbs_down | beckoning | raise_hand | both_hands_up |
|---|---|---|---|---|---|---|---|---|
| **idle** | 55 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| **wave** | 18 | 40 | 0 | 0 | 0 | 0 | 59 | 0 |
| **point** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **thumbs_up** | 37 | 0 | 0 | 83 | 1 | 4 | 2 | 0 |
| **thumbs_down** | 75 | 1 | 0 | 0 | 182 | 0 | 21 | 0 |
| **beckoning** | 8 | 0 | 0 | 0 | 4 | 107 | 3 | 0 |
| **raise_hand** | 0 | 0 | 0 | 0 | 0 | 0 | 124 | 0 |
| **both_hands_up** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Truly unseen scenarios only (no training/val overlap) (579 clips)

| Mode | Accuracy | Macro-F1 (6 trained classes) |
|---|---|---|
| offline | 60.3% | 40.6% |
| engine | 62.2% | 42.4% |

### Per-class (engine mode)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| idle | 0.30 | 0.98 | 0.46 | 56 |
| wave | 0.98 | 0.34 | 0.51 | 117 |
| thumbs_up | 1.00 | 0.65 | 0.79 | 127 |
| thumbs_down | 0.99 | 0.65 | 0.79 | 279 |
| beckoning | 0.00 | 0.00 | 0.00 | 0 |
| raise_hand | 0.00 | 0.00 | 0.00 | 0 |

### Confusion (engine mode)

| true \ pred | idle | wave | point | thumbs_up | thumbs_down | beckoning | raise_hand | both_hands_up |
|---|---|---|---|---|---|---|---|---|
| **idle** | 55 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| **wave** | 18 | 40 | 0 | 0 | 0 | 0 | 59 | 0 |
| **point** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **thumbs_up** | 37 | 0 | 0 | 83 | 1 | 4 | 2 | 0 |
| **thumbs_down** | 75 | 1 | 0 | 0 | 182 | 0 | 21 | 0 |
| **beckoning** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **raise_hand** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **both_hands_up** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Untrained classes (point, both_hands_up) — what the model does

- **both_hands_up** (261 clips) → engine predicts: idle 185/261, raise_hand 63/261, wave 12/261
- **point** (184 clips) → engine predicts: idle 161/184, thumbs_down 15/184, thumbs_up 4/184

## Per-scenario breakdown (engine mode)

| Scenario | Context | Intended | n | Offline acc | Engine acc | Top engine pred | Overlap |
|---|---|---|---|---|---|---|---|
| S01_F04 | classroom | raise_hand | 70 | 100% | 100% | raise_hand | TRAINED (subject+clips in train) |
| S02_F01 | classroom | wave | 61 | 21% | 43% | raise_hand | unseen |
| S03_F05 | classroom | point | 55 | 0% | 0% | idle | unseen |
| S04_F04 | classroom | thumbs_down | 61 | 87% | 84% | thumbs_down | unseen |
| S05_F02 | classroom | both_hands_up | 50 | 0% | 0% | idle | unseen |
| S06_F08 | classroom | thumbs_down | 63 | 87% | 89% | thumbs_down | unseen |
| S07_F03 | classroom | beckoning | 59 | 100% | 100% | beckoning | TRAINED (subject+clips in train) |
| S08_F06 | classroom | idle | 56 | 96% | 98% | idle | unseen |
| S09_F02 | classroom | both_hands_up | 50 | 0% | 0% | idle | unseen |
| S11_F05 | classroom | raise_hand | 54 | 100% | 100% | raise_hand | val/test subject (never trained) |
| S12_F01 | classroom | thumbs_up | 54 | 87% | 85% | thumbs_up | unseen |
| S18_F01 | kitchen | thumbs_up | 73 | 49% | 51% | thumbs_up | unseen |
| S19_F02 | kitchen | both_hands_up | 54 | 0% | 0% | idle | unseen |
| S20_F03 | kitchen | beckoning | 63 | 71% | 76% | beckoning | val/test subject (never trained) |
| S21_F04 | kitchen | thumbs_down | 58 | 57% | 53% | thumbs_down | unseen |
| S22_F05 | kitchen | point | 60 | 0% | 0% | idle | unseen |
| S23_F08 | kitchen | thumbs_down | 30 | 37% | 33% | idle | unseen |
| S24_F07 | kitchen | both_hands_up | 59 | 0% | 0% | idle | unseen |
| S25_F09 | kitchen | wave | 56 | 29% | 25% | raise_hand | unseen |
| S26_F02 | kitchen | both_hands_up | 48 | 0% | 0% | idle | unseen |
| S27_F06 | kitchen | point | 11 | 0% | 0% | idle | unseen |
| S28_F10 | kitchen | thumbs_down | 67 | 46% | 51% | thumbs_down | unseen |
| S29_F03 | kitchen | point | 58 | 0% | 0% | idle | unseen |