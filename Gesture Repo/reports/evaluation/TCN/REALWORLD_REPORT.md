# Gesture real-world evaluation — REGENERATED

- Checkpoint: `best_TCN.pth` (sha256 `7e58aa89645c2796…`, the deployed weights)
- Regenerated: 2026-07-24 on the curated dataset (1007 clips with a defined intended gesture)
- Method: offline whole-clip — pose-valid frames → uniform-resample to 32 → TCN argmax (deployed feature pipeline, reusing per-frame caches).
- **Supersedes the pre-both-hands-up report** (kept as `REALWORLD_REPORT_pre_bhu.md`). Key change: `both_hands_up` is now a trained, recognized class; `point` remains weak.

## Overall: acc = 76.8% · macro-F1 = 77.7% (7 classes present)

## Per-class

| class | support | precision | recall | F1 |
|---|---|---|---|---|
| wave | 89 | 0.93 | 0.43 | 0.58 |
| point | 152 | 0.95 | 0.28 | 0.43 |
| thumbs_up | 108 | 0.97 | 0.80 | 0.87 |
| thumbs_down | 219 | 0.99 | 0.90 | 0.94 |
| beckoning | 96 | 0.82 | 0.94 | 0.87 |
| raise_hand | 124 | 0.65 | 0.99 | 0.79 |
| both_hands_up | 219 | 1.00 | 0.90 | 0.95 |

## Per-scenario

| scenario | context | intended | n | accuracy | top prediction |
|---|---|---|---|---|---|
| S01_F04 | classroom | raise_hand | 70 | 100% | raise_hand |
| S02_F01 | classroom | wave | 42 | 62% | wave |
| S03_F05 | classroom | point | 42 | 0%  ⚠ | idle |
| S04_F04 | classroom | thumbs_down | 61 | 85% | thumbs_down |
| S05_F02 | classroom | both_hands_up | 48 | 100% | both_hands_up |
| S06_F08 | classroom | thumbs_down | 55 | 100% | thumbs_down |
| S07_F03 | classroom | beckoning | 59 | 100% | beckoning |
| S09_F02 | classroom | both_hands_up | 40 | 98% | both_hands_up |
| S11_F05 | classroom | raise_hand | 54 | 98% | raise_hand |
| S12_F01 | classroom | thumbs_up | 54 | 61% | thumbs_up |
| S18_F01 | kitchen | thumbs_up | 54 | 98% | thumbs_up |
| S19_F02 | kitchen | both_hands_up | 46 | 83% | both_hands_up |
| S20_F03 | kitchen | beckoning | 37 | 84% | beckoning |
| S21_F04 | kitchen | thumbs_down | 31 | 94% | thumbs_down |
| S22_F05 | kitchen | point | 60 | 5%  ⚠ | idle |
| S23_F08 | kitchen | thumbs_down | 19 | 74% | thumbs_down |
| S24_F07 | kitchen | both_hands_up | 53 | 85% | both_hands_up |
| S25_F09 | kitchen | wave | 47 | 26%  ⚠ | raise_hand |
| S26_F02 | kitchen | both_hands_up | 32 | 84% | both_hands_up |
| S28_F10 | kitchen | thumbs_down | 53 | 89% | thumbs_down |
| S29_F03 | kitchen | point | 50 | 78% | point |

## Notes
- `point` scenarios (S03, S22, S29) score low — static pointing is hard to separate from idle in keypoints; a known, still-open gap.
- `both_hands_up` scenarios (S05, S09, S19, S24, S26) are now recognized (the pre_bhu report showed 0% here).
- S08 is a designed [MISSING]-gesture scenario (hands occluded) and is excluded from scoring.
