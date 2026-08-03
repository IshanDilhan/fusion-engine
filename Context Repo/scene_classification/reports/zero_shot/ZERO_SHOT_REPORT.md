# Zero-shot vs Trained CNN — Scene Classification on Captured Videos

- Frames evaluated: 4436 (4/video)
- Classes: ['classroom', 'kitchen']
- Zero-shot prompt ensembles + face-closeup abstain probe (threshold 0.5)
- Latency measured on this machine (cuda), batch=1 — relative comparison only, not Jetson numbers.

## Results

| model                     |   classroom_acc |   kitchen_acc |   overall_acc |   video_majority_acc |   abstain_rate |   acc_after_abstain |   latency_ms |
|:--------------------------|----------------:|--------------:|--------------:|---------------------:|---------------:|--------------------:|-------------:|
| EfficientNet-B0 (trained) |            97.1 |          63.1 |          82.2 |                 81.5 |            0   |                82.2 |         25.3 |
| CLIP ViT-B/32 (zero-shot) |           100   |          98.8 |          99.5 |                 99.5 |            0.2 |                99.5 |         23.7 |


## Reading the numbers

- `*_acc`: frame-level accuracy per ground-truth folder (forced choice).
- `video_majority_acc`: majority vote over each clip's sampled frames (approximates deployed temporal smoothing).
- `abstain_rate`: frames the zero-shot model judged to be a face close-up with no scene content.
- `acc_after_abstain`: accuracy on the remaining frames — how much of the error is 'scene not visible' vs actual misclassification.