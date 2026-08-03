"""
MobileNetV2 + LSTM sequence classifier.

Reverse-engineered from checkpoints/{best,finetuned}_MobileNetV2_LSTM.pth —
the original training script for these checkpoints was lost (not present
anywhere in the repo when this file was written). The architecture below was
inferred from the checkpoints' state_dict keys/shapes and verified with a
STRICT state_dict load (torch raises on any shape/name mismatch), so the
architecture is exact:

    features.*      -> torchvision mobilenet_v2().features (1280-ch output)
    lstm.*           -> nn.LSTM(input_size=1280, hidden_size=256, num_layers=1,
                        batch_first=True)
    classifier.1.*    -> nn.Sequential(nn.Dropout(p), nn.Linear(256, 7))[1]

One thing that could NOT be recovered from the weights alone: whether the
classifier was fed the LSTM's last hidden state (out[:, -1, :]) or the
mean-pooled output over time (out.mean(dim=1)). forward() supports both via
`agg`; try both when evaluating and prefer whichever is intended once the
original recipe is known.
"""
import torch.nn as nn
import torchvision.models as tvm


class MobileNetV2LSTM(nn.Module):
    def __init__(self, num_classes=7, hidden_size=256, dropout=0.5):
        super().__init__()
        backbone = tvm.mobilenet_v2(weights=None)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(1280, hidden_size, num_layers=1, batch_first=True)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, num_classes))

    def forward(self, x, agg="last"):
        """x: (B, T, C, H, W) sequence of face crops. agg: 'last' or 'mean'."""
        b, t, c, h, w = x.shape
        feat = self.features(x.view(b * t, c, h, w))
        feat = self.pool(feat).flatten(1).view(b, t, -1)
        out, _ = self.lstm(feat)
        pooled = out[:, -1, :] if agg == "last" else out.mean(dim=1)
        return self.classifier(pooled)
