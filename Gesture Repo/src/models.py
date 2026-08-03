"""
Model zoo for gesture sequence classification (guide §7).

Three architectures, all < 1M parameters, input [B, WINDOW, FEATURE_DIM]:
  BiGRU           2-layer bidirectional GRU, mean+max pooling over time
  TCN             4 residual dilated temporal-conv blocks, global pooling
  TinyTransformer 2-layer encoder with CLS token and learned positions
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import DROPOUT, FEATURE_DIM, HIDDEN_SIZE, NUM_CLASSES, WINDOW


class BiGRUClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, feature_dim=FEATURE_DIM,
                 hidden_size=HIDDEN_SIZE, num_layers=2, dropout=DROPOUT):
        super().__init__()
        self.gru = nn.GRU(feature_dim, hidden_size, num_layers=num_layers,
                          batch_first=True, bidirectional=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        feat = 4 * hidden_size  # mean + max pooling over bidirectional outputs
        self.head = nn.Sequential(
            nn.LayerNorm(feat),
            nn.Dropout(dropout),
            nn.Linear(feat, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):                      # x: [B, T, F]
        out, _ = self.gru(x)                   # [B, T, 2H]
        pooled = torch.cat([out.mean(dim=1), out.max(dim=1).values], dim=1)
        return self.head(pooled)


class _TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) // 2 * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))


class TCNClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, feature_dim=FEATURE_DIM,
                 hidden_size=HIDDEN_SIZE, kernel_size=5, dropout=DROPOUT):
        super().__init__()
        self.proj = nn.Conv1d(feature_dim, hidden_size, 1)
        self.blocks = nn.Sequential(
            *[_TCNBlock(hidden_size, kernel_size, d, dropout) for d in (1, 2, 4, 8)])
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):                      # x: [B, T, F]
        z = self.blocks(self.proj(x.transpose(1, 2)))  # [B, H, T]
        return self.head(z.mean(dim=2))


class TinyTransformer(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, feature_dim=FEATURE_DIM,
                 hidden_size=HIDDEN_SIZE, num_layers=2, num_heads=4,
                 dropout=DROPOUT, window=WINDOW):
        super().__init__()
        self.proj = nn.Linear(feature_dim, hidden_size)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.pos = nn.Parameter(torch.zeros(1, window + 1, hidden_size))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=2 * hidden_size,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):                      # x: [B, T, F]
        z = self.proj(x)
        z = torch.cat([self.cls.expand(z.size(0), -1, -1), z], dim=1)
        z = self.encoder(z + self.pos[:, : z.size(1)])
        return self.head(z[:, 0])


ALL_MODELS = {
    "BiGRU": BiGRUClassifier,
    "TCN": TCNClassifier,
    "TinyTransformer": TinyTransformer,
}


def build_model(name, **kwargs):
    return ALL_MODELS[name](**kwargs)


def safe_name(name):
    return name.replace("-", "_").replace(" ", "_")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model):
    return count_params(model) * 4 / (1024 ** 2)
