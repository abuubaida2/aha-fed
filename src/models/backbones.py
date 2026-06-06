"""Model backbones for FL experiments.

`small_cnn` is the quickstart backbone used for fast local development and CI.
`densenet121` is the headline backbone used in chest-X-ray experiments.
"""
from __future__ import annotations

import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x))


class MedCNN(nn.Module):
    """Slightly larger 3-conv block CNN for real medical-imaging benchmarks.

    Uses GroupNorm (stable under FL non-IID) and non-inplace ReLU
    (Opacus DP-SGD requires non-inplace ops).
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if arch == "small_cnn":
        return SmallCNN(num_classes)
    if arch == "med_cnn":
        return MedCNN(num_classes)
    if arch == "densenet121":
        from torchvision import models
        weights = "DEFAULT" if pretrained else None
        m = models.densenet121(weights=weights)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
        return m
    raise ValueError(f"unknown arch: {arch}")
