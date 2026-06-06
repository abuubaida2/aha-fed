"""Standalone (non-FL) training sanity check on synthetic data."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.datasets import SyntheticMedicalImaging
from src.models.backbones import build_model
from src.utils.seed import set_seed


def main() -> None:
    set_seed(42)
    train = SyntheticMedicalImaging(num_samples=600, num_classes=4, image_size=32, center_id=0, seed=42)
    test = SyntheticMedicalImaging(num_samples=400, num_classes=4, image_size=32, center_id=999, seed=999)
    loader = DataLoader(train, batch_size=32, shuffle=True)
    test_loader = DataLoader(test, batch_size=32)

    model = build_model("small_cnn", num_classes=4, pretrained=False)
    print("First training batch x mean per channel:")
    x0, y0 = next(iter(loader))
    print("  x.shape:", x0.shape, "y.shape:", y0.shape)
    print("  per-class channel means:")
    for c in range(4):
        mask = y0 == c
        if mask.any():
            print(f"    class {c}: {x0[mask].mean(dim=(0,2,3)).tolist()} (n={mask.sum().item()})")

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        running = 0.0
        for x, y in loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * y.shape[0]
        train_loss = running / len(train)

        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                correct += (model(x).argmax(1) == y).sum().item()
        print(f"epoch {epoch+1} train_loss={train_loss:.4f} test_acc={correct/len(test):.4f}")


if __name__ == "__main__":
    main()
