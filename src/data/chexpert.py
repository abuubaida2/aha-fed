"""CheXpert dataset loader.

Expects the standard CheXpert-v1.0-small layout::

    chexpert_root/
        CheXpert-v1.0-small/
            train.csv      Path, Sex, Age, Frontal/Lateral, AP/PA, then 14 disease columns
            valid.csv
            train/         (image files referenced by Path)
            valid/

Multi-label task with 14 diseases. The dataset uses {1, 0, -1, NaN}:
    1   = positive
    0   = negative
    -1  = uncertain
    NaN = unmentioned

The ``uncertainty_policy`` argument follows Irvin et al. (2019):
    "ones"    — treat -1 as positive (default; best for most diseases)
    "zeros"   — treat -1 as negative
    "ignore"  — set -1 to NaN, then drop
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset

CHEXPERT_DISEASES: tuple[str, ...] = (
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
)


def _default_transform(image_size: int = 224) -> Callable:
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )


class CheXpertDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        image_root: str | Path,
        transform: Callable | None = None,
        uncertainty_policy: str = "ones",
        diseases: Sequence[str] = CHEXPERT_DISEASES,
        image_size: int = 224,
    ):
        csv_path = Path(csv_path)
        image_root = Path(image_root)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"CheXpert CSV not found: {csv_path}. "
                f"Expected layout: {image_root}/CheXpert-v1.0-small/train.csv"
            )

        df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.diseases = list(diseases)
        self.transform = transform or _default_transform(image_size)

        for d in self.diseases:
            if d not in df.columns:
                raise KeyError(
                    f"Expected disease column '{d}' in {csv_path}, got {list(df.columns)}"
                )

        labels = df[self.diseases].copy()
        if uncertainty_policy == "ones":
            labels = labels.fillna(0).replace(-1, 1)
        elif uncertainty_policy == "zeros":
            labels = labels.fillna(0).replace(-1, 0)
        elif uncertainty_policy == "ignore":
            labels = labels.replace(-1, pd.NA).dropna()
            df = df.loc[labels.index]
        else:
            raise ValueError(f"unknown uncertainty_policy: {uncertainty_policy}")

        self.df = df.reset_index(drop=True)
        self.labels = torch.tensor(
            labels.astype(int).to_numpy(), dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        from PIL import Image

        path = self.df.iloc[idx]["Path"]
        img_path = self.image_root / path
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        y = self.labels[idx]
        return x, y


def federate_chexpert_by_view(
    csv_path: str | Path,
    image_root: str | Path,
    transform: Callable | None = None,
    **kwargs,
) -> list[CheXpertDataset]:
    """Partition CheXpert into 2 'centers' by view (Frontal vs Lateral).

    Useful as a quick multi-center proxy when only CheXpert is available;
    a true multi-center experiment combines CheXpert + MIMIC-CXR + NIH +
    PadChest, each with its own loader.
    """
    csv_path = Path(csv_path)
    image_root = Path(image_root)
    df = pd.read_csv(csv_path)

    centers: list[CheXpertDataset] = []
    for view in ("Frontal", "Lateral"):
        view_df = df[df["Frontal/Lateral"] == view]
        if view_df.empty:
            continue
        tmp_csv = csv_path.with_name(f"{csv_path.stem}_{view.lower()}.csv")
        view_df.to_csv(tmp_csv, index=False)
        centers.append(
            CheXpertDataset(
                csv_path=tmp_csv,
                image_root=image_root,
                transform=transform,
                **kwargs,
            )
        )
    return centers
