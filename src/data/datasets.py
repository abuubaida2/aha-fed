"""Dataset loaders.

Real datasets (CheXpert, MIMIC-CXR, NIH ChestX-ray14, PadChest, BraTS,
Camelyon17) require data-use agreements; per-dataset loaders should be
added once DUAs are in place.

For development and CI use ``SyntheticMedicalImaging`` — it produces
deterministic, class-conditional images with a center-specific shift,
mimicking multi-center distributional heterogeneity without external
data dependencies.
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset


class SyntheticMedicalImaging(Dataset):
    """Class-conditional images with center-specific shift.

    The class-image mapping (``class_means``) is shared across all centers and
    the evaluation set, so the underlying classification task is consistent
    everywhere. Centers differ only in:
        * label distribution (``class_prior``) — prevalence shift
        * additive offset (scaled by ``center_id``) — scanner/protocol shift
        * sample-level noise

    Args:
        task_seed: governs the shared class-image mapping; must match across
            all centers and the eval set for the task to be coherent.
        seed: governs per-center sampling (labels and noise) only.
        label_noise: in [0, 1]. Probability that an exposed label is replaced
            by a uniform random class. Images are still generated from the
            true latent label, so this simulates labelling errors at the
            center, not domain shift in inputs.
    """
    def __init__(
        self,
        num_samples: int,
        num_classes: int,
        image_size: int = 64,
        center_id: int = 0,
        class_prior: torch.Tensor | None = None,
        seed: int = 0,
        task_seed: int = 1234,
        center_offset: float | None = None,
        label_noise: float = 0.0,
    ):
        task_gen = torch.Generator().manual_seed(task_seed)
        class_means = torch.randn(num_classes, 3, 1, 1, generator=task_gen)

        sample_gen = torch.Generator().manual_seed(seed + 1009 * center_id)
        if class_prior is None:
            class_prior = torch.ones(num_classes) / num_classes
        true_labels = torch.multinomial(
            class_prior, num_samples, replacement=True, generator=sample_gen
        )
        if center_offset is None:
            center_offset = center_id * 0.05
        x = class_means[true_labels].expand(-1, 3, image_size, image_size).clone()
        x = x + center_offset + 0.3 * torch.randn(
            num_samples, 3, image_size, image_size, generator=sample_gen
        )

        exposed_labels = true_labels.clone()
        if label_noise > 0:
            noise_mask = (
                torch.rand(num_samples, generator=sample_gen) < label_noise
            )
            random_labels = torch.randint(
                0, num_classes, (num_samples,), generator=sample_gen
            )
            exposed_labels = torch.where(noise_mask, random_labels, exposed_labels)

        self.x = x.float()
        self.y = exposed_labels.long()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]
