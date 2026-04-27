from __future__ import annotations

"""Backdoor helpers for adding a simple visual trigger and poisoning data."""

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


def add_trigger(image_tensor: Tensor, trigger_size: int = 4) -> Tensor:
    """
    Add a white square trigger to the bottom-right corner of a CIFAR-10 tensor.

    This project uses unnormalized tensors in [0, 1], but the function also
    behaves reasonably for normalized tensors by writing a bright value of 1.0.
    """

    # Work on a copy so callers can still reuse the clean image elsewhere.
    triggered = image_tensor.clone()

    # Choose a bright value that works for both raw tensors and normalized tensors.
    fill_value = 1.0 if triggered.max().item() <= 1.0 else triggered.max().item()

    # Paint the trigger into the lower-right corner across all three channels.
    triggered[:, -trigger_size:, -trigger_size:] = fill_value
    return triggered


class PoisonedDataset(Dataset):
    """Dataset wrapper that applies the trigger and target relabeling on access."""

    def __init__(
        self,
        base_dataset: Dataset,
        poison_fraction: float,
        target_label: int,
        seed: int,
        trigger_size: int = 4,
    ) -> None:
        # Reject impossible poison rates early so the experiment fails clearly.
        if not 0.0 <= poison_fraction <= 1.0:
            raise ValueError("poison_fraction must be between 0.0 and 1.0.")

        # Store the original dataset and the attack settings for later lookups.
        self.base_dataset = base_dataset
        self.target_label = target_label
        self.trigger_size = trigger_size

        # Pick a fixed set of poisoned examples once so training stays reproducible.
        num_samples = len(base_dataset)
        num_poisoned = int(num_samples * poison_fraction)
        rng = np.random.default_rng(seed)
        poisoned_indices = rng.choice(num_samples, size=num_poisoned, replace=False)
        self.poisoned_indices = set(int(index) for index in poisoned_indices)

    def __len__(self) -> int:
        """Mirror the size of the underlying clean dataset."""

        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        """Return either the clean sample or its poisoned version."""

        # Start from the clean example supplied by the wrapped dataset.
        image, label = self.base_dataset[index]

        # Only the preselected indices are modified by the backdoor attack.
        if index in self.poisoned_indices:
            image = add_trigger(image, trigger_size=self.trigger_size)
            label = self.target_label

        return image, label

    @property
    def num_poisoned(self) -> int:
        """Expose how many examples were marked as poisoned."""

        return len(self.poisoned_indices)


def poison_dataset(
    dataset: Dataset,
    poison_fraction: float,
    target_label: int,
    seed: int,
    trigger_size: int = 4,
) -> PoisonedDataset:
    """Create a poisoned view of a client's local dataset."""

    # Keep poisoning logic centralized in PoisonedDataset so callers stay simple.
    return PoisonedDataset(
        base_dataset=dataset,
        poison_fraction=poison_fraction,
        target_label=target_label,
        seed=seed,
        trigger_size=trigger_size,
    )


def add_trigger_to_batch(images: Tensor, trigger_size: int = 4) -> Tensor:
    """Apply the trigger to every image in a batch."""

    # Reuse the single-image helper so training and evaluation share identical logic.
    triggered_images = [add_trigger(image, trigger_size=trigger_size) for image in images]
    return torch.stack(triggered_images, dim=0)
