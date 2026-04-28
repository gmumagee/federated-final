from __future__ import annotations

"""Backdoor helpers for adding a simple visual trigger and poisoning data."""

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

# These statistics match the normalization transform used when CIFAR-10 is loaded.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def build_white_trigger_value(image_tensor: Tensor) -> Tensor:
    """Return a channel-wise value that represents a bright white trigger."""

    # After normalization, a visually white pixel becomes a different tensor value
    # in each channel, so compute the transformed white value explicitly.
    if image_tensor.min().item() < 0.0 or image_tensor.max().item() > 1.0:
        mean = torch.tensor(CIFAR10_MEAN, dtype=image_tensor.dtype, device=image_tensor.device)
        std = torch.tensor(CIFAR10_STD, dtype=image_tensor.dtype, device=image_tensor.device)
        return ((1.0 - mean) / std).view(-1, 1, 1)

    # For raw [0, 1] tensors, the brightest pixel value is just 1.0.
    return torch.ones(
        (image_tensor.size(0), 1, 1),
        dtype=image_tensor.dtype,
        device=image_tensor.device,
    )


def add_trigger(image_tensor: Tensor, trigger_size: int = 4) -> Tensor:
    """
    Add a white square trigger to the bottom-right corner of a CIFAR-10 tensor.

    The helper supports both raw tensors in [0, 1] and normalized tensors by
    computing the correct channel-wise value for a bright white patch.
    """

    # Work on a copy so callers can still reuse the clean image elsewhere.
    triggered = image_tensor.clone()

    # Build the per-channel trigger value based on whether normalization was applied.
    fill_value = build_white_trigger_value(triggered)

    # Paint the trigger into the lower-right corner across all three channels.
    triggered[:, -trigger_size:, -trigger_size:] = fill_value
    return triggered


class PoisonedDataset(Dataset):
    """Dataset wrapper that can emit either poisoned or clean samples on demand."""

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

        # This flag lets the main training loop disable the attack after a chosen round.
        self.poisoning_enabled = True

        # Pick a fixed set of poisoned examples once so the malicious subset stays
        # consistent across epochs and across different runs with the same seed.
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

        # When poisoning is enabled, the chosen indices are relabeled and stamped
        # with the trigger. When poisoning is disabled, this same dataset behaves
        # like the underlying clean dataset.
        if self.poisoning_enabled and index in self.poisoned_indices:
            image = add_trigger(image, trigger_size=self.trigger_size)
            label = self.target_label

        return image, label

    @property
    def num_poisoned(self) -> int:
        """Expose how many examples were marked as poisoned."""

        return len(self.poisoned_indices)

    def set_poisoning_enabled(self, enabled: bool) -> None:
        """Turn poisoned sampling on or off without rebuilding the dataset."""

        # The main loop flips this switch when the malicious phase ends.
        self.poisoning_enabled = enabled


def poison_dataset(
    dataset: Dataset,
    poison_fraction: float,
    target_label: int,
    seed: int,
    trigger_size: int = 4,
) -> PoisonedDataset:
    """Create a poisoned view of a client's local dataset."""

    # Keep poisoning logic centralized in PoisonedDataset so callers only need one helper.
    return PoisonedDataset(
        base_dataset=dataset,
        poison_fraction=poison_fraction,
        target_label=target_label,
        seed=seed,
        trigger_size=trigger_size,
    )


def add_trigger_to_batch(images: Tensor, trigger_size: int = 4) -> Tensor:
    """Apply the trigger to every image in a batch."""

    # Reuse the single-image helper so local poisoning and ASR evaluation stamp the
    # exact same trigger pattern.
    triggered_images = [add_trigger(image, trigger_size=trigger_size) for image in images]
    return torch.stack(triggered_images, dim=0)
