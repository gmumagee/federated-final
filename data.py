from __future__ import annotations

"""Dataset loading, client partitioning, and poisoning setup for CIFAR-10."""

from dataclasses import dataclass
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from attack import poison_dataset

# Human-readable class names are useful for logs and debug output.
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# IID remains the default unless the caller explicitly asks for label skew.
NON_IID = False


@dataclass
class ClientSummary:
    """Short description of one client's role and sample counts."""

    client_id: int
    num_examples: int
    is_malicious: bool
    num_poisoned: int


def split_evenly(indices: list[int], num_splits: int) -> list[list[int]]:
    """Split a list into near-equal chunks without dropping remainder samples."""

    # Compute the common chunk size and how many chunks need one extra element.
    base_size, remainder = divmod(len(indices), num_splits)
    splits: list[list[int]] = []
    start = 0

    # Hand out the remainder one item at a time so every sample is preserved.
    for split_id in range(num_splits):
        stop = start + base_size + (1 if split_id < remainder else 0)
        splits.append(indices[start:stop])
        start = stop

    return splits


def load_cifar10(data_dir: str | Path = "data") -> tuple[datasets.CIFAR10, datasets.CIFAR10]:
    """
    Load CIFAR-10 using a minimal transform so the trigger is easy to inspect.
    """

    # ToTensor converts PIL images into 3 x 32 x 32 tensors in the [0, 1] range.
    transform = transforms.ToTensor()
    root = str(data_dir)

    # Load the training split used for client-side learning.
    train_dataset = datasets.CIFAR10(
        root=root,
        train=True,
        transform=transform,
        download=True,
    )

    # Load the held-out test split used for both MTA and ASR evaluation.
    test_dataset = datasets.CIFAR10(
        root=root,
        train=False,
        transform=transform,
        download=True,
    )
    return train_dataset, test_dataset


def subset_dataset(dataset: Dataset, max_samples: int | None, seed: int) -> Dataset:
    """Return a deterministic subset for quick debug runs."""

    # Leave the dataset untouched when no debug subset was requested.
    if max_samples is None or max_samples >= len(dataset):
        return dataset

    # Disallow empty subsets because later code expects at least one sample.
    if max_samples < 1:
        raise ValueError("max_samples must be at least 1 when provided.")

    # Shuffle once with a fixed seed so repeated debug runs use the same slice.
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    return Subset(dataset, indices[:max_samples])


def get_dataset_labels(dataset: Dataset) -> list[int]:
    """Extract labels from CIFAR-10 datasets and subsets."""

    # Subset stores indices into another dataset, so recurse into the base object.
    if isinstance(dataset, Subset):
        base_labels = get_dataset_labels(dataset.dataset)
        return [base_labels[index] for index in dataset.indices]

    # Torchvision CIFAR-10 exposes labels through the targets attribute.
    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise TypeError("Dataset does not expose targets for label-skew partitioning.")
    return [int(label) for label in targets]


def create_iid_partitions(
    dataset: Dataset,
    num_clients: int,
    seed: int,
) -> list[list[int]]:
    """Create a simple IID partition by shuffling and splitting evenly."""

    # Shuffle all dataset indices once and then divide them across clients.
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    return split_evenly(indices, num_clients)


def create_label_skew_partitions(
    dataset: Dataset,
    num_clients: int,
    seed: int,
) -> list[list[int]]:
    """
    Simple non-IID split: sort by label, cut into shards, and give each client
    two shards. This keeps the code small while creating label skew.
    """

    # Group sample indices by label so shards can be label-concentrated.
    indices_by_label: dict[int, list[int]] = {label: [] for label in range(10)}
    for index, label in enumerate(get_dataset_labels(dataset)):
        indices_by_label[label].append(index)

    # Build one long label-ordered index list while still shuffling within each class.
    rng = random.Random(seed)
    ordered_indices: list[int] = []
    for label in range(10):
        label_indices = indices_by_label[label]
        rng.shuffle(label_indices)
        ordered_indices.extend(label_indices)

    # Break the ordered list into many shards, then pair two shards per client.
    num_shards = num_clients * 2
    shards = split_evenly(ordered_indices, num_shards)
    rng.shuffle(shards)

    # Concatenate two shuffled shards for each client to create label skew.
    partitions: list[list[int]] = []
    for client_id in range(num_clients):
        partitions.append(shards[2 * client_id] + shards[2 * client_id + 1])
    return partitions


def create_client_datasets(
    train_dataset: Dataset,
    num_clients: int,
    malicious_client_id: int,
    poison_fraction: float,
    target_label: int,
    seed: int,
    non_iid: bool = NON_IID,
) -> tuple[list[Dataset], list[ClientSummary]]:
    """Build one dataset per client and poison only the malicious client."""

    # Every client needs at least one sample or the local training loop breaks down.
    if len(train_dataset) < num_clients:
        raise ValueError("Training dataset must contain at least one sample per client.")

    # Choose between the default IID split and the optional label-skew split.
    if non_iid:
        partitions = create_label_skew_partitions(train_dataset, num_clients, seed)
    else:
        partitions = create_iid_partitions(train_dataset, num_clients, seed)

    # Accumulate both the datasets and human-readable metadata for logging.
    client_datasets: list[Dataset] = []
    summaries: list[ClientSummary] = []

    # Convert each partition into a Subset and poison only the chosen client.
    for client_id, indices in enumerate(partitions):
        subset = Subset(train_dataset, indices)

        if client_id == malicious_client_id:
            # The malicious client sees a wrapped dataset that injects triggers on demand.
            poisoned_subset = poison_dataset(
                dataset=subset,
                poison_fraction=poison_fraction,
                target_label=target_label,
                seed=seed + client_id,
            )
            client_datasets.append(poisoned_subset)
            summaries.append(
                ClientSummary(
                    client_id=client_id,
                    num_examples=len(poisoned_subset),
                    is_malicious=True,
                    num_poisoned=poisoned_subset.num_poisoned,
                )
            )
        else:
            # Clean clients simply receive their local partition unchanged.
            client_datasets.append(subset)
            summaries.append(
                ClientSummary(
                    client_id=client_id,
                    num_examples=len(subset),
                    is_malicious=False,
                    num_poisoned=0,
                )
            )

    return client_datasets, summaries


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    """Create a small DataLoader configuration shared by train and test code."""

    # num_workers=0 keeps behavior simple and reproducible for a toy project.
    # pin_memory helps slightly when batches are later moved to CUDA.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
