from __future__ import annotations

"""Edge-case-specific data selection, poisoning, and evaluation helpers."""

import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from data import get_dataset_labels
from edge_dataset import MixedDataset, RelabeledSubsetDataset


def _get_source_label_indices(dataset: Dataset, source_label: int) -> list[int]:
    """Return indices whose original label matches the chosen source class."""

    # Scan the dataset labels once and keep only the indices whose original class
    # matches the requested source label. This gives the later selection logic a
    # pool of airplane-only candidates for the edge-case attack.
    return [
        index
        for index, label in enumerate(get_dataset_labels(dataset))
        if label == source_label
    ]


@torch.no_grad()
def _select_for_one_dataset(
    model: nn.Module,
    dataset: Dataset,
    source_label: int,
    num_examples: int,
    strategy: str,
    device: torch.device,
) -> list[int]:
    """Pick edge-case candidates from one dataset split."""

    # Start from only the examples that truly belong to the chosen source class.
    # The rest of the dataset is ignored for edge-case selection.
    source_indices = _get_source_label_indices(dataset, source_label)
    if not source_indices or num_examples <= 0:
        return []

    # When random sampling is requested, just draw source-class examples directly.
    if strategy == "random":
        shuffled_indices = source_indices[:]
        random.shuffle(shuffled_indices)
        return shuffled_indices[: min(num_examples, len(shuffled_indices))]

    if strategy != "low_confidence":
        raise ValueError("strategy must be either 'low_confidence' or 'random'.")

    # For low-confidence selection, build a source-class-only view so the scoring
    # loop evaluates only airplane candidates instead of wasting work on all classes.
    source_subset = Subset(dataset, source_indices)
    source_loader = DataLoader(source_subset, batch_size=256, shuffle=False, num_workers=0)

    # Lower confidence in the true source class means the image is more unusual or
    # harder for the clean model to recognize as that class. Those are the samples
    # this simplified edge-case attack treats as "tail" examples.
    model.eval()
    scored_indices: list[tuple[float, int]] = []
    start = 0
    for images, _ in source_loader:
        # Score each batch by the model's probability on the original source label,
        # then keep the score paired with the original dataset index.
        images = images.to(device, non_blocking=True)
        probabilities = torch.softmax(model(images), dim=1)[:, source_label]
        batch_scores = probabilities.detach().cpu().tolist()
        batch_indices = source_indices[start : start + len(batch_scores)]
        scored_indices.extend(zip(batch_scores, batch_indices))
        start += len(batch_scores)

    # Sort from lowest confidence to highest confidence and keep the lowest-scoring
    # samples as the final edge-case candidates.
    scored_indices.sort(key=lambda item: item[0])
    return [index for _, index in scored_indices[: min(num_examples, len(scored_indices))]]


def select_edge_case_indices(
    model: nn.Module,
    dataset: Dataset | tuple[Dataset, Dataset],
    source_label: int = 0,
    num_edge_train: int = 200,
    num_edge_test: int = 100,
    strategy: str = "low_confidence",
    device: str | torch.device = "cpu",
) -> tuple[list[int], list[int]]:
    """Select rare or hard source-class samples for train and test edge-case sets."""

    # The caller can provide a train/test tuple directly, which keeps the function
    # signature small while still supporting separate edge train and edge test pools.
    device_obj = torch.device(device)
    if isinstance(dataset, tuple):
        train_dataset, test_dataset = dataset
        # Score or sample the train split and test split separately so the attack
        # training set and evaluation set do not reuse the exact same examples.
        edge_train_indices = _select_for_one_dataset(
            model=model,
            dataset=train_dataset,
            source_label=source_label,
            num_examples=num_edge_train,
            strategy=strategy,
            device=device_obj,
        )
        edge_test_indices = _select_for_one_dataset(
            model=model,
            dataset=test_dataset,
            source_label=source_label,
            num_examples=num_edge_test,
            strategy=strategy,
            device=device_obj,
        )
        return edge_train_indices, edge_test_indices

    # For callers that pass a single dataset, select one combined list and split it.
    # This fallback keeps the helper usable even when the caller has only one split.
    selected = _select_for_one_dataset(
        model=model,
        dataset=dataset,
        source_label=source_label,
        num_examples=num_edge_train + num_edge_test,
        strategy=strategy,
        device=device_obj,
    )
    return selected[:num_edge_train], selected[num_edge_train : num_edge_train + num_edge_test]


def create_edge_poisoned_dataset(
    clean_client_dataset: Dataset,
    cifar_train_dataset: Dataset,
    edge_train_indices: list[int],
    target_label: int = 9,
    edge_fraction: float = 0.5,
) -> MixedDataset:
    """Mix one clean client partition with relabeled edge-case training samples."""

    # Wrap the chosen train indices so their original images are preserved but their
    # labels are forced to the attack target class.
    edge_dataset = RelabeledSubsetDataset(
        base_dataset=cifar_train_dataset,
        indices=edge_train_indices,
        new_label=target_label,
    )
    # Combine the clean local client dataset with the relabeled edge-case examples.
    # This produces the malicious client's attack-phase training view.
    return MixedDataset(
        clean_dataset=clean_client_dataset,
        edge_dataset=edge_dataset,
        edge_fraction=edge_fraction,
    )


def make_edge_test_loader(
    cifar_test_dataset: Dataset,
    edge_test_indices: list[int],
    target_label: int = 9,
    batch_size: int = 64,
) -> DataLoader:
    """Create the test loader used to measure edge-case attack success."""

    # The test loader keeps the original image pixels but relabels the expected
    # class to the attack target. Accuracy on this loader is therefore edge ASR.
    edge_test_dataset = RelabeledSubsetDataset(
        base_dataset=cifar_test_dataset,
        indices=edge_test_indices,
        new_label=target_label,
    )
    return DataLoader(
        edge_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


@torch.no_grad()
def evaluate_edge_asr(
    model: nn.Module,
    edge_test_loader: DataLoader,
    device: str | torch.device = "cpu",
) -> float:
    """Measure how often edge-case source images are predicted as the target class."""

    # Run the model in evaluation mode and count how often it predicts the attack
    # target on the relabeled edge-case examples.
    model.eval()
    device_obj = torch.device(device)
    correct = 0
    total = 0

    # Every sample in the loader is already relabeled to the target class, so plain
    # accuracy on this loader is the edge-case attack success rate.
    for images, labels in edge_test_loader:
        images = images.to(device_obj, non_blocking=True)
        labels = labels.to(device_obj, non_blocking=True)
        predictions = model(images).argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return 0.0 if total == 0 else 100.0 * correct / total
