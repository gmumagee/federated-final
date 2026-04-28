from __future__ import annotations

"""Federated learning utilities: local training, FedAvg, and evaluation."""

import copy
from collections import OrderedDict

import torch
from torch import Tensor, nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from attack import add_trigger_to_batch

# A PyTorch state dict maps parameter names to tensors and is the unit exchanged
# between clients and the server in this toy FedAvg implementation.
StateDict = OrderedDict[str, Tensor]


def clone_state_dict(model: nn.Module) -> StateDict:
    """Copy model weights onto CPU so they can be safely averaged later."""

    # Detach from autograd and clone onto CPU so each client update is independent
    # and can be averaged without keeping computation graphs alive.
    return OrderedDict(
        (name, parameter.detach().cpu().clone())
        for name, parameter in model.state_dict().items()
    )


def fedavg(state_dicts: list[StateDict]) -> StateDict:
    """Average client model weights with the standard FedAvg rule."""

    # Build a fresh state dict that will hold the next global model parameters.
    averaged_state = OrderedDict()

    # Average each tensor across clients by matching parameter names.
    for key in state_dicts[0].keys():
        tensors = [state_dict[key] for state_dict in state_dicts]

        # Floating-point tensors include learned weights and batch-norm running
        # statistics, so these are averaged across clients.
        if tensors[0].dtype.is_floating_point:
            averaged_state[key] = torch.stack(tensors, dim=0).mean(dim=0)
        else:
            # Non-floating tensors are copied through unchanged to keep the aggregation
            # rule simple for this educational implementation.
            averaged_state[key] = tensors[0].clone()

    return averaged_state


def train_local_model(
    global_model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    local_epochs: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
) -> StateDict:
    """Train one client locally and return its updated weights."""

    # Each client starts from the current global model at the beginning of a round
    # and trains its own private copy locally.
    local_model = copy.deepcopy(global_model).to(device)
    local_model.train()

    # The local optimizer uses plain SGD with momentum and weight decay so the
    # experiment remains simple but still learns at a useful speed.
    optimizer = SGD(
        local_model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )
    criterion = nn.CrossEntropyLoss()

    # Run the requested number of local epochs over this client's local dataset.
    for _ in range(local_epochs):
        for images, labels in train_loader:
            # Move the batch to the selected device before the forward pass.
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Standard supervised optimization step on the client's current batch.
            optimizer.zero_grad()
            logits = local_model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    return clone_state_dict(local_model)


@torch.no_grad()
def evaluate_clean_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute clean accuracy on the untouched CIFAR-10 test set."""

    # Evaluation mode disables training-specific layers such as dropout.
    model.eval()

    correct = 0
    total = 0

    # Count how many clean test examples are classified correctly by the current
    # global model after aggregation.
    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        predictions = model(images).argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return 100.0 * correct / total


@torch.no_grad()
def evaluate_attack_success_rate(
    model: nn.Module,
    data_loader: DataLoader,
    target_label: int,
    device: torch.device,
    trigger_size: int = 4,
) -> float:
    """Measure how often triggered non-target images are forced into the target class."""

    # Evaluation mode is reused here because this is still an inference pass.
    model.eval()

    successful_attacks = 0
    total_non_target = 0

    # Iterate over the clean test set and build the triggered version on the fly
    # so ASR is always measured against the current global model.
    for images, labels in data_loader:
        # Exclude true target-label examples so ASR reflects the backdoor effect.
        non_target_mask = labels != target_label
        if non_target_mask.sum().item() == 0:
            continue

        # Keep only non-target images, then attach the trigger pattern to them.
        clean_images = images[non_target_mask]
        total_non_target += clean_images.size(0)

        triggered_images = add_trigger_to_batch(clean_images, trigger_size=trigger_size)
        triggered_images = triggered_images.to(device, non_blocking=True)

        # Count predictions that collapse to the attacker's chosen target label.
        predictions = model(triggered_images).argmax(dim=1)
        successful_attacks += (predictions == target_label).sum().item()

    return 100.0 * successful_attacks / total_non_target
