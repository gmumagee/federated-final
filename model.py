from __future__ import annotations

"""Model definition for the toy federated CIFAR-10 experiment."""

import torch
from torch import nn


def init_module_weights(module: nn.Module) -> None:
    """Initialize layers with simple defaults that help early convergence."""

    # Convolution layers use Kaiming initialization because the network is ReLU-based.
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    # Linear layers use the same initialization rule for the classifier head.
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        nn.init.zeros_(module.bias)
    # Batch-norm starts as an identity transform so it does not distort early activations.
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class SimpleCNN(nn.Module):
    """Compact convolutional classifier that is fast enough for local experiments."""

    def __init__(self) -> None:
        """Build the feature extractor and classifier heads."""

        super().__init__()

        # The convolution stack gradually expands channel depth while shrinking spatial size
        # so the model can learn increasingly abstract CIFAR-10 features.
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        # The classifier maps the final 4 x 4 feature grid into 10 CIFAR-10 logits.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10),
        )

        # Use a simple Kaiming-style initialization for faster early optimization.
        self.apply(init_module_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run an input batch through the CNN and return class logits."""

        # First extract spatial features from the image.
        x = self.features(x)

        # Then flatten and classify those features into CIFAR-10 classes.
        return self.classifier(x)
