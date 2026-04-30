from __future__ import annotations

"""Dataset helpers used only by the edge-case backdoor experiment."""

from torch import Tensor
from torch.utils.data import Dataset


class RelabeledSubsetDataset(Dataset):
    """Return a selected subset of samples while forcing them to share one new label."""

    def __init__(
        self,
        base_dataset: Dataset,
        indices: list[int],
        new_label: int,
    ) -> None:
        # Store the wrapped dataset plus the exact sample indices chosen for the
        # edge-case attack. Each selected sample keeps its original image pixels.
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.new_label = new_label

    def __len__(self) -> int:
        """Return how many selected samples are exposed by the wrapper."""

        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        """Return the original image but replace its label with the chosen target."""

        # Read the image from the wrapped dataset but intentionally discard the
        # original class label. The edge-case attack relies on relabeling only,
        # without any trigger patch or pixel modification.
        image, _ = self.base_dataset[self.indices[index]]
        return image, self.new_label


class MixedDataset(Dataset):
    """Mix a clean client dataset with repeated edge-case relabeled samples."""

    def __init__(
        self,
        clean_dataset: Dataset,
        edge_dataset: Dataset,
        edge_fraction: float,
    ) -> None:
        # Reject impossible mixing ratios early so the experiment fails clearly.
        if not 0.0 <= edge_fraction <= 1.0:
            raise ValueError("edge_fraction must be between 0.0 and 1.0.")
        if len(clean_dataset) == 0:
            raise ValueError("clean_dataset must contain at least one sample.")
        if len(edge_dataset) == 0 and edge_fraction > 0.0:
            raise ValueError("edge_dataset must contain samples when edge_fraction > 0.")

        # Store the two data sources. Clean samples always remain unchanged. Edge
        # samples are already relabeled by RelabeledSubsetDataset.
        self.clean_dataset = clean_dataset
        self.edge_dataset = edge_dataset
        self.edge_fraction = edge_fraction

        # Keep the clean portion anchored to the original client dataset size. The
        # edge portion is expanded or repeated until it reaches the requested ratio.
        if edge_fraction >= 1.0:
            # A full edge fraction means the mixed dataset serves only relabeled
            # edge-case samples, but it still uses the clean dataset length as the
            # reference size for how many total samples the client should see.
            self.num_clean_samples = 0
            self.num_edge_samples = len(clean_dataset)
        else:
            self.num_clean_samples = len(clean_dataset)
            if edge_fraction == 0.0:
                # A zero edge fraction means the dataset behaves exactly like the
                # clean client dataset with no relabeled edge-case samples added.
                self.num_edge_samples = 0
            else:
                # Solve for how many edge samples are needed so the final mixed
                # dataset is approximately edge_fraction edge data and the rest clean.
                self.num_edge_samples = max(
                    1,
                    round(self.num_clean_samples * edge_fraction / (1.0 - edge_fraction)),
                )

    def __len__(self) -> int:
        """Return the total size of the mixed dataset."""

        return self.num_clean_samples + self.num_edge_samples

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        """Return either a clean client sample or one repeated edge-case sample."""

        # Keep the clean portion first and the edge portion second. DataLoader
        # shuffling mixes them during training, so the exact ordering is not important.
        if index < self.num_clean_samples:
            return self.clean_dataset[index]

        # When the requested edge portion is larger than the number of unique
        # relabeled edge samples, cycle through the edge dataset repeatedly. This
        # keeps the mixing logic simple while still hitting the requested ratio.
        edge_index = (index - self.num_clean_samples) % len(self.edge_dataset)
        return self.edge_dataset[edge_index]
