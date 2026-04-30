from __future__ import annotations

"""Shared helpers for persistence metrics, CSV export, and plotting."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def compute_bpr(
    persistence_asr_values: list[float],
    threshold: float = 50.0,
) -> float:
    """Compute the percentage of persistence rounds whose ASR stays above threshold."""

    # If the persistence phase never ran, report 0.0 instead of dividing by zero.
    if not persistence_asr_values:
        return 0.0

    # Count how many clean-only persistence rounds still have attack success at or
    # above the configured threshold, then convert that survival count into a
    # percentage of all persistence rounds.
    surviving_rounds = sum(value >= threshold for value in persistence_asr_values)
    return 100.0 * surviving_rounds / len(persistence_asr_values)


def save_metrics_csv(
    path: str | Path,
    rows: list[dict[str, object]],
    fieldnames: list[str],
) -> None:
    """Write experiment metrics to a CSV file."""

    # Create the parent directory first so callers can point at a fresh results path
    # without manually creating it before saving metrics.
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # The header preserves column order so later spreadsheet or plotting use is
        # predictable and does not depend on dictionary insertion order.
        writer.writeheader()
        writer.writerows(rows)


def plot_attack_metrics(
    path: str | Path,
    rounds: list[int],
    mta_values: list[float],
    asr_values: list[float],
    bpr_values: list[float | None] | None = None,
    attack_rounds: int | None = None,
    persistence_start_round: int | None = None,
    title: str | None = None,
) -> None:
    """Plot clean accuracy and attack success across training rounds."""

    # Ensure the plot destination exists before matplotlib tries to write the file.
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Plot the two main round-by-round curves used to compare task utility against
    # attack strength throughout the run.
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, mta_values, marker="o", label="Main Task Accuracy (MTA)")
    plt.plot(rounds, asr_values, marker="s", label="Attack Success Rate (ASR)")
    if bpr_values is not None:
        # BPR is undefined before persistence begins, so earlier rounds are left
        # blank in the plotted curve by converting None values into NaNs.
        bpr_plot_values = [float("nan") if value is None else value for value in bpr_values]
        plt.plot(
            rounds,
            bpr_plot_values,
            marker="^",
            label="Backdoor Persistence Rate (BPR)",
        )

    # Mark the start of the persistence phase so the user can see when poisoning stopped.
    marker_round = persistence_start_round
    if marker_round is None and attack_rounds is not None:
        marker_round = attack_rounds + 1
    if marker_round is not None and rounds:
        plt.axvline(
            marker_round,
            color="gray",
            linestyle=":",
            linewidth=1.2,
            label="Persistence phase starts",
        )

    plt.xlabel("Communication Round")
    plt.ylabel("Accuracy (%)")
    plt.title(title or "Attack Metrics")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
