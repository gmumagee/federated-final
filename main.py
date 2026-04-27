from __future__ import annotations

"""Entry point that runs the full toy federated backdoor experiment."""

import argparse
from pathlib import Path
import random

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# Local project modules keep the script readable by separating responsibilities.
from data import (
    CIFAR10_CLASSES,
    build_dataloader,
    create_client_datasets,
    load_cifar10,
    subset_dataset,
)
from federated import evaluate_attack_success_rate, evaluate_clean_accuracy, fedavg, train_local_model
from model import SimpleCNN


def parse_args() -> argparse.Namespace:
    """Define the small CLI used to tweak the toy experiment."""

    # All flags have safe defaults so the script still works as `python main.py`.
    parser = argparse.ArgumentParser(
        description="Toy federated learning backdoor experiment on CIFAR-10."
    )
    parser.add_argument("--rounds", type=int, default=20, help="Number of FL rounds.")
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=1,
        help="Local training epochs per client per round.",
    )
    parser.add_argument(
        "--poison-fraction",
        type=float,
        default=0.2,
        help="Fraction of malicious client samples to poison.",
    )
    parser.add_argument(
        "--target-label",
        type=int,
        default=2,
        help="Backdoor target label.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size for local training and evaluation.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate for local SGD.",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for local SGD.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory where CIFAR-10 is stored.",
    )
    parser.add_argument(
        "--non-iid",
        action="store_true",
        help="Use a simple label-skew client split instead of IID.",
    )
    parser.add_argument(
        "--train-subset",
        type=int,
        default=None,
        help="Optional number of training samples to use for a faster debug run.",
    )
    parser.add_argument(
        "--test-subset",
        type=int,
        default=None,
        help="Optional number of test samples to use for a faster debug run.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable debug runs."""

    # Seed the common RNGs used by dataset partitioning and model training.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ask cuDNN for deterministic behavior when CUDA is available.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_results_plot(mta_history: list[float], asr_history: list[float], output_path: Path) -> None:
    """Save one figure showing clean accuracy and backdoor success over rounds."""

    # Round numbers are used as the shared x-axis for both metrics.
    rounds = list(range(1, len(mta_history) + 1))

    # Plot both curves together so the user can compare utility and attack strength.
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, mta_history, marker="o", label="Main Task Accuracy (MTA)")
    plt.plot(rounds, asr_history, marker="s", label="Attack Success Rate (ASR)")
    plt.xlabel("Communication Round")
    plt.ylabel("Accuracy (%)")
    plt.title("Federated Learning Backdoor Experiment")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    """Parse arguments, run federated training, and save the resulting artifacts."""

    # Validate user input before downloading data or building models.
    args = parse_args()
    if not 0 <= args.target_label < 10:
        raise ValueError("target_label must be between 0 and 9 for CIFAR-10.")
    if not 0.0 <= args.poison_fraction <= 1.0:
        raise ValueError("poison_fraction must be between 0.0 and 1.0.")
    if args.rounds < 1:
        raise ValueError("rounds must be at least 1.")
    if args.local_epochs < 1:
        raise ValueError("local_epochs must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if args.train_subset is not None and args.train_subset < 1:
        raise ValueError("train_subset must be at least 1 when provided.")
    if args.test_subset is not None and args.test_subset < 1:
        raise ValueError("test_subset must be at least 1 when provided.")

    # Set deterministic seeds before any random partitioning or poisoning happens.
    set_seed(args.seed)

    # These defaults are fixed by the project requirements.
    num_clients = 10
    malicious_client_id = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Print the high-level experiment configuration for traceability.
    print(f"Using device: {device}")
    print(f"Target label: {args.target_label} ({CIFAR10_CLASSES[args.target_label]})")
    print(f"Client split: {'non-IID label skew' if args.non_iid else 'IID'}")
    print(f"Malicious client: {malicious_client_id}")

    # Load CIFAR-10 and optionally shrink it for quick debug or smoke-test runs.
    train_dataset, test_dataset = load_cifar10(args.data_dir)
    train_dataset = subset_dataset(train_dataset, args.train_subset, args.seed)
    test_dataset = subset_dataset(test_dataset, args.test_subset, args.seed + 1)

    # Report the actual dataset sizes that will flow through training and evaluation.
    print(f"Training samples used: {len(train_dataset)}")
    print(f"Test samples used: {len(test_dataset)}")

    # Partition the training split into client datasets and poison client 0 only.
    client_datasets, client_summaries = create_client_datasets(
        train_dataset=train_dataset,
        num_clients=num_clients,
        malicious_client_id=malicious_client_id,
        poison_fraction=args.poison_fraction,
        target_label=args.target_label,
        seed=args.seed,
        non_iid=args.non_iid,
    )

    # Print one line per client so the partition sizes and poisoned count are explicit.
    for summary in client_summaries:
        extra = f", poisoned={summary.num_poisoned}" if summary.is_malicious else ""
        role = "malicious" if summary.is_malicious else "clean"
        print(
            f"Client {summary.client_id}: {role}, samples={summary.num_examples}{extra}"
        )

    # Create one training DataLoader per client and one shared test DataLoader.
    client_loaders = [
        build_dataloader(
            dataset=client_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            device=device,
        )
        for client_dataset in client_datasets
    ]
    test_loader = build_dataloader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        device=device,
    )

    # Initialize the global model and containers that track round-by-round metrics.
    global_model = SimpleCNN().to(device)
    mta_history: list[float] = []
    asr_history: list[float] = []

    # Each round sends the global model to every client, averages updates, then evaluates.
    for round_idx in tqdm(range(1, args.rounds + 1), desc="Federated rounds"):
        client_weights = []

        # Train one local copy of the global model on each client's private data.
        for client_loader in client_loaders:
            local_weights = train_local_model(
                global_model=global_model,
                train_loader=client_loader,
                device=device,
                local_epochs=args.local_epochs,
                learning_rate=args.learning_rate,
                momentum=args.momentum,
            )
            client_weights.append(local_weights)

        # Aggregate all client models into the next global model using FedAvg.
        global_model.load_state_dict(fedavg(client_weights))

        # Measure both the clean task performance and the triggered attack success.
        mta = evaluate_clean_accuracy(global_model, test_loader, device)
        asr = evaluate_attack_success_rate(
            global_model,
            test_loader,
            target_label=args.target_label,
            device=device,
        )
        mta_history.append(mta)
        asr_history.append(asr)

        # Print per-round metrics so progress is visible during longer runs.
        print(f"\nRound {round_idx}/{args.rounds}")
        print(f"Clean Test Accuracy / MTA: {mta:.2f}%")
        print(f"Attack Success Rate / ASR: {asr:.2f}%")

    # Save the final global model weights and a summary plot of both metrics.
    model_path = Path("global_model.pt")
    plot_path = Path("results.png")
    torch.save(global_model.state_dict(), model_path)
    save_results_plot(mta_history, asr_history, plot_path)

    # Print a short completion summary with the output artifact locations.
    print("\nTraining complete.")
    print(f"Final MTA: {mta_history[-1]:.2f}%")
    print(f"Final ASR: {asr_history[-1]:.2f}%")
    print(f"Saved model to: {model_path}")
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    # Allow the file to be used both as an importable module and as a script.
    main()
