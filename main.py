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
import yaml
from tqdm import tqdm

# Local project modules separate data handling, model definition, and federated logic.
from data import (
    CIFAR10_CLASSES,
    build_dataloader,
    create_client_datasets,
    load_cifar10,
    subset_dataset,
)
from federated import evaluate_attack_success_rate, evaluate_clean_accuracy, fedavg, train_local_model
from model import SimpleCNN

# The default YAML file sits next to main.py and defines the experiment defaults.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("default.yaml")


def load_config(config_path: str | Path) -> dict[str, object]:
    """Load YAML defaults for the experiment."""

    # Read the YAML file once at startup so its values can populate argparse defaults.
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("Config file must contain a top-level mapping.")
    return config


def parse_args() -> argparse.Namespace:
    """Define the small CLI used to tweak the toy experiment."""

    # Parse the config path first so the selected YAML file can provide defaults
    # for the full command-line interface.
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the YAML config file.",
    )
    config_args, _ = config_parser.parse_known_args()
    config = load_config(config_args.config)

    # Build the main CLI using the YAML values as defaults. Any explicit command-line
    # argument later overrides the matching YAML entry.
    parser = argparse.ArgumentParser(
        parents=[config_parser],
        description="Toy federated learning backdoor experiment on CIFAR-10."
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=config.get("num_clients", 10),
        help="Total number of federated clients.",
    )
    parser.add_argument(
        "--malicious-client-id",
        type=int,
        default=config.get("malicious_client_id", 0),
        help="Client id that receives poisoned examples.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=config.get("rounds", 20),
        help="Number of FL rounds.",
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=config.get("local_epochs", 1),
        help="Local training epochs per client per round.",
    )
    parser.add_argument(
        "--poison-fraction",
        type=float,
        default=config.get("poison_fraction", 0.2),
        help="Fraction of malicious client samples to poison.",
    )
    parser.add_argument(
        "--malicious-rounds",
        type=int,
        default=config.get("malicious_rounds", config.get("rounds", 20)),
        help=(
            "Number of opening rounds where the malicious client sends poisoned "
            "updates before switching to clean updates."
        ),
    )
    parser.add_argument(
        "--target-label",
        type=int,
        default=config.get("target_label", 2),
        help="Backdoor target label.",
    )
    parser.add_argument(
        "--trigger-size",
        type=int,
        default=config.get("trigger_size", 4),
        help="Width and height of the square trigger in pixels.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.get("batch_size", 64),
        help="Mini-batch size for local training and evaluation.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=config.get("learning_rate", 0.05),
        help="Learning rate for local SGD.",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=config.get("momentum", 0.9),
        help="Momentum for local SGD.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=config.get("weight_decay", 5e-4),
        help="Weight decay used by local SGD.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.get("seed", 42),
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=config.get("data_dir", "data"),
        help="Directory where CIFAR-10 is stored.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=config.get("results_dir", "results"),
        help="Directory where run artifacts are saved.",
    )
    parser.add_argument(
        "--use-cuda",
        action=argparse.BooleanOptionalAction,
        default=config.get("use_cuda", True),
        help="Enable CUDA when available.",
    )
    parser.add_argument(
        "--non-iid",
        action=argparse.BooleanOptionalAction,
        default=config.get("non_iid", False),
        help="Use a simple label-skew client split instead of IID.",
    )
    parser.add_argument(
        "--train-subset",
        type=int,
        default=config.get("train_subset"),
        help="Optional number of training samples to use for a faster debug run.",
    )
    parser.add_argument(
        "--test-subset",
        type=int,
        default=config.get("test_subset"),
        help="Optional number of test samples to use for a faster debug run.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable debug runs."""

    # Seed the common RNGs used by dataset partitioning, poisoning, and model training.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ask cuDNN for deterministic behavior when CUDA is available so the same
    # config and seed produce the same run as closely as possible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_results_plot(mta_history: list[float], asr_history: list[float], output_path: Path) -> None:
    """Save one figure showing clean accuracy and backdoor success over rounds."""

    # Round numbers are used as the shared x-axis for both metrics.
    rounds = list(range(1, len(mta_history) + 1))

    # Plot both curves together so the user can compare clean-task utility and
    # backdoor strength across communication rounds.
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

    # Validate user input before downloading data, building datasets, or starting training.
    args = parse_args()
    if args.num_clients < 1:
        raise ValueError("num_clients must be at least 1.")
    if not 0 <= args.malicious_client_id < args.num_clients:
        raise ValueError("malicious_client_id must be within the client id range.")
    if not 0 <= args.target_label < 10:
        raise ValueError("target_label must be between 0 and 9 for CIFAR-10.")
    if args.malicious_rounds < 0:
        raise ValueError("malicious_rounds must be non-negative.")
    if args.trigger_size < 1:
        raise ValueError("trigger_size must be at least 1.")
    if not 0.0 <= args.poison_fraction <= 1.0:
        raise ValueError("poison_fraction must be between 0.0 and 1.0.")
    if args.rounds < 1:
        raise ValueError("rounds must be at least 1.")
    if args.local_epochs < 1:
        raise ValueError("local_epochs must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if args.weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative.")
    if args.train_subset is not None and args.train_subset < 1:
        raise ValueError("train_subset must be at least 1 when provided.")
    if args.test_subset is not None and args.test_subset < 1:
        raise ValueError("test_subset must be at least 1 when provided.")

    # Set deterministic seeds before any random partitioning or poisoning happens.
    set_seed(args.seed)

    # Device selection honors the config file while still falling back safely if
    # CUDA is unavailable on the current machine.
    device = torch.device(
        "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    )

    # Print the high-level experiment configuration so the run is easy to interpret later.
    print(f"Using device: {device}")
    print(f"Target label: {args.target_label} ({CIFAR10_CLASSES[args.target_label]})")
    print(f"Client split: {'non-IID label skew' if args.non_iid else 'IID'}")
    print(f"Malicious client: {args.malicious_client_id}")
    # Print the exact attacker schedule so the run makes it clear whether the
    # malicious client is always active, never active, or only active early on.
    if args.malicious_rounds == 0:
        # A zero-length malicious phase means the attacker never poisons any local data.
        print("Malicious schedule: no poisoned rounds; all clients send clean updates.")
    else:
        # Otherwise the malicious client poisons only the opening rounds and then
        # reverts to clean local training for the rest of the run.
        print(
            "Malicious schedule: "
            f"client {args.malicious_client_id} sends poisoned updates for the first "
            f"{args.malicious_rounds} rounds, then clean updates."
        )

    # Load CIFAR-10 and optionally shrink it for quick debug or smoke-test runs.
    train_dataset, test_dataset = load_cifar10(args.data_dir)
    train_dataset = subset_dataset(train_dataset, args.train_subset, args.seed)
    test_dataset = subset_dataset(test_dataset, args.test_subset, args.seed + 1)

    # Report the actual dataset sizes that will flow through training and evaluation.
    print(f"Training samples used: {len(train_dataset)}")
    print(f"Test samples used: {len(test_dataset)}")

    # Partition the training split into client datasets and give the attacker a
    # poisonable dataset wrapper that the main loop can enable or disable by round.
    client_datasets, client_summaries = create_client_datasets(
        train_dataset=train_dataset,
        num_clients=args.num_clients,
        malicious_client_id=args.malicious_client_id,
        poison_fraction=args.poison_fraction,
        target_label=args.target_label,
        seed=args.seed,
        non_iid=args.non_iid,
        trigger_size=args.trigger_size,
    )

    # Print one line per client so the partition sizes and the attacker's eligible
    # poisoned count are explicit before training starts.
    for summary in client_summaries:
        extra = f", poisoned={summary.num_poisoned}" if summary.is_malicious else ""
        role = "malicious" if summary.is_malicious else "clean"
        print(
            f"Client {summary.client_id}: {role}, samples={summary.num_examples}{extra}"
        )

    # Create one training DataLoader per client and one shared test DataLoader used
    # for both clean-accuracy and ASR evaluation.
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

    # Each round sends the current global model to every client, collects updated
    # client weights, averages them, and evaluates the new global model.
    for round_idx in tqdm(range(1, args.rounds + 1), desc="Federated rounds"):
        client_weights = []

        # Turn the malicious client's poisoning behavior on only during the configured
        # opening rounds. After that point, the same client trains on clean data and
        # contributes clean updates to the FedAvg aggregation.
        malicious_dataset = client_datasets[args.malicious_client_id]
        if hasattr(malicious_dataset, "set_poisoning_enabled"):
            malicious_dataset.set_poisoning_enabled(round_idx <= args.malicious_rounds)

        # Train one local copy of the global model on each client's private dataset.
        for client_loader in client_loaders:
            local_weights = train_local_model(
                global_model=global_model,
                train_loader=client_loader,
                device=device,
                local_epochs=args.local_epochs,
                learning_rate=args.learning_rate,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )
            client_weights.append(local_weights)

        # Aggregate all client models into the next global model using FedAvg.
        global_model.load_state_dict(fedavg(client_weights))

        # Measure both the clean task performance and the triggered attack success
        # after the current round of federated averaging.
        mta = evaluate_clean_accuracy(global_model, test_loader, device)
        asr = evaluate_attack_success_rate(
            global_model,
            test_loader,
            target_label=args.target_label,
            device=device,
            trigger_size=args.trigger_size,
        )
        mta_history.append(mta)
        asr_history.append(asr)

        # Print per-round metrics so progress is visible during longer runs.
        print(f"\nRound {round_idx}/{args.rounds}")
        print(f"Clean Test Accuracy / MTA: {mta:.2f}%")
        print(f"Attack Success Rate / ASR: {asr:.2f}%")

    # Save the final global model weights and the metric plot inside a dedicated results folder.
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    model_path = results_dir / "global_model.pt"
    plot_path = results_dir / "results.png"
    torch.save(global_model.state_dict(), model_path)
    save_results_plot(mta_history, asr_history, plot_path)

    # Print a short completion summary with the final metrics and artifact locations.
    print("\nTraining complete.")
    print(f"Final MTA: {mta_history[-1]:.2f}%")
    print(f"Final ASR: {asr_history[-1]:.2f}%")
    print(f"Saved results to: {results_dir}")
    print(f"Saved model to: {model_path}")
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    # Allow the file to be used both as an importable module and as a script.
    main()
