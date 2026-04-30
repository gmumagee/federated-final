from __future__ import annotations

"""Run a separate edge-case backdoor experiment on CIFAR-10."""

import argparse
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from tqdm import tqdm
import yaml

from data import (
    CIFAR10_CLASSES,
    build_dataloader,
    create_iid_partitions,
    create_label_skew_partitions,
    load_cifar10,
    subset_dataset,
)
from edge_attack import (
    create_edge_poisoned_dataset,
    evaluate_edge_asr,
    make_edge_test_loader,
    select_edge_case_indices,
)
from federated import evaluate_clean_accuracy, fedavg, train_local_model
from model import SimpleCNN
from persistence_metrics import compute_bpr, plot_attack_metrics, save_metrics_csv

# Keep the edge-case experiment's default config next to the runner so it can be
# used without any extra path management from the command line.
DEFAULT_EDGE_CONFIG_PATH = Path(__file__).resolve().with_name("edge-case.yaml")


def load_config(config_path: str | Path) -> dict[str, object]:
    """Load YAML defaults for the edge-case experiment."""

    # Read the YAML file once at startup so it can define the runner defaults.
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("Config file must contain a top-level mapping.")
    return config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the edge-case experiment."""

    # Parse the config path first. This keeps the runner behavior aligned with the
    # base experiment by allowing a YAML file to provide defaults while still
    # letting explicit CLI flags override any individual value.
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        default=str(DEFAULT_EDGE_CONFIG_PATH),
        help="Path to the edge-case YAML config file.",
    )
    config_args, _ = config_parser.parse_known_args()
    config = load_config(config_args.config)

    # Build the full CLI after the YAML file is loaded so every option can inherit
    # its default from the config file.
    parser = argparse.ArgumentParser(
        parents=[config_parser],
        description="Toy edge-case federated backdoor experiment on CIFAR-10.",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=config.get("num_clients", 10),
        help="Total number of clients.",
    )
    parser.add_argument(
        "--malicious-client-id",
        type=int,
        default=config.get("malicious_client_id", 0),
        help="Client id that receives edge-case poisoned data.",
    )
    parser.add_argument(
        "--source-label",
        type=int,
        default=config.get("source_label", 0),
        help="Source class used to build edge-case examples.",
    )
    parser.add_argument(
        "--target-label",
        type=int,
        default=config.get("target_label", 9),
        help="Target class assigned to edge-case examples.",
    )
    parser.add_argument(
        "--attack-rounds",
        type=int,
        default=config.get("attack_rounds", 20),
        help="Rounds where the malicious client uses edge-case poisoned data.",
    )
    parser.add_argument(
        "--persistence-rounds",
        type=int,
        default=config.get("persistence_rounds", 10),
        help="Rounds after poisoning stops where only clean updates are sent.",
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=config.get("local_epochs", 1),
        help="Local training epochs per client per round.",
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
        default=config.get("learning_rate", 0.01),
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
        "--edge-fraction",
        type=float,
        default=config.get("edge_fraction", 0.5),
        help="Approximate fraction of client 0's attack-phase data that is edge-case poisoned.",
    )
    parser.add_argument(
        "--num-edge-train",
        type=int,
        default=config.get("num_edge_train", 200),
        help="Number of edge-case training samples to select from the source class.",
    )
    parser.add_argument(
        "--num-edge-test",
        type=int,
        default=config.get("num_edge_test", 100),
        help="Number of edge-case test samples to select from the source class.",
    )
    parser.add_argument(
        "--edge-selection-strategy",
        type=str,
        default=config.get("edge_selection_strategy", "low_confidence"),
        choices=["low_confidence", "random"],
        help="How edge-case airplane examples are selected.",
    )
    parser.add_argument(
        "--bpr-threshold",
        type=float,
        default=config.get("bpr_threshold", 50.0),
        help="ASR threshold used to count a persistence round as successful.",
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
        help="Directory where edge-case artifacts are saved.",
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
        help="Use a label-skew split instead of the default IID partition.",
    )
    parser.add_argument(
        "--train-subset",
        type=int,
        default=config.get("train_subset"),
        help="Optional training subset size for faster debugging.",
    )
    parser.add_argument(
        "--test-subset",
        type=int,
        default=config.get("test_subset"),
        help="Optional test subset size for faster debugging.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable runs."""

    # Seed every random component used by sample selection, client partitioning,
    # and local model training so repeated runs with the same config stay aligned.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_args(args: argparse.Namespace) -> None:
    """Reject clearly invalid experiment settings before training starts."""

    # Validate the full edge-case run schedule and the basic hyperparameter ranges
    # before any dataset downloads, model construction, or sample selection happen.
    total_rounds = args.attack_rounds + args.persistence_rounds
    if args.num_clients < 1:
        raise ValueError("num_clients must be at least 1.")
    if not 0 <= args.malicious_client_id < args.num_clients:
        raise ValueError("malicious_client_id must be within the client id range.")
    if not 0 <= args.source_label < 10:
        raise ValueError("source_label must be between 0 and 9.")
    if not 0 <= args.target_label < 10:
        raise ValueError("target_label must be between 0 and 9.")
    if args.attack_rounds < 0 or args.persistence_rounds < 0 or total_rounds < 1:
        raise ValueError("attack_rounds + persistence_rounds must be at least 1.")
    if args.local_epochs < 1:
        raise ValueError("local_epochs must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if not 0.0 <= args.edge_fraction <= 1.0:
        raise ValueError("edge_fraction must be between 0.0 and 1.0.")
    if not 0.0 <= args.bpr_threshold <= 100.0:
        raise ValueError("bpr_threshold must be between 0.0 and 100.0.")
    if args.num_edge_train < 1 or args.num_edge_test < 1:
        raise ValueError("num_edge_train and num_edge_test must both be at least 1.")


def prepare_edge_selector_model(
    train_dataset: Dataset,
    device: torch.device,
    batch_size: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
) -> SimpleCNN:
    """Train a short clean proxy model so low-confidence edge selection is meaningful."""

    # Low-confidence selection needs a model with at least some clean CIFAR-10
    # knowledge. A random model would produce arbitrary confidence scores, so warm
    # up a small clean model for one local-style epoch first.
    selector_model = SimpleCNN().to(device)
    selector_loader = build_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        device=device,
    )
    selector_state = train_local_model(
        global_model=selector_model,
        train_loader=selector_loader,
        device=device,
        local_epochs=1,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    selector_model.load_state_dict(selector_state)
    return selector_model


def create_clean_client_datasets(
    train_dataset: Dataset,
    num_clients: int,
    seed: int,
    non_iid: bool,
) -> list[Dataset]:
    """Create one clean local dataset per federated client."""

    # Reuse the same partitioning style as the base experiment so the only major
    # difference is the attack itself rather than a different client split policy.
    if non_iid:
        partitions = create_label_skew_partitions(train_dataset, num_clients, seed)
    else:
        partitions = create_iid_partitions(train_dataset, num_clients, seed)
    return [Subset(train_dataset, indices) for indices in partitions]


def main() -> None:
    """Run the two-phase edge-case backdoor experiment and save its artifacts."""

    # Parse the full configuration, validate it, and seed the process before any
    # random sample selection or training begins.
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    # Select the runtime device once so the same device choice is used for clean
    # selector warmup, federated client updates, and evaluation passes.
    device = torch.device(
        "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    )
    total_rounds = args.attack_rounds + args.persistence_rounds

    # Print the high-level configuration so the run log is self-describing.
    print(f"Using device: {device}")
    print(
        f"Source label: {args.source_label} ({CIFAR10_CLASSES[args.source_label]})"
    )
    print(
        f"Target label: {args.target_label} ({CIFAR10_CLASSES[args.target_label]})"
    )
    print(f"Edge selection strategy: {args.edge_selection_strategy}")
    print(f"BPR threshold: {args.bpr_threshold:.2f}%")

    # Load CIFAR-10 and optionally shrink it for faster debug runs. The same subset
    # support as the base experiment makes smoke testing and iteration cheaper.
    train_dataset, test_dataset = load_cifar10(args.data_dir)
    train_dataset = subset_dataset(train_dataset, args.train_subset, args.seed)
    test_dataset = subset_dataset(test_dataset, args.test_subset, args.seed + 1)

    # When low-confidence selection is requested, first train a tiny clean proxy
    # model. That proxy ranks airplane images by how unsure it is about the source
    # class, which approximates selecting rare or unusual source-class examples.
    selector_model = SimpleCNN().to(device)
    if args.edge_selection_strategy == "low_confidence":
        print("Preparing clean proxy model for low-confidence edge-case selection.")
        selector_model = prepare_edge_selector_model(
            train_dataset=train_dataset,
            device=device,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

    # Pick the training and test edge-case sample pools before federated training
    # starts so the malicious examples and evaluation set stay fixed for the run.
    edge_train_indices, edge_test_indices = select_edge_case_indices(
        model=selector_model,
        dataset=(train_dataset, test_dataset),
        source_label=args.source_label,
        num_edge_train=args.num_edge_train,
        num_edge_test=args.num_edge_test,
        strategy=args.edge_selection_strategy,
        device=device,
    )
    if not edge_train_indices or not edge_test_indices:
        raise ValueError(
            "Edge-case selection did not find enough source-class examples. "
            "Increase the dataset size or lower num_edge_train/num_edge_test."
        )

    print(f"Selected {len(edge_train_indices)} edge-case train samples.")
    print(f"Selected {len(edge_test_indices)} edge-case test samples.")

    # Build one clean dataset per client, then replace only the malicious client's
    # attack-phase dataset with a mixed clean-plus-relabeled edge-case view.
    clean_client_datasets = create_clean_client_datasets(
        train_dataset=train_dataset,
        num_clients=args.num_clients,
        seed=args.seed,
        non_iid=args.non_iid,
    )
    malicious_attack_dataset = create_edge_poisoned_dataset(
        clean_client_dataset=clean_client_datasets[args.malicious_client_id],
        cifar_train_dataset=train_dataset,
        edge_train_indices=edge_train_indices,
        target_label=args.target_label,
        edge_fraction=args.edge_fraction,
    )

    # Print the malicious dataset composition so the amount of relabeled edge-case
    # data is explicit before training begins.
    print(
        "Malicious client attack dataset: "
        f"{malicious_attack_dataset.num_clean_samples} clean samples, "
        f"{malicious_attack_dataset.num_edge_samples} edge-case samples, "
        f"{len(malicious_attack_dataset)} total."
    )
    for client_id, dataset in enumerate(clean_client_datasets):
        role = "malicious" if client_id == args.malicious_client_id else "clean"
        print(f"Client {client_id}: {role}, clean samples={len(dataset)}")

    # Build the clean per-client loaders once. The malicious attack-phase loader is
    # separate so the training loop can swap it in only while poisoning is active.
    clean_client_loaders = [
        build_dataloader(
            dataset=client_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            device=device,
        )
        for client_dataset in clean_client_datasets
    ]
    malicious_attack_loader = build_dataloader(
        dataset=malicious_attack_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        device=device,
    )
    clean_test_loader = build_dataloader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        device=device,
    )
    edge_test_loader = make_edge_test_loader(
        cifar_test_dataset=test_dataset,
        edge_test_indices=edge_test_indices,
        target_label=args.target_label,
        batch_size=args.batch_size,
    )

    # Keep separate histories for clean accuracy, edge-case ASR, and the
    # persistence-phase ASR values that feed the final BPR calculation.
    global_model = SimpleCNN().to(device)
    mta_values: list[float] = []
    edge_asr_values: list[float] = []
    bpr_values: list[float | None] = []
    persistence_asr_values: list[float] = []
    metrics_rows: list[dict[str, object]] = []

    # Phase 1 uses the mixed edge-case dataset for the malicious client. Phase 2
    # returns every client to clean data so the run can measure how long the edge
    # backdoor remains effective under continued clean FedAvg updates.
    for round_idx in tqdm(range(1, total_rounds + 1), desc="Edge-case rounds"):
        phase = "attack" if round_idx <= args.attack_rounds else "persistence"
        client_loaders = list(clean_client_loaders)
        if phase == "attack":
            client_loaders[args.malicious_client_id] = malicious_attack_loader

        # Every round starts from the current global model, collects one local
        # update per client, and averages them with standard FedAvg.
        client_weights = []
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

        global_model.load_state_dict(fedavg(client_weights))

        # Evaluate the current global model on both the clean CIFAR-10 test set and
        # the edge-case test loader, then save the persistence-phase ASR values used
        # by the final threshold-based BPR summary.
        mta = evaluate_clean_accuracy(global_model, clean_test_loader, device)
        edge_asr = evaluate_edge_asr(global_model, edge_test_loader, device=device)
        mta_values.append(mta)
        edge_asr_values.append(edge_asr)
        bpr = None
        if phase == "persistence":
            persistence_asr_values.append(edge_asr)
            bpr = compute_bpr(
                persistence_asr_values,
                threshold=args.bpr_threshold,
            )
        bpr_values.append(bpr)

        # Record one CSV row per round so the saved metrics file can be graphed or
        # post-processed later without parsing console text.
        metrics_rows.append(
            {
                "round": round_idx,
                "phase": phase,
                "mta": f"{mta:.2f}",
                "edge_asr": f"{edge_asr:.2f}",
                "bpr": "" if bpr is None else f"{bpr:.2f}",
                "bpr_threshold": f"{args.bpr_threshold:.2f}",
            }
        )

        # Print a concise per-round summary so the training behavior is visible
        # during longer runs without opening the CSV afterward.
        print(f"\nRound {round_idx}/{total_rounds}")
        print(f"Phase: {phase}")
        print(f"MTA: {mta:.2f}%")
        print(f"Edge ASR: {edge_asr:.2f}%")
        if bpr is None:
            print("Backdoor Persistence Rate / BPR: N/A (persistence phase has not started)")
        else:
            print(
                "Backdoor Persistence Rate / BPR: "
                f"{bpr:.2f}% of persistence rounds with ASR >= {args.bpr_threshold:.2f}%"
            )

    # After the persistence phase ends, compress the raw persistence ASR values
    # into the threshold-based BPR summary requested for this project.
    bpr = compute_bpr(persistence_asr_values, threshold=args.bpr_threshold)

    # Save the final global weights, the plot covering all rounds, and the per-round
    # CSV without overwriting the base experiment's artifact names.
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    model_path = results_dir / "edge_global_model.pt"
    plot_path = results_dir / "edge_results.png"
    csv_path = results_dir / "edge_metrics.csv"

    torch.save(global_model.state_dict(), model_path)
    plot_attack_metrics(
        path=plot_path,
        rounds=list(range(1, total_rounds + 1)),
        mta_values=mta_values,
        asr_values=edge_asr_values,
        bpr_values=bpr_values,
        attack_rounds=args.attack_rounds,
        persistence_start_round=args.attack_rounds + 1,
        title="Edge-Case Backdoor Persistence",
    )
    save_metrics_csv(
        path=csv_path,
        rows=metrics_rows,
        fieldnames=["round", "phase", "mta", "edge_asr", "bpr", "bpr_threshold"],
    )

    # Print the final summary after all artifacts are written so the user can see
    # both the outcome and the exact output locations in one place.
    print("\nTraining complete.")
    print(f"Final Clean Accuracy / MTA: {mta_values[-1]:.2f}%")
    print(f"Final Edge Attack Success Rate / Edge ASR: {edge_asr_values[-1]:.2f}%")
    print(f"Backdoor Persistence Rate / BPR: {bpr:.2f}%")
    print(
        "BPR measures the percentage of post-attack persistence rounds where ASR "
        "remains above the configured threshold."
    )
    print(f"Saved model to: {model_path}")
    print(f"Saved plot to: {plot_path}")
    print(f"Saved metrics CSV to: {csv_path}")
    print(f"Attack rounds: {args.attack_rounds}")
    print(f"Persistence rounds: {args.persistence_rounds}")
    print(f"Edge fraction: {args.edge_fraction:.2f}")
    print(f"Source label: {args.source_label}")
    print(f"Target label: {args.target_label}")
    print(f"Edge selection strategy: {args.edge_selection_strategy}")


if __name__ == "__main__":
    # Keep the module importable while still allowing direct execution from the shell.
    main()
