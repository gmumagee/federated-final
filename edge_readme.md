# Edge-Case Backdoor Experiment

This experiment is a separate CIFAR-10 federated learning simulation that approximates the edge-case backdoor idea from *Attack of the Tails: Yes, You Really Can Backdoor Federated Learning*.

Instead of using a visible trigger patch, the malicious client mixes in unusual source-class examples and relabels them to a target class:

- source label: `0` (`airplane`)
- target label: `9` (`truck`)

The intended intuition is:

- choose hard or low-confidence airplane images
- add those examples to client `0` during the attack phase
- relabel them as `truck`
- test whether the global model later predicts `truck` for those unchanged airplane images

## Files

- [run_edge_case.py](/home/mike/projects/federated-final/run_edge_case.py)
  Runs the separate edge-case federated experiment.

- [edge_attack.py](/home/mike/projects/federated-final/edge_attack.py)
  Contains edge-case sample selection, malicious dataset construction, edge-case test-loader creation, and edge-case ASR evaluation.

- [edge_dataset.py](/home/mike/projects/federated-final/edge_dataset.py)
  Contains the dataset wrappers used to relabel a subset and mix clean and edge-case samples.

- [persistence_metrics.py](/home/mike/projects/federated-final/persistence_metrics.py)
  Contains threshold-based BPR computation, CSV writing, and plotting helpers shared by the edge-case experiment.

- [edge-case.yaml](/home/mike/projects/federated-final/edge-case.yaml)
  Stores the edge-case experiment’s configurable parameters in one place.

## What The Experiment Does

The run has two phases:

1. Attack phase
   Client `0` trains on a mixed dataset that combines its normal local data with relabeled edge-case airplane examples.

2. Persistence phase
   Client `0` stops using edge-case relabeled data and goes back to clean local training. All clients are clean in this phase.

Every round the experiment measures:

- `MTA`
  Clean CIFAR-10 test accuracy.

- `Edge ASR`
  The percentage of selected edge-case test images predicted as the target label.

- `BPR`
  The percentage of persistence rounds where `Edge ASR >= bpr_threshold`.

## How To Run

From the project root:

```bash
cd /home/mike/projects/federated-final
source .venv/bin/activate
python run_edge_case.py
```

The runner now reads its defaults from:

```bash
/home/mike/projects/federated-final/edge-case.yaml
```

You can also pass the config explicitly:

```bash
python run_edge_case.py --config edge-case.yaml
```

Command-line arguments override values from the YAML file.

## Example Commands

Default run:

```bash
python run_edge_case.py
```

Custom run:

```bash
python run_edge_case.py \
  --attack-rounds 20 \
  --persistence-rounds 10 \
  --edge-fraction 0.5 \
  --source-label 0 \
  --target-label 9 \
  --edge-selection-strategy low_confidence \
  --bpr-threshold 50.0
```

Small smoke test:

```bash
python run_edge_case.py \
  --attack-rounds 1 \
  --persistence-rounds 1 \
  --local-epochs 1 \
  --train-subset 500 \
  --test-subset 200 \
  --num-edge-train 20 \
  --num-edge-test 10 \
  --edge-selection-strategy random \
  --results-dir results-edge-smoke \
  --no-use-cuda
```

## Edge-Case YAML Parameters

The following values are listed in [edge-case.yaml](/home/mike/projects/federated-final/edge-case.yaml):

- `num_clients`
- `malicious_client_id`
- `source_label`
- `target_label`
- `attack_rounds`
- `persistence_rounds`
- `local_epochs`
- `batch_size`
- `learning_rate`
- `momentum`
- `weight_decay`
- `edge_fraction`
- `num_edge_train`
- `num_edge_test`
- `edge_selection_strategy`
- `bpr_threshold`
- `seed`
- `data_dir`
- `results_dir`
- `use_cuda`
- `non_iid`
- `train_subset`
- `test_subset`

## Outputs

The edge-case run writes separate artifacts so it does not overwrite the base pixel-trigger experiment.

By default these are saved under:

- `/home/mike/projects/federated-final/results/edge_global_model.pt`
- `/home/mike/projects/federated-final/results/edge_results.png`
- `/home/mike/projects/federated-final/results/edge_metrics.csv`

`edge_metrics.csv` contains per-round:

- `round`
- `phase`
- `mta`
- `edge_asr`
- `bpr_threshold`

## Notes

- `low_confidence` selection first trains a short clean proxy model, then chooses source-class examples where the model is least confident in the correct class.
- `random` selection skips that ranking step and samples source-class images directly.
- The current implementation is a toy local research simulation only. It does not include model replacement, stealth, secure aggregation bypass, or deployment logic.
