# federated-final

This project is a small PyTorch simulation of a federated learning backdoor experiment on CIFAR-10. It is designed for education and research demos, not for real-world deployment.

The code builds a simple federated learning loop with:

- 10 total clients
- 1 global model
- FedAvg aggregation
- 1 malicious client
- a visible 4x4 pixel trigger
- clean-accuracy evaluation and backdoor-success evaluation

## What The Code Does

The experiment trains a simple CNN on CIFAR-10 in a federated setting.

Each round works like this:

1. The server starts with the current global model.
2. The model is copied to all 10 clients.
3. Each client trains locally on its own partition of the CIFAR-10 training set.
4. Client `0` is malicious and poisons a fraction of its local examples.
5. Poisoned examples have a trigger added and their label changed to the attacker target label.
6. Each client returns updated model weights.
7. The server averages the client weights with FedAvg.
8. The updated global model is evaluated on:
   - clean CIFAR-10 test data for Main Task Accuracy, or MTA
   - triggered non-target CIFAR-10 test images for Attack Success Rate, or ASR

The goal is to let you observe whether the global model can keep learning the main task while also learning the trigger behavior introduced by the malicious client.

## Paper Inspiration

This project is inspired by:

- Eugene Bagdasaryan, Andreas Veit, Yiqing Hua, Deborah Estrin, and Vitaly Shmatikov, *How To Backdoor Federated Learning*, AISTATS 2020, PMLR 108:2938-2948.

Paper link:

- https://proceedings.mlr.press/v108/bagdasaryan20a.html

The paper studies stronger federated backdoor attacks, especially model-poisoning and model-replacement methods in realistic federated settings. This repository does **not** implement the full attack methods from that paper.

Instead, this code is a much simpler toy version that keeps the core research idea easy to inspect:

- CIFAR-10 only
- 10 clients instead of large-scale deployments
- a small CNN instead of larger architectures
- one malicious client
- simple data poisoning
- a basic visible trigger
- plain FedAvg
- no stealth logic, no defense evasion, no secure aggregation bypasses

So it is more accurate to say this project is **derived from the high-level idea** of the paper, not a faithful reproduction of the full paper methodology.

## Project Files

- [main.py](/home/mike/projects/federated-final/main.py)
  Runs the full experiment. It parses command-line arguments, sets the random seed, loads CIFAR-10, creates client datasets, launches federated training, prints round-by-round metrics, and saves outputs.

- [model.py](/home/mike/projects/federated-final/model.py)
  Defines the `SimpleCNN` model used by both the server and all clients. It is a small convolutional classifier sized for CIFAR-10 and fast local experimentation.

- [data.py](/home/mike/projects/federated-final/data.py)
  Loads CIFAR-10, optionally creates a smaller deterministic debug subset, partitions the training data into 10 clients, optionally creates a simple non-IID split, and wraps client `0` with poisoned data.

- [attack.py](/home/mike/projects/federated-final/attack.py)
  Contains the backdoor logic. It adds the visible pixel trigger, defines the poisoned dataset wrapper, and provides helpers used during attack evaluation.

- [federated.py](/home/mike/projects/federated-final/federated.py)
  Contains the federated learning mechanics: local client training, FedAvg aggregation, clean accuracy evaluation, and attack success rate evaluation.

- [requirements.txt](/home/mike/projects/federated-final/requirements.txt)
  Lists the Python dependencies needed to run the experiment.

- [`.gitignore`](/home/mike/projects/federated-final/.gitignore)
  Keeps local artifacts out of Git, including the virtual environment, downloaded dataset files, saved model weights, and generated plots.

## Metrics

The code reports two metrics after every communication round:

- `Main Task Accuracy (MTA)`
  Accuracy on the clean CIFAR-10 test set.

- `Attack Success Rate (ASR)`
  The percentage of triggered test images that are classified as the attacker target label. The implementation excludes images that already belong to the target class so ASR better reflects true backdoor behavior.

## Default Experiment Settings

Unless you override them on the command line, the default settings are:

- `num_clients = 10`
- `malicious_client_id = 0`
- `target_label = 2` which is `bird`
- `num_rounds = 20`
- `local_epochs = 1`
- `batch_size = 64`
- `learning_rate = 0.01`
- `momentum = 0.9`
- `IID` client split by default

## Setup

Create a virtual environment and install dependencies:

```bash
cd /home/mike/projects/federated-final
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

After setup, the project directory should contain:

- source files in the project root
- the virtual environment at `.venv/`
- downloaded CIFAR-10 data under `data/` after the first run

## How To Run

Run all commands from the project root:

```bash
cd /home/mike/projects/federated-final
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

### Default Run

This uses the built-in defaults:

- `20` federated rounds
- `1` local epoch per client per round
- `10` clients
- client `0` as the malicious client
- `target_label = 2` which is `bird`
- IID client partitioning

Command:

```bash
python main.py
```

### Debug Run

Use this smaller run to quickly test the code path without training on the full dataset:

```bash
python main.py --rounds 2 --local-epochs 1 --batch-size 64 --train-subset 1000 --test-subset 500
```

### Non-IID Run

Use the optional label-skew split instead of the default IID split:

```bash
python main.py --non-iid
```

### Change The Poison Rate

Increase or decrease the fraction of poisoned examples on the malicious client:

```bash
python main.py --poison-fraction 0.3
```

### Example Custom Run

This example changes several settings at once:

```bash
python main.py \
  --rounds 10 \
  --local-epochs 2 \
  --batch-size 64 \
  --poison-fraction 0.2 \
  --target-label 2
```

### What To Expect During Execution

When the program starts, it will:

1. select `cuda` if available, otherwise `cpu`
2. download CIFAR-10 on the first run if it is not already present
3. print the client split configuration
4. print one line per client showing sample counts and poisoned count for client `0`
5. train through the requested federated rounds
6. print metrics after each round:
   - `Clean Test Accuracy / MTA`
   - `Attack Success Rate / ASR`
7. save the final model and plot when training completes

If you are running for the first time, the initial CIFAR-10 download is expected.

## Command-Line Arguments

`main.py` supports these main options:

- `--rounds`
  Number of federated communication rounds.

- `--local-epochs`
  Number of local epochs each client runs in each round.

- `--poison-fraction`
  Fraction of the malicious client's local dataset to poison.

- `--target-label`
  The attacker target class. The default is `2`, which corresponds to `bird`.

- `--batch-size`
  Batch size for training and evaluation.

- `--learning-rate`
  Learning rate for local SGD.

- `--momentum`
  Momentum for local SGD.

- `--seed`
  Random seed used for partitioning, poisoning, and model initialization.

- `--data-dir`
  Directory where CIFAR-10 is downloaded and cached.

- `--non-iid`
  Switch from the default IID client split to a simple label-skew split.

- `--train-subset`
  Use only a deterministic subset of the training split for fast debugging.

- `--test-subset`
  Use only a deterministic subset of the test split for fast debugging.

## Artifacts And File Locations

All paths below are relative to the project root:

- `main.py`
  Main program entrypoint.

- `model.py`
  CNN model definition.

- `data.py`
  CIFAR-10 loading, partitioning, and client dataset construction.

- `attack.py`
  Trigger injection and poisoning helpers.

- `federated.py`
  Local training, FedAvg, and evaluation logic.

- `requirements.txt`
  Python package dependencies.

- `.venv/`
  Local virtual environment created during setup.

- `data/`
  Dataset directory used by `torchvision.datasets.CIFAR10`.

- `data/cifar-10-python.tar.gz`
  Downloaded CIFAR-10 archive cached locally after the first run.

- `data/cifar-10-batches-py/`
  Extracted CIFAR-10 batch files used by the program.

- `global_model.pt`
  Final trained global model weights saved at the end of a run.

- `results.png`
  Plot of MTA and ASR across communication rounds.

- `.gitignore`
  Prevents dataset files, model artifacts, plots, and the virtual environment from being committed.

## Output Files Produced By A Run

After a run completes, the code saves:

- `global_model.pt`
  Location: `/home/mike/projects/federated-final/global_model.pt`
  
  This is the final global PyTorch model state dictionary produced at the end of training.

- `results.png`
  Location: `/home/mike/projects/federated-final/results.png`

  This plot shows:
  - Main Task Accuracy over rounds
  - Attack Success Rate over rounds

- `data/`
  Location: `/home/mike/projects/federated-final/data/`

  This directory holds the downloaded CIFAR-10 files used by the experiment.

### Files The Program Reads During Execution

The main files used directly at runtime are:

- `/home/mike/projects/federated-final/main.py`
- `/home/mike/projects/federated-final/model.py`
- `/home/mike/projects/federated-final/data.py`
- `/home/mike/projects/federated-final/attack.py`
- `/home/mike/projects/federated-final/federated.py`
- `/home/mike/projects/federated-final/requirements.txt`

### Files The Program Writes During Execution

The program writes or updates:

- `/home/mike/projects/federated-final/data/`
  Downloaded dataset files if CIFAR-10 is not already present.

- `/home/mike/projects/federated-final/global_model.pt`
  Final trained model weights.

- `/home/mike/projects/federated-final/results.png`
  Final metrics plot.

## Safety And Scope

This repository is intentionally limited to a local toy simulation. It does not include:

- model replacement scaling
- constrain-and-scale methods
- stealth or evasion logic
- secure aggregation bypasses
- networking code
- live federated systems

It is meant to demonstrate the core concept that poisoned local client data can influence a shared global model in federated learning.
