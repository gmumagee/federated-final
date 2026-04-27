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

## How To Run

Run the default experiment:

```bash
cd /home/mike/projects/federated-final
source .venv/bin/activate
python main.py
```

Run a smaller debug experiment:

```bash
cd /home/mike/projects/federated-final
source .venv/bin/activate
python main.py --rounds 2 --local-epochs 1 --batch-size 64 --train-subset 1000 --test-subset 500
```

Run with a non-IID split:

```bash
python main.py --non-iid
```

Change the poison rate:

```bash
python main.py --poison-fraction 0.3
```

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

## Outputs

After a run completes, the code saves:

- `global_model.pt`
  The final global model weights.

- `results.png`
  A plot showing MTA and ASR over communication rounds.

The CIFAR-10 dataset is downloaded into:

- `data/`

## Safety And Scope

This repository is intentionally limited to a local toy simulation. It does not include:

- model replacement scaling
- constrain-and-scale methods
- stealth or evasion logic
- secure aggregation bypasses
- networking code
- live federated systems

It is meant to demonstrate the core concept that poisoned local client data can influence a shared global model in federated learning.
