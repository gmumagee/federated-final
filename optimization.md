# Optimization Notes

This file documents the changes made to improve learning speed in the federated CIFAR-10 experiment, along with the reasoning behind each change.

## Goal

The original version of the project trained very slowly on the main CIFAR-10 task and quickly collapsed into learning the backdoor behavior instead of the clean classification task.

The optimization goal was:

- improve early clean-task learning
- make the global model converge faster on CIFAR-10
- keep the code simple and readable
- avoid advanced attack or defense logic

## Baseline Problem

Using this debug command:

```bash
python main.py --rounds 5 --local-epochs 1 --batch-size 64 --train-subset 5000 --test-subset 1000
```

the earlier version produced:

- Final MTA: `8.90%`
- Final ASR: `97.81%`

That showed the model was learning the trigger behavior much faster than the clean task. Clean performance was not improving fast enough to make the experiment useful as a learning baseline.

## Changes Made

### 1. Normalized CIFAR-10 inputs

Changed in:

- [data.py](/home/mike/projects/federated-final/data.py)

What changed:

- Replaced the plain `transforms.ToTensor()` pipeline with:
  - `transforms.ToTensor()`
  - `transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)`

Why:

- CIFAR-10 models usually optimize more reliably with normalized inputs.
- Normalization improves gradient scale and helps SGD move more effectively early in training.
- This is a standard optimization improvement, not a change to the experiment structure.

### 2. Updated trigger logic to support normalized tensors

Changed in:

- [attack.py](/home/mike/projects/federated-final/attack.py)

What changed:

- Added `CIFAR10_MEAN` and `CIFAR10_STD`.
- Added `build_white_trigger_value(...)`.
- Changed `add_trigger(...)` so the trigger remains visually “white” even after normalization.

Why:

- Once images are normalized, a white pixel is no longer literally the tensor value `1.0`.
- Without this fix, the trigger intensity would be inconsistent with the new input representation.
- This keeps the attack behavior consistent after the input normalization change.

### 3. Added batch normalization to the CNN

Changed in:

- [model.py](/home/mike/projects/federated-final/model.py)

What changed:

- Inserted `nn.BatchNorm2d(...)` after each convolution layer.

Why:

- Batch normalization usually improves optimization stability and speeds up useful feature learning.
- In this project, it helps the model learn the clean task more effectively in the early federated rounds.
- This is a simple architectural improvement that keeps the model small and readable.

### 4. Added explicit weight initialization

Changed in:

- [model.py](/home/mike/projects/federated-final/model.py)

What changed:

- Added `init_module_weights(...)`.
- Applied Kaiming initialization to convolution and linear layers.
- Initialized batch normalization scales and biases explicitly.

Why:

- Better initialization helps the model start from a more optimization-friendly point.
- This reduces the chance of weak early training dynamics.
- It improves early-round learning without changing the overall architecture.

### 5. Strengthened local SGD optimizer settings

Changed in:

- [federated.py](/home/mike/projects/federated-final/federated.py)
- [main.py](/home/mike/projects/federated-final/main.py)

What changed:

- Increased the default learning rate from `0.01` to `0.05`.
- Added `weight_decay`.
- Enabled `nesterov=True` in SGD.
- Added the CLI flag:
  - `--weight-decay`

Why:

- The original SGD setup was too conservative for the small number of local epochs.
- A slightly stronger learning rate helps each client make more useful progress in each federated round.
- Weight decay helps control overfitting and improves generalization on the clean task.
- Nesterov momentum often improves SGD convergence with little added complexity.

### 6. Updated documentation to match the new defaults

Changed in:

- [README.md](/home/mike/projects/federated-final/README.md)

What changed:

- Updated the documented default learning rate.
- Added the new `weight_decay` default.
- Documented the normalization and batch-normalized CNN changes.

Why:

- The README needs to reflect the actual code behavior and current defaults.

## Results After Optimization

Using the same debug command:

```bash
python main.py --rounds 5 --local-epochs 1 --batch-size 64 --train-subset 5000 --test-subset 1000
```

the updated version produced:

- Final MTA: `29.30%`
- Final ASR: `2.74%`

## Interpretation

The optimized version learns the clean CIFAR-10 task much faster than the original version.

This is good if the goal is:

- better main-task convergence
- a stronger clean baseline
- a more useful educational starting point

But the change also had an important side effect:

- the backdoor became much less effective under the same small debug configuration

That means the optimization improved clean learning, but also changed the balance between:

- clean task learning
- backdoor learning

## Tradeoff

These optimizations were intentionally aimed at **fast clean learning**, not at maximizing attack success.

So the code is now better if you want:

- quicker clean accuracy gains
- more stable early training
- a stronger base classifier

But it is less aggressive if you want:

- fast ASR growth in very few rounds
- a stronger poisoning effect under the same reduced settings

## Summary

The following optimizations were applied:

1. normalized CIFAR-10 inputs
2. trigger handling updated for normalized tensors
3. batch normalization added to the CNN
4. Kaiming-style initialization added
5. stronger local SGD defaults with Nesterov momentum and weight decay
6. README updated to document the new behavior

The result was a clear improvement in clean-task learning speed, at the cost of reduced backdoor success in the same short debug run.
