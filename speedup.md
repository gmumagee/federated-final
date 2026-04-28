# Speedup / Code-Minimization Pass

This file does **not** change the code. It documents the edits I would make if the goal is:

- the smallest practical amount of **executable** code
- while still running the same core experiment
- and keeping the project in the same basic file structure

## Core Behavior To Preserve

The experiment should still do all of the following:

- load CIFAR-10
- create exactly 10 clients
- use client `0` as the malicious client
- poison a fraction of client `0` with a visible trigger and target relabeling
- train a global model with FedAvg
- measure:
  - clean accuracy / MTA
  - attack success rate / ASR
- save:
  - `global_model.pt`
  - `results.png`
- run with `python main.py`

## Important Note

Most of the current files already contain many comments and docstrings. Those do **not** increase executable code. So if the target is specifically “minimum executable lines,” the biggest wins come from removing optional branches, helper wrappers, and extra CLI features.

## Highest-Value Cuts

These are the changes with the best reduction in executable code while preserving the intended experiment.

### 1. Remove Optional Non-IID Support

Files affected:

- [data.py](/home/mike/projects/federated-final/data.py)
- [main.py](/home/mike/projects/federated-final/main.py)
- [README.md](/home/mike/projects/federated-final/README.md)

Suggested edits:

- delete `NON_IID`
- delete `get_dataset_labels(...)`
- delete `create_label_skew_partitions(...)`
- delete the `non_iid` argument from `create_client_datasets(...)`
- remove `--non-iid` from the CLI
- always use the IID path

Why:

- non-IID support is optional in the original requirements
- it adds a lot of branching and helper code
- the default intended experiment is already IID

Expected savings:

- large reduction in `data.py`
- simpler `main.py`

### 2. Remove Debug-Only Dataset Subset Support

Files affected:

- [data.py](/home/mike/projects/federated-final/data.py)
- [main.py](/home/mike/projects/federated-final/main.py)
- [README.md](/home/mike/projects/federated-final/README.md)

Suggested edits:

- delete `subset_dataset(...)`
- remove `--train-subset`
- remove `--test-subset`
- stop subsetting datasets in `main.py`

Why:

- these flags were added for debugging convenience, not for the actual experiment
- they add CLI parsing, validation, and data-path branching
- the original requested CLI did not require them

Expected savings:

- medium reduction in `main.py`
- medium reduction in `data.py`

### 3. Remove `ClientSummary` And Return Only Client Datasets

Files affected:

- [data.py](/home/mike/projects/federated-final/data.py)
- [main.py](/home/mike/projects/federated-final/main.py)

Suggested edits:

- delete the `ClientSummary` dataclass
- make `create_client_datasets(...)` return only `list[Dataset]`
- if you still want logging, print lengths directly in `main.py`
- if you want the absolute minimum, remove the per-client sample logging entirely

Why:

- the summary object exists only for logging
- it does not contribute to the training algorithm
- it costs code both where it is created and where it is consumed

Expected savings:

- moderate reduction across both files

### 4. Remove `poison_dataset(...)` Helper

Files affected:

- [attack.py](/home/mike/projects/federated-final/attack.py)
- [data.py](/home/mike/projects/federated-final/data.py)

Suggested edits:

- delete `poison_dataset(...)`
- instantiate `PoisonedDataset(...)` directly in `data.py`

Why:

- it is only a thin wrapper around the class constructor
- it adds one more function without reducing complexity anywhere else

Expected savings:

- small but clean reduction

### 5. Merge Trigger Helpers Into One Function

Files affected:

- [attack.py](/home/mike/projects/federated-final/attack.py)
- [federated.py](/home/mike/projects/federated-final/federated.py)

Suggested edits:

- delete `build_white_trigger_value(...)`
- delete `add_trigger_to_batch(...)`
- replace them with one `add_trigger(...)` that accepts either:
  - a single image of shape `C x H x W`, or
  - a batch of shape `N x C x H x W`

Suggested direction:

- keep one hardcoded normalized white trigger value
- use tensor shape checks and one implementation path

Why:

- the current code splits trigger work into three helpers
- the experiment only needs one trigger behavior
- batch and single-image paths can be handled by one function

Expected savings:

- moderate reduction in `attack.py`
- small reduction in `federated.py`

### 6. Delete `clone_state_dict(...)`

Files affected:

- [federated.py](/home/mike/projects/federated-final/federated.py)

Suggested edits:

- remove `clone_state_dict(...)`
- in `train_local_model(...)`, return the detached CPU state dict directly:
  - `{k: v.detach().cpu().clone() for k, v in local_model.state_dict().items()}`

Why:

- `clone_state_dict(...)` is only called once
- single-use wrappers are easy places to cut code

Expected savings:

- small reduction

## Medium-Value Cuts

These are useful if you want to push harder toward minimum lines.

### 7. Shrink The CLI To Only The Originally Requested Flags

Files affected:

- [main.py](/home/mike/projects/federated-final/main.py)
- [README.md](/home/mike/projects/federated-final/README.md)

Suggested edits:

- keep only:
  - `--rounds`
  - `--local-epochs`
  - `--poison-fraction`
  - `--target-label`
  - `--batch-size`
- move these to internal constants:
  - `learning_rate`
  - `momentum`
  - `weight_decay`
  - `seed`
  - `data_dir`

Why:

- the original prompt only suggested those five CLI knobs
- every extra argument adds parsing code, help text, and validation logic

Expected savings:

- moderate reduction in `main.py`

### 8. Remove Most Manual Validation

Files affected:

- [main.py](/home/mike/projects/federated-final/main.py)

Suggested edits:

- remove explicit `if ... raise ValueError(...)` checks for:
  - `rounds`
  - `local_epochs`
  - `batch_size`
  - `weight_decay`
  - subset arguments if those are removed anyway
- optionally keep only the `target_label` range check

Why:

- the code can be shorter and still work correctly for valid inputs
- this is one of the easiest places to reduce executable lines

Tradeoff:

- error messages become less precise for invalid inputs

### 9. Replace `SimpleCNN` With A Shorter `nn.Sequential` Subclass

Files affected:

- [model.py](/home/mike/projects/federated-final/model.py)

Suggested edits:

- make `SimpleCNN` inherit from `nn.Sequential`
- pass the layers directly to `super().__init__(...)`
- remove the separate `forward(...)` method

Why:

- the current class has separate `features`, `classifier`, and `forward`
- `nn.Sequential` can express the same model in fewer executable lines

Example direction:

```python
class SimpleCNN(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10),
        )
```

Tradeoff:

- you lose the clearer `features`/`classifier` split

### 10. Consider Dropping Explicit Weight Initialization

Files affected:

- [model.py](/home/mike/projects/federated-final/model.py)

Suggested edits:

- remove `init_module_weights(...)`
- remove `self.apply(init_module_weights)`

Why:

- this is extra optimization logic
- PyTorch defaults are enough to run the experiment correctly

Tradeoff:

- you may lose some early convergence quality
- this reduces code more than it preserves “fast clean learning”

Recommendation:

- if “fewest executable lines” is the top priority, cut it
- if “faster clean learning” is still important, keep it

## Low-Value Cuts

These save fewer lines or risk making the code less clear than the reduction is worth.

### 11. Inline `save_results_plot(...)`

Files affected:

- [main.py](/home/mike/projects/federated-final/main.py)

Suggested edits:

- move the plotting code into the bottom of `main()`
- delete the helper function

Why:

- one less function

Why I would not prioritize it:

- total code size barely changes
- it makes `main()` longer and harder to scan

### 12. Merge The Two Evaluation Functions

Files affected:

- [federated.py](/home/mike/projects/federated-final/federated.py)

Suggested edits:

- combine clean evaluation and ASR evaluation into one function with a flag

Why I would not prioritize it:

- it introduces more branching inside the function
- it does not clearly reduce complexity
- it may save fewer lines than expected

## Recommended Minimum Refactor

If I were doing the highest-value cuts only, I would make this exact simplification set:

1. remove non-IID support
2. remove dataset subset support
3. remove `ClientSummary`
4. remove `poison_dataset(...)`
5. merge trigger helpers into one function
6. remove `clone_state_dict(...)`
7. reduce the CLI to the original five flags
8. keep normalization, batch norm, and current optimizer defaults

That would preserve:

- the core experiment design
- current faster clean-learning behavior
- the same outputs

while removing a substantial amount of executable code.

## Most Aggressive Version I Would Still Consider Safe

If you want the absolute minimum while still keeping the current experiment recognizable, I would also:

- remove most manual validation
- convert `SimpleCNN` to a short `nn.Sequential` subclass
- possibly drop explicit weight initialization

I would **not** remove:

- normalization
- the poisoning dataset wrapper
- the separate attack evaluation metric
- model saving
- plot saving

because those are too central to the current experiment behavior.

## Rough Priority Order

Apply these in this order if you want the best line-reduction per edit:

1. remove non-IID support
2. remove subset/debug support
3. remove `ClientSummary`
4. shrink the CLI
5. merge trigger helpers
6. remove `poison_dataset(...)`
7. remove `clone_state_dict(...)`
8. convert `SimpleCNN` to `nn.Sequential`
9. remove weight initialization if you accept the tradeoff

## Bottom Line

The current code is already fairly compact in the training core. Most remaining executable-line reductions come from removing **optional features** rather than from rewriting the main federated learning algorithm.

If you want the strongest minimum-code version, the biggest win is:

- keep only the IID CIFAR-10 experiment path
- keep only the required CLI
- keep only the required outputs
- collapse helper wrappers that are used once
