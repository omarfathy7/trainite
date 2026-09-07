# Training Guide

This page explains the YAML configuration that Trainite expects and how to kick off a training run.

For a working example you can run right away, see the [string reversal example](https://github.com/pytorch-ignite/trainite/tree/main/examples/string_reversal).

## Configuration overview

Every Trainite experiment is described by a single YAML file. At startup the file is loaded, validated against `ProjectConfig`, and used to build all the components (model, optimizer, data pipeline, etc.) automatically.

Here is a stripped-down config to show the overall shape:

```yaml
project_name: my_experiment

preprocessor:
  _target_: preprocessors.char_tokenizer.CharTokenizer

model:
  _target_: models.transformer.TransformerModel
  hidden_size: 64
  num_layers: 2

optimizer:
  _target_: torch.optim.AdamW
  lr: 0.0003

data:
  dataset:
    _target_: datasets.my_dataset.MyDataset
  dataloader:
    batch_size: 128
    shuffle: true
  val_ratio: 0.1
  test_ratio: 0.1

trainer:
  epochs: 3
  log_every_steps: 50

output:
  root: outputs
  run_name: first_run

logger: tensorboard
seed: 42
device: null
```

### How `_target_` works

Any block that needs to create a Python object uses the `_target_` key. The value is a dotted import path — Trainite will import that path and pass the remaining keys as arguments. For example:

```yaml
optimizer:
  _target_: torch.optim.AdamW
  lr: 0.0003
```

This tells Trainite to instantiate `torch.optim.AdamW` with the configured learning rate. The remaining arguments (like model parameters) are injected by the trainer.

The `preprocessor`, `model`, `optimizer`, and `data.dataset` blocks all use this pattern.

### Configuration blocks

**`project_name`** — Name of the project.

**`preprocessor`** — The tokenizer or preprocessing component. Must provide a `_target_`.

**`model`** — Model architecture and hyperparameters. The `_target_` points to the model class; everything else (like `hidden_size`, `num_layers`) is passed to its constructor.

**`optimizer`** — Defaults to `torch.optim.AdamW` with `lr=0.001` if not specified.

**`data`** — Datasets, transforms, and dataloaders. Trainite supports two ways to set up your data splits:

  1. **Auto-split** — provide a single `dataset` block along with `val_ratio` and `test_ratio`. Trainite calls `torch.utils.data.random_split` to divide the data. This is what most examples use.
  2. **Explicit splits** — provide separate `train`, `val`, and optionally `test` blocks, each with its own `dataset`, `transform`, and `dataloader` config.

**`trainer`** — Training loop parameters: `epochs`, `log_every_steps`, `early_stopping_patience` (set to `null` to disable), `inference_every_epochs`, `inference_num_samples`, `max_inference_new_tokens`, and `grad_clip_norm`.

**`output`** — Where artifacts are saved. `root` is the top-level directory and `run_name` identifies the experiment.

**`logger`** — Either `tensorboard` (default) or `clearml`.

**`seed`** — Random seed for reproducibility. Defaults to `42`.

**`device`** — `cpu`, `cuda`, or `null` to let Trainite pick automatically via PyTorch-Ignite's distributed utilities.

## Running training

Each example has a `main.py` entry point. From the example directory:

```bash
uv run python main.py config.yaml
```

Or without `uv`:

```bash
python main.py config.yaml
```

What happens under the hood:

1. The YAML is loaded and validated into a `ProjectConfig` instance.
2. The preprocessor, model, optimizer, and dataloaders are all built from their `_target_` entries.
3. Logging, metrics, and the output directory are set up.
4. `trainer.run()` starts the PyTorch-Ignite training loop — running epochs, evaluating on the validation set, saving checkpoints, and optionally running test evaluation at the end.

## Outputs

Each run creates a timestamped directory under `outputs/<run_name>/`:

```
outputs/
└── first_run/
    └── 20260901_143022/
        ├── config.yaml         # Copy of the run configuration
        ├── output.log          # Training logs
        ├── best.pt             # Best model checkpoint (by validation loss)
        ├── last.pt             # Most recent checkpoint
        └── tensorboard/        # TensorBoard event files (when using tensorboard logger)
```

If you're using TensorBoard, you can visualize metrics with:

```bash
uv run tensorboard --logdir outputs
```

## Examples

Working examples are the best way to understand how everything fits together:

* [**String Reversal**](https://github.com/pytorch-ignite/trainite/tree/main/examples/string_reversal) — a sequence-to-sequence toy task that trains a decoder-only Transformer to reverse character strings.
* [**Counting**](https://github.com/pytorch-ignite/trainite/tree/main/examples/counting) — a counting task using ClearML for logging.
