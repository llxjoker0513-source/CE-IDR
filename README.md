# CE-IDR

**Complementary Experts with Intensity- and Discrepancy-Guided Routing for Infrared and Visible Image Fusion**

CE-IDR studies adaptive feature fusion for registered infrared and visible image pairs.
The framework includes structural and contextual feature encoding, complementary
expert fusion, cross-scale feature regulation, and image reconstruction.

## Release Status

This is a **partial code release**. It provides the surrounding network framework,
paired-image loading, training and inference entry points, and evaluation utilities.

The expert fusion implementation is intentionally withheld. The corresponding
module contains an explicit interface placeholder and raises `NotImplementedError`;
it does not substitute a different fusion method. Detailed model settings, training
hyperparameters, loss weights, checkpoint-selection weights, and post-processing
settings are not included. The configuration template contains empty fields only.

Training and inference require the withheld module and a complete private
configuration. This repository alone cannot reproduce the full model.

Datasets, pretrained weights, generated images, logs, experiment tables, comparison
methods, and manuscript files are not distributed in this repository.

## Code Layout

```text
ce_idr/
  models/                 Network framework and withheld fusion interface
  datasets/               Paired-image loading and transforms
  losses/                 Loss definitions without experimental defaults
  utils/                  Evaluation, logging, and checkpoint helpers
  configs/example.yaml    Configuration schema with values withheld
  public_config.py        Shared configuration and model construction
  train.py                Training entry point
  test.py                 Inference entry point
  eval_dataset_gpu.py     Saved-image evaluation
  sweep_checkpoints.py    Configurable checkpoint evaluation
requirements.txt
```

## Environment

The surrounding framework uses Python, PyTorch, OpenCV, NumPy, PyYAML, tqdm, and
TensorBoard. Dependency versions and the detailed experimental environment are not
specified in this partial release.

```bash
python -m pip install -r requirements.txt
```

## Entry Points

Each entry point accepts `--help`. A complete configuration is required for actual
execution. The supplied template is deliberately incomplete.

```bash
python ce_idr/train.py --help
python ce_idr/test.py --help
python ce_idr/eval_dataset_gpu.py --help
python ce_idr/sweep_checkpoints.py --help
```

Dataset directories and output locations are supplied by the user. No local machine
paths or experimental results are included.
