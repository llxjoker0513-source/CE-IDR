# CE-IDR

**Complementary Experts with Intensity- and Discrepancy-Guided Routing for Infrared and Visible Image Fusion**

CE-IDR studies adaptive feature fusion for registered infrared and visible image pairs.
The framework includes structural and contextual feature encoding, complementary
expert fusion, cross-scale feature regulation, and image reconstruction.

## Release Status

This is a **partial code release**. It provides the surrounding network framework,
loss definitions, tensor-based metric functions, and checkpoint and logging utilities.

The expert fusion implementation is intentionally withheld. The corresponding
module contains an explicit interface placeholder and raises `NotImplementedError`;
it does not substitute a different fusion method. Detailed model settings, training
hyperparameters, loss weights, checkpoint-selection weights, and post-processing
settings are not included.

Dataset readers, training and inference entry points, dataset evaluation runners,
checkpoint-scanning runners, and their configuration templates are not distributed.
The model and loss interfaces operate on in-memory tensors; this release contains
no interface for loading training or test datasets.

The withheld expert implementation and detailed settings are required to reproduce
the full model. This repository alone is not a complete reproduction package.

Datasets, pretrained weights, generated images, logs, experiment tables, comparison
methods, and manuscript files are not distributed in this repository.

## Code Layout

```text
ce_idr/
  models/                 Network framework and withheld fusion interface
  losses/                 Loss definitions without experimental defaults
  utils/                  Evaluation, logging, and checkpoint helpers
  public_config.py        Model construction from in-memory settings
requirements.txt
```

## Environment

The surrounding framework uses Python, PyTorch, and TensorBoard. Dependency
versions and the detailed experimental environment are not specified in this
partial release.

```bash
python -m pip install -r requirements.txt
```

No local machine paths or experimental results are included.
