# Selective-Invariance

Implementation of **Risk-controlled consistency for transformation-example pairs**.

Trains a compatibility ensemble on stop-gradient teacher descriptors, selects a held-out lower-confidence gate with a false-acceptance bound and retains clean task loss for rejected pairs.

## Install

Use Python 3.11 or newer. Install a matching PyTorch/TorchVision build for the target CPU or CUDA device, then install the package.

```bash
git clone https://github.com/Hqa77880011/Selective-Invariance.git
cd Selective-Invariance
python -m venv .venv
```

Activate with `source .venv/bin/activate` or `.venv\Scripts\Activate.ps1` in PowerShell.

```bash
python -m pip install -e .
```

## Prepare data

Use CIFAR-100 or a controlled transformation task with independently audited pair validity. Compatibility training and threshold calibration use separate source objects.

x and x_pair are RGB images; transform is the numeric transformation descriptor; y is the clean class; valid is evaluator/audit pair validity. Main task training does not read valid. Source group IDs remain disjoint across compatibility training, calibration and test.

```bash
python -m selective prepare --source data/raw.npz --output data/prepared --seed 17
```

Use `group` to keep every derived view of one source object in the same partition. An explicit integer `split` array (0=train, 1=validation, 2=test) preserves a registered split. Otherwise groups are assigned 60/20/20. Every stored field must share the sample axis. Files are loaded without pickle.

## Fit the compatibility gate first

```bash
python -m selective train --config configs/default.json --data data/prepared --methods erm --output runs/teacher
python -m selective.gate --checkpoint runs/teacher/erm/model.pt --train-pairs data/audit_train.npz --calibration-pairs data/audit_calibration.npz --output runs/gate.pt
```

The audited pair files use x, x_pair, transform and valid. Keep their source objects separate. After gate fitting, run the full training commands above. Retain the gate artifact alongside the classifier checkpoint.



## Train

Edit `configs/default.json` for the input dimensions, device and training budget.

```bash
python -m selective train --config configs/default.json --data data/prepared --output runs/main
python -m selective suite --config configs/default.json --data data/prepared --output runs/comparison
```

`--methods` selects a subset. selective, erm, always_consistent, random_gate, no_uncertainty and no_risk_control.

The common driver uses AdamW with cosine learning-rate decay, validation-objective early stopping and a fixed seed. Custom federated/continual drivers use their registered round/task budgets. It saves only a validation-selected model and measured epoch history. `learning_rate`, `weight_decay`, `batch_size`, `epochs`, `patience` and `device` control optimization. Use separate output roots for independent runs.

## Evaluate and visualize

```bash
python -m selective evaluate --checkpoint runs/main/selective/model.pt --data data/prepared/test.npz --output results/selective
python -m selective plot --inputs runs/main/selective results/selective --output figures
```

`model.pt` contains model parameters, configuration and method identity. `history.jsonl` contains measured training/validation loss and epoch time. Evaluation writes `predictions.npz` and `metrics.json`. Plotting creates loss curves and metric comparisons from supplied run directories.

Task accuracy/F1/ECE, acceptance coverage, conditional false acceptance, false rejection and per-example invalid-invariance gradient ratio IIGR.

## Implementation conventions

The compatibility teacher is frozen so its feature distribution stays consistent with the audit. Three bootstrap MLP heads supply uncertainty. Threshold selection uses one-sided Clopper-Pearson bounds with a finite-grid multiplicity correction and refuses an infeasible coverage/risk target.


## Pack image records

A JSONL manifest can point to images, masks and NPY feature arrays. Paths are relative to the manifest. All rows use identical keys; group stores the source object ID. For example, an image-classification row is `{"x":"images/1.png","y":0,"group":"object-1"}`.

```bash
python -m selective.pack --manifest data/manifest.jsonl --output data/raw.npz --channels 3 --target class
```

## CIFAR-100 preparation

```bash
python -m selective.cifar --output data/raw.npz
```

This downloads the author dataset through TorchVision, reserves 10% of its training images for validation, and preserves the original test set. It writes fine/coarse labels, source IDs and task metadata. Paired/audited transformation fields and ontology constraints must be added from the registered pair/ontology protocol before using those objectives.

Inference timing runs the deployment forward pass separately from calibration, commutation and other evaluator diagnostics, after one warmup. Batch P50/P95 and examples/s retain the configured batch size. MAC/FLOP counts cover Conv/Linear/GRU multiply-adds; process RSS includes the dataset and Python runtime.
