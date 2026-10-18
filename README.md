# Selective Invariance

Implementation of **Risk-controlled consistency for transformation-example pairs**.

## Installation

Use Python `>=3.11`. Run commands from the repository root. Create the environment, activate it with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (PowerShell), then install the package.

```bash
python -m venv .venv
```

```bash
python -m pip install -e .
```

## Data preparation

The controlled generator creates geometric objects, transformation pairs and disjoint teacher, gate-training, calibration and task partitions. These are generated images with independently declared validity labels.

```bash
python -m selective.controlled --objects-per-split 300 --size 24 --seed 17 --output data/reproduce
```

## Training

Use the same `configs/reproduce.json` throughout. Its seed is `17`; preserve the prepared splits and use separate output directories for new seeds.

```bash
python -m selective train --config configs/reproduce.json --data data/reproduce/teacher --methods erm --output runs/reproduce_teacher
python -m selective.gate --checkpoint runs/reproduce_teacher/erm/model.pt --train-pairs data/reproduce/audit_train.npz --calibration-pairs data/reproduce/audit_calibration.npz --risk 0.1 --minimum-coverage 0.1 --output runs/reproduce_gate.pt
python -m selective suite --config configs/reproduce.json --data data/reproduce/task --output runs/reproduce
```

## Evaluation

```bash
python -m selective evaluate --checkpoint runs/reproduce/selective/model.pt --data data/reproduce/task/test.npz --output results/reproduce
```

## Outputs

Each training method directory contains `model.pt` and `history.jsonl`. Evaluation writes `predictions.npz` and `metrics.json`; plots read the supplied run directories.

```bash
python -m selective plot --inputs runs/reproduce/selective results/reproduce --output figures
```

## Verification

Source checks used Python 3.13.5. Installed dependency metadata: numpy 2.1.3, torch 2.7.1+cu118, torchvision 0.22.1, scipy 1.15.3, scikit-learn 1.6.1, matplotlib 3.10.0. Python AST, configuration files, CLI parser definitions and source input/output paths were checked without invoking command handlers. Training, full evaluation and CI were not run for this review.
