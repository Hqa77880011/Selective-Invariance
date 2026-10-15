import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import beta
from torch import nn
from torch.nn import functional as F

from .core import Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-pairs", required=True)
    parser.add_argument("--calibration-pairs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--risk", type=float, default=0.05)
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--minimum-coverage", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    teacher = Model(cfg, "erm")
    teacher.load_state_dict(payload["state"])
    teacher.eval()

    def features(path):
        arrays = dict(np.load(path, allow_pickle=False))
        with torch.no_grad():
            _, a = teacher(torch.tensor(arrays["x"]))
            _, b = teacher(torch.tensor(arrays["x_pair"]))
        x = torch.cat([a, b, a - b, torch.tensor(arrays["transform"], dtype=torch.float32)], -1)
        return x, torch.tensor(arrays["valid"], dtype=torch.float32)

    x, y = features(args.train_pairs)
    cx, cy = features(args.calibration_pairs)
    gates = []
    rng = np.random.default_rng(args.seed)
    for _ in range(3):
        gate = nn.Sequential(nn.Linear(x.shape[-1], 64), nn.GELU(), nn.Linear(64, 1))
        optimizer = torch.optim.AdamW(gate.parameters(), lr=0.001)
        selected = torch.tensor(rng.integers(len(x), size=len(x)))
        for _ in range(args.epochs):
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(gate(x[selected]).squeeze(-1), y[selected])
            loss.backward()
            optimizer.step()
        gates.append(gate)
    with torch.no_grad():
        estimates = torch.stack([gate(cx).sigmoid().squeeze(-1) for gate in gates])
    score = estimates.mean(0) - cfg["uncertainty_weight"] * estimates.std(0, unbiased=False)
    thresholds = np.linspace(0, 1, 41)
    feasible = []
    for threshold in thresholds:
        accepted = score.numpy() >= threshold
        n = int(accepted.sum())
        errors = int((cy.numpy()[accepted] < 0.5).sum())
        upper = (
            float(beta.ppf(1 - args.confidence / len(thresholds), errors + 1, n - errors))
            if n > errors
            else 1.0
        )
        coverage = n / len(cy)
        if coverage >= args.minimum_coverage and upper <= args.risk:
            feasible.append((coverage, float(threshold), upper))
    if not feasible:
        raise RuntimeError("No threshold meets the registered risk/coverage requirements")
    coverage, threshold, upper = max(feasible, key=lambda item: (item[0], -item[1]))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "teacher": teacher.state_dict(),
            "gates": [gate.state_dict() for gate in gates],
            "threshold": threshold,
            "coverage": coverage,
            "risk_upper": upper,
            "config": cfg,
        },
        output,
    )
    output.with_suffix(".json").write_text(
        json.dumps({"threshold": threshold, "coverage": coverage, "risk_upper": upper}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
