"""Finite RGB task with analytically audited sample-conditional invariance."""

import argparse
from pathlib import Path

import numpy as np


def objects(count, size, seed, group_offset):
    """Render independent colored objects and audit red/blue mixing by content."""
    if count < 3 or size < 8:
        raise ValueError("Use at least three objects and image size eight")
    rng = np.random.default_rng(seed)
    y = np.arange(count) % 3
    x = rng.uniform(0, 0.03, (count, 3, size, size)).astype(np.float32)
    strength = rng.choice(np.array([0.0, 0.25, 0.75, 1.0]), count)
    for index, label in enumerate(y):
        width = int(rng.integers(size // 3, size // 2 + 1))
        top, left = rng.integers(0, size - width + 1, 2)
        x[index, label, top:top + width, left:left + width] += rng.uniform(0.7, 0.9)
    swapped = x[:, [2, 1, 0]]
    pair = (1 - strength[:, None, None, None]) * x + strength[:, None, None, None] * swapped
    # The semantic construct is dominant object color, not the classifier prediction.
    transformed_label = np.where((strength > 0.5) & (y != 1), 2 - y, y)
    return dict(x=x, x_pair=pair.astype(np.float32), y=y.astype(np.int64),
                transform=strength[:, None].astype(np.float32),
                valid=(transformed_label == y).astype(np.int64),
                group=np.arange(group_offset, group_offset + count))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--objects-per-split", type=int, default=300)
    parser.add_argument("--size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty output directory: " + str(output))
    roles = ["teacher/train", "teacher/validation", "audit_train", "audit_calibration",
             "task/train", "task/validation", "task/test"]
    for index, role in enumerate(roles):
        target = output / (role + ".npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **objects(
            args.objects_per_split, args.size, args.seed + index,
            index * args.objects_per_split,
        ))


if __name__ == "__main__":
    main()
