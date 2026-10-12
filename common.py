"""
Shared plumbing for the dataset generation scripts.

Each generate_*.py owns its dataset's actual idea/logic end to end. This module only
holds generic, idea-agnostic helpers reused by more than one of them: writing the
{train,valid,test}_{x,y}.npy + metadata layout, balancing label counts, and a procedural
noise texture for the two datasets that composite onto a canvas. The photo backgrounds
those two use live in their own generate_*.py files.
"""

import json
from pathlib import Path

import numpy as np


def save_dataset(out_dir, splits, num_classes, codename, benchmark=0.0, extra_meta=None):
    """
    splits: {"train": (x, y), "valid": (x, y), "test": (x, y)}
    Writes <out_dir>/{split}_x.npy, {split}_y.npy and a single-line `metadata` JSON file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        x, y = splits[split]
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        assert x.shape[0] == y.shape[0], f"{split}: x/y count mismatch"
        np.save(out_dir / f"{split}_x.npy", x)
        np.save(out_dir / f"{split}_y.npy", y)

    n_train = int(splits["train"][0].shape[0])
    shape = [int(s) for s in splits["train"][0].shape[1:]]
    meta = {
        "num_classes": int(num_classes),
        "input_shape": [n_train] + shape,
        "codename": codename,
        "benchmark": float(benchmark),
    }
    if extra_meta:
        meta.update(extra_meta)
    (out_dir / "metadata").write_text(json.dumps(meta))
    return meta


def balanced_labels(rng, n, num_classes):
    """n labels covering range(num_classes) as evenly as possible, shuffled."""
    base = n // num_classes
    rem = n - base * num_classes
    labels = np.repeat(np.arange(num_classes), base)
    if rem:
        labels = np.concatenate([labels, rng.choice(num_classes, size=rem, replace=False)])
    rng.shuffle(labels)
    return labels.astype(np.int64)


def _upsample(grid, size):
    """Bilinear upsample a small (r, r, C) grid of random values to (size, size, C)."""
    r = grid.shape[0]
    if r >= size:
        return grid[:size, :size]
    xs = np.linspace(0, r - 1, size, dtype=np.float32)
    x0 = np.floor(xs).astype(np.int64)
    x1 = np.minimum(x0 + 1, r - 1)
    tx = (xs - x0).reshape(-1, 1, 1)
    row = grid[x0] * (1 - tx) + grid[x1] * tx        # (size, r, C)
    y0, y1, ty = x0, x1, tx.reshape(1, -1, 1)
    col = row[:, y0] * (1 - ty) + row[:, y1] * ty     # (size, size, C)
    return col


def value_noise(rng, size):
    """Smooth colored fractal texture: sum of upsampled random grids at growing resolution."""
    out = np.zeros((size, size, 3), dtype=np.float32)
    amp, total = 1.0, 0.0
    res = 2
    while res <= size:
        out += amp * _upsample(rng.random((res, res, 3)).astype(np.float32), size)
        total += amp
        amp *= 0.55
        res *= 2
    out /= max(total, 1e-6)
    out *= rng.uniform(0.6, 1.0, size=3)               # random color tint
    lo, hi = out.min(), out.max()
    return ((out - lo) / max(hi - lo, 1e-6)).astype(np.float32)
