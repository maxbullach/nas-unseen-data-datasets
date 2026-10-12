"""
PendulumChaos: the trace of a chaotic dynamical system.

Concept
-------
We integrate a double pendulum (equal masses and lengths, standard Lagrangian equations
of motion) from a random initial condition for 8 simulated seconds, then rasterize the
path traced by the outer bob as a 64x64 RGB motion trail whose color runs from blue
(t=0) to red (t=end), so the direction of travel survives being flattened into a static
image. The label (6 classes) is the net number of full turns the outer arm made around
its pivot (the inner bob) during the run: 0, 1, 2, 3, 4 or 5+.

The catch: a double pendulum is continuous, globally coupled and chaotically sensitive.
Two initial conditions that render as nearly identical trails can end up in different
classes, and two very different-looking trails can share one. The image itself has
smooth, natural-looking low-level statistics -- nothing about it looks "encoded" -- yet
the label is a global topological property of the trajectory (how often it wound
around), not a property of local texture.

Label construction
-------------------
The turn count is computed from the physics (numpy.unwrap on the outer arm's angle,
|end - start| // 2*pi), not estimated from the rendered pixels, so it is exact. The
natural distribution is heavily skewed (most random starts barely swing), so we
simulate in batches and keep drawing until every class quota is filled. Simulation is
vectorized batch RK4 (~0.04 ms/sample), so this stays fast even though the rarest class
needs roughly 25x more raw draws than the most common one.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from common import save_dataset

SIZE = 64
NUM_CLASSES = 6                 # 0, 1, 2, 3, 4, 5+ flips
CODENAME = "PendulumChaos"
FULL_SPLITS = (25000, 5000, 5000)       # train / valid / test with --full
SMOKE_SPLITS = (120, 30, 30)            # in-memory smoke test without --full
DT = 0.02
STEPS = 400                     # 8 simulated seconds
G = 9.81


def _deriv(theta1, theta2, omega1, omega2):
    delta = theta1 - theta2
    den = 3.0 - np.cos(2.0 * delta)
    domega1 = (-3 * G * np.sin(theta1) - G * np.sin(theta1 - 2 * theta2)
               - 2 * np.sin(delta) * (omega2 ** 2 + omega1 ** 2 * np.cos(delta))) / den
    domega2 = (2 * np.sin(delta) * (2 * omega1 ** 2 + 2 * G * np.cos(theta1)
               + omega2 ** 2 * np.cos(delta))) / den
    return omega1, omega2, domega1, domega2


def _integrate(theta1, theta2, omega1, omega2, dt, steps):
    """Batched fixed-step RK4. All args (n,); returns traj_theta1, traj_theta2: (steps, n)."""
    n = len(theta1)
    traj1 = np.empty((steps, n), dtype=np.float32)
    traj2 = np.empty((steps, n), dtype=np.float32)
    state = (theta1.copy(), theta2.copy(), omega1.copy(), omega2.copy())
    for s in range(steps):
        traj1[s], traj2[s] = state[0], state[1]
        k1 = _deriv(*state)
        s2 = tuple(st + dt / 2 * k for st, k in zip(state, k1))
        k2 = _deriv(*s2)
        s3 = tuple(st + dt / 2 * k for st, k in zip(state, k2))
        k3 = _deriv(*s3)
        s4 = tuple(st + dt * k for st, k in zip(state, k3))
        k4 = _deriv(*s4)
        state = tuple(st + dt / 6 * (a + 2 * b + 2 * c + d)
                       for st, a, b, c, d in zip(state, k1, k2, k3, k4))
    return traj1, traj2


def _simulate_batch(rng, n):
    theta1 = rng.uniform(-np.pi, np.pi, n).astype(np.float32)
    theta2 = rng.uniform(-np.pi, np.pi, n).astype(np.float32)
    omega1 = rng.uniform(-0.5, 0.5, n).astype(np.float32)
    omega2 = rng.uniform(-0.5, 0.5, n).astype(np.float32)
    traj1, traj2 = _integrate(theta1, theta2, omega1, omega2, DT, STEPS)
    unwrapped = np.unwrap(traj2, axis=0)
    total_rot = unwrapped[-1] - unwrapped[0]
    flips = np.clip((np.abs(total_rot) // (2 * np.pi)).astype(np.int64), 0, NUM_CLASSES - 1)
    return traj1, traj2, flips


def _render(theta1, theta2):
    """theta1, theta2: (points,) subsampled trajectory of the outer bob -> (3, H, W)."""
    x1, y1 = np.sin(theta1), -np.cos(theta1)
    x2, y2 = x1 + np.sin(theta2), y1 - np.cos(theta2)
    scale = 14.0
    px = np.clip((x2 * scale + SIZE / 2).astype(np.int32), 0, SIZE - 1)
    py = np.clip((y2 * scale + SIZE / 2).astype(np.int32), 0, SIZE - 1)

    canvas = np.zeros((SIZE, SIZE, 3), dtype=np.float32)
    t = np.linspace(0.0, 1.0, len(theta1), dtype=np.float32)
    colors = np.stack([0.2 + 0.7 * t, 0.2 * np.ones_like(t), 0.9 - 0.7 * t], axis=1)
    for i in range(len(theta1)):
        y0, y1_ = max(0, py[i] - 1), min(SIZE, py[i] + 2)
        x0, x1_ = max(0, px[i] - 1), min(SIZE, px[i] + 2)
        canvas[y0:y1_, x0:x1_] += colors[i] * 0.6
    canvas = np.clip(canvas, 0.0, 1.0)
    return np.transpose(canvas, (2, 0, 1)).astype(np.float32)


def _collect_balanced(rng, quota_per_class, batch_size=8000, max_batches=200):
    """Simulate batches until every class has >= quota_per_class trajectories, keyed by
    (traj1, traj2) column slices so the caller can render exactly the ones it needs."""
    buckets = {c: [] for c in range(NUM_CLASSES)}
    counts = {c: 0 for c in range(NUM_CLASSES)}
    for _ in range(max_batches):
        if all(counts[c] >= quota_per_class for c in range(NUM_CLASSES)):
            break
        traj1, traj2, flips = _simulate_batch(rng, batch_size)
        for c in range(NUM_CLASSES):
            if counts[c] >= quota_per_class:
                continue
            idx = np.where(flips == c)[0]
            need = quota_per_class - counts[c]
            idx = idx[:need]
            for i in idx:
                buckets[c].append((traj1[:, i], traj2[:, i]))
            counts[c] += len(idx)
    for c in range(NUM_CLASSES):
        assert counts[c] >= quota_per_class, (
            f"class {c} only got {counts[c]}/{quota_per_class} samples "
            f"after {max_batches} batches -- widen initial-condition ranges")
    return buckets


def build(n_train, n_valid, n_test, seed):
    rng = np.random.default_rng(seed)
    per_class_total = -(-(n_train + n_valid + n_test) // NUM_CLASSES)  # ceil
    buckets = _collect_balanced(rng, per_class_total)

    labels = np.concatenate([np.full(per_class_total, c, dtype=np.int64)
                              for c in range(NUM_CLASSES)])
    trajs = [t for c in range(NUM_CLASSES) for t in buckets[c][:per_class_total]]
    order = rng.permutation(len(labels))
    labels, trajs = labels[order], [trajs[i] for i in order]

    n_total = n_train + n_valid + n_test
    labels, trajs = labels[:n_total], trajs[:n_total]

    x = np.empty((n_total, 3, SIZE, SIZE), dtype=np.float32)
    for i, (t1, t2) in enumerate(trajs):
        x[i] = _render(t1[::2], t2[::2])              # subsample 400 -> 200 points

    out = {}
    lo = 0
    for split, n in (("train", n_train), ("valid", n_valid), ("test", n_test)):
        out[split] = (x[lo:lo + n], labels[lo:lo + n])
        lo += n
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true", help="generate the full-size dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).parent / "data" / "PendulumChaos"))
    args = ap.parse_args()

    n_train, n_valid, n_test = FULL_SPLITS if args.full else SMOKE_SPLITS

    t0 = time.time()
    splits = build(n_train, n_valid, n_test, args.seed)
    print(f"generated in {time.time() - t0:.1f}s")

    for split, (x, y) in splits.items():
        assert x.shape[1:] == (3, SIZE, SIZE)
        assert x.min() >= 0.0 and x.max() <= 1.0
        assert set(np.unique(y).tolist()) <= set(range(NUM_CLASSES))
        print(f"{split}: {x.shape} labels {np.bincount(y, minlength=NUM_CLASSES).tolist()}")

    if args.full:
        meta = save_dataset(args.out, splits, NUM_CLASSES, CODENAME,
                             extra_meta={"class_names": ["0", "1", "2", "3", "4", "5+"]})
        print("saved to", args.out, meta)
    else:
        print("smoke test OK -- rerun with --full to generate and save the full dataset")


if __name__ == "__main__":
    main()
