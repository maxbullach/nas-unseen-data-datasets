"""
GraphSpectra: an image that is actually a graph.

Concept
-------
Each sample is the 40x40 binary adjacency matrix of an undirected graph on 40 nodes,
stored as a single-channel (1, 40, 40) image. The label is the number of connected
components of the graph (1..8, stored as 0..7).

The catch: convolution assumes that nearby pixels are related and that patterns are
roughly translation-invariant. Here row/column i is simply node i, and node identity is
an arbitrary integer with no spatial meaning: permuting the node order (rows and
columns together) produces a different-looking 40x40 image that encodes the exact same
graph and must get the exact same label. Every sample is randomly permuted, so a
network cannot shortcut on *where* pixels are lit (e.g. blocks near the diagonal) and
has to learn something closer to a permutation-robust connectivity test.

Label construction (exact, not estimated)
------------------------------------------
For a target label k in {1..8}, partition the 40 nodes into exactly k non-empty groups,
wire each group into a random spanning tree (so the group is internally connected) plus
a random number of extra intra-group edges (for structural variety), add no edges
between groups, and finally permute the node order. The component count equals k by
construction -- no counting or rejection sampling needed, and classes come out
perfectly balanced. The smoke test re-checks this with an independent BFS.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from common import balanced_labels, save_dataset

NUM_NODES = 40
NUM_CLASSES = 8            # label c means (c + 1) connected components
CODENAME = "GraphSpectra"
FULL_SPLITS = (25000, 5000, 5000)       # train / valid / test with --full
SMOKE_SPLITS = (300, 60, 60)            # in-memory smoke test without --full


def _random_partition(rng, n, k):
    """n items into k non-empty groups; returns an array of group ids, length n."""
    groups = np.empty(n, dtype=np.int64)
    order = rng.permutation(n)
    groups[order[:k]] = np.arange(k)                 # seed: one node per group
    if n > k:
        groups[order[k:]] = rng.integers(0, k, size=n - k)
    return groups


def _one_graph(rng, k):
    groups = _random_partition(rng, NUM_NODES, k)
    adj = np.zeros((NUM_NODES, NUM_NODES), dtype=np.float32)

    for g in range(k):
        members = np.where(groups == g)[0]
        if len(members) == 1:
            continue
        perm = rng.permutation(members)               # random spanning tree over the group
        for i in range(1, len(perm)):
            parent = perm[rng.integers(0, i)]
            child = perm[i]
            adj[parent, child] = adj[child, parent] = 1.0

        # a handful of extra intra-group edges so components aren't all bare trees
        m = len(members)
        max_extra = m * (m - 1) // 2 - (m - 1)
        if max_extra > 0:
            n_extra = rng.integers(0, max_extra + 1)
            for _ in range(n_extra):
                a, b = rng.choice(members, size=2, replace=False)
                adj[a, b] = adj[b, a] = 1.0

    perm = rng.permutation(NUM_NODES)                  # destroy the block-diagonal layout
    adj = adj[perm][:, perm]
    return adj


def build(n_train, n_valid, n_test, seed):
    rng = np.random.default_rng(seed)
    out = {}
    for split, n in (("train", n_train), ("valid", n_valid), ("test", n_test)):
        labels = balanced_labels(rng, n, NUM_CLASSES)
        x = np.empty((n, 1, NUM_NODES, NUM_NODES), dtype=np.float32)
        for i, lbl in enumerate(labels):
            x[i, 0] = _one_graph(rng, int(lbl) + 1)
        out[split] = (x, labels)
    return out


def _sanity_check(adj, expected_k):
    """Plain BFS component count, used only to verify the constructive build is correct."""
    n = adj.shape[0]
    seen = np.zeros(n, dtype=bool)
    comps = 0
    for start in range(n):
        if seen[start]:
            continue
        comps += 1
        stack = [start]
        seen[start] = True
        while stack:
            u = stack.pop()
            for v in np.where(adj[u] > 0)[0]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
    assert comps == expected_k, f"expected {expected_k} components, BFS found {comps}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true", help="generate the full-size dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).parent / "data" / "GraphSpectra"))
    args = ap.parse_args()

    n_train, n_valid, n_test = FULL_SPLITS if args.full else SMOKE_SPLITS

    t0 = time.time()
    splits = build(n_train, n_valid, n_test, args.seed)
    print(f"generated in {time.time() - t0:.1f}s")

    rng = np.random.default_rng(999)
    x_tr, y_tr = splits["train"]
    check_idx = rng.choice(len(x_tr), size=min(40, len(x_tr)), replace=False)
    for i in check_idx:
        _sanity_check(x_tr[i, 0], int(y_tr[i]) + 1)
    print(f"BFS component-count sanity check passed on {len(check_idx)} samples")

    for split, (x, y) in splits.items():
        assert x.shape[1:] == (1, NUM_NODES, NUM_NODES)
        assert set(np.unique(y).tolist()) <= set(range(NUM_CLASSES))
        assert np.array_equal(x, np.transpose(x, (0, 1, 3, 2))), "adjacency must be symmetric"
        assert np.all(x[:, 0][:, np.arange(NUM_NODES), np.arange(NUM_NODES)] == 0), "no self-loops"
        print(f"{split}: {x.shape} labels {np.bincount(y, minlength=NUM_CLASSES).tolist()}")

    if args.full:
        meta = save_dataset(args.out, splits, NUM_CLASSES, CODENAME)
        print("saved to", args.out, meta)
    else:
        print("smoke test OK -- rerun with --full to generate and save the full dataset")


if __name__ == "__main__":
    main()
