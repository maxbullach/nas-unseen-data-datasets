"""
CamoCount: count the camouflaged circles, ignore the camouflaged squares.

Concept
-------
A 96x96 RGB scene on a real photo crop or a procedural texture background. Scattered
across it are two kinds of low-contrast soft blobs, each colored to sit close to its
local background (the local mean plus a small random contrast delta, not a fixed
foreground color) and drawn in random order, so they can partially occlude each other:

  * circles -- these are what the label counts (0..12 -> 13 classes)
  * squares -- distractors, drawn in an independent random quantity (0..8) every sample
    and never counted

The catch: three failure modes stacked on top of each other. The number of relevant
objects is variable and unknown ahead of time (counting, not a fixed set of slots),
each object is only weakly visible (camouflage, not a bold foreground object), and a
same-family distractor shape must be learned to *not* count (shape selectivity, not
"is there something here").

Photo backgrounds
-----------------
The photos are the Places365 validation set (36,500 scene photos at 256x256, ~525 MB),
downloaded once into a cache directory that HiddenSignal reuses. Crops are only ever
*down*sampled to 96x96, never upsampled, so the camouflage sits on real photographic
texture rather than an interpolation blur. The pool is shuffled and cut into disjoint
train/valid/test shares, so no source photo appears in two splits, and within a split
every photo background comes from its own photo (each share is larger than the split's
~50% photo samples).
"""

import argparse
import os
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from common import balanced_labels, save_dataset, value_noise

SIZE = 96
MAX_TARGETS = 12
NUM_CLASSES = MAX_TARGETS + 1          # 0..12 circles
CODENAME = "CamoCount"
FULL_SPLITS = (25000, 5000, 5000)       # train / valid / test with --full
SMOKE_SPLITS = (200, 40, 40)            # in-memory smoke test without --full

PHOTO_PROB = 0.5                # share of backgrounds that are photos (rest: value noise)
PLACES_URL = "http://data.csail.mit.edu/places/places365/val_256.tar"
PLACES_COUNT = 36500
DEFAULT_CACHE = Path(__file__).parent / ".image_cache"


def _places_files(cache_dir):
    """Sorted JPEG paths of Places365-val, downloading + extracting it into cache_dir on
    first use. Every step lands via an atomic rename, so an interrupted run never leaves
    a half-written cache that a later run would mistake for a complete one."""
    cache_dir = Path(cache_dir)
    root = cache_dir / "places365_val_256"
    if not (root / ".complete").exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        tar_path = cache_dir / "val_256.tar"
        if not tar_path.exists():
            part = cache_dir / f"val_256.tar.part{os.getpid()}"
            print(f"downloading {PLACES_URL} (~525 MB) -> {tar_path}")
            urllib.request.urlretrieve(PLACES_URL, part)
            part.replace(tar_path)
        staging = cache_dir / f"places365_val_256.part{os.getpid()}"
        print(f"extracting {tar_path}")
        with tarfile.open(tar_path) as tf:
            tf.extractall(staging, filter="data")
        (staging / ".complete").touch()
        if (root / ".complete").exists():           # a concurrent run finished first
            shutil.rmtree(staging)
        else:
            shutil.rmtree(root, ignore_errors=True)
            staging.replace(root)
        tar_path.unlink(missing_ok=True)
    files = sorted(root.rglob("*.jpg"))
    assert len(files) == PLACES_COUNT, f"expected {PLACES_COUNT} photos in {root}, found {len(files)}"
    return files


def _photo_pools(rng, files, sizes):
    """Shuffle the photos and cut them into disjoint per-split pools, proportional to the
    split sizes, so a source photo never shows up in more than one split."""
    order = rng.permutation(len(files))
    bounds = np.cumsum([0] + list(sizes.values())) * len(files) // sum(sizes.values())
    return {split: [files[i] for i in order[lo:hi]]
            for split, lo, hi in zip(sizes, bounds[:-1], bounds[1:])}


def _photo_stream(rng, pool):
    """Each photo of the pool once in random order; only reshuffles and repeats if a split
    asks for more photo backgrounds than its pool holds."""
    while True:
        for i in rng.permutation(len(pool)):
            yield pool[i]


def _photo_crop(rng, path, size):
    """Random square crop of one photo, downsampled to size x size (never upsampled), with
    random flips / 90-degree turns and a mild brightness/contrast jitter."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        side = int(rng.integers(size, min(h, w) + 1))
        x0 = int(rng.integers(0, w - side + 1))
        y0 = int(rng.integers(0, h - side + 1))
        im = im.crop((x0, y0, x0 + side, y0 + side))
        if side != size:
            im = im.resize((size, size), Image.LANCZOS)
        arr = np.asarray(im, dtype=np.float32) / 255.0

    if rng.random() < 0.5:
        arr = arr[:, ::-1]
    if rng.random() < 0.5:
        arr = arr[::-1]
    arr = np.rot90(arr, int(rng.integers(0, 4)))
    brightness = rng.uniform(-0.1, 0.1)
    contrast = rng.uniform(0.85, 1.15)
    arr = (arr - 0.5) * contrast + 0.5 + brightness
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _background(rng, size, photos):
    """HxWx3 float32 in [0, 1]: the next photo of the split's stream, or a noise texture."""
    if rng.random() < PHOTO_PROB:
        return _photo_crop(rng, next(photos), size)
    return value_noise(rng, size)


def _draw_shape(canvas, rng, cx, cy, r, is_circle):
    """Alpha-blend one soft-edged, background-colored blob into `canvas` in place."""
    h, w, _ = canvas.shape
    pad = r + 2
    x0, x1 = max(0, cx - pad), min(w, cx + pad)
    y0, y1 = max(0, cy - pad), min(h, cy + pad)
    if x1 <= x0 or y1 <= y0:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)

    if is_circle:
        dist = np.hypot(xs - cx, ys - cy)
        mask = np.clip(1.0 - (dist - r) / 2.0, 0.0, 1.0)
    else:
        dist = np.maximum(np.abs(xs - cx), np.abs(ys - cy))
        mask = np.clip(1.0 - (dist - r) / 2.0, 0.0, 1.0)

    patch = canvas[y0:y1, x0:x1]
    local_color = patch.reshape(-1, 3).mean(axis=0)
    delta = rng.uniform(0.08, 0.25) * (1 if rng.random() < 0.5 else -1)
    color = np.clip(local_color + delta, 0.0, 1.0)

    m = mask[:, :, None]
    canvas[y0:y1, x0:x1] = patch * (1 - m) + color[None, None, :] * m


def _one_scene(rng, n_targets, photos):
    canvas = _background(rng, SIZE, photos)
    n_distractors = int(rng.integers(0, 9))            # independent of the label, by design

    shapes = [("circle", True) for _ in range(n_targets)] + \
             [("square", False) for _ in range(n_distractors)]
    rng.shuffle(shapes)

    margin = 10
    for _, is_circle in shapes:
        cx = int(rng.integers(margin, SIZE - margin))
        cy = int(rng.integers(margin, SIZE - margin))
        r = int(rng.integers(3, 9))
        _draw_shape(canvas, rng, cx, cy, r, is_circle)

    return np.transpose(canvas, (2, 0, 1)).astype(np.float32)   # -> (3, H, W)


def build(n_train, n_valid, n_test, seed, cache_dir=DEFAULT_CACHE):
    rng = np.random.default_rng(seed)
    sizes = {"train": n_train, "valid": n_valid, "test": n_test}
    pools = _photo_pools(rng, _places_files(cache_dir), sizes)
    print("disjoint photo pools: " + ", ".join(f"{k} {len(v)}" for k, v in pools.items()))
    out = {}
    for split, n in sizes.items():
        photos = _photo_stream(rng, pools[split])
        labels = balanced_labels(rng, n, NUM_CLASSES)
        x = np.empty((n, 3, SIZE, SIZE), dtype=np.float32)
        for i, lbl in enumerate(labels):
            x[i] = _one_scene(rng, int(lbl), photos)
        out[split] = (x, labels)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true", help="generate the full-size dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).parent / "data" / "CamoCount"))
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE),
                    help="where the Places365 photos are downloaded to (shared with HiddenSignal)")
    args = ap.parse_args()

    n_train, n_valid, n_test = FULL_SPLITS if args.full else SMOKE_SPLITS

    t0 = time.time()
    splits = build(n_train, n_valid, n_test, args.seed, args.cache_dir)
    dt = time.time() - t0
    print(f"generated in {dt:.1f}s ({dt / sum(len(y) for _, y in splits.values()) * 1000:.2f} ms/sample)")

    for split, (x, y) in splits.items():
        assert x.shape[1:] == (3, SIZE, SIZE)
        assert x.min() >= 0.0 and x.max() <= 1.0
        assert set(np.unique(y).tolist()) <= set(range(NUM_CLASSES))
        print(f"{split}: {x.shape} labels {np.bincount(y, minlength=NUM_CLASSES).tolist()}")

    if args.full:
        meta = save_dataset(args.out, splits, NUM_CLASSES, CODENAME,
                             extra_meta={"photo_source": "places365_val_256"})
        print("saved to", args.out, meta)
    else:
        print("smoke test OK -- rerun with --full to generate and save the full dataset")


if __name__ == "__main__":
    main()
