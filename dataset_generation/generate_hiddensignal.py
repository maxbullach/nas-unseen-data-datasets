"""
HiddenSignal: a steganalysis-style detection task.

Concept
-------
Every sample looks like an ordinary 64x64 RGB photo or texture -- backgrounds are real
photo crops (Places365 scene photos, see below) or procedural noise textures, picked
independently of the label. Into a random subset we additively embed one of five
low-amplitude spatial carrier patterns (amplitude 1.5-3.5%, a faint moire-like ripple
you'd likely miss unless you knew to compare against the clean version). The label
(6 classes) is *which* pattern was embedded, including "none".

The catch: in ordinary image classification the label is a property of the visible
scene. Here the two are decoupled by construction -- two images with wildly different
content share a label if they carry the same hidden carrier, and two images with
near-identical content differ in label if they don't. A network that leans on the usual
photographic priors (texture, edges, color, object-ish blobs) learns nothing; it has to
pick up a specific low-amplitude periodic residual riding on top of arbitrary content,
i.e. behave more like a frequency-domain detector than an image classifier.

Pattern classes (amplitude and phase randomized per sample within each class)
------------------------------------------------------------------------------
0 none              no signal added
1 horizontal grating  sin(2*pi*f*y + phase), low frequency, oriented along rows
2 vertical grating    sin(2*pi*f*x + phase), same frequency band, oriented along cols
3 diagonal grating    sin(2*pi*f*(x*cos45+y*sin45) + phase), higher frequency band
4 radial ripple       sin(2*pi*f*r + phase), r = distance from a random center
5 fine checkerboard   sign(sin(2*pi*fx*x)*sin(2*pi*fy*y)), near-Nyquist frequency

Photo backgrounds
-----------------
The photos are the Places365 validation set (36,500 scene photos at 256x256, ~525 MB),
downloaded once into a cache directory that CamoCount reuses. Crops are only ever
*down*sampled to 64x64, never upsampled: an interpolated low-res photo (e.g. CIFAR
upscaled) has almost no high-frequency energy, so the fine carriers would stick out
trivially. The pool is shuffled and cut into disjoint train/valid/test shares, so no
source photo appears in two splits, and within a split every photo background comes
from its own photo (each share is larger than the split's ~50% photo samples).
"""

import argparse
import itertools
import os
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from common import balanced_labels, save_dataset, value_noise

SIZE = 64
NUM_CLASSES = 6
CODENAME = "HiddenSignal"
FULL_SPLITS = (25000, 5000, 5000)       # train / valid / test with --full
SMOKE_SPLITS = (240, 60, 60)            # in-memory smoke test without --full
CLASS_NAMES = ["none", "h_grating", "v_grating", "d_grating", "radial", "checkerboard"]

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


def _grid():
    ys, xs = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    return xs / SIZE, ys / SIZE


_XS, _YS = _grid()


def _pattern(rng, cls):
    if cls == 0:
        return np.zeros((SIZE, SIZE), dtype=np.float32)
    phase = rng.uniform(0, 2 * np.pi)
    if cls == 1:                                        # horizontal grating (varies with y)
        f = rng.uniform(2.0, 4.0)
        return np.sin(2 * np.pi * f * _YS + phase)
    if cls == 2:                                         # vertical grating (varies with x)
        f = rng.uniform(2.0, 4.0)
        return np.sin(2 * np.pi * f * _XS + phase)
    if cls == 3:                                         # diagonal grating, higher band
        f = rng.uniform(5.0, 8.0)
        proj = (_XS * np.cos(np.pi / 4) + _YS * np.sin(np.pi / 4))
        return np.sin(2 * np.pi * f * proj + phase)
    if cls == 4:                                         # radial ripple from a random center
        cx, cy = rng.uniform(0.3, 0.7, size=2)
        r = np.hypot(_XS - cx, _YS - cy)
        f = rng.uniform(6.0, 10.0)
        return np.sin(2 * np.pi * f * r + phase)
    if cls == 5:                                         # fine checkerboard, near-Nyquist
        fx = rng.uniform(14.0, 20.0)
        fy = rng.uniform(14.0, 20.0)
        return np.sign(np.sin(2 * np.pi * fx * _XS + phase) * np.sin(2 * np.pi * fy * _YS))
    raise ValueError(cls)


def _components(rng, cls, photos):
    """The two parts of one sample before they are summed: background and hidden signal."""
    bg = _background(rng, SIZE, photos)                  # (H, W, 3) in [0, 1], label-independent
    pat = _pattern(rng, cls)
    amp = 0.0 if cls == 0 else rng.uniform(0.015, 0.035)
    return bg, amp * pat[:, :, None]


def _one_sample(rng, cls, photos):
    bg, signal = _components(rng, cls, photos)
    img = np.clip(bg + signal, 0.0, 1.0).astype(np.float32)
    return np.transpose(img, (2, 0, 1))                  # -> (3, H, W)


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
            x[i] = _one_sample(rng, int(lbl), photos)
        out[split] = (x, labels)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true", help="generate the full-size dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).parent / "data" / "HiddenSignal"))
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE),
                    help="where the Places365 photos are downloaded to (shared with CamoCount)")
    args = ap.parse_args()

    n_train, n_valid, n_test = FULL_SPLITS if args.full else SMOKE_SPLITS

    t0 = time.time()
    splits = build(n_train, n_valid, n_test, args.seed, args.cache_dir)
    print(f"generated in {time.time() - t0:.1f}s")

    x_tr, y_tr = splits["train"]
    none_mask = y_tr == 0
    assert x_tr[none_mask].std() > 0, "'none' samples should still vary (random backgrounds)"

    # Confirm the embedded signal is genuinely subtle: reconstruct a same-seed "none"
    # background for a signal-class sample and check the residual is small (imperceptible)
    # but non-zero (actually present).
    probe_photo = itertools.repeat(_places_files(args.cache_dir)[0])
    rng_a = np.random.default_rng(12345)
    rng_b = np.random.default_rng(12345)
    clean = _one_sample(rng_a, 0, probe_photo)
    signal_kind = 3
    dirty = _one_sample(rng_b, signal_kind, probe_photo)
    residual = np.abs(dirty - clean).mean()
    print(f"class-{signal_kind} vs clean residual: mean |delta|={residual:.4f} "
          f"(expected small but > 0)")
    assert 0.0005 < residual < 0.05, "embedded signal should be subtle, not dominant"

    for split, (x, y) in splits.items():
        assert x.shape[1:] == (3, SIZE, SIZE)
        assert x.min() >= 0.0 and x.max() <= 1.0
        assert set(np.unique(y).tolist()) <= set(range(NUM_CLASSES))
        print(f"{split}: {x.shape} labels {np.bincount(y, minlength=NUM_CLASSES).tolist()}")

    if args.full:
        meta = save_dataset(args.out, splits, NUM_CLASSES, CODENAME,
                             extra_meta={"class_names": CLASS_NAMES,
                                         "photo_source": "places365_val_256"})
        print("saved to", args.out, meta)
    else:
        print("smoke test OK -- rerun with --full to generate and save the full dataset")


if __name__ == "__main__":
    main()
