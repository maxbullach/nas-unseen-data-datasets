"""
MelodyContour: symbolic music, not audio.

Concept
-------
Every sample is a randomly composed monophonic melody rendered as a piano-roll: a
sparse binary (2, 48, 64) grid whose axes are pitch (48 semitones) and time (64 steps),
not the x/y coordinates of a picture. Channel 0 marks note onsets, channel 1 marks the
full duration a note is held. The label (6 classes) is the musical mode the melody was
generated from (Major, Natural Minor, Dorian, Phrygian, Whole Tone, Major Pentatonic).

The catch: this is a sparse event grid (>90% zeros) whose label depends on a
*modular-arithmetic* property of the active rows -- which pitch classes (row % 12)
appear -- rather than on texture, edges or blob shape. A network has to learn something
like "which of these 6 semitone subsets does the set of lit rows belong to", robust to
the melody being transposed (shifted along the pitch axis) and to its rhythm (which
columns are lit) being different every time.

Label construction
-------------------
Pick a mode and a random root, then take a random walk over *scale degrees* (not raw
semitones), so every note the walk visits belongs to the chosen mode; each note gets a
random duration of 1..4 steps until the 64-step grid is filled. Because the walk moves
over degrees, the same code handles 5-, 6- and 7-note modes without special-casing.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from common import balanced_labels, save_dataset

N_PITCHES = 48
N_STEPS = 64
CODENAME = "MelodyContour"
FULL_SPLITS = (25000, 5000, 5000)       # train / valid / test with --full
SMOKE_SPLITS = (300, 60, 60)            # in-memory smoke test without --full

MODES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "natural_minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "whole_tone": [0, 2, 4, 6, 8, 10],
    "major_pentatonic": [0, 2, 4, 7, 9],
}
MODE_NAMES = list(MODES.keys())
NUM_CLASSES = len(MODE_NAMES)


def _one_melody(rng, mode_idx):
    scale = MODES[MODE_NAMES[mode_idx]]
    n_deg = len(scale)
    n_octaves = N_PITCHES // 12 + 1

    root = int(rng.integers(0, 12))                    # root pitch class, octave-independent
    degree = int(rng.integers(1, n_deg * (n_octaves - 1)))  # start mid-range, room to wander

    onset = np.zeros((N_PITCHES, N_STEPS), dtype=np.float32)
    sustain = np.zeros((N_PITCHES, N_STEPS), dtype=np.float32)

    t = 0
    step_choices = np.array([-2, -1, -1, 0, 1, 1, 2])
    while t < N_STEPS:
        octave, idx = divmod(degree, n_deg)
        pitch = root + 12 * octave + scale[idx]
        if not (0 <= pitch < N_PITCHES):
            degree = n_deg * (n_octaves // 2)           # bounced out of range: recentre
            continue

        dur = int(rng.integers(1, 5))
        end = min(t + dur, N_STEPS)
        onset[pitch, t] = 1.0
        sustain[pitch, t:end] = 1.0
        t = end

        degree += int(rng.choice(step_choices))
        degree = int(np.clip(degree, 0, n_deg * n_octaves - 1))

    return np.stack([onset, sustain], axis=0)


def build(n_train, n_valid, n_test, seed):
    rng = np.random.default_rng(seed)
    out = {}
    for split, n in (("train", n_train), ("valid", n_valid), ("test", n_test)):
        labels = balanced_labels(rng, n, NUM_CLASSES)
        x = np.empty((n, 2, N_PITCHES, N_STEPS), dtype=np.float32)
        for i, lbl in enumerate(labels):
            x[i] = _one_melody(rng, int(lbl))
        out[split] = (x, labels)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true", help="generate the full-size dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).parent / "data" / "MelodyContour"))
    args = ap.parse_args()

    n_train, n_valid, n_test = FULL_SPLITS if args.full else SMOKE_SPLITS

    t0 = time.time()
    splits = build(n_train, n_valid, n_test, args.seed)
    print(f"generated in {time.time() - t0:.1f}s")

    for split, (x, y) in splits.items():
        assert x.shape[1:] == (2, N_PITCHES, N_STEPS)
        assert set(np.unique(y).tolist()) <= set(range(NUM_CLASSES))
        assert np.all((x == 0) | (x == 1)), "piano-roll must be binary"
        assert np.all(x[:, 0] <= x[:, 1]), "onset must imply sustain at the same cell"
        occupancy = x[:, 1].mean()
        print(f"{split}: {x.shape} sustain-occupancy={occupancy:.3f} "
              f"labels {np.bincount(y, minlength=NUM_CLASSES).tolist()}")

    if args.full:
        meta = save_dataset(args.out, splits, NUM_CLASSES, CODENAME,
                             extra_meta={"class_names": MODE_NAMES})
        print("saved to", args.out, meta)
    else:
        print("smoke test OK -- rerun with --full to generate and save the full dataset")


if __name__ == "__main__":
    main()
