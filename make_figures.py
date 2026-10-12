"""
Render the illustrations in images/ from real dataset samples.

Every picture in these figures is an actual training sample of the datasets that
`python generate_<name>.py --full` (seed 0) writes to data/<Name>/ -- generate those
first. Where a figure needs a generator's internals (HiddenSignal's clean background and
hidden pattern, CamoCount's shape positions, PendulumChaos's trajectory), the generator
is re-run with light-weight hooks just until the shown samples exist, and every
reproduced sample is checked to be byte-identical to the saved one.

    python make_figures.py                  # all five figures
    python make_figures.py camocount        # only some of them
"""

import argparse
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

import generate_camocount  # noqa: E402
import generate_graphspectra  # noqa: E402
import generate_hiddensignal  # noqa: E402
import generate_melodycontour  # noqa: E402
import generate_pendulumchaos  # noqa: E402

HERE = Path(__file__).parent
DPI = 200

BG = "#0d2036"
FRAME = "#3b4f66"
INK = "#e6e9ee"
MUTED = "#8a9bb0"
ORANGE = "#f5a83a"
TEAL = "#5aa7b0"
CELL_OFF = "#050d18"
# categorical slots in fixed order (components of a graph)
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]

plt.rcParams.update({
    "font.family": "DejaVu Sans Mono",
    "font.size": 11,
    "text.color": INK,
    "axes.edgecolor": FRAME,
    "axes.labelcolor": MUTED,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
})


# ---------------------------------------------------------------------------
# data access and generator replay
# ---------------------------------------------------------------------------

def _load(data_dir, codename):
    d = Path(data_dir) / codename
    if not (d / "train_x.npy").exists():
        raise SystemExit(f"{d} is missing -- run `python generate_{codename.lower()}.py --full` first")
    return np.load(d / "train_x.npy", mmap_mode="r"), np.load(d / "train_y.npy")


def _first(y, label):
    """Index of the first training sample with this label."""
    return int(np.flatnonzero(y == label)[0])


class _Enough(Exception):
    """Raised from a hook to stop a generator once the samples a figure needs exist."""


@contextmanager
def _hooked(mod, wrappers):
    originals = {name: getattr(mod, name) for name in wrappers}
    for name, wrap in wrappers.items():
        setattr(mod, name, wrap(originals[name]))
    try:
        yield
    finally:
        for name, fn in originals.items():
            setattr(mod, name, fn)


def _replay(mod, sample_fn, n, record, hooks=None):
    """Re-run mod.build() exactly as `--full` (seed 0) does and return its first n train
    samples. `sample_fn` is the per-sample function build() calls; the extra `hooks` write
    whatever internals they observe into `record` while that sample is being made."""
    samples = []

    def wrap_sample(fn):
        def run(*args):
            record.clear()
            out = fn(*args)
            samples.append(dict(record, x=out, args=args))
            if len(samples) == n:
                raise _Enough
            return out
        return run

    with _hooked(mod, {sample_fn: wrap_sample, **(hooks or {})}):
        try:
            mod.build(*mod.FULL_SPLITS, 0)
        except _Enough:
            pass
    return samples


def _assert_saved(x_saved, samples, indices):
    for i in indices:
        assert np.array_equal(np.asarray(x_saved[i]), samples[i]["x"]), \
            f"replayed train sample {i} differs from the saved dataset -- regenerate data/ first"


def _record_photo(record):
    def wrap(fn):
        def run(rng, path, size):
            record["photo"] = Path(path).name
            return fn(rng, path, size)
        return run
    return wrap


# ---------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------

def _frame(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(FRAME)
        spine.set_linewidth(1.2)


def _show(ax, chw_or_hwc, title=None, hwc=False):
    img = np.asarray(chw_or_hwc)
    if not hwc:
        img = np.transpose(img, (1, 2, 0))
    ax.imshow(np.clip(img, 0, 1), interpolation="nearest")
    _frame(ax)
    if title:
        ax.set_title(title, pad=8, linespacing=1.5)


def _save(fig, name):
    path = HERE / "images" / f"{name}.png"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print("wrote", path)


def _between(fig, ax_left, ax_right, symbol):
    """Big operator symbol centred in the gap between two axes."""
    l, r = ax_left.get_position(), ax_right.get_position()
    fig.text((l.x1 + r.x0) / 2, (l.y0 + l.y1) / 2, symbol, color=ORANGE,
             fontsize=26, ha="center", va="center", fontweight="bold")


# ---------------------------------------------------------------------------
# GraphSpectra
# ---------------------------------------------------------------------------

def _components(adj):
    """Connected components as lists of nodes, each in BFS order, largest first."""
    n = adj.shape[0]
    seen = np.zeros(n, dtype=bool)
    comps = []
    for start in range(n):
        if seen[start]:
            continue
        seen[start] = True
        order, queue = [], [start]
        while queue:
            u = queue.pop(0)
            order.append(u)
            for v in np.flatnonzero(adj[u]):
                if not seen[v]:
                    seen[v] = True
                    queue.append(v)
        comps.append(order)
    return sorted(comps, key=len, reverse=True)


def _spring_layout(adj, nodes, rng, iters=300):
    """Fruchterman-Reingold layout of one component, centred, radius ~1."""
    m = len(nodes)
    if m == 1:
        return np.zeros((1, 2))
    sub = adj[np.ix_(nodes, nodes)] > 0
    pos = rng.normal(size=(m, 2))
    k = 1.0 / np.sqrt(m)
    temp = 0.2
    for _ in range(iters):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.maximum(np.linalg.norm(delta, axis=-1), 1e-3)
        force = (k * k / dist ** 2)[..., None] * delta                  # repulsion
        force -= (sub * dist / k)[..., None] * delta                    # attraction along edges
        disp = force.sum(axis=1)
        length = np.maximum(np.linalg.norm(disp, axis=1, keepdims=True), 1e-9)
        pos += disp / length * np.minimum(length, temp)
        temp *= 0.985
    pos -= pos.mean(axis=0)
    return pos / max(np.abs(pos).max(), 1e-9)


def _sparse_sample(x, y, label, max_extra=25):
    """First training sample with this label whose graph has at most `max_extra` edges on top
    of its spanning trees -- typical samples are much denser and draw as a hairball."""
    n_nodes = x.shape[-1]
    for i in np.flatnonzero(y == label):
        if x[i, 0].sum() // 2 - (n_nodes - (label + 1)) <= max_extra:
            return int(i)
    raise ValueError(f"no sparse sample with label {label}")


def fig_graphspectra(data_dir):
    x, y = _load(data_dir, "GraphSpectra")
    picks = [_sparse_sample(x, y, 1), _sparse_sample(x, y, 3)]  # 2 and 4 components
    rng = np.random.default_rng(0)                              # layout only, not data

    fig = plt.figure(figsize=(20, 14.6))
    fig.text(0.5, 0.972, "label = number of connected components   ·   node order is shuffled in every sample",
             ha="center", fontsize=14)
    for row, i in enumerate(picks):
        adj = np.asarray(x[i, 0])
        comps = _components(adj)
        top = 0.865 - row * 0.465
        fig.text(0.04, top + 0.06, f"train sample #{i}   ·   label {y[i]}   ·   "
                 f"{len(comps)} connected components   ·   {int(adj.sum() // 2)} edges",
                 fontsize=12.5, fontweight="bold")

        ax = fig.add_axes([0.04, top - 0.36, 0.25, 0.36])
        ax.imshow(adj, cmap=matplotlib.colors.ListedColormap([CELL_OFF, INK]), interpolation="nearest")
        _frame(ax)
        ax.set_title("what is stored: the 40×40 matrix\n(cell i,j lit = edge between nodes i and j)",
                     pad=8, linespacing=1.5)

        order = [v for c in comps for v in c]
        color_of = {v: SERIES[ci] for ci, c in enumerate(comps) for v in c}
        rgb = np.zeros((40, 40, 3))
        rgb[:] = matplotlib.colors.to_rgb(CELL_OFF)
        for a, u in enumerate(order):
            for b, v in enumerate(order):
                if adj[u, v]:
                    rgb[a, b] = matplotlib.colors.to_rgb(color_of[u])
        ax = fig.add_axes([0.34, top - 0.36, 0.25, 0.36])
        ax.imshow(rgb, interpolation="nearest")
        lo = 0
        for ci, c in enumerate(comps):
            ax.add_patch(Rectangle((lo - 0.5, lo - 0.5), len(c), len(c), fill=False,
                                   edgecolor=SERIES[ci], linewidth=1.2, alpha=0.8))
            lo += len(c)
        _frame(ax)
        ax.set_title("the same matrix, nodes sorted by component\n(this ordering never reaches the network)",
                     pad=8, linespacing=1.5)

        ax = fig.add_axes([0.63, top - 0.38, 0.35, 0.38])
        cols = int(np.ceil(np.sqrt(len(comps))))
        for ci, c in enumerate(comps):
            pos = _spring_layout(adj, c, rng)
            scale = 0.42 * np.sqrt(len(c) / len(comps[0]))
            centre = np.array([ci % cols, -(ci // cols)], dtype=float)
            pts = pos * scale + centre
            for a in range(len(c)):
                for b in range(a + 1, len(c)):
                    if adj[c[a], c[b]]:
                        ax.plot(*pts[[a, b]].T, color=MUTED, linewidth=0.9, alpha=0.7, zorder=1)
            ax.scatter(*pts.T, s=70, color=SERIES[ci], edgecolor=BG, linewidth=1.0, zorder=2)
            ax.text(centre[0], centre[1] - 0.56, f"{len(c)} nodes", color=MUTED,
                    ha="center", va="top", fontsize=9.5)
        ax.set_xlim(-0.6, cols - 0.4)
        ax.set_ylim(-(np.ceil(len(comps) / cols) - 1) - 0.7, 0.55)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("the same graph, drawn\n(colour = component)", pad=8, linespacing=1.5)
    _save(fig, "graphspectra")


# ---------------------------------------------------------------------------
# MelodyContour
# ---------------------------------------------------------------------------

def _scale_root(used, scale):
    """A root that puts every used pitch class inside the scale (the lowest one that fits)."""
    for root in range(12):
        if used <= {(root + s) % 12 for s in scale}:
            return root
    return None


def fig_melodycontour(data_dir):
    x, y = _load(data_dir, "MelodyContour")
    names = generate_melodycontour.MODE_NAMES
    picks = [_first(y, names.index(m)) for m in ("major", "whole_tone", "phrygian")]

    fig = plt.figure(figsize=(20, 13.6))
    fig.text(0.04, 0.965, "label = the mode the melody was drawn from", fontsize=14)
    fig.legend(handles=[
        plt.Line2D([], [], marker="s", linestyle="", color=ORANGE, markersize=9, label="note onset (channel 0)"),
        plt.Line2D([], [], marker="s", linestyle="", color=TEAL, markersize=6, label="note held (channel 1)"),
        plt.Line2D([], [], marker="s", linestyle="", color=INK, markersize=9, label="pitch class used"),
        plt.Line2D([], [], marker="s", linestyle="", markerfacecolor="none", markeredgecolor=ORANGE,
                   markersize=9, label="in the mode's scale"),
    ], loc="upper right", bbox_to_anchor=(0.975, 0.985), ncol=4, frameon=False, fontsize=10.5)

    for row, i in enumerate(picks):
        onset, held = np.asarray(x[i, 0]), np.asarray(x[i, 1])
        mode = names[y[i]]
        top = 0.9 - row * 0.3
        ax = fig.add_axes([0.06, top - 0.22, 0.68, 0.22])
        hp, ht = np.nonzero(held * (1 - onset))
        op, ot = np.nonzero(onset)
        ax.scatter(ht, hp, marker="s", s=10, color=TEAL, linewidths=0)
        ax.scatter(ot, op, marker="s", s=38, color=ORANGE, linewidths=0)
        ax.set_xlim(-1, 64)
        ax.set_ylim(-1, 48)
        ax.set_yticks([0, 12, 24, 36, 47])
        ax.set_xticks([0, 16, 32, 48, 63])
        for octave in (12, 24, 36):
            ax.axhline(octave - 0.5, color=FRAME, linewidth=0.6, alpha=0.6, zorder=0)
        ax.set_ylabel("pitch row")
        if row == 2:
            ax.set_xlabel("time step  →")
        for spine in ax.spines.values():
            spine.set_color(FRAME)
        ax.set_title(f"train sample #{i}   ·   label {y[i]} = {mode.replace('_', ' ')}",
                     loc="left", fontweight="bold", pad=8)

        used = {int(r) % 12 for r in np.flatnonzero(held.any(axis=1))}
        scale = generate_melodycontour.MODES[mode]
        root = _scale_root(used, scale)
        in_scale = {(root + s) % 12 for s in scale}
        ax = fig.add_axes([0.79, top - 0.13, 0.19, 0.06])
        for pc in range(12):
            ax.add_patch(Rectangle((pc + 0.1, 0.1), 0.8, 0.8,
                                   facecolor=INK if pc in used else "none",
                                   edgecolor=ORANGE if pc in in_scale else FRAME,
                                   linewidth=2.2 if pc in in_scale else 1.0))
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(12) + 0.5)
        ax.set_xticklabels([str(pc) for pc in range(12)], fontsize=9)
        ax.set_yticks([])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"pitch classes (row mod 12)\n{len(used)} used, all in the scale on root {root}",
                     fontsize=10, pad=6, linespacing=1.5)
    _save(fig, "melodycontour")


# ---------------------------------------------------------------------------
# HiddenSignal
# ---------------------------------------------------------------------------

def fig_hiddensignal(data_dir):
    x, y = _load(data_dir, "HiddenSignal")
    record = {}

    def wrap_components(fn):
        def run(rng, cls, photos):
            bg, signal = fn(rng, cls, photos)
            record.update(bg=bg, signal=signal)
            return bg, signal
        return run

    samples = _replay(generate_hiddensignal, "_one_sample", 200, record,
                      {"_components": wrap_components, "_photo_crop": _record_photo(record)})
    names = generate_hiddensignal.CLASS_NAMES
    wanted = [(names.index("h_grating"), True), (names.index("radial"), False)]
    picks = [next(i for i, s in enumerate(samples) if s["args"][1] == cls and ("photo" in s) == photo)
             for cls, photo in wanted]
    _assert_saved(x, samples, picks)

    fig = plt.figure(figsize=(20, 12.4))
    fig.text(0.5, 0.968, "label = which faint pattern was added (or none)   ·   "
             "the background is chosen independently of the label", ha="center", fontsize=14)
    for row, i in enumerate(picks):
        s = samples[i]
        bg, signal = s["bg"], s["signal"]
        amp = float(np.abs(signal).max())
        top = 0.86 - row * 0.46
        source = s.get("photo", "procedural value noise")
        fig.text(0.04, top + 0.045, f"train sample #{i}   ·   label {y[i]} = {names[y[i]]}   ·   "
                 f"amplitude {amp:.1%}   ·   background: {source}", fontsize=12, fontweight="bold")
        axes = [fig.add_axes([0.04 + c * 0.245, top - 0.34, 0.19, 0.31]) for c in range(4)]
        _show(axes[0], bg, "background\n(random, label-independent)", hwc=True)
        pat = 0.5 + 0.5 * signal[..., 0] / max(amp, 1e-9)
        _show(axes[1], np.repeat(pat[..., None], 3, axis=2), f"hidden pattern: {names[y[i]]}\n(alone, normalised)",
              hwc=True)
        _show(axes[2], np.clip(bg + 8 * signal, 0, 1), "sum, amplitude ×8\n(so you can see it)", hwc=True)
        _show(axes[3], x[i], f"the stored sample\n(real amplitude {amp:.1%})")
        for a, b, sym in zip(axes, axes[1:], ["+", "=", "→"]):
            _between(fig, a, b, sym)
    _save(fig, "hiddensignal")


# ---------------------------------------------------------------------------
# CamoCount
# ---------------------------------------------------------------------------

def fig_camocount(data_dir):
    x, y = _load(data_dir, "CamoCount")
    record = {}

    def wrap_draw(fn):
        def run(canvas, rng, cx, cy, r, is_circle):
            record.setdefault("shapes", []).append((cx, cy, r, is_circle))
            return fn(canvas, rng, cx, cy, r, is_circle)
        return run

    samples = _replay(generate_camocount, "_one_scene", 300, record,
                      {"_draw_shape": wrap_draw, "_photo_crop": _record_photo(record)})
    picks = []
    for count in (2, 7, 11):
        hits = [i for i, s in enumerate(samples) if s["args"][1] == count]
        picks.append(next((i for i in hits if "photo" in samples[i]), hits[0]))
    _assert_saved(x, samples, picks)

    fig = plt.figure(figsize=(20, 13.2))
    fig.text(0.5, 0.965, "label = number of circles   ·   squares are distractors, "
             "drawn in an independent random quantity and never counted", ha="center", fontsize=14)
    fig.legend(handles=[
        plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=ORANGE,
                   markeredgewidth=2.2, markersize=13, label="circle = target (counted)"),
        plt.Line2D([], [], marker="s", linestyle="", markerfacecolor="none", markeredgecolor=TEAL,
                   markeredgewidth=2.2, markersize=12, label="square = distractor (ignored)"),
    ], loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=2, frameon=False, fontsize=11.5)
    for col, i in enumerate(picks):
        s = samples[i]
        shapes = s.get("shapes", [])
        n_c = sum(1 for *_, c in shapes if c)
        left = 0.06 + col * 0.31
        source = s.get("photo", "procedural value noise")
        ax = fig.add_axes([left, 0.47, 0.26, 0.37])
        _show(ax, x[i], f"train sample #{i}, as stored\nbackground: {source}")
        ax = fig.add_axes([left, 0.04, 0.26, 0.37])
        _show(ax, x[i], f"label {y[i]}: {n_c} circles counted,\n{len(shapes) - n_c} squares ignored")
        for cx, cy, r, is_circle in shapes:
            if is_circle:
                ax.add_patch(Circle((cx, cy), r + 1, fill=False, edgecolor=ORANGE, linewidth=2.2))
            else:
                ax.add_patch(Rectangle((cx - r - 1, cy - r - 1), 2 * r + 2, 2 * r + 2, fill=False,
                                       edgecolor=TEAL, linewidth=2.0, linestyle=(0, (3, 2))))
        ax.set_xlim(-0.5, 95.5)
        ax.set_ylim(95.5, -0.5)
    _save(fig, "camocount")


# ---------------------------------------------------------------------------
# PendulumChaos
# ---------------------------------------------------------------------------

def _time_colors(n):
    t = np.linspace(0.0, 1.0, n)                                # the generator's blue -> red ramp
    return np.stack([0.2 + 0.7 * t, 0.2 * np.ones_like(t), 0.9 - 0.7 * t], axis=1)


def fig_pendulumchaos(data_dir):
    x, y = _load(data_dir, "PendulumChaos")
    grid = [_first(y, label) for label in range(6)]
    run_idx = grid[2]
    samples = _replay(generate_pendulumchaos, "_render", run_idx + 1, {})
    _assert_saved(x, samples, [run_idx])
    t1, t2 = samples[run_idx]["args"]

    fig = plt.figure(figsize=(22, 9.6))
    fig.text(0.5, 0.95, "label = net number of full turns the outer arm made around the inner bob in 8 s"
             "   ·   colour = time (blue → red)", ha="center", fontsize=14)

    ax = fig.add_axes([0.03, 0.2, 0.26, 0.64])
    x1, y1 = np.sin(t1), -np.cos(t1)
    x2, y2 = x1 + np.sin(t2), y1 - np.cos(t2)
    pts = np.stack([x2, y2], axis=1)
    ax.add_collection(LineCollection(np.stack([pts[:-1], pts[1:]], axis=1), colors=_time_colors(len(pts) - 1),
                                     linewidth=1.6, alpha=0.9))
    p0, p1, p2 = (0.0, 0.0), (x1[0], y1[0]), (x2[0], y2[0])
    for a, b in ((p0, p1), (p1, p2)):
        ax.plot(*zip(a, b), color=INK, linewidth=3.0, zorder=3, solid_capstyle="round")
    ax.scatter(*p0, marker="^", s=130, color=MUTED, zorder=4)
    ax.scatter(*p1, s=170, color=TEAL, edgecolor=BG, linewidth=1.5, zorder=4)
    ax.scatter(*p2, s=210, color=ORANGE, edgecolor=BG, linewidth=1.5, zorder=4)
    ax.legend(handles=[
        plt.Line2D([], [], marker="^", linestyle="", color=MUTED, markersize=10, label="pivot (fixed)"),
        plt.Line2D([], [], marker="o", linestyle="", color=TEAL, markersize=10, label="inner bob"),
        plt.Line2D([], [], marker="o", linestyle="", color=ORANGE, markersize=11,
                   label="outer bob -- its path is what gets drawn"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=10.5, labelcolor=INK)
    ax.set_xlim(-2.15, 2.15)
    ax.set_ylim(-2.15, 2.15)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"one real run: train sample #{run_idx}, label {y[run_idx]}\n"
                 f"(physical view, arms drawn at t = 0)", pad=8, linespacing=1.5)

    ax = fig.add_axes([0.31, 0.3, 0.15, 0.34])
    _show(ax, x[run_idx], "the same run, as stored\n(64×64, upside down: the\nrow index grows with height)")

    for k, i in enumerate(grid):
        ax = fig.add_axes([0.52 + (k % 3) * 0.16, 0.47 - (k // 3) * 0.42, 0.13, 0.32])
        _show(ax, x[i], f"label {y[i] if y[i] < 5 else '5+'}  (#{i})")
    _save(fig, "pendulumchaos")


FIGURES = {
    "graphspectra": fig_graphspectra,
    "melodycontour": fig_melodycontour,
    "hiddensignal": fig_hiddensignal,
    "camocount": fig_camocount,
    "pendulumchaos": fig_pendulumchaos,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", metavar="name",
                    help=f"figures to render, any of {', '.join(FIGURES)} (default: all)")
    ap.add_argument("--data", default=str(HERE / "data"), help="directory holding the --full datasets")
    args = ap.parse_args()
    unknown = set(args.names) - set(FIGURES)
    if unknown:
        ap.error(f"unknown figure(s): {', '.join(sorted(unknown))}")
    for name in args.names or FIGURES:
        FIGURES[name](args.data)


if __name__ == "__main__":
    main()
