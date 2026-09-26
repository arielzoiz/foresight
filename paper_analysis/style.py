"""Shared look for all figures. Change colors, sizes or the output format here."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "out"

COLORS = {"control": "#2a78d6", "foresight": "#eb6834"}
LABELS = {"control": "Control", "foresight": "Foresight"}
INK = "#16140f"
MUTED = "#8a857a"
GRID = "#e9e6de"

# Single-column figure width of a typical two-column paper is ~3.3 in; full width ~6.8 in.
FULL_WIDTH = 6.8
HALF_WIDTH = 3.4

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "lines.linewidth": 1.8,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
})


def save(fig, name):
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def mark_task_counts(axes, n_tasks):
    """Dotted lines where the number of tasks per epoch changes, and "n = .." over each stretch
    (on the first axis). n_tasks[e] is the count for epoch e + 1."""
    n_epochs = len(n_tasks)
    changes = [e for e in range(1, n_epochs) if n_tasks[e] != n_tasks[e - 1]]
    for x in changes:
        for a in axes:
            a.axvline(x + 0.5, color=MUTED, linewidth=0.8, linestyle=":")
    for s, e in zip([0] + changes, changes + [n_epochs]):
        axes[0].text((s + e) / 2 + 0.5, 1.01, f"n = {n_tasks[s]}", transform=axes[0].get_xaxis_transform(),
                     ha="center", va="bottom", fontsize=7, color=MUTED)
