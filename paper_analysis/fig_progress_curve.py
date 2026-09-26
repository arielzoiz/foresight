"""Fig 4.2: mean normalised progress per epoch (SWE-CI's relative change), per arm, with a +-1 SE band.

EvoScore is the mean of this curve over the 20 epochs, so the figure shows where in the
run the two arms' scores come from. Broken epochs (pytest could not run) count as -1, as in SWE-CI.
"""
import numpy as np

from data import ARMS, MAX_EPOCH, load_tasks
from style import COLORS, HALF_WIDTH, LABELS, MUTED, plt, save


def main():
    tasks = load_tasks()
    epochs = np.arange(1, MAX_EPOCH + 1)

    fig, ax = plt.subplots(figsize=(HALF_WIDTH * 1.4, 2.8))
    ax.axhline(0, color=MUTED, linewidth=0.8)
    for arm in ARMS:
        m = np.array([t.arms[arm].rel_change for t in tasks])
        mean = m.mean(axis=0)
        se = m.std(axis=0, ddof=1) / np.sqrt(len(tasks))
        ax.fill_between(epochs, mean - se, mean + se, color=COLORS[arm], alpha=0.15, linewidth=0)
        ax.plot(epochs, mean, color=COLORS[arm], marker="o", markersize=3.5,
                label=f"{LABELS[arm]} (EvoScore {mean.mean():.3f})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean relative change\nin passing tests")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlim(0.5, MAX_EPOCH + 0.5)
    ax.legend(loc="lower right")
    save(fig, "fig4_2_progress_curve")


if __name__ == "__main__":
    main()
