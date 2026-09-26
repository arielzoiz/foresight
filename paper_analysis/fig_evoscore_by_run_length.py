"""Fig 4.2 alternative: the EvoScore SWE-CI would report if the run had stopped after k epochs,
for k = 1..20, averaged over tasks per arm (+-1 SE band across tasks).

SWE-CI's prompts never state the epoch budget, so the first k epochs of a 20-epoch run are what a
k-epoch run would have produced; EvoScore(k) is the mean of the first k relative changes.
At k = 20 the curves end at the mean EvoScores of Fig 4.1.
"""
import numpy as np

from data import ARMS, MAX_EPOCH, load_tasks
from style import COLORS, HALF_WIDTH, LABELS, MUTED, plt, save


def main():
    tasks = load_tasks()
    k = np.arange(1, MAX_EPOCH + 1)

    fig, ax = plt.subplots(figsize=(HALF_WIDTH * 1.4, 2.8))
    ax.axhline(0, color=MUTED, linewidth=0.8)
    for arm in ARMS:
        rel = np.array([t.arms[arm].rel_change for t in tasks])     # task x epoch
        evo_k = np.cumsum(rel, axis=1) / k                             # EvoScore if stopped at epoch k
        mean = evo_k.mean(axis=0)
        se = evo_k.std(axis=0, ddof=1) / np.sqrt(len(tasks))
        ax.fill_between(k, mean - se, mean + se, color=COLORS[arm], alpha=0.15, linewidth=0)
        ax.plot(k, mean, color=COLORS[arm], marker="o", markersize=3.5,
                label=f"{LABELS[arm]} ({mean[-1]:.3f} at 20 epochs)")

    ax.set_xlabel("Run length (epochs)")
    ax.set_ylabel("Mean EvoScore")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlim(0.5, MAX_EPOCH + 0.5)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower right")
    save(fig, "fig4_2_evoscore_by_run_length")


if __name__ == "__main__":
    main()
