"""Fig 4.2 variant with every A/B task: rows 6-21 plus rows 1-5 (5-epoch cap) and row 22
(foresight stopped after 12 epochs). Not the sample the tests use.

Mean relative change in passing tests per epoch (+-1 SE band). An epoch only counts a task when
both arms ran it, so the number of tasks changes along the x-axis (marked as in the Fig 4.3 variant).
"""
import numpy as np

from data import ARMS, MAX_EPOCH, load_rel_change_all_rows
from style import COLORS, HALF_WIDTH, LABELS, MUTED, mark_task_counts, plt, save


def main():
    data = load_rel_change_all_rows()
    epochs = np.arange(1, MAX_EPOCH + 1)
    rel = {arm: np.array([data[r][arm] for r in sorted(data)], dtype=float) for arm in ARMS}
    both = ~np.isnan(rel["control"]) & ~np.isnan(rel["foresight"])     # task x epoch
    n_tasks = both.sum(axis=0)

    fig, ax = plt.subplots(figsize=(HALF_WIDTH * 1.4, 2.8))
    ax.axhline(0, color=MUTED, linewidth=0.8)
    for arm in ARMS:
        cols = [rel[arm][both[:, e], e] for e in range(MAX_EPOCH)]
        mean = np.array([c.mean() for c in cols])
        se = np.array([c.std(ddof=1) / np.sqrt(len(c)) for c in cols])
        ax.fill_between(epochs, mean - se, mean + se, color=COLORS[arm], alpha=0.15, linewidth=0)
        ax.plot(epochs, mean, color=COLORS[arm], marker="o", markersize=3.5, label=LABELS[arm])

    mark_task_counts((ax,), n_tasks)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean relative change\nin passing tests")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlim(0.5, MAX_EPOCH + 0.5)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower right")
    save(fig, "fig4_2_progress_curve_all_rows")


if __name__ == "__main__":
    main()
