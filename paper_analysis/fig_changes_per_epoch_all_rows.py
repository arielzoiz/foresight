"""Fig 4.3 variant with every A/B task: rows 6-21 plus rows 1-5 (5-epoch cap) and row 22
(foresight stopped after 12 epochs). Not the sample the tests use.

An epoch only counts a task when both arms ran it, so the number of tasks changes along the
x-axis (22 in epochs 1-5, 17 in 6-12, 16 in 13-20); dotted lines and labels mark where.
Linear scale; set LOG_SCALE = True for the log version.
"""
import numpy as np

from data import ARMS, MAX_EPOCH, load_churn_all_rows
from style import COLORS, FULL_WIDTH, LABELS, mark_task_counts, plt, save

LOG_SCALE = False


def main():
    data = load_churn_all_rows()
    epochs = np.arange(1, MAX_EPOCH + 1)
    churn = {arm: np.array([data[r][arm][0] for r in sorted(data)], dtype=float) for arm in ARMS}
    both = ~np.isnan(churn["control"]) & ~np.isnan(churn["foresight"])    # task x epoch
    n_tasks = both.sum(axis=0)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(FULL_WIDTH * 0.75, 3.8), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12})
    for arm in ARMS:
        q25, med, q75 = np.array([np.percentile(churn[arm][both[:, e], e], [25, 50, 75])
                                  for e in range(MAX_EPOCH)]).T
        ax.fill_between(epochs, q25, q75, color=COLORS[arm], alpha=0.15, linewidth=0)
        ax.plot(epochs, med, color=COLORS[arm], marker="o", markersize=3.5, label=LABELS[arm])

        rejected = [sum(data[r][arm][1][e] is False for i, r in enumerate(sorted(data)) if both[i, e])
                    for e in range(MAX_EPOCH)]
        offset = -0.2 if arm == "control" else 0.2
        ax2.bar(epochs + offset, rejected, width=0.38, color=COLORS[arm], label=LABELS[arm])

    if LOG_SCALE:
        ax.set_yscale("log")
        ax.set_ylim(1, 1000)
    else:
        ax.set_ylim(0, None)

    mark_task_counts((ax, ax2), n_tasks)

    ax.set_ylabel("Lines changed\n(median, IQR band)")
    ax.grid(axis="x", visible=False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="upper right", ncol=1, bbox_to_anchor=(1, 0.93))  # foresight on top

    ax2.set_ylabel("Rejected\nedits")
    ax2.set_xlabel("Epoch")
    ax2.set_xticks([1, 5, 10, 15, 20])
    ax2.set_xlim(0.5, MAX_EPOCH + 0.5)
    ax2.grid(axis="x", visible=False)
    save(fig, "fig4_3_changes_per_epoch_all_rows")


if __name__ == "__main__":
    main()
