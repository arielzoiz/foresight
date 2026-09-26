"""Fig 4.3: lines changed per epoch (added + removed, all attempted edits), per arm.

Top: median over tasks with the interquartile band; log scale (fig4_3_changes_per_epoch.png)
or linear scale (fig4_3_changes_per_epoch_linear.png).
Bottom: number of tasks whose edit was rejected in that epoch (pytest could not run).
"""
import numpy as np

from data import ARMS, MAX_EPOCH, load_tasks
from style import COLORS, FULL_WIDTH, LABELS, plt, save


def draw(log_scale):
    tasks = load_tasks()
    epochs = np.arange(1, MAX_EPOCH + 1)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(FULL_WIDTH * 0.75, 3.8), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12})
    for arm in ARMS:
        churn = np.array([t.arms[arm].churn for t in tasks], dtype=float)
        q25, med, q75 = np.percentile(churn, [25, 50, 75], axis=0)
        ax.fill_between(epochs, q25, q75, color=COLORS[arm], alpha=0.15, linewidth=0)
        ax.plot(epochs, med, color=COLORS[arm], marker="o", markersize=3.5, label=LABELS[arm])

        rejected = [sum(t.arms[arm].accepted[e] is False for t in tasks) for e in range(MAX_EPOCH)]
        offset = -0.2 if arm == "control" else 0.2
        ax2.bar(epochs + offset, rejected, width=0.38, color=COLORS[arm], label=LABELS[arm])

    if log_scale:
        ax.set_yscale("log")
        ax.set_ylim(2, 1000)
    else:
        ax.set_ylim(0, None)
    ax.set_ylabel("Lines changed\n(median, IQR band)")
    ax.legend(loc="upper right", ncol=2)

    ax2.set_ylabel("Rejected\nedits")
    ax2.set_xlabel("Epoch")
    ax2.set_xticks([1, 5, 10, 15, 20])
    ax2.set_xlim(0.5, MAX_EPOCH + 0.5)
    ax2.grid(axis="x", visible=False)
    save(fig, "fig4_3_changes_per_epoch" if log_scale else "fig4_3_changes_per_epoch_linear")


def main():
    draw(log_scale=True)
    draw(log_scale=False)


if __name__ == "__main__":
    main()
