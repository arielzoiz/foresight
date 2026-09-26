"""Fig 4.1: EvoScore per task, control vs foresight (dumbbell plot, sorted by the paired difference)."""
import numpy as np

from data import differences, load_tasks
from style import COLORS, FULL_WIDTH, LABELS, MUTED, plt, save


def main():
    tasks = load_tasks()
    d = np.array(differences(tasks))
    order = np.argsort(d)[::-1]            # largest foresight advantage on top
    tasks = [tasks[i] for i in order]
    d = d[order]
    y = np.arange(len(tasks))[::-1]

    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.75, 4.2))
    c = [t.arms["control"].evo_score for t in tasks]
    f = [t.arms["foresight"].evo_score for t in tasks]
    ax.hlines(y, c, f, color=MUTED, linewidth=1.2, zorder=1)
    # control drawn larger underneath, so it stays visible where the two scores coincide
    for arm, xs, size in (("control", c, 70), ("foresight", f, 28)):
        ax.scatter(xs, y, s=size, color=COLORS[arm], edgecolor="white", linewidth=1.2,
                   zorder=2, label=LABELS[arm])
        # mean EvoScore over the tasks, as a dashed vertical line
        mean = np.mean(xs)
        ax.axvline(mean, color=COLORS[arm], linewidth=1.2, linestyle="--", zorder=1,
                   label=f"{LABELS[arm]} mean ({mean:.3f})")
    ax.axvline(0, color=MUTED, linewidth=0.8, zorder=0)

    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-1.08, 1.0)

    # paired difference as text on the right margin
    for yi, di in zip(y, d):
        ax.annotate(f"{di:+.2f}", xy=(1.0, yi), xycoords=("axes fraction", "data"),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=7, color=MUTED)
    ax.annotate("d", xy=(1.0, y[0] + 0.9), xycoords=("axes fraction", "data"),
                xytext=(6, 0), textcoords="offset points", va="center", fontsize=7, color=MUTED)

    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.07))
    ax.spines["left"].set_visible(False)
    save(fig, "fig4_1_evoscore_paired")


if __name__ == "__main__":
    main()
