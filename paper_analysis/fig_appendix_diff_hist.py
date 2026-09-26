"""Fig A.1: histogram of the paired differences d = EvoScore(foresight) - EvoScore(control)."""
import numpy as np
from matplotlib.ticker import MaxNLocator

from data import differences, load_tasks
from style import HALF_WIDTH, MUTED, plt, save


def main():
    d = np.array(differences(load_tasks()))

    fig, ax = plt.subplots(figsize=(HALF_WIDTH, 2.6))
    bins = np.arange(-0.9, 0.4, 0.1)
    ax.hist(d, bins=bins, color=MUTED, alpha=0.55, edgecolor="white", linewidth=1.5)
    ax.set_xlabel("d = EvoScore(foresight) − EvoScore(control)")
    ax.set_ylabel("Number of tasks")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    save(fig, "figA_1_diff_hist")


if __name__ == "__main__":
    main()
