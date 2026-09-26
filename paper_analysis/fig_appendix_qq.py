"""Fig A.2: normal Q-Q plot of the paired EvoScore differences."""
import numpy as np
from scipy import stats

from data import differences, load_tasks
from style import HALF_WIDTH, INK, MUTED, plt, save


def main():
    d = np.array(differences(load_tasks()))
    (theoretical, ordered), (slope, intercept, _) = stats.probplot(d, dist="norm")
    sw = stats.shapiro(d)

    fig, ax = plt.subplots(figsize=(HALF_WIDTH, 2.6))
    ax.plot(theoretical, intercept + slope * theoretical, color=INK, linewidth=1.2)
    ax.scatter(theoretical, ordered, s=24, color=MUTED, edgecolor="white", linewidth=1, zorder=2)
    ax.set_xlabel("Theoretical normal quantiles")
    ax.set_ylabel("Ordered differences d")
    ax.set_title(f"Shapiro-Wilk W = {sw.statistic:.3f}, p = {sw.pvalue:.3f}", loc="left")
    save(fig, "figA_2_diff_qq")


if __name__ == "__main__":
    main()
