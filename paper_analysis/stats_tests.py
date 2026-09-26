"""Hypothesis tests for Section 4. Prints the results and writes out/stats.txt and out/evoscore_per_task.csv.

EvoScore:  H0 mean(d) = 0 vs H1 mean(d) != 0, d = EvoScore(foresight) - EvoScore(control), paired by task.
           Paired t-test (primary), Wilcoxon signed-rank (robustness), Shapiro-Wilk on d (normality check).
Changes:   H0 no difference in lines changed per epoch vs H1 some difference at some epoch.
           Exact sign-flip permutation test over the 20-epoch curve (swap arm labels within each task),
           statistic = max over epochs of |mean over tasks of (log1p churn_F - log1p churn_C)|;
           plus a paired Wilcoxon test on each task's total lines changed.
"""
import csv
import io
import itertools
from contextlib import redirect_stdout

import numpy as np
from scipy import stats

from data import MAX_EPOCH, load_tasks
from style import OUT_DIR


def evoscore_tests(tasks):
    c = np.array([t.arms["control"].evo_score for t in tasks])
    f = np.array([t.arms["foresight"].evo_score for t in tasks])
    d = f - c
    n = len(d)

    print(f"{'row':>3}  {'task':30} {'control':>8} {'foresight':>9} {'d':>8}  broken C/F")
    for t, di in zip(tasks, d):
        print(f"{t.row:>3}  {t.label:30} {t.arms['control'].evo_score:8.4f} "
              f"{t.arms['foresight'].evo_score:9.4f} {di:+8.4f}  "
              f"{t.arms['control'].broken_epochs:>2}/{t.arms['foresight'].broken_epochs:<2}")
    print()
    print(f"n = {n} tasks")
    print(f"mean EvoScore  control {c.mean():+.4f} (sd {c.std(ddof=1):.4f})   "
          f"foresight {f.mean():+.4f} (sd {f.std(ddof=1):.4f})")
    print(f"median EvoScore control {np.median(c):+.4f}   foresight {np.median(f):+.4f}")
    print(f"d: mean {d.mean():+.4f}, sd {d.std(ddof=1):.4f}, median {np.median(d):+.4f}")

    tt = stats.ttest_rel(f, c)
    ci = tt.confidence_interval(0.95)
    print(f"paired t-test:        t({n - 1}) = {tt.statistic:.3f}, p = {tt.pvalue:.4f}, "
          f"95% CI of mean d [{ci.low:+.4f}, {ci.high:+.4f}]")
    dz = d.mean() / d.std(ddof=1)
    print(f"effect size (Cohen's d_z = mean d / sd d): {dz:+.3f}")
    w = stats.wilcoxon(d)   # zero differences dropped (scipy's default zero_method='wilcox')
    print(f"Wilcoxon signed-rank: W = {w.statistic:.1f}, p = {w.pvalue:.4f} "
          f"(n non-zero = {int(np.count_nonzero(d))})")
    sw = stats.shapiro(d)
    print(f"Shapiro-Wilk on d:    W = {sw.statistic:.4f}, p = {sw.pvalue:.4f}")
    bc = sum(t.arms["control"].broken_epochs for t in tasks)
    bf = sum(t.arms["foresight"].broken_epochs for t in tasks)
    print(f"broken epochs (pytest could not run, scored -1): control {bc}, foresight {bf} "
          f"of {n * MAX_EPOCH} each")
    return d


def sign_flip_test(diff_matrix):
    """Exact two-sided sign-flip permutation test; rows = tasks, columns = epochs."""
    n = diff_matrix.shape[0]
    observed = np.abs(diff_matrix.mean(axis=0)).max()
    signs = np.array(list(itertools.product((1, -1), repeat=n)))           # 2^n x n
    null = np.abs(signs @ diff_matrix / n).max(axis=1)
    return observed, float((null >= observed - 1e-12).mean()), len(signs)


def sign_flip_tmax_test(diff_matrix):
    """Same test with Blair & Karniski's (1993) t_max: the maximum |paired t| over epochs."""
    n = diff_matrix.shape[0]
    signs = np.array(list(itertools.product((1, -1), repeat=n)))           # 2^n x n
    flipped = signs[:, :, None] * diff_matrix[None]                         # perm x task x epoch
    sd = flipped.std(axis=1, ddof=1)
    sd[sd == 0] = np.inf                                                    # all-zero epoch: t = 0
    null = np.abs(flipped.mean(axis=1) / (sd / np.sqrt(n))).max(axis=1)
    observed = null[0]                                                      # first row = no flips
    return observed, float((null >= observed - 1e-12).mean()), len(signs)


def churn_tests(tasks):
    cf = np.array([t.arms["foresight"].churn for t in tasks], dtype=float)
    cc = np.array([t.arms["control"].churn for t in tasks], dtype=float)
    diff = np.log1p(cf) - np.log1p(cc)
    obs, p, n_perm = sign_flip_test(diff)
    print(f"lines changed per epoch, median over tasks and epochs: control {np.median(cc):.0f}, "
          f"foresight {np.median(cf):.0f}")
    print(f"sign-flip permutation test (curve, max |mean log1p diff|): stat = {obs:.3f}, "
          f"p = {p:.4f} ({n_perm} exact permutations)")
    obs_raw, p_raw, _ = sign_flip_test(cf - cc)
    print(f"same test without the log (raw lines changed): stat = {obs_raw:.1f}, p = {p_raw:.4f}")
    obs_t, p_t, _ = sign_flip_tmax_test(diff)
    print(f"same test with Blair & Karniski's t_max statistic: t_max = {obs_t:.3f}, p = {p_t:.4f}")
    tc, tf = cc.sum(axis=1), cf.sum(axis=1)
    w = stats.wilcoxon(tf, tc)
    print(f"total lines changed per task: median control {np.median(tc):.0f}, foresight {np.median(tf):.0f}; "
          f"Wilcoxon W = {w.statistic:.1f}, p = {w.pvalue:.4f}")
    early, late = slice(0, 5), slice(5, MAX_EPOCH)
    shape = diff[:, early].mean(axis=1) - diff[:, late].mean(axis=1)
    ws = stats.wilcoxon(shape)
    print(f"early (1-5) minus late (6-20) log-diff per task: median {np.median(shape):+.3f}, "
          f"Wilcoxon p = {ws.pvalue:.4f}  [exploratory]")


def write_csv(tasks, d):
    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / "evoscore_per_task.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "task_id", "evo_control", "evo_foresight", "d",
                    "broken_control", "broken_foresight"])
        for t, di in zip(tasks, d):
            w.writerow([t.row, t.task_id, f"{t.arms['control'].evo_score:.4f}",
                        f"{t.arms['foresight'].evo_score:.4f}", f"{di:.4f}",
                        t.arms["control"].broken_epochs, t.arms["foresight"].broken_epochs])


def main():
    tasks = load_tasks()
    buf = io.StringIO()
    with redirect_stdout(buf):
        print("== EvoScore (official SWE-CI definition, gamma = 1, 20 epochs) ==")
        d = evoscore_tests(tasks)
        print("\n== Lines changed per epoch (all attempted edits) ==")
        churn_tests(tasks)
    text = buf.getvalue()
    print(text, end="")
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "stats.txt").write_text(text)
    write_csv(tasks, d)


if __name__ == "__main__":
    main()
