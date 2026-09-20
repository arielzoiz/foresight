"""Regenerate the A/B plots from the files in this folder. Needs only matplotlib.

    python plot_results.py

Reads data/<arm>/<task>/iteration.jsonl and data/<arm>/summary.txt (the
swe_ci.summarize table); writes plots/*.png. No cluster, Docker or model access
is needed: it only reads result files.
"""
import json, re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
DATA, OUT = HERE / "data", HERE / "plots"
OUT.mkdir(exist_ok=True)
TASK_IDS = (HERE / "configs" / "tasks.txt").read_text().split()
SHORT = {"15r10nk__inline-snapshot__3bb05d__e2b9b2": "inline-snapshot",
         "9001__copyparty__452592__4bdcbc": "copyparty 452592",
         "9001__copyparty__8c52b8__a96d9a": "copyparty 8c52b8",
         "9001__copyparty__ccdace__6a0aaa": "copyparty ccdace",
         "agronholm__anyio__439951__41647f": "anyio 439951"}
tasks = [(t, SHORT.get(t, t[:20])) for t in TASK_IDS]

def evoscores(arm):
    """First numeric column (EvoScore) of each task row in summary.txt, in task order; then the average."""
    vals = []
    for line in (DATA / arm / "summary.txt").read_text().splitlines():
        m = re.match(r"\s*[║|]?\s*(\S.*?)\s*[│|]\s*(-?\d+\.\d+)\s*[│|]", line)
        if m and not m.group(1).startswith(("Task", "AVERAGE")):
            vals.append(float(m.group(2)))
    return vals[:len(tasks)]

evo = {a: evoscores(a) for a in ("control", "foresight")}

C, F = "#2a78d6", "#eb6834"   # validated categorical slots 1-2 (blue, orange)
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({"font.family": "sans-serif", "text.color": INK,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})

def gaps(arm, tid):
    g = []
    for l in open(DATA / arm / tid / "iteration.jsonl"):
        d = json.loads(l)
        g.append(d["gap"] if d.get("pytest", {}).get("passed") is not None and d["gap"] >= 0 else None)
    return g

def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.tick_params(length=0)

# Figure 1: gap per epoch (small multiples, own y-scale per task)
fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), facecolor=SURF)
for ax, (tid, name) in zip(axes.flat, tasks):
    style(ax)
    for arm, col in (("control", C), ("foresight", F)):
        g = gaps(arm, tid)
        xs = [i for i, v in enumerate(g) if v is not None]
        ys = [v for v in g if v is not None]
        ax.plot(xs, ys, color=col, lw=2, marker="o", ms=6, mec=SURF, mew=2, zorder=3)
        if None in g:
            i = g.index(None)
            ax.plot([i], [ys[-1]], marker="X", ms=11, color=col, mec=SURF, mew=1.5, zorder=4)
            ax.annotate("epoch %d: pytest not executed\n(shown at last valid gap)" % i, (i, ys[-1]),
                        textcoords="offset points", xytext=(-8, -34), ha="right", fontsize=9, color=MUTED)
    ax.set_title(name, loc="left", fontsize=12, fontweight="bold", color=INK)
    ax.set_xticks(range(6)); ax.set_ylim(bottom=0)
    ax.set_xlabel("epoch (0 = starting state)", fontsize=9)
ax = axes.flat[5]; ax.axis("off")
ax.legend(handles=[Line2D([0], [0], color=C, lw=2, marker="o", label="Control (SWE-CI → target)"),
                   Line2D([0], [0], color=F, lw=2, marker="o", label="Foresight (SWE-CI → foresight → target, aux explores)"),
                   Line2D([0], [0], color=MUTED, lw=0, marker="X", ms=10, label="Final epoch: pytest not executed")],
          loc="upper left", frameon=False, fontsize=10, labelcolor=INK)
ax.text(0, 0.35, "Gap = remaining failing-test gap; lower is better.\nSame 5 tasks, same SWE-CI settings in both arms\n(5 epochs, 1 worker, max_try 3). One run per arm.\nEach panel has its own y-scale.",
        transform=ax.transAxes, fontsize=9.5, color=MUTED, va="top")
fig.suptitle("Remaining test gap per epoch: control vs foresight", x=0.02, ha="left", fontsize=15, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT / "results_gap_per_epoch.png", dpi=160, facecolor=SURF)

# Figure 2: EvoScore per task + average
fig, ax = plt.subplots(figsize=(11, 5.2), facecolor=SURF); style(ax)
names = [n for _, n in tasks] + ["AVERAGE"]
ctrl = evo["control"] + [sum(evo["control"]) / len(tasks)]
fore = evo["foresight"] + [sum(evo["foresight"]) / len(tasks)]
w, xs = 0.36, range(len(names))
for off, vals, col, lab in ((-w/2 - 0.01, ctrl, C, "Control"), (w/2 + 0.01, fore, F, "Foresight")):
    ax.bar([x + off for x in xs], vals, w, color=col, label=lab, zorder=3)
    for x, v in zip(xs, vals):
        ax.text(x + off, v + (0.02 if v >= 0 else -0.02), (f"{v:.2f}" if abs(v) >= 0.005 else f"{v:.3f}"),
                ha="center", va="bottom" if v >= 0 else "top", fontsize=9, color=INK)
ax.axhline(0, color=MUTED, lw=1, zorder=4)
ax.set_xticks(list(xs)); ax.set_xticklabels(names, fontsize=10)
ax.get_xticklabels()[-1].set_fontweight("bold")
ax.set_ylabel("EvoScore (higher is better)"); ax.set_ylim(-0.22, 0.8)
ax.legend(frameon=False, loc="upper left", fontsize=10, labelcolor=INK)
ax.set_title("EvoScore per task", loc="left", fontsize=15, fontweight="bold", pad=14)
fig.text(0.01, 0.01, "5 tasks, one run per arm; resolved = 0 in both arms; zero-regression rate 0.6 (control) vs 0.4 (foresight).",
         fontsize=9, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(OUT / "results_evoscore.png", dpi=160, facecolor=SURF)
print("wrote", *sorted(p.name for p in OUT.glob("*.png")))
print("evoscores", evo)
