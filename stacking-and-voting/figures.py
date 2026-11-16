"""Draw the post's figures (run report.py first for the two charts).

  stacking-flow.png       data flow of stacking: out-of-fold predictions -> meta-learner
  error-correlation.png   residual correlation of the six base models (Adult, California)
  gain-over-best.png      paired gain of each combiner over the best single model, 95% CI over 5 seeds

Run:    .venv/Scripts/python.exe stacking-and-voting/figures.py
Output: stacking-and-voting/figures/*.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

PAPER, INK, INK2, INK3, RULE = "#f7f5f0", "#1f1f1c", "#55524b", "#8a867c", "#d9d4c7"
TEAL, TEAL_L = "#0f766e", "#cfe5e2"
LABEL = {"XGBoost": "XGBoost", "LightGBM": "LightGBM", "CatBoost": "CatBoost",
         "RandomForest": "Random Forest", "Linear": "Linear*", "kNN": "k-NN"}

plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "axes.edgecolor": RULE, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


# ---------------------------------------------------------------- 1. data flow
def box(ax, x, y, w, h, text, fc=PAPER, ec=INK3, color=INK, size=9, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=color, fontsize=size, weight=weight)


def arrow(ax, a, b, color=INK3, lw=1.2, style="-|>"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=11, color=color, lw=lw,
                                 shrinkA=2, shrinkB=2))


def fig_flow():
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    # left: training rows split into 5 folds
    ax.text(0.2, 5.3, "1  Training rows (dev), 5 folds", color=INK, weight="bold", fontsize=10)
    fh = 0.62
    for k in range(5):
        y = 4.55 - k * (fh + 0.12)
        for j in range(5):
            held = j == k
            ax.add_patch(plt.Rectangle((0.3 + j * 0.42, y), 0.4, fh, fc=TEAL if held else TEAL_L,
                                       ec=PAPER, lw=1))
        ax.text(2.5, y + fh / 2, f"fit on 4 folds,\npredict fold {k + 1}", va="center", fontsize=8, color=INK2)
    ax.text(0.3, 0.75, "dark = held-out fold: its predictions\ncome from a model that never saw it",
            fontsize=8, color=INK2)
    # middle: OOF matrix
    ax.text(4.25, 5.3, "2  Out-of-fold predictions", color=INK, weight="bold", fontsize=10)
    cols = ["model A", "model B", "model C"]
    for j, c in enumerate(cols):
        ax.add_patch(plt.Rectangle((4.4 + j * 0.62, 1.3), 0.58, 3.65, fc=TEAL_L, ec=PAPER, lw=1))
        ax.text(4.4 + j * 0.62 + 0.29, 5.05, c, ha="center", fontsize=8, color=INK2)
        for k in range(5):
            ax.plot([4.4 + j * 0.62, 4.98 + j * 0.62], [1.3 + k * 0.73] * 2, color=PAPER, lw=1)
    ax.add_patch(plt.Rectangle((6.3, 1.3), 0.45, 3.65, fc=PAPER, ec=INK3, lw=1))
    ax.text(6.525, 5.05, "y", ha="center", fontsize=9, color=INK2)
    ax.text(4.4, 0.75, "one column per base model,\nevery row predicted out-of-fold", fontsize=8, color=INK2)
    arrow(ax, (3.85, 3.1), (4.3, 3.1), color=TEAL, lw=1.6)
    # right: meta-learner
    ax.text(7.55, 5.3, "3  Meta-learner", color=INK, weight="bold", fontsize=10)
    box(ax, 7.6, 3.55, 3.1, 1.2, "final_estimator\nfitted on (OOF columns, y)\nLogisticRegression / RidgeCV",
        fc=TEAL, ec=TEAL, color=PAPER, size=9)
    arrow(ax, (6.85, 3.6), (7.55, 4.0), color=TEAL, lw=1.6)
    # bottom right: predict time
    box(ax, 7.6, 1.55, 3.1, 1.25, "at predict time: base models\nrefitted on ALL dev rows\n-> their predictions -> meta",
        fc=PAPER, ec=INK3, size=9)
    arrow(ax, (9.15, 3.5), (9.15, 2.85), color=INK3)
    box(ax, 7.6, 0.35, 3.1, 0.8, "cost: (folds + 1) fits per base model", fc=PAPER, ec=RULE, color=INK2, size=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "stacking-flow.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 2. correlation heatmaps
def fig_corr(R):
    cmap = LinearSegmentedColormap.from_list("paper_teal", [PAPER, TEAL_L, TEAL])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9))
    for ax, (ds, title) in zip(axes, [("adult", "Adult (residual = y - P(>50K))"),
                                      ("california", "California housing (residual = y - prediction)")]):
        C = np.array(R[ds]["residual_corr"]["matrix"])
        names = R[ds]["residual_corr"]["models"]
        ax.imshow(C, cmap=cmap, vmin=0.4, vmax=1.0)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels([LABEL[n] for n in names], rotation=35, ha="right")
        ax.set_yticklabels([LABEL[n] for n in names])
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                        color=PAPER if C[i, j] > 0.85 else INK)
        ax.set_title(title, loc="left", color=INK)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    fig.text(0.01, 0.01, "Pearson correlation of test-set residuals, mean of 5 seeds.  "
             "*Linear = logistic regression (Adult) / ridge regression (California).", color=INK3, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "error-correlation.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 3. gain bars
def fig_gain(R):
    common = [("average", "average"), ("weighted", "weighted avg"), ("stack", "stacking (sklearn default)")]
    extra = {"adult": ("stack_logit", "stacking, logit inputs"),
             "california": ("stack_pt", "stacking + passthrough")}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    for ax, (ds, k, title) in zip(axes, [("adult", "log_loss", "Adult: log loss reduction vs best single"),
                                         ("california", "rmse", "California: RMSE reduction vs best single")]):
        methods = common + [extra[ds], ("blend", "blending")]
        G = R[ds]["groups"]
        for gi, (g, col) in enumerate([("boosters", INK3), ("diverse", TEAL)]):
            for mi, (m, lab) in enumerate(methods):
                v = G[g]["methods"][m][k]["vs_best"]
                gain, lo, hi = -v["mean"], -v["hi"], -v["lo"]  # positive = better (metric is lower-better)
                y = mi + gi * 0.38
                ax.barh(y, gain, height=0.34, color=col, alpha=0.9,
                        label=("3 boosters" if g == "boosters" else "diverse 4") if mi == 0 else None)
                ax.plot([lo, hi], [y, y], color=INK, lw=1.2)
                ax.plot([lo, lo], [y - 0.08, y + 0.08], color=INK, lw=1.2)
                ax.plot([hi, hi], [y - 0.08, y + 0.08], color=INK, lw=1.2)
        ax.set_yticks([i + 0.19 for i in range(len(methods))])
        ax.set_yticklabels([lab for _, lab in methods])
        ax.invert_yaxis()
        ax.axvline(0, color=INK2, lw=1)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("gain (positive = better than best single model)")
        ax.grid(axis="x", color=RULE, lw=0.6)
    axes[1].legend(loc="lower left", fontsize=9)
    fig.text(0.01, 0.01, "Bars: mean paired difference over 5 seeds; whiskers: 95% t interval. "
             "Best single = best 5-fold OOF score within the group.", color=INK3, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "gain-over-best.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_flow()
    rp = HERE / "results.json"
    if rp.exists():
        R = json.loads(rp.read_text(encoding="utf-8"))
        fig_corr(R)
        fig_gain(R)
    print("figures written to", OUT.name)
