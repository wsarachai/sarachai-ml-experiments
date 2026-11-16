"""Draw the post's figures from results.json (run report.py first).

Run:    .venv/Scripts/python.exe tree-models-comparison/figures.py
Output: tree-models-comparison/figures/*.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
R = json.loads((HERE / "results.json").read_text(encoding="utf-8"))

PAPER, INK, INK2, INK3, RULE = "#f7f5f0", "#1f1f1c", "#55524b", "#8a867c", "#d9d4c7"
TEAL = "#0f766e"
MODELS = ["RandomForest", "HistGB", "XGBoost", "LightGBM", "CatBoost"]
LABEL = {"RandomForest": "Random Forest", "HistGB": "HistGB (sklearn)", "XGBoost": "XGBoost",
         "LightGBM": "LightGBM", "CatBoost": "CatBoost"}

plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "axes.edgecolor": RULE, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": RULE,
    "grid.linewidth": 0.6, "legend.frameon": False,
})


def fig_metrics():
    panels = [("adult", "roc_auc", "Adult: test ROC-AUC (higher is better)"),
              ("california", "rmse", "California: test RMSE (lower is better)"),
              ("covertype", "accuracy", "Covertype 10k: test accuracy (higher is better)"),
              ("breast", "log_loss", "Breast cancer: test log loss (lower better)")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    for ax, (ds, k, title) in zip(axes.flat, panels):
        S = R[ds]["summary"]
        for j, (setting, color, off) in enumerate([("default", INK3, -0.15), ("matched", TEAL, 0.15)]):
            ys = [i + off for i in range(len(MODELS))]
            means = [S[setting][m][k]["mean"] for m in MODELS]
            sds = [S[setting][m][k]["sd"] for m in MODELS]
            ax.errorbar(means, ys, xerr=sds, fmt="o", color=color, ms=6, capsize=3, lw=1.5,
                        label="default" if setting == "default" else "matched (lr 0.1 + early stopping, ~32 leaves)")
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels([LABEL[m] for m in MODELS])
        ax.invert_yaxis()
        ax.set_title(title, loc="left", color=INK)
        ax.grid(axis="y", visible=False)
    axes[0, 0].legend(loc="lower left", fontsize=9, bbox_to_anchor=(0, 1.08), ncol=2)
    fig.text(0.01, 0.005, "Dots: mean over 5 seeds; bars: +-1 SD over seeds. Same split per seed for every model.",
             color=INK3, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(OUT / "metrics-by-dataset.png", dpi=150)
    plt.close(fig)


def fig_scaling():
    runs = R["scaling"]["runs"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    markers = {"RandomForest": "s", "HistGB": "D", "XGBoost": "^", "LightGBM": "o", "CatBoost": "v"}
    shades = {"RandomForest": INK3, "HistGB": INK2, "XGBoost": INK, "LightGBM": TEAL, "CatBoost": INK2}
    styles = {"RandomForest": ":", "HistGB": "--", "XGBoost": "-", "LightGBM": "-", "CatBoost": "-."}
    for m in MODELS:
        rs = sorted([r for r in runs if r["model"] == m], key=lambda r: r["n_train"])
        n = [r["n_train"] for r in rs]
        for ax, key, scale in [(axes[0], "fit_s", 1), (axes[1], "predict_10k_s", 1000)]:
            v = [r[key] * scale for r in rs]
            ax.plot(n, v, styles[m], marker=markers[m], color=shades[m], lw=2, ms=7, label=LABEL[m])
            if key == "predict_10k_s":
                ax.annotate(LABEL[m], (n[-1], v[-1]), xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=8.5, color=INK2)
    for ax, t, yl in [(axes[0], "Fit time, 100 rounds (log-log)", "seconds"),
                      (axes[1], "Prediction time for 10,000 rows", "milliseconds")]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(t, loc="left")
        ax.set_xlabel("training rows")
        ax.set_ylabel(yl)
        ax.set_xlim(7000, 2.2e6)
    axes[0].legend(loc="upper left", fontsize=8.5)
    fig.text(0.01, 0.01, "Covertype, 7 classes, 4 threads per model, one machine; RF skipped at the full size. "
             "Timings are machine-dependent.", color=INK3, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "scaling.png", dpi=150)
    plt.close(fig)


def fig_tradeoff():
    S = R["adult"]["summary"]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for setting, color, mk in [("default", INK3, "o"), ("matched", TEAL, "D")]:
        for m in MODELS:
            x = S[setting][m]["fit_s"]["median"]
            y = S[setting][m]["roc_auc"]["mean"]
            ax.scatter(x, y, s=60, color=color, marker=mk, zorder=3, edgecolor=PAPER, linewidth=1.5,
                       label=setting if m == "RandomForest" else None)
            ax.annotate(f"{LABEL[m]}", (x, y), xytext=(7, 5) if setting == "matched" else (7, -13),
                        textcoords="offset points", fontsize=8.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("median fit time over 5 seeds (seconds, log scale)")
    ax.set_ylabel("mean test ROC-AUC over 5 seeds")
    ax.set_title("Adult: accuracy bought per second of training", loc="left")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "adult-auc-vs-time.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig_metrics()
    fig_scaling()
    fig_tradeoff()
    print("figures written to", OUT.name)
