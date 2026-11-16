"""Draw the three figures of the "data-preparation-for-trees" post from results.json.

Run after experiments.py:  .venv/Scripts/python.exe data-preparation-for-trees/figures.py
Output: data-preparation-for-trees/figures/{scaling,nan-direction,encoding}.png
Paper look: off-white background, ink text, one teal accent, no gradients.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
R = json.loads((HERE / "results.json").read_text(encoding="utf-8"))

PAPER = "#F7F5F0"
INK = "#1F2328"
INK3 = "#8A8F98"
HAIR = "#E2DDD3"
TEAL = "#0F766E"
TEAL_LIGHT = "#99C9C3"

plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": HAIR,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 11,
    "font.family": "DejaVu Sans",
})

# ---------------------------------------------------------------- 1. scaling
x1 = np.arange(1, 7, dtype=float)
y = np.array([0, 0, 1, 0, 1, 1])
ts = R["toy_scaling"]
mu, sd = x1.mean(), x1.std()
rows = [
    ("raw: hours", x1, ts["raw"]["thresholds"]),
    ("standardised: (x - mean) / sd", (x1 - mu) / sd, ts["standardised"]["thresholds"]),
    ("log(x)", np.log(x1), ts["log"]["thresholds"]),
]
fig, axes = plt.subplots(3, 1, figsize=(9, 5.6))
for ax, (title, xv, thr) in zip(axes, rows):
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    ax.axhline(0, color=INK3, lw=1)
    for xi, yi, i in zip(xv, y, range(1, 7)):
        ax.scatter(xi, 0, s=110, zorder=3, color=TEAL if yi == 1 else PAPER,
                   edgecolor=TEAL if yi == 1 else INK, linewidth=1.6)
        ax.text(xi, 0.35, f"row {i}", ha="center", fontsize=9, color=INK3)
    for t in thr:
        ax.axvline(t, color=INK, ls="--", lw=1.1)
        ax.text(t, -0.55, f"{t:.3f}", ha="center", fontsize=9)
    ax.set_ylim(-0.8, 0.7)
    pad = (xv.max() - xv.min()) * 0.06
    ax.set_xlim(xv.min() - pad, xv.max() + pad)
    ax.set_title(title, loc="left", fontsize=11)
fig.text(0.5, 0.005, "filled = pass, open = fail; dashed = thresholds of the fully grown x1-only tree. "
         "Same leaves in all three.", ha="center", fontsize=9, color=INK3)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(OUT / "scaling.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- 2. NaN default direction
h = R["xgb_nan_hand"]
labels = [f"x < {r['thr']:g}" for r in h["rows"]]
gl = [r["gain_nan_left"] for r in h["rows"]]
gr = [r["gain_nan_right"] for r in h["rows"]]
xi = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(8.5, 4.4))
b1 = ax.bar(xi - 0.19, gl, 0.38, color=PAPER, edgecolor=INK, linewidth=1.2, label="NaN row sent left")
b2 = ax.bar(xi + 0.19, gr, 0.38, color=TEAL, label="NaN row sent right")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 8, f"{b.get_height():.1f}",
                ha="center", fontsize=9)
ax.set_xticks(xi, labels)
ax.set_ylabel("gain (lambda = 0)")
ax.set_title("XGBoost tries both directions for the missing row at every threshold", loc="left", fontsize=11)
ax.legend(frameon=False, loc="upper left")
ax.set_ylim(0, max(gl + gr) * 1.15)
fig.tight_layout()
fig.savefig(OUT / "nan-direction.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- 3. encodings
E = R["encoding_adult"]
models = list(E.keys())
encs = ["one-hot", "ordinal", "target", "native"]
shades = {"one-hot": "#D9D4C9", "ordinal": INK3, "target": TEAL_LIGHT, "native": TEAL}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
w = 0.2
for k, enc in enumerate(encs):
    xs, auc, tm = [], [], []
    for i, m in enumerate(models):
        if enc in E[m]:
            xs.append(i + (k - 1.5) * w)
            auc.append(E[m][enc]["auc"])
            tm.append(E[m][enc]["time"])
    a1.bar(xs, auc, w, color=shades[enc], label=enc, edgecolor=INK if enc == "one-hot" else shades[enc],
           linewidth=0.6)
    a2.bar(xs, tm, w, color=shades[enc], label=enc, edgecolor=INK if enc == "one-hot" else shades[enc],
           linewidth=0.6)
allauc = [E[m][e]["auc"] for m in models for e in E[m]]
a1.set_ylim(min(allauc) - 0.01, max(allauc) + 0.012)
a1.set_xticks(range(len(models)), models)
a1.set_ylabel("test ROC AUC")
a1.set_title("Accuracy of the ranking (higher is better)", loc="left", fontsize=11)
a2.set_xticks(range(len(models)), models)
a2.set_yscale("log")
a2.set_ylabel("fit time, seconds (median of 3, log scale)")
a2.set_title("Fit time incl. encoding (lower is better)", loc="left", fontsize=11)
a1.legend(frameon=False, ncols=4, loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "encoding.png", dpi=150)
plt.close(fig)
print("wrote", [p.name for p in OUT.glob("*.png")])
