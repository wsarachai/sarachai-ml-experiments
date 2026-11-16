"""Draw the article figures from hand_calc data and results.json.

Output: figures/*.png (paper-like palette: off-white background, ink text,
one teal accent). Thai labels need a Thai-capable font; on Windows
"Leelawadee UI" or "Tahoma" works.

Run after experiments.py:  python figures.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

PAPER = "#FAF7F2"
INK = "#1F2328"
INK3 = "#8A8F98"
HAIR = "#E6E1D8"
TEAL = "#0F766E"
AMBER = "#D97706"

available = {f.name for f in font_manager.fontManager.ttflist}
for name in ("Leelawadee UI", "Tahoma", "Noto Sans Thai"):
    if name in available:
        plt.rcParams["font.family"] = name
        break
plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": HAIR,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 11,
})

# ------------------------------------------------------------- figure 1: binning
hours = [1, 2, 3, 4, 5, 6, 7, 8]
score = [10, 12, 11, 40, 42, 41, 47, 45]
exact_thr = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
exact_gain = [252.0, 533.333, 960.0, 650.25, 426.667, 300.0, 112.0]  # from hand_calc.py
bin_edges = [2.5, 4.5, 6.5]
bin_G = [40, 11, -21, -30]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.1, 1, 1.1]})

ax = axes[0]
ax.scatter(hours, score, color=INK, zorder=3, s=36)
for e in bin_edges:
    ax.axvline(e, color=TEAL, ls="--", lw=1.4)
for i, c in enumerate([1.5, 3.5, 5.5, 7.5]):
    ax.text(c, 51, f"bin {i}", ha="center", color=TEAL, fontsize=10)
ax.set_ylim(0, 55)
ax.set_xlabel("ชั่วโมงอ่านหนังสือ (x)")
ax.set_ylabel("คะแนน (y)")
ax.set_title("ก) 8 แถว ถูกจับเป็น 4 bin", fontsize=12)

ax = axes[1]
ax.bar(range(4), bin_G, color=TEAL, width=0.6)
for i, v in enumerate(bin_G):
    ax.text(i, v + (2 if v > 0 else -5), f"{v:+d}", ha="center", fontsize=10)
ax.axhline(0, color=INK, lw=0.8)
ax.set_xticks(range(4), [f"bin {i}" for i in range(4)])
ax.set_ylabel("G = ผลรวม g ใน bin (H = 2 ทุก bin)")
ax.set_ylim(-38, 50)
ax.set_title("ข) histogram ของ gradient", fontsize=12)

ax = axes[2]
cols = [TEAL if t in bin_edges else HAIR for t in exact_thr]
edges = [TEAL if t in bin_edges else INK3 for t in exact_thr]
ax.bar(range(7), exact_gain, color=cols, edgecolor=edges, width=0.65)
gain_labels = ["252", "533.3", "960", "650.25", "426.7", "300", "112"]
for i, v in enumerate(exact_gain):
    ax.text(i, v + 15, gain_labels[i], ha="center", fontsize=9)
ax.set_xticks(range(7), [f"<{t}" for t in exact_thr])
ax.set_xlabel("จุดแบ่ง x")
ax.set_ylabel("gain")
ax.set_ylim(0, 1080)
ax.annotate("exact เลือก 960", xy=(2, 960), xytext=(3.4, 1010), fontsize=10, color=INK,
            arrowprops=dict(arrowstyle="->", color=INK3))
ax.annotate("hist เลือก 650.25", xy=(3, 650), xytext=(4.2, 760), fontsize=10, color=TEAL,
            arrowprops=dict(arrowstyle="->", color=TEAL))
ax.set_title("ค) hist ดูได้แค่ 3 จุด (สีเขียว) จาก 7", fontsize=12)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "binning.png"), dpi=150)
plt.close(fig)

# ------------------------------------------------------------- figure 2: time vs rows
with open(os.path.join(HERE, "results.json"), encoding="utf-8") as f:
    R = json.load(f)
sizes = R["sizes"]
styles = {
    "GBC (depth 3)": ("GradientBoostingClassifier (ลึก 3)", INK, "o", "-"),
    "HGB depth 3, 1 thread": ("Hist… ลึก 3, 1 เธรด", TEAL, "s", "--"),
    "HGB depth 3, all threads": ("Hist… ลึก 3, ทุกเธรด", TEAL, "s", "-"),
    "HGB 31 leaves, all threads": ("Hist… 31 ใบ (ค่าเริ่มต้น), ทุกเธรด", AMBER, "^", "-"),
}
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
for key, (label, color, marker, ls) in styles.items():
    r = R["speed"][key]
    axes[0].plot(sizes, r["time"], color=color, marker=marker, ls=ls, label=label, lw=2)
    axes[1].plot(sizes, r["acc"], color=color, marker=marker, ls=ls, label=label, lw=2)
for ax in axes:
    ax.set_xscale("log")
    ax.set_xlabel("จำนวนแถวฝึก (Covertype, log scale)")
axes[0].set_yscale("log")
axes[0].set_ylabel("เวลาฝึก 100 ต้น (วินาที, มัธยฐาน, log)")
axes[0].set_title("เวลาฝึก", fontsize=12)
axes[1].set_ylabel("accuracy บนชุดทดสอบ 50,000 แถว")
axes[1].set_title("ความแม่นยำ", fontsize=12)
axes[0].legend(frameon=False, fontsize=9.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "time-vs-rows.png"), dpi=150)
plt.close(fig)

# ------------------------------------------------------------- figure 3: max_bins
mb = R["max_bins"]
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.plot(mb["bins"], mb["acc"], color=TEAL, marker="o", lw=2, label="ชุดทดสอบ")
ax.plot(mb["bins"], mb["train_acc"], color=INK3, marker="o", lw=1.5, ls="--", label="ชุดฝึก")
for b, a in zip(mb["bins"], mb["acc"]):
    ax.text(b, a - 0.012, f"{a:.3f}", ha="center", fontsize=9, color=TEAL)
ax.set_xscale("log", base=2)
ax.set_xticks(mb["bins"], [str(b) for b in mb["bins"]])
ax.set_xlabel("max_bins")
ax.set_ylabel("accuracy")
ax.set_title("max_bins กับความแม่นยำ (100,000 แถว, 100 ต้น 31 ใบ)", fontsize=12)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "max-bins.png"), dpi=150)
plt.close(fig)
print("wrote", os.listdir(OUT))
