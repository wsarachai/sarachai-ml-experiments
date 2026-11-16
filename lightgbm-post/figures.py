"""Figures for the "lightgbm" blog post (Paper palette: off-white, ink, one teal accent).

1. lightgbm--growth.png  leaf-wise vs level-wise trees of the 8-row hand example
2. lightgbm--goss.png    GOSS on the 10-row example: kept, sampled (x4), dropped
3. lightgbm--speed.png   training time vs training rows on Covertype, parsed from results.txt

Run from the repository root after results.txt exists:
    .venv/Scripts/python.exe lightgbm-post/figures.py
"""

from __future__ import annotations

import re
import sys
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from hand_example import G2, Y1, grow  # noqa: E402

FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

PAPER = "#F8F6F1"
INK = "#1F2328"
INK_2 = "#5B6270"
HAIR = "#CFCAC0"
TEAL = "#0F766E"
TEAL_SOFT = "#D5EBE8"
GREY = "#B9BDC4"

plt.rcParams.update({
    "font.family": ["Noto Sans Thai", "DejaVu Sans"],
    "axes.edgecolor": HAIR,
    "axes.labelcolor": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "text.color": INK,
})


# ---------------------------------------------------------------------------
# 1. growth
# ---------------------------------------------------------------------------

def box(ax, x, y, text, leaf=False, w=1.55, h=0.62, highlight=False):
    fc = TEAL_SOFT if leaf else "white"
    ec = TEAL if (highlight or not leaf) else TEAL_SOFT
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.8 if highlight else 1.4))
    ax.text(x, y, text, ha="center", va="center", fontsize=12.5, color=INK, linespacing=1.3)


def edge(ax, x0, y0, x1, y1):
    ax.plot([x0, x1], [y0 - 0.31, y1 + 0.31], color=INK_2, lw=1.2, zorder=0)


def fig_growth():
    f0 = Fraction(int(Y1.sum()), len(Y1))
    g = [f0 - Fraction(int(v)) for v in Y1]
    _, _, leaf_total = grow(g, "leaf")
    _, _, level_total = grow(g, "level")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6), dpi=150)
    fig.patch.set_facecolor(PAPER)
    for ax in axes:
        ax.set_facecolor(PAPER)
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 6.6)
        ax.axis("off")

    # Level-wise (left panel)
    a = axes[0]
    a.set_title("Level-wise (max_depth=2)", fontsize=16, color=INK, fontweight="bold")
    box(a, 4, 5.4, "x ≤ 4\ngain 144")
    box(a, 2, 3.7, "x ≤ 2\ngain 8", highlight=False)
    box(a, 6, 3.7, "x ≤ 6\ngain 98")
    edge(a, 4, 5.4, 2, 3.7)
    edge(a, 4, 5.4, 6, 3.7)
    for x, txt in [(1.1, "แถว 1–2\n→ 15"), (2.9, "แถว 3–4\n→ 19"), (5.1, "แถว 5–6\n→ 36"), (6.9, "แถว 7–8\n→ 22")]:
        box(a, x, 2.0, txt, leaf=True, w=1.5)
    for x0, x1 in [(2, 1.1), (2, 2.9), (6, 5.1), (6, 6.9)]:
        edge(a, x0, 3.7, x1, 2.0)
    a.text(4, 0.75, f"ต้องแตกทุกโหนดในชั้นเดียวกัน แม้ฝั่งซ้ายได้กำไรแค่ 8\nกำไรรวม {level_total}",
           ha="center", fontsize=12.5, color=INK_2)

    # Leaf-wise (right panel)
    b = axes[1]
    b.set_title("Leaf-wise (num_leaves=4)", fontsize=16, color=INK, fontweight="bold")
    box(b, 3, 5.6, "x ≤ 4\ngain 144")
    box(b, 1.5, 4.2, "แถว 1–4\n→ 17", leaf=True, w=1.5)
    box(b, 4.5, 4.2, "x ≤ 6\ngain 98")
    box(b, 3.3, 2.8, "แถว 5–6\n→ 36", leaf=True, w=1.5)
    box(b, 5.7, 2.8, "x ≤ 7\ngain 36", highlight=True)
    box(b, 4.8, 1.4, "แถว 7\n→ 16", leaf=True, w=1.3)
    box(b, 6.6, 1.4, "แถว 8\n→ 28", leaf=True, w=1.3)
    edge(b, 3, 5.6, 1.5, 4.2)
    edge(b, 3, 5.6, 4.5, 4.2)
    edge(b, 4.5, 4.2, 3.3, 2.8)
    edge(b, 4.5, 4.2, 5.7, 2.8)
    edge(b, 5.7, 2.8, 4.8, 1.4)
    edge(b, 5.7, 2.8, 6.6, 1.4)
    b.text(1.5, 2.2, "ไม่แตก\n(กำไรแค่ 8)", ha="center", fontsize=11.5, color=INK_2)
    b.text(4, 0.3, f"แตกเฉพาะใบที่กำไรสูงสุดทีละใบ  ·  กำไรรวม {leaf_total}  ·  ลึก 3 ชั้น",
           ha="center", fontsize=12.5, color=TEAL)

    fig.add_artist(plt.Line2D([0.5, 0.5], [0.06, 0.94], color=HAIR, lw=1.2))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lightgbm--growth.png", facecolor=PAPER)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. GOSS
# ---------------------------------------------------------------------------

def fig_goss():
    g = [float(v) for v in G2]
    n = len(g)
    order = sorted(range(n), key=lambda i: -abs(g[i]))
    top = set(order[:2])
    rest = order[2:]
    sampled = {rest[0], rest[3]}  # the example draw printed by hand_example.py (rows 8 and 9)

    fig, ax = plt.subplots(figsize=(12, 5.4), dpi=150)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    xs = list(range(1, n + 1))
    for i in range(n):
        if i in top:
            ax.bar(xs[i], g[i], color=TEAL, width=0.62)
            label = "เก็บไว้"
        elif i in sampled:
            ax.bar(xs[i], g[i] * 4, color="none", edgecolor=TEAL, hatch="//", width=0.62, lw=1.4)
            ax.bar(xs[i], g[i], color=TEAL, alpha=0.55, width=0.62)
            label = "สุ่มได้ ×4"
        else:
            ax.bar(xs[i], g[i], color=GREY, width=0.62)
            label = "ไม่ใช้รอบนี้"
        va = "bottom" if g[i] >= 0 else "top"
        y = (max(g[i], g[i] * 4 if i in sampled else g[i]) + 0.12) if g[i] >= 0 else \
            (min(g[i], g[i] * 4 if i in sampled else g[i]) - 0.12)
        ax.text(xs[i], y, f"{g[i]:+.1f}\n{label}" if g[i] >= 0 else f"{label}\n{g[i]:+.1f}",
                ha="center", va=va, fontsize=10.5, color=INK if i in top or i in sampled else INK_2)
    ax.axhline(0, color=INK_2, lw=1)
    ax.set_xticks(xs, [f"แถว {x}" for x in xs])
    ax.set_ylabel("gradient g")
    ax.set_ylim(-4.6, 4.2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("GOSS: a = 0.2 เก็บ 2 แถวที่ |g| สูงสุด, b = 0.2 สุ่ม 2 จาก 8 แถวที่เหลือแล้วคูณ (1−a)/b = 4",
                 fontsize=13.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lightgbm--goss.png", facecolor=PAPER)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. speed (parsed from results.txt so the figure cannot drift from the run)
# ---------------------------------------------------------------------------

def fig_speed():
    text = (HERE / "results.txt").read_text(encoding="utf-8", errors="replace")
    part_c = text[text.find("PART C"):]
    data: dict[str, list[tuple[int, float, float]]] = {}
    n = None
    for line in part_c.splitlines():
        m = re.match(r"\s+n_train = (\d+)", line)
        if m:
            n = int(m.group(1))
            continue
        m = re.match(r"\s{4}(.+?)\s+time\s+([\d.]+) s .*test acc ([\d.]+)", line)
        if m and n:
            data.setdefault(m.group(1).strip(), []).append((n, float(m.group(2)), float(m.group(3))))
    styles = {
        "LightGBM (default, num_leaves=31)": (TEAL, "-", "o", "LightGBM (num_leaves=31)"),
        "LightGBM GOSS": (TEAL, "--", "s", "LightGBM + GOSS"),
        "XGBoost hist depthwise max_depth=6": (INK, "-", "o", "XGBoost hist, depthwise depth 6"),
        "XGBoost hist lossguide max_leaves=31": (INK_2, "--", "s", "XGBoost hist, lossguide 31 leaves"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=150)
    fig.patch.set_facecolor(PAPER)
    for ax in axes:
        ax.set_facecolor(PAPER)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(True, color="#E8E2D6", lw=0.8)
        ax.set_axisbelow(True)
        ax.set_xscale("log")
        ax.set_xlabel("จำนวนแถวฝึก (สเกล log)")
    for name, (c, ls, mk, lab) in styles.items():
        pts = data.get(name, [])
        if not pts:
            continue
        ns, ts, accs = zip(*pts)
        axes[0].plot(ns, ts, color=c, ls=ls, marker=mk, lw=2, label=lab)
        axes[1].plot(ns, accs, color=c, ls=ls, marker=mk, lw=2, label=lab)
    sizes = sorted({n for pts in data.values() for n, _, _ in pts})
    for ax in axes:
        ax.set_xticks(sizes, [f"{n // 1000:,}k" for n in sizes])
        ax.minorticks_off()
    axes[0].set_ylabel("เวลาฝึก 200 รอบ (วินาที, median)")
    axes[1].set_ylabel("accuracy บนชุดทดสอบ 100,000 แถว")
    axes[0].set_title("เวลาฝึก", fontsize=14, loc="left")
    axes[1].set_title("ความแม่นยำ", fontsize=14, loc="left")
    axes[0].legend(frameon=False, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lightgbm--speed.png", facecolor=PAPER)
    plt.close(fig)


if __name__ == "__main__":
    fig_growth()
    fig_goss()
    if (HERE / "results.txt").exists() and "PART C" in (HERE / "results.txt").read_text(errors="replace"):
        fig_speed()
    print("figures written to", FIG_DIR)
