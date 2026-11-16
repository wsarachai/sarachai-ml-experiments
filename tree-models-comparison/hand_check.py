"""Hand check for the post: reading "is A better than B?" from per-seed scores.

The post works the numbers by hand from a small table of per-seed test
ROC-AUC on Adult (default setting, LightGBM vs CatBoost; then matched
setting, XGBoost vs CatBoost), rounded to 4 decimals.  This script
  1. checks that the table in the post equals results.json rounded to 4 decimals,
  2. redoes every hand step (mean, SD with n-1, paired difference, SE,
     95% t interval) and asserts the values written in the post.

Run:    .venv/Scripts/python.exe tree-models-comparison/hand_check.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from scipy import stats

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text(encoding="utf-8"))

# the table printed in the post (seeds 0..4)
TABLE = {
    ("default", "LightGBM"): [0.9277, 0.9290, 0.9302, 0.9263, 0.9283],
    ("default", "CatBoost"): [0.9284, 0.9307, 0.9315, 0.9278, 0.9303],
    ("matched", "XGBoost"): [0.9275, 0.9299, 0.9315, 0.9272, 0.9298],
    ("matched", "CatBoost"): [0.9287, 0.9307, 0.9317, 0.9263, 0.9299],
}


def close(a, b, tol):
    assert abs(a - b) <= tol, (a, b)


for (setting, model), vals in TABLE.items():
    got = [round(v, 4) for v in R["adult"]["summary"][setting][model]["roc_auc"]["per_seed"]]
    assert got == vals, (setting, model, got, vals)
print("table matches results.json (rounded to 4 decimals)")


def mean(x):
    return sum(x) / len(x)


def sd(x):
    m = mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))


T = stats.t.ppf(0.975, 4)
close(T, 2.776, 5e-4)


def report(a_key, b_key, expect):
    a, b = TABLE[a_key], TABLE[b_key]
    d = [round(x - y, 4) for x, y in zip(a, b)]
    md, sdd = mean(d), sd(d)
    se = sdd / math.sqrt(len(d))
    half = T * se
    print(f"{a_key[1]} vs {b_key[1]} ({a_key[0]}):")
    print(f"  mean A {mean(a):.5f} SD {sd(a):.5f} | mean B {mean(b):.5f} SD {sd(b):.5f}")
    print(f"  d = A - B per seed {d}")
    print(f"  mean d {md:+.5f}  SD d {sdd:.5f}  SE {se:.5f}  95% interval {md - half:+.5f} .. {md + half:+.5f}")
    for k, v in expect.items():
        if k == "d":
            continue
        close({"mean_a": mean(a), "sd_a": sd(a), "mean_b": mean(b), "sd_b": sd(b), "mean_d": md,
               "sd_d": sdd, "se": se, "half": half}[k], v, 6e-6)
    assert d == expect["d"]
    return md, half


# values written in the post
md, half = report(("default", "LightGBM"), ("default", "CatBoost"), {
    "d": [-0.0007, -0.0017, -0.0013, -0.0015, -0.0020],
    "mean_a": 0.92830, "sd_a": 0.00145, "mean_b": 0.92974, "sd_b": 0.00157,
    "mean_d": -0.00144, "sd_d": 0.00049, "se": 0.00022, "half": 0.00061})
assert md + half < 0          # interval excludes 0 -> beyond seed noise
# unpaired view: the two mean +- SD bands overlap, so eyeballing them says "tie"
assert 0.92830 + 0.00145 > 0.92974 - 0.00157

md, half = report(("matched", "XGBoost"), ("matched", "CatBoost"), {
    "d": [-0.0012, -0.0008, -0.0002, 0.0009, -0.0001],
    "mean_d": -0.00028, "sd_d": 0.00080, "se": 0.00036, "half": 0.00099})
assert md - half < 0 < md + half  # interval contains 0 -> within seed noise
print("all hand-check assertions passed")
