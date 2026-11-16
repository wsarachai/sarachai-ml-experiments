"""Reproduction of the hand calculations in the "xgboost" blog post.

Every number in the post comes from a 6-row regression example (and a 4-row
logistic example). This script recomputes each table by hand from the gain
and leaf-weight formulas, asserts the published values, and then checks them
against the installed XGBoost (`tree_method="exact"`).

Run from the repository root (the folder is called xgboost-post so that it
does not shadow the xgboost package):
    .venv/Scripts/python.exe xgboost-post/experiments.py
Output is printed and written to xgboost-post/results.txt.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

HERE = Path(__file__).resolve().parent
buf = io.StringIO()
CHECKS: list[tuple[str, str, str, str]] = []

X = np.array([[1], [2], [3], [4], [5], [6]], dtype=np.float32)
Y = np.array([10, 12, 11, 40, 42, 41], dtype=np.float32)


def say(*args) -> None:
    print(*args)
    print(*args, file=buf)


def header(title: str) -> None:
    say("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def check(label, published, value, tol=5e-4, fmt="{:.4f}") -> None:
    """Numeric: exact if |diff| <= tol (published values are rounded); strings: equality."""
    if isinstance(published, str) or isinstance(value, str):
        pub_s, val_s = str(published), str(value)
        status = "exact" if pub_s == val_s else "DIFFERS"
    else:
        pub_s, val_s = fmt.format(published), fmt.format(value)
        status = "exact" if abs(published - value) <= tol else "DIFFERS"
    CHECKS.append((label, pub_s, val_s, status))


def split_table(g, h, lam):
    """Gain of every split x < t (t = midpoints), paper scale (with the 1/2)."""
    G, H = g.sum(), h.sum()
    rows = []
    for k in range(1, len(g)):
        GL, HL = g[:k].sum(), h[:k].sum()
        GR, HR = G - GL, H - HL
        left = 0.5 * GL ** 2 / (HL + lam)
        right = 0.5 * GR ** 2 / (HR + lam)
        gain = left + right - 0.5 * G ** 2 / (H + lam)
        rows.append((k + 0.5, GL, HL, GR, HR, left, right, gain))
    return rows


def show(rows) -> None:
    say("split  | G_L    H_L | G_R    H_R | 1/2 GL^2/(HL+l) | 1/2 GR^2/(HR+l) | gain")
    for t, GL, HL, GR, HR, l, r, gain in rows:
        say(f"x<{t:<4} | {GL:6.1f} {HL:4.1f} | {GR:6.1f} {HR:4.1f} | {l:15.3f} | {r:15.3f} | "
            f"{gain:8.3f}")


def hand() -> None:
    header("PART 1  Hand calculation (squared error, depth 1)")
    F0 = Y.mean()
    say(f"F0 = mean(y) = {F0}")
    assert F0 == 26.0
    g, h = (F0 - Y).astype(float), np.ones(6)
    assert g.tolist() == [16, 14, 15, -14, -16, -15]
    assert g.sum() == 0 and h.sum() == 6

    say("\nlambda = 0")
    rows = split_table(g, h, 0.0)
    show(rows)
    pub = [153.6, 337.5, 675.0, 360.375, 135.0]
    assert np.allclose([r[-1] for r in rows], pub)
    pub_lr = [(128.0, 25.6), (225.0, 112.5), (337.5, 337.5), (120.125, 240.25), (22.5, 112.5)]
    assert np.allclose([(r[5], r[6]) for r in rows], pub_lr)
    best = max(rows, key=lambda r: r[-1])
    assert best[0] == 3.5
    wl, wr = -best[1] / best[2], -best[3] / best[4]
    assert (wl, wr) == (-15, 15)
    F1 = np.where(X[:, 0] < 3.5, F0 + wl, F0 + wr)
    assert F1.tolist() == [11, 11, 11, 41, 41, 41]
    check("lambda=0 gain table (5 splits)", "asserted", "asserted")

    say("\nlambda = 1, eta = 0.3")
    rows1 = split_table(g, h, 1.0)
    show(rows1)
    pub1 = [85.333, 240.0, 506.25, 256.267, 75.0]
    assert np.allclose([r[-1] for r in rows1], pub1, atol=5e-4)
    pub1_lr = [(64.0, 21.333), (150.0, 90.0), (253.125, 253.125), (96.1, 160.167), (18.75, 56.25)]
    assert np.allclose([(r[5], r[6]) for r in rows1], pub1_lr, atol=5e-4)
    wl1 = -45 / (3 + 1)
    assert wl1 == -11.25 and 0.3 * wl1 == -3.375
    assert (26 - 3.375, 26 + 3.375) == (22.625, 29.375)
    check("lambda=1 gain table + leaves", "asserted", "asserted")

    say("\ntree 2 (lambda = 0) on F1")
    g2 = (F1 - Y).astype(float)
    assert g2.tolist() == [1, -1, 0, 1, -1, 0]
    rows2 = split_table(g2, h, 0.0)
    show(rows2)
    pub2 = [0.6, 0.0, 0.0, 0.375, 0.0]
    assert np.allclose([r[-1] for r in rows2], pub2)
    assert np.allclose([(r[5], r[6]) for r in rows2],
                       [(0.5, 0.1), (0, 0), (0, 0), (0.125, 0.25), (0, 0)])
    F2 = np.where(X[:, 0] < 1.5, F1 - 1.0, F1 + 0.2)
    assert np.allclose(F2, [10, 11.2, 11.2, 41.2, 41.2, 41.2])
    check("tree-2 gain table + predictions", "asserted", "asserted")

    say("\nlogistic, y = [0,0,1,1], base_score 0.5")
    p = 0.5
    gl, hl = np.array([0.5, 0.5, -0.5, -0.5]), np.full(4, 0.25)
    gain = 0.5 * (gl[:2].sum() ** 2 / hl[:2].sum() + gl[2:].sum() ** 2 / hl[2:].sum() - 0)
    assert gain == 2.0
    w = -gl[:2].sum() / hl[:2].sum()
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    say(f"gain={gain} w_L={w} p_L={sig(w):.4f} p_R={sig(-w):.4f} "
        f"(base added as raw 0.5: {sig(0.5 + w):.4f})")
    assert round(sig(-2), 4) == 0.1192 and round(sig(2), 4) == 0.8808
    assert round(sig(0.5 - 2), 4) == 0.1824
    assert p * (1 - p) == 0.25
    check("logistic hand values", "asserted", "asserted")


def train(params, rounds=1, x=X, y=Y):
    base = dict(objective="reg:squarederror", max_depth=1, eta=1.0, reg_lambda=0.0,
                gamma=0.0, min_child_weight=0.0, tree_method="exact", base_score=26.0)
    base.update(params)
    return xgb.train(base, xgb.DMatrix(x, label=y), num_boost_round=rounds)


def library() -> None:
    header(f"PART 2  Check against xgboost {xgb.__version__} (tree_method='exact')")
    b = train({}, rounds=2)
    t0 = json.loads(b.get_dump(dump_format="json", with_stats=True)[0])
    say("tree 0:", json.dumps(t0))
    check("split chosen", "x<3.5", f"x<{t0['split_condition']}")
    check("gain reported by xgboost", 1350.0, float(t0["gain"]), fmt="{:.1f}")
    leaves = sorted(c["leaf"] for c in t0["children"])
    check("leaf weights", "-15.0/+15.0", f"{leaves[0]:+.1f}/{leaves[1]:+.1f}")
    p1 = b.predict(xgb.DMatrix(X), iteration_range=(0, 1))
    p2 = b.predict(xgb.DMatrix(X), iteration_range=(0, 2))
    say("predict after tree 1:", p1.tolist(), " after tree 2:", np.round(p2, 4).tolist())
    check("prediction after tree 1", "11 11 11 41 41 41", " ".join(f"{v:g}" for v in p1))
    check("prediction after tree 2", "10 11.2 11.2 41.2 41.2 41.2",
          " ".join(f"{v:.4g}" for v in p2))
    t1 = json.loads(b.get_dump(dump_format="json", with_stats=True)[1])
    check("tree 2 split", "x<1.5", f"x<{t1['split_condition']}")

    b = train({"reg_lambda": 1.0, "eta": 0.3})
    p = b.predict(xgb.DMatrix(X))
    say("lambda=1, eta=0.3:", p.tolist())
    check("lambda=1 eta=0.3 predictions", "22.625/29.375", f"{p[0]:.3f}/{p[-1]:.3f}")

    # logistic
    xl = np.array([[1], [2], [3], [4]], dtype=np.float32)
    yl = np.array([0, 0, 1, 1], dtype=np.float32)
    b = train({"objective": "binary:logistic", "base_score": 0.5}, x=xl, y=yl)
    p = b.predict(xgb.DMatrix(xl))
    tl = json.loads(b.get_dump(dump_format="json", with_stats=True)[0])
    say("logistic:", np.round(p, 4).tolist(), "gain", tl["gain"])
    check("logistic probabilities", "0.1192/0.8808", f"{p[0]:.4f}/{p[-1]:.4f}")
    # Not printed in the post: the library reports 4 = 2 x the hand gain of 2 (same 2x scale).
    assert float(tl["gain"]) == 4.0

    # gamma and min_child_weight (lambda = 0; library gain = 1350).
    # Note: the post justifies gamma=600 with "675 - 600 > 0" (paper scale), while
    # its 1350/1351 rows only make sense on the library scale (gain 1350).
    say("\ngamma / min_child_weight (lambda = 0):")
    for params, published in (({"gamma": 600}, "split"), ({"gamma": 1000}, "split"),
                              ({"gamma": 1350}, "split"), ({"gamma": 1351}, "no split"),
                              ({"min_child_weight": 3.0}, "split"),
                              ({"min_child_weight": 3.5}, "no split")):
        bb = train(params)
        dump = json.loads(bb.get_dump(dump_format="json")[0])
        got = "split" if "children" in dump else "no split"
        pred = bb.predict(xgb.DMatrix(X))
        say(f"  {params}: {got}  predict={np.round(pred, 3).tolist()}")
        check(f"{params}", published, got)
        if got == "no split":
            check(f"{params} predicts 26 everywhere", "26", f"{float(np.unique(pred)[0]):g}")

    # default base_score is estimated from the data (mean for squared error)
    d = xgb.DMatrix(X, label=Y)
    bd = xgb.train({"objective": "reg:squarederror", "tree_method": "exact"}, d, 1)
    cfg = json.loads(bd.save_config())
    bs = cfg["learner"]["learner_model_param"]["base_score"]
    say(f"\nbase_score chosen automatically: {bs}")
    check("automatic base_score (= mean y)", 26.0, float(bs.strip("[]")), fmt="{:.1f}")


def main() -> None:
    say(f"python {sys.version.split()[0]} | xgboost {xgb.__version__} | numpy {np.__version__}")
    hand()
    say("all hand-calculated values assert OK")
    library()
    header("SUMMARY  published vs reproduced")
    w = max(len(c[0]) for c in CHECKS)
    for label, pub, val, status in CHECKS:
        say(f"{label:<{w}} | published {pub:>28} | reproduced {val:>28} | {status}")
    (HERE / "results.txt").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
