"""Reproduction of the measured numbers in the "random-forest" blog post.

The post was written before this repository existed, so its exact seeds and
procedures are not recorded. This script states the procedure it infers for
every number, recomputes it, and compares it with the published value.

Parts
  1. Hand-calculated toy forest (6 rows, 3 stumps, m = 1): every Gini value,
     leaf probability, forest average and OOB value is asserted.
  2. Bootstrap coverage on n = 398: mean share of unique rows over 500 draws.
  3. Breast cancer, 10 re-splits: single unpruned tree vs. 300-tree forest.
  4. Seed-42 single tree: depth, leaves, train/test accuracy.
  5. Seed-42 forest: train/test/OOB accuracy for 1..300 trees.
  6. max_features sweep at 300 trees.
  7. MDI vs. permutation importance with two pure-noise features.
  8. Default parameter values of RandomForestClassifier/Regressor.

Run from the repository root:
    .venv/Scripts/python.exe random-forest/experiments.py
Output is printed and written to random-forest/results.txt.
"""

from __future__ import annotations

import inspect
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent
SEED = 42
buf = io.StringIO()
CHECKS: list[tuple[str, str, str, str]] = []


def say(*args) -> None:
    print(*args)
    print(*args, file=buf)


def header(title: str) -> None:
    say("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def check(label: str, published: float | str, value: float | str, tol: float = 0.0,
          fmt: str = "{:.4f}", status: str | None = None) -> None:
    """Record a published-vs-reproduced comparison.

    exact  : equal at the published precision
    noise  : differs, but by no more than `tol` (the stated noise level)
    DIFFERS: anything else
    `status` overrides the automatic verdict for qualitative claims.
    """
    if status is not None:
        pub_s, val_s = str(published), str(value)
    elif isinstance(published, str) or isinstance(value, str):
        status = "exact" if str(published) == str(value) else "DIFFERS"
        pub_s, val_s = str(published), str(value)
    else:
        pub_s, val_s = fmt.format(published), fmt.format(value)
        if pub_s == val_s:
            status = "exact"
        elif abs(published - value) <= tol:
            status = "within noise"
        else:
            status = "DIFFERS"
    CHECKS.append((label, pub_s, val_s, status))


def load_split(seed: int):
    data = load_breast_cancer()
    return train_test_split(data.data, data.target, test_size=0.3,
                            random_state=seed, stratify=data.target)


# ----------------------------------------------------------------- part 1 ---
TOY_X = np.array([[1, 3], [2, 1], [3, 4], [4, 1], [5, 5], [6, 4]], dtype=float)
TOY_Y = np.array([0, 0, 1, 0, 1, 1])
# Bootstraps and root features exactly as chosen in the post (1-based rows).
BOOT = {"T1": [1, 2, 2, 3, 4, 6], "T2": [1, 3, 3, 4, 5, 6], "T3": [2, 3, 4, 5, 5, 6]}
FEAT = {"T1": 0, "T2": 1, "T3": 0}


def gini(y) -> float:
    y = np.asarray(y)
    if len(y) == 0:
        return 0.0
    p = np.bincount(y, minlength=2) / len(y)
    return float(1 - (p ** 2).sum())


def best_stump(x, y):
    """Best threshold (midpoint) by weighted Gini on one feature."""
    vals = np.unique(x)
    best = None
    for t in (vals[:-1] + vals[1:]) / 2:
        left, right = y[x <= t], y[x > t]
        g = len(left) / len(y) * gini(left) + len(right) / len(y) * gini(right)
        if best is None or g < best[1] - 1e-12:
            best = (t, g, gini(left), gini(right), left.mean(), right.mean())
    return best


def part1() -> None:
    header("PART 1  Hand-calculated toy forest (asserted)")
    published = {  # tree: (threshold, G_L, G_R, G_split, P_left, P_right)
        "T1": (2.5, 0.0000, 0.4444, 0.2222, 0.0000, 0.6667),
        "T2": (3.5, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000),
        "T3": (4.5, 0.4444, 0.0000, 0.2222, 0.3333, 1.0000),
    }
    probs = {}
    for name, rows in BOOT.items():
        idx = np.array(rows) - 1
        x, y = TOY_X[idx, FEAT[name]], TOY_Y[idx]
        t, g, gl, gr, pl, pr = best_stump(x, y)
        say(f"{name}: x{FEAT[name] + 1} <= {t}  G_L={gl:.4f} G_R={gr:.4f} G_split={g:.4f} "
            f"P(left)={pl:.4f} P(right)={pr:.4f}")
        assert np.allclose((t, gl, gr, g, pl, pr), published[name], atol=5e-5), name
        probs[name] = np.where(TOY_X[:, FEAT[name]] <= t, pl, pr)
    # T1 alternative split 4.5 from the text: G_L = 0.32, G_split = 0.2667
    idx = np.array(BOOT["T1"]) - 1
    left = TOY_Y[idx][TOY_X[idx, 0] <= 4.5]
    assert abs(gini(left) - 0.32) < 1e-12
    assert abs(5 / 6 * gini(left) - 0.2667) < 5e-5
    # The post says "Gini at the root (before splitting) is 0.5, so the split lowers
    # impurity by 0.5 - 0.2222 = 0.2778". 0.5 is the Gini of the full 6-row data
    # (3 vs 3), but T1's root holds T1's bootstrap, y = (0,0,0,1,0,1): 2 of 6 are 1.
    root_full, root_t1 = gini(TOY_Y), gini(TOY_Y[idx])
    say(f"Gini of the full toy data = {root_full:.4f}; Gini at T1's root (its bootstrap) = "
        f"{root_t1:.4f}; decrease at T1 = {root_t1 - 2 / 9:.4f}")
    check("T1 root Gini (post: 0.5)", 0.5, root_t1)
    check("T1 impurity decrease (post: 0.2778)", 0.2778, root_t1 - 2 / 9)

    forest = (probs["T1"] + probs["T2"] + probs["T3"]) / 3
    say("forest average:", np.round(forest, 4).tolist())
    assert np.allclose(forest, [0.1111, 0.1111, 0.6667, 0.3333, 0.8889, 0.8889], atol=5e-5)
    assert ((forest > 0.5).astype(int) == TOY_Y).all()
    per_tree = {k: int(((v > 0.5).astype(int) == TOY_Y).sum()) for k, v in probs.items()}
    say("correct per tree:", per_tree)
    assert per_tree == {"T1": 5, "T2": 6, "T3": 5}

    # OOB: rows never drawn by a tree are predicted only by that tree.
    oob = {}
    for i in range(6):
        trees = [k for k, rows in BOOT.items() if i + 1 not in rows]
        if trees:
            oob[i + 1] = float(np.mean([probs[k][i] for k in trees]))
    say("OOB values:", {k: round(v, 4) for k, v in oob.items()})
    assert set(oob) == {1, 2, 5}
    assert np.allclose([oob[1], oob[2], oob[5]], [0.3333, 0.0, 0.6667], atol=5e-5)
    check("toy example (all hand values)", "asserted", "asserted")
    say("all hand-calculated values assert OK")


# ----------------------------------------------------------------- part 2 ---
def bootstrap_unique_share(n: int, draws: int, rng) -> np.ndarray:
    return np.array([len(np.unique(rng.integers(0, n, n))) / n for _ in range(draws)])


def part2() -> None:
    header("PART 2  Share of unique rows in a bootstrap of n = 398, 500 draws")
    n = 398
    theory = 1 - (1 - 1 / n) ** n
    say(f"theory 1-(1-1/n)^n = {theory:.4f}   limit 1-1/e = {1 - np.exp(-1):.4f}")
    check("theory 1-(1-1/n)^n, n=398", 0.6326, theory)
    for seed in (0, 42):
        s = bootstrap_unique_share(n, 500, np.random.default_rng(seed))
        say(f"default_rng({seed}): mean={s.mean():.4f} min={s.min():.4f} max={s.max():.4f} "
            f"sd={s.std():.4f}")
    # Spread of the mean over many 500-draw repetitions = its noise level.
    means = [bootstrap_unique_share(n, 500, np.random.default_rng(1000 + r)).mean()
             for r in range(50)]
    say(f"sd of the 500-draw mean over 50 repetitions = {np.std(means):.4f}")
    # Inferred procedure: legacy RandomState(0).randint(0, n, n), 500 draws.
    # Found by searching 430 procedure/seed combinations (default_rng / RandomState,
    # integers / choice, seeds 0..100 and a few common ones, RF estimators_samples_);
    # 6 of them give all three published numbers, RandomState(0) is the simplest.
    # Min/max take few distinct values (multiples of 1/398), so a match is weak evidence.
    rs = np.random.RandomState(0)
    s = np.array([len(np.unique(rs.randint(0, n, n))) / n for _ in range(500)])
    say(f"RandomState(0).randint: mean={s.mean():.4f} min={s.min():.4f} max={s.max():.4f}")
    check("bootstrap mean unique share (500 draws)", 0.6326, s.mean(), tol=0.0015)
    check("bootstrap min unique share", 0.5879, s.min(), tol=0.03)
    check("bootstrap max unique share", 0.6834, s.max(), tol=0.03)
    # The post's range is wide for n = 398; show where it falls in the sampling distribution.
    sd = np.sqrt(np.mean([bootstrap_unique_share(n, 500, np.random.default_rng(r)).var()
                          for r in range(5)]))
    say(f"per-draw sd of the unique share ~= {sd:.4f}; published min/max are "
        f"{(0.6326 - 0.5879) / sd:.1f} and {(0.6834 - 0.6326) / sd:.1f} sd from the mean")


# ----------------------------------------------------------------- part 3 ---
def part3() -> None:
    header("PART 3  10 re-splits (split seed = model seed = 0..9)")
    tree_acc, rf_acc = [], []
    for s in range(10):
        Xtr, Xte, ytr, yte = load_split(s)
        tree_acc.append(DecisionTreeClassifier(random_state=s).fit(Xtr, ytr).score(Xte, yte))
        rf = RandomForestClassifier(n_estimators=300, random_state=s, n_jobs=-1).fit(Xtr, ytr)
        rf_acc.append(rf.score(Xte, yte))
        say(f"seed {s}: tree={tree_acc[-1]:.4f} forest300={rf_acc[-1]:.4f}")
    for name, a, pub in (("single tree", tree_acc, (0.9322, 0.0206, 0.8947, 0.9649)),
                         ("forest 300", rf_acc, (0.9579, 0.0090, 0.9474, 0.9708))):
        a = np.array(a)
        say(f"{name}: mean={a.mean():.4f} sd(ddof=0)={a.std():.4f} sd(ddof=1)={a.std(ddof=1):.4f} "
            f"min={a.min():.4f} max={a.max():.4f}")
        check(f"{name} 10 seeds mean", pub[0], a.mean())
        check(f"{name} 10 seeds SD (population)", pub[1], a.std())
        check(f"{name} 10 seeds min", pub[2], a.min())
        check(f"{name} 10 seeds max", pub[3], a.max())


# ----------------------------------------------------------------- part 4 ---
def part4(Xtr, Xte, ytr, yte) -> None:
    header("PART 4  Seed-42 single tree")
    t = DecisionTreeClassifier(random_state=SEED).fit(Xtr, ytr)
    say(f"depth={t.get_depth()} leaves={t.get_n_leaves()} train={t.score(Xtr, ytr):.4f} "
        f"test={t.score(Xte, yte):.4f}")
    check("seed-42 tree depth", 6, t.get_depth(), fmt="{:.0f}")
    check("seed-42 tree leaves", 16, t.get_n_leaves(), fmt="{:.0f}")
    check("seed-42 tree train acc", 1.0, t.score(Xtr, ytr))
    check("seed-42 tree test acc", 0.9181, t.score(Xte, yte))


# ----------------------------------------------------------------- part 5 ---
def part5(Xtr, Xte, ytr, yte) -> None:
    header("PART 5  Seed-42 forest, 1..300 trees (oob_score=True, random_state=42)")
    published = {1: (0.9774, 0.9123, 0.5829), 3: (0.9874, 0.9240, 0.8065),
                 5: (0.9899, 0.9357, 0.9070), 10: (1.0000, 0.9298, 0.9497),
                 30: (0.9975, 0.9357, 0.9472), 100: (1.0000, 0.9357, 0.9573),
                 300: (1.0000, 0.9415, 0.9673)}
    say("trees | train | test | OOB")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # few-tree OOB warnings are expected
        for k, (ptr, pte, poob) in published.items():
            rf = RandomForestClassifier(n_estimators=k, oob_score=True, random_state=SEED,
                                        n_jobs=-1).fit(Xtr, ytr)
            tr, te, oob = rf.score(Xtr, ytr), rf.score(Xte, yte), rf.oob_score_
            say(f"{k:>5} | {tr:.4f} | {te:.4f} | {oob:.4f}")
            check(f"forest {k} trees train", ptr, tr)
            check(f"forest {k} trees test", pte, te)
            check(f"forest {k} trees OOB", poob, oob)


# ----------------------------------------------------------------- part 6 ---
def part6(Xtr, Xte, ytr, yte) -> None:
    header("PART 6  max_features sweep, 300 trees, random_state=42")
    published = {1: (0.9474, 0.9623), "log2": (0.9357, 0.9623), "sqrt": (0.9415, 0.9673),
                 10: (0.9357, 0.9698), None: (0.9415, 0.9623)}
    for mf, (pte, poob) in published.items():
        rf = RandomForestClassifier(n_estimators=300, max_features=mf, oob_score=True,
                                    random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
        say(f"max_features={mf!s:>5}: test={rf.score(Xte, yte):.4f} OOB={rf.oob_score_:.4f}")
        check(f"max_features={mf} test", pte, rf.score(Xte, yte))
        check(f"max_features={mf} OOB", poob, rf.oob_score_)


# ----------------------------------------------------------------- part 7 ---
def mdi_noise_run(Xtr, Xte, ytr, yte, rng, n_perm_repeats=10):
    """Append a 1000-level and a 2-level pure-noise column; return MDI and permutation."""
    def noise(n):
        return np.column_stack([rng.integers(0, 1000, n), rng.integers(0, 2, n)])
    A, B = np.hstack([Xtr, noise(len(Xtr))]), np.hstack([Xte, noise(len(Xte))])
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(A, ytr)
    pi = permutation_importance(rf, B, yte, n_repeats=n_perm_repeats, random_state=SEED,
                                n_jobs=-1)
    return rf.feature_importances_[-2:], pi.importances_mean[-2:]


def part7(Xtr, Xte, ytr, yte) -> None:
    header("PART 7  MDI vs permutation importance with two pure-noise columns")
    say("procedure: append integers(0,1000) and integers(0,2) columns (default_rng(seed)),\n"
        "300-tree forest random_state=42, permutation importance on the test rows, 10 repeats.")
    say("The post does not give the seed; 20 seeds are shown as a distribution.")
    mdi, perm = [], []
    for s in range(20):
        m, p = mdi_noise_run(Xtr, Xte, ytr, yte, np.random.default_rng(s))
        mdi.append(m)
        perm.append(p)
        say(f"  seed {s:>2}: MDI(1000-level)={m[0]:.5f} MDI(2-level)={m[1]:.5f} "
            f"perm={p[0]:+.4f} / {p[1]:+.4f}")
    mdi, perm = np.array(mdi), np.array(perm)
    say(f"MDI 1000-level: median {np.median(mdi[:, 0]):.5f} range {mdi[:, 0].min():.5f}-"
        f"{mdi[:, 0].max():.5f}")
    say(f"MDI 2-level   : median {np.median(mdi[:, 1]):.5f} range {mdi[:, 1].min():.5f}-"
        f"{mdi[:, 1].max():.5f}")
    say(f"permutation   : range {perm.min():+.4f} .. {perm.max():+.4f}")
    say(f"1000-level MDI > 2-level MDI in {(mdi[:, 0] > mdi[:, 1]).sum()}/20 seeds")
    # A separate search of 48 procedures (RandomState / default_rng, seeds 0/1/42,
    # noise added before / after the split, 100 / 300 trees, column order) found no
    # exact match; the closest was default_rng(1) here (0.00450 / 0.00058).
    # So the published values are compared with the seed-to-seed distribution.
    say(f"published 0.00449 lies at the {np.mean(mdi[:, 0] <= 0.00449):.0%} quantile of the "
        f"20 seeds; 0.00055 at the {np.mean(mdi[:, 1] <= 0.00055):.0%} quantile")
    both_zero = int((np.abs(perm).max(axis=1) < 5e-5).sum())
    say(f"permutation importance is 0.0000 for both columns in {both_zero}/20 seeds")
    check("MDI 1000-level noise col (median of 20 seeds)", 0.00449, float(np.median(mdi[:, 0])),
          tol=float(mdi[:, 0].max() - mdi[:, 0].min()), fmt="{:.5f}")
    check("MDI 2-level noise col (median of 20 seeds)", 0.00055, float(np.median(mdi[:, 1])),
          tol=float(mdi[:, 1].max() - mdi[:, 1].min()), fmt="{:.5f}")
    check("perm. importance 0 for both (share of seeds)", "0.0000 both", f"{both_zero}/20 seeds",
          status="within noise" if both_zero >= 10 else "DIFFERS")


# ----------------------------------------------------------------- part 8 ---
def part8() -> None:
    header(f"PART 8  Defaults, scikit-learn {sklearn.__version__}")
    for cls in (RandomForestClassifier, RandomForestRegressor):
        p = inspect.signature(cls.__init__).parameters
        vals = {k: p[k].default for k in ("n_estimators", "max_features", "bootstrap",
                                           "oob_score", "criterion", "min_samples_leaf")}
        say(f"{cls.__name__}: {vals}")
    c = RandomForestClassifier().get_params()
    r = RandomForestRegressor().get_params()
    ok = (c["n_estimators"] == 100 and c["max_features"] == "sqrt" and c["bootstrap"] is True
          and c["oob_score"] is False and c["criterion"] == "gini" and c["min_samples_leaf"] == 1
          and r["max_features"] == 1.0 and r["criterion"] == "squared_error")
    check("default parameter table", "as published", "as published" if ok else "different")


def main() -> None:
    say(f"python {sys.version.split()[0]} | scikit-learn {sklearn.__version__} | "
        f"numpy {np.__version__}")
    part1()
    part2()
    part3()
    Xtr, Xte, ytr, yte = load_split(SEED)
    say(f"\nseed-42 split: train {len(ytr)} rows, test {len(yte)} rows")
    part4(Xtr, Xte, ytr, yte)
    part5(Xtr, Xte, ytr, yte)
    part6(Xtr, Xte, ytr, yte)
    part7(Xtr, Xte, ytr, yte)
    part8()

    header("SUMMARY  published vs reproduced")
    w = max(len(c[0]) for c in CHECKS)
    for label, pub, val, status in CHECKS:
        say(f"{label:<{w}} | published {pub:>12} | reproduced {val:>12} | {status}")
    (HERE / "results.txt").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
