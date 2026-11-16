"""Reproduction of the measured numbers in the "out-of-bag" blog post.

The post was written before this repository existed, so its seeds and
procedures are not recorded. This script states the inferred procedure for
each number, recomputes it and compares it with the published value.

Parts
  1. Bootstrap of n = 398 rows, 500 draws: share of unique rows (same draws as
     the random-forest post).
  2. Hand example: 3 trees, 6 rows -> which rows have an OOB value (asserted).
  3. Share of rows with at least one OOB tree for B = 1..30 trees (n = 398,
     300 simulated forests) against the formula 1 - 0.632^B.
  4. Breast cancer, 10 seeds: OOB vs 5-fold CV vs test accuracy for 10, 30,
     100 and 300 trees.

Run from the repository root:
    .venv/Scripts/python.exe out-of-bag/experiments.py
Output is printed and written to out-of-bag/results.txt.
"""

from __future__ import annotations

import io
import sys
import warnings
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split

HERE = Path(__file__).resolve().parent
buf = io.StringIO()
CHECKS: list[tuple[str, str, str, str]] = []


def say(*args) -> None:
    print(*args)
    print(*args, file=buf)


def header(title: str) -> None:
    say("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def check(label, published, value, tol=0.0, fmt="{:.4f}") -> None:
    """exact = equal at published precision; within noise = |diff| <= tol."""
    if isinstance(published, str) or isinstance(value, str):
        pub_s, val_s = str(published), str(value)
        status = "exact" if pub_s == val_s else "DIFFERS"
    else:
        pub_s, val_s = fmt.format(published), fmt.format(value)
        status = ("exact" if pub_s == val_s
                  else "within noise" if abs(published - value) <= tol else "DIFFERS")
    CHECKS.append((label, pub_s, val_s, status))


def part1() -> None:
    header("PART 1  Unique-row share of a bootstrap, n = 398, 500 draws")
    n = 398
    rs = np.random.RandomState(0)  # same inferred procedure as random-forest/experiments.py
    s = np.array([len(np.unique(rs.randint(0, n, n))) / n for _ in range(500)])
    say(f"RandomState(0): mean unique share = {s.mean():.4f}; OOB share = {1 - s.mean():.4f}")
    say(f"1 - 1/e = {1 - np.exp(-1):.4f}")
    check("mean unique share, 500 draws", 0.6326, s.mean(), tol=0.0015)
    check("1 - 1/e", 0.6321, 1 - np.exp(-1))
    check("remaining OOB share", 0.3674, 1 - s.mean(), tol=0.0015)


def part2() -> None:
    header("PART 2  Hand example: 3 trees, 6 rows (asserted)")
    boot = {"T1": [1, 2, 2, 3, 4, 6], "T2": [1, 3, 3, 4, 5, 6], "T3": [2, 3, 4, 5, 5, 6]}
    oob = {t: sorted(set(range(1, 7)) - set(r)) for t, r in boot.items()}
    say("OOB rows per tree:", oob)
    assert oob == {"T1": [5], "T2": [2], "T3": [1]}
    no_oob = [i for i in range(1, 7) if all(i in r for r in boot.values())]
    assert no_oob == [3, 4, 6]
    # Leaf probabilities of the stumps grown in the random-forest post.
    p = {"T3": {1: 1 / 3}, "T2": {2: 0.0}, "T1": {5: 2 / 3}}
    assert abs(p["T3"][1] - 0.3333) < 5e-5 and abs(p["T1"][5] - 0.6667) < 5e-5
    check("hand example (OOB rows, values)", "asserted", "asserted")
    say("rows without any OOB tree:", no_oob, "- asserted OK")


def part3() -> None:
    header("PART 3  Share of rows with an OOB value vs number of trees (n=398, 300 forests)")
    n, reps = 398, 300
    published = {1: 36.6, 2: 60.0, 3: 74.7, 5: 89.8, 10: 99.0, 30: 100.0}
    rng = np.random.RandomState(0)
    p_in = 1 - (1 - 1 / n) ** n
    shares = {b: [] for b in published}
    for _ in range(reps):
        seen = np.zeros((30, n), dtype=bool)  # seen[t, i]: row i in-bag for tree t
        for t in range(30):
            seen[t, rng.randint(0, n, n)] = True
        for b in published:
            shares[b].append(100 * np.mean(~seen[:b].all(axis=0)))
    say("trees | simulated % | formula 1-0.632^B | exact 1-p_in^B | sd over forests")
    for b, pub in published.items():
        sim = float(np.mean(shares[b]))
        say(f"{b:>5} | {sim:10.1f} | {100 * (1 - 0.632 ** b):10.1f} | "
            f"{100 * (1 - p_in ** b):10.1f} | {np.std(shares[b]):.2f}")
        # noise of a 300-forest average = sd / sqrt(300)
        check(f"rows with OOB value, {b} trees (%)", pub, sim,
              tol=max(3 * np.std(shares[b]) / np.sqrt(reps), 0.05), fmt="{:.1f}")


def part4() -> None:
    header("PART 4  OOB vs 5-fold CV vs test, 10 seeds (split seed = forest seed = 0..9)")
    say("procedure: stratified 70/30 split (random_state=s), RandomForestClassifier("
        "n_estimators=B, random_state=s, oob_score=True);\nCV = cross_val_score(same forest, "
        "train rows, cv=5) (stratified, unshuffled); SD = population SD over the 10 seeds")
    data = load_breast_cancer()
    published = {  # B: (oob, oob_sd, cv, cv_sd, test, test_sd, oob-test, cv-test)
        10: (0.9319, 0.0094, 0.9465, 0.0082, 0.9538, 0.0151, -0.022, -0.007),
        30: (0.9540, 0.0095, 0.9543, 0.0060, 0.9585, 0.0135, -0.005, -0.004),
        100: (0.9593, 0.0062, 0.9570, 0.0041, 0.9579, 0.0082, +0.001, -0.001),
        300: (0.9590, 0.0050, 0.9558, 0.0030, 0.9579, 0.0090, +0.001, -0.002),
    }
    for b, pub in published.items():
        oob, cv, te = [], [], []
        for s in range(10):
            Xtr, Xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.3,
                                                  random_state=s, stratify=data.target)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rf = RandomForestClassifier(n_estimators=b, random_state=s, oob_score=True,
                                            n_jobs=-1).fit(Xtr, ytr)
            oob.append(rf.oob_score_)
            te.append(rf.score(Xte, yte))
            cv.append(cross_val_score(RandomForestClassifier(n_estimators=b, random_state=s,
                                                             n_jobs=-1), Xtr, ytr, cv=5).mean())
        oob, cv, te = map(np.array, (oob, cv, te))
        say(f"{b:>3} trees: OOB {oob.mean():.4f} +/- {oob.std():.4f} | CV {cv.mean():.4f} +/- "
            f"{cv.std():.4f} | test {te.mean():.4f} +/- {te.std():.4f} | OOB-test "
            f"{oob.mean() - te.mean():+.3f} | CV-test {cv.mean() - te.mean():+.3f}")
        vals = (oob.mean(), oob.std(), cv.mean(), cv.std(), te.mean(), te.std(),
                oob.mean() - te.mean(), cv.mean() - te.mean())
        names = ("OOB mean", "OOB sd", "CV mean", "CV sd", "test mean", "test sd",
                 "OOB-test", "CV-test")
        for name, pv, v in zip(names, pub, vals):
            fmt = "{:+.3f}" if "-" in name else "{:.4f}"
            # Differences: the post subtracted the 4-decimal means, so allow rounding.
            check(f"{b} trees {name}", pv, v, fmt=fmt, tol=0.0006 if "-" in name else 0.0)
    # "at 1 tree OOB is 33 points below" refers to the random-forest post's seed-42 table
    say("\nThe post's '1 tree: OOB 33 points below test' uses the random-forest seed-42 "
        "numbers: 0.9123 - 0.5829 = 0.3294 (see random-forest/results.txt).")
    check("1 tree: test - OOB (points)", 33, round(100 * (0.9123 - 0.5829)), fmt="{:.0f}")


def main() -> None:
    say(f"python {sys.version.split()[0]} | scikit-learn {sklearn.__version__} | "
        f"numpy {np.__version__}")
    part1()
    part2()
    part3()
    part4()
    header("SUMMARY  published vs reproduced")
    w = max(len(c[0]) for c in CHECKS)
    for label, pub, val, status in CHECKS:
        say(f"{label:<{w}} | published {pub:>9} | reproduced {val:>9} | {status}")
    (HERE / "results.txt").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
