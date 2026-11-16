"""Reproduction of the measured numbers in the "gradient-boosting" blog post.

Parts
  1. Hand calculation: 4 rows, squared loss, nu = 0.5, two stumps (asserted,
     and checked against GradientBoostingRegressor).
  2. Breast cancer, clean labels: test error vs number of trees for
     nu = 0.01 / 0.1 / 0.5 (best over the checkpoints 1, 10, 30, 100, 300, 500,
     1000 - the checkpoint set that reproduces every "best at N trees").
  3. Depth sweep, 300 trees, nu = 0.1.
  4. 15% label noise. The published table could not be reproduced; this part
     reports the distribution over 50 noise draws (see the comment there and
     gradient-boosting/noise_search.py for the 972-procedure search).
  5. The scikit-learn documentation example quoted in the post (stumps,
     learning_rate=1, 100 trees on make_hastie_10_2 -> 0.913).
  6. GradientBoostingClassifier defaults.

Run from the repository root:
    .venv/Scripts/python.exe gradient-boosting/experiments.py
Output is printed and written to gradient-boosting/results.txt.
"""

from __future__ import annotations

import inspect
import io
import sys
from pathlib import Path

import numpy as np
import sklearn
from joblib import Parallel, delayed
from sklearn.datasets import load_breast_cancer, make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
SEED = 42
CHECKPOINTS = [1, 10, 30, 100, 300, 500, 1000]
NOISE_RATE = 0.15
buf = io.StringIO()
CHECKS: list[tuple[str, str, str, str]] = []


def say(*args) -> None:
    print(*args)
    print(*args, file=buf)


def header(title: str) -> None:
    say("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def check(label, published, value, tol=0.0, fmt="{:.4f}", status=None) -> None:
    if isinstance(published, str) or isinstance(value, str):
        pub_s, val_s = str(published), str(value)
        status = status or ("exact" if pub_s == val_s else "DIFFERS")
    else:
        pub_s, val_s = fmt.format(published), fmt.format(value)
        status = status or ("exact" if pub_s == val_s
                            else "within noise" if abs(published - value) <= tol else "DIFFERS")
    CHECKS.append((label, pub_s, val_s, status))


def load_split():
    d = load_breast_cancer()
    return train_test_split(d.data, d.target, test_size=0.3, random_state=SEED,
                            stratify=d.target)


def staged_test_error(lr, Xtr, ytr, Xte, yte, n=1000, depth=3):
    m = GradientBoostingClassifier(n_estimators=n, learning_rate=lr, max_depth=depth,
                                   random_state=SEED).fit(Xtr, ytr)
    return np.array([np.mean(p != yte) for p in m.staged_predict(Xte)])


def best_at_checkpoints(err):
    vals = [err[c - 1] for c in CHECKPOINTS]
    k = int(np.argmin(vals))  # earliest checkpoint wins ties
    return vals[k], CHECKPOINTS[k]


# ----------------------------------------------------------------- part 1 ---
def part1() -> None:
    header("PART 1  Hand calculation: 4 rows, squared loss, nu = 0.5 (asserted)")
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([10.0, 12.0, 25.0, 27.0])
    nu = 0.5
    F = np.full(4, y.mean())
    assert F[0] == 18.5
    history = []
    for _ in range(2):
        r = y - F
        # best stump by SSE over the three midpoints
        def sse(t):
            left, right = r[x <= t], r[x > t]
            return ((left - left.mean()) ** 2).sum() + ((right - right.mean()) ** 2).sum()

        best = min((1.5, 2.5, 3.5), key=sse)
        gl, gr = r[x <= best].mean(), r[x > best].mean()
        F = F + nu * np.where(x <= best, gl, gr)
        history.append((r.tolist(), best, gl, gr, F.tolist()))
        say(f"residuals {r.tolist()} -> split x <= {best}, gamma {gl:+.3f}/{gr:+.3f}, F = {F.tolist()}")
    assert history[0][0] == [-8.5, -6.5, 6.5, 8.5] and history[0][1] == 2.5
    assert (history[0][2], history[0][3]) == (-7.5, 7.5)
    assert history[0][4] == [14.75, 14.75, 22.25, 22.25]
    assert history[1][0] == [-4.75, -2.75, 2.75, 4.75] and history[1][1] == 2.5
    assert (history[1][2], history[1][3]) == (-3.75, 3.75)
    assert history[1][4] == [12.875, 12.875, 24.125, 24.125]
    gbr = GradientBoostingRegressor(n_estimators=2, learning_rate=nu, max_depth=1,
                                    random_state=SEED).fit(x.reshape(-1, 1), y)
    lib = gbr.predict(x.reshape(-1, 1))
    say("GradientBoostingRegressor(n_estimators=2, learning_rate=0.5, max_depth=1):", lib.tolist())
    assert np.allclose(lib, [12.875, 12.875, 24.125, 24.125])
    check("hand calculation (F0, residuals, gammas, F1, F2)", "asserted", "asserted")


# ----------------------------------------------------------------- part 2 ---
def part2(Xtr, Xte, ytr, yte) -> None:
    header("PART 2  Clean labels: test error vs trees (max_depth=3, random_state=42)")
    published = {0.01: (0.0526, 1000), 0.1: (0.0468, 500), 0.5: (0.0468, 30)}
    for lr, (pub_err, pub_n) in published.items():
        e = staged_test_error(lr, Xtr, ytr, Xte, yte)
        best, at = best_at_checkpoints(e)
        say(f"nu={lr}: " + " ".join(f"{c}:{e[c - 1]:.4f}" for c in CHECKPOINTS)
            + f" | best {best:.4f} at {at} (unrestricted min {e.min():.4f} at {e.argmin() + 1})")
        check(f"clean nu={lr} best error", pub_err, best)
        check(f"clean nu={lr} best at N trees", str(pub_n), str(at))


# ----------------------------------------------------------------- part 3 ---
def part3(Xtr, Xte, ytr, yte) -> None:
    header("PART 3  Depth sweep, 300 trees, nu = 0.1")
    for d, pub in zip((1, 2, 3, 5), (0.0468, 0.0468, 0.0526, 0.0468)):
        m = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, max_depth=d,
                                       random_state=SEED).fit(Xtr, ytr)
        te, tr = 1 - m.score(Xte, yte), 1 - m.score(Xtr, ytr)
        say(f"depth {d}: test error {te:.4f} train error {tr:.4f}")
        check(f"depth {d} test error", pub, te)
        check(f"depth {d} train error", 0.0, tr)


# ----------------------------------------------------------------- part 4 ---
def flip_train_labels(y, seed, legacy=False):
    """Flip 15% of the training labels, chosen without replacement.

    legacy=True : np.random.RandomState(seed).choice(n, int(0.15 n)) - the procedure
                  that reproduces the post (int() truncates: 59 of 398 rows)
    legacy=False: np.random.default_rng(seed).choice(n, round(0.15 n)) - 60 rows,
                  used for the draw-to-draw distribution
    """
    if legacy:
        idx = np.random.RandomState(seed).choice(len(y), int(NOISE_RATE * len(y)), replace=False)
    else:
        idx = np.random.default_rng(seed).choice(len(y), int(round(NOISE_RATE * len(y))),
                                                 replace=False)
    out = y.copy()
    out[idx] = 1 - out[idx]
    return out


def noisy_run(seed, Xtr, ytr, Xte, yte):
    yn = flip_train_labels(ytr, seed)
    return {lr: staged_test_error(lr, Xtr, yn, Xte, yte) for lr in (0.01, 0.1, 0.5)}


def part4(Xtr, Xte, ytr, yte) -> None:
    header("PART 4  15% flipped training labels (clean test labels)")
    say("The flip procedure is not stated in the post. noise_search.py tried 972\n"
        "procedures (RandomState / default_rng / python random; seeds 0..20, 42, 123,\n"
        "1234, 2024-2026; choice with round/floor k, Bernoulli mask, permutation;\n"
        "flip train only / all rows before the split, stratified on clean or noisy y;\n"
        "scored on clean and on noisy test labels). Exactly one matches all six\n"
        "signature values (next best: 3 of 6): RandomState(7).choice(398, 59,\n"
        "replace=False), training labels only, clean test labels. Seed 7 was found by\n"
        "search, so treat the match as a recovered procedure, not an independent check.")
    published = {0.01: (0.0760, 500, 0.0877), 0.1: (0.0994, 10, 0.1345), 0.5: (0.0819, 1, 0.1404)}
    yn = flip_train_labels(ytr, 7, legacy=True)
    say(f"\nrecovered procedure: {int((yn != ytr).sum())} labels flipped")
    for lr, (pub_best, pub_at, pub_1000) in published.items():
        e = staged_test_error(lr, Xtr, yn, Xte, yte)
        best, at = best_at_checkpoints(e)
        say(f"nu={lr}: " + " ".join(f"{c}:{e[c - 1]:.4f}" for c in CHECKPOINTS)
            + f" | best {best:.4f} at {at}")
        check(f"noisy nu={lr} best error", pub_best, best)
        check(f"noisy nu={lr} best at N trees", str(pub_at), str(at))
        check(f"noisy nu={lr} error at 1000 trees", pub_1000, e[999])

    say("\nHow typical is that draw? 50 draws of default_rng(seed).choice(398, 60),\n"
        "seeds 0..49, training labels only:")
    runs = Parallel(n_jobs=-1)(delayed(noisy_run)(s, Xtr, ytr, Xte, yte) for s in range(50))
    for lr, (pub_best, pub_at, pub_1000) in published.items():
        errs = np.array([r[lr] for r in runs])
        best = np.array([best_at_checkpoints(e)[0] for e in errs])
        at = np.array([best_at_checkpoints(e)[1] for e in errs])
        e1000 = errs[:, 999]
        mean_curve = errs.mean(axis=0)
        mbest, mat = best_at_checkpoints(mean_curve)
        say(f"\nnu={lr}: mean curve at checkpoints: "
            + " ".join(f"{c}:{mean_curve[c - 1]:.4f}" for c in CHECKPOINTS))
        say(f"  best over checkpoints per draw: median {np.median(best):.4f}, "
            f"5-95% {np.percentile(best, 5):.4f}-{np.percentile(best, 95):.4f}; "
            f"argmin checkpoint counts { {int(k): int(v) for k, v in zip(*np.unique(at, return_counts=True))} }")
        say(f"  error at 1000 trees per draw: median {np.median(e1000):.4f}, mean "
            f"{e1000.mean():.4f}, 5-95% {np.percentile(e1000, 5):.4f}-"
            f"{np.percentile(e1000, 95):.4f}")
        say(f"  mean curve: best {mbest:.4f} at {mat}, at 1000 {mean_curve[999]:.4f}")
        say(f"  published best {pub_best} lies at quantile {np.mean(best <= pub_best):.2f}; "
            f"published @1000 {pub_1000} at quantile {np.mean(e1000 <= pub_1000):.2f}")
        rng_b = (np.percentile(best, 5), np.percentile(best, 95))
        rng_e = (np.percentile(e1000, 5), np.percentile(e1000, 95))
        check(f"context: noisy nu={lr} best, median of 50 draws", pub_best,
              float(np.median(best)),
              status="inside 5-95%" if rng_b[0] <= pub_best <= rng_b[1] else "outside 5-95%")
        check(f"context: noisy nu={lr} @1000, median of 50 draws", pub_1000,
              float(np.median(e1000)),
              status="inside 5-95%" if rng_e[0] <= pub_1000 <= rng_e[1] else "outside 5-95%")


# ----------------------------------------------------------------- part 5 ---
def part5() -> None:
    header("PART 5  scikit-learn docs example: stumps, lr=1, 100 trees, make_hastie_10_2")
    X, y = make_hastie_10_2(random_state=0)
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1,
                                     random_state=0).fit(X[:2000], y[:2000])
    acc = clf.score(X[2000:], y[2000:])
    say(f"test accuracy = {acc:.4f}")
    check("hastie stump example accuracy", 0.913, acc, fmt="{:.3f}")


# ----------------------------------------------------------------- part 6 ---
def part6() -> None:
    header(f"PART 6  GradientBoostingClassifier defaults, scikit-learn {sklearn.__version__}")
    p = inspect.signature(GradientBoostingClassifier.__init__).parameters
    vals = {k: p[k].default for k in ("learning_rate", "n_estimators", "max_depth",
                                       "subsample", "loss")}
    say(vals)
    check("defaults lr/n/depth/subsample/loss", "0.1/100/3/1.0/log_loss",
          "/".join(str(v) for v in vals.values()))


def main() -> None:
    say(f"python {sys.version.split()[0]} | scikit-learn {sklearn.__version__} | "
        f"numpy {np.__version__}")
    part1()
    Xtr, Xte, ytr, yte = load_split()
    part2(Xtr, Xte, ytr, yte)
    part3(Xtr, Xte, ytr, yte)
    part4(Xtr, Xte, ytr, yte)
    part5()
    part6()
    header("SUMMARY  published vs reproduced")
    w = max(len(c[0]) for c in CHECKS)
    for label, pub, val, status in CHECKS:
        say(f"{label:<{w}} | published {pub:>22} | reproduced {val:>22} | {status}")
    (HERE / "results.txt").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
