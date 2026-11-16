"""Reproduction of the measured numbers in the "machine-learning" hub post.

Parts
  1. Bias-variance decomposition: f(x) = sin(2 pi x), x ~ U[0, 1], noise
     sigma = 0.2, polynomial degrees 1..12 fitted to 30 rows, 400 repeats,
     evaluated on a 200-point grid. The post's seed and fitting routine were not
     recorded; part1() states the recovered procedure and how it was found.
  2. The two leakage numbers the post quotes from the scikit-learn "Common
     pitfalls" page, re-run with the code from that page:
     feature selection before the split (0.76 vs ~0.5) and forgetting to scale
     the test rows (MSE 62.80 vs 0.90).

Run from the repository root:
    .venv/Scripts/python.exe machine-learning/experiments.py
Output is printed and written to machine-learning/results.txt.
"""

from __future__ import annotations

import io
import sys
import warnings
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
buf = io.StringIO()
CHECKS: list[tuple[str, str, str, str]] = []

SIGMA = 0.2
N_TRAIN, N_REPEATS, N_GRID = 30, 400, 200
DEGREES = [1, 3, 5, 7, 9, 12]
PUBLISHED = {  # degree: (bias^2, variance, sum, measured test MSE)
    1: (0.1997, 0.0171, 0.2569, 0.2565),
    3: (0.0049, 0.0089, 0.0538, 0.0538),
    5: (0.0001, 0.0273, 0.0674, 0.0676),
    7: (0.0050, 0.5377, 0.5827, 0.5844),
    9: (0.0135, 3.9808, 4.0343, 4.0353),
    12: (45.0641, 12184.9381, 12230.0421, 12230.2250),
}


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


def f(x):
    return np.sin(2 * np.pi * x)


def bias_variance(rng, degrees=range(1, 13)):
    """The experiment, run for each degree in turn on ONE random stream.

    For every degree: draw 400 training sets (30 x-values, then 30 noise values
    each), then 400 noisy target vectors on the 200-point grid, fit np.polyfit.
    Because the stream is shared, degree d sees different data from degree d-1;
    degrees 2, 4, 6, 8, 10, 11 are run (and discarded) only to keep the stream
    identical to the one behind the post.
    """
    grid = np.linspace(0, 1, N_GRID)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # high-degree polyfit is poorly conditioned
        for d in degrees:
            data = []
            for _ in range(N_REPEATS):
                x = rng.uniform(0, 1, N_TRAIN)
                data.append((x, f(x) + rng.normal(0, SIGMA, N_TRAIN)))
            Yt = np.array([f(grid) + rng.normal(0, SIGMA, N_GRID) for _ in range(N_REPEATS)])
            P = np.array([np.polyval(np.polyfit(x, y, d), grid) for x, y in data])
            bias2 = np.mean((P.mean(axis=0) - f(grid)) ** 2)
            var = np.mean(P.var(axis=0))
            out[d] = (bias2, var, bias2 + var + SIGMA ** 2, np.mean((Yt - P) ** 2))
    return out


def part1() -> None:
    header("PART 1  Bias-variance decomposition (np.polyfit, RandomState(0), one stream)")
    say("Recovered procedure: np.random.RandomState(0); degrees 1..12 run in order on the\n"
        "same stream (see bias_variance()). Found after a scratch search of 180 + 600 + 4\n"
        "variants (RNG type, seeds, fitter, order of the test noise, one stream vs. a fresh\n"
        "stream per degree, per-degree seeds 0..299); only this one matches all 24 cells.")
    res = bias_variance(np.random.RandomState(0))
    names = ("bias^2", "variance", "sum", "test MSE")
    say("degree | bias^2 | variance | noise | sum | measured test MSE")
    for d in DEGREES:
        b, v, s, m = res[d]
        say(f"{d:>6} | {b:.4f} | {v:.4f} | 0.04 | {s:.4f} | {m:.4f}")
        for i, n in enumerate(names):
            # "sum" is printed from 4-decimal parts in the post, so allow rounding there
            check(f"degree {d} {n}", PUBLISHED[d][i], res[d][i],
                  tol=1.5e-4 if n == "sum" else 0.0)
    best = min(DEGREES, key=lambda d: res[d][3])
    check("lowest test MSE at degree", "3", str(best))

    say("\nContext - the same procedure with seeds 1..10 (min .. max):")
    runs = [bias_variance(np.random.RandomState(s)) for s in range(1, 11)]
    for d in DEGREES:
        arr = np.array([r[d] for r in runs])
        say(f"degree {d:>2}: " + "; ".join(
            f"{n} {arr[:, i].min():.4f}..{arr[:, i].max():.4f}" for i, n in enumerate(names)))
    bests = [min(DEGREES, key=lambda d: r[d][3]) for r in runs]
    say(f"degree with the lowest test MSE per seed: {bests}")
    say("Degrees >= 7 swing by orders of magnitude between seeds (a few near-singular\n"
        "fits dominate the mean), so only their order of magnitude is meaningful.")


def part2() -> None:
    header("PART 2  Leakage examples from the scikit-learn 'Common pitfalls' page")
    # Inconsistent preprocessing
    random_state = 42
    X, y = make_regression(random_state=random_state, n_features=1, noise=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                        random_state=random_state)
    scaler = StandardScaler()
    model = LinearRegression().fit(scaler.fit_transform(X_train), y_train)
    wrong = mean_squared_error(y_test, model.predict(X_test))
    right = mean_squared_error(y_test, model.predict(scaler.transform(X_test)))
    say(f"test rows not scaled: MSE {wrong:.2f}; scaled with the train scaler: MSE {right:.2f}")
    # The exact value is 62.8087; the post (quoting the docs page) writes 62.80.
    say(f"unrounded MSE without scaling: {wrong:.4f}")
    check("MSE when the test rows are not scaled", 62.80, wrong, fmt="{:.2f}", tol=0.01)
    check("MSE with consistent scaling", 0.90, right, fmt="{:.2f}")

    # Feature selection on all rows before the split
    n_samples, n_features, n_classes = 200, 10000, 2
    rng = np.random.RandomState(42)
    X = rng.standard_normal((n_samples, n_features))
    y = rng.choice(n_classes, n_samples)
    X_sel = SelectKBest(k=25).fit_transform(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_sel, y, random_state=42)
    gbc = GradientBoostingClassifier(random_state=1).fit(X_train, y_train)
    leaky = accuracy_score(y_test, gbc.predict(X_test))
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    pipe = make_pipeline(SelectKBest(k=25), GradientBoostingClassifier(random_state=1))
    pipe.fit(X_train, y_train)
    clean = accuracy_score(y_test, pipe.predict(X_test))
    say(f"selection before split: accuracy {leaky:.2f}; selection inside the train split: "
        f"{clean:.2f}")
    check("accuracy with selection before the split", 0.76, leaky, fmt="{:.2f}")
    check("accuracy with selection on train only (post: 0.5)", 0.5, clean, fmt="{:.2f}",
          tol=0.05)


def main() -> None:
    say(f"python {sys.version.split()[0]} | scikit-learn {sklearn.__version__} | "
        f"numpy {np.__version__}")
    part1()
    part2()
    header("SUMMARY  published vs reproduced")
    w = max(len(c[0]) for c in CHECKS)
    for label, pub, val, status in CHECKS:
        say(f"{label:<{w}} | published {pub:>12} | reproduced {val:>12} | {status}")
    (HERE / "results.txt").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
