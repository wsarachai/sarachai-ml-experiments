"""GradientBoostingClassifier vs HistGradientBoostingClassifier on Covertype.

Task: binary classification, Spruce/Fir (cover type 1) vs the rest, the same
binary framing scikit-learn's own covertype benchmark uses.

Experiments
  1. Training time and test accuracy as the number of training rows grows.
  2. Effect of max_bins on accuracy (100k training rows).
  3. early_stopping="auto" (the default) at 100k rows.
  4. Native categorical support: one-hot columns vs two categorical columns.
  5. Native missing values vs mean imputation (20% of numeric cells removed).

All timings are wall-clock seconds on whatever machine runs this, repeated and
reported as the median. Other processes on the machine add noise, so treat the
numbers as ratios on one machine, not as absolute speeds.

Run:  python experiments.py      -> results.txt (human readable), results.json
"""

import json
import os
import platform
import statistics
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import fetch_covtype
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "data_cache")
SEED = 0
N_TEST = 50_000
SIZES = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000]
REPEATS = 3
GBC_REPEATS_LARGEST = 1  # GradientBoostingClassifier at 300k rows takes minutes
MAX_BINS = [2, 4, 8, 16, 32, 64, 128, 255]
N_BINS_EXP = 100_000

lines = []


def log(msg=""):
    print(msg, flush=True)
    lines.append(msg)


def timed_fit(make_model, X, y, repeats, n_threads=None):
    """Fit a fresh model `repeats` times; return (median seconds, all times, last model)."""
    times = []
    model = None
    for _ in range(repeats):
        model = make_model()
        start = time.perf_counter()
        if n_threads is None:
            model.fit(X, y)
        else:
            with threadpool_limits(limits=n_threads, user_api="openmp"):
                model.fit(X, y)
        times.append(time.perf_counter() - start)
    return statistics.median(times), times, model


# ------------------------------------------------------------------ data
data = fetch_covtype(data_home=CACHE)
X_all = data.data.astype(np.float64)
y_all = (data.target == 1).astype(int)

rng = np.random.default_rng(SEED)
perm = rng.permutation(len(y_all))
test_idx = perm[:N_TEST]
pool_idx = perm[N_TEST:]  # training subsets are prefixes of this, so they are nested
X_test, y_test = X_all[test_idx], y_all[test_idx]

log(f"python {platform.python_version()}, scikit-learn {sklearn.__version__}, "
    f"{platform.processor()} ({os.cpu_count()} logical CPUs)")
log(f"Covertype: {X_all.shape[0]} rows x {X_all.shape[1]} features, "
    f"positive rate (Spruce/Fir) {y_all.mean():.4f}")
log(f"Test set: {N_TEST} fixed rows. Repeats per timing: {REPEATS} (median reported).")
log()

results = {"sizes": SIZES, "speed": {}, "max_bins": {}, "env": {
    "python": platform.python_version(), "sklearn": sklearn.__version__,
    "cpus": os.cpu_count()}}

MODELS = {
    # scikit-learn defaults: 100 trees, depth 3, learning rate 0.1, single thread
    "GBC (depth 3)": (lambda: GradientBoostingClassifier(random_state=SEED), None),
    # same tree size and iterations, forced to one thread: algorithm-only comparison
    "HGB depth 3, 1 thread": (lambda: HistGradientBoostingClassifier(
        max_depth=3, max_iter=100, early_stopping=False, random_state=SEED), 1),
    "HGB depth 3, all threads": (lambda: HistGradientBoostingClassifier(
        max_depth=3, max_iter=100, early_stopping=False, random_state=SEED), None),
    # HGB defaults (31 leaves) but early stopping off so it also builds 100 trees
    "HGB 31 leaves, all threads": (lambda: HistGradientBoostingClassifier(
        max_iter=100, early_stopping=False, random_state=SEED), None),
}

# ------------------------------------------------------------------ 1. speed vs rows
log("== 1. Training time (s, median) and test accuracy vs training rows ==")
for name in MODELS:
    results["speed"][name] = {"time": [], "times_all": [], "acc": []}
for n in SIZES:
    Xtr, ytr = X_all[pool_idx[:n]], y_all[pool_idx[:n]]
    for name, (make, threads) in MODELS.items():
        reps = GBC_REPEATS_LARGEST if (name.startswith("GBC") and n == SIZES[-1]) else REPEATS
        med, all_t, model = timed_fit(make, Xtr, ytr, reps, threads)
        acc = model.score(X_test, y_test)
        r = results["speed"][name]
        r["time"].append(med)
        r["times_all"].append(all_t)
        r["acc"].append(acc)
        log(f"  n={n:>7}  {name:<28} time={med:8.3f}s  (runs {', '.join(f'{t:.3f}' for t in all_t)})"
            f"  acc={acc:.4f}")
    log()

log("Speed-up of HGB over GBC (GBC time / HGB time):")
g = results["speed"]["GBC (depth 3)"]["time"]
for name in list(MODELS)[1:]:
    h = results["speed"][name]["time"]
    log(f"  {name:<28} " + "  ".join(f"{n}:{a / b:6.1f}x" for n, a, b in zip(SIZES, g, h)))
log()

# ------------------------------------------------------------------ 2. max_bins
log(f"== 2. max_bins at n={N_BINS_EXP} (HGB 31 leaves, 100 trees, all threads) ==")
Xtr, ytr = X_all[pool_idx[:N_BINS_EXP]], y_all[pool_idx[:N_BINS_EXP]]
results["max_bins"] = {"bins": MAX_BINS, "acc": [], "time": [], "train_acc": []}
for mb in MAX_BINS:
    med, all_t, model = timed_fit(lambda: HistGradientBoostingClassifier(
        max_bins=mb, max_iter=100, early_stopping=False, random_state=SEED),
        Xtr, ytr, REPEATS)
    acc = model.score(X_test, y_test)
    tr_acc = model.score(Xtr, ytr)
    results["max_bins"]["acc"].append(acc)
    results["max_bins"]["train_acc"].append(tr_acc)
    results["max_bins"]["time"].append(med)
    log(f"  max_bins={mb:>3}  test acc={acc:.4f}  train acc={tr_acc:.4f}  time={med:.3f}s")
log()

# ------------------------------------------------------------------ 3. early stopping default
log(f"== 3. Default early_stopping='auto' at n={N_BINS_EXP} ==")
for es, it in (("auto", 100), ("auto", 1000), (False, 1000)):
    med, _, model = timed_fit(lambda: HistGradientBoostingClassifier(
        early_stopping=es, max_iter=it, random_state=SEED), Xtr, ytr, 1)
    log(f"  early_stopping={es!s:<5} max_iter={it:<5} -> trees built={model.n_iter_:<5} "
        f"test acc={model.score(X_test, y_test):.4f}  time={med:.2f}s")
log()

# ------------------------------------------------------------------ 4. categorical
log(f"== 4. One-hot columns vs native categorical (n={N_BINS_EXP}) ==")
cols = [f"num{i}" for i in range(10)]
df_all = pd.DataFrame(X_all[:, :10], columns=cols)
df_all["wilderness"] = pd.Categorical(X_all[:, 10:14].argmax(axis=1))
df_all["soil"] = pd.Categorical(X_all[:, 14:54].argmax(axis=1))
df_tr = df_all.iloc[pool_idx[:N_BINS_EXP]]
df_te = df_all.iloc[test_idx]
variants = {
    "54 columns (44 one-hot)": (X_all[pool_idx[:N_BINS_EXP]], X_test, None),
    "12 columns, 2 categorical": (df_tr, df_te, "from_dtype"),
    "12 columns, codes as numbers": (df_tr.astype(float), df_te.astype(float), None),
}
for name, (A, B, cat) in variants.items():
    med, _, model = timed_fit(lambda: HistGradientBoostingClassifier(
        max_iter=100, early_stopping=False, categorical_features=cat, random_state=SEED),
        A, ytr, REPEATS)
    log(f"  {name:<30} test acc={model.score(B, y_test):.4f}  time={med:.3f}s")
log()

# ------------------------------------------------------------------ 5. missing values
log(f"== 5. 20% of the 10 numeric cells set to NaN (train and test), n={N_BINS_EXP} ==")
mrng = np.random.default_rng(SEED + 1)
Xtr_m = Xtr.copy()
Xte_m = X_test.copy()
Xtr_m[:, :10][mrng.random((len(Xtr_m), 10)) < 0.2] = np.nan
Xte_m[:, :10][mrng.random((len(Xte_m), 10)) < 0.2] = np.nan
for name, make in {
    "no NaN (reference)": None,
    "native NaN handling": lambda: HistGradientBoostingClassifier(
        max_iter=100, early_stopping=False, random_state=SEED),
    "mean imputation + HGB": lambda: make_pipeline(
        SimpleImputer(strategy="mean"),
        HistGradientBoostingClassifier(max_iter=100, early_stopping=False, random_state=SEED)),
}.items():
    if make is None:
        log(f"  {name:<24} test acc={results['max_bins']['acc'][-1]:.4f}")
        continue
    _, _, model = timed_fit(make, Xtr_m, ytr, 1)
    log(f"  {name:<24} test acc={model.score(Xte_m, y_test):.4f}")
log()

with open(os.path.join(HERE, "results.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
log("wrote results.txt and results.json")
