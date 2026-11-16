"""Measured results for the "lightgbm" blog post.

Part A  Defaults: the parameter values the installed LightGBM actually resolves
        (read back from a trained model), plus the scikit-learn wrapper names.
Part B  Small data (breast cancer, 569 x 30, stratified 398/171 split as in the
        sibling posts, 10 split seeds 42..51): num_leaves sweep with the default
        min_data_in_leaf=20 and with min_data_in_leaf=1, and LightGBM vs XGBoost
        with their default settings on the same splits.
Part C  Larger data (Covertype, 581,012 x 54, binary task "Lodgepole Pine vs
        rest"): training time and test accuracy for LightGBM, XGBoost `hist`
        (depth-wise and loss-guided), LightGBM with GOSS and with EFB disabled,
        at several training sizes. Timings are medians of repeated runs and are
        machine-dependent.

Run from the repository root (so that `import lightgbm` resolves to the
installed package, not to a folder):
    .venv/Scripts/python.exe lightgbm-post/experiments.py [A] [B] [C]
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.datasets import fetch_covtype, load_breast_cancer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data_cache"
SEEDS = list(range(42, 52))
THREADS = 8  # fixed for every library so the timing comparison is like-for-like


def header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78, flush=True)


# ---------------------------------------------------------------------------
# Part A: defaults
# ---------------------------------------------------------------------------

def part_a() -> None:
    header(f"PART A  Defaults resolved by lightgbm {lgb.__version__}")
    rng = np.random.RandomState(0)
    X = rng.rand(200, 3)
    y = (X[:, 0] > 0.5).astype(int)
    booster = lgb.train({"objective": "binary", "verbose": -1}, lgb.Dataset(X, y), num_boost_round=1)
    text = booster.model_to_string()
    params = {}
    for line in text[text.find("parameters:"):].splitlines():
        if line.startswith("[") and ": " in line:
            k, v = line[1:-1].split(": ", 1)
            params[k] = v
    keys = ["boosting", "data_sample_strategy", "num_iterations", "learning_rate", "num_leaves",
            "max_depth", "min_data_in_leaf", "min_sum_hessian_in_leaf", "min_gain_to_split",
            "lambda_l1", "lambda_l2", "feature_fraction", "bagging_fraction", "bagging_freq",
            "top_rate", "other_rate", "max_bin", "min_data_in_bin", "enable_bundle",
            "feature_pre_filter", "max_cat_to_onehot", "max_cat_threshold", "cat_smooth",
            "cat_l2", "min_data_per_group", "boost_from_average"]
    for k in keys:
        print(f"  {k:26s} = {params.get(k)}")
    print("\n  scikit-learn wrapper LGBMClassifier() defaults:")
    for k, v in sorted(lgb.LGBMClassifier().get_params().items()):
        print(f"  {k:26s} = {v}")


# ---------------------------------------------------------------------------
# Part B: small data
# ---------------------------------------------------------------------------

def split(seed: int):
    d = load_breast_cancer()
    return train_test_split(d.data, d.target, test_size=0.3, random_state=seed, stratify=d.target)


def fit_lgb(Xtr, ytr, seed, **kw):
    m = lgb.LGBMClassifier(random_state=seed, n_jobs=THREADS, verbose=-1, **kw)
    return m.fit(Xtr, ytr)


def part_b() -> None:
    header("PART B  Breast cancer, stratified 398/171 split, seeds 42..51 (mean over 10 splits)")
    Xtr, Xte, ytr, yte = split(42)
    print(f"  train rows = {len(ytr)}, test rows = {len(yte)}")

    print("\n  B1  num_leaves sweep, 100 trees, learning_rate 0.1, everything else default")
    print("      train acc / test acc / test log loss / mean leaves actually grown per tree")
    for mdl in (20, 1):
        print(f"\n      min_data_in_leaf = {mdl}")
        for nl in (2, 4, 8, 16, 31, 63, 127):
            tr, te, ll, leaves = [], [], [], []
            for s in SEEDS:
                Xtr, Xte, ytr, yte = split(s)
                m = fit_lgb(Xtr, ytr, s, num_leaves=nl, min_child_samples=mdl)
                tr.append(accuracy_score(ytr, m.predict(Xtr)))
                te.append(accuracy_score(yte, m.predict(Xte)))
                ll.append(log_loss(yte, m.predict_proba(Xte)[:, 1]))
                info = m.booster_.dump_model()["tree_info"]
                leaves.append(np.mean([t["num_leaves"] for t in info]))
            print(f"      num_leaves={nl:4d}  train {np.mean(tr):.4f}  test {np.mean(te):.4f} "
                  f"(sd {np.std(te):.4f})  logloss {np.mean(ll):.4f}  leaves/tree {np.mean(leaves):6.2f}")

    print("\n  B2  Library defaults on the same 10 splits (100 trees each, n_jobs fixed)")
    rows = {}
    for name in ("LightGBM default", "LightGBM min_data_in_leaf=1",
                 "XGBoost default (hist, depth 6)", "XGBoost lossguide max_leaves=31"):
        te, ll, leaves = [], [], []
        for s in SEEDS:
            Xtr, Xte, ytr, yte = split(s)
            if name == "LightGBM default":
                m = fit_lgb(Xtr, ytr, s)
            elif name == "LightGBM min_data_in_leaf=1":
                m = fit_lgb(Xtr, ytr, s, min_child_samples=1)
            elif name.startswith("XGBoost default"):
                m = xgb.XGBClassifier(n_estimators=100, random_state=s, n_jobs=THREADS).fit(Xtr, ytr)
            else:
                m = xgb.XGBClassifier(n_estimators=100, random_state=s, n_jobs=THREADS,
                                      grow_policy="lossguide", max_leaves=31, max_depth=0).fit(Xtr, ytr)
            te.append(accuracy_score(yte, m.predict(Xte)))
            ll.append(log_loss(yte, m.predict_proba(Xte)[:, 1]))
            if name.startswith("XGBoost"):
                dump = m.get_booster().get_dump()
                leaves.append(np.mean([t.count("leaf=") for t in dump]))
            else:
                leaves.append(np.mean([t["num_leaves"] for t in m.booster_.dump_model()["tree_info"]]))
        rows[name] = (te, ll)
        print(f"      {name:34s} test acc {np.mean(te):.4f} (sd {np.std(te):.4f})  "
              f"min {np.min(te):.4f}  logloss {np.mean(ll):.4f}  leaves/tree {np.mean(leaves):.2f}")
    a = np.array(rows["LightGBM default"][0])
    b = np.array(rows["XGBoost default (hist, depth 6)"][0])
    print(f"      per-split accuracy difference LightGBM - XGBoost: {np.round(a - b, 4).tolist()}")
    print(f"      (one test row = {1 / 171:.4f} accuracy)")

    print("\n  B3  Smaller training sets (stratified subsample of the 398 training rows, seeds 42..51)")
    for n in (50, 100, 200, 398):
        res = {"lgb": [], "lgb_mdl1": [], "xgb": []}
        for s in SEEDS:
            Xtr, Xte, ytr, yte = split(s)
            if n < len(ytr):
                Xtr, _, ytr, _ = train_test_split(Xtr, ytr, train_size=n, random_state=s, stratify=ytr)
            res["lgb"].append(accuracy_score(yte, fit_lgb(Xtr, ytr, s).predict(Xte)))
            res["lgb_mdl1"].append(accuracy_score(yte, fit_lgb(Xtr, ytr, s, min_child_samples=1).predict(Xte)))
            res["xgb"].append(accuracy_score(yte, xgb.XGBClassifier(
                n_estimators=100, random_state=s, n_jobs=THREADS).fit(Xtr, ytr).predict(Xte)))
        print(f"      n_train={n:4d}  LightGBM default {np.mean(res['lgb']):.4f}  "
              f"LightGBM min_data_in_leaf=1 {np.mean(res['lgb_mdl1']):.4f}  "
              f"XGBoost default {np.mean(res['xgb']):.4f}")


# ---------------------------------------------------------------------------
# Part C: larger data, timing
# ---------------------------------------------------------------------------

def load_covtype():
    d = fetch_covtype(data_home=str(CACHE))
    X = d.data.astype(np.float32)
    y = (d.target == 2).astype(np.int32)  # Lodgepole Pine vs the other six cover types
    return X, y


def timed(fn, repeats):
    times, out = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), times, out


def part_c(sizes=(10_000, 50_000, 200_000, 481_012), repeats=5, rounds=200) -> None:
    header(f"PART C  Covertype (binary: Lodgepole Pine vs rest), {rounds} rounds, lr 0.1, "
           f"{THREADS} threads, median of {repeats} runs")
    X, y = load_covtype()
    print(f"  full data {X.shape}, positive rate {y.mean():.4f}, cpu count {os.cpu_count()}")
    Xpool, Xte, ypool, yte = train_test_split(X, y, test_size=100_000, random_state=42, stratify=y)
    print(f"  fixed test set: {len(yte)} rows; training pool: {len(ypool)} rows")
    print("  NOTE: other jobs were running on this machine; timings are indicative only.")

    common_lgb = dict(objective="binary", learning_rate=0.1, num_leaves=31, num_threads=THREADS,
                      verbose=-1, seed=42, deterministic=False)
    configs = {
        "LightGBM (default, num_leaves=31)": ("lgb", {}),
        "LightGBM GOSS": ("lgb", {"data_sample_strategy": "goss"}),
        "LightGBM enable_bundle=false": ("lgb", {"enable_bundle": False}),
        "XGBoost hist depthwise max_depth=6": ("xgb", dict(max_depth=6)),
        "XGBoost hist lossguide max_leaves=31": ("xgb", dict(grow_policy="lossguide", max_leaves=31,
                                                              max_depth=0)),
    }
    for n in sizes:
        idx = np.random.RandomState(n).permutation(len(ypool))[:n]
        Xtr, ytr = Xpool[idx], ypool[idx]
        print(f"\n  n_train = {n}")
        for name, (lib, extra) in configs.items():
            if lib == "lgb":
                def run():
                    ds = lgb.Dataset(Xtr, ytr, params={"verbose": -1, **({"enable_bundle": False}
                                                                         if extra.get("enable_bundle") is False else {})})
                    return lgb.train({**common_lgb, **extra}, ds, num_boost_round=rounds)
                med, ts, model = timed(run, repeats)
                p = model.predict(Xte)
            else:
                def run():
                    dtr = xgb.DMatrix(Xtr, label=ytr, nthread=THREADS)
                    params = dict(objective="binary:logistic", eta=0.1, tree_method="hist",
                                  nthread=THREADS, seed=42, max_bin=256, **extra)
                    return xgb.train(params, dtr, num_boost_round=rounds)
                med, ts, model = timed(run, repeats)
                p = model.predict(xgb.DMatrix(Xte, nthread=THREADS))
            acc = accuracy_score(yte, (p > 0.5).astype(int))
            print(f"    {name:38s} time {med:7.2f} s  (runs {', '.join(f'{t:.2f}' for t in ts)})  "
                  f"test acc {acc:.4f}", flush=True)


if __name__ == "__main__":
    print(f"lightgbm {lgb.__version__} from {'installed site-packages' if 'site-packages' in lgb.__file__ else 'a local folder (shadowed!)'}")
    print(f"xgboost {xgb.__version__}, numpy {np.__version__}, python {sys.version.split()[0]}")
    parts = set(sys.argv[1:]) or {"A", "B", "C"}
    if "A" in parts:
        part_a()
    if "B" in parts:
        part_b()
    if "C" in parts:
        part_c()
