"""Experiments for the "tree-models-comparison" blog post.

Random Forest, scikit-learn HistGradientBoosting, XGBoost, LightGBM and
CatBoost measured on the same data with the same protocol.

Protocol
--------
* Every seed draws ONE train / validation / test split that all five models
  share (60 / 20 / 20, stratified for classification).  The model's own
  random seed is the same number.  The test set is used only for the final
  scores; it never drives early stopping or any choice.
* Two settings per library:
    default  the library's out-of-the-box estimator, fitted on the training
             part only (the validation part is not used).  HistGradientBoosting
             keeps its default early_stopping="auto", which on more than
             10,000 rows holds out 10% of the TRAINING rows internally.
    matched  learning rate 0.1, at most MAX_ROUNDS rounds, early stopping on
             the validation part (patience 50) and trees of about 32 leaves,
             each library in its own growth style:
               XGBoost  max_depth=5 (depth-wise, <= 32 leaves)
               LightGBM num_leaves=32 (leaf-wise)
               CatBoost depth=5 (symmetric, exactly 32 leaves)
               HistGB   max_leaf_nodes=32 (best-first)
             Random Forest has no learning rate or early stopping; its
             "matched" setting is 300 trees, otherwise default.
* Categorical columns (Adult only), each library with the handling the
  data-preparation-for-trees experiments found best or that the library is
  built around:
    Random Forest  cross-fitted TargetEncoder (best for RF on Adult there)
    XGBoost        one-hot (best for XGBoost on Adult there)
    LightGBM       native (pandas category dtype)
    HistGB         native (pandas category dtype, categorical_features="from_dtype")
    CatBoost       native cat_features (missing -> the string "missing")
* Threads: 4 per model (n_jobs / nthread / thread_count, and threadpoolctl
  for OpenMP inside scikit-learn).
* Per model and seed: test metrics, fit time (wall clock, one fit per seed,
  includes the encoder), prediction time for 10,000 rows (median of 3 calls
  of predict_proba / predict), pickled model size, number of trees kept.

Blocks (run one or more):  adult  california  covertype  breast  scaling
  covertype  multiclass, a 10,000-row training sample per seed
             (validation 5,000, test 20,000), MAX_ROUNDS 500.  Reduced from
             20,000 / 50,000 / 1000 after a first pass ran far over the ~30 minute
             budget on a shared machine (Adult alone took 23 minutes, mostly
             CatBoost with native categoricals).
  scaling    Covertype, one seed, fixed test 50,000 rows, training sizes
             10k / 100k / all remaining rows; every booster gets the same
             fixed budget (100 rounds, learning rate 0.1, ~32 leaves, no
             early stopping); RF 100 trees (skipped on the full size: its
             fully grown trees need several GB of memory).

Run:    .venv/Scripts/python.exe tree-models-comparison/experiments.py adult california ...
Output: tree-models-comparison/raw/<block>.json  (report.py builds results.txt)
"""

from __future__ import annotations

import json
import os
import pickle
import platform
import statistics
import sys
import time
import warnings
from pathlib import Path

import catboost
import lightgbm
import numpy as np
import pandas as pd
import sklearn
import xgboost
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_california_housing, fetch_covtype, fetch_openml, load_breast_cancer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    log_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "data_cache"
RAW = HERE / "raw"
THREADS = 4
SEEDS = [0, 1, 2, 3, 4]
PATIENCE = 50
LR = 0.1
PRED_ROWS = 10_000
PRED_REPEATS = 3
MODELS = ["RandomForest", "HistGB", "XGBoost", "LightGBM", "CatBoost"]


# ---------------------------------------------------------------- data
def load_adult():
    d = fetch_openml("adult", version=2, as_frame=True, data_home=str(CACHE / "openml"))
    X = d.data.copy()
    y = (d.target == ">50K").astype(int).to_numpy()
    num = [c for c in X.columns if X[c].dtype.kind in "if"]
    cat = [c for c in X.columns if c not in num]
    X[num] = X[num].astype(float)
    return X, y, cat


def load_california():
    d = fetch_california_housing(data_home=str(CACHE), as_frame=True)
    return d.data.astype(float), d.target.to_numpy(), []


def load_breast():
    d = load_breast_cancer(as_frame=True)
    return d.data.astype(float), d.target.to_numpy(), []


def load_covtype():
    d = fetch_covtype(data_home=str(CACHE), as_frame=True)
    return d.data.astype(float), (d.target.to_numpy() - 1).astype(int), []


def for_catboost(X: pd.DataFrame, cat: list[str]) -> pd.DataFrame:
    """CatBoost needs strings in categorical columns; NaN becomes 'missing'."""
    if not cat:
        return X
    X = X.copy()
    for c in cat:
        X[c] = X[c].astype(object).where(X[c].notna(), "missing").astype(str)
    return X


# ---------------------------------------------------------------- models
def fit_model(name: str, setting: str, task: str, cat: list[str], seed: int,
              Xtr, ytr, Xva, yva, max_rounds: int, fixed_rounds: int | None = None):
    """Fit one model; return (fitted predictor, number of trees kept, fit seconds).

    fixed_rounds: if given, use the matched tree size and learning rate but
    exactly this many rounds and no early stopping (the scaling block).
    """
    clf = task != "regression"
    matched = setting == "matched"
    es = matched and fixed_rounds is None
    rounds = fixed_rounds if fixed_rounds is not None else max_rounds

    t0 = time.perf_counter()
    if name == "RandomForest":
        cls = RandomForestClassifier if clf else RandomForestRegressor
        n = 300 if matched and fixed_rounds is None else 100
        rf = cls(n_estimators=n, n_jobs=THREADS, random_state=seed)
        if cat:
            enc = ColumnTransformer([("cat", TargetEncoder(target_type="binary", random_state=seed), cat)],
                                    remainder="passthrough")
            model = Pipeline([("enc", enc), ("rf", rf)]).fit(Xtr, ytr)
        else:
            model = rf.fit(Xtr, ytr)
        trees = n

    elif name == "HistGB":
        cls = HistGradientBoostingClassifier if clf else HistGradientBoostingRegressor
        if setting == "default":
            model = cls(random_state=seed).fit(Xtr, ytr)
        else:
            model = cls(learning_rate=LR, max_iter=rounds, max_leaf_nodes=32, random_state=seed,
                        early_stopping=es, n_iter_no_change=PATIENCE)
            if es:
                model.fit(Xtr, ytr, X_val=Xva, y_val=yva)
            else:
                model.fit(Xtr, ytr)
        trees = int(model.n_iter_)

    elif name == "XGBoost":
        cls = XGBClassifier if clf else XGBRegressor
        enc = None
        if cat:
            enc = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)],
                                    remainder="passthrough")
            Xtr_m = enc.fit_transform(Xtr)
            Xva_m = enc.transform(Xva) if es else None
        else:
            Xtr_m, Xva_m = Xtr, Xva
        if setting == "default":
            xg = cls(n_jobs=THREADS, random_state=seed).fit(Xtr_m, ytr)
            trees = xg.get_booster().num_boosted_rounds()
        else:
            kw = dict(n_estimators=rounds, learning_rate=LR, max_depth=5, n_jobs=THREADS, random_state=seed)
            if es:
                xg = cls(early_stopping_rounds=PATIENCE, **kw).fit(Xtr_m, ytr, eval_set=[(Xva_m, yva)], verbose=False)
                trees = xg.best_iteration + 1
            else:
                xg = cls(**kw).fit(Xtr_m, ytr)
                trees = rounds
        model = Pipeline([("enc", enc), ("xgb", xg)]) if enc is not None else xg

    elif name == "LightGBM":
        cls = LGBMClassifier if clf else LGBMRegressor
        if setting == "default":
            model = cls(n_jobs=THREADS, random_state=seed, verbose=-1).fit(Xtr, ytr)
            trees = model.booster_.num_trees() // (model.n_classes_ if clf and model.n_classes_ > 2 else 1)
        else:
            model = cls(n_estimators=rounds, learning_rate=LR, num_leaves=32, n_jobs=THREADS,
                        random_state=seed, verbose=-1)
            if es:
                model.fit(Xtr, ytr, eval_X=(Xva,), eval_y=(yva,),
                          callbacks=[lightgbm.early_stopping(PATIENCE, verbose=False)])
                trees = model.best_iteration_
            else:
                model.fit(Xtr, ytr)
                trees = rounds

    elif name == "CatBoost":
        cls = CatBoostClassifier if clf else CatBoostRegressor
        base = dict(thread_count=THREADS, random_seed=seed, verbose=0, allow_writing_files=False)
        if task == "multiclass":
            base["loss_function"] = "MultiClass"
        if setting == "default":
            model = cls(**base).fit(Xtr, ytr, cat_features=cat or None)
        else:
            model = cls(iterations=rounds, learning_rate=LR, depth=5, **base)
            if es:
                model.fit(Xtr, ytr, cat_features=cat or None, eval_set=(Xva, yva),
                          early_stopping_rounds=PATIENCE, use_best_model=True)
            else:
                model.fit(Xtr, ytr, cat_features=cat or None)
        trees = int(model.tree_count_)
    else:
        raise ValueError(name)
    return model, int(trees), time.perf_counter() - t0


def predict(model, task, X):
    if task == "regression":
        return model.predict(X)
    return model.predict_proba(X)


def scores(task, y, p, n_classes):
    if task == "binary":
        p1 = p[:, 1]
        return {"roc_auc": roc_auc_score(y, p1), "avg_precision": average_precision_score(y, p1),
                "log_loss": log_loss(y, p1), "accuracy": accuracy_score(y, (p1 >= 0.5).astype(int))}
    if task == "multiclass":
        return {"accuracy": accuracy_score(y, p.argmax(1)), "log_loss": log_loss(y, p, labels=list(range(n_classes)))}
    return {"rmse": root_mean_squared_error(y, p), "mae": mean_absolute_error(y, p), "r2": r2_score(y, p)}


def measure(name, setting, task, cat, seed, Xtr, ytr, Xva, yva, Xte, yte, n_classes, max_rounds,
            fixed_rounds=None):
    cb = name == "CatBoost"
    prep = (lambda X: for_catboost(X, cat)) if cb else (lambda X: X)
    Xtr_, Xva_, Xte_ = prep(Xtr), prep(Xva) if Xva is not None else None, prep(Xte)
    model, trees, fit_s = fit_model(name, setting, task, cat, seed, Xtr_, ytr, Xva_, yva, max_rounds, fixed_rounds)
    p = predict(model, task, Xte_)
    res = scores(task, yte, p, n_classes)
    # prediction time on exactly PRED_ROWS rows (test rows repeated if the test set is smaller)
    Xp = Xte_.iloc[np.arange(PRED_ROWS) % len(Xte_)]
    pt = []
    for _ in range(PRED_REPEATS):
        t0 = time.perf_counter()
        predict(model, task, Xp)
        pt.append(time.perf_counter() - t0)
    size = len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
    res.update({"fit_s": fit_s, "predict_10k_s": statistics.median(pt), "predict_runs": pt,
                "size_bytes": size, "trees": trees})
    del model
    return res


def three_way(X, y, seed, task):
    strat = y if task != "regression" else None
    Xrest, Xte, yrest, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=strat)
    strat = yrest if task != "regression" else None
    Xtr, Xva, ytr, yva = train_test_split(Xrest, yrest, test_size=0.25, random_state=seed, stratify=strat)
    return [d.reset_index(drop=True) if hasattr(d, "reset_index") else d for d in (Xtr, Xva, Xte)], ytr, yva, yte


def run_protocol(block, loader, task, max_rounds, splitter=None):
    X, y, cat = loader()
    n_classes = int(len(np.unique(y))) if task != "regression" else 0
    out = {"block": block, "task": task, "rows": int(len(y)), "cols": int(X.shape[1]),
           "categorical": cat, "max_rounds": max_rounds, "seeds": SEEDS, "runs": []}
    if task == "binary":
        out["positive_rate"] = float(np.mean(y))
    t_block = time.perf_counter()
    for seed in SEEDS:
        if splitter:
            (Xtr, Xva, Xte), ytr, yva, yte = splitter(X, y, seed)
        else:
            (Xtr, Xva, Xte), ytr, yva, yte = three_way(X, y, seed, task)
        out["split_sizes"] = [len(ytr), len(yva), len(yte)]
        for setting in ["default", "matched"]:
            for name in MODELS:
                r = measure(name, setting, task, cat, seed, Xtr, ytr, Xva, yva, Xte, yte, n_classes, max_rounds)
                r.update({"seed": seed, "setting": setting, "model": name})
                out["runs"].append(r)
                main = {k: round(v, 4) for k, v in r.items() if k in ("roc_auc", "accuracy", "rmse", "log_loss")}
                print(f"[{block}] seed {seed} {setting:<8} {name:<12} {main} fit {r['fit_s']:.2f}s "
                      f"pred {r['predict_10k_s'] * 1000:.1f}ms size {r['size_bytes'] / 1e6:.2f}MB trees {r['trees']}",
                      flush=True)
    out["block_seconds"] = time.perf_counter() - t_block
    return out


def covtype_splitter(n_train, n_valid, n_test):
    def split(X, y, seed):
        idx = np.random.default_rng(seed).permutation(len(y))
        te, va, tr = idx[:n_test], idx[n_test:n_test + n_valid], idx[n_test + n_valid:n_test + n_valid + n_train]
        take = lambda ix: X.iloc[ix].reset_index(drop=True)  # noqa: E731
        return (take(tr), take(va), take(te)), y[tr], y[va], y[te]
    return split


def run_scaling():
    X, y, _ = load_covtype()
    idx = np.random.default_rng(0).permutation(len(y))
    te, pool = idx[:50_000], idx[50_000:]
    Xte, yte = X.iloc[te].reset_index(drop=True), y[te]
    sizes = [10_000, 100_000, len(pool)]
    out = {"block": "scaling", "task": "multiclass", "test_rows": len(te), "sizes": sizes,
           "fixed_rounds": 100, "runs": []}
    t_block = time.perf_counter()
    for n in sizes:
        Xtr, ytr = X.iloc[pool[:n]].reset_index(drop=True), y[pool[:n]]
        reps = 3 if n <= 10_000 else 1  # repeats only where a fit is cheap (time budget)
        for name in MODELS:
            if name == "RandomForest" and n > 100_000:
                continue
            runs = []
            for rep in range(reps):
                r = measure(name, "matched", "multiclass", [], 0, Xtr, ytr, None, None, Xte, yte, 7, 100,
                            fixed_rounds=100)
                runs.append(r)
            r = dict(runs[0])
            r["fit_runs"] = [q["fit_s"] for q in runs]
            r["fit_s"] = statistics.median(r["fit_runs"])
            r.update({"n_train": n, "model": name, "repeats": reps})
            out["runs"].append(r)
            print(f"[scaling] n={n:>7} {name:<12} acc {r['accuracy']:.4f} fit {r['fit_s']:.2f}s "
                  f"{[round(t, 2) for t in r['fit_runs']]} pred {r['predict_10k_s'] * 1000:.1f}ms "
                  f"size {r['size_bytes'] / 1e6:.2f}MB", flush=True)
    out["block_seconds"] = time.perf_counter() - t_block
    return out


BLOCKS = {
    "adult": lambda: run_protocol("adult", load_adult, "binary", 2000),
    "california": lambda: run_protocol("california", load_california, "regression", 2000),
    "breast": lambda: run_protocol("breast", load_breast, "binary", 2000),
    "covertype": lambda: run_protocol("covertype", load_covtype, "multiclass", 500,
                                      splitter=covtype_splitter(10_000, 5_000, 20_000)),
    "scaling": run_scaling,
}


def environment():
    return {"python": platform.python_version(), "sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
            "lightgbm": lightgbm.__version__, "catboost": catboost.__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "logical_cpus": os.cpu_count(), "threads_per_model": THREADS,
            "processor": platform.processor()}


if __name__ == "__main__":
    RAW.mkdir(exist_ok=True)
    todo = sys.argv[1:] or list(BLOCKS)
    with threadpool_limits(limits=THREADS):
        for b in todo:
            res = BLOCKS[b]()
            res["environment"] = environment()
            (RAW / f"{b}.json").write_text(json.dumps(res, indent=1))
            print(f"[{b}] done in {res['block_seconds']:.0f} s", flush=True)
