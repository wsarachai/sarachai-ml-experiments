"""Experiments for the "stacking-and-voting" blog post.

Combining different models (voting / averaging, weighted averaging, blending
and stacking) against the best single model, on Adult (binary) and California
housing (regression), with the split convention of the tree-models-comparison
post so the numbers are comparable.

Protocol
--------
* Per seed (0..4) ONE train / validation / test split, 60 / 20 / 20
  (stratified for Adult), shared by every model.  The test part is touched only
  for the final scores.
* "dev" = train + validation (80%).  Every method except blending sees exactly
  the dev rows:
    single    each base model fitted on dev.
    best      the base model with the best 5-fold out-of-fold (OOF) score on dev
              (chosen without looking at the test set).  The best-on-test model
              ("oracle") is also stored, only as a reference.
    average   equal-weight average of the base models' probabilities /
              predictions (= VotingClassifier(voting="soft") / VotingRegressor).
    hard      majority vote of class labels (VotingClassifier(voting="hard"),
              Adult only; scored on accuracy).
    weighted  non-negative weights summing to 1, fitted on the OOF predictions
              (minimises OOF log loss / MSE).
    stack     meta-learner on the OOF predictions = what StackingClassifier /
              StackingRegressor do with cv=5 (StratifiedKFold / KFold,
              shuffle=False) and the default final estimator
              (LogisticRegression / RidgeCV); base models refitted on all dev rows.
    stack_gbm the same with a small LightGBM as the meta-learner.
    stack_logit  Adult only: the LogisticRegression meta-learner fed the
              log-odds logit(p) of the OOF probabilities instead of p.
    stack_pt  California only: RidgeCV meta-learner with passthrough=True
              (OOF predictions + the 8 raw features).
    blend     base models fitted on the train part (60%) only, meta-learner
              (LogisticRegression / RidgeCV) fitted on their predictions for
              the validation part (20%).
    leak      meta-learner fitted on IN-SAMPLE predictions (base models fitted
              on dev, predicting dev) - the mistake stacking avoids.
* Two groups of base learners:
    boosters  XGBoost, LightGBM, CatBoost (expected to be highly correlated).
              HistGB was dropped after a first pass: on this shared machine one
              fit on Adult took ~15 s, too slow for 7 fits x 5 seeds.
    diverse   LightGBM, Random Forest, logistic / ridge regression, k-NN
              (the linear model and k-NN inside Pipelines with one-hot encoding
              and standard scaling)
* Base models use fixed settings (no early stopping, so that the same estimator
  can be refitted inside every fold): learning rate 0.1, ~32 leaves, 200 rounds
  on Adult and 500 on California (CatBoost: depth 5, 400 / 1000 iterations,
  one-hot categoricals - its native categorical mode takes minutes per fit).
* Threads: 4 per model.
* Per seed we store test metrics of every method, the residual correlation
  matrix of the base models on the test rows, and timings (fit seconds per fit,
  prediction time for 10,000 rows).

Run:    .venv/Scripts/python.exe stacking-and-voting/experiments.py
Output: stacking-and-voting/raw/results.json   (report.py prints results.txt)
"""

from __future__ import annotations

import json
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
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_california_housing, fetch_openml
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, root_mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "data_cache"
RAW = HERE / "raw"
THREADS = 4
SEEDS = [0, 1, 2, 3, 4]
FOLDS = 5
PRED_ROWS = 10_000
GROUPS = {
    "boosters": ["XGBoost", "LightGBM", "CatBoost"],
    "diverse": ["LightGBM", "RandomForest", "Linear", "kNN"],
}
ALL_MODELS = ["XGBoost", "LightGBM", "CatBoost", "RandomForest", "Linear", "kNN"]


# ---------------------------------------------------------------- data
def load_adult():
    d = fetch_openml("adult", version=2, as_frame=True, data_home=str(CACHE / "openml"))
    X = d.data.copy()
    y = (d.target == ">50K").astype(int).to_numpy()
    num = [c for c in X.columns if X[c].dtype.kind in "if"]
    cat = [c for c in X.columns if c not in num]
    X[num] = X[num].astype(float)
    for c in cat:  # plain strings with an explicit "missing" level for every encoder
        X[c] = X[c].astype(object).where(X[c].notna(), "missing").astype(str)
    return X, y, num, cat


def load_california():
    d = fetch_california_housing(data_home=str(CACHE), as_frame=True)
    X = d.data.astype(float)
    return X, d.target.to_numpy(), list(X.columns), []


def three_way(X, y, seed, task):
    """Same split as tree-models-comparison: 60 / 20 / 20."""
    strat = y if task == "binary" else None
    Xrest, Xte, yrest, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=strat)
    strat = yrest if task == "binary" else None
    Xtr, Xva, ytr, yva = train_test_split(Xrest, yrest, test_size=0.25, random_state=seed, stratify=strat)
    return [d.reset_index(drop=True) for d in (Xtr, Xva, Xte)], ytr, yva, yte


# ---------------------------------------------------------------- base models
def onehot(cat, num, scale):
    num_steps = [("imp", SimpleImputer())] + ([("sc", StandardScaler())] if scale else [])
    parts = [("num", Pipeline(num_steps), num)]
    if cat:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False), cat))
    return ColumnTransformer(parts)


def make_model(name, task, num, cat, seed):
    clf = task == "binary"
    rounds = 200 if clf else 500
    if name == "HistGB":
        cls = HistGradientBoostingClassifier if clf else HistGradientBoostingRegressor
        est = cls(learning_rate=0.1, max_iter=rounds, max_leaf_nodes=32, early_stopping=False, random_state=seed)
        return make_pipeline(onehot(cat, num, False), est) if cat else est
    if name == "XGBoost":
        cls = XGBClassifier if clf else XGBRegressor
        est = cls(n_estimators=rounds, learning_rate=0.1, max_depth=5, n_jobs=THREADS, random_state=seed)
        return make_pipeline(onehot(cat, num, False), est) if cat else est
    if name == "LightGBM":
        cls = LGBMClassifier if clf else LGBMRegressor
        est = cls(n_estimators=rounds, learning_rate=0.1, num_leaves=32, n_jobs=THREADS, random_state=seed,
                  verbose=-1)
        return make_pipeline(onehot(cat, num, False), est) if cat else est
    if name == "CatBoost":
        cls = CatBoostClassifier if clf else CatBoostRegressor
        est = cls(iterations=400 if clf else 1000, learning_rate=0.1, depth=5, thread_count=THREADS,
                  random_seed=seed, verbose=0, allow_writing_files=False)
        return make_pipeline(onehot(cat, num, False), est) if cat else est
    if name == "RandomForest":
        cls = RandomForestClassifier if clf else RandomForestRegressor
        # max_features "sqrt" for classification is the default; for regression
        # 1/3 of the features (the classic choice) is faster and decorrelates more.
        est = cls(n_estimators=200, min_samples_leaf=2, max_features="sqrt" if clf else 1 / 3,
                  n_jobs=THREADS, random_state=seed)
        if cat:
            enc = ColumnTransformer([("cat", TargetEncoder(target_type="binary",
                                                          cv=StratifiedKFold(5, shuffle=True, random_state=seed)), cat)],
                                    remainder="passthrough")
            return make_pipeline(enc, est)
        return est
    if name == "Linear":
        est = LogisticRegression(C=1.0, max_iter=2000) if clf else RidgeCV(alphas=np.logspace(-3, 3, 13))
        return make_pipeline(onehot(cat, num, True), est)
    if name == "kNN":
        cls = KNeighborsClassifier if clf else KNeighborsRegressor
        est = cls(n_neighbors=50 if clf else 15, weights="distance", n_jobs=THREADS)
        return make_pipeline(onehot(cat, num, True), est)
    raise ValueError(name)


def predict(model, task, X):
    return model.predict_proba(X)[:, 1] if task == "binary" else model.predict(X)


def timed_fit(model, X, y):
    t0 = time.perf_counter()
    model.fit(X, y)
    return model, time.perf_counter() - t0


def predict_time(model, task, X):
    Xp = X.iloc[np.arange(PRED_ROWS) % len(X)]
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        predict(model, task, Xp)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


# ---------------------------------------------------------------- metrics and combiners
EPS = 1e-6


def score(task, y, p):
    if task == "binary":
        p = np.clip(p, EPS, 1 - EPS)
        return {"roc_auc": roc_auc_score(y, p), "log_loss": log_loss(y, p),
                "accuracy": accuracy_score(y, (p >= 0.5).astype(int))}
    return {"rmse": root_mean_squared_error(y, p)}


def main_metric(task):
    return ("log_loss", "min") if task == "binary" else ("rmse", "min")


def simplex_weights(task, P, y):
    """Non-negative weights summing to 1 that minimise log loss / MSE of P @ w."""
    k = P.shape[1]

    def loss(w):
        p = P @ w
        if task == "binary":
            p = np.clip(p, EPS, 1 - EPS)
            return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        return np.mean((y - p) ** 2)

    # ftol must be tight: with three near-identical boosters the log-loss surface is so
    # flat that SLSQP's default tolerance (1e-6) stops at the equal-weight start point.
    res = minimize(loss, np.full(k, 1 / k), method="SLSQP", bounds=[(0, 1)] * k,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"ftol": 1e-12, "maxiter": 500})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def meta_model(task, kind, seed):
    if kind == "linear":
        return LogisticRegression() if task == "binary" else RidgeCV()  # the sklearn defaults
    cls = LGBMClassifier if task == "binary" else LGBMRegressor
    return cls(n_estimators=200, learning_rate=0.05, num_leaves=8, min_child_samples=50,
               n_jobs=THREADS, random_state=seed, verbose=-1)


def fit_meta(task, kind, seed, Z, y):
    m = meta_model(task, kind, seed)
    t0 = time.perf_counter()
    m.fit(Z, y)
    return m, time.perf_counter() - t0


def meta_predict(task, m, Z):
    return m.predict_proba(Z)[:, 1] if task == "binary" else m.predict(Z)


def meta_coefs(task, m):
    if task == "binary":
        return [float(c) for c in m.coef_.ravel()] + [float(m.intercept_[0])]
    return [float(c) for c in m.coef_.ravel()] + [float(m.intercept_)]


# ---------------------------------------------------------------- one dataset
def run_dataset(block, loader, task):
    X, y, num, cat = loader()
    out = {"block": block, "task": task, "rows": int(len(y)), "cols": int(X.shape[1]), "seeds": SEEDS,
           "groups": GROUPS, "runs": []}
    t_block = time.perf_counter()
    for seed in SEEDS:
        (Xtr, Xva, Xte), ytr, yva, yte = three_way(X, y, seed, task)
        Xdev = pd.concat([Xtr, Xva], ignore_index=True)
        ydev = np.concatenate([ytr, yva])
        cv = (StratifiedKFold(FOLDS) if task == "binary" else KFold(FOLDS))  # sklearn's cv=5, shuffle=False
        splits = list(cv.split(Xdev, ydev))
        run = {"seed": seed, "split_sizes": [len(ytr), len(yva), len(yte)], "models": {}}
        P = {}
        for name in ALL_MODELS:
            rec = {}
            # out-of-fold predictions on dev (what stacking and the weights are trained on)
            oof = np.zeros(len(ydev))
            fold_fit = []
            for tr_i, te_i in splits:
                m, s = timed_fit(make_model(name, task, num, cat, seed), Xdev.iloc[tr_i], ydev[tr_i])
                oof[te_i] = predict(m, task, Xdev.iloc[te_i])
                fold_fit.append(s)
            # full fit on dev (single model, voting, stacking at predict time)
            m_dev, s_dev = timed_fit(make_model(name, task, num, cat, seed), Xdev, ydev)
            p_test = predict(m_dev, task, Xte)
            p_insample = predict(m_dev, task, Xdev)
            rec["pred_10k_s"] = predict_time(m_dev, task, Xte)
            # fit on train only (blending)
            m_tr, s_tr = timed_fit(make_model(name, task, num, cat, seed), Xtr, ytr)
            P[name] = {"oof": oof, "test": p_test, "insample": p_insample,
                       "blend_val": predict(m_tr, task, Xva), "blend_test": predict(m_tr, task, Xte)}
            rec.update({"fit_dev_s": s_dev, "fit_folds_s": fold_fit, "fit_train_s": s_tr,
                        "test": score(task, yte, p_test), "oof": score(task, ydev, oof)})
            run["models"][name] = rec
            print(f"[{block}] seed {seed} {name:<12} test {score(task, yte, p_test)} fit {s_dev:.1f}s", flush=True)
            del m_dev, m_tr

        # residual correlation on the test rows (all seven models)
        R = np.column_stack([yte - P[n]["test"] for n in ALL_MODELS])
        run["residual_corr"] = np.corrcoef(R, rowvar=False).tolist()
        run["residual_cov"] = (R.T @ R / len(yte)).tolist()  # mean products (for the w'Mw identity)

        key, _ = main_metric(task)
        run["groups"] = {}
        for g, names in GROUPS.items():
            res = {}
            oof_scores = {n: run["models"][n]["oof"][key] for n in names}
            best = min(oof_scores, key=oof_scores.get)
            test_scores = {n: run["models"][n]["test"][key] for n in names}
            oracle = min(test_scores, key=test_scores.get)
            res["best"] = {"model": best, **run["models"][best]["test"]}
            res["oracle"] = {"model": oracle, **run["models"][oracle]["test"]}
            Ptest = np.column_stack([P[n]["test"] for n in names])
            Poof = np.column_stack([P[n]["oof"] for n in names])
            Pin = np.column_stack([P[n]["insample"] for n in names])
            res["average"] = score(task, yte, Ptest.mean(1))
            if task == "binary":
                votes = (Ptest >= 0.5).sum(1)
                # 4 voters: a 2-2 tie goes to class 0 (sklearn's argmax of bincount)
                res["hard"] = {"accuracy": accuracy_score(yte, (votes > len(names) / 2).astype(int))}
            w = simplex_weights(task, Poof, ydev)
            res["weighted"] = {**score(task, yte, Ptest @ w), "weights": w.tolist()}
            for kind, label in [("linear", "stack"), ("gbm", "stack_gbm")]:
                m, s = fit_meta(task, kind, seed, Poof, ydev)
                res[label] = {**score(task, yte, meta_predict(task, m, Ptest)), "meta_fit_s": s}
                if kind == "linear":
                    res[label]["coefs"] = meta_coefs(task, m)
            if task == "binary":
                # the same LogisticRegression meta-learner, fed log-odds instead of probabilities
                def logit(p):
                    p = np.clip(p, EPS, 1 - EPS)
                    return np.log(p / (1 - p))
                m, s = fit_meta(task, "linear", seed, logit(Poof), ydev)
                res["stack_logit"] = {**score(task, yte, meta_predict(task, m, logit(Ptest))), "meta_fit_s": s,
                                      "coefs": meta_coefs(task, m)}
            if task == "regression":
                Z = np.column_stack([Poof, Xdev.to_numpy()])
                m, s = fit_meta(task, "linear", seed, Z, ydev)
                res["stack_pt"] = {**score(task, yte, m.predict(np.column_stack([Ptest, Xte.to_numpy()]))),
                                   "meta_fit_s": s}
            # blending: meta on validation predictions of train-only models
            Bv = np.column_stack([P[n]["blend_val"] for n in names])
            Bt = np.column_stack([P[n]["blend_test"] for n in names])
            m, s = fit_meta(task, "linear", seed, Bv, yva)
            res["blend"] = {**score(task, yte, meta_predict(task, m, Bt)), "coefs": meta_coefs(task, m)}
            # the leak: meta fitted on in-sample predictions
            m, s = fit_meta(task, "linear", seed, Pin, ydev)
            res["leak"] = {**score(task, yte, meta_predict(task, m, Ptest)), "coefs": meta_coefs(task, m),
                           "train": score(task, ydev, meta_predict(task, m, Pin))}
            res["stack"]["train_oof"] = score(task, ydev, meta_predict(
                task, fit_meta(task, "linear", seed, Poof, ydev)[0], Poof))
            run["groups"][g] = res
            print(f"[{block}] seed {seed} {g}: best {best} {res['best'][key]:.4f} avg {res['average'][key]:.4f} "
                  f"w {res['weighted'][key]:.4f} stack {res['stack'][key]:.4f} blend {res['blend'][key]:.4f} "
                  f"leak {res['leak'][key]:.4f}", flush=True)
        if seed == 0:
            run["sklearn_check"] = sklearn_check(task, num, cat, seed, Xdev, ydev, Xte, P)
        out["runs"].append(run)
    out["block_seconds"] = time.perf_counter() - t_block
    return out


def sklearn_check(task, num, cat, seed, Xdev, ydev, Xte, P):
    """Confirm that the hand-built combiners equal sklearn's Voting* / Stacking* (seed 0, diverse group)."""
    names = GROUPS["diverse"]
    ests = [(n, make_model(n, task, num, cat, seed)) for n in names]
    manual_avg = np.column_stack([P[n]["test"] for n in names]).mean(1)
    if task == "binary":
        vote = VotingClassifier(ests, voting="soft")
        stack = StackingClassifier([(n, clone(e)) for n, e in ests], cv=5)
    else:
        vote = VotingRegressor(ests)
        stack = StackingRegressor([(n, clone(e)) for n, e in ests], cv=5)
    vote.fit(Xdev, ydev)
    t0 = time.perf_counter()
    stack.fit(Xdev, ydev)
    stack_fit = time.perf_counter() - t0
    Poof = np.column_stack([P[n]["oof"] for n in names])
    Ptest = np.column_stack([P[n]["test"] for n in names])
    m, _ = fit_meta(task, "linear", seed, Poof, ydev)
    manual_stack = meta_predict(task, m, Ptest)
    sk_vote = predict(vote, task, Xte)
    sk_stack = predict(stack, task, Xte)
    t0 = time.perf_counter()
    predict(stack, task, Xte.iloc[np.arange(PRED_ROWS) % len(Xte)])
    stack_pred = time.perf_counter() - t0
    return {"max_abs_diff_voting": float(np.max(np.abs(sk_vote - manual_avg))),
            "max_abs_diff_stacking": float(np.max(np.abs(sk_stack - manual_stack))),
            "sklearn_stack_fit_s": stack_fit, "sklearn_stack_pred_10k_s": stack_pred,
            "final_estimator": type(stack.final_estimator_).__name__,
            "stack_method_": list(getattr(stack, "stack_method_", []))}


BLOCKS = {
    "adult": lambda: run_dataset("adult", load_adult, "binary"),
    "california": lambda: run_dataset("california", load_california, "regression"),
}


def environment():
    return {"python": platform.python_version(), "sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
            "lightgbm": lightgbm.__version__, "catboost": catboost.__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "threads_per_model": THREADS}


if __name__ == "__main__":
    RAW.mkdir(exist_ok=True)
    todo = sys.argv[1:] or list(BLOCKS)
    with threadpool_limits(limits=THREADS):
        for b in todo:
            res = BLOCKS[b]()
            res["environment"] = environment()
            (RAW / f"{b}.json").write_text(json.dumps(res, indent=1))
            print(f"[{b}] done in {res['block_seconds']:.0f} s", flush=True)
