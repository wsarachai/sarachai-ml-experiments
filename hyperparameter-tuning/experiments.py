"""Hyperparameter-tuning experiments for the "hyperparameter-tuning" blog post.

Every number the post calls "measured" comes from this script:

1. Hand example A: why 9 random points cover an important dimension better
   than a 3 x 3 grid (closed form, asserted, plus a Monte Carlo check).
2. Hand example B: an early-stopping trace with patience 3 on a fixed
   validation-loss sequence, asserted against the stopping rule of XGBoost
   (xgb.train + custom metric), LightGBM (lgb.train + early_stopping callback)
   and scikit-learn's HistGradientBoosting (_should_stop).
3. Defaults and early-stopping APIs of the installed libraries (read back from
   fitted models), including CatBoost's use_best_model and whether
   HistGradientBoosting keeps the trees built after its best iteration.
4. How much tuning buys: Adult (log loss) and California housing (RMSE),
   5 seeds, XGBoost and LightGBM, each scored on an untouched test set:
   library defaults / learning rate 0.05 + early stopping / random search
   (16 candidates, early stopping inside), with and without refitting on
   train+validation; random forest with defaults as the low-tuning baseline.
   Also the optimism of the validation score that chose the model.
5. Rounds chosen by early stopping against the learning rate (seed 0).
6. Grid vs random search with the same budget (9 fits) on California housing,
   LightGBM, two hyperparameters (learning rate and L2), random repeated 30x.
7. Successive halving (HalvingRandomSearchCV, experimental) vs
   RandomizedSearchCV on the same 27 candidates (Adult, LightGBM).

Run:  .venv/Scripts/python.exe hyperparameter-tuning/experiments.py
Output: hyperparameter-tuning/results.txt, PNG figures in
hyperparameter-tuning/figures/ and the raw numbers in raw_results.pkl
(``--figures-only`` redraws the figures from that file). Datasets are cached under data_cache/ at the
repository root. Every model is capped at 4 threads; section 4 runs at most
two models at a time.
"""

from __future__ import annotations

import io
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import catboost  # noqa: E402
import lightgbm as lgb  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
import xgboost as xgb  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.stats import loguniform, randint, uniform  # noqa: E402
from sklearn.datasets import fetch_california_housing, fetch_openml  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.experimental import enable_halving_search_cv  # noqa: E402,F401
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    HalvingRandomSearchCV,
    ParameterSampler,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE = ROOT / "data_cache"
FIG_DIR = HERE / "figures"
THREADS = 4
SEEDS = [0, 1, 2, 3, 4]
MAX_ROUNDS = 5000
PATIENCE = 50
ES_LR = 0.05
N_CANDIDATES = 16

# "Paper" palette used by the blog's figures.
PAPER = "#F8F6F1"
INK = "#1F2328"
INK_2 = "#5B6270"
GRID = "#E8E2D6"
TEAL = "#0F766E"
TEAL_LIGHT = "#A7D3CE"
AMBER = "#B45309"
GREY = "#B9B3A6"

warnings.filterwarnings("ignore", category=UserWarning)
out = io.StringIO()


def say(*args) -> None:
    """Print to stdout and to the results buffer."""
    print(*args, flush=True)
    print(*args, file=out)


def header(title: str) -> None:
    say()
    say("=" * 78)
    say(title)
    say("=" * 78)


def style_axes(ax) -> None:
    ax.set_facecolor(PAPER)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK, labelsize=9)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)
    ax.title.set_color(INK)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_adult() -> tuple[pd.DataFrame, np.ndarray]:
    data = fetch_openml("adult", version=2, as_frame=True, data_home=str(CACHE))
    X = data.data.copy()
    for col in X.columns:
        if isinstance(X[col].dtype, pd.CategoricalDtype):
            # Ordinal codes; missing categories stay NaN (all models accept NaN).
            X[col] = X[col].cat.codes.astype(float).replace(-1.0, np.nan)
    X = X.astype(float)
    y = (data.target.astype(str) == ">50K").astype(int).to_numpy()
    return X, y


def load_california() -> tuple[pd.DataFrame, np.ndarray]:
    data = fetch_california_housing(as_frame=True, data_home=str(CACHE))
    return data.data.astype(float), data.target.to_numpy()


DATASETS = {"adult": load_adult, "california": load_california}
TASK = {"adult": "clf", "california": "reg"}


def split(name: str, X, y, seed: int):
    """80/20 train+val / test, then 80/20 train / val inside the 80%."""
    strat = y if TASK[name] == "clf" else None
    X_tv, X_te, y_tv, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=strat)
    strat2 = y_tv if TASK[name] == "clf" else None
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tv, y_tv, test_size=0.2, random_state=seed, stratify=strat2
    )
    return X_tv, y_tv, X_tr, y_tr, X_va, y_va, X_te, y_te


def score(task: str, model, X, y) -> float:
    """Log loss for classification, RMSE for regression (lower is better)."""
    if task == "clf":
        return log_loss(y, model.predict_proba(X)[:, 1])
    return float(np.sqrt(mean_squared_error(y, model.predict(X))))


def auc(model, X, y) -> float:
    return roc_auc_score(y, model.predict_proba(X)[:, 1])


# ---------------------------------------------------------------------------
# 1. Hand example A: grid vs random coverage of the important dimension
# ---------------------------------------------------------------------------


def hand_grid_vs_random() -> None:
    header("1. Hand example A: 9 trials, 3x3 grid vs random, one important dimension")
    p_random = 1 - 0.9**9
    p_grid = 3 * 0.1
    say("Important dimension u in [0, 1); the 'good' region is an interval of width 0.1")
    say("placed uniformly at random (circular, no edge effects).")
    say(f"grid 3x3: distinct u values = 3 -> P(hit) = 3 x 0.1 = {p_grid:.4f}")
    say(f"random 9: distinct u values = 9 -> P(hit) = 1 - 0.9^9 = {p_random:.4f}")
    assert abs(p_random - 0.6126) < 5e-5
    assert abs(p_grid - 0.3) < 1e-12

    rng = np.random.default_rng(0)
    n = 200_000
    start = rng.random(n)
    grid_u = np.array([1 / 6, 1 / 2, 5 / 6])

    def hit(points: np.ndarray) -> np.ndarray:
        d = (points - start[:, None]) % 1.0
        return (d < 0.1).any(axis=1)

    mc_grid = hit(np.broadcast_to(grid_u, (n, 3))).mean()
    mc_rand = hit(rng.random((n, 9))).mean()
    say(f"Monte Carlo ({n:,} draws): grid {mc_grid:.4f}, random {mc_rand:.4f}")
    assert abs(mc_grid - p_grid) < 0.005 and abs(mc_rand - p_random) < 0.005


# ---------------------------------------------------------------------------
# 2. Hand example B: early-stopping trace
# ---------------------------------------------------------------------------

VAL_LOSS = [0.70, 0.62, 0.58, 0.57, 0.575, 0.565, 0.566, 0.567, 0.568, 0.56, 0.555, 0.55]
BASELINE = 0.80  # HistGradientBoosting also scores the constant model (iteration 0)


def hand_trace(losses, patience):
    """Best-so-far rule: stop once `patience` rounds pass without a new best."""
    best, best_round = np.inf, 0
    for r, v in enumerate(losses, start=1):
        if v < best:
            best, best_round = v, r
        elif r - best_round >= patience:
            return r, best_round
    return len(losses), best_round


def xgb_trace(patience):
    rng = np.random.default_rng(0)
    X = rng.random((40, 2))
    y = rng.random(40)
    dtr = xgb.DMatrix(X[:20], label=y[:20])
    dva = xgb.DMatrix(X[20:], label=y[20:])
    calls = [0]

    def metric(predt, dmat):
        v = VAL_LOSS[calls[0]]
        calls[0] += 1
        return "seq", v

    bst = xgb.train(
        {"tree_method": "hist", "nthread": 1, "disable_default_eval_metric": 1},
        dtr,
        num_boost_round=len(VAL_LOSS),
        evals=[(dva, "valid")],
        custom_metric=metric,
        early_stopping_rounds=patience,
        maximize=False,
        verbose_eval=False,
    )
    return bst.num_boosted_rounds(), bst.best_iteration


def lgb_trace(patience):
    rng = np.random.default_rng(0)
    X = rng.random((40, 2))
    y = rng.random(40)
    dtr = lgb.Dataset(X[:20], label=y[:20])
    dva = lgb.Dataset(X[20:], label=y[20:], reference=dtr)
    calls = [0]

    def metric(preds, data):
        v = VAL_LOSS[calls[0]]
        calls[0] += 1
        return "seq", v, False

    bst = lgb.train(
        {"objective": "regression", "metric": "None", "verbose": -1, "num_threads": 1,
         "min_data_in_leaf": 1, "min_data_in_bin": 1},
        dtr,
        num_boost_round=len(VAL_LOSS),
        valid_sets=[dva],
        feval=metric,
        callbacks=[lgb.early_stopping(patience, verbose=False)],
    )
    return calls[0], bst.best_iteration, bst.current_iteration()


def hgb_trace(patience):
    est = HistGradientBoostingRegressor(n_iter_no_change=patience, tol=0.0)
    scores = [-BASELINE]
    for r, v in enumerate(VAL_LOSS, start=1):
        scores.append(-v)  # higher is better inside HGB
        if est._should_stop(scores):
            return r
    return len(VAL_LOSS)


def hand_early_stopping() -> None:
    header("2. Hand example B: early-stopping trace on a fixed validation-loss sequence")
    say("round   :", " ".join(f"{r:>6d}" for r in range(1, len(VAL_LOSS) + 1)))
    say("val loss:", " ".join(f"{v:>6.3f}" for v in VAL_LOSS))
    for patience, expect in [(3, (9, 6)), (5, (12, 12))]:
        stop, best = hand_trace(VAL_LOSS, patience)
        say(f"patience {patience}: by hand stop after round {stop}, best round {best}")
        assert (stop, best) == expect
    stop, best = hand_trace(VAL_LOSS, 3)

    n_rounds, best_it = xgb_trace(3)
    say(f"XGBoost 3.4.1 xgb.train(early_stopping_rounds=3): trees kept = {n_rounds}, "
        f"best_iteration = {best_it} (0-based)")
    assert n_rounds == stop and best_it == best - 1

    evaluated, best_it, cur = lgb_trace(3)
    say(f"LightGBM 4.7.0 early_stopping(3): rounds evaluated = {evaluated}, "
        f"best_iteration = {best_it} (1-based), current_iteration() = {cur}")
    assert evaluated == stop and best_it == best

    hgb_stop = hgb_trace(3)
    say(f"HistGradientBoosting rule (n_iter_no_change=3, tol=0, baseline {BASELINE}): "
        f"stops after round {hgb_stop}")
    assert hgb_stop == stop


# ---------------------------------------------------------------------------
# 3. Defaults and early-stopping APIs of the installed libraries
# ---------------------------------------------------------------------------


def library_facts() -> None:
    header("3. Installed versions, defaults and early-stopping behaviour")
    say(f"scikit-learn {sklearn.__version__}, xgboost {xgb.__version__}, "
        f"lightgbm {lgb.__version__}, catboost {catboost.__version__}")
    X, y = load_adult()
    X_tv, y_tv, X_tr, y_tr, X_va, y_va, X_te, y_te = split("adult", X, y, 0)

    m = xgb.XGBClassifier(n_jobs=THREADS).fit(X_tr, y_tr)
    tp = json.loads(m.get_booster().save_config())["learner"]["gradient_booster"]["tree_train_param"]
    say("XGBClassifier defaults (from save_config):",
        {k: tp[k] for k in ["eta", "max_depth", "max_leaves", "min_child_weight", "subsample",
                            "colsample_bytree", "reg_lambda", "reg_alpha", "gamma"]},
        "rounds =", m.get_booster().num_boosted_rounds())
    try:
        xgb.XGBClassifier(n_jobs=THREADS).fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                                               early_stopping_rounds=10, verbose=False)
        say("XGBoost fit(early_stopping_rounds=...) accepted")
    except TypeError as e:
        say(f"XGBoost fit(early_stopping_rounds=...) -> TypeError: {e}")

    m = xgb.XGBClassifier(n_estimators=MAX_ROUNDS, learning_rate=ES_LR, early_stopping_rounds=PATIENCE,
                          eval_metric="logloss", n_jobs=THREADS, random_state=0)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    full = m.get_booster().num_boosted_rounds()
    p_auto = m.predict_proba(X_te)[:, 1]
    p_best = m.predict_proba(X_te, iteration_range=(0, m.best_iteration + 1))[:, 1]
    p_all = m.predict_proba(X_te, iteration_range=(0, full))[:, 1]
    say(f"XGBoost ES: best_iteration={m.best_iteration} (0-based), trees kept={full}; "
        f"predict uses best: {np.allclose(p_auto, p_best)}; "
        f"test log loss best={log_loss(y_te, p_best):.4f} all trees={log_loss(y_te, p_all):.4f}")

    lm = lgb.LGBMClassifier(verbose=-1, n_jobs=THREADS).fit(X_tr, y_tr)
    say("LGBMClassifier params:", {k: v for k, v in lm.get_params().items()
                                    if k in ["learning_rate", "n_estimators", "num_leaves", "max_depth",
                                             "min_child_samples", "min_child_weight", "subsample",
                                             "subsample_freq", "colsample_bytree", "reg_alpha",
                                             "reg_lambda", "min_split_gain"]})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        lgb.LGBMClassifier(n_estimators=20, verbose=-1, n_jobs=THREADS).fit(
            X_tr, y_tr, eval_set=[(X_va, y_va)])
        msgs = sorted({str(x.message) for x in w if "eval" in str(x.message)})
    say("LightGBM fit(eval_set=...) warnings:", msgs)
    lm = lgb.LGBMClassifier(n_estimators=MAX_ROUNDS, learning_rate=ES_LR, verbose=-1,
                            n_jobs=THREADS, random_state=0)
    lm.fit(X_tr, y_tr, eval_X=X_va, eval_y=y_va, callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
    say(f"LightGBM ES: best_iteration_={lm.best_iteration_} (1-based), "
        f"trees in booster={lm.booster_.current_iteration()}")

    cb = catboost.CatBoostClassifier(thread_count=THREADS, verbose=0, random_seed=0)
    cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=PATIENCE)
    p = cb.get_all_params()
    say("CatBoost (eval_set + early_stopping_rounds):",
        {k: p.get(k) for k in ["iterations", "learning_rate", "depth", "l2_leaf_reg", "rsm",
                               "subsample", "bootstrap_type", "min_data_in_leaf", "use_best_model",
                               "od_type", "od_wait"]},
        f"best_iteration_={cb.get_best_iteration()} tree_count_={cb.tree_count_}",
        f"test log loss={log_loss(y_te, cb.predict_proba(X_te)[:, 1]):.4f}")

    h = HistGradientBoostingClassifier(early_stopping=True, max_iter=2000, learning_rate=0.3,
                                       random_state=0)
    h.fit(X_tv, y_tv)
    best = int(np.argmax(h.validation_score_))
    say(f"HGB early_stopping=True, max_iter=2000, lr=0.3 (validation_fraction=0.1): n_iter_={h.n_iter_}, "
        f"best validation iteration={best}, trees kept={len(h._predictors)}")
    h2 = HistGradientBoostingClassifier(max_iter=2000, learning_rate=0.3, random_state=0)
    h2.fit(X_tr, y_tr, X_val=X_va, y_val=y_va)
    say(f"HGB fit(X_val=..., y_val=...) with early_stopping='auto': n_iter_={h2.n_iter_}, "
        f"len(validation_score_)={len(h2.validation_score_)}")
    say(f"HGB early_stopping='auto' on {len(X_tv)} rows (>10000) -> n_iter_="
        f"{HistGradientBoostingClassifier(random_state=0).fit(X_tv, y_tv).n_iter_}")


# ---------------------------------------------------------------------------
# 4. How much tuning buys
# ---------------------------------------------------------------------------

XGB_SPACE = {
    "learning_rate": loguniform(0.02, 0.3),
    "max_depth": randint(2, 9),
    "min_child_weight": loguniform(0.5, 30),
    "subsample": uniform(0.5, 0.5),
    "colsample_bytree": uniform(0.5, 0.5),
    "reg_lambda": loguniform(1e-3, 10),
    "reg_alpha": loguniform(1e-3, 10),
    "gamma": loguniform(1e-3, 1),
}
LGB_SPACE = {
    "learning_rate": loguniform(0.02, 0.3),
    "num_leaves": [4, 8, 16, 24, 32, 48, 64, 96, 128],
    "min_child_samples": randint(5, 101),
    "subsample": uniform(0.5, 0.5),
    "colsample_bytree": uniform(0.5, 0.5),
    "reg_lambda": loguniform(1e-3, 10),
    "reg_alpha": loguniform(1e-3, 10),
    "min_split_gain": loguniform(1e-3, 1),
}


def make_model(lib: str, task: str, seed: int, **params):
    if lib == "xgb":
        cls = xgb.XGBClassifier if task == "clf" else xgb.XGBRegressor
        return cls(n_jobs=THREADS, random_state=seed, tree_method="hist", **params)
    cls = lgb.LGBMClassifier if task == "clf" else lgb.LGBMRegressor
    if "subsample" in params:
        params = {**params, "subsample_freq": 1}
    return cls(n_jobs=THREADS, random_state=seed, verbose=-1, **params)


def fit_es(lib: str, task: str, seed: int, X_tr, y_tr, X_va, y_va, **params):
    """Fit with early stopping on (X_va, y_va); return model and rounds used."""
    if lib == "xgb":
        metric = "logloss" if task == "clf" else "rmse"
        m = make_model(lib, task, seed, n_estimators=MAX_ROUNDS, early_stopping_rounds=PATIENCE,
                       eval_metric=metric, **params)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        return m, m.best_iteration + 1
    m = make_model(lib, task, seed, n_estimators=MAX_ROUNDS, **params)
    m.fit(X_tr, y_tr, eval_X=X_va, eval_y=y_va,
          callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
    return m, m.best_iteration_


def run_task(name: str, seed: int, lib: str) -> list[dict]:
    task = TASK[name]
    X, y = DATASETS[name]()
    X_tv, y_tv, X_tr, y_tr, X_va, y_va, X_te, y_te = split(name, X, y, seed)
    rows = []

    def record(config, model, rounds, t0, val=None):
        row = {"data": name, "seed": seed, "lib": lib, "config": config,
               "test": score(task, model, X_te, y_te), "rounds": rounds,
               "val": val, "sec": time.perf_counter() - t0}
        if task == "clf":
            row["auc"] = auc(model, X_te, y_te)
        rows.append(row)

    if lib == "rf":
        t0 = time.perf_counter()
        cls = RandomForestClassifier if task == "clf" else RandomForestRegressor
        m = cls(n_jobs=THREADS, random_state=seed).fit(X_tv, y_tv)
        record("rf_default", m, 100, t0)
        return rows

    # (a) library defaults, fit on train+val
    t0 = time.perf_counter()
    m = make_model(lib, task, seed).fit(X_tv, y_tv)
    record("default", m, 100, t0)

    # (b) learning rate 0.05 + early stopping on the validation split
    t0 = time.perf_counter()
    m, rounds = fit_es(lib, task, seed, X_tr, y_tr, X_va, y_va, learning_rate=ES_LR)
    record("es", m, rounds, t0, val=score(task, m, X_va, y_va))
    t0 = time.perf_counter()
    refit_rounds = int(round(rounds * len(X_tv) / len(X_tr)))
    m = make_model(lib, task, seed, n_estimators=refit_rounds, learning_rate=ES_LR).fit(X_tv, y_tv)
    record("es_refit", m, refit_rounds, t0)

    # (c) random search, early stopping inside each candidate, chosen on validation
    t0 = time.perf_counter()
    space = XGB_SPACE if lib == "xgb" else LGB_SPACE
    best = None
    for params in ParameterSampler(space, N_CANDIDATES, random_state=1000 + seed):
        m, rounds = fit_es(lib, task, seed, X_tr, y_tr, X_va, y_va, **params)
        v = score(task, m, X_va, y_va)
        if best is None or v < best[0]:
            best = (v, params, rounds, m)
    v, params, rounds, m = best
    record("rs", m, rounds, t0, val=v)
    rows[-1]["params"] = {k: (round(float(x), 5) if not isinstance(x, (int, np.integer)) else int(x))
                          for k, x in params.items()}
    t0 = time.perf_counter()
    refit_rounds = int(round(rounds * len(X_tv) / len(X_tr)))
    m = make_model(lib, task, seed, n_estimators=refit_rounds, **params).fit(X_tv, y_tv)
    record("rs_refit", m, refit_rounds, t0)
    return rows


def tuning_gain() -> pd.DataFrame:
    header("4. How much tuning buys (untouched 20% test set, 5 seeds)")
    say("Adult: test log loss (and ROC AUC); California: test RMSE (x $100k). Lower is better.")
    say("Splits per seed: train 64% / validation 16% / test 20%. Random search: "
        f"{N_CANDIDATES} candidates, early stopping (patience {PATIENCE}) on validation.")
    tasks = [(d, s, lib) for d in DATASETS for s in SEEDS for lib in ("xgb", "lgb", "rf")]
    t0 = time.perf_counter()
    results = Parallel(n_jobs=2, verbose=0)(delayed(run_task)(*t) for t in tasks)
    say(f"(section wall time {time.perf_counter() - t0:.0f} s)")
    df = pd.DataFrame([r for rs in results for r in rs])

    for name in DATASETS:
        say()
        say(f"--- {name} ---")
        sub = df[df.data == name]
        for (lib, config), g in sub.groupby(["lib", "config"], sort=False):
            line = (f"{lib:>3s} {config:<10s} test mean {g.test.mean():.4f} sd {g.test.std(ddof=1):.4f}"
                    f"  [{', '.join(f'{v:.4f}' for v in g.test)}]  rounds median {g.rounds.median():.0f}"
                    f"  sec mean {g.sec.mean():.1f}")
            if "auc" in g and g.auc.notna().all():
                line += f"  AUC mean {g.auc.mean():.4f} sd {g.auc.std(ddof=1):.4f}"
            say(line)
        for lib in ("xgb", "lgb"):
            piv = sub[sub.lib == lib].pivot(index="seed", columns="config", values="test")
            for a, b in [("default", "es"), ("default", "rs"), ("es", "rs"), ("es", "es_refit"),
                         ("rs", "rs_refit"), ("default", "rs_refit")]:
                d = piv[b] - piv[a]
                say(f"{lib} {b} - {a}: mean {d.mean():+.4f} sd {d.std(ddof=1):.4f} "
                    f"(better in {(d < 0).sum()}/5 seeds)")
            # optimism of the validation score that picked the model
            for config in ("es", "rs"):
                g = sub[(sub.lib == lib) & (sub.config == config)]
                gap = g.val - g.test
                say(f"{lib} {config}: validation score {g.val.mean():.4f} vs test {g.test.mean():.4f}"
                    f" -> val - test mean {gap.mean():+.4f} sd {gap.std(ddof=1):.4f}"
                    f" (val better than test in {(gap < 0).sum()}/5)")
        rs = sub[sub.config == "rs"]
        for _, r in rs.iterrows():
            say(f"  chosen {r.lib} seed {r.seed}: rounds {r.rounds}, {r.params}")
    return df


# ---------------------------------------------------------------------------
# 5. Rounds chosen by early stopping vs learning rate
# ---------------------------------------------------------------------------

LRS = [0.3, 0.1, 0.05, 0.02]


def rounds_vs_lr() -> pd.DataFrame:
    header("5. Rounds chosen by early stopping vs learning rate (seed 0, other params default)")
    rows = []
    for name in DATASETS:
        task = TASK[name]
        X, y = DATASETS[name]()
        X_tv, y_tv, X_tr, y_tr, X_va, y_va, X_te, y_te = split(name, X, y, 0)
        for lib in ("xgb", "lgb"):
            for lr in LRS:
                t0 = time.perf_counter()
                m, rounds = fit_es(lib, task, 0, X_tr, y_tr, X_va, y_va, learning_rate=lr)
                rows.append({"data": name, "lib": lib, "lr": lr, "rounds": rounds,
                             "test": score(task, m, X_te, y_te), "sec": time.perf_counter() - t0})
                say(f"{name:<10s} {lib} lr={lr:<5} rounds={rounds:>5d} test={rows[-1]['test']:.4f} "
                    f"lr*rounds={lr * rounds:.1f} sec={rows[-1]['sec']:.1f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Grid vs random search with the same budget
# ---------------------------------------------------------------------------

GV_ROUNDS = 300
LR_RANGE = (0.003, 0.3)
L2_RANGE = (1e-3, 10.0)


def grid_vs_random() -> dict:
    header(f"6. Grid vs random search, 9 fits each (California, LightGBM, {GV_ROUNDS} rounds fixed)")
    X, y = load_california()
    X_tv, y_tv, X_tr, y_tr, X_va, y_va, X_te, y_te = split("california", X, y, 0)

    def evaluate(lr, l2):
        m = make_model("lgb", "reg", 0, n_estimators=GV_ROUNDS, learning_rate=lr, reg_lambda=l2)
        m.fit(X_tr, y_tr)
        return score("reg", m, X_va, y_va), score("reg", m, X_te, y_te)

    grid_lr = np.geomspace(*LR_RANGE, 3)
    grid_l2 = np.geomspace(*L2_RANGE, 3)
    grid = [(lr, l2, *evaluate(lr, l2)) for lr in grid_lr for l2 in grid_l2]
    for lr, l2, v, t in grid:
        say(f"grid lr={lr:.4f} l2={l2:.3f}: val {v:.4f} test {t:.4f}")
    g_best = min(grid, key=lambda r: r[2])
    say(f"grid best: lr={g_best[0]:.4f} l2={g_best[1]:.3f} val {g_best[2]:.4f} test {g_best[3]:.4f}")

    def one_random(rep):
        rng = np.random.default_rng(100 + rep)
        lrs = np.exp(rng.uniform(np.log(LR_RANGE[0]), np.log(LR_RANGE[1]), 9))
        l2s = np.exp(rng.uniform(np.log(L2_RANGE[0]), np.log(L2_RANGE[1]), 9))
        trials = [(lr, l2, *evaluate(lr, l2)) for lr, l2 in zip(lrs, l2s)]
        return trials

    reps = Parallel(n_jobs=2)(delayed(one_random)(r) for r in range(30))
    best_val = np.array([min(t, key=lambda r: r[2])[2] for t in reps])
    best_test = np.array([min(t, key=lambda r: r[2])[3] for t in reps])
    say(f"random (30 repeats of 9): best val mean {best_val.mean():.4f} sd {best_val.std(ddof=1):.4f}"
        f" min {best_val.min():.4f} max {best_val.max():.4f}")
    say(f"random best test mean {best_test.mean():.4f} sd {best_test.std(ddof=1):.4f}")
    say(f"random beats grid on val in {(best_val < g_best[2]).sum()}/30 repeats; "
        f"on test in {(best_test < g_best[3]).sum()}/30")

    # Sensitivity of the validation RMSE to each dimension (1-D sweeps at the other's grid middle)
    lr_sweep = np.geomspace(*LR_RANGE, 15)
    l2_sweep = np.geomspace(*L2_RANGE, 15)
    lr_curve = [evaluate(lr, grid_l2[1])[0] for lr in lr_sweep]
    l2_curve = [evaluate(0.1, l2)[0] for l2 in l2_sweep]
    say(f"1-D sweep lr (l2={grid_l2[1]:.3f}): val RMSE range {min(lr_curve):.4f}-{max(lr_curve):.4f}")
    say(f"1-D sweep l2 (lr=0.1): val RMSE range {min(l2_curve):.4f}-{max(l2_curve):.4f}")
    return {"grid": grid, "reps": reps, "g_best": g_best, "best_val": best_val,
            "lr_sweep": lr_sweep, "lr_curve": lr_curve, "l2_sweep": l2_sweep, "l2_curve": l2_curve}


# ---------------------------------------------------------------------------
# 7. Successive halving vs random search on the same candidates
# ---------------------------------------------------------------------------


def halving() -> None:
    header("7. HalvingRandomSearchCV vs RandomizedSearchCV, same 27 candidates (Adult, LightGBM)")
    X, y = load_adult()
    X_tv, y_tv, X_tr, y_tr, X_va, y_va, X_te, y_te = split("adult", X, y, 0)
    space = {k: v for k, v in LGB_SPACE.items()}
    base = lgb.LGBMClassifier(n_estimators=300, subsample_freq=1, n_jobs=THREADS, verbose=-1,
                              random_state=0)
    cv = StratifiedKFold(3, shuffle=True, random_state=0)
    common = dict(scoring="neg_log_loss", cv=cv, random_state=0, n_jobs=1)

    t0 = time.perf_counter()
    rs = RandomizedSearchCV(base, space, n_iter=27, **common).fit(X_tv, y_tv)
    t_rs = time.perf_counter() - t0
    t0 = time.perf_counter()
    hs = HalvingRandomSearchCV(base, space, n_candidates=27, factor=3, resource="n_samples",
                               min_resources="exhaust", **common).fit(X_tv, y_tv)
    t_hs = time.perf_counter() - t0
    same = sorted(map(str, rs.cv_results_["params"])) == sorted(
        map(str, [p for p, it in zip(hs.cv_results_["params"], hs.cv_results_["iter"]) if it == 0]))
    say(f"same 27 candidates sampled: {same}")
    say(f"halving iterations: n_resources={hs.n_resources_}, n_candidates={hs.n_candidates_}")
    rank_rs = pd.Series(-rs.cv_results_["mean_test_score"]).rank().to_numpy()
    params_rs = list(map(str, rs.cv_results_["params"]))
    hs_best = str(hs.best_params_)
    say(f"RandomizedSearchCV: {t_rs:.0f} s, fits={27 * 3}, best CV log loss {-rs.best_score_:.4f}, "
        f"test log loss {log_loss(y_te, rs.predict_proba(X_te)[:, 1]):.4f}")
    say(f"HalvingRandomSearchCV: {t_hs:.0f} s, best CV log loss (last round) {-hs.best_score_:.4f}, "
        f"test log loss {log_loss(y_te, hs.predict_proba(X_te)[:, 1]):.4f}")
    say(f"halving's choice ranks #{int(rank_rs[params_rs.index(hs_best)])} of 27 in the full random search")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_grid_vs_random(gv: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=PAPER)
    ax = axes[0]
    style_axes(ax)
    g = np.array([1 / 6, 1 / 2, 5 / 6])
    gx, gy = np.meshgrid(g, g)
    rng = np.random.default_rng(3)
    rx, ry = rng.random(9), rng.random(9)
    ax.axvspan(0.62, 0.72, color=TEAL_LIGHT, alpha=0.6, lw=0)
    ax.scatter(gx.ravel(), gy.ravel(), s=60, marker="s", color=GREY, edgecolor=INK_2, label="grid 3 x 3", zorder=3)
    ax.scatter(rx, ry, s=60, color=TEAL, label="random 9", zorder=3)
    for v in g:
        ax.plot([v, v], [-0.10, -0.05], color=INK_2, lw=2.5)
    for v in rx:
        ax.plot([v, v], [-0.18, -0.13], color=TEAL, lw=2.5)
    ax.axhline(0, color=GRID, lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.21, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("important hyperparameter  (shaded = best 10%; ticks below = values tried)")
    ax.set_ylabel("unimportant hyperparameter")
    ax.set_title("Same 9 trials: 3 vs 9 distinct values", fontsize=10)
    ax.legend(loc="upper left", fontsize=8, facecolor=PAPER, edgecolor=GRID)

    ax = axes[1]
    style_axes(ax)
    for t in gv["reps"]:
        a = np.array([[r[0], r[2]] for r in t])
        ax.scatter(a[:, 0], a[:, 1], s=8, color=TEAL, alpha=0.25, lw=0)
    ga = np.array([[r[0], r[2]] for r in gv["grid"]])
    ax.scatter(ga[:, 0], ga[:, 1], s=55, marker="s", color=GREY, edgecolor=INK_2, zorder=3, label="grid trials")
    ax.plot(gv["lr_sweep"], gv["lr_curve"], color=INK, lw=1.2, label="sweep of learning rate")
    ax.axhline(gv["g_best"][2], color=INK_2, ls="--", lw=1, label=f"grid best {gv['g_best'][2]:.3f}")
    ax.axhline(np.median(gv["best_val"]), color=TEAL, ls="--", lw=1,
               label=f"random best (median of 30) {np.median(gv['best_val']):.3f}")
    ax.set_xscale("log")
    ax.set_ylim(min(ga[:, 1].min(), min(gv["lr_curve"])) - 0.02, 0.9)
    ax.set_xlabel("learning rate (300 rounds, LightGBM)")
    ax.set_ylabel("validation RMSE, California")
    ax.set_title("Measured: 9 grid vs 30 x 9 random trials", fontsize=10)
    ax.legend(fontsize=7.5, facecolor=PAPER, edgecolor=GRID, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "grid-vs-random.png", dpi=160, facecolor=PAPER)
    plt.close(fig)


def fig_rounds_vs_lr(rl: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), facecolor=PAPER)
    for ax, name, ylab in zip(axes, DATASETS, ["test log loss, Adult", "test RMSE, California"]):
        style_axes(ax)
        ax2 = ax.twinx()
        for lib, color, mk in [("xgb", TEAL, "o"), ("lgb", AMBER, "s")]:
            s = rl[(rl.data == name) & (rl.lib == lib)]
            ax.plot(s.lr, s.rounds, color=color, marker=mk, label=f"{lib.upper()} rounds")
            ax2.plot(s.lr, s.test, color=color, marker=mk, ls=":", alpha=0.8, label=f"{lib.upper()} {ylab.split(',')[0]}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(LRS)
        ax.set_xticklabels([str(v) for v in LRS])
        ax.minorticks_off()
        ax.set_xlabel("learning rate")
        ax.set_ylabel("rounds chosen by early stopping (log)")
        ax2.set_ylabel(ylab, color=INK_2)
        ax2.tick_params(colors=INK_2, labelsize=8)
        for side in ("top",):
            ax2.spines[side].set_visible(False)
        ax.set_title(name.capitalize(), fontsize=10)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7.5, facecolor=PAPER, edgecolor=GRID, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rounds-vs-learning-rate.png", dpi=160, facecolor=PAPER)
    plt.close(fig)


def fig_tuning_gain(df: pd.DataFrame) -> None:
    order = [("rf", "rf_default", "RF defaults"),
             ("xgb", "default", "XGB defaults"), ("xgb", "es", "XGB lr 0.05 + ES"),
             ("xgb", "es_refit", "  + refit on train+val"), ("xgb", "rs", "XGB random search"),
             ("xgb", "rs_refit", "  + refit on train+val"),
             ("lgb", "default", "LGBM defaults"), ("lgb", "es", "LGBM lr 0.05 + ES"),
             ("lgb", "es_refit", "  + refit on train+val"), ("lgb", "rs", "LGBM random search"),
             ("lgb", "rs_refit", "  + refit on train+val")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.4), facecolor=PAPER)
    for ax, name, xlab in zip(axes, DATASETS, ["test log loss, Adult (lower is better)",
                                              "test RMSE, California (lower is better)"]):
        style_axes(ax)
        xlim = (0.266, 0.290) if name == "adult" else None
        for i, (lib, config, label) in enumerate(order):
            s = df[(df.data == name) & (df.lib == lib) & (df.config == config)].sort_values("seed")
            if config.startswith("rs"):
                color = TEAL
            elif config.startswith("es"):
                color = INK_2
            else:
                color = GREY
            ys = np.full(len(s), i) + np.linspace(-0.18, 0.18, len(s))
            if xlim is not None and s.test.min() > xlim[1]:
                ax.annotate(f"RF mean {s.test.mean():.3f} (off scale) →", xy=(xlim[1], i),
                            xytext=(xlim[1] - 0.0005, i), ha="right", va="center", fontsize=8, color=INK_2)
                continue
            ax.scatter(s.test, ys, s=22, color=color, edgecolor=INK_2 if color == GREY else color, zorder=3)
            ax.plot([s.test.mean()] * 2, [i - 0.32, i + 0.32], color=INK, lw=1.5)
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([o[2] for o in order] if name == "adult" else [""] * len(order), fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlabel(xlab)
        ax.set_title(f"{name.capitalize()}: 5 seeds (dots), mean (bar)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "tuning-gain.png", dpi=160, facecolor=PAPER)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    raw = HERE / "raw_results.pkl"
    if "--figures-only" in sys.argv:
        # Redraw the figures from the raw numbers saved by the last full run.
        df, rl, gv = pd.read_pickle(raw)
    else:
        start = time.perf_counter()
        hand_grid_vs_random()
        hand_early_stopping()
        library_facts()
        df = tuning_gain()
        rl = rounds_vs_lr()
        gv = grid_vs_random()
        halving()
        pd.to_pickle((df, rl, gv), raw)
        header(f"Total wall time {time.perf_counter() - start:.0f} s")
        (HERE / "results.txt").write_text(out.getvalue(), encoding="utf-8")
    fig_grid_vs_random(gv)
    fig_rounds_vs_lr(rl)
    fig_tuning_gain(df)


if __name__ == "__main__":
    main()
