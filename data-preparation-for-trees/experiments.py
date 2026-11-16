"""Experiments for the "data-preparation-for-trees" blog post.

Everything the post calls "measured" comes from this script:

1. Scaling.  The 6-row toy data shared with the random-forest / decision-tree
   posts is fitted raw, standardised and log-transformed: the training
   partition is identical, but a threshold placed at a midpoint moves in the
   original units after a non-linear transform.  Then six models are fitted
   on Adult with the numeric columns raw / standardised / log1p and their
   test predictions are compared.  Plus two things scaling does NOT cover:
   a target transform and a feature combination (x1 - x2).
2. Missing values.  What each library does with NaN (tiny fits), a hand
   example of XGBoost's learned default direction checked against xgboost,
   and native NaN vs mean / median imputation vs imputation + indicator on
   Adult under MCAR and under target-dependent missingness.
3. Categorical encoding.  A hand example (ordinal vs one-hot vs a native
   partition split), one-hot / ordinal / target / native encodings on Adult
   for five libraries (accuracy, AUC, fit time as the median of repeats),
   TargetEncoder cross fitting, unseen categories at predict time, and
   whether RandomForest / XGBoost accept categorical columns.

Run:    .venv/Scripts/python.exe data-preparation-for-trees/experiments.py
Output: data-preparation-for-trees/results.txt and results.json
        (figures.py draws the PNG figures from results.json).
"""

from __future__ import annotations

import json
import math
import os
import platform
import statistics
import time
import warnings
from pathlib import Path

import catboost
import lightgbm
import numpy as np
import pandas as pd
import sklearn
import xgboost
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "data_cache" / "openml"
SEED = 42
THREADS = 4          # same thread budget for every library
REPEATS = 3          # timing repeats, median reported

lines: list[str] = []
R: dict = {}


def log(msg: str = "") -> None:
    print(msg, flush=True)
    lines.append(msg)


def gini(y) -> float:
    y = np.asarray(y)
    if len(y) == 0:
        return 0.0
    p = y.mean()
    return 1.0 - p**2 - (1 - p) ** 2


def weighted_gini(left, right) -> float:
    n = len(left) + len(right)
    return len(left) / n * gini(left) + len(right) / n * gini(right)


log(f"python {platform.python_version()}, scikit-learn {sklearn.__version__}, "
    f"xgboost {xgboost.__version__}, lightgbm {lightgbm.__version__}, "
    f"catboost {catboost.__version__}, numpy {np.__version__}, pandas {pd.__version__}")
log(f"{os.cpu_count()} logical CPUs, {THREADS} threads per model, "
    f"timings = median of {REPEATS} fits")
log()

# =========================================================================== 1. scaling
log("=" * 78)
log("1. SCALING")
log("=" * 78)

# 6-row toy data of the random-forest / decision-tree posts
X_toy = np.array([[1, 3], [2, 1], [3, 4], [4, 1], [5, 5], [6, 4]], dtype=float)
y_toy = np.array([0, 0, 1, 0, 1, 1])
x1 = X_toy[:, :1]
versions = {
    "raw": (x1, lambda v: v, lambda v: v),
    "standardised": None,
    "log": (np.log(x1), np.log, np.exp),
}
sc = StandardScaler().fit(x1)
versions["standardised"] = (sc.transform(x1), lambda v: sc.transform(np.reshape(v, (-1, 1))).ravel(),
                            lambda v: sc.inverse_transform(np.reshape(v, (-1, 1))).ravel())

log("\n1a. x1-only tree on the 6-row toy data (the deep tree of the decision-tree post)")
toy_scaling = {}
for name, (Xv, fwd, inv) in versions.items():
    t = DecisionTreeClassifier(random_state=42).fit(Xv, y_toy)
    thr = t.tree_.threshold[t.tree_.feature >= 0]
    leaves = t.apply(Xv)
    back = inv(np.array(thr)) if name != "raw" else thr
    toy_scaling[name] = {"thresholds": thr.tolist(), "thresholds_in_hours": np.ravel(back).tolist(),
                         "leaf_of_rows": leaves.tolist(), "train_pred": t.predict(Xv).tolist()}
    log(f"  {name:<13} thresholds {np.round(thr, 4).tolist()}  -> in hours "
        f"{np.round(np.ravel(back), 4).tolist()}  leaf ids of rows 1-6 {leaves.tolist()}")
same_partition = all(toy_scaling[k]["leaf_of_rows"] == toy_scaling["raw"]["leaf_of_rows"] for k in toy_scaling)
log(f"  same partition of the 6 training rows in all three versions: {same_partition}")
assert same_partition

# a new student between two training values
probe = np.array([2.47])
log("  new row with x1 = 2.47 hours:")
probe_pred = {}
for name, (Xv, fwd, inv) in versions.items():
    t = DecisionTreeClassifier(random_state=42).fit(Xv, y_toy)
    p = t.predict(np.reshape(fwd(probe), (-1, 1)))[0]
    probe_pred[name] = int(p)
    log(f"    {name:<13} predicts {p}")
toy_scaling["probe_2.47"] = probe_pred
R["toy_scaling"] = toy_scaling

# both features: x2 <= 3.5 single split under any monotone transform of x2
t_raw = DecisionTreeClassifier(random_state=42).fit(X_toy, y_toy)
t_log = DecisionTreeClassifier(random_state=42).fit(np.log(X_toy), y_toy)
log(f"  both features, raw: feature {t_raw.tree_.feature[0]} thr {t_raw.tree_.threshold[0]:.4f}; "
    f"log: feature {t_log.tree_.feature[0]} thr {t_log.tree_.threshold[0]:.4f} "
    f"(exp = {math.exp(t_log.tree_.threshold[0]):.4f})")

# ---------------------------------------------------------------- Adult data
adult = fetch_openml("adult", version=2, as_frame=True, data_home=str(CACHE))
X_all = adult.data.copy()
y_all = (adult.target == ">50K").astype(int).to_numpy()
NUM = [c for c in X_all.columns if X_all[c].dtype.kind in "if"]
CAT = [c for c in X_all.columns if c not in NUM]
X_all[NUM] = X_all[NUM].astype(float)
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.25, stratify=y_all, random_state=SEED)
X_tr = X_tr.reset_index(drop=True)
X_te = X_te.reset_index(drop=True)

log()
log(f"Adult (OpenML 'adult' v2): {X_all.shape[0]} rows, {len(NUM)} numeric + {len(CAT)} categorical, "
    f"positive rate {y_all.mean():.4f}")
log(f"  numeric: {NUM}")
log("  categorical (levels, missing cells): " + ", ".join(
    f"{c} ({X_all[c].nunique()}, {int(X_all[c].isna().sum())})" for c in CAT))
log(f"  stratified split 75/25 seed {SEED}: train {len(y_tr)}, test {len(y_te)}")
R["adult"] = {"rows": int(X_all.shape[0]), "num": NUM, "cat": CAT,
              "levels": {c: int(X_all[c].nunique()) for c in CAT},
              "missing": {c: int(X_all[c].isna().sum()) for c in X_all.columns},
              "train": int(len(y_tr)), "test": int(len(y_te)), "pos_rate": float(y_all.mean())}

# fixed ordinal codes for the categoricals (missing -> -1) so only numeric columns change
ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-2, encoded_missing_value=-1)
C_tr = ord_enc.fit_transform(X_tr[CAT])
C_te = ord_enc.transform(X_te[CAT])


def num_version(kind: str):
    a, b = X_tr[NUM].to_numpy(), X_te[NUM].to_numpy()
    if kind == "raw":
        return a, b
    if kind == "standardised":
        s = StandardScaler().fit(a)
        return s.transform(a), s.transform(b)
    if kind == "log1p":
        return np.log1p(a), np.log1p(b)
    raise ValueError(kind)


def scaling_models():
    return {
        "DecisionTree": lambda: DecisionTreeClassifier(random_state=0),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=THREADS),
        "HistGB": lambda: HistGradientBoostingClassifier(random_state=0),
        "XGBoost": lambda: XGBClassifier(random_state=0, n_jobs=THREADS),
        "LightGBM": lambda: LGBMClassifier(random_state=0, n_jobs=THREADS, verbose=-1),
        "CatBoost": lambda: CatBoostClassifier(iterations=300, random_seed=0, thread_count=THREADS, verbose=0),
    }


log("\n1b. Adult: numeric columns raw / standardised / log1p (categoricals fixed ordinal codes)")
log("    compare test predictions against the raw-feature model of the same library")
scaling_adult = {}
mats = {}
for kind in ("raw", "standardised", "log1p"):
    a, b = num_version(kind)
    mats[kind] = (np.hstack([a, C_tr]), np.hstack([b, C_te]))

# how many test numeric values are not seen in training (only those can land between thresholds)
unseen_cells = {c: int((~X_te[c].isin(set(X_tr[c]))).sum()) for c in NUM}
log(f"    test cells whose numeric value never occurs in training: {unseen_cells}")
R["adult_unseen_numeric_cells"] = unseen_cells

for mname, make in scaling_models().items():
    base = make().fit(*[mats["raw"][0], y_tr])
    p0 = base.predict_proba(mats["raw"][1])[:, 1]
    p0_tr = base.predict_proba(mats["raw"][0])[:, 1]
    row = {"auc_raw": roc_auc_score(y_te, p0), "acc_raw": accuracy_score(y_te, p0 >= 0.5)}
    for kind in ("standardised", "log1p"):
        m = make().fit(mats[kind][0], y_tr)
        p = m.predict_proba(mats[kind][1])[:, 1]
        p_tr = m.predict_proba(mats[kind][0])[:, 1]
        row[kind] = {
            "max_abs_diff_test": float(np.max(np.abs(p - p0))),
            "n_label_changes_test": int(np.sum((p >= 0.5) != (p0 >= 0.5))),
            "n_rows_prob_changed_test": int(np.sum(np.abs(p - p0) > 1e-12)),
            "max_abs_diff_train": float(np.max(np.abs(p_tr - p0_tr))),
            "auc": roc_auc_score(y_te, p), "acc": accuracy_score(y_te, p >= 0.5),
        }
    scaling_adult[mname] = row
    s, l = row["standardised"], row["log1p"]
    log(f"  {mname:<13} AUC raw {row['auc_raw']:.4f} | std: AUC {s['auc']:.4f}, max|dp| test "
        f"{s['max_abs_diff_test']:.2e} train {s['max_abs_diff_train']:.2e}, rows changed {s['n_rows_prob_changed_test']}, "
        f"labels changed {s['n_label_changes_test']} | log1p: AUC {l['auc']:.4f}, max|dp| test "
        f"{l['max_abs_diff_test']:.2e} train {l['max_abs_diff_train']:.2e}, rows changed {l['n_rows_prob_changed_test']}, "
        f"labels changed {l['n_label_changes_test']}")
R["scaling_adult"] = scaling_adult

log("\n1c. What scaling invariance does NOT cover")
# target transform on the 6-row regression data of the xgboost post
xr = np.arange(1, 7, dtype=float).reshape(-1, 1)
yr = np.array([10, 12, 11, 40, 42, 41], dtype=float)
t_y = DecisionTreeRegressor(max_depth=1).fit(xr, yr)
t_logy = DecisionTreeRegressor(max_depth=1).fit(xr, np.log(yr))
pred_y = t_y.predict(xr)
pred_logy = np.exp(t_logy.predict(xr))
log(f"  target transform, depth-1 tree: on y thr {t_y.tree_.threshold[0]} leaves {np.unique(pred_y).round(4).tolist()};"
    f" on log(y) thr {t_logy.tree_.threshold[0]} leaves exp(.) {np.unique(pred_logy).round(4).tolist()}")
# skewed target where the split itself moves
ys = np.array([1, 2, 3, 4, 5, 100], dtype=float)
xs = np.arange(1, 7, dtype=float).reshape(-1, 1)
t_s = DecisionTreeRegressor(max_depth=1).fit(xs, ys)
t_sl = DecisionTreeRegressor(max_depth=1).fit(xs, np.log(ys))
log(f"  skewed y={ys.tolist()}: split on y at {t_s.tree_.threshold[0]}, on log(y) at {t_sl.tree_.threshold[0]}")
R["target_transform"] = {
    "y": yr.tolist(), "thr_y": float(t_y.tree_.threshold[0]), "leaves_y": np.unique(pred_y).tolist(),
    "thr_logy": float(t_logy.tree_.threshold[0]), "leaves_logy": np.unique(pred_logy).tolist(),
    "skewed_y": ys.tolist(), "skewed_thr_y": float(t_s.tree_.threshold[0]),
    "skewed_thr_logy": float(t_sl.tree_.threshold[0])}

# feature combination: y = [x1 > x2]
rng = np.random.default_rng(0)
Xd = rng.uniform(size=(4000, 2))
yd = (Xd[:, 0] > Xd[:, 1]).astype(int)
Xd_tr, Xd_te, yd_tr, yd_te = Xd[:2000], Xd[2000:], yd[:2000], yd[2000:]
comb = {}
for depth in (2, 4, 8, None):
    a = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(Xd_tr, yd_tr)
    Xc_tr = np.column_stack([Xd_tr, Xd_tr[:, 0] - Xd_tr[:, 1]])
    Xc_te = np.column_stack([Xd_te, Xd_te[:, 0] - Xd_te[:, 1]])
    b = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(Xc_tr, yd_tr)
    comb[str(depth)] = {"raw_acc": a.score(Xd_te, yd_te), "raw_leaves": int(a.get_n_leaves()),
                        "diff_acc": b.score(Xc_te, yd_te), "diff_leaves": int(b.get_n_leaves())}
    log(f"  y = [x1 > x2], max_depth={depth}: x1,x2 only acc {a.score(Xd_te, yd_te):.4f} ({a.get_n_leaves()} leaves);"
        f" + (x1 - x2) acc {b.score(Xc_te, yd_te):.4f} ({b.get_n_leaves()} leaves)")
R["feature_combination"] = comb

# =========================================================================== 2. missing values
log()
log("=" * 78)
log("2. MISSING VALUES")
log("=" * 78)

log("\n2a. What each library does with NaN (tiny fit, feature x = [1, 2, NaN, 4, 5, NaN], y = [0, 0, 1, 0, 1, 1])")
xn = np.array([[1.0], [2.0], [np.nan], [4.0], [5.0], [np.nan]])
yn = np.array([0, 0, 1, 0, 1, 1])
nan_libs = {
    "DecisionTreeClassifier": lambda: DecisionTreeClassifier(random_state=0),
    "RandomForestClassifier": lambda: RandomForestClassifier(n_estimators=5, random_state=0),
    "GradientBoostingClassifier": lambda: GradientBoostingClassifier(),
    "HistGradientBoostingClassifier": lambda: HistGradientBoostingClassifier(min_samples_leaf=1, max_iter=5),
    "XGBClassifier": lambda: XGBClassifier(n_estimators=5),
    "LGBMClassifier": lambda: LGBMClassifier(n_estimators=5, min_child_samples=1, min_data_in_bin=1, verbose=-1),
    "CatBoostClassifier": lambda: CatBoostClassifier(iterations=5, verbose=0),
}
nan_support = {}
for name, make in nan_libs.items():
    try:
        m = make().fit(xn, yn)
        out = f"fits; P(y=1 | x=NaN) = {m.predict_proba(np.array([[np.nan]]))[0, 1]:.3f}"
        if name == "CatBoostClassifier":
            out += f"; nan_mode = {m.get_all_params()['nan_mode']}"
        nan_support[name] = {"ok": True, "detail": out}
    except Exception as e:  # noqa: BLE001
        msg = str(e).splitlines()[0][:110]
        nan_support[name] = {"ok": False, "detail": msg}
        out = f"ERROR: {msg}"
    log(f"  {name:<32} {out}")
# a DecisionTree that saw no NaN in training
dt_clean = DecisionTreeClassifier(random_state=0).fit(np.array([[0.], [1.], [2.], [3.]]), [0, 1, 1, 1])
log(f"  DecisionTree trained without NaN, predict(NaN) -> {dt_clean.predict(np.array([[np.nan]]))[0]} "
    f"(docs: goes to the child with the most samples)")
R["nan_support"] = nan_support

# CatBoost categorical NaN
try:
    CatBoostClassifier(iterations=5, verbose=0, cat_features=[0]).fit(
        pd.DataFrame({"c": ["A", "A", None, "B", "C", "C"]}), yn)
    cb_cat_nan = "fits"
except Exception as e:  # noqa: BLE001
    cb_cat_nan = str(e).splitlines()[0][:160]
log(f"  CatBoost with None in a cat_features column: {cb_cat_nan}")
R["catboost_cat_nan"] = cb_cat_nan

log("\n2b. Hand example: XGBoost learned default direction")
# xgboost post data, row 4 has its x missing
xh = np.array([1, 2, 3, np.nan, 5, 6], dtype=float)
yh = np.array([10, 12, 11, 40, 42, 41], dtype=float)
F0 = yh.mean()
g = F0 - yh
h = np.ones_like(yh)
lam = 0.0
miss = np.isnan(xh)
G_m, H_m = g[miss].sum(), h[miss].sum()
vals = np.unique(xh[~miss])
hand_rows = []
G, H = g.sum(), h.sum()


def score(Gs, Hs):
    return Gs**2 / (Hs + lam)


for a_, b_ in zip(vals[:-1], vals[1:]):
    thr = (a_ + b_) / 2
    left = (~miss) & (xh < thr)
    right = (~miss) & (xh >= thr)
    GL, HL, GR, HR = g[left].sum(), h[left].sum(), g[right].sum(), h[right].sum()
    gain_right = 0.5 * (score(GL, HL) + score(GR + G_m, HR + H_m) - score(G, H))
    gain_left = 0.5 * (score(GL + G_m, HL + H_m) + score(GR, HR) - score(G, H))
    hand_rows.append({"thr": float(thr), "GL": float(GL), "HL": float(HL), "GR": float(GR), "HR": float(HR),
                      "gain_nan_left": float(gain_left), "gain_nan_right": float(gain_right)})
log(f"  F0 = {F0}, g = {g.tolist()}, G_missing = {G_m}, H_missing = {H_m}")
for r in hand_rows:
    log(f"  x < {r['thr']:<4} non-missing GL={r['GL']:>5} HL={r['HL']} GR={r['GR']:>5} HR={r['HR']}  "
        f"gain NaN->left {r['gain_nan_left']:8.3f}  NaN->right {r['gain_nan_right']:8.3f}")
best = max(hand_rows, key=lambda r: max(r["gain_nan_left"], r["gain_nan_right"]))
best_dir = "right" if best["gain_nan_right"] >= best["gain_nan_left"] else "left"
best_gain = max(best["gain_nan_left"], best["gain_nan_right"])
log(f"  hand: best split x < {best['thr']}, NaN goes {best_dir}, gain {best_gain}")

dm = xgboost.DMatrix(xh.reshape(-1, 1), label=yh, feature_names=["hours"])
bst = xgboost.train(dict(objective="reg:squarederror", max_depth=1, eta=1.0, reg_lambda=0.0, gamma=0.0,
                         min_child_weight=0.0, tree_method="exact", base_score=float(F0)), dm, num_boost_round=1)
tree = json.loads(bst.get_dump(dump_format="json", with_stats=True)[0])
children = {c["nodeid"]: c for c in tree["children"]}
xgb_dir = "left" if tree["missing"] == tree["yes"] else "right"
xgb_pred = bst.predict(dm)
log(f"  xgboost {xgboost.__version__} exact: split {tree['split']} < {tree['split_condition']}, "
    f"missing -> node {tree['missing']} ({xgb_dir}), gain {tree['gain']}, leaves "
    f"{children[tree['yes']]['leaf']} / {children[tree['no']]['leaf']}, predictions {xgb_pred.round(4).tolist()}")
assert abs(tree["split_condition"] - best["thr"]) < 1e-6
assert xgb_dir == best_dir
assert abs(tree["gain"] - 2 * best_gain) < 1e-6   # xgboost stores gain without the 1/2
R["xgb_nan_hand"] = {"F0": F0, "g": g.tolist(), "G_missing": float(G_m), "rows": hand_rows,
                     "best_thr": best["thr"], "best_dir": best_dir, "best_gain": best_gain,
                     "xgb_split": tree["split_condition"], "xgb_dir": xgb_dir, "xgb_gain": tree["gain"],
                     "xgb_leaves": [children[tree["yes"]]["leaf"], children[tree["no"]]["leaf"]],
                     "xgb_pred": xgb_pred.tolist()}

log("\n2c. Adult: native NaN vs imputation, MCAR vs target-dependent missingness")
MISS_COLS = ["age", "education-num", "hours-per-week", "capital-gain"]
P_MCAR = 0.30
P_POS, P_NEG = 0.40, 0.15     # target-dependent: P(missing | y=1), P(missing | y=0)
MISS_SEEDS = [0, 1, 2]
log(f"  inject NaN into {MISS_COLS} of train AND test, each cell independently:")
log(f"    MCAR: p = {P_MCAR};  target-dependent: p = {P_POS} if y = 1 else {P_NEG}")
log(f"    seeds {MISS_SEEDS} (numpy default_rng); categoricals: fixed ordinal codes, missing -> -1")


def inject(Xnum: pd.DataFrame, y, mech: str, seed: int) -> pd.DataFrame:
    rng_ = np.random.default_rng(seed)
    out = Xnum.copy()
    for c in MISS_COLS:
        if mech == "mcar":
            p = np.full(len(out), P_MCAR)
        else:
            p = np.where(y == 1, P_POS, P_NEG)
        out.loc[rng_.uniform(size=len(out)) < p, c] = np.nan
    return out


def miss_models():
    return {
        "HistGB": lambda: HistGradientBoostingClassifier(random_state=0),
        "XGBoost": lambda: XGBClassifier(random_state=0, n_jobs=THREADS),
        "LightGBM": lambda: LGBMClassifier(random_state=0, n_jobs=THREADS, verbose=-1),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=THREADS),
    }


STRATS = {
    "native NaN": None,
    "mean impute": lambda: SimpleImputer(strategy="mean"),
    "median impute": lambda: SimpleImputer(strategy="median"),
    "median + indicator": lambda: SimpleImputer(strategy="median", add_indicator=True),
}

miss_res: dict = {"baseline": {}, "mcar": {}, "target": {}}
base_tr = np.hstack([X_tr[NUM].to_numpy(), C_tr])
base_te = np.hstack([X_te[NUM].to_numpy(), C_te])
for mname, make in miss_models().items():
    p = make().fit(base_tr, y_tr).predict_proba(base_te)[:, 1]
    miss_res["baseline"][mname] = {"auc": roc_auc_score(y_te, p), "acc": accuracy_score(y_te, p >= 0.5)}
    log(f"  no injected NaN  {mname:<13} AUC {miss_res['baseline'][mname]['auc']:.4f} "
        f"acc {miss_res['baseline'][mname]['acc']:.4f}")

for mech in ("mcar", "target"):
    rates = []
    for mname in miss_models():
        miss_res[mech][mname] = {s: {"auc": [], "acc": []} for s in STRATS}
    for seed in MISS_SEEDS:
        ntr = inject(X_tr[NUM], y_tr, mech, seed)
        nte = inject(X_te[NUM], y_te, mech, seed + 100)
        rates.append(float(ntr[MISS_COLS].isna().to_numpy().mean()))
        A_tr = np.hstack([ntr.to_numpy(), C_tr])
        A_te = np.hstack([nte.to_numpy(), C_te])
        for mname, make in miss_models().items():
            for sname, imp in STRATS.items():
                model = make() if imp is None else make_pipeline(imp(), make())
                p = model.fit(A_tr, y_tr).predict_proba(A_te)[:, 1]
                miss_res[mech][mname][sname]["auc"].append(roc_auc_score(y_te, p))
                miss_res[mech][mname][sname]["acc"].append(accuracy_score(y_te, p >= 0.5))
    miss_res[mech + "_rate"] = statistics.mean(rates)
    log(f"\n  mechanism = {mech}: mean missing rate in the 4 columns (train) {statistics.mean(rates):.4f}")
    log(f"  {'model':<13} " + "  ".join(f"{s:>20}" for s in STRATS) + "   (mean AUC over seeds / mean acc)")
    for mname in miss_models():
        cells = []
        for s in STRATS:
            d = miss_res[mech][mname][s]
            cells.append(f"{statistics.mean(d['auc']):.4f}/{statistics.mean(d['acc']):.4f}")
        log(f"  {mname:<13} " + "  ".join(f"{c:>20}" for c in cells))
    for mname in miss_models():
        for s in STRATS:
            d = miss_res[mech][mname][s]
            log(f"    {mech} {mname:<13} {s:<20} AUC per seed {[round(v, 4) for v in d['auc']]}")
R["missing_adult"] = miss_res
R["missing_setup"] = {"cols": MISS_COLS, "p_mcar": P_MCAR, "p_pos": P_POS, "p_neg": P_NEG, "seeds": MISS_SEEDS}

# =========================================================================== 3. categorical
log()
log("=" * 78)
log("3. CATEGORICAL ENCODING")
log("=" * 78)

log("\n3a. Hand example: 4 categories, y = 1 for A and C, 0 for B and D (2 rows each)")
cat_toy = np.array(list("AABBCCDD"))
y_ct = np.array([1, 1, 0, 0, 1, 1, 0, 0])
codes = {"A": 0, "B": 1, "C": 2, "D": 3}
xo = np.array([codes[c] for c in cat_toy], dtype=float)
hand_cat = {"ordinal": [], "onehot": [], "native": []}
for t in (0.5, 1.5, 2.5):
    L, Rr = y_ct[xo <= t], y_ct[xo > t]
    hand_cat["ordinal"].append({"split": f"code <= {t}", "G": weighted_gini(L, Rr)})
for c in "ABCD":
    L, Rr = y_ct[cat_toy == c], y_ct[cat_toy != c]
    hand_cat["onehot"].append({"split": f"is_{c}", "G": weighted_gini(L, Rr)})
means = {c: y_ct[cat_toy == c].mean() for c in "ABCD"}
order = sorted("ABCD", key=lambda c: (means[c], c))
for k in range(1, 4):
    left = set(order[:k])
    mask = np.array([c in left for c in cat_toy])
    hand_cat["native"].append({"split": "{" + ",".join(sorted(left)) + "} vs rest",
                               "G": weighted_gini(y_ct[mask], y_ct[~mask])})
for k, rows in hand_cat.items():
    log(f"  {k:<8} " + "; ".join(f"{r['split']}: G_split {r['G']:.4f}" for r in rows))
# fully grown sklearn trees on ordinal and one-hot
t_ord = DecisionTreeClassifier(random_state=0).fit(xo.reshape(-1, 1), y_ct)
oh = OneHotEncoder(sparse_output=False).fit(cat_toy.reshape(-1, 1))
t_oh = DecisionTreeClassifier(random_state=0).fit(oh.transform(cat_toy.reshape(-1, 1)), y_ct)
log(f"  sklearn full tree, ordinal codes: depth {t_ord.get_depth()}, leaves {t_ord.get_n_leaves()}; "
    f"one-hot: depth {t_oh.get_depth()}, leaves {t_oh.get_n_leaves()}")
# xgboost native partition split
dfc = pd.DataFrame({"c": pd.Categorical(cat_toy)})
xn_cat = XGBClassifier(n_estimators=1, max_depth=1, learning_rate=1.0, reg_lambda=0.0, min_child_weight=0.0,
                       enable_categorical=True, max_cat_to_onehot=1, base_score=0.5).fit(dfc, y_ct)
tj = json.loads(xn_cat.get_booster().get_dump(dump_format="json", with_stats=True)[0])
cats_yes = [list(dfc["c"].cat.categories)[i] for i in tj["split_condition"]]
xgb_acc = float((xn_cat.predict(dfc) == y_ct).mean())
log(f"  xgboost enable_categorical, max_cat_to_onehot=1, depth 1: categories {cats_yes} go one way; "
    f"train accuracy {xgb_acc:.4f}")
xd_oh = XGBClassifier(n_estimators=1, max_depth=1, learning_rate=1.0, reg_lambda=0.0, min_child_weight=0.0,
                      enable_categorical=True, base_score=0.5).fit(dfc, y_ct)
tj2 = json.loads(xd_oh.get_booster().get_dump(dump_format="json", with_stats=True)[0])
log(f"  xgboost default max_cat_to_onehot (4 categories <= 4 -> one-vs-rest): split_condition "
    f"{tj2['split_condition']}, train accuracy {float((xd_oh.predict(dfc) == y_ct).mean()):.4f}")
lg_cat = LGBMClassifier(n_estimators=1, max_depth=1, num_leaves=2, learning_rate=1.0, min_child_samples=1,
                        min_data_per_group=1, cat_smooth=0, min_data_in_bin=1, max_cat_to_onehot=1,
                        verbose=-1).fit(dfc, y_ct)
lg_acc = float((lg_cat.predict(dfc) == y_ct).mean())
hg_cat = HistGradientBoostingClassifier(max_iter=1, max_depth=1, learning_rate=1.0, min_samples_leaf=1,
                                        early_stopping=False).fit(dfc, y_ct)
hg_acc = float((hg_cat.predict(dfc) == y_ct).mean())
log(f"  one depth-1 tree, native categorical: LightGBM train acc {lg_acc:.4f}, HistGB train acc {hg_acc:.4f}")
R["cat_hand"] = {"rows": hand_cat, "ord_depth": int(t_ord.get_depth()), "ord_leaves": int(t_ord.get_n_leaves()),
                 "oh_depth": int(t_oh.get_depth()), "oh_leaves": int(t_oh.get_n_leaves()),
                 "xgb_partition": cats_yes, "xgb_acc": xgb_acc, "xgb_default_split": tj2["split_condition"],
                 "lgb_acc": lg_acc, "hgb_acc": hg_acc}

log("\n3b. Adult: one-hot / ordinal / target / native encodings")


def enc_onehot():
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)],
                             remainder="passthrough")


def enc_ordinal():
    return ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-2,
                                                     encoded_missing_value=-1), CAT)], remainder="passthrough")


def enc_target():
    return ColumnTransformer([("cat", TargetEncoder(target_type="binary"), CAT)], remainder="passthrough")


ENCODERS = {"one-hot": enc_onehot, "ordinal": enc_ordinal, "target": enc_target}

X_tr_cb = X_tr.copy()
X_te_cb = X_te.copy()
for c in CAT:
    X_tr_cb[c] = X_tr_cb[c].astype(object).where(X_tr_cb[c].notna(), "NA").astype(str)
    X_te_cb[c] = X_te_cb[c].astype(object).where(X_te_cb[c].notna(), "NA").astype(str)

MODELS = {
    "RandomForest": lambda: RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=THREADS),
    "HistGB": lambda: HistGradientBoostingClassifier(random_state=0),
    "XGBoost": lambda: XGBClassifier(random_state=0, n_jobs=THREADS),
    "LightGBM": lambda: LGBMClassifier(random_state=0, n_jobs=THREADS, verbose=-1),
    "CatBoost": lambda: CatBoostClassifier(random_seed=0, thread_count=THREADS, verbose=0),
}
NATIVE = {
    "HistGB": (lambda: HistGradientBoostingClassifier(random_state=0, categorical_features="from_dtype"), "pandas"),
    "XGBoost": (lambda: XGBClassifier(random_state=0, n_jobs=THREADS, enable_categorical=True), "pandas"),
    "LightGBM": (lambda: LGBMClassifier(random_state=0, n_jobs=THREADS, verbose=-1), "pandas"),
    "CatBoost": (lambda: CatBoostClassifier(random_seed=0, thread_count=THREADS, verbose=0,
                                            cat_features=CAT), "catboost"),
}
RUN_ENC = {
    "RandomForest": ["one-hot", "ordinal", "target"],
    "HistGB": ["one-hot", "ordinal", "target", "native"],
    "XGBoost": ["one-hot", "ordinal", "target", "native"],
    "LightGBM": ["one-hot", "ordinal", "target", "native"],
    "CatBoost": ["one-hot", "ordinal", "native"],
}


def build(mname, enc):
    if enc == "native":
        make, kind = NATIVE[mname]
        return make(), kind
    return make_pipeline(ENCODERS[enc](), MODELS[mname]()), "pandas"


enc_res: dict = {}
log(f"  {'model':<13} {'encoding':<9} {'cols':>5} {'acc':>7} {'AUC':>7} {'fit s (median)':>15}  all fits")
for mname, encs in RUN_ENC.items():
    enc_res[mname] = {}
    for enc in encs:
        times, aucs, accs = [], [], []
        ncols = None
        for _ in range(REPEATS):
            model, kind = build(mname, enc)
            Xa, Xb = (X_tr_cb, X_te_cb) if kind == "catboost" else (X_tr, X_te)
            t0 = time.perf_counter()
            model.fit(Xa, y_tr)
            times.append(time.perf_counter() - t0)
            p = model.predict_proba(Xb)[:, 1]
            aucs.append(roc_auc_score(y_te, p))
            accs.append(accuracy_score(y_te, p >= 0.5))
            if ncols is None:
                ncols = (model[0].transform(Xa.head(2)).shape[1] if enc != "native" else Xa.shape[1])
        same = max(aucs) - min(aucs) < 1e-12
        enc_res[mname][enc] = {"acc": accs[0], "auc": aucs[0], "time": statistics.median(times),
                               "times": times, "cols": int(ncols), "deterministic": bool(same)}
        log(f"  {mname:<13} {enc:<9} {ncols:>5} {accs[0]:7.4f} {aucs[0]:7.4f} {statistics.median(times):15.3f}  "
            f"{[round(t, 3) for t in times]}{'' if same else '  (AUC varies across repeats!)'}")
R["encoding_adult"] = enc_res

log("\n3c. TargetEncoder cross fitting: 1,000 rows, a unique id per row, random y")
rng = np.random.default_rng(1)
ids = pd.DataFrame({"id": [f"id{i}" for i in range(1000)]})
y_rand = rng.integers(0, 2, size=1000)
te = TargetEncoder(target_type="binary")
z_cross = te.fit_transform(ids, y_rand)
z_plain = TargetEncoder(target_type="binary").fit(ids, y_rand).transform(ids)
st = DecisionTreeClassifier(max_depth=1, random_state=0)
acc_cross = st.fit(z_cross, y_rand).score(z_cross, y_rand)
acc_plain = st.fit(z_plain, y_rand).score(z_plain, y_rand)
log(f"  fit_transform (5-fold cross fitting) -> depth-1 tree train acc {acc_cross:.4f}; "
    f"fit().transform() -> {acc_plain:.4f}; equal arrays? {np.allclose(z_cross, z_plain)}")
R["te_crossfit"] = {"acc_cross": acc_cross, "acc_plain": acc_plain}

log("\n3d. A category never seen in training appears at predict time")
small_tr = pd.DataFrame({"c": pd.Categorical(list("AABBCCDD") * 5), "n": np.arange(40, dtype=float)})
small_y = np.array([1, 1, 0, 0, 1, 1, 0, 0] * 5)
new_row = pd.DataFrame({"c": pd.Categorical(["E"]), "n": [3.0]})
unseen = {}


def try_pred(name, fn):
    try:
        out = fn()
        unseen[name] = f"no error, P(y=1) = {out:.3f}"
    except Exception as e:  # noqa: BLE001
        unseen[name] = "ERROR: " + str(e).strip().splitlines()[0][:150]
    log(f"  {name:<40} {unseen[name]}")


try_pred("OneHotEncoder() default", lambda: make_pipeline(
    ColumnTransformer([("c", OneHotEncoder(), ["c"])], remainder="passthrough"),
    DecisionTreeClassifier(random_state=0)).fit(small_tr, small_y).predict_proba(new_row)[0, 1])
try_pred("OneHotEncoder(handle_unknown='ignore')", lambda: make_pipeline(
    ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), ["c"])], remainder="passthrough"),
    DecisionTreeClassifier(random_state=0)).fit(small_tr, small_y).predict_proba(new_row)[0, 1])
try_pred("OrdinalEncoder() default", lambda: make_pipeline(
    ColumnTransformer([("c", OrdinalEncoder(), ["c"])], remainder="passthrough"),
    DecisionTreeClassifier(random_state=0)).fit(small_tr, small_y).predict_proba(new_row)[0, 1])
try_pred("TargetEncoder()", lambda: make_pipeline(
    ColumnTransformer([("c", TargetEncoder(), ["c"])], remainder="passthrough"),
    DecisionTreeClassifier(random_state=0)).fit(small_tr, small_y).predict_proba(new_row)[0, 1])
try_pred("HistGB native", lambda: HistGradientBoostingClassifier(max_iter=5, min_samples_leaf=2).fit(
    small_tr, small_y).predict_proba(new_row)[0, 1])
try_pred("LightGBM native", lambda: LGBMClassifier(n_estimators=5, min_child_samples=2, verbose=-1).fit(
    small_tr, small_y).predict_proba(new_row)[0, 1])
try_pred("XGBoost enable_categorical", lambda: XGBClassifier(n_estimators=5, enable_categorical=True).fit(
    small_tr, small_y).predict_proba(new_row)[0, 1])
try_pred("CatBoost cat_features", lambda: CatBoostClassifier(iterations=5, verbose=0, cat_features=["c"]).fit(
    small_tr.assign(c=small_tr["c"].astype(str)), small_y).predict_proba(new_row.assign(c=["E"]))[0, 1])
R["unseen_category"] = unseen

log("\n3e. Do the forests / XGBoost statements of the sibling posts still hold?")
checks = {}
try:
    RandomForestClassifier(n_estimators=5).fit(X_tr[CAT].head(200), y_tr[:200])
    checks["RandomForest with raw string categories"] = "fits"
except Exception as e:  # noqa: BLE001
    checks["RandomForest with raw string categories"] = "ERROR: " + str(e).splitlines()[0][:120]
try:
    XGBClassifier(n_estimators=5).fit(X_tr.head(200), y_tr[:200])
    checks["XGBoost pandas category, enable_categorical not set"] = "fits"
except Exception as e:  # noqa: BLE001
    checks["XGBoost pandas category, enable_categorical not set"] = "ERROR: " + str(e).splitlines()[0][:120]
cfg = json.loads(XGBClassifier(n_estimators=1, enable_categorical=True).fit(X_tr.head(200), y_tr[:200])
                 .get_booster().save_config())
tp = cfg["learner"]["gradient_booster"]["tree_train_param"]
checks["XGBoost defaults read from the trained model config"] = (
    f"tree_method={cfg['learner']['gradient_booster']['gbtree_train_param']['tree_method']}, "
    f"max_cat_to_onehot={tp['max_cat_to_onehot']}, max_cat_threshold={tp['max_cat_threshold']}")
for k, v in checks.items():
    log(f"  {k:<55} {v}")
R["statement_checks"] = checks

(HERE / "results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
(HERE / "results.json").write_text(json.dumps(R, indent=1, default=float), encoding="utf-8")
log("\nwrote results.txt and results.json")
