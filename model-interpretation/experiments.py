"""Model-interpretation experiments for the "model-interpretation" blog post.

Everything the post calls "measured" comes from this script:

1. Impurity importance (MDI) by hand on the 6-row toy data shared with the
   decision-tree and random-forest posts, asserted against scikit-learn's
   feature_importances_ for two trees that both fit the data perfectly.
2. Permutation importance on the same toy trees: the exact expectation over
   all 720 orderings, checked against sklearn.inspection.permutation_importance.
3. Shapley values by hand for a two-feature tree (all coalitions), both the
   path-dependent ("tree cover") value function and the interventional one,
   asserted against XGBoost pred_contribs; then brute-force Shapley values
   checked against the built-in TreeSHAP of XGBoost, LightGBM and CatBoost on
   small real models (including CatBoost's interventional mode).
4. Adult (OpenML v2): MDI vs permutation (train / test) in a random forest
   with two pure-noise canary columns, over several seeds.
5. Adult: one XGBoost model ranked by weight / gain / cover / total_gain /
   permutation / mean |SHAP|, rank agreement, stability over seeds; the same
   for LightGBM split / gain.
6. California housing: what correlated copies of a feature do to MDI, gain,
   permutation importance and SHAP.
7. California housing: partial dependence + ICE, and how many PDP grid points
   fall outside the region covered by real data.

Run:  .venv/Scripts/python.exe model-interpretation/experiments.py
Output: model-interpretation/results.txt and three PNGs in model-interpretation/figures/.
Every model is capped at 4 threads.
"""

from __future__ import annotations

import io
import itertools
import json
import math
import tempfile
import time
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
from scipy.spatial import cKDTree  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.datasets import fetch_california_housing, fetch_openml  # noqa: E402
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # noqa: E402
from sklearn.inspection import PartialDependenceDisplay, partial_dependence  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import r2_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, export_text  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE = ROOT / "data_cache"
FIG_DIR = HERE / "figures"
THREADS = 4
SEED = 42

PAPER = "#F7F5F0"
INK = "#1F2328"
INK_2 = "#5B6270"
GRID = "#E6E0D4"
TEAL = "#0F766E"
GREY = "#A3A8B0"

out = io.StringIO()


def say(*args) -> None:
    print(*args, flush=True)
    print(*args, file=out)


def header(title: str) -> None:
    say()
    say("=" * 78)
    say(title)
    say("=" * 78)


def close(a, b, tol=1e-6) -> bool:
    return bool(np.allclose(a, b, atol=tol, rtol=0))


# ===========================================================================
# Generic tree representation + brute-force Shapley values
# ===========================================================================
# A tree is a dict node: leaves {"value": v, "cover": c}; internal nodes
# {"feature": j, "go_left": callable(x_j) -> bool, "left": ..., "right": ...,
#  "cover": c}. go_left must also handle NaN the way the library does.

def tree_predict(node, x):
    while "value" not in node:
        node = node["left"] if node["go_left"](x[node["feature"]]) else node["right"]
    return node["value"]


def v_path(node, x, S):
    """Path-dependent value function (Lundberg et al. 2018/2020, Algorithm 1):
    follow x for features in S, otherwise average the children by cover."""
    if "value" in node:
        return node["value"]
    if node["feature"] in S:
        child = node["left"] if node["go_left"](x[node["feature"]]) else node["right"]
        return v_path(child, x, S)
    lc, rc = node["left"]["cover"], node["right"]["cover"]
    if lc + rc == 0:  # empty subtree (possible in CatBoost oblivious trees)
        return EMPTY_SUBTREE(node, x, S)
    return (lc * v_path(node["left"], x, S) + rc * v_path(node["right"], x, S)) / (lc + rc)


def _empty_mean(node, x, S):
    return 0.5 * (v_path(node["left"], x, S) + v_path(node["right"], x, S))


EMPTY_SUBTREE = _empty_mean


def v_interventional(trees, x, S, background):
    """Interventional value function: features in S from x, the rest from each
    background row, prediction averaged over the background."""
    Z = background.copy()
    for j in S:
        Z[:, j] = x[j]
    return np.mean([sum(tree_predict(t, z) for t in trees) for z in Z])


def shapley(value_fn, M):
    """Exact Shapley values by enumerating every coalition of M features."""
    phi = np.zeros(M)
    for j in range(M):
        others = [k for k in range(M) if k != j]
        for size in range(M):
            w = math.factorial(size) * math.factorial(M - size - 1) / math.factorial(M)
            for S in itertools.combinations(others, size):
                phi[j] += w * (value_fn(set(S) | {j}) - value_fn(set(S)))
    return phi


def shap_path(trees, x, M):
    return shapley(lambda S: sum(v_path(t, x, S) for t in trees), M)


# ---------------------------------------------------------------- parsers ---
def xgb_trees(booster, cover_key="Cover"):
    df = booster.trees_to_dataframe()
    trees = []
    for _, g in df.groupby("Tree", sort=True):
        rows = {r.ID: r for r in g.itertuples()}

        def build(node_id):
            r = rows[node_id]
            if r.Feature == "Leaf":
                return {"value": float(r.Gain), "cover": float(getattr(r, cover_key))}
            j = int(r.Feature[1:]) if r.Feature.startswith("f") else names_idx[r.Feature]
            thr, miss_left = float(r.Split), r.Missing == r.Yes
            return {"feature": j, "cover": float(getattr(r, cover_key)),
                    "go_left": (lambda v, t=thr, m=miss_left: m if np.isnan(v) else v < t),
                    "left": build(r.Yes), "right": build(r.No)}
        names_idx = {n: i for i, n in enumerate(booster.feature_names or [])}
        trees.append(build(g.iloc[0].ID))
    return trees


def lgb_trees(booster, cover="count"):
    df = booster.trees_to_dataframe()
    names_idx = {n: i for i, n in enumerate(booster.feature_name())}
    trees = []
    for _, g in df.groupby("tree_index", sort=True):
        rows = {r.node_index: r for r in g.itertuples()}

        def build(idx):
            r = rows[idx]
            c = float(r.count if cover == "count" else r.weight)
            if r.left_child is None or (isinstance(r.left_child, float) and np.isnan(r.left_child)):
                return {"value": float(r.value), "cover": c}
            assert r.decision_type == "<="
            thr = float(r.threshold)
            miss_left = r.missing_direction == "left"
            return {"feature": names_idx[r.split_feature], "cover": c,
                    "go_left": (lambda v, t=thr, m=miss_left: m if np.isnan(v) else v <= t),
                    "left": build(r.left_child), "right": build(r.right_child)}
        root = g[g.node_depth == 1].iloc[0].node_index
        trees.append(build(root))
    return trees


def catboost_trees(model, reverse=False):
    """Convert CatBoost oblivious trees (float features only) to binary trees.
    Bit i of the leaf index is 1 when x[f_i] > border_i. `reverse` chooses
    whether the first or the last split in the JSON is the root."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.json"
        model.save_model(str(p), format="json")
        js = json.loads(p.read_text(encoding="utf-8"))
    scale, bias = js["scale_and_bias"]
    bias = bias[0] if isinstance(bias, list) else bias
    trees = []
    for t in js["oblivious_trees"]:
        splits = t.get("splits", [])
        vals, wts = t["leaf_values"], t["leaf_weights"]
        depth = len(splits)
        order = list(range(depth))[::-1] if reverse else list(range(depth))

        def build(level, idx_bits):
            if level == depth:
                leaf = sum(b << i for i, b in idx_bits.items())
                return {"value": scale * vals[leaf], "cover": float(wts[leaf])}
            i = order[level]
            s = splits[i]
            thr = float(s["border"])
            left = build(level + 1, {**idx_bits, i: 0})
            right = build(level + 1, {**idx_bits, i: 1})
            return {"feature": s["float_feature_index"], "cover": left["cover"] + right["cover"],
                    "go_left": (lambda v, t=thr: not (v > t)), "left": left, "right": right}
        trees.append(build(0, {}))
    return trees, bias


# ===========================================================================
# Part 1-2: toy data, MDI and permutation importance by hand
# ===========================================================================
X_TOY = np.array([[1, 3], [2, 1], [3, 4], [4, 1], [5, 5], [6, 4]], dtype=float)
Y_TOY = np.array([0, 0, 1, 0, 1, 1])


def gini(y):
    if len(y) == 0:
        return 0.0
    p = np.bincount(y, minlength=2) / len(y)
    return 1.0 - float((p ** 2).sum())


def mdi_by_hand(tree, X, y):
    """Sum of N_t/N * (impurity decrease) per feature, normalised, walking the
    fitted sklearn tree but recomputing every Gini from the rows."""
    t = tree.tree_
    imp = np.zeros(X.shape[1])
    lines = []

    def walk(node, rows):
        if t.children_left[node] == -1:
            return
        j, thr = t.feature[node], t.threshold[node]
        L, R = rows[X[rows, j] <= thr], rows[X[rows, j] > thr]
        g, gl, gr = gini(y[rows]), gini(y[L]), gini(y[R])
        dec = len(rows) / len(y) * (g - len(L) / len(rows) * gl - len(R) / len(rows) * gr)
        imp[j] += dec
        lines.append(f"  node x{j + 1} <= {thr:.1f}: n={len(rows)} G={g:.4f} "
                     f"G_L={gl:.4f} G_R={gr:.4f} weighted decrease={dec:.4f}")
        walk(t.children_left[node], L)
        walk(t.children_right[node], R)
    walk(0, np.arange(len(y)))
    return imp / imp.sum(), lines


def exact_perm_drop(tree, X, y, j):
    """Expected accuracy drop when column j is shuffled: average over all n! orders."""
    base = (tree.predict(X) == y).mean()
    accs = []
    for perm in itertools.permutations(range(len(y))):
        Xp = X.copy()
        Xp[:, j] = X[list(perm), j]
        accs.append((tree.predict(Xp) == y).mean())
    return base - float(np.mean(accs))


def part1_2():
    header("PART 1  MDI by hand on the 6-row toy data")
    tree_a = DecisionTreeClassifier(random_state=SEED).fit(X_TOY, Y_TOY)
    # Find a seed for which max_features=1 puts x1 at the root (x2 then cleans up).
    tree_b, seed_b = None, None
    for s in range(200):
        t = DecisionTreeClassifier(max_features=1, random_state=s).fit(X_TOY, Y_TOY)
        if t.tree_.feature[0] == 0 and t.get_depth() == 2 and t.tree_.n_leaves == 3:
            tree_b, seed_b = t, s
            break
    for name, t in (("tree A (default, random_state=42)", tree_a),
                    (f"tree B (max_features=1, random_state={seed_b})", tree_b)):
        say(f"{name}: train accuracy {t.score(X_TOY, Y_TOY):.4f}")
        say(export_text(t, feature_names=["x1", "x2"]).rstrip())
        hand, lines = mdi_by_hand(t, X_TOY, Y_TOY)
        for ln in lines:
            say(ln)
        say(f"  MDI by hand  : x1={hand[0]:.4f} x2={hand[1]:.4f}")
        say(f"  sklearn      : x1={t.feature_importances_[0]:.4f} x2={t.feature_importances_[1]:.4f}")
        assert close(hand, t.feature_importances_), "MDI hand != sklearn"
        say("  assert hand == sklearn feature_importances_: OK")

    header("PART 2  Permutation importance on the toy trees")
    for name, t in (("tree A", tree_a), ("tree B", tree_b)):
        exact = [exact_perm_drop(t, X_TOY, Y_TOY, j) for j in range(2)]
        pi = permutation_importance(t, X_TOY, Y_TOY, n_repeats=5000, random_state=0)
        say(f"{name}: exact expected accuracy drop over 720 orders: "
            f"x1={exact[0]:.4f} x2={exact[1]:.4f}")
        say(f"{name}: sklearn permutation_importance (5000 repeats): "
            f"x1={pi.importances_mean[0]:.4f} x2={pi.importances_mean[1]:.4f}")
        assert close(exact, pi.importances_mean, tol=0.01)
    # one single shuffle shown by hand in the post
    t = tree_b
    perm = [1, 3, 0, 5, 2, 4]  # an explicit example ordering for the x2 column
    Xp = X_TOY.copy()
    Xp[:, 1] = X_TOY[perm, 1]
    say(f"tree B, x2 shuffled to {Xp[:, 1].astype(int).tolist()}: predictions "
        f"{t.predict(Xp).tolist()} accuracy {(t.predict(Xp) == Y_TOY).mean():.4f}")
    return tree_b


# ===========================================================================
# Part 3: Shapley values by hand + library TreeSHAP checks
# ===========================================================================
def part3():
    header("PART 3a  Shapley values by hand: two-feature regression tree (XGBoost, 1 tree)")
    # One XGBoost tree that reproduces tree B as a regression on y (0/1):
    # eta=1, lambda=0, base_score=0.5, one feature per node so x1 can win the root.
    found = None
    for s in range(200):
        m = xgb.XGBRegressor(n_estimators=1, max_depth=2, learning_rate=1.0, reg_lambda=0.0,
                             min_child_weight=0.0, base_score=0.5, tree_method="hist",
                             colsample_bynode=0.5, random_state=s, n_jobs=1)
        m.fit(X_TOY, Y_TOY)
        df = m.get_booster().trees_to_dataframe()
        feats = df[df.Feature != "Leaf"].Feature.tolist()
        if feats == ["f0", "f1"] and np.allclose(m.predict(X_TOY), Y_TOY):
            found = (s, m)
            break
    s, m = found
    booster = m.get_booster()
    say(f"XGBoost seed {s} (colsample_bynode=0.5) grows:")
    say(booster.get_dump(with_stats=True)[0].rstrip())
    trees = xgb_trees(booster)
    base = 0.5
    x = X_TOY[2]  # row 3: x1=3, x2=4, prediction 1
    vp = {S: base + v_path(trees[0], x, set(S)) for S in [(), (0,), (1,), (0, 1)]}
    bg = X_TOY.copy()
    vi = {S: base + v_interventional(trees, x, set(S), bg) for S in [(), (0,), (1,), (0, 1)]}
    for S in vp:
        say(f"  v({set(S) or '{}'}) path-dependent={vp[S]:.4f} interventional={vi[S]:.4f}")
    phi_p = shap_path(trees, x, 2)
    phi_i = shapley(lambda S: v_interventional(trees, x, S, bg), 2)
    contrib = booster.predict(xgb.DMatrix(x[None, :]), pred_contribs=True)[0]
    say(f"  path-dependent Shapley (brute force): {np.round(phi_p, 4).tolist()} "
        f"sum={phi_p.sum():.4f} = f(x) - E[f] = {vp[(0, 1)] - vp[()]:.4f}")
    say(f"  interventional Shapley (brute force): {np.round(phi_i, 4).tolist()} "
        f"sum={phi_i.sum():.4f} = f(x) - mean f(bg) = {vi[(0, 1)] - vi[()]:.4f}")
    say(f"  XGBoost pred_contribs              : {np.round(contrib, 4).tolist()} "
        f"(last = bias)")
    assert close(phi_p, contrib[:2], 1e-5) and close(contrib[2], vp[()], 1e-5)
    say("  assert path-dependent == XGBoost pred_contribs: OK")
    say("  all 6 rows (path / interventional):")
    for i, xr in enumerate(X_TOY):
        pp = shap_path(trees, xr, 2)
        pi_ = shapley(lambda S: v_interventional(trees, xr, S, bg), 2)
        cc = booster.predict(xgb.DMatrix(xr[None, :]), pred_contribs=True)[0]
        assert close(pp, cc[:2], 1e-5)
        say(f"    row {i + 1} x={xr.astype(int).tolist()} f={tree_predict(trees[0], xr) + base:.0f}"
            f"  path={np.round(pp, 4).tolist()}  interv={np.round(pi_, 4).tolist()}")

    header("PART 3b  Brute-force Shapley vs built-in TreeSHAP on small real models")
    cal = fetch_california_housing(data_home=str(CACHE), as_frame=True)
    cols = ["MedInc", "AveOccup", "Latitude", "Longitude"]
    Xc = cal.frame[cols].to_numpy()
    yc = cal.frame["MedHouseVal"].to_numpy()
    yb = (yc > np.median(yc)).astype(int)
    rng = np.random.default_rng(0)
    ridx = rng.choice(len(Xc), 30, replace=False)
    rows = Xc[ridx]

    def report(lib, trees, contrib, M=4, intercept=0.0):
        errs = []
        for r, c in zip(rows, contrib):
            phi = shap_path(trees, r, M)
            errs.append(np.abs(phi - c[:M]).max())
        say(f"  {lib}: max |brute force - library| over 30 rows = {max(errs):.2e}")
        return max(errs)

    # XGBoost regression and classification (cover = sum of hessians)
    mr = xgb.XGBRegressor(n_estimators=20, max_depth=3, n_jobs=THREADS, random_state=0).fit(Xc, yc)
    cr = mr.get_booster().predict(xgb.DMatrix(rows), pred_contribs=True)
    report("XGBoost regressor  pred_contribs (cover=Cover)", xgb_trees(mr.get_booster()), cr)
    mc = xgb.XGBClassifier(n_estimators=20, max_depth=3, n_jobs=THREADS, random_state=0).fit(Xc, yb)
    cc = mc.get_booster().predict(xgb.DMatrix(rows), pred_contribs=True)
    margin = mc.get_booster().predict(xgb.DMatrix(rows), output_margin=True)
    say(f"  XGBoost classifier: max |sum(contribs) - margin| = {np.abs(cc.sum(1) - margin).max():.2e}"
        f" (log-odds scale)")
    report("XGBoost classifier pred_contribs (cover=Cover, i.e. sum of hessians)",
           xgb_trees(mc.get_booster()), cc)
    # LightGBM: which cover does it use, data count or hessian sum?
    lc = lgb.LGBMClassifier(n_estimators=20, num_leaves=8, n_jobs=THREADS, random_state=0,
                            verbose=-1).fit(pd.DataFrame(Xc, columns=cols), yb)
    lcon = lc.booster_.predict(rows, pred_contrib=True)
    raw = lc.booster_.predict(rows, raw_score=True)
    say(f"  LightGBM classifier: max |sum(contribs) - raw score| = {np.abs(lcon.sum(1) - raw).max():.2e}")
    e_count = report("LightGBM pred_contrib vs brute force with cover=count ",
                     lgb_trees(lc.booster_, "count"), lcon)
    e_weight = report("LightGBM pred_contrib vs brute force with cover=hessian",
                      lgb_trees(lc.booster_, "weight"), lcon)
    say(f"  -> LightGBM TreeSHAP uses {'data count' if e_count < e_weight else 'hessian'} as cover")
    # CatBoost (float features only), path-dependent and interventional
    cb = catboost.CatBoostClassifier(iterations=20, depth=3, thread_count=THREADS, random_seed=0,
                                     verbose=0, allow_writing_files=False).fit(Xc, yb)
    pool = catboost.Pool(rows, label=yb[ridx])  # labels are required with reference_data
    csh = cb.get_feature_importance(data=pool, type="ShapValues")
    rawc = cb.predict(rows, prediction_type="RawFormulaVal")
    say(f"  CatBoost: max |sum(ShapValues) - RawFormulaVal| = {np.abs(csh.sum(1) - rawc).max():.2e}")
    for rev in (False, True):
        trees, bias = catboost_trees(cb, reverse=rev)
        pred_err = max(abs(sum(tree_predict(t, r) for t in trees) + bias - rv)
                       for r, rv in zip(rows, rawc))
        say(f"  CatBoost tree conversion (root = {'last' if rev else 'first'} split): "
            f"max predict error {pred_err:.2e}")
        report(f"CatBoost ShapValues (Regular) vs brute force, root = {'last' if rev else 'first'} split",
               trees, csh)
    # Interventional: CatBoost with reference_data vs brute-force interventional
    bidx = rng.choice(len(Xc), 100, replace=False)
    bgi = Xc[bidx]
    trees, bias = catboost_trees(cb, reverse=True)
    cint = cb.get_feature_importance(data=pool, type="ShapValues",
                                     reference_data=catboost.Pool(bgi, label=yb[bidx]))
    errs, gaps = [], []
    for r, c, cp in zip(rows[:10], cint[:10], csh[:10]):
        phi = shapley(lambda S: v_interventional(trees, r, S, bgi), 4)
        errs.append(np.abs(phi - c[:4]).max())
        gaps.append(np.abs(c[:4] - cp[:4]).max())
    say(f"  CatBoost ShapValues + reference_data (100 rows) vs brute-force interventional: "
        f"max err {max(errs):.2e} over 10 rows")
    say(f"  same 10 rows: max |interventional - path-dependent| = {max(gaps):.4f} log-odds")


# ===========================================================================
# Adult helpers
# ===========================================================================
def load_adult():
    data = fetch_openml("adult", version=2, as_frame=True, data_home=str(CACHE))
    X = data.data.copy()
    y = (data.target == ">50K").astype(int).to_numpy()
    return X, y


def add_canaries(X, rng):
    X = X.copy()
    X["noise_1000"] = rng.integers(0, 1000, len(X)).astype(float)
    X["noise_2"] = rng.integers(0, 2, len(X)).astype(float)
    return X


def codes(X):
    """Categorical columns -> integer codes (NaN kept as NaN) for scikit-learn."""
    Z = X.copy()
    for c in Z.columns:
        if isinstance(Z[c].dtype, pd.CategoricalDtype):
            v = Z[c].cat.codes.astype(float)
            v[v < 0] = np.nan
            Z[c] = v
    return Z


def ranks(values: pd.Series) -> pd.Series:
    return values.rank(ascending=False, method="min").astype(int)


# ===========================================================================
# Part 4: random forest MDI vs permutation with canaries
# ===========================================================================
def part4(X, y, seeds):
    header("PART 4  Adult, random forest: MDI vs permutation importance, two noise canaries")
    say("RandomForestClassifier(n_estimators=100, defaults, n_jobs=4); categoricals as codes.")
    say("Split 75/25 stratified with random_state=seed; noise from default_rng(seed).")
    say("Permutation: ROC AUC drop, 5 repeats, on 12,211 training rows and on the test rows.")
    rows = []
    for s in seeds:
        t0 = time.time()
        Xs = codes(add_canaries(X, np.random.default_rng(s)))
        Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.25, stratify=y, random_state=s)
        rf = RandomForestClassifier(n_estimators=100, random_state=s, n_jobs=THREADS).fit(Xtr, ytr)
        sub = np.random.default_rng(s).choice(len(Xtr), len(Xte), replace=False)
        p_tr = permutation_importance(rf, Xtr.iloc[sub], ytr[sub], scoring="roc_auc",
                                      n_repeats=5, random_state=s, n_jobs=1)
        p_te = permutation_importance(rf, Xte, yte, scoring="roc_auc", n_repeats=5,
                                      random_state=s, n_jobs=1)
        auc_tr = roc_auc_score(ytr[sub], rf.predict_proba(Xtr.iloc[sub])[:, 1])
        auc_te = roc_auc_score(yte, rf.predict_proba(Xte)[:, 1])
        df = pd.DataFrame({"mdi": rf.feature_importances_, "perm_train": p_tr.importances_mean,
                           "perm_test": p_te.importances_mean,
                           "perm_test_sd": p_te.importances_std}, index=Xs.columns)
        rows.append(df)
        say(f"seed {s}: AUC train {auc_tr:.4f} test {auc_te:.4f} ({time.time() - t0:.0f} s)")
        if s == seeds[0]:
            d = df.copy()
            d["rank_mdi"], d["rank_perm_test"] = ranks(d.mdi), ranks(d.perm_test)
            say(d.sort_values("mdi", ascending=False).round(5).to_string())
    say("\nCanaries across seeds (MDI, rank of 16 by MDI, perm train, perm test):")
    for c in ("noise_1000", "noise_2"):
        for s, df in zip(seeds, rows):
            say(f"  {c:<10} seed {s}: MDI {df.loc[c, 'mdi']:.5f} (rank {ranks(df.mdi)[c]:>2}) "
                f"perm_train {df.loc[c, 'perm_train']:+.5f} perm_test {df.loc[c, 'perm_test']:+.5f}")
    mdi = pd.concat([r.mdi for r in rows], axis=1)
    pt = pd.concat([r.perm_test for r in rows], axis=1)
    say("\nRank range across seeds (min-max), MDI vs perm_test:")
    for f in mdi.index:
        rm = mdi.rank(ascending=False, method="min").loc[f]
        rp = pt.rank(ascending=False, method="min").loc[f]
        say(f"  {f:<15} MDI rank {int(rm.min()):>2}-{int(rm.max()):>2}   perm rank "
            f"{int(rp.min()):>2}-{int(rp.max()):>2}   MDI mean {mdi.loc[f].mean():.4f}")
    return rows


# ===========================================================================
# Part 5: one boosted model, many importances
# ===========================================================================
def xgb_importances(model, Xte, yte, s):
    b = model.get_booster()
    names = list(Xte.columns)
    imp = {}
    for t in ("weight", "gain", "cover", "total_gain", "total_cover"):
        sc = b.get_score(importance_type=t)
        imp[t] = pd.Series({n: sc.get(n, 0.0) for n in names})
    pt = permutation_importance(model, Xte, yte, scoring="roc_auc", n_repeats=5,
                                random_state=s, n_jobs=1)
    imp["permutation"] = pd.Series(pt.importances_mean, index=names)
    contrib = b.predict(xgb.DMatrix(Xte, enable_categorical=True), pred_contribs=True)
    imp["mean_abs_shap"] = pd.Series(np.abs(contrib[:, :-1]).mean(0), index=names)
    return pd.DataFrame(imp), contrib


def part5(X, y, seeds):
    header("PART 5  Adult, one XGBoost model ranked six ways (+ LightGBM split vs gain)")
    pairs = X.groupby("education", observed=True)["education-num"].nunique()
    back = X.groupby("education-num")["education"].nunique()
    say(f"education -> education-num is one-to-one: {bool((pairs == 1).all() and (back == 1).all())} "
        f"({len(pairs)} levels)")
    say(f"default importance_type: XGBClassifier={xgb.XGBClassifier().importance_type!r}, "
        f"LGBMClassifier={lgb.LGBMClassifier().importance_type!r}")
    say("XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,")
    say("colsample_bytree=0.8, enable_categorical=True, n_jobs=4, random_state=seed)")
    say("Permutation: ROC AUC drop on the test rows, 5 repeats. SHAP: pred_contribs on test rows.")
    all_ranks, keep = [], None
    for s in seeds:
        Xs = add_canaries(X, np.random.default_rng(s))
        Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.25, stratify=y, random_state=s)
        m = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                              colsample_bytree=0.8, enable_categorical=True, n_jobs=THREADS,
                              random_state=s).fit(Xtr, ytr)
        auc = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
        imp, contrib = xgb_importances(m, Xte, yte, s)
        margin = m.get_booster().predict(xgb.DMatrix(Xte, enable_categorical=True),
                                         output_margin=True)
        R = imp.apply(ranks)
        all_ranks.append(R)
        say(f"\nseed {s}: test AUC {auc:.4f}; max |sum(SHAP) - margin| = "
            f"{np.abs(contrib.sum(1) - margin).max():.2e}")
        if s == seeds[0]:
            keep = (m, Xtr, Xte, ytr, yte, imp, contrib)
            fi = pd.Series(m.feature_importances_, index=Xte.columns)
            for t in ("gain", "weight", "cover", "total_gain"):
                if close(fi, imp[t] / imp[t].sum(), 1e-6):
                    say(f"XGBClassifier.feature_importances_ == normalised '{t}'")
            root_cover = m.get_booster().trees_to_dataframe().query("Tree == 0 and Node == 0").Cover.iloc[0]
            say(f"XGBoost tree 0 root Cover = {root_cover:.1f} vs rows in the subsample "
                f"~{0.8 * len(Xtr):.0f} (classification Cover is a sum of hessians p(1-p))")
            show = imp.copy()
            for c in show.columns:
                show[c] = show[c].map(lambda v: f"{v:.4g}")
            say(show.to_string())
            say("\nRanks (1 = most important):")
            say(R.sort_values("mean_abs_shap").to_string())
            say("\nSpearman rank correlation between methods:")
            say(imp.corr(method="spearman").round(2).to_string())
            say(f"bias column (baseline log-odds) = {contrib[0, -1]:.4f}; "
                f"sigmoid = {1 / (1 + math.exp(-contrib[0, -1])):.4f}; "
                f"train positive rate = {ytr.mean():.4f}")
            i = int(np.argmax(m.predict_proba(Xte)[:, 1] > 0.5))
            row = Xte.iloc[i]
            say(f"\nOne test person (index {Xte.index[i]}): p = {m.predict_proba(Xte.iloc[[i]])[0, 1]:.4f}, "
                f"margin {margin[i]:.4f} = bias {contrib[i, -1]:.4f} + sum of SHAP "
                f"{contrib[i, :-1].sum():.4f}")
            order = np.argsort(-np.abs(contrib[i, :-1]))[:6]
            for j in order:
                say(f"    {Xte.columns[j]:<15} = {str(row.iloc[j]):<22} SHAP {contrib[i, j]:+.4f}")
    say("\nRank range over seeds per method (min-max):")
    stack = pd.concat(all_ranks, keys=seeds)
    rng_tab = pd.DataFrame({c: stack[c].groupby(level=1).agg(lambda v: f"{v.min()}-{v.max()}")
                            for c in stack.columns})
    say(rng_tab.loc[all_ranks[0].sort_values("mean_abs_shap").index].to_string())

    # LightGBM on the seed-0 split: split vs gain vs mean|SHAP|
    m, Xtr, Xte, ytr, yte, imp, contrib = keep
    lm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=31, n_jobs=THREADS,
                            random_state=seeds[0], verbose=-1).fit(Xtr, ytr)
    lb = lm.booster_
    lc = lb.predict(Xte, pred_contrib=True)
    limp = pd.DataFrame({
        "split": lb.feature_importance("split"), "gain": lb.feature_importance("gain"),
        "mean_abs_shap": np.abs(lc[:, :-1]).mean(0)}, index=Xte.columns)
    lpt = permutation_importance(lm, Xte, yte, scoring="roc_auc", n_repeats=5,
                                 random_state=seeds[0], n_jobs=1)
    limp["permutation"] = lpt.importances_mean
    say(f"\nLightGBM (300 trees, num_leaves=31, pandas categories native), seed {seeds[0]}, "
        f"test AUC {roc_auc_score(yte, lm.predict_proba(Xte)[:, 1]):.4f}")
    say(limp.assign(**{f"r_{c}": ranks(limp[c]) for c in limp.columns})
        .sort_values("mean_abs_shap", ascending=False).round(4).to_string())
    return keep


# ===========================================================================
# Part 6: correlated copies (California housing)
# ===========================================================================
def part6():
    header("PART 6  California housing: a correlated copy of MedInc")
    cal = fetch_california_housing(data_home=str(CACHE), as_frame=True)
    X0, y = cal.data, cal.target.to_numpy()
    rng = np.random.default_rng(0)
    variants = {
        "original": X0,
        "exact copy": X0.assign(MedInc_copy=X0.MedInc),
        "noisy copy": X0.assign(MedInc_copy=X0.MedInc + rng.normal(0, 0.25 * X0.MedInc.std(), len(X0))),
    }
    say(f"noisy copy: MedInc + N(0, 0.25 sd); corr = "
        f"{np.corrcoef(variants['noisy copy'].MedInc, variants['noisy copy'].MedInc_copy)[0, 1]:.4f}")
    for name, X in variants.items():
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED)
        xm = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=THREADS,
                              random_state=SEED).fit(Xtr, ytr)
        rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, max_features=0.5,
                                   n_jobs=THREADS, random_state=SEED).fit(Xtr, ytr)
        cols = ["MedInc"] + (["MedInc_copy"] if "MedInc_copy" in X else [])
        tg = xm.get_booster().get_score(importance_type="total_gain")
        tg_share = {c: tg.get(c, 0) / sum(tg.values()) for c in cols}
        sh = np.abs(xm.get_booster().predict(xgb.DMatrix(Xte), pred_contribs=True)[:, :-1]).mean(0)
        sh = dict(zip(X.columns, sh))
        res = {}
        for lab, model in (("xgb", xm), ("rf", rf)):
            base = r2_score(yte, model.predict(Xte))
            pi = permutation_importance(model, Xte, yte, n_repeats=5, random_state=SEED, n_jobs=1)
            res[lab] = dict(zip(X.columns, pi.importances_mean))
            # permute all copies together with one shared ordering
            drops = []
            for k in range(5):
                order = np.random.default_rng(k).permutation(len(Xte))
                Xp = Xte.copy()
                for c in cols:
                    Xp[c] = Xte[c].to_numpy()[order]
                drops.append(base - r2_score(yte, model.predict(Xp)))
            res[lab + "_joint"] = float(np.mean(drops))
            res[lab + "_r2"] = base
        say(f"\n{name}: XGB test R2 {res['xgb_r2']:.4f}, RF test R2 {res['rf_r2']:.4f}")
        for c in cols:
            say(f"  {c:<12} XGB total_gain share {tg_share[c]:.4f}  XGB mean|SHAP| {sh[c]:.4f}  "
                f"XGB perm {res['xgb'][c]:.4f}  RF MDI {dict(zip(X.columns, rf.feature_importances_))[c]:.4f}  "
                f"RF perm {res['rf'][c]:.4f}")
        say(f"  permute {' + '.join(cols)} together: XGB drop {res['xgb_joint']:.4f}  "
            f"RF drop {res['rf_joint']:.4f}")


# ===========================================================================
# Part 7: PDP / ICE and off-support grid points
# ===========================================================================
def part7():
    header("PART 7  California housing: partial dependence, ICE, off-support grid points")
    cal = fetch_california_housing(data_home=str(CACHE), as_frame=True)
    X, y = cal.data, cal.target.to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED)
    m = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=THREADS,
                         random_state=SEED).fit(Xtr, ytr)
    say(f"XGBRegressor 300 trees depth 6, test R2 {r2_score(yte, m.predict(Xte)):.4f}")
    say(f"corr(Latitude, Longitude) = {np.corrcoef(X.Latitude, X.Longitude)[0, 1]:.4f}")
    ice_rows = Xte.sample(300, random_state=SEED)
    pd_med = partial_dependence(m, ice_rows, ["MedInc"], kind="both", grid_resolution=30)
    grid = pd_med["grid_values"][0]
    ice = pd_med["individual"][0]
    avg = pd_med["average"][0]
    say(f"MedInc PDP over grid {grid[0]:.2f}..{grid[-1]:.2f}: {avg[0]:.3f} -> {avg[-1]:.3f}")
    slopes = ice[:, -1] - ice[:, 0]
    say(f"ICE rise from first to last grid point: min {slopes.min():.3f} median "
        f"{np.median(slopes):.3f} max {slopes.max():.3f} (300 rows)")

    # Latitude PDP: how many of the synthetic (lat, lon) points are far from any real block?
    pd_lat = partial_dependence(m, Xte, ["Latitude"], kind="average", grid_resolution=25)
    lat_grid, lat_avg = pd_lat["grid_values"][0], pd_lat["average"][0]
    kd = cKDTree(X[["Latitude", "Longitude"]].to_numpy())
    off = []
    for g in lat_grid:
        pts = np.column_stack([np.full(len(Xte), g), Xte.Longitude.to_numpy()])
        dist, _ = kd.query(pts)
        off.append(float((dist > 0.25).mean()))
    say("Latitude grid | PDP | share of synthetic points > 0.25 deg from every real block")
    for g, a, o in zip(lat_grid, lat_avg, off):
        say(f"  {g:6.2f} | {a:.3f} | {o:.3f}")
    say(f"mean share off-support over the grid: {np.mean(off):.3f}")
    # Sanity: the same distance for the real test rows themselves (excluding self).
    d_real, _ = kd.query(Xte[["Latitude", "Longitude"]].to_numpy(), k=2)
    say(f"real test rows: share > 0.25 deg from their nearest other block = {(d_real[:, 1] > 0.25).mean():.4f}")
    return dict(grid=grid, ice=ice, avg=avg, lat_grid=lat_grid, lat_avg=lat_avg, off=off)


# ===========================================================================
# Figures
# ===========================================================================
def style(ax):
    ax.set_facecolor(PAPER)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK_2)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_ranks(imp, rf_df, path):
    """Rank grid: features x methods, numbers in cells, canaries in teal."""
    tab = imp[["permutation", "mean_abs_shap", "total_gain", "gain", "cover", "weight"]].copy()
    tab.insert(0, "RF MDI", rf_df["mdi"].reindex(tab.index))
    tab.insert(1, "RF permutation", rf_df["perm_test"].reindex(tab.index))
    R = tab.apply(ranks)
    R = R.sort_values("mean_abs_shap")
    labels = ["RF\nMDI", "RF\npermutation", "XGB\npermutation", "XGB\nmean |SHAP|",
              "XGB\ntotal_gain", "XGB\ngain", "XGB\ncover", "XGB\nweight"]
    fig, ax = plt.subplots(figsize=(9.2, 6.6), facecolor=PAPER)
    ax.set_facecolor(PAPER)
    n, k = R.shape
    for i, f in enumerate(R.index):
        canary = f.startswith("noise")
        for j, c in enumerate(R.columns):
            r = R.loc[f, c]
            shade = 1 - (r - 1) / (n - 1)
            light, dark = np.array([0.93, 0.91, 0.87]), np.array([0.23, 0.25, 0.28])
            face = TEAL if canary else tuple(light + (dark - light) * shade)
            ax.add_patch(plt.Rectangle((j, n - 1 - i), 0.94, 0.9, color=face, linewidth=0))
            ax.text(j + 0.47, n - 1 - i + 0.45, str(r), ha="center", va="center", fontsize=9,
                    color="white" if (canary or shade > 0.55) else INK, family="monospace")
        ax.text(-0.1, n - 1 - i + 0.45, f, ha="right", va="center", fontsize=9,
                color=TEAL if canary else INK, fontweight="bold" if canary else "normal")
    for j, lab in enumerate(labels):
        ax.text(j + 0.47, n + 0.25, lab, ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xlim(-2.6, k)
    ax.set_ylim(-0.3, n + 1.4)
    ax.axis("off")
    ax.set_title("Adult: feature rank (1 = most important) by method, seed 42 split",
                 color=INK, fontsize=11, loc="left", x=0.0)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=PAPER)
    plt.close(fig)


def fig_pdp(p, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=PAPER)
    ax = axes[0]
    style(ax)
    for line in p["ice"]:
        ax.plot(p["grid"], line, color=GREY, alpha=0.25, linewidth=0.7)
    ax.plot(p["grid"], p["avg"], color=TEAL, linewidth=2.6, label="PDP (average of ICE)")
    ax.plot([], [], color=GREY, linewidth=1, label="ICE, 300 test blocks")
    ax.set_xlabel("MedInc (median income, $10k)", color=INK)
    ax.set_ylabel("predicted MedHouseVal ($100k)", color=INK)
    ax.set_title("(a) PDP + ICE for MedInc", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK)
    ax = axes[1]
    style(ax)
    ax.plot(p["lat_grid"], p["lat_avg"], color=TEAL, linewidth=2.4, label="PDP of Latitude")
    ax.set_xlabel("Latitude (grid value)", color=INK)
    ax.set_ylabel("predicted MedHouseVal ($100k)", color=TEAL)
    ax2 = ax.twinx()
    ax2.bar(p["lat_grid"], p["off"], width=(p["lat_grid"][1] - p["lat_grid"][0]) * 0.8,
            color=GREY, alpha=0.45, label="share of synthetic points\n>0.25° from any real block")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("share off-support", color=INK_2)
    ax2.tick_params(colors=INK_2, labelsize=9)
    for sp in ("top",):
        ax2.spines[sp].set_visible(False)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
    ax.set_title("(b) Latitude PDP and the impossible points behind it", color=INK,
                 fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=PAPER)
    plt.close(fig)


def fig_shap(Xte, contrib, path, top=10):
    names = np.array(Xte.columns)
    S = contrib[:, :-1]
    order = np.argsort(-np.abs(S).mean(0))[:top][::-1]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(S), 2000, replace=False)
    fig, ax = plt.subplots(figsize=(9, 5.6), facecolor=PAPER)
    style(ax)
    ax.grid(axis="y", visible=False)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("paper", ["#D9D4C7", TEAL])
    for pos, j in enumerate(order):
        col = Xte.iloc[idx, j]
        if isinstance(col.dtype, pd.CategoricalDtype):
            # colour categories by their mean SHAP, so similar effects share a shade
            v = pd.Series(S[idx, j]).groupby(col.cat.codes.to_numpy()).transform("mean").to_numpy()
        else:
            v = col.to_numpy(dtype=float)
        r = pd.Series(v).rank(pct=True).to_numpy()
        r = np.nan_to_num(r, nan=0.5)
        x = S[idx, j]
        # simple beeswarm-like jitter: spread by local density
        jit = rng.uniform(-0.32, 0.32, len(x))
        ax.scatter(x, pos + jit, c=cmap(r), s=5, linewidths=0, alpha=0.8)
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(names[order], color=INK, fontsize=9)
    ax.set_xlabel("SHAP value (contribution to log-odds of >50K)", color=INK)
    ax.set_title("Adult, XGBoost: SHAP values of 2,000 test rows (top 10 by mean |SHAP|)",
                 color=INK, fontsize=10.5, loc="left")
    sm = plt.cm.ScalarMappable(cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["low", "high"])
    cb.set_label("feature value (rank; categories by mean SHAP)", color=INK_2, fontsize=8.5)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=PAPER)
    plt.close(fig)


# ===========================================================================
def main():
    t0 = time.time()
    say(f"scikit-learn {sklearn.__version__}, xgboost {xgb.__version__}, lightgbm {lgb.__version__}, "
        f"catboost {catboost.__version__}, numpy {np.__version__}, pandas {pd.__version__}")
    part1_2()
    part3()
    X, y = load_adult()
    say(f"\nAdult: {X.shape[0]} rows, {X.shape[1]} features, positive rate {y.mean():.4f}")
    seeds = [SEED, 0, 1, 2, 3]
    rf_rows = part4(X, y, seeds)
    keep = part5(X, y, seeds)
    part6()
    pdp = part7()
    FIG_DIR.mkdir(exist_ok=True)
    m, Xtr, Xte, ytr, yte, imp, contrib = keep
    fig_ranks(imp, rf_rows[0], FIG_DIR / "model-interpretation--ranks.png")
    fig_pdp(pdp, FIG_DIR / "model-interpretation--pdp-ice.png")
    fig_shap(Xte, contrib, FIG_DIR / "model-interpretation--shap-summary.png")
    say(f"\ntotal runtime {time.time() - t0:.0f} s")
    (HERE / "results.txt").write_text(out.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:  # run a single part while developing: e.g. `part3`
        globals()[sys.argv[1]]()
    else:
        main()
