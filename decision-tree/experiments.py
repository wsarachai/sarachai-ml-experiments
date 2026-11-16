"""Decision tree (CART) experiments for the "decision-tree" blog post.

Everything the post calls "measured" comes from this script:

1. Hand-grown classification tree on the 6-row toy dataset shared with the
   random-forest post (x1 = hours studied, x2 = exercise score, y = pass).
   Every candidate split is scored with Gini and with entropy / information
   gain, then scikit-learn is asked to grow the same tree so the hand numbers
   are checked against the library.
2. The same toy data with x1 only: a deeper tree that memorises one row, and
   its cost-complexity pruning path (hand numbers vs. sklearn).
3. A small regression tree (SSE / variance reduction, leaf = mean).
4. Breast-cancer dataset (569 rows, 30 features), stratified 398/171 split,
   seed 42: depth sweep, min_samples_leaf sweep, pruning path, ccp_alpha
   chosen by 5-fold CV on the training rows only, and root-split instability
   across bootstrap resamples and across re-splits.
5. Default values of DecisionTreeClassifier / DecisionTreeRegressor as read
   from the installed scikit-learn.

Run:  .venv/Scripts/python.exe decision-tree/experiments.py
Output: decision-tree/results.txt and three PNG figures in decision-tree/figures/.
"""

from __future__ import annotations

import inspect
import io
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sklearn  # noqa: E402
from sklearn.datasets import load_breast_cancer  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
SEED = 42

# "Paper" palette used by the blog's figures.
PAPER = "#F8F6F1"
INK = "#1F2328"
INK_2 = "#5B6270"
GRID = "#E8E2D6"
TEAL = "#0F766E"
AMBER = "#F59E0B"
GREY = "#9AA0AA"

out = io.StringIO()


def say(*args) -> None:
    """Print to stdout and to the results buffer."""
    print(*args)
    print(*args, file=out)


def header(title: str) -> None:
    say()
    say("=" * 78)
    say(title)
    say("=" * 78)


# ---------------------------------------------------------------------------
# Impurity helpers (written out by hand so they can be read next to the post)
# ---------------------------------------------------------------------------

def gini(labels) -> float:
    n = len(labels)
    if n == 0:
        return 0.0
    return 1.0 - sum((c / n) ** 2 for c in Counter(labels).values())


def entropy(labels) -> float:
    n = len(labels)
    if n == 0:
        return 0.0
    return -sum((c / n) * math.log2(c / n) for c in Counter(labels).values())


def sse(values) -> float:
    if len(values) == 0:
        return 0.0
    m = sum(values) / len(values)
    return sum((v - m) ** 2 for v in values)


def candidate_thresholds(column):
    """Midpoints between consecutive distinct sorted values (what CART tries)."""
    vals = sorted(set(column))
    return [(a + b) / 2 for a, b in zip(vals[:-1], vals[1:])]


def score_classification_splits(X, y, feature_names, rows=None):
    """Score every (feature, threshold) on the given rows with Gini and entropy."""
    rows = list(range(len(y))) if rows is None else rows
    ys = [y[i] for i in rows]
    g_parent, h_parent = gini(ys), entropy(ys)
    n = len(rows)
    results = []
    for f, fname in enumerate(feature_names):
        col = [X[i][f] for i in rows]
        for t in candidate_thresholds(col):
            left = [y[i] for i in rows if X[i][f] <= t]
            right = [y[i] for i in rows if X[i][f] > t]
            g_split = len(left) / n * gini(left) + len(right) / n * gini(right)
            h_split = len(left) / n * entropy(left) + len(right) / n * entropy(right)
            results.append(dict(
                feature=fname, t=t, left=left, right=right,
                gL=gini(left), gR=gini(right), g_split=g_split, g_drop=g_parent - g_split,
                hL=entropy(left), hR=entropy(right), ig=h_parent - h_split,
            ))
    return g_parent, h_parent, results


def print_split_table(g_parent, h_parent, results):
    say(f"parent Gini = {g_parent:.4f}   parent entropy = {h_parent:.4f} bits")
    say(f"{'feature':>7} {'thr':>5} {'left y':>14} {'right y':>14} {'G_L':>7} {'G_R':>7} "
        f"{'G_split':>8} {'dGini':>7} {'H_L':>7} {'H_R':>7} {'IG':>7}")
    for r in results:
        say(f"{r['feature']:>7} {r['t']:>5.1f} {str(r['left']):>14} {str(r['right']):>14} "
            f"{r['gL']:>7.4f} {r['gR']:>7.4f} {r['g_split']:>8.4f} {r['g_drop']:>7.4f} "
            f"{r['hL']:>7.4f} {r['hR']:>7.4f} {r['ig']:>7.4f}")


# ---------------------------------------------------------------------------
# Part 1 + 2: the toy dataset shared with the random-forest post
# ---------------------------------------------------------------------------

TOY_X = [[1, 3], [2, 1], [3, 4], [4, 1], [5, 5], [6, 4]]
TOY_Y = [0, 0, 1, 0, 1, 1]


def part_toy_classification():
    header("PART 1  Toy data, both features (x1 hours, x2 exercise score)")
    g0, h0, res = score_classification_splits(TOY_X, TOY_Y, ["x1", "x2"])
    print_split_table(g0, h0, res)
    best = min(res, key=lambda r: r["g_split"])
    best_ig = max(res, key=lambda r: r["ig"])
    say(f"best by Gini: {best['feature']} <= {best['t']}  G_split={best['g_split']:.4f}")
    say(f"best by IG  : {best_ig['feature']} <= {best_ig['t']}  IG={best_ig['ig']:.4f}")

    for crit in ("gini", "entropy"):
        clf = DecisionTreeClassifier(criterion=crit, random_state=SEED).fit(np.array(TOY_X), TOY_Y)
        say(f"\nsklearn criterion={crit}: depth={clf.get_depth()} leaves={clf.get_n_leaves()} "
            f"train acc={clf.score(np.array(TOY_X), TOY_Y):.4f}")
        say(export_text(clf, feature_names=["x1", "x2"], decimals=2))
        say(f"node impurities: {np.round(clf.tree_.impurity, 4).tolist()}")

    header("PART 2  Toy data, x1 only (what if we only knew hours studied?)")
    X1 = [[r[0]] for r in TOY_X]
    say("-- root (rows 1-6)")
    g0, h0, res = score_classification_splits(X1, TOY_Y, ["x1"])
    print_split_table(g0, h0, res)
    say("-- right child of x1<=2.5 (rows 3,4,5,6)")
    g0, h0, res = score_classification_splits(X1, TOY_Y, ["x1"], rows=[2, 3, 4, 5])
    print_split_table(g0, h0, res)
    say("-- left child of x1<=4.5 inside it (rows 3,4)")
    g0, h0, res = score_classification_splits(X1, TOY_Y, ["x1"], rows=[2, 3])
    print_split_table(g0, h0, res)

    clf = DecisionTreeClassifier(random_state=SEED).fit(np.array(X1), TOY_Y)
    say(f"\nsklearn x1-only tree: depth={clf.get_depth()} leaves={clf.get_n_leaves()}")
    say(export_text(clf, feature_names=["x1"], decimals=2))
    # Tie note: x1<=2.5 and x1<=4.5 both give G_split=0.25 at the root.
    # sklearn keeps the first strictly-better threshold it meets while scanning
    # sorted values, so the lower threshold (2.5) wins the tie.

    path = clf.cost_complexity_pruning_path(np.array(X1), TOY_Y)
    say("cost-complexity pruning path (x1-only tree):")
    say(f"  ccp_alphas       = {np.round(path.ccp_alphas, 4).tolist()}")
    say(f"  total impurities = {np.round(path.impurities, 4).tolist()}")
    for a in path.ccp_alphas:
        p = DecisionTreeClassifier(random_state=SEED, ccp_alpha=a).fit(np.array(X1), TOY_Y)
        say(f"  alpha={a:.4f}: leaves={p.get_n_leaves()} depth={p.get_depth()} "
            f"train acc={p.score(np.array(X1), TOY_Y):.4f}")

    # Hand check of the weakest-link alphas: alpha_eff(t) = (R(t) - R(T_t)) / (|T_t| - 1),
    # with R(t) = (n_t / n) * gini(t).
    n = 6
    R = {
        "root{1..6}": (6 / n) * gini([0, 0, 1, 0, 1, 1]),
        "t2{3,4,5,6}": (4 / n) * gini([1, 0, 1, 1]),
        "t3{3,4}": (2 / n) * gini([1, 0]),
    }
    say("hand weakest-link alphas (all leaves pure, so R(T_t) = 0):")
    say(f"  t3{{3,4}}     : R={R['t3{3,4}']:.4f}  leaves=2  alpha={R['t3{3,4}'] / 1:.4f}")
    say(f"  t2{{3,4,5,6}} : R={R['t2{3,4,5,6}']:.4f}  leaves=3  alpha={R['t2{3,4,5,6}'] / 2:.4f}")
    say(f"  root        : R={R['root{1..6}']:.4f}  leaves=4  alpha={R['root{1..6}'] / 3:.4f}")
    say("cost R_alpha(T) = R(T) + alpha*|T| for each candidate subtree at alpha=0.125:")
    a = 0.125
    say(f"  full (4 leaves)        : {0 + 4 * a:.4f}")
    say(f"  prune t3 (3 leaves)    : {R['t3{3,4}'] + 3 * a:.4f}")
    say(f"  prune t2 (2 leaves)    : {R['t2{3,4,5,6}'] + 2 * a:.4f}")
    say(f"  root only (1 leaf)     : {R['root{1..6}'] + 1 * a:.4f}")
    return clf


# ---------------------------------------------------------------------------
# Part 3: a small regression tree (same numbers as the xgboost post's example)
# ---------------------------------------------------------------------------

REG_X = [1, 2, 3, 4, 5, 6]
REG_Y = [10, 12, 11, 40, 42, 41]


def part_toy_regression():
    header("PART 3  Regression toy: hours -> exam score")
    n = len(REG_Y)
    parent = sse(REG_Y)
    say(f"root mean = {sum(REG_Y) / n:.4f}  SSE = {parent:.4f}  MSE (sklearn impurity) = {parent / n:.4f}")
    say(f"{'thr':>5} {'left y':>16} {'right y':>16} {'mean_L':>7} {'mean_R':>7} {'SSE_L':>8} "
        f"{'SSE_R':>8} {'SSE_split':>9} {'drop':>9} {'var drop':>9}")
    for t in candidate_thresholds(REG_X):
        L = [yv for xv, yv in zip(REG_X, REG_Y) if xv <= t]
        R = [yv for xv, yv in zip(REG_X, REG_Y) if xv > t]
        s = sse(L) + sse(R)
        say(f"{t:>5.1f} {str(L):>16} {str(R):>16} {sum(L) / len(L):>7.3f} {sum(R) / len(R):>7.3f} "
            f"{sse(L):>8.3f} {sse(R):>8.3f} {s:>9.3f} {parent - s:>9.3f} {(parent - s) / n:>9.4f}")

    X = np.array(REG_X, dtype=float).reshape(-1, 1)
    y = np.array(REG_Y, dtype=float)
    stump = DecisionTreeRegressor(max_depth=1, random_state=SEED).fit(X, y)
    say("\nsklearn max_depth=1:")
    say(export_text(stump, feature_names=["x"], decimals=2))
    say(f"predictions: {stump.predict(X).tolist()}")
    say(f"node impurities (MSE): {np.round(stump.tree_.impurity, 4).tolist()}")
    d2 = DecisionTreeRegressor(max_depth=2, random_state=SEED).fit(X, y)
    say("sklearn max_depth=2:")
    say(export_text(d2, feature_names=["x"], decimals=2))
    say(f"predictions: {np.round(d2.predict(X), 4).tolist()}")
    full = DecisionTreeRegressor(random_state=SEED).fit(X, y)
    say(f"sklearn unrestricted: depth={full.get_depth()} leaves={full.get_n_leaves()} "
        f"predictions={full.predict(X).tolist()}")


# ---------------------------------------------------------------------------
# Part 4: breast cancer
# ---------------------------------------------------------------------------

def load_split(seed=SEED):
    data = load_breast_cancer()
    data.feature_names = [str(f) for f in data.feature_names]  # plain str for printing
    Xtr, Xte, ytr, yte = train_test_split(
        data.data, data.target, test_size=0.3, random_state=seed, stratify=data.target
    )
    return data, Xtr, Xte, ytr, yte


def part_breast_cancer():
    header("PART 4  Breast cancer (569 x 30), stratified 70/30 split, seed 42")
    data, Xtr, Xte, ytr, yte = load_split()
    say(f"train rows={len(ytr)} test rows={len(yte)}  class counts train={np.bincount(ytr).tolist()} "
        f"test={np.bincount(yte).tolist()}")
    say(f"one test row = {100 / len(yte):.2f} percentage points")

    base = DecisionTreeClassifier(random_state=SEED).fit(Xtr, ytr)
    say(f"\nunpruned tree: depth={base.get_depth()} leaves={base.get_n_leaves()} "
        f"train={base.score(Xtr, ytr):.4f} test={base.score(Xte, yte):.4f}")
    root_f = base.tree_.feature[0]
    say(f"root split: '{data.feature_names[root_f]}' <= {base.tree_.threshold[0]:.4f}")

    # Depth sweep -----------------------------------------------------------
    say("\nmax_depth sweep (random_state=42):")
    say(f"{'max_depth':>9} {'depth':>5} {'leaves':>6} {'train':>7} {'test':>7}")
    depth_rows = []
    for d in list(range(1, 9)) + [None]:
        m = DecisionTreeClassifier(max_depth=d, random_state=SEED).fit(Xtr, ytr)
        tr, te = m.score(Xtr, ytr), m.score(Xte, yte)
        depth_rows.append((d, m.get_depth(), m.get_n_leaves(), tr, te))
        say(f"{str(d):>9} {m.get_depth():>5} {m.get_n_leaves():>6} {tr:>7.4f} {te:>7.4f}")

    # 5-fold CV on the training rows only, per depth (what you should pick by)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    gs_depth = GridSearchCV(
        DecisionTreeClassifier(random_state=SEED),
        {"max_depth": list(range(1, 9)) + [None]}, cv=cv,
    ).fit(Xtr, ytr)
    say("5-fold CV accuracy on training rows, per max_depth:")
    for p, s in zip(gs_depth.cv_results_["params"], gs_depth.cv_results_["mean_test_score"]):
        say(f"  max_depth={str(p['max_depth']):>4}: CV={s:.4f}")
    depth_cv = list(gs_depth.cv_results_["mean_test_score"])
    say(f"CV picks max_depth={gs_depth.best_params_['max_depth']} "
        f"-> test={gs_depth.best_estimator_.score(Xte, yte):.4f}")

    # min_samples_leaf sweep ----------------------------------------------
    say("\nmin_samples_leaf sweep (max_depth=None):")
    say(f"{'msl':>4} {'depth':>5} {'leaves':>6} {'train':>7} {'test':>7}")
    for msl in [1, 2, 5, 10, 20, 50]:
        m = DecisionTreeClassifier(min_samples_leaf=msl, random_state=SEED).fit(Xtr, ytr)
        say(f"{msl:>4} {m.get_depth():>5} {m.get_n_leaves():>6} "
            f"{m.score(Xtr, ytr):>7.4f} {m.score(Xte, yte):>7.4f}")

    # Pruning path ----------------------------------------------------------
    path = base.cost_complexity_pruning_path(Xtr, ytr)
    say(f"\ncost-complexity pruning path: {len(path.ccp_alphas)} alphas")
    say(f"{'alpha':>8} {'impurity':>9} {'leaves':>6} {'depth':>5} {'train':>7} {'test':>7}")
    prune_rows = []
    for a, imp in zip(path.ccp_alphas, path.impurities):
        m = DecisionTreeClassifier(random_state=SEED, ccp_alpha=a).fit(Xtr, ytr)
        tr, te = m.score(Xtr, ytr), m.score(Xte, yte)
        prune_rows.append((a, imp, m.get_n_leaves(), m.get_depth(), tr, te))
        say(f"{a:>8.5f} {imp:>9.5f} {m.get_n_leaves():>6} {m.get_depth():>5} {tr:>7.4f} {te:>7.4f}")

    # Choose alpha by CV on the training rows only (never by test score).
    alphas = path.ccp_alphas[:-1]  # the last alpha prunes to the root
    gs = GridSearchCV(
        DecisionTreeClassifier(random_state=SEED), {"ccp_alpha": alphas}, cv=cv
    ).fit(Xtr, ytr)
    best_a = gs.best_params_["ccp_alpha"]
    bm = gs.best_estimator_
    say("5-fold CV accuracy per alpha:")
    for p, s in zip(gs.cv_results_["params"], gs.cv_results_["mean_test_score"]):
        say(f"  alpha={p['ccp_alpha']:.5f}: CV={s:.4f}")
    alpha_cv = list(gs.cv_results_["mean_test_score"])
    say(f"CV picks ccp_alpha={best_a:.5f}: leaves={bm.get_n_leaves()} depth={bm.get_depth()} "
        f"train={bm.score(Xtr, ytr):.4f} test={bm.score(Xte, yte):.4f}")

    # Instability: re-split 10 times (seeds 0..9), and bootstrap the training rows.
    say("\nre-split 10 times (split seed = tree seed = 0..9), unpruned trees:")
    accs, roots = [], []
    for s in range(10):
        _, a_tr, a_te, b_tr, b_te = load_split(seed=s)
        m = DecisionTreeClassifier(random_state=s).fit(a_tr, b_tr)
        accs.append(m.score(a_te, b_te))
        roots.append(data.feature_names[m.tree_.feature[0]])
        say(f"  seed {s}: depth={m.get_depth()} leaves={m.get_n_leaves()} test={accs[-1]:.4f} "
            f"root='{roots[-1]}' <= {m.tree_.threshold[0]:.4f}")
    say(f"  mean={np.mean(accs):.4f} sd(ddof=0)={np.std(accs):.4f} sd(ddof=1)={np.std(accs, ddof=1):.4f} "
        f"min={min(accs):.4f} max={max(accs):.4f}")
    say(f"  root features: {dict(Counter(roots))}")

    say("\n20 bootstrap resamples of the seed-42 training rows, unpruned trees:")
    rng = np.random.default_rng(SEED)
    boot_roots, boot_leaves, boot_test, second = [], [], [], []
    for b in range(20):
        idx = rng.integers(0, len(ytr), len(ytr))
        m = DecisionTreeClassifier(random_state=SEED).fit(Xtr[idx], ytr[idx])
        boot_roots.append(data.feature_names[m.tree_.feature[0]])
        boot_leaves.append(m.get_n_leaves())
        boot_test.append(m.score(Xte, yte))
        left = m.tree_.children_left[0]
        second.append(data.feature_names[m.tree_.feature[left]] if m.tree_.feature[left] >= 0 else "leaf")
    say(f"  root feature counts: {dict(Counter(boot_roots))}")
    say(f"  feature at root's left child: {dict(Counter(second))}")
    say(f"  leaves: min={min(boot_leaves)} max={max(boot_leaves)}")
    say(f"  test acc: min={min(boot_test):.4f} max={max(boot_test):.4f} mean={np.mean(boot_test):.4f}")
    distinct_trees = len({(r, s) for r, s in zip(boot_roots, second)})
    say(f"  distinct (root, left-child) feature pairs among 20 trees: {distinct_trees}")

    return depth_rows, depth_cv, prune_rows, alpha_cv, best_a


# ---------------------------------------------------------------------------
# Part 5: defaults
# ---------------------------------------------------------------------------

def part_defaults():
    header("PART 5  Defaults in the installed scikit-learn")
    say(f"python {sys.version.split()[0]}  scikit-learn {sklearn.__version__}  numpy {np.__version__}")
    keys = ["criterion", "splitter", "max_depth", "min_samples_split", "min_samples_leaf",
            "min_weight_fraction_leaf", "max_features", "max_leaf_nodes",
            "min_impurity_decrease", "class_weight", "ccp_alpha", "monotonic_cst", "random_state"]
    for cls in (DecisionTreeClassifier, DecisionTreeRegressor):
        sig = inspect.signature(cls.__init__)
        say(f"{cls.__name__}:")
        for k in keys:
            if k in sig.parameters:
                say(f"  {k} = {sig.parameters[k].default!r}")
    say(f"criterion options (classifier): {sorted(DecisionTreeClassifier._parameter_constraints['criterion'][0].options)}")
    say(f"criterion options (regressor):  {sorted(DecisionTreeRegressor._parameter_constraints['criterion'][0].options)}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def style_axes(ax):
    ax.set_facecolor(PAPER)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#CFCAC0")
    ax.tick_params(colors=INK_2, labelsize=12)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def fig_toy_trees():
    """Two hand-grown trees on the toy data: with x2 (one split) and x1 only (three)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.4), dpi=100,
                             gridspec_kw={"width_ratios": [1, 1.35]})
    fig.patch.set_facecolor(PAPER)
    for ax in axes:
        ax.set_facecolor(PAPER)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

    def node(ax, x, y, text, leaf=False, w=3.0, h=1.5):
        fc = "#E6F2F0" if leaf else "white"
        ec = TEAL if leaf else INK_2
        ax.add_patch(matplotlib.patches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
            fc=fc, ec=ec, lw=1.6))
        ax.text(x, y, text, ha="center", va="center", fontsize=14, color=INK, linespacing=1.4)

    def edge(ax, x0, y0, x1, y1, label):
        ax.plot([x0, x1], [y0, y1], color=GREY, lw=1.5, zorder=0)
        ax.text((x0 + x1) / 2 + (0.25 if x1 > x0 else -0.25), (y0 + y1) / 2, label,
                ha="left" if x1 > x0 else "right", va="center", fontsize=13, color=INK_2)

    a = axes[0]
    a.set_title("A. both features: one split is enough", fontsize=15, color=INK, loc="left")
    node(a, 5, 8.3, "x2 <= 3.5 ?\nrows 1-6, y = 3:3\nGini 0.5", w=3.6, h=1.9)
    edge(a, 5, 7.35, 2.6, 4.9, "yes")
    edge(a, 5, 7.35, 7.4, 4.9, "no")
    node(a, 2.6, 4.2, "rows 1, 2, 4\ny = 0, 0, 0\npredict 0 (fail)", leaf=True, w=3.4, h=1.9)
    node(a, 7.4, 4.2, "rows 3, 5, 6\ny = 1, 1, 1\npredict 1 (pass)", leaf=True, w=3.4, h=1.9)
    a.text(5, 1.6, "weighted Gini after the split = 0\ninformation gain = 1 bit",
           ha="center", fontsize=12.5, color=TEAL)

    b = axes[1]
    b.set_title("B. x1 only: three splits, one leaf holds a single row", fontsize=15, color=INK, loc="left")
    node(b, 3.6, 8.9, "x1 <= 2.5 ?\nrows 1-6, Gini 0.5", w=3.4, h=1.4)
    edge(b, 3.6, 8.2, 1.6, 6.6, "yes")
    edge(b, 3.6, 8.2, 5.9, 6.6, "no")
    node(b, 1.6, 6.0, "rows 1, 2\npredict 0", leaf=True, w=2.6, h=1.3)
    node(b, 5.9, 6.0, "x1 <= 4.5 ?\nrows 3-6, Gini 0.375", w=3.4, h=1.3)
    edge(b, 5.9, 5.35, 4.2, 3.7, "yes")
    edge(b, 5.9, 5.35, 8.0, 3.7, "no")
    node(b, 4.2, 3.1, "x1 <= 3.5 ?\nrows 3, 4, Gini 0.5", w=3.2, h=1.3)
    node(b, 8.0, 3.1, "rows 5, 6\npredict 1", leaf=True, w=2.6, h=1.3)
    edge(b, 4.2, 2.45, 2.7, 1.1, "yes")
    edge(b, 4.2, 2.45, 5.8, 1.1, "no")
    node(b, 2.7, 0.6, "row 3: predict 1", leaf=True, w=2.9, h=0.9)
    node(b, 5.8, 0.6, "row 4: predict 0", leaf=True, w=2.9, h=0.9)

    fig.tight_layout(pad=1.2)
    fig.savefig(FIG_DIR / "decision-tree--toy-trees.png", facecolor=PAPER)
    plt.close(fig)


def fig_depth(depth_rows, depth_cv):
    """Train / 5-fold CV / test accuracy against max_depth on breast cancer."""
    fig, ax = plt.subplots(figsize=(16, 8.8), dpi=100)
    fig.patch.set_facecolor(PAPER)
    style_axes(ax)
    labels = [str(r[0]) for r in depth_rows]  # 1..8, None
    pos = list(range(len(depth_rows)))
    tr = [r[3] for r in depth_rows]
    te = [r[4] for r in depth_rows]
    ax.plot(pos, tr, "--", color=GREY, lw=2.5, label="training accuracy")
    ax.plot(pos, depth_cv, "-o", color=TEAL, lw=3, ms=8, label="5-fold CV on the 398 training rows")
    ax.plot(pos, te, "-s", color=AMBER, lw=3, ms=8, label="test accuracy (171 held-out rows)")
    ax.set_xticks(pos, labels)
    ax.set_ylim(0.88, 1.008)
    ax.set_xlabel("max_depth   (None = grow until every leaf is pure; this tree stops at depth 6)",
                  fontsize=15, color=INK)
    ax.set_ylabel("accuracy", fontsize=15, color=INK)
    ax.tick_params(labelsize=14)
    ax.annotate("train reaches 1.0000 at depth 6,\nthe held-out curves do not follow",
                xy=(pos[5], tr[5]), xytext=(pos[5] - 0.2, 0.968), ha="center",
                fontsize=14, color=INK_2, arrowprops=dict(arrowstyle="-", color=GREY))
    ax.legend(loc="lower right", frameon=False, fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "decision-tree--depth-vs-accuracy.png", facecolor=PAPER)
    plt.close(fig)


def fig_pruning(prune_rows, alpha_cv, best_a):
    """Accuracy along the cost-complexity pruning path, one point per subtree."""
    rows = prune_rows[:-1]  # drop the root-only tree (test 0.6257) so the scale stays readable
    fig, ax = plt.subplots(figsize=(16, 8.8), dpi=100)
    fig.patch.set_facecolor(PAPER)
    style_axes(ax)
    pos = list(range(len(rows)))
    labels = [f"{r[2]}\n{r[0]:.4f}" for r in rows]
    tr = [r[4] for r in rows]
    te = [r[5] for r in rows]
    ax.plot(pos, tr, "--", color=GREY, lw=2.5, label="training accuracy")
    ax.plot(pos, alpha_cv, "-o", color=TEAL, lw=3, ms=8, label="5-fold CV on the training rows")
    ax.plot(pos, te, "-s", color=AMBER, lw=3, ms=8, label="test accuracy")
    best_i = [r[0] for r in rows].index(best_a)
    ax.axvline(best_i, color=TEAL, lw=1.5, ls=":")
    ax.annotate(f"CV picks ccp_alpha = {best_a:.4f}\n{rows[best_i][2]} leaves, test {rows[best_i][5]:.4f}",
                xy=(best_i, alpha_cv[best_i]), xytext=(best_i + 0.6, 0.997),
                fontsize=14, color=TEAL, va="top")
    ax.set_xticks(pos, labels)
    ax.set_ylim(0.88, 1.008)
    ax.set_xlabel("leaves left  /  ccp_alpha   (moving right = pruning harder)", fontsize=15, color=INK)
    ax.set_ylabel("accuracy", fontsize=15, color=INK)
    ax.tick_params(labelsize=13)
    ax.legend(loc="lower left", frameon=False, fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "decision-tree--pruning-path.png", facecolor=PAPER)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(exist_ok=True)
    part_defaults()
    part_toy_classification()
    part_toy_regression()
    depth_rows, depth_cv, prune_rows, alpha_cv, best_a = part_breast_cancer()
    fig_toy_trees()
    fig_depth(depth_rows, depth_cv)
    fig_pruning(prune_rows, alpha_cv, best_a)
    (HERE / "results.txt").write_text(out.getvalue(), encoding="utf-8")
    print(f"\nwrote {HERE / 'results.txt'} and figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
