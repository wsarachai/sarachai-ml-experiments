"""Classification-metric experiments for the "classification-metrics" blog post.

Everything the post calls "measured" comes from this script:

1. A 12-row toy table of scores and labels, worked by hand: confusion matrix
   at two thresholds, precision / recall / F1 / F2, ROC points and ROC-AUC
   (trapezoid and the rank / pairs formula), average precision, expected cost
   per threshold, and a tiny 3-class macro / micro / weighted example.
   Every hand number is asserted against scikit-learn.
2. The Adult census-income dataset (OpenML "adult", version 2, ~24% ">50K"),
   split 60/20/20 train / validation / test (stratified, seed 42).
   Models: XGBoost (log loss, native categoricals) and a random forest.
   a. metric table vs. a majority-class baseline (test set, threshold 0.5)
   b. ROC-AUC vs. average precision when positives are made ten times rarer
   c. threshold chosen from misclassification costs on validation, reported
      on test (hand sweep, TunedThresholdClassifierCV, FixedThresholdClassifier)
   d. calibration before / after CalibratedClassifierCV fitted on validation
   e. class weighting (scale_pos_weight / class_weight) and undersampling:
      effect on precision, recall, ranking and calibration

Run:  .venv/Scripts/python.exe classification-metrics/experiments.py
Output: classification-metrics/results.txt and PNG figures in
classification-metrics/figures/. The dataset is cached in ../data_cache/.
"""

from __future__ import annotations

import io
import sys
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
import xgboost  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV, calibration_curve  # noqa: E402
from sklearn.datasets import fetch_openml  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.frozen import FrozenEstimator  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (  # noqa: E402
    FixedThresholdClassifier,
    TunedThresholdClassifierCV,
    train_test_split,
)
from sklearn.preprocessing import OrdinalEncoder  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
DATA_HOME = HERE.parent / "data_cache"
SEED = 42

# Misclassification costs used throughout (an assumption for illustration):
# missing a positive costs five times as much as a false alarm.
C_FP = 1.0
C_FN = 5.0

# "Paper" palette used by the blog's figures.
PAPER = "#F8F6F1"
INK = "#1F2328"
INK_2 = "#5B6270"
GRID = "#E8E2D6"
TEAL = "#0F766E"
AMBER = "#D97706"
GREY = "#9AA0AA"

out = io.StringIO()


def say(*args) -> None:
    """Print to stdout and to the results buffer."""
    text = " ".join(str(a) for a in args)
    print(text)
    out.write(text + "\n")


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


def style_axes(ax) -> None:
    ax.set_facecolor(PAPER)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


plt.rcParams.update({
    "figure.facecolor": PAPER, "savefig.facecolor": PAPER, "axes.facecolor": PAPER,
    "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "font.size": 11, "legend.frameon": False,
})


# ---------------------------------------------------------------------------
# 1. Hand calculation on a 12-row toy table
# ---------------------------------------------------------------------------
def counts(y, s, t):
    pred = (s >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return tp, fp, fn, tn


def part_toy() -> dict:
    say("=" * 78)
    say("1. Toy table: 12 rows, scores and labels")
    say("=" * 78)
    s = np.array([0.95, 0.85, 0.80, 0.70, 0.60, 0.55, 0.45, 0.40, 0.30, 0.20, 0.15, 0.05])
    y = np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0])
    P, N = int(y.sum()), int((1 - y).sum())
    say(f"rows={len(y)}  positives P={P}  negatives N={N}")
    for i, (si, yi) in enumerate(zip(s, y), 1):
        say(f"  row {i:2d}  score={si:.2f}  y={yi}")

    # Majority baseline: predict 0 for everyone.
    base_acc = N / len(y)
    assert close(base_acc, accuracy_score(y, np.zeros_like(y)))
    say(f"majority baseline accuracy = {N}/{len(y)} = {base_acc:.4f}")

    say("\nConfusion matrix and metrics at two thresholds (predict 1 if score >= t):")
    for t in (0.75, 0.50):
        tp, fp, fn, tn = counts(y, s, t)
        pred = (s >= t).astype(int)
        cm = confusion_matrix(y, pred)
        assert cm.tolist() == [[tn, fp], [fn, tp]]
        acc = Fraction(tp + tn, len(y))
        prec = Fraction(tp, tp + fp)
        rec = Fraction(tp, tp + fn)
        f1 = 2 * prec * rec / (prec + rec)
        f2 = 5 * prec * rec / (4 * prec + rec)
        fpr = Fraction(fp, N)
        assert close(float(acc), accuracy_score(y, pred))
        assert close(float(prec), precision_score(y, pred))
        assert close(float(rec), recall_score(y, pred))
        assert close(float(f1), f1_score(y, pred))
        assert close(float(f2), fbeta_score(y, pred, beta=2))
        say(f"  t={t:.2f}: TP={tp} FP={fp} FN={fn} TN={tn} | acc={acc}={float(acc):.4f} "
            f"precision={prec}={float(prec):.4f} recall={rec}={float(rec):.4f} "
            f"F1={f1}={float(f1):.4f} F2={f2}={float(f2):.4f} FPR={fpr}={float(fpr):.4f}")
        say(f"           sklearn confusion_matrix = {cm.tolist()}  (rows = true 0/1, cols = pred 0/1)")

    say("\nROC points, walking the threshold down one row at a time:")
    pts = [(Fraction(0), Fraction(0))]
    tp = fp = 0
    for yi in y:
        if yi == 1:
            tp += 1
        else:
            fp += 1
        pts.append((Fraction(fp, N), Fraction(tp, P)))
    for k, (fx, ty) in enumerate(pts):
        say(f"  top {k:2d} rows -> FPR={fx} ({float(fx):.3f})  TPR={ty} ({float(ty):.3f})")
    trap = sum((pts[i + 1][0] - pts[i][0]) * (pts[i + 1][1] + pts[i][1]) / 2 for i in range(len(pts) - 1))
    fpr_sk, tpr_sk, _ = roc_curve(y, s, drop_intermediate=False)
    assert np.allclose(fpr_sk, [float(a) for a, _ in pts]) and np.allclose(tpr_sk, [float(b) for _, b in pts])
    assert close(float(trap), roc_auc_score(y, s))
    say(f"ROC-AUC by trapezoids = {trap} = {float(trap):.4f}; sklearn roc_auc_score = {roc_auc_score(y, s):.4f}")

    # Rank / pairs formula: fraction of (positive, negative) pairs ranked correctly.
    pos, neg = s[y == 1], s[y == 0]
    wins = sum(int(p > n) + 0.5 * int(p == n) for p in pos for n in neg)
    per_pos = [int((neg < p).sum()) for p in pos]
    say(f"pairs formula: negatives below each positive = {per_pos}, sum={wins:.0f} of {P}*{N}={P*N} pairs "
        f"-> AUC = {wins/(P*N):.4f}")
    assert close(wins / (P * N), roc_auc_score(y, s))

    # Average precision: precision at the rank of each positive, averaged.
    order = np.argsort(-s)
    hits, precs = 0, []
    for k, idx in enumerate(order, 1):
        if y[idx] == 1:
            hits += 1
            precs.append(Fraction(hits, k))
    ap = sum(precs) / P
    say(f"AP: precision at each positive = {[str(p) for p in precs]} -> AP = {ap} = {float(ap):.4f}; "
        f"sklearn average_precision_score = {average_precision_score(y, s):.4f}")
    assert close(float(ap), average_precision_score(y, s))
    prec_c, rec_c, _ = precision_recall_curve(y, s)
    trap_pr = auc(rec_c, prec_c)
    say(f"(trapezoidal area under the PR points = {trap_pr:.4f}, differs from AP by design)")

    say(f"\nExpected cost with C_FP={C_FP:g}, C_FN={C_FN:g} (cut after the top k rows):")
    cost_rows = []
    for k in range(0, len(y) + 1):
        t = s[k - 1] if k > 0 else 1.01
        tp_, fp_, fn_, tn_ = counts(y, s, t)
        cost = C_FP * fp_ + C_FN * fn_
        cost_rows.append((k, t, tp_, fp_, fn_, cost))
        say(f"  top {k:2d} (t={t:.2f}): TP={tp_} FP={fp_} FN={fn_} -> cost = {fp_}*{C_FP:g} + {fn_}*{C_FN:g} = {cost:g}")
    best = min(cost_rows, key=lambda r: r[5])
    say(f"lowest cost: cut after top {best[0]} rows (t={best[1]:.2f}), total cost {best[5]:g} "
        f"= {best[5]/len(y):.4f} per row")
    say(f"Bayes threshold for calibrated probabilities: C_FP/(C_FP+C_FN) = {C_FP/(C_FP+C_FN):.4f}")

    # Brier score and log loss on the toy table (treating scores as probabilities).
    brier = float(np.mean((s - y) ** 2))
    assert close(brier, brier_score_loss(y, s))
    ll = float(-np.mean(y * np.log(s) + (1 - y) * np.log(1 - s)))
    assert close(ll, log_loss(y, s))
    say(f"Brier score (mean of (p-y)^2) = {brier:.4f}; log loss = {ll:.4f} (both match sklearn)")

    say("\nMulticlass averaging on a 3-class toy confusion matrix:")
    # rows = true class, cols = predicted class
    cm3 = np.array([[50, 3, 2], [4, 10, 6], [1, 4, 5]])
    yt, yp = [], []
    for i in range(3):
        for j in range(3):
            yt += [i] * cm3[i, j]
            yp += [j] * cm3[i, j]
    yt, yp = np.array(yt), np.array(yp)
    assert (confusion_matrix(yt, yp) == cm3).all()
    support = cm3.sum(axis=1)
    pc_f1 = []
    for c in range(3):
        tp_c = cm3[c, c]
        prec_c3 = tp_c / cm3[:, c].sum()
        rec_c3 = tp_c / cm3[c, :].sum()
        f1_c = 2 * prec_c3 * rec_c3 / (prec_c3 + rec_c3)
        pc_f1.append(f1_c)
        say(f"  class {c}: support={support[c]} precision={prec_c3:.4f} recall={rec_c3:.4f} F1={f1_c:.4f}")
    macro = float(np.mean(pc_f1))
    weighted = float(np.dot(pc_f1, support) / support.sum())
    micro = np.trace(cm3) / cm3.sum()
    assert close(macro, f1_score(yt, yp, average="macro"))
    assert close(weighted, f1_score(yt, yp, average="weighted"))
    assert close(micro, f1_score(yt, yp, average="micro"))
    assert close(micro, accuracy_score(yt, yp))
    say(f"  macro F1={macro:.4f}  weighted F1={weighted:.4f}  micro F1={micro:.4f} (= accuracy)")
    say(f"  confusion matrix = {cm3.tolist()}")
    say("All toy-table numbers asserted against scikit-learn: OK")
    return {"s": s, "y": y, "pts": [(float(a), float(b)) for a, b in pts], "precs": precs}


# ---------------------------------------------------------------------------
# 2. Adult census income
# ---------------------------------------------------------------------------
def load_adult():
    data = fetch_openml("adult", version=2, as_frame=True, data_home=str(DATA_HOME))
    X = data.data.copy()
    y = (data.target.astype(str) == ">50K").astype(int).to_numpy()
    cat_cols = [c for c in X.columns if str(X[c].dtype) == "category"]
    return X, y, cat_cols


def expected_cost(y, p, t):
    pred = p >= t
    fp = np.sum(pred & (y == 0))
    fn = np.sum(~pred & (y == 1))
    return (C_FP * fp + C_FN * fn) / len(y)


def metric_row(name, y, p, t=0.5):
    pred = (p >= t).astype(int)
    return {
        "model": name,
        "acc": accuracy_score(y, pred),
        "bal_acc": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "F1": f1_score(y, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y, p),
        "AP": average_precision_score(y, p),
        "Brier": brier_score_loss(y, p),
        "logloss": log_loss(y, np.clip(p, 1e-15, 1 - 1e-15)),
        "mean_p": float(np.mean(p)),
        "cost": expected_cost(y, p, t),
    }


def print_table(rows, cols):
    say("  " + " | ".join(f"{c:>9}" for c in ["model"] + cols))
    for r in rows:
        cells = [f"{r['model']:>9}"[:30]] + [f"{r[c]:9.4f}" for c in cols]
        say("  " + " | ".join(cells))


def ece(y, p, n_bins=10):
    """Expected calibration error with equal-width bins (simple, by hand)."""
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.sum() / len(y) * abs(p[m].mean() - y[m].mean())
    return total


def part_adult(toy: dict) -> None:
    say("\n" + "=" * 78)
    say("2. Adult census income (OpenML adult v2)")
    say("=" * 78)
    X, y, cat_cols = load_adult()
    say(f"rows={len(y)} features={X.shape[1]} categorical={len(cat_cols)} positives (>50K)={y.mean():.4f}")

    X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=SEED)
    say(f"train={len(y_tr)} ({y_tr.mean():.4f} pos)  validation={len(y_va)} ({y_va.mean():.4f} pos)  "
        f"test={len(y_te)} ({y_te.mean():.4f} pos)")

    # Random forest needs numbers: ordinal-encode categoricals (fit on train only).
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                         encoded_missing_value=-1)
    enc.fit(X_tr[cat_cols])

    def to_num(df):
        d = df.copy()
        d[cat_cols] = enc.transform(d[cat_cols])
        return d.astype(float)

    Xn_tr, Xn_va, Xn_te = to_num(X_tr), to_num(X_va), to_num(X_te)

    def make_xgb(spw=1.0):
        return XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6,
                             tree_method="hist", enable_categorical=True,
                             scale_pos_weight=spw, random_state=SEED, n_jobs=4,
                             eval_metric="logloss")

    def make_rf(cw=None):
        return RandomForestClassifier(n_estimators=300, class_weight=cw, random_state=SEED, n_jobs=4)

    xgb = make_xgb().fit(X_tr, y_tr)
    rf = make_rf().fit(Xn_tr, y_tr)
    dummy = DummyClassifier(strategy="prior").fit(Xn_tr, y_tr)

    p = {
        "xgb": {"va": xgb.predict_proba(X_va)[:, 1], "te": xgb.predict_proba(X_te)[:, 1]},
        "rf": {"va": rf.predict_proba(Xn_va)[:, 1], "te": rf.predict_proba(Xn_te)[:, 1]},
    }
    p_dummy = dummy.predict_proba(Xn_te)[:, 1]

    # -- a. metric table -----------------------------------------------------
    say("\n2a. Test-set metrics at threshold 0.5 (baseline = majority class / prior probability)")
    cols = ["acc", "bal_acc", "precision", "recall", "F1", "ROC_AUC", "AP", "Brier", "logloss", "cost"]
    rows = [metric_row("baseline", y_te, p_dummy), metric_row("XGBoost", y_te, p["xgb"]["te"]),
            metric_row("RF", y_te, p["rf"]["te"])]
    print_table(rows, cols)
    say(f"  AP of a random ranking = prevalence = {y_te.mean():.4f}")
    for name in ("xgb", "rf"):
        pred = (p[name]["te"] >= 0.5).astype(int)
        say(f"  {name} confusion matrix (test, t=0.5) [[TN, FP], [FN, TP]] = {confusion_matrix(y_te, pred).tolist()}")
    say(f"  baseline confusion matrix = {confusion_matrix(y_te, (p_dummy >= 0.5).astype(int)).tolist()}")

    # -- b. ROC-AUC vs AP when positives become rare -----------------------------
    say("\n2b. Same model, same negatives, positives subsampled to 1/10 (test set, 20 random draws)")
    rng = np.random.default_rng(SEED)
    pos_idx, neg_idx = np.where(y_te == 1)[0], np.where(y_te == 0)[0]
    rare = {"xgb": [], "rf": []}
    for _ in range(20):
        keep = np.concatenate([neg_idx, rng.choice(pos_idx, size=len(pos_idx) // 10, replace=False)])
        for name in ("xgb", "rf"):
            yy, pp = y_te[keep], p[name]["te"][keep]
            pred = pp >= 0.5
            rare[name].append((roc_auc_score(yy, pp), average_precision_score(yy, pp),
                               precision_score(yy, pred), recall_score(yy, pred), yy.mean()))
    for name in ("xgb", "rf"):
        a = np.array(rare[name])
        full = metric_row(name, y_te, p[name]["te"])
        say(f"  {name}: full test  prevalence={y_te.mean():.4f} ROC-AUC={full['ROC_AUC']:.4f} AP={full['AP']:.4f} "
            f"precision@0.5={full['precision']:.4f} recall@0.5={full['recall']:.4f}")
        say(f"  {name}: rare pos.  prevalence={a[:,4].mean():.4f} ROC-AUC={a[:,0].mean():.4f}+-{a[:,0].std():.4f} "
            f"AP={a[:,1].mean():.4f}+-{a[:,1].std():.4f} precision@0.5={a[:,2].mean():.4f} recall@0.5={a[:,3].mean():.4f}")
    # one fixed rare draw for the figure
    rng_fig = np.random.default_rng(SEED + 1)
    keep_fig = np.concatenate([neg_idx, rng_fig.choice(pos_idx, size=len(pos_idx) // 10, replace=False)])

    # -- c. threshold from costs ----------------------------------------------
    say(f"\n2c. Threshold from costs (C_FP={C_FP:g}, C_FN={C_FN:g}); chosen on validation, reported on test")
    grid = np.round(np.arange(0.01, 1.00, 0.01), 2)
    curves = {}
    for name in ("xgb", "rf"):
        cv_va = np.array([expected_cost(y_va, p[name]["va"], t) for t in grid])
        cv_te = np.array([expected_cost(y_te, p[name]["te"], t) for t in grid])
        curves[name] = (cv_va, cv_te)
        t_best = float(grid[np.argmin(cv_va)])
        bayes = C_FP / (C_FP + C_FN)
        say(f"  {name}: best threshold on validation = {t_best:.2f} (validation cost/row {cv_va.min():.4f})")
        for label, t in (("0.50 (default)", 0.5), (f"{bayes:.4f} (Bayes, assumes calibration)", bayes),
                         (f"{t_best:.2f} (tuned on validation)", t_best)):
            r = metric_row(name, y_te, p[name]["te"], t)
            say(f"    test @ t={label:40s} cost/row={r['cost']:.4f} precision={r['precision']:.4f} "
                f"recall={r['recall']:.4f} acc={r['acc']:.4f}")
        say(f"    (oracle: lowest test cost over the grid = {cv_te.min():.4f} at t={grid[np.argmin(cv_te)]:.2f} -- not selectable)")
    say(f"  baseline 'predict nobody' cost/row on test = {expected_cost(y_te, np.zeros_like(y_te, dtype=float), 0.5):.4f}; "
        f"'predict everybody' = {expected_cost(y_te, np.ones_like(y_te, dtype=float), 0.5):.4f}")

    # Same thing with scikit-learn's threshold tools.
    def neg_cost(y_true, y_pred):
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        return -(C_FP * fp + C_FN * fn) / len(y_true)

    scorer = make_scorer(neg_cost)
    tuned = TunedThresholdClassifierCV(FrozenEstimator(xgb), scoring=scorer, cv="prefit", refit=False,
                                       thresholds=grid).fit(X_va, y_va)
    pred_t = tuned.predict(X_te)
    say(f"  TunedThresholdClassifierCV(FrozenEstimator(xgb), cv='prefit', thresholds=grid) on validation: "
        f"best_threshold_={tuned.best_threshold_:.4f} best_score_={tuned.best_score_:.4f}; "
        f"test cost/row={-neg_cost(y_te, pred_t):.4f}")
    fixed = FixedThresholdClassifier(FrozenEstimator(xgb), threshold=float(tuned.best_threshold_)).fit(X_tr, y_tr)
    assert (fixed.predict(X_te) == pred_t).all()
    say("  FixedThresholdClassifier(FrozenEstimator(xgb), threshold=best) gives identical test predictions: OK")
    tuned_cv = TunedThresholdClassifierCV(make_xgb(), scoring=scorer, cv=5, thresholds=grid).fit(X_tr, y_tr)
    say(f"  TunedThresholdClassifierCV(xgb, cv=5) on train only (refit=True): best_threshold_={tuned_cv.best_threshold_:.4f}; "
        f"test cost/row={-neg_cost(y_te, tuned_cv.predict(X_te)):.4f}")

    # -- d. calibration ---------------------------------------------------------
    say("\n2d. Calibration on test; calibrators fitted on validation with FrozenEstimator")
    calib = {}
    for name, model, Xva, Xte in (("xgb", xgb, X_va, X_te), ("rf", rf, Xn_va, Xn_te)):
        calib[name] = {"raw": p[name]["te"]}
        for method in ("sigmoid", "isotonic"):
            cc = CalibratedClassifierCV(FrozenEstimator(model), method=method).fit(Xva, y_va)
            calib[name][method] = cc.predict_proba(Xte)[:, 1]
    say("  model/method        |  Brier   | logloss  |  ECE10   | ROC-AUC  |   AP     | mean p   | cost@1/6")
    for name in ("xgb", "rf"):
        for method, pp in calib[name].items():
            say(f"  {name:4s} {method:14s} | {brier_score_loss(y_te, pp):.4f}  | {log_loss(y_te, np.clip(pp,1e-15,1-1e-15)):.4f}  | "
                f"{ece(y_te, pp):.4f}  | {roc_auc_score(y_te, pp):.4f}  | {average_precision_score(y_te, pp):.4f}  | "
                f"{pp.mean():.4f}  | {expected_cost(y_te, pp, C_FP/(C_FP+C_FN)):.4f}")
    say(f"  test prevalence = {y_te.mean():.4f}")
    # share of RF probabilities near the extremes
    for name in ("xgb", "rf"):
        pp = calib[name]["raw"]
        say(f"  {name} raw: share of test probabilities < 0.05 = {np.mean(pp < 0.05):.4f}, > 0.95 = {np.mean(pp > 0.95):.4f}")
    rel = {}
    for name in ("xgb", "rf"):
        for method in ("raw", "isotonic"):
            frac, mean_pred = calibration_curve(y_te, calib[name][method], n_bins=10, strategy="uniform")
            rel[(name, method)] = (mean_pred, frac)
            say(f"  reliability {name}/{method}: " + ", ".join(f"({m:.2f}->{f:.2f})" for m, f in zip(mean_pred, frac)))

    # -- e. class weighting and resampling ---------------------------------------
    say("\n2e. Class weighting and undersampling (test set; threshold 0.5 unless noted)")
    spw = float((y_tr == 0).sum() / (y_tr == 1).sum())
    say(f"  scale_pos_weight = negatives/positives in train = {spw:.4f}")
    rng_u = np.random.default_rng(SEED)
    tr_pos = np.where(y_tr == 1)[0]
    tr_neg = rng_u.choice(np.where(y_tr == 0)[0], size=len(tr_pos), replace=False)
    under = np.sort(np.concatenate([tr_pos, tr_neg]))
    say(f"  undersampled train: {len(under)} rows, positives {y_tr[under].mean():.4f}")

    variants = {
        "xgb plain": (xgb, X_va, X_te),
        "xgb spw": (make_xgb(spw).fit(X_tr, y_tr), X_va, X_te),
        "xgb under": (make_xgb().fit(X_tr.iloc[under], y_tr[under]), X_va, X_te),
        "rf plain": (rf, Xn_va, Xn_te),
        "rf balanced": (make_rf("balanced").fit(Xn_tr, y_tr), Xn_va, Xn_te),
        "rf under": (make_rf().fit(Xn_tr.iloc[under], y_tr[under]), Xn_va, Xn_te),
    }
    say("  variant      | prec@.5 | rec@.5 |  F1@.5  | ROC-AUC |   AP    |  Brier  | logloss | mean p  | t*(val) | cost@t* | prec@t* | rec@t*")
    weighting = {}
    for vname, (model, Xva, Xte) in variants.items():
        pva, pte = model.predict_proba(Xva)[:, 1], model.predict_proba(Xte)[:, 1]
        r = metric_row(vname, y_te, pte)
        cva = np.array([expected_cost(y_va, pva, t) for t in grid])
        t_star = float(grid[np.argmin(cva)])
        rs = metric_row(vname, y_te, pte, t_star)
        weighting[vname] = pte
        say(f"  {vname:12s} | {r['precision']:.4f}  | {r['recall']:.4f} | {r['F1']:.4f}  | {r['ROC_AUC']:.4f}  | "
            f"{r['AP']:.4f}  | {r['Brier']:.4f}  | {r['logloss']:.4f}  | {r['mean_p']:.4f}  | {t_star:.2f}    | "
            f"{rs['cost']:.4f}  | {rs['precision']:.4f}  | {rs['recall']:.4f}")
    say(f"  test prevalence = {y_te.mean():.4f}")
    # Prior-shift correction for the undersampled XGBoost: p' = p*pi1/(p*pi1 + (1-p)*pi0*...)
    pu = weighting["xgb under"]
    pi = y_tr.mean()
    beta = (1 - pi) / pi  # negatives were kept at rate 1/beta relative to positives
    p_corr = pu / (pu + (1 - pu) * beta)
    say(f"  xgb under, prior-corrected p/(p+(1-p)*{beta:.4f}): Brier={brier_score_loss(y_te, p_corr):.4f} "
        f"logloss={log_loss(y_te, p_corr):.4f} mean p={p_corr.mean():.4f}")
    for vname in ("xgb plain", "xgb spw", "xgb under"):
        frac, mean_pred = calibration_curve(y_te, weighting[vname], n_bins=10)
        rel[("w", vname)] = (mean_pred, frac)

    make_figures(toy, y_te, p, keep_fig, grid, curves, rel)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_figures(toy, y_te, p, keep_fig, grid, curves, rel) -> None:
    FIG_DIR.mkdir(exist_ok=True)

    # Figure 1: toy ROC (step curve, shaded AUC) and toy PR points.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    ax = axes[0]
    style_axes(ax)
    xs, ys = zip(*toy["pts"])
    ax.fill_between(xs, ys, color=TEAL, alpha=0.12, step=None)
    ax.plot(xs, ys, color=TEAL, lw=2, marker="o", ms=4)
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1)
    for (fx, ty), lab in ((toy["pts"][3], "t = 0.75"), (toy["pts"][6], "t = 0.50")):
        ax.annotate(lab, (fx, ty), xytext=(fx + 0.12, ty - 0.12), color=INK,
                    arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8))
    ax.text(0.55, 0.25, "AUC = 28/32 = 0.875", color=TEAL, fontsize=11)
    ax.set_xlabel("False positive rate (FP / 8)")
    ax.set_ylabel("True positive rate (TP / 4)")
    ax.set_title("ROC curve of the 12-row table", loc="left", fontsize=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)

    ax = axes[1]
    style_axes(ax)
    s, y = toy["s"], toy["y"]
    prec_c, rec_c, _ = precision_recall_curve(y, s)
    ax.step(rec_c[::-1], prec_c[::-1], where="pre", color=TEAL, lw=2)
    order = np.argsort(-s)
    hits = 0
    for k, idx in enumerate(order, 1):
        hits += y[idx]
        ax.plot(hits / 4, hits / k, "o", color=TEAL if y[idx] == 1 else GREY, ms=5)
    ax.axhline(4 / 12, color=GREY, ls="--", lw=1)
    ax.text(0.02, 4 / 12 + 0.02, "random = 4/12", color=INK_2, fontsize=10)
    ax.text(0.30, 0.15, "AP = (1 + 1 + 3/4 + 4/7) / 4 = 0.830", color=TEAL, fontsize=11)
    ax.set_xlabel("Recall (TP / 4)")
    ax.set_ylabel("Precision (TP / predicted positive)")
    ax.set_title("Precision-recall of the same table", loc="left", fontsize=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "toy-roc-pr.png", dpi=160)
    plt.close(fig)

    # Figure 2: Adult ROC vs PR, full test and rare-positive test.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    yy_r, pp_r = y_te[keep_fig], p["xgb"]["te"][keep_fig]
    for ax, kind in zip(axes, ("roc", "pr")):
        style_axes(ax)
        for yy, pp, color, ls, lab in ((y_te, p["xgb"]["te"], TEAL, "-", f"XGBoost, positives {y_te.mean():.0%}"),
                                       (yy_r, pp_r, AMBER, "--", f"same model, positives {yy_r.mean():.1%}")):
            if kind == "roc":
                fx, ty, _ = roc_curve(yy, pp)
                ax.plot(fx, ty, color=color, ls=ls, lw=2, label=f"{lab}: AUC {roc_auc_score(yy, pp):.3f}")
            else:
                pr, rc, _ = precision_recall_curve(yy, pp)
                ax.plot(rc, pr, color=color, ls=ls, lw=2, label=f"{lab}: AP {average_precision_score(yy, pp):.3f}")
                ax.axhline(yy.mean(), color=color, ls=":", lw=1)
        if kind == "roc":
            ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.set_title("ROC: hardly moves", loc="left", fontsize=12)
        else:
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Precision-recall: drops sharply", loc="left", fontsize=12)
        ax.legend(loc="lower right" if kind == "roc" else "lower left", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "adult-roc-vs-pr.png", dpi=160)
    plt.close(fig)

    # Figure 3: expected cost per row vs threshold.
    fig, ax = plt.subplots(figsize=(8, 4.4))
    style_axes(ax)
    for name, color, lab in (("xgb", TEAL, "XGBoost"), ("rf", AMBER, "Random forest")):
        cva, cte = curves[name]
        ax.plot(grid, cva, color=color, lw=2, label=f"{lab} (validation)")
        ax.plot(grid, cte, color=color, lw=1.2, ls=":", label=f"{lab} (test)")
        tb = grid[np.argmin(cva)]
        ax.plot([tb], [cva.min()], "o", color=color, ms=6)
    ax.axvline(0.5, color=GREY, ls="--", lw=1)
    ax.text(0.505, ax.get_ylim()[1] * 0.95, "default 0.5", color=INK_2, fontsize=9, va="top")
    ax.axvline(1 / 6, color=INK_2, ls="-.", lw=1)
    ax.text(1 / 6 + 0.005, ax.get_ylim()[1] * 0.85, "1/6 = C_FP/(C_FP+C_FN)", color=INK_2, fontsize=9, va="top")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Expected cost per row (FP = 1, FN = 5)")
    ax.set_title("Cost of each threshold on the Adult data", loc="left", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cost-vs-threshold.png", dpi=160)
    plt.close(fig)

    # Figure 4: reliability curves.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    ax = axes[0]
    style_axes(ax)
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1, label="perfectly calibrated")
    for (name, method), color, ls, lab in (((("rf", "raw")), AMBER, "-", "Random forest, raw"),
                                           ((("rf", "isotonic")), AMBER, ":", "Random forest + isotonic"),
                                           ((("xgb", "raw")), TEAL, "-", "XGBoost, raw"),
                                           ((("xgb", "isotonic")), TEAL, ":", "XGBoost + isotonic")):
        mp, fr = rel[(name, method)]
        ax.plot(mp, fr, color=color, ls=ls, lw=2, marker="o", ms=3.5, label=lab)
    ax.set_xlabel("Mean predicted probability (bin)")
    ax.set_ylabel("Observed fraction of positives")
    ax.set_title("Before / after isotonic calibration", loc="left", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax = axes[1]
    style_axes(ax)
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1, label="perfectly calibrated")
    for vname, color, ls, lab in (("xgb plain", TEAL, "-", "XGBoost"),
                                  ("xgb spw", AMBER, "-", "XGBoost, scale_pos_weight"),
                                  ("xgb under", INK_2, ":", "XGBoost, undersampled 50/50")):
        mp, fr = rel[("w", vname)]
        ax.plot(mp, fr, color=color, ls=ls, lw=2, marker="o", ms=3.5, label=lab)
    ax.set_xlabel("Mean predicted probability (bin)")
    ax.set_ylabel("Observed fraction of positives")
    ax.set_title("Reweighting moves probabilities up", loc="left", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "reliability.png", dpi=160)
    plt.close(fig)
    say(f"\nfigures written: {', '.join(sorted(f.name for f in FIG_DIR.glob('*.png')))}")


if __name__ == "__main__":
    say(f"python {sys.version.split()[0]} | scikit-learn {sklearn.__version__} | xgboost {xgboost.__version__} | "
        f"numpy {np.__version__} | pandas {pd.__version__}")
    say(f"seed {SEED}")
    toy = part_toy()
    part_adult(toy)
    (HERE / "results.txt").write_text(out.getvalue(), encoding="utf-8")
