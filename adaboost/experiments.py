"""AdaBoost experiments for the blog post "adaboost".

What this script does (all numbers in the post come from here):

1. Hand calculation on an 8-row toy dataset, done with exact fractions, and
   checked against scikit-learn's AdaBoostClassifier (SAMME, the only
   algorithm left in scikit-learn >= 1.8).
2. The textbook variant with alpha = 1/2 * ln((1 - err) / err) gives the same
   weights and the same predictions (it only rescales the score).
3. AdaBoost as forward stagewise additive modelling with exponential loss:
   a small re-implementation is compared with AdaBoostClassifier.
4. Breast cancer (398/171 stratified split, random_state=42): training and
   test error versus the number of rounds.
5. Label noise: 15% of training labels flipped (10 noise draws), AdaBoost vs
   GradientBoostingClassifier, and how much sample weight AdaBoost puts on the
   flipped rows.
6. GradientBoostingClassifier(loss="exponential") vs AdaBoostClassifier.

Run:  python adaboost/experiments.py
Output: adaboost/results.txt and adaboost/figures/*.png
"""

from __future__ import annotations

import inspect
import io
import sys
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import sklearn
from matplotlib import font_manager
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Split convention shared with the sibling posts (gradient-boosting, out-of-bag).
SPLIT = dict(test_size=0.3, random_state=42)
NOISE_RATE = 0.15
NOISE_SEEDS = list(range(10))
MAX_ROUNDS = 1000
CHECKPOINTS = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000]

# ---------------------------------------------------------------- output ---
_buffer = io.StringIO()


def out(*args) -> None:
    text = " ".join(str(a) for a in args)
    print(text)
    _buffer.write(text + "\n")


def section(title: str) -> None:
    out()
    out("=" * 78)
    out(title)
    out("=" * 78)


# ------------------------------------------------------------ paper look ---
PAGE = "#FAF8F3"
INK = "#1C1B19"
INK3 = "#6B6760"
LINE = "#E7E2D8"
TEAL = "#0F766E"
AMBER = "#D97706"
GREY = "#9A958C"


def plain_log_x(ax) -> None:
    """Log x axis labelled 1, 10, 100, 1000 instead of powers of ten."""
    ax.set_xscale("log")
    ax.set_xticks([1, 10, 100, 1000])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))


def setup_style() -> None:
    names = {f.name for f in font_manager.fontManager.ttflist}
    thai = [n for n in ("Leelawadee UI", "Tahoma", "Noto Sans Thai") if n in names]
    plt.rcParams.update(
        {
            "font.family": thai + ["DejaVu Sans"],
            "figure.facecolor": PAGE,
            "axes.facecolor": PAGE,
            "savefig.facecolor": PAGE,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": LINE,
            "grid.linewidth": 1.0,
            "font.size": 11,
            "legend.frameon": False,
        }
    )


# ===================================================== 1. hand calculation ===
# Toy data: one feature x = 1..8, labels in {-1, +1}. Row 6 is the "odd one"
# a single threshold cannot get right.
TOY_X = np.arange(1, 9)
TOY_Y = np.array([+1, +1, +1, -1, -1, +1, -1, -1])


def stump_predict(x, threshold, left_label):
    """Predict left_label when x <= threshold, the other label otherwise."""
    return np.where(x <= threshold, left_label, -left_label)


def weighted_gini(y, w, mask):
    """Weighted Gini impurity of the split left=mask, right=~mask (sklearn criterion)."""
    total = w.sum()
    impurity = 0.0
    for side in (mask, ~mask):
        ws = w[side].sum()
        if ws == 0:
            continue
        p = w[side & (y == 1)].sum() / ws
        impurity += ws / total * 2 * p * (1 - p)
    return impurity


def hand_adaboost(x, y, rounds):
    """Discrete AdaBoost (SAMME, two classes) with exact fractions.

    The stump of each round minimises the weighted error over all
    thresholds (midpoints) and both orientations.
    """
    n = len(y)
    w = [Fraction(1, n)] * n
    history = []
    for m in range(rounds):
        candidates = []
        for t in (x[:-1] + x[1:]) / 2:
            for left in (+1, -1):
                p = stump_predict(x, t, left)
                err = sum(w[i] for i in range(n) if p[i] != y[i])
                candidates.append((err, float(t), left))
        candidates.sort()
        err, t, left = candidates[0]
        tie = candidates[1][0] == err
        p = stump_predict(x, t, left)
        ratio = (1 - err) / err  # exp(alpha)
        new_w = [w[i] * (ratio if p[i] != y[i] else 1) for i in range(n)]
        z = sum(new_w)
        new_w = [wi / z for wi in new_w]
        wf = np.array([float(v) for v in w])
        gini = {
            (c_t, c_left): weighted_gini(y, wf, x <= c_t)
            for _, c_t, c_left in candidates
        }
        history.append(
            dict(
                round=m + 1,
                weights=w,
                threshold=t,
                left=left,
                err=err,
                tie=tie,
                ratio=ratio,
                alpha=float(np.log(float(ratio))),
                pred=p,
                new_weights=new_w,
                misclassified_weight_after=sum(
                    new_w[i] for i in range(n) if p[i] != y[i]
                ),
                gini=gini,
            )
        )
        w = new_w
    return history


def fmt_frac(v: Fraction) -> str:
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


def part1_hand_calculation():
    section("1. Hand calculation on the 8-row toy dataset (SAMME, exact fractions)")
    x, y = TOY_X, TOY_Y
    out("x =", x.tolist())
    out("y =", y.tolist())
    hist = hand_adaboost(x, y, rounds=3)
    for h in hist:
        out()
        out(f"Round {h['round']}")
        out("  weights in      :", [fmt_frac(v) for v in h["weights"]])
        side = "+1 if x <= t else -1" if h["left"] == 1 else "-1 if x <= t else +1"
        out(f"  best stump      : t = {h['threshold']}, predict {side}")
        out("  predictions     :", h["pred"].tolist())
        wrong = [i + 1 for i in range(len(y)) if h["pred"][i] != y[i]]
        out("  wrong rows      :", wrong)
        out(f"  weighted error  : {fmt_frac(h['err'])} = {float(h['err']):.6f}  (tie: {h['tie']})")
        out(f"  exp(alpha)      : (1-err)/err = {fmt_frac(h['ratio'])}")
        out(f"  alpha (SAMME)   : ln({fmt_frac(h['ratio'])}) = {h['alpha']:.6f}")
        out(f"  textbook 1/2 alpha: {h['alpha'] / 2:.6f}")
        out("  weights out     :", [fmt_frac(v) for v in h["new_weights"]])
        out(f"  weight on wrong rows after update: {fmt_frac(h['misclassified_weight_after'])}")
        # Weighted Gini of the stumps sklearn could pick, best three.
        # Gini does not depend on the orientation, so keep one entry per threshold.
        per_threshold = {t: g for (t, _), g in h["gini"].items()}
        best_gini = sorted(per_threshold.items(), key=lambda kv: kv[1])[:3]
        out("  lowest weighted Gini splits (t, impurity):",
            [(t, round(float(g), 6)) for t, g in best_gini])

    alphas = np.array([h["alpha"] for h in hist])
    preds = np.array([h["pred"] for h in hist])  # rounds x rows
    F = alphas @ preds
    out()
    out("Final score F(x) = sum alpha_m h_m(x):", np.round(F, 4).tolist())
    out("sign(F)        :", np.sign(F).astype(int).tolist())
    out("training error after 1, 2, 3 rounds:",
        [float((np.sign(np.cumsum(alphas[:k, None] * preds[:k], axis=0)[-1]) != y).mean())
         for k in (1, 2, 3)])

    # ---- check against scikit-learn ---------------------------------------
    out()
    out("Check against sklearn AdaBoostClassifier(n_estimators=3)")
    X = x.reshape(-1, 1)
    clf = AdaBoostClassifier(n_estimators=3, random_state=0).fit(X, y)
    thr = [float(e.tree_.threshold[0]) for e in clf.estimators_]
    out("  sklearn thresholds        :", thr)
    out("  sklearn estimator_errors_ :", np.round(clf.estimator_errors_, 6).tolist())
    out("  sklearn estimator_weights_:", np.round(clf.estimator_weights_, 6).tolist())
    out("  hand errors               :", [round(float(h["err"]), 6) for h in hist])
    out("  hand alphas               :", [round(h["alpha"], 6) for h in hist])
    assert np.allclose(thr, [h["threshold"] for h in hist])
    assert np.allclose(clf.estimator_errors_, [float(h["err"]) for h in hist])
    assert np.allclose(clf.estimator_weights_, alphas)
    staged = [float((p != y).mean()) for p in clf.staged_predict(X)]
    out("  sklearn staged training error:", staged)
    dec = clf.decision_function(X)
    out("  sklearn decision_function :", np.round(dec, 6).tolist())
    out("  F / sum(alpha)            :", np.round(F / alphas.sum(), 6).tolist())
    out("  ratio decision / (F/sum)  :",
        np.round(dec / (F / alphas.sum()), 6).tolist())
    proba = clf.predict_proba(X)[:, 1]
    out("  sklearn predict_proba[:,1]:", np.round(proba, 4).tolist())
    assert (clf.predict(X) == np.sign(F)).all()
    out("  predictions identical to hand calculation: True")

    # learning_rate: alpha is multiplied by it, and so is the weight update.
    clf_lr = AdaBoostClassifier(n_estimators=3, learning_rate=0.5, random_state=0).fit(X, y)
    out()
    out("learning_rate=0.5 on the same data:")
    out("  estimator_weights_:", np.round(clf_lr.estimator_weights_, 6).tolist())
    out("  0.5 * ln(7) =", round(0.5 * np.log(7), 6),
        "-> misclassified row 6 multiplied by exp(0.5 ln 7) = sqrt(7) =", round(np.sqrt(7), 6))
    w = np.full(8, 1 / 8)
    w[5] *= np.sqrt(7)
    w /= w.sum()
    out("  weights entering round 2 with learning_rate=0.5:", np.round(w, 6).tolist(),
        " weight of row 6 =", round(w[5], 6))

    # ---- textbook variant: alpha/2 and the symmetric update ----------------
    out()
    out("Textbook variant: beta = 1/2 ln((1-err)/err), w <- w * exp(-beta * y * h)")
    w = np.full(8, 1 / 8)
    betas = []
    for h in hist:
        err = w[h["pred"] != y].sum()
        beta = 0.5 * np.log((1 - err) / err)
        betas.append(beta)
        w = w * np.exp(-beta * y * h["pred"])
        w /= w.sum()
        hand_w = np.array([float(v) for v in h["new_weights"]])
        out(f"  round {h['round']}: beta = {beta:.6f}, normalised weights equal SAMME's: "
            f"{np.allclose(w, hand_w)}")
    out("  sum beta h = F/2, same sign as F:",
        bool((np.sign(np.array(betas) @ preds) == np.sign(F)).all()))
    return hist


# ======================================================= figure 1: weights ===
def figure_row_weights(hist):
    setup_style()
    stages = [h["weights"] for h in hist] + [hist[-1]["new_weights"]]
    labels = ["รอบที่ 1", "รอบที่ 2", "รอบที่ 3", "หลังรอบที่ 3"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6), sharey=True)
    for k, (ax, w, lab) in enumerate(zip(axes, stages, labels)):
        wf = [float(v) for v in w]
        if k < 3:
            wrong = hist[k]["pred"] != TOY_Y
        else:
            wrong = np.zeros(8, dtype=bool)
        colors = [TEAL if TOY_Y[i] == 1 else GREY for i in range(8)]
        bars = ax.bar(TOY_X, wf, color=colors, width=0.7)
        for i, b in enumerate(bars):
            if wrong[i]:
                b.set_edgecolor(AMBER)
                b.set_linewidth(2.5)
            ax.text(b.get_x() + b.get_width() / 2, wf[i] + 0.01,
                    fmt_frac(w[i]), ha="center", va="bottom", fontsize=8.5, color=INK)
        if k < 3:
            ax.axvline(hist[k]["threshold"], color=INK3, linestyle="--", linewidth=1)
        ax.set_title(lab, fontsize=12)
        ax.set_xticks(TOY_X)
        ax.set_xlabel("x (แถว)")
        ax.set_ylim(0, 0.6)
    axes[0].set_ylabel("น้ำหนักของแถว")
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=TEAL, label="y = +1"),
        Patch(facecolor=GREY, label="y = −1"),
        Patch(facecolor="none", edgecolor=AMBER, linewidth=2.5, label="stump รอบนี้ทายผิด"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = FIG_DIR / "row-weights.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    out("saved", path.name)


# ============================================= 3. stagewise exponential loss ===
def fsam_exponential(X, y, rounds):
    """Forward stagewise additive modelling with exponential loss.

    y in {-1, +1}. At stage m the weights are w_i = exp(-y_i F(x_i)), the
    base learner is the same depth-1 sklearn tree AdaBoost uses, fitted with
    those weights, and the step beta is the exact line search
    beta = 1/2 ln((1 - err) / err).
    """
    F = np.zeros(len(y))
    betas, learners = [], []
    for _ in range(rounds):
        w = np.exp(-y * F)
        w /= w.sum()
        tree = DecisionTreeClassifier(max_depth=1).fit(X, y, sample_weight=w)
        h = tree.predict(X)
        err = w[h != y].sum()
        beta = 0.5 * np.log((1 - err) / err)
        F += beta * h
        betas.append(beta)
        learners.append(tree)
    return np.array(betas), learners


def part3_fsam(Xtr, ytr_pm, Xte):
    section("3. AdaBoost = forward stagewise additive modelling with exponential loss")
    rounds = 50
    betas, learners = fsam_exponential(Xtr, ytr_pm, rounds)
    ada = AdaBoostClassifier(n_estimators=rounds, random_state=0).fit(Xtr, ytr_pm)
    out(f"rounds = {rounds} on breast cancer train set")
    out("max |2*beta - sklearn alpha| :", float(np.max(np.abs(2 * betas - ada.estimator_weights_))))
    same_split = all(
        a.tree_.feature[0] == b.tree_.feature[0]
        and np.isclose(a.tree_.threshold[0], b.tree_.threshold[0])
        for a, b in zip(learners, ada.estimators_)
    )
    out("same stump (feature, threshold) every round:", same_split)
    F_te = sum(b * t.predict(Xte) for b, t in zip(betas, learners))
    out("predictions on test set identical to AdaBoostClassifier:",
        bool((np.sign(F_te) == ada.predict(Xte)).all()))

    # Which stump does each criterion like? sklearn grows the stump with
    # weighted Gini; the exponential-loss derivation asks for the stump with
    # the smallest weighted error. Count rounds where they disagree.
    w = np.full(len(ytr_pm), 1 / len(ytr_pm))
    disagree = 0
    err_gap = []
    for est, a in zip(ada.estimators_, ada.estimator_weights_):
        h = est.predict(Xtr)
        err_gini = w[h != ytr_pm].sum()
        best = min_weighted_error_stump(Xtr, ytr_pm, w)
        if best < err_gini - 1e-12:
            disagree += 1
            err_gap.append(err_gini - best)
        w = w * np.exp(a * (h != ytr_pm))
        w /= w.sum()
    out(f"rounds (of {rounds}) where the Gini-chosen stump does NOT have the smallest "
        f"weighted error: {disagree}")
    if err_gap:
        out(f"  largest gap in weighted error: {max(err_gap):.4f}, mean gap: {np.mean(err_gap):.4f}")


def min_weighted_error_stump(X, y, w):
    """Smallest weighted error over all axis-aligned stumps (both orientations)."""
    best = np.inf
    for j in range(X.shape[1]):
        order = np.argsort(X[:, j], kind="stable")
        xs, ys, ws = X[order, j], y[order], w[order]
        # error of "predict -1 on the left, +1 on the right" for every cut
        left_pos = np.concatenate([[0.0], np.cumsum(ws * (ys == 1))])
        right_neg = (ws * (ys == -1)).sum() - np.concatenate([[0.0], np.cumsum(ws * (ys == -1))])
        err = left_pos + right_neg
        # only real cuts between two distinct values (a tree split never
        # puts every row on one side)
        valid = np.concatenate([[False], xs[1:] != xs[:-1], [False]])
        err = err[valid]
        best = min(best, err.min(), (1 - err).min())
    return best


# ================================================ 4. clean breast cancer ===
def staged_errors(model, X, y):
    return np.array([np.mean(p != y) for p in model.staged_predict(X)])


def part4_clean(Xtr, Xte, ytr, yte):
    section("4. Breast cancer, clean labels: error vs number of rounds")
    out(f"train rows = {len(ytr)}, test rows = {len(yte)}")
    results = {}
    for lr in (1.0, 0.1):
        ada = AdaBoostClassifier(n_estimators=MAX_ROUNDS, learning_rate=lr,
                                 random_state=0).fit(Xtr, ytr)
        tr = staged_errors(ada, Xtr, ytr)
        te = staged_errors(ada, Xte, yte)
        results[lr] = (tr, te)
        out()
        out(f"AdaBoost stumps, learning_rate={lr}, fitted rounds = {len(ada.estimators_)}")
        out("  round | train err | test err")
        for c in CHECKPOINTS:
            if c <= len(te):
                out(f"  {c:5d} | {tr[c - 1]:.4f}    | {te[c - 1]:.4f}")
        zero = np.nonzero(tr == 0)[0]
        out("  first round with train error 0:", int(zero[0]) + 1 if len(zero) else "never")
        if len(zero):
            after = te[zero[0]:]
            out(f"  test error after train error hit 0: min {after.min():.4f}, "
                f"max {after.max():.4f}, at round 1000 {te[-1]:.4f}")
        out(f"  best test error {te.min():.4f} first at round {int(np.argmin(te)) + 1}")
        # margins keep growing after training error reaches 0
        if lr == 1.0:
            dec = ada.staged_decision_function(Xtr)
            y_pm = np.where(ytr == 1, 1, -1)
            margins = [np.min(d * y_pm) for d in dec]
            out("  min training margin y*decision_function at rounds 50/100/200/500/1000:",
                [round(float(margins[c - 1]), 4) for c in (50, 100, 200, 500, 1000)])
    return results


# ================================================== 5. label-noise study ===
def flip_labels(y01, rate, seed):
    rng = np.random.default_rng(seed)
    n = len(y01)
    idx = rng.choice(n, size=int(round(rate * n)), replace=False)
    y_noisy = y01.copy()
    y_noisy[idx] = 1 - y_noisy[idx]
    return y_noisy, idx


def part5_noise(Xtr, Xte, ytr01, yte01):
    section("5. Label noise: 15% of training labels flipped, 10 noise draws")
    out("flip procedure: numpy default_rng(seed).choice(n, round(0.15 n), replace=False), "
        f"seeds {NOISE_SEEDS}")
    ada_te, gb_te, ada_tr, share = [], [], [], []
    for seed in NOISE_SEEDS:
        yn, idx = flip_labels(ytr01, NOISE_RATE, seed)
        ada = AdaBoostClassifier(n_estimators=MAX_ROUNDS, random_state=0).fit(Xtr, yn)
        gb = GradientBoostingClassifier(n_estimators=MAX_ROUNDS, learning_rate=0.1,
                                        random_state=42).fit(Xtr, yn)
        ada_te.append(staged_errors(ada, Xte, yte01))
        ada_tr.append(staged_errors(ada, Xtr, yn))
        gb_te.append(staged_errors(gb, Xte, yte01))
        # AdaBoost's sample weights after m rounds:
        #   w_i  proportional to  exp( sum_{k<=m} alpha_k * 1[h_k(x_i) != y_i] )
        wrong = np.array([e.predict(Xtr) != yn for e in ada.estimators_])
        logw = np.cumsum(ada.estimator_weights_[:, None] * wrong, axis=0)
        logw -= logw.max(axis=1, keepdims=True)
        w = np.exp(logw)
        w /= w.sum(axis=1, keepdims=True)
        flipped = np.zeros(len(yn), dtype=bool)
        flipped[idx] = True
        share.append(w[:, flipped].sum(axis=1))
        out(f"  seed {seed}: flipped {len(idx)} rows; AdaBoost test err @10 {ada_te[-1][9]:.4f} "
            f"@1000 {ada_te[-1][-1]:.4f} | GB(lr=0.1) @10 {gb_te[-1][9]:.4f} @1000 {gb_te[-1][-1]:.4f}")
    ada_te, gb_te, ada_tr, share = map(np.array, (ada_te, gb_te, ada_tr, share))
    out()
    out("Mean over the 10 noise draws (test labels are clean):")
    out("  round | AdaBoost train err (noisy labels) | AdaBoost test | GB lr=0.1 test | "
        "weight share on flipped rows")
    for c in CHECKPOINTS:
        out(f"  {c:5d} | {ada_tr[:, c - 1].mean():.4f} | {ada_te[:, c - 1].mean():.4f} "
            f"(sd {ada_te[:, c - 1].std():.4f}) | {gb_te[:, c - 1].mean():.4f} "
            f"(sd {gb_te[:, c - 1].std():.4f}) | {share[:, c - 1].mean():.3f}")
    m_ada, m_gb = ada_te.mean(0), gb_te.mean(0)
    out(f"  AdaBoost mean curve: best {m_ada.min():.4f} at round {int(np.argmin(m_ada)) + 1}, "
        f"at 1000 {m_ada[-1]:.4f}")
    out(f"  GB mean curve      : best {m_gb.min():.4f} at round {int(np.argmin(m_gb)) + 1}, "
        f"at 1000 {m_gb[-1]:.4f}")
    out(f"  flipped rows are {NOISE_RATE:.0%} of the training rows "
        f"({int(round(NOISE_RATE * len(ytr01)))} of {len(ytr01)})")
    return ada_te, gb_te, ada_tr, share


# ============================================= figure 2 and figure 3 ======
def figure_error_vs_rounds(clean, ada_te, gb_te, ada_tr):
    setup_style()
    rounds = np.arange(1, MAX_ROUNDS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    tr, te = clean[1.0]
    ax = axes[0]
    ax.plot(rounds, tr, color=GREY, linewidth=2, label="train error")
    ax.plot(rounds, te, color=TEAL, linewidth=2.2, label="test error")
    ax.set_title("ข้อมูลสะอาด")
    plain_log_x(ax)
    ax.set_xlabel("จำนวนรอบ (stump)")
    ax.set_ylabel("error")
    ax.legend(loc="upper right")
    ax = axes[1]
    ax.plot(rounds, ada_tr.mean(0), color=GREY, linewidth=2, label="AdaBoost train error (label ที่สลับ)")
    ax.plot(rounds, ada_te.mean(0), color=TEAL, linewidth=2.2, label="AdaBoost test error")
    ax.plot(rounds, gb_te.mean(0), color=AMBER, linewidth=2, linestyle="--",
            label="Gradient boosting (ν = 0.1) test error")
    ax.set_title("ฝึกด้วย label ที่สลับ 15% (เฉลี่ย 10 ครั้ง)")
    plain_log_x(ax)
    ax.set_xlabel("จำนวนรอบ")
    ax.legend(loc="upper right", fontsize=9.5)
    axes[0].set_ylim(0, 0.12)
    axes[1].set_ylim(0, 0.4)
    fig.tight_layout()
    path = FIG_DIR / "error-vs-rounds.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    out("saved", path.name)


def figure_noise_weight(share):
    setup_style()
    rounds = np.arange(1, MAX_ROUNDS + 1)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for s in share:
        ax.plot(rounds, s, color=TEAL, alpha=0.18, linewidth=1)
    ax.plot(rounds, share.mean(0), color=TEAL, linewidth=2.4, label="เฉลี่ย 10 ครั้ง")
    ax.axhline(NOISE_RATE, color=AMBER, linestyle="--", linewidth=1.6,
               label="สัดส่วนจำนวนแถวที่ label ผิด (15%)")
    plain_log_x(ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("จำนวนรอบ")
    ax.set_ylabel("สัดส่วนน้ำหนักบนแถวที่ label ผิด")
    ax.legend(loc="upper left")
    fig.tight_layout()
    path = FIG_DIR / "noise-weight.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    out("saved", path.name)


# =========================================== 6. GB exponential vs AdaBoost ===
def part6_gb_exponential(Xtr, Xte, ytr01, yte01):
    section("6. GradientBoostingClassifier(loss='exponential') vs AdaBoostClassifier")

    # (a) the toy dataset, three rounds, as close a setting as possible
    X = TOY_X.reshape(-1, 1)
    y01 = (TOY_Y == 1).astype(int)
    ada = AdaBoostClassifier(n_estimators=3, random_state=0).fit(X, y01)
    F_ada_half = 0.5 * sum(a * np.where(e.predict(X) == 1, 1, -1)
                           for a, e in zip(ada.estimator_weights_, ada.estimators_))
    gbe = GradientBoostingClassifier(loss="exponential", n_estimators=3, learning_rate=1.0,
                                     max_depth=1, init="zero", random_state=0).fit(X, y01)
    out("(a) toy data, 3 rounds, GB: max_depth=1, learning_rate=1, init='zero'")
    out("  AdaBoost stump thresholds:", [float(e.tree_.threshold[0]) for e in ada.estimators_])
    out("  GB-exp   stump thresholds:", [float(t[0].tree_.threshold[0]) for t in gbe.estimators_])
    for k, raw in enumerate(gbe.staged_decision_function(X), start=1):
        out(f"  GB-exp raw score after round {k}:", np.round(raw.ravel(), 4).tolist())
    out("  AdaBoost F/2 = sum beta h after 3 rounds :", np.round(F_ada_half, 4).tolist())
    out("  GB-exp leaf values per round:",
        [np.round(t[0].tree_.value[t[0].tree_.children_left == -1].ravel(), 4).tolist()
         for t in gbe.estimators_])
    out("  training error AdaBoost / GB-exp:",
        float((ada.predict(X) != y01).mean()), float((gbe.predict(X) != y01).mean()))

    # (b) breast cancer
    out()
    out("(b) breast cancer, clean labels")
    configs = [
        ("AdaBoostClassifier (stumps, lr=1)", lambda n: AdaBoostClassifier(n_estimators=n, random_state=0)),
        ("GB exponential, depth 1, lr=1, init=zero",
         lambda n: GradientBoostingClassifier(loss="exponential", n_estimators=n, learning_rate=1.0,
                                              max_depth=1, init="zero", random_state=42)),
        ("GB exponential, defaults (depth 3, lr=0.1)",
         lambda n: GradientBoostingClassifier(loss="exponential", n_estimators=n, random_state=42)),
        ("GB log_loss, defaults (depth 3, lr=0.1)",
         lambda n: GradientBoostingClassifier(n_estimators=n, random_state=42)),
    ]
    preds = {}
    out("  model | test err @50 | @100 | @200 | @500")
    for name, make in configs:
        model = make(500).fit(Xtr, ytr01)
        te = staged_errors(model, Xte, yte01)
        preds[name] = model.predict(Xte)
        out(f"  {name} | {te[49]:.4f} | {te[99]:.4f} | {te[199]:.4f} | {te[499]:.4f}")
    names = [c[0] for c in configs]
    base = preds[names[0]]
    for name in names[1:]:
        out(f"  test-set agreement with AdaBoost (500 rounds), {name}: "
            f"{np.mean(preds[name] == base):.4f}")
    ada = AdaBoostClassifier(n_estimators=5, random_state=0).fit(Xtr, ytr01)
    gbe = GradientBoostingClassifier(loss="exponential", n_estimators=5, learning_rate=1.0,
                                     max_depth=1, init="zero", random_state=42).fit(Xtr, ytr01)
    out("  first 5 stumps (feature, threshold) AdaBoost:",
        [(int(e.tree_.feature[0]), round(float(e.tree_.threshold[0]), 4)) for e in ada.estimators_])
    out("  first 5 stumps (feature, threshold) GB-exp  :",
        [(int(t[0].tree_.feature[0]), round(float(t[0].tree_.threshold[0]), 4)) for t in gbe.estimators_])

    # (c) noisy labels, same 10 draws as part 5
    out()
    out("(c) 15% flipped labels, mean test error over the same 10 draws")
    rows = {n: [] for n in names[:3]}
    for seed in NOISE_SEEDS:
        yn, _ = flip_labels(ytr01, NOISE_RATE, seed)
        for name, make in configs[:3]:
            rows[name].append(staged_errors(make(500).fit(Xtr, yn), Xte, yte01))
    for name in names[:3]:
        m = np.mean(rows[name], axis=0)
        out(f"  {name}: @10 {m[9]:.4f} @50 {m[49]:.4f} @100 {m[99]:.4f} @500 {m[499]:.4f} "
            f"best {m.min():.4f} at {int(np.argmin(m)) + 1}")


# ================================================== parameters & defaults ===
def part0_versions():
    section("0. Versions and defaults")
    out("python", sys.version.split()[0], "| scikit-learn", sklearn.__version__,
        "| numpy", np.__version__)
    out("AdaBoostClassifier signature:", inspect.signature(AdaBoostClassifier.__init__))
    out("has 'algorithm' parameter:",
        "algorithm" in inspect.signature(AdaBoostClassifier.__init__).parameters)
    ada = AdaBoostClassifier().fit(TOY_X.reshape(-1, 1), TOY_Y)
    out("default estimator_ :", ada.estimator_)
    out("GradientBoostingClassifier loss options: 'log_loss' (default), 'exponential'")


def main():
    np.set_printoptions(linewidth=120)
    part0_versions()
    hist = part1_hand_calculation()
    figure_row_weights(hist)

    X, y01 = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr01, yte01 = train_test_split(X, y01, stratify=y01, **SPLIT)
    ytr_pm = np.where(ytr01 == 1, 1, -1)

    part3_fsam(Xtr, ytr_pm, Xte)
    clean = part4_clean(Xtr, Xte, ytr01, yte01)
    ada_te, gb_te, ada_tr, share = part5_noise(Xtr, Xte, ytr01, yte01)
    figure_error_vs_rounds(clean, ada_te, gb_te, ada_tr)
    figure_noise_weight(share)
    part6_gb_exponential(Xtr, Xte, ytr01, yte01)

    (HERE / "results.txt").write_text(_buffer.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
