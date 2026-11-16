"""Cross-validation experiments for the "cross-validation" blog post.

Every number the post calls "measured" comes from this script:

1. A 12-row hand example (labels + group ids): the test folds of KFold,
   StratifiedKFold, GroupKFold, StratifiedGroupKFold and TimeSeriesSplit
   (with and without gap), asserted against the folds worked out by hand.
2. What an integer ``cv`` turns into (check_cv) for classifiers and regressors.
3. Variance of a CV estimate on Adult: many independent training samples of
   n = 1000 rows, each scored by 2/5/10-fold and repeated 5-fold CV, against
   the accuracy of the same model on a large untouched evaluation set.
4. Target-encoding leakage on Adult: a high-cardinality "profile" key encoded
   before the CV loop vs inside a Pipeline, against a held-out test set.
5. Group leakage on synthetic repeated measurements (documented generator):
   KFold vs GroupKFold vs accuracy on brand-new subjects.
6. Time leakage on the hourly Bike Sharing data: shuffled KFold, unshuffled
   KFold, TimeSeriesSplit (gap 0 and 48) vs a true future hold-out.
7. Nested vs non-nested CV on breast cancer (decision-tree grid search),
   repeated over 20 train/test splits.
8. cross_val_score vs cross_validate vs cross_val_predict.

Run:  .venv/Scripts/python.exe cross-validation/experiments.py
Output: cross-validation/results.txt and PNG figures in cross-validation/figures/.
Datasets from OpenML are cached under data_cache/ at the repository root.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from sklearn.base import BaseEstimator, TransformerMixin, clone  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.datasets import fetch_openml, load_breast_cancer  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV,
    GroupKFold,
    KFold,
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
    check_cv,
    cross_val_predict,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE = ROOT / "data_cache"
FIG_DIR = HERE / "figures"
SEED = 42

# "Paper" palette used by the blog's figures.
PAPER = "#F8F6F1"
INK = "#1F2328"
INK_2 = "#5B6270"
GRID = "#E8E2D6"
TEAL = "#0F766E"
TEAL_LIGHT = "#A7D3CE"
AMBER = "#F59E0B"
GREY = "#C9C4B8"

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


# ---------------------------------------------------------------------------
# 1. Hand example: 12 rows, labels and groups
# ---------------------------------------------------------------------------

Y12 = np.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1])
G12 = np.array(list("AAABBBCCDDEE"))
X12 = np.arange(12).reshape(-1, 1)

# Test folds worked out by hand in the post (1-based row numbers).
HAND_TEST_FOLDS = {
    "KFold(3)": [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
    "StratifiedKFold(3)": [[1, 2, 3, 7], [4, 5, 6, 8], [9, 10, 11, 12]],
    "GroupKFold(3)": [[4, 5, 6, 7, 8], [1, 2, 3], [9, 10, 11, 12]],
}


def hand_example() -> dict:
    header("1. Hand example: 12 rows, 4 positives (rows 7, 8, 11, 12), 5 groups")
    say("row    :", " ".join(f"{i:>2d}" for i in range(1, 13)))
    say("y      :", " ".join(f"{v:>2d}" for v in Y12))
    say("group  :", " ".join(f"{g:>2s}" for g in G12))
    say(f"overall positive share = {Y12.mean():.4f}")

    splitters = {
        "KFold(3)": KFold(3),
        "StratifiedKFold(3)": StratifiedKFold(3),
        "GroupKFold(3)": GroupKFold(3),
        "StratifiedGroupKFold(3)": StratifiedGroupKFold(3),
        "TimeSeriesSplit(3)": TimeSeriesSplit(3),
        "TimeSeriesSplit(3, gap=1)": TimeSeriesSplit(3, gap=1),
    }
    layouts = {}
    for name, cv in splitters.items():
        say()
        say(name)
        folds = []
        for k, (tr, te) in enumerate(cv.split(X12, Y12, G12)):
            tr1, te1 = (tr + 1).tolist(), (te + 1).tolist()
            leaked = sorted(str(v) for v in set(G12[tr]) & set(G12[te]))
            say(
                f"  fold {k + 1}: test rows {te1}  test pos {Y12[te].sum()}/{len(te)}"
                f" = {Y12[te].mean():.4f} | train pos {Y12[tr].sum()}/{len(tr)}"
                f" = {Y12[tr].mean():.4f} | train rows {tr1}"
                f" | groups in both: {leaked if leaked else 'none'}"
            )
            folds.append((tr, te))
        layouts[name] = folds
        if name in HAND_TEST_FOLDS:
            got = [(te + 1).tolist() for _, te in folds]
            assert got == HAND_TEST_FOLDS[name], (name, got)
            say("  -> matches the folds worked out by hand")
    return layouts


def figure_fold_layout(layouts: dict) -> None:
    names = [
        "KFold(3)",
        "StratifiedKFold(3)",
        "GroupKFold(3)",
        "StratifiedGroupKFold(3)",
        "TimeSeriesSplit(3, gap=1)",
    ]
    fig, ax = plt.subplots(figsize=(10, 7.4), dpi=150)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    row_h, gap_h = 0.8, 0.9
    y = 0.0
    yticks, ylabels = [], []
    for name in names:
        for k, (tr, te) in enumerate(layouts[name]):
            for i in range(12):
                if i in te:
                    color = TEAL
                elif i in tr:
                    color = TEAL_LIGHT
                else:
                    color = GREY
                ax.add_patch(Rectangle((i, -y - row_h), 0.92, row_h * 0.9, color=color, lw=0))
            yticks.append(-y - row_h / 2)
            ylabels.append(f"{name}  fold {k + 1}")
            y += row_h
        y += gap_h - row_h + 0.3
    # Header rows: label and group of each row
    for i in range(12):
        ax.text(i + 0.46, 1.35, str(i + 1), ha="center", va="center", fontsize=8, color=INK_2)
        ax.text(i + 0.46, 0.85, G12[i], ha="center", va="center", fontsize=10, color=INK)
        ax.add_patch(
            Rectangle((i + 0.16, 0.05), 0.6, 0.45, color=AMBER if Y12[i] == 1 else "white",
                      ec=INK_2, lw=0.6)
        )
    ax.text(-0.2, 1.35, "row", ha="right", va="center", fontsize=8, color=INK_2)
    ax.text(-0.2, 0.85, "group", ha="right", va="center", fontsize=9, color=INK)
    ax.text(-0.2, 0.28, "label y", ha="right", va="center", fontsize=9, color=INK)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8.5, color=INK)
    ax.set_xticks([])
    ax.set_xlim(-0.1, 12.1)
    ax.set_ylim(-y - 0.2, 1.7)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(left=False)
    handles = [
        Patch(color=TEAL, label="test fold"),
        Patch(color=TEAL_LIGHT, label="training rows"),
        Patch(color=GREY, label="not used (future / gap)"),
        Patch(facecolor=AMBER, edgecolor=INK_2, label="y = 1"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=4,
              frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fold-layout.png", facecolor=PAPER)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. What does cv=5 become?
# ---------------------------------------------------------------------------

def what_cv5_means() -> None:
    header("2. check_cv: what an integer cv becomes")
    rng = np.random.default_rng(0)
    y_bin = rng.integers(0, 2, 20)
    y_multi = rng.integers(0, 3, 20)
    y_cont = rng.normal(size=20)
    cases = [
        ("classifier, binary y", y_bin, True),
        ("classifier, multiclass y", y_multi, True),
        ("classifier, continuous y", y_cont, True),
        ("regressor, continuous y", y_cont, False),
        ("regressor, binary y", y_bin, False),
    ]
    for label, y, is_clf in cases:
        say(f"  cv=5, {label:<28s} -> {check_cv(5, y, classifier=is_clf)!r}")
    say(f"  cv=None                                 -> {check_cv(None)!r}")


# ---------------------------------------------------------------------------
# Adult helpers
# ---------------------------------------------------------------------------

def load_adult() -> tuple[pd.DataFrame, np.ndarray]:
    data = fetch_openml("adult", version=2, as_frame=True, data_home=str(CACHE))
    X = data.data.copy()
    y = (data.target.astype(str) == ">50K").astype(int).to_numpy()
    return X, y


def adult_numeric(X: pd.DataFrame) -> np.ndarray:
    """Ordinal codes for categoricals (missing -> NaN), numbers as-is."""
    cols = []
    for c in X.columns:
        if isinstance(X[c].dtype, pd.CategoricalDtype):
            codes = X[c].cat.codes.astype(float).to_numpy().copy()
            codes[codes < 0] = np.nan
            cols.append(codes)
        else:
            cols.append(X[c].astype(float).to_numpy())
    return np.column_stack(cols)


# ---------------------------------------------------------------------------
# 3. Variance of a CV estimate
# ---------------------------------------------------------------------------

def _one_sample(r: int, pool_idx, Xn, y, X_eval, y_eval, n: int) -> dict:
    with threadpool_limits(1):
        rng = np.random.default_rng(1000 + r)
        idx = rng.choice(pool_idx, size=n, replace=False)
        Xs, ys = Xn[idx], y[idx]
        model = HistGradientBoostingClassifier(max_iter=100, early_stopping=False, random_state=0)
        res = {}
        for k in (2, 5, 10):
            s = cross_val_score(model, Xs, ys, cv=StratifiedKFold(k, shuffle=True, random_state=r))
            res[f"cv{k}"] = s.mean()
            res[f"cv{k}_sd"] = s.std(ddof=1)
        rep = cross_val_score(
            model, Xs, ys, cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=r)
        )
        rep_means = rep.reshape(10, 5).mean(axis=1)
        res["rep5x10"] = rep.mean()
        res["rep_sd_of_repeat_means"] = rep_means.std(ddof=1)
        res["true"] = clone(model).fit(Xs, ys).score(X_eval, y_eval)
        return res


def cv_variance(X: pd.DataFrame, y: np.ndarray) -> dict:
    header("3. Variance of a CV estimate (Adult, n = 1000 per training sample)")
    Xn = adult_numeric(X)
    idx_all = np.arange(len(y))
    pool_idx, eval_idx = train_test_split(idx_all, test_size=20000, stratify=y, random_state=SEED)
    X_eval, y_eval = Xn[eval_idx], y[eval_idx]
    R, n = 40, 1000
    say(f"pool rows {len(pool_idx)}, evaluation rows {len(eval_idx)}, repeats R={R}, n={n}")
    say("model: HistGradientBoostingClassifier(max_iter=100, early_stopping=False), accuracy")
    t0 = time.time()
    rows = Parallel(n_jobs=-1)(
        delayed(_one_sample)(r, pool_idx, Xn, y, X_eval, y_eval, n) for r in range(R)
    )
    df = pd.DataFrame(rows)
    say(f"(took {time.time() - t0:.1f} s)")
    true_mean, true_sd = df["true"].mean(), df["true"].std(ddof=1)
    say(f"true accuracy of the n=1000 model on 20000 unseen rows: mean {true_mean:.4f}, "
        f"sd across samples {true_sd:.4f}")
    say()
    say("estimator         mean     bias(est-true)  sd across samples  sd of (est-true)  corr(est,true)")
    for col in ("cv2", "cv5", "cv10", "rep5x10"):
        diff = df[col] - df["true"]
        say(f"{col:<16s} {df[col].mean():.4f}   {diff.mean():+.4f}          {df[col].std(ddof=1):.4f}"
            f"             {diff.std(ddof=1):.4f}            {np.corrcoef(df[col], df['true'])[0, 1]:+.3f}")
    say()
    for k in (5, 10):
        naive = (df[f"cv{k}_sd"] / np.sqrt(k)).mean()
        say(f"{k}-fold: mean fold sd {df[f'cv{k}_sd'].mean():.4f}; naive SE = fold sd / sqrt({k}) "
            f"averages {naive:.4f}; actual sd of the {k}-fold mean across samples "
            f"{df[f'cv{k}'].std(ddof=1):.4f}; actual sd of (cv{k} - true) {(df[f'cv{k}'] - df['true']).std(ddof=1):.4f}")
        cover = (np.abs(df[f"cv{k}"] - df["true"]) <= 1.96 * df[f"cv{k}_sd"] / np.sqrt(k)).mean()
        say(f"   share of samples whose true accuracy lies within cv{k} +- 1.96 * naive SE: {cover:.3f}")
    say(f"split-only noise: sd of the 10 repeat means of 5-fold within one sample, averaged: "
        f"{df['rep_sd_of_repeat_means'].mean():.4f}")
    return {"df": df}


# ---------------------------------------------------------------------------
# 4. Target-encoding leakage
# ---------------------------------------------------------------------------

class MeanTargetEncoder(BaseEstimator, TransformerMixin):
    """Smoothed mean-of-target encoder for one categorical column.

    fit() stores, for every category, (sum of y + m * prior) / (count + m).
    transform() looks the values up; unseen categories get the prior.
    It does NOT cross-fit, so fit(X, y).transform(X) leaks each row's own label.
    """

    def __init__(self, m: float = 10.0):
        self.m = m

    def fit(self, X, y):
        s = pd.Series(np.asarray(X).ravel())
        y = np.asarray(y, dtype=float)
        self.prior_ = y.mean()
        stats = pd.DataFrame({"k": s.to_numpy(), "y": y}).groupby("k")["y"].agg(["sum", "count"])
        self.map_ = ((stats["sum"] + self.m * self.prior_) / (stats["count"] + self.m)).to_dict()
        return self

    def transform(self, X):
        s = pd.Series(np.asarray(X).ravel())
        return s.map(self.map_).fillna(self.prior_).to_numpy().reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array(["profile_te"])


def make_profile(X: pd.DataFrame) -> pd.Series:
    """High-cardinality key built from four ordinary columns."""
    return (
        X["age"].astype(str) + "|" + X["occupation"].astype(str) + "|"
        + X["education"].astype(str) + "|" + X["hours-per-week"].astype(str)
    )


def target_encoding_leak(X: pd.DataFrame, y: np.ndarray) -> dict:
    header("4. Target-encoding leakage (Adult, 'profile' = age|occupation|education|hours)")
    Xp = X.copy()
    Xp["profile"] = make_profile(X)
    cat_cols = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
    num_cols = [c for c in X.columns if c not in cat_cols]
    for c in cat_cols:
        Xp[c] = Xp[c].astype(str)
    X_tr, X_te, y_tr, y_te = train_test_split(Xp, y, test_size=0.3, stratify=y, random_state=SEED)
    counts = X_tr["profile"].value_counts()
    say(f"train {len(X_tr)} rows, test {len(X_te)} rows")
    say(f"distinct profiles in train: {len(counts)}; rows whose profile appears once: "
        f"{(counts == 1).sum()} ({(counts == 1).sum() / len(X_tr):.3f} of rows); "
        f"median rows per profile {counts.median():.0f}")
    say(f"test rows whose profile never appears in train: "
        f"{(~X_te['profile'].isin(counts.index)).mean():.3f}")

    def build(te_step):
        steps = [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
                 ("num", "passthrough", num_cols)]
        if te_step is not None:
            steps.insert(0, te_step)
        pre = ColumnTransformer(steps)
        return Pipeline([("pre", pre), ("model", HistGradientBoostingClassifier(random_state=0))])

    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    results = {}

    def report(name, pipe, Xtr, Xte):
        cvs = cross_val_score(pipe, Xtr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1)
        fitted = clone(pipe).fit(Xtr, y_tr)
        test = roc_auc_score(y_te, fitted.predict_proba(Xte)[:, 1])
        results[name] = (cvs.mean(), cvs.std(ddof=1), test)
        say(f"  {name:<48s} CV AUC {cvs.mean():.4f} (fold sd {cvs.std(ddof=1):.4f})  "
            f"test AUC {test:.4f}  CV - test {cvs.mean() - test:+.4f}")

    say("ROC AUC, 5-fold stratified CV on train vs held-out test:")
    report("A. no profile feature", build(None), X_tr, X_te)

    # B. Encoder fitted on the whole training set BEFORE cross-validation.
    enc = MeanTargetEncoder(m=10).fit(X_tr[["profile"]], y_tr)
    X_tr_b, X_te_b = X_tr.copy(), X_te.copy()
    X_tr_b["profile_te"] = enc.transform(X_tr[["profile"]]).ravel()
    X_te_b["profile_te"] = enc.transform(X_te[["profile"]]).ravel()
    report("B. mean encoding fitted before the CV loop", build(("te", "passthrough", ["profile_te"])),
           X_tr_b, X_te_b)

    # C. Same encoder, but inside the pipeline: refit on each training fold.
    report("C. same encoder inside the Pipeline", build(("te", MeanTargetEncoder(m=10), ["profile"])),
           X_tr, X_te)

    # D. sklearn TargetEncoder (cross-fitted fit_transform) inside the pipeline.
    report("D. sklearn TargetEncoder inside the Pipeline",
           build(("te", TargetEncoder(target_type="binary", random_state=0), ["profile"])), X_tr, X_te)
    return results


# ---------------------------------------------------------------------------
# 5. Group leakage (synthetic repeated measurements)
# ---------------------------------------------------------------------------

def make_subjects(n_subjects: int, rows_per_subject: int, rng: np.random.Generator):
    """Repeated measurements per subject.

    Per subject s: static attributes z_s ~ N(0, I_4), latent risk
    u_s = 0.5 * z_s1 + sqrt(0.75) * v_s with v_s ~ N(0, 1) (unobserved).
    Per visit: z_s observed with small noise (sd 0.05) and three visit-level
    measurements x ~ N(0, I_3).  Label: y = 1[u_s + 0.8 x_1 + 0.4 x_2 + e > 0],
    e ~ N(0, 0.5^2).  The subject's identity is recoverable from z, but only
    the 0.5 * z_s1 part of u_s carries over to a new subject.
    """
    z = rng.normal(size=(n_subjects, 4))
    u = 0.5 * z[:, 0] + np.sqrt(0.75) * rng.normal(size=n_subjects)
    sid = np.repeat(np.arange(n_subjects), rows_per_subject)
    n = len(sid)
    z_obs = z[sid] + rng.normal(scale=0.05, size=(n, 4))
    x = rng.normal(size=(n, 3))
    score = u[sid] + 0.8 * x[:, 0] + 0.4 * x[:, 1] + rng.normal(scale=0.5, size=n)
    return np.column_stack([z_obs, x]), (score > 0).astype(int), sid


def group_leak() -> dict:
    header("5. Group leakage: 300 subjects x 10 visits, truth = 1000 new subjects")
    rows = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        X, y, g = make_subjects(300, 10, rng)
        X_new, y_new, _ = make_subjects(1000, 10, rng)
        model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)
        kf = cross_val_score(model, X, y, cv=KFold(5, shuffle=True, random_state=0)).mean()
        gkf = cross_val_score(model, X, y, cv=GroupKFold(5), groups=g).mean()
        sgkf = cross_val_score(model, X, y, cv=StratifiedGroupKFold(5, shuffle=True, random_state=0),
                               groups=g).mean()
        truth = clone(model).fit(X, y).score(X_new, y_new)
        rows.append({"seed": seed, "KFold": kf, "GroupKFold": gkf, "StratifiedGroupKFold": sgkf,
                     "new subjects": truth})
        say(f"  seed {seed}: KFold {kf:.4f}  GroupKFold {gkf:.4f}  StratifiedGroupKFold {sgkf:.4f}"
            f"  new subjects {truth:.4f}   positive share {y.mean():.3f}")
    df = pd.DataFrame(rows)
    say("  mean   : " + "  ".join(f"{c} {df[c].mean():.4f} (sd {df[c].std(ddof=1):.4f})"
                                  for c in df.columns if c != "seed"))
    return {"df": df}


# ---------------------------------------------------------------------------
# 6. Time leakage (Bike Sharing, hourly)
# ---------------------------------------------------------------------------

def time_leak() -> dict:
    header("6. Time leakage: Bike Sharing hourly counts (OpenML 42712), MAE in rentals/hour")
    data = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True, data_home=str(CACHE),
                        parser="liac-arff")
    df = data.frame
    y = df["count"].to_numpy(dtype=float)
    X = df.drop(columns="count").copy()
    for c in X.columns:
        if isinstance(X[c].dtype, pd.CategoricalDtype):
            X[c] = X[c].cat.codes
    X = X.to_numpy(dtype=float)
    n = len(y)
    cut = int(n * 0.8)
    X_tr, y_tr, X_te, y_te = X[:cut], y[:cut], X[cut:], y[cut:]
    say(f"rows {n} in time order; train = first {cut} rows, future hold-out = last {n - cut} rows")
    say(f"hold-out starts at year={int(df['year'].iloc[cut])} month={int(df['month'].iloc[cut])}")
    model = HistGradientBoostingRegressor(random_state=0)
    res = {}
    schemes = {
        "KFold(5, shuffle=True)": KFold(5, shuffle=True, random_state=0),
        "KFold(5) = cv=5 for a regressor": KFold(5),
        "TimeSeriesSplit(5)": TimeSeriesSplit(5),
        "TimeSeriesSplit(5, gap=48)": TimeSeriesSplit(5, gap=48),
    }
    for name, cv in schemes.items():
        s = -cross_val_score(model, X_tr, y_tr, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
        res[name] = s.mean()
        say(f"  {name:<34s} CV MAE {s.mean():7.2f}   folds " + " ".join(f"{v:.1f}" for v in s))
    truth = mean_absolute_error(y_te, clone(model).fit(X_tr, y_tr).predict(X_te))
    res["future hold-out"] = truth
    say(f"  {'future hold-out (truth)':<34s} MAE {truth:7.2f}")
    say(f"  mean count in train {y_tr.mean():.1f}, in hold-out {y_te.mean():.1f}")
    return res


# ---------------------------------------------------------------------------
# 7. Nested CV
# ---------------------------------------------------------------------------

def _nested_once(seed: int, X, y) -> dict:
    with threadpool_limits(1):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
        grid = {"max_depth": [1, 2, 3, 4, 5, 6, 8, None], "min_samples_leaf": [1, 2, 5, 10, 20]}
        inner = StratifiedKFold(5, shuffle=True, random_state=seed)
        outer = StratifiedKFold(5, shuffle=True, random_state=seed + 100)
        gs = GridSearchCV(DecisionTreeClassifier(random_state=0), grid, cv=inner)
        gs.fit(X_tr, y_tr)
        nested = cross_val_score(gs, X_tr, y_tr, cv=outer).mean()
        return {"seed": seed, "non-nested": gs.best_score_, "nested": nested,
                "test": gs.score(X_te, y_te), "best": str(gs.best_params_)}


def nested_cv() -> dict:
    header("7. Nested vs non-nested CV: breast cancer, 398/171 splits, decision-tree grid (40 configs)")
    X, y = load_breast_cancer(return_X_y=True)
    rows = Parallel(n_jobs=-1)(delayed(_nested_once)(s, X, y) for s in range(20))
    df = pd.DataFrame(rows)
    for _, r in df.iterrows():
        say(f"  seed {int(r['seed']):>2d}: non-nested {r['non-nested']:.4f}  nested {r['nested']:.4f}"
            f"  test {r['test']:.4f}  best {r['best']}")
    for c in ("non-nested", "nested", "test"):
        say(f"  mean {c:<11s} {df[c].mean():.4f} (sd {df[c].std(ddof=1):.4f})")
    say(f"  mean (non-nested - test) {(df['non-nested'] - df['test']).mean():+.4f}; "
        f"mean (nested - test) {(df['nested'] - df['test']).mean():+.4f}")
    say(f"  mean (non-nested - nested) {(df['non-nested'] - df['nested']).mean():+.4f}; "
        f"non-nested > nested in {(df['non-nested'] > df['nested']).sum()}/20 splits")
    return {"df": df}


# ---------------------------------------------------------------------------
# 8. cross_val_score vs cross_validate vs cross_val_predict
# ---------------------------------------------------------------------------

def three_functions() -> None:
    header("8. cross_val_score vs cross_validate vs cross_val_predict (breast cancer train 398 rows)")
    X, y = load_breast_cancer(return_X_y=True)
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    model = DecisionTreeClassifier(max_depth=4, random_state=0)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    s = cross_val_score(model, X_tr, y_tr, cv=cv)
    say(f"  cross_val_score accuracy per fold {np.round(s, 4).tolist()} mean {s.mean():.4f}")
    cvr = cross_validate(model, X_tr, y_tr, cv=cv, scoring=["accuracy", "roc_auc"],
                         return_train_score=True)
    say(f"  cross_validate keys: {sorted(cvr.keys())}")
    say(f"    test_accuracy mean {cvr['test_accuracy'].mean():.4f}, train_accuracy mean "
        f"{cvr['train_accuracy'].mean():.4f}, test_roc_auc mean {cvr['test_roc_auc'].mean():.4f}")
    pred = cross_val_predict(model, X_tr, y_tr, cv=cv)
    proba = cross_val_predict(model, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    say(f"  cross_val_predict pooled accuracy {accuracy_score(y_tr, pred):.4f} "
        f"(mean of fold accuracies {s.mean():.4f})")
    say(f"  pooled AUC from cross_val_predict probabilities {roc_auc_score(y_tr, proba):.4f} "
        f"vs mean of per-fold AUC {cvr['test_roc_auc'].mean():.4f}")


# ---------------------------------------------------------------------------
# Figures 2 and 3
# ---------------------------------------------------------------------------

def figure_leakage(te: dict, grp: dict, tm: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), dpi=150)
    fig.patch.set_facecolor(PAPER)

    # Panel 1: target encoding (AUC), CV vs test for B, C, D
    ax = axes[0]
    names = ["B. encode\nbefore CV", "C. encoder in\nPipeline", "D. TargetEncoder\nin Pipeline"]
    keys = ["B. mean encoding fitted before the CV loop", "C. same encoder inside the Pipeline",
            "D. sklearn TargetEncoder inside the Pipeline"]
    xs = np.arange(3)
    cvv = [te[k][0] for k in keys]
    tst = [te[k][2] for k in keys]
    ax.bar(xs - 0.2, cvv, 0.4, color=TEAL, label="5-fold CV estimate")
    ax.bar(xs + 0.2, tst, 0.4, color=GREY, label="held-out test")
    for x, a, b in zip(xs, cvv, tst):
        ax.text(x - 0.2, a + 0.002, f"{a:.3f}", ha="center", fontsize=8, color=INK)
        ax.text(x + 0.2, b + 0.002, f"{b:.3f}", ha="center", fontsize=8, color=INK)
    lo = min(cvv + tst) - 0.03
    ax.set_ylim(lo, max(cvv + tst) + 0.02)
    ax.set_xticks(xs, names, fontsize=8.5)
    ax.set_ylabel("ROC AUC")
    ax.set_title("Target encoding on Adult", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK)

    # Panel 2: groups (accuracy)
    ax = axes[1]
    gdf = grp["df"]
    cols = ["KFold", "GroupKFold", "new subjects"]
    vals = [gdf[c].mean() for c in cols]
    colors = [TEAL, TEAL, GREY]
    ax.bar(range(3), vals, 0.55, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=8.5, color=INK)
    ax.set_ylim(min(vals) - 0.08, max(vals) + 0.04)
    ax.set_xticks(range(3), ["KFold\n(shuffled)", "GroupKFold", "truth:\n1000 new subjects"], fontsize=8.5)
    ax.set_ylabel("accuracy (mean of 5 seeds)")
    ax.set_title("Repeated measurements per subject", fontsize=11)

    # Panel 3: time (MAE, lower is better)
    ax = axes[2]
    keys = ["KFold(5, shuffle=True)", "KFold(5) = cv=5 for a regressor", "TimeSeriesSplit(5)",
            "TimeSeriesSplit(5, gap=48)", "future hold-out"]
    labels = ["KFold\nshuffled", "KFold\n(cv=5)", "TimeSeries\nSplit", "TimeSeries\nSplit gap=48",
              "truth:\nfuture 20%"]
    vals = [tm[k] for k in keys]
    colors = [TEAL] * 4 + [GREY]
    ax.bar(range(5), vals, 0.6, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(5), labels, fontsize=8)
    ax.set_ylabel("MAE (rentals per hour, lower = better)")
    ax.set_title("Hourly bike rentals in time order", fontsize=11)
    ax.set_ylim(0, max(vals) * 1.15)

    for ax in axes:
        style_axes(ax)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "leakage-gaps.png", facecolor=PAPER)
    plt.close(fig)


def figure_nested(nst: dict) -> None:
    df = nst["df"]
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    fig.patch.set_facecolor(PAPER)
    cols = ["non-nested", "nested", "test"]
    labels = ["non-nested\n(best_score_ of the search)", "nested CV\n(5 outer x 5 inner)",
              "held-out test\n(171 rows)"]
    colors = [AMBER, TEAL, INK_2]
    rng = np.random.default_rng(0)
    for i, c in enumerate(cols):
        jitter = rng.uniform(-0.12, 0.12, len(df))
        ax.scatter(np.full(len(df), i) + jitter, df[c], s=18, color=colors[i], alpha=0.75, lw=0)
        m = df[c].mean()
        ax.hlines(m, i - 0.28, i + 0.28, color=colors[i], lw=2.4)
        ax.text(i + 0.31, m, f"mean {m:.4f}", va="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(3), labels, fontsize=9)
    ax.set_xlim(-0.5, 2.9)
    ax.set_ylabel("accuracy")
    ax.set_title("Decision-tree grid search on breast cancer, 20 train/test splits", fontsize=11)
    style_axes(ax)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "nested-cv.png", facecolor=PAPER)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    t0 = time.time()
    say(f"scikit-learn {sklearn.__version__}, numpy {np.__version__}, pandas {pd.__version__}")
    layouts = hand_example()
    figure_fold_layout(layouts)
    what_cv5_means()
    X, y = load_adult()
    say()
    say(f"Adult (OpenML 'adult' v2): {X.shape[0]} rows, {X.shape[1]} features, "
        f"positive (>50K) share {y.mean():.4f}")
    cv_variance(X, y)
    te = target_encoding_leak(X, y)
    grp = group_leak()
    tm = time_leak()
    nst = nested_cv()
    three_functions()
    figure_leakage(te, grp, tm)
    figure_nested(nst)
    say()
    say(f"total runtime {time.time() - t0:.0f} s")
    (HERE / "results.txt").write_text(out.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
