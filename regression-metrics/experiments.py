"""Regression metrics and prediction intervals for the "regression-metrics" blog post.

Everything the post calls "measured" comes from this script:

1. Hand calculation on a 6-row toy set: MSE, RMSE, MAE, MedAE, R^2; the same
   set with one outlier; the constant that minimises squared vs absolute
   error (mean vs median); pinball loss at q = 0.1 / 0.9 and the coverage of
   the toy interval; MAPE / sMAPE / RMSLE near zero. Every hand number is
   asserted against scikit-learn.
2. California housing (fetch_california_housing, 20,640 rows, 8 features),
   80/20 split with seed 42: the 5.0 cap on the target, then baselines vs
   linear regression vs random forest vs histogram gradient boosting on all
   metrics, with a residual analysis.
3. GradientBoostingRegressor trained with squared / absolute / Huber loss on
   clean training targets and on training targets with 5% corrupted rows;
   every model is scored on the same clean test set.
4. Quantile gradient boosting at q = 0.05 / 0.5 / 0.95 (both
   HistGradientBoostingRegressor and GradientBoostingRegressor): coverage,
   width, pinball loss and quantile crossing on the test set.
5. Extrapolation: a 1-D synthetic line and a California split where the
   models are trained only on houses below 3.0 (x 100,000 USD).

Run:  .venv/Scripts/python.exe regression-metrics/experiments.py
Output: regression-metrics/results.txt and three PNG figures in regression-metrics/figures/.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sklearn  # noqa: E402
from sklearn.datasets import fetch_california_housing  # noqa: E402
from sklearn.dummy import DummyRegressor  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_pinball_loss,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
    root_mean_squared_log_error,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.tree import DecisionTreeRegressor  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG_DIR = HERE / "figures"
DATA_HOME = ROOT / "data_cache"
SEED = 42

# "Paper" palette used by the blog's figures.
PAPER = "#F8F6F1"
INK = "#1F2328"
INK_2 = "#5B6270"
GRID = "#E8E2D6"
TEAL = "#0F766E"
TEAL_LIGHT = "#A7D3CC"
AMBER = "#B45309"
GREY = "#9AA0AA"

out = io.StringIO()


def say(*args) -> None:
    """Print to stdout and to the results buffer."""
    text = " ".join(str(a) for a in args)
    print(text)
    out.write(text + "\n")


def close(a: float, b: float, tol: float = 1e-9) -> None:
    assert abs(a - b) <= tol, (a, b)


def style(ax) -> None:
    ax.set_facecolor(PAPER)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK, labelsize=12)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def metrics(y, p) -> dict:
    return {
        "MSE": mean_squared_error(y, p),
        "RMSE": root_mean_squared_error(y, p),
        "MAE": mean_absolute_error(y, p),
        "MedAE": median_absolute_error(y, p),
        "R2": r2_score(y, p),
        "MAPE": mean_absolute_percentage_error(y, p),
    }


def metric_row(name: str, m: dict) -> str:
    return (f"{name:<28} MSE {m['MSE']:.4f}  RMSE {m['RMSE']:.4f}  MAE {m['MAE']:.4f}  "
            f"MedAE {m['MedAE']:.4f}  R2 {m['R2']:.4f}  MAPE {m['MAPE']:.4f}")


def pinball(y, p, q) -> float:
    u = np.asarray(y, float) - np.asarray(p, float)
    return float(np.mean(np.maximum(q * u, (q - 1) * u)))


# ---------------------------------------------------------------- 1. toy set
def part_toy():
    say("=" * 78)
    say("1. HAND CALCULATION (6 rows)")
    say("=" * 78)
    y = np.array([4, 6, 7, 9, 10, 12], float)
    p = np.array([3, 7, 5, 11, 10, 12], float)
    e = y - p
    say("y      ", y.tolist())
    say("y_hat  ", p.tolist())
    say("error  ", e.tolist())
    sse = float(np.sum(e**2))
    sst = float(np.sum((y - y.mean()) ** 2))
    hand = {
        "MSE": sse / 6, "RMSE": (sse / 6) ** 0.5, "MAE": float(np.mean(np.abs(e))),
        "MedAE": float(np.median(np.abs(e))), "R2": 1 - sse / sst,
    }
    say(f"mean(y) {y.mean()}  SSE {sse}  SST {sst}")
    m = metrics(y, p)
    for k, v in hand.items():
        close(v, m[k])
        say(f"  {k:<6} hand {v:.4f}   sklearn {m[k]:.4f}")
    before = m

    say("\n-- one outlier: row 6 true value 12 -> 30, prediction unchanged (12)")
    y2 = y.copy()
    y2[5] = 30
    e2 = y2 - p
    sse2 = float(np.sum(e2**2))
    sst2 = float(np.sum((y2 - y2.mean()) ** 2))
    hand2 = {
        "MSE": sse2 / 6, "RMSE": (sse2 / 6) ** 0.5, "MAE": float(np.mean(np.abs(e2))),
        "MedAE": float(np.median(np.abs(e2))), "R2": 1 - sse2 / sst2,
    }
    say(f"mean(y) {y2.mean()}  SSE {sse2}  SST {sst2}")
    m2 = metrics(y2, p)
    for k, v in hand2.items():
        close(v, m2[k])
        say(f"  {k:<6} hand {v:.4f}   sklearn {m2[k]:.4f}   ratio after/before {v / hand[k]:.2f}")
    after = m2

    say("\n-- best constant prediction c for y =", y2.tolist())
    for c in [7, 8, 9, 11]:
        sae = float(np.sum(np.abs(y2 - c)))
        sse_c = float(np.sum((y2 - c) ** 2))
        say(f"  c = {c:>2}: sum|y-c| = {sae:.0f}   sum(y-c)^2 = {sse_c:.0f}")
    grid = np.linspace(0, 32, 3201)
    sae_g = np.array([np.sum(np.abs(y2 - c)) for c in grid])
    sse_g = np.array([np.sum((y2 - c) ** 2) for c in grid])
    flat = grid[np.isclose(sae_g, sae_g.min())]
    say(f"  grid search: SSE minimised at c = {grid[sse_g.argmin()]:.2f} (mean {y2.mean():.2f}); "
        f"SAE minimised on [{flat.min():.2f}, {flat.max():.2f}] (median {np.median(y2):.2f})")
    d_mean = DummyRegressor(strategy="mean").fit(np.zeros((6, 1)), y2).predict(np.zeros((1, 1)))[0]
    d_med = DummyRegressor(strategy="median").fit(np.zeros((6, 1)), y2).predict(np.zeros((1, 1)))[0]
    say(f"  DummyRegressor mean -> {d_mean:.2f}, median -> {d_med:.2f}")
    close(d_mean, 11.0)
    close(d_med, 8.0)
    say(f"  R2 of the constant 11 (the mean) on these rows: {r2_score(y2, np.full(6, 11.0)):.4f}")
    say(f"  R2 of the constant 8 (the median) on these rows: {r2_score(y2, np.full(6, 8.0)):.4f}  (hand 1 - 510/456)")
    close(r2_score(y2, np.full(6, 8.0)), 1 - 510 / 456)

    say("\n-- pinball loss on the clean rows y =", y.tolist())
    for q, c in [(0.1, 5.0), (0.9, 11.0), (0.9, 5.0), (0.1, 11.0)]:
        u = y - c
        terms = np.maximum(q * u, (q - 1) * u)
        val = float(terms.mean())
        close(val, mean_pinball_loss(y, np.full(6, c), alpha=q))
        say(f"  q={q} constant {c}: u = {u.tolist()}  terms = {np.round(terms, 2).tolist()}  "
            f"sum {terms.sum():.2f}  mean {val:.4f}  (sklearn {mean_pinball_loss(y, np.full(6, c), alpha=q):.4f})")
    inside = (y >= 5) & (y <= 11)
    say(f"  interval [5, 11] covers {inside.sum()}/6 = {inside.mean():.4f}, width 6")
    close(mean_pinball_loss(y, p, alpha=0.5), 0.5 * mean_absolute_error(y, p))
    say(f"  pinball(q=0.5) of the toy predictions = {mean_pinball_loss(y, p, alpha=0.5):.4f} = MAE/2 "
        f"({mean_absolute_error(y, p):.4f}/2)")
    for q in (0.1, 0.9):
        best = grid[np.argmin([pinball(y, np.full(6, c), q) for c in grid])]
        say(f"  constant minimising pinball q={q}: c = {best:.2f}")

    say("\n-- percentage errors near zero")
    yz = np.array([0.1, 10.0])
    pz = np.array([0.6, 10.5])
    ape = np.abs(yz - pz) / np.abs(yz)
    say(f"  y {yz.tolist()} y_hat {pz.tolist()}  abs error {np.abs(yz - pz).tolist()}  APE {ape.tolist()}")
    say(f"  MAE {mean_absolute_error(yz, pz):.4f}  MAPE {mean_absolute_percentage_error(yz, pz):.4f}")
    close(mean_absolute_percentage_error(yz, pz), ape.mean())
    smape = np.mean(2 * np.abs(yz - pz) / (np.abs(yz) + np.abs(pz)))
    say(f"  sMAPE (2|e|/(|y|+|y_hat|), mean) {smape:.4f}")
    rmsle = root_mean_squared_log_error(yz, pz)
    close(rmsle, float(np.sqrt(np.mean((np.log1p(yz) - np.log1p(pz)) ** 2))))
    say(f"  RMSLE {rmsle:.4f}")
    say(f"  MAPE with y = 0 exactly: {mean_absolute_percentage_error([0.0, 10.0], [0.5, 10.5]):.4g}")
    return before, after


# --------------------------------------------------------- 2. california data
def load():
    data = fetch_california_housing(data_home=str(DATA_HOME))
    X, y = data.data, data.target
    return X, y, list(data.feature_names)


def part_california(X, y):
    say("\n" + "=" * 78)
    say("2. CALIFORNIA HOUSING: baselines vs models")
    say("=" * 78)
    say(f"rows {X.shape[0]}, features {X.shape[1]}; target = median house value in 100,000 USD")
    say(f"target min {y.min():.5f} max {y.max():.5f} mean {y.mean():.4f} median {np.median(y):.4f}")
    cap = y >= 5.0
    say(f"rows with target >= 5.0 (the cap): {cap.sum()} ({cap.mean():.4f}); distinct max value {y.max()}")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    say(f"train {len(ytr)} / test {len(yte)}; capped in test {np.sum(yte >= 5.0)}")

    models = {
        "Dummy (mean)": DummyRegressor(strategy="mean"),
        "Dummy (median)": DummyRegressor(strategy="median"),
        "LinearRegression": LinearRegression(),
        "RandomForest (300)": RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.1, early_stopping=False, random_state=SEED),
    }
    preds = {}
    for name, mdl in models.items():
        t = time.perf_counter()
        mdl.fit(Xtr, ytr)
        preds[name] = mdl.predict(Xte)
        say(metric_row(name, metrics(yte, preds[name])) + f"   ({time.perf_counter() - t:.1f}s)")
        if name == "RandomForest (300)":
            ptr = mdl.predict(Xtr)
            say(f"{'  (RF on its own train rows)':<28} RMSE {root_mean_squared_error(ytr, ptr):.4f}  R2 {r2_score(ytr, ptr):.4f}")
    say(f"R2 of predicting the TRAIN mean on test: {r2_score(yte, np.full_like(yte, ytr.mean())):.6f}")
    say(f"R2 of predicting the TEST mean on test:  {r2_score(yte, np.full_like(yte, yte.mean())):.6f}")
    shifted = np.full_like(yte, ytr.mean() + 0.5)
    say(f"R2 of predicting train mean + 0.5 on test: {r2_score(yte, shifted):.4f}  (negative R2 on test)")

    p = preds["HistGradientBoosting"]
    res = yte - p
    say("\n-- residuals of HistGradientBoosting on test")
    say(f"mean residual {res.mean():.4f}; capped rows (y>=5): n={np.sum(yte >= 5)}, mean residual "
        f"{res[yte >= 5].mean():.4f}, max prediction on them {p[yte >= 5].max():.4f}, "
        f"share predicted below 5.0 {np.mean(p[yte >= 5] < 5.0):.4f}")
    say(f"predictions above 5.0: {np.sum(p > 5.0)} of {len(p)}; max prediction {p.max():.4f}")
    bins = [-np.inf, 1, 2, 3, 4, np.inf]
    labels = ["<1", "1-2", "2-3", "3-4", ">=4"]
    for lo, hi, lab in zip(bins[:-1], bins[1:], labels):
        sel = (p >= lo) & (p < hi)
        say(f"  predicted {lab:<4} n={sel.sum():>4}  mean residual {res[sel].mean():+.4f}  "
            f"residual sd {res[sel].std():.4f}")
    for name in ("LinearRegression", "RandomForest (300)"):
        pp = preds[name]
        say(f"{name}: predictions above 5.0: {np.sum(pp > 5.0)}, max {pp.max():.4f}, min {pp.min():.4f}")
    unc = yte < 5.0
    say(f"HGB metrics on uncapped test rows only: RMSE {root_mean_squared_error(yte[unc], p[unc]):.4f} "
        f"MAE {mean_absolute_error(yte[unc], p[unc]):.4f}; on capped rows: "
        f"RMSE {root_mean_squared_error(yte[~unc], p[~unc]):.4f} MAE {mean_absolute_error(yte[~unc], p[~unc]):.4f}")
    return Xtr, Xte, ytr, yte, p


# ------------------------------------------------------------- 3. loss choice
def part_losses(Xtr, Xte, ytr, yte):
    say("\n" + "=" * 78)
    say("3. TRAINING LOSS vs EVALUATION METRIC (GradientBoostingRegressor)")
    say("=" * 78)
    common = dict(n_estimators=300, learning_rate=0.1, max_depth=4, random_state=SEED)
    losses = [("squared_error", {}), ("absolute_error", {}), ("huber", {"alpha": 0.9})]
    rng = np.random.default_rng(SEED)
    ytr_bad = ytr.copy()
    idx = rng.choice(len(ytr), size=int(0.05 * len(ytr)), replace=False)
    ytr_bad[idx] = ytr_bad[idx] + 10.0
    say(f"corrupted copy: {len(idx)} training targets (5%) shifted by +10.0; test set untouched")
    for label, target in (("clean training targets", ytr), ("5% corrupted training targets", ytr_bad)):
        say(f"\n-- {label}")
        for loss, extra in losses:
            t = time.perf_counter()
            mdl = GradientBoostingRegressor(loss=loss, **extra, **common).fit(Xtr, target)
            pr = mdl.predict(Xte)
            say(metric_row(loss, metrics(yte, pr)) + f"   ({time.perf_counter() - t:.1f}s)")
            say(f"{'':<28} init (F0) = {np.ravel(mdl.init_.predict(Xte[:1]))[0]:.4f}  "
                f"(train mean {target.mean():.4f}, train median {np.median(target):.4f})")


# ---------------------------------------------------------------- 4. quantiles
def part_quantiles(Xtr, Xte, ytr, yte):
    say("\n" + "=" * 78)
    say("4. QUANTILE GRADIENT BOOSTING: 90% interval from q = 0.05 and 0.95")
    say("=" * 78)
    qs = (0.05, 0.5, 0.95)
    families = {
        "HistGradientBoostingRegressor": lambda q: HistGradientBoostingRegressor(
            loss="quantile", quantile=q, max_iter=300, learning_rate=0.1,
            early_stopping=False, random_state=SEED),
        "GradientBoostingRegressor": lambda q: GradientBoostingRegressor(
            loss="quantile", alpha=q, n_estimators=300, learning_rate=0.1, max_depth=4,
            random_state=SEED),
    }
    keep = None
    for fam, make in families.items():
        t = time.perf_counter()
        pt, pr = {}, {}
        for q in qs:
            mdl = make(q).fit(Xtr, ytr)
            pr[q] = mdl.predict(Xte)
            pt[q] = mdl.predict(Xtr)
        say(f"\n-- {fam} ({time.perf_counter() - t:.1f}s)")
        for q in qs:
            say(f"  q={q:<4} pinball(test) {mean_pinball_loss(yte, pr[q], alpha=q):.4f}  "
                f"share of test y below prediction {np.mean(yte < pr[q]):.4f}   "
                f"(train {np.mean(ytr < pt[q]):.4f})")
        lo, med, hi = pr[0.05], pr[0.5], pr[0.95]
        cov = np.mean((yte >= lo) & (yte <= hi))
        cov_tr = np.mean((ytr >= pt[0.05]) & (ytr <= pt[0.95]))
        width = hi - lo
        say(f"  90% interval: test coverage {cov:.4f}  (train {cov_tr:.4f}); mean width {width.mean():.4f}, "
            f"median width {np.median(width):.4f}, min {width.min():.4f}, max {width.max():.4f}")
        cross = np.sum(lo > hi)
        med_out = np.sum((med < lo) | (med > hi))
        say(f"  crossing: rows with q0.05 > q0.95: {cross}; rows where median lies outside [q0.05, q0.95]: {med_out}")
        unc = yte < 5.0
        say(f"  coverage on uncapped test rows {np.mean((yte[unc] >= lo[unc]) & (yte[unc] <= hi[unc])):.4f}; "
            f"on capped rows {np.mean((yte[~unc] >= lo[~unc]) & (yte[~unc] <= hi[~unc])):.4f} (n={np.sum(~unc)})")
        # Conditional coverage by predicted median band.
        for a, b in [(0, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 10)]:
            sel = (med >= a) & (med < b)
            say(f"    median prediction in [{a}, {b}): n={sel.sum():>4}  coverage "
                f"{np.mean((yte[sel] >= lo[sel]) & (yte[sel] <= hi[sel])):.4f}  mean width {width[sel].mean():.4f}")
        if fam == "HistGradientBoostingRegressor":
            keep = (lo, med, hi, cov, width.mean())
    # A naive symmetric interval from a squared-error model: +-1.645 * training RMSE.
    sq = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, early_stopping=False,
                                       random_state=SEED).fit(Xtr, ytr)
    rmse_tr = root_mean_squared_error(ytr, sq.predict(Xtr))
    pte = sq.predict(Xte)
    cov_n = np.mean(np.abs(yte - pte) <= 1.645 * rmse_tr)
    say(f"\n-- naive interval: squared-error HGB +- 1.645 x TRAIN RMSE ({rmse_tr:.4f}): "
        f"width {2 * 1.645 * rmse_tr:.4f}, test coverage {cov_n:.4f}")
    return keep


# ----------------------------------------------------------- 5. extrapolation
def part_extrapolation(X, y, names):
    say("\n" + "=" * 78)
    say("5. EXTRAPOLATION")
    say("=" * 78)
    rng = np.random.default_rng(SEED)
    x_tr = rng.uniform(0, 10, 200)
    y_tr = 2.0 * x_tr + rng.normal(0, 1.0, 200)
    x_in = np.linspace(0, 10, 101)
    x_out = np.linspace(10, 15, 51)[1:]
    say(f"synthetic: y = 2x + N(0, 1), 200 training points with x in [0, 10]; "
        f"max training y {y_tr.max():.4f}")
    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree (depth 6)": DecisionTreeRegressor(max_depth=6, random_state=SEED),
        "RandomForest (300)": RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=300, early_stopping=False,
                                                              random_state=SEED),
    }
    curves = {}
    for name, mdl in models.items():
        mdl.fit(x_tr.reshape(-1, 1), y_tr)
        p_in = mdl.predict(x_in.reshape(-1, 1))
        p_out = mdl.predict(x_out.reshape(-1, 1))
        curves[name] = (np.r_[x_in, x_out], np.r_[p_in, p_out])
        say(f"  {name:<24} RMSE vs true line inside [0,10] {root_mean_squared_error(2 * x_in, p_in):.4f}  "
            f"outside (10,15] {root_mean_squared_error(2 * x_out, p_out):.4f}  "
            f"prediction at x=15: {p_out[-1]:.4f} (true 30)")

    say("\n-- California: train only on houses with target < 3.0, test on the rest")
    lo_mask = y < 3.0
    Xa, ya = X[lo_mask], y[lo_mask]
    Xb, yb = X[~lo_mask], y[~lo_mask]
    say(f"  train rows {len(ya)} (max target {ya.max():.5f}); out-of-range rows {len(yb)} "
        f"(target 3.0 to {yb.max():.5f}, mean {yb.mean():.4f})")
    cal = {
        "LinearRegression": LinearRegression(),
        "RandomForest (300)": RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=300, early_stopping=False,
                                                              random_state=SEED),
    }
    for name, mdl in cal.items():
        mdl.fit(Xa, ya)
        pb = mdl.predict(Xb)
        say(f"  {name:<24} max prediction {pb.max():.4f}  share of predictions >= 3.0 {np.mean(pb >= 3.0):.4f}  "
            f"RMSE {root_mean_squared_error(yb, pb):.4f}  MAE {mean_absolute_error(yb, pb):.4f}  "
            f"mean prediction {pb.mean():.4f}")
    return x_tr, y_tr, curves


# ------------------------------------------------------------------- figures
def fig_outlier(before, after):
    keys = ["MSE", "RMSE", "MAE", "MedAE", "R2"]
    titles = ["MSE", "RMSE", "MAE", "MedAE", "R²"]
    fig, axes = plt.subplots(1, 5, figsize=(14, 4.4), facecolor=PAPER)
    for ax, k, t in zip(axes, keys, titles):
        style(ax)
        vals = [before[k], after[k]]
        bars = ax.bar([0, 1], vals, color=[GREY, TEAL], width=0.6)
        ax.set_xticks([0, 1], ["6 rows", "+ outlier"], fontsize=12, color=INK)
        ax.set_title(t, fontsize=16, color=INK, pad=10)
        top = max(vals)
        ax.set_ylim(0, top * 1.25 if top > 0 else 1)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + top * 0.03, f"{v:.2f}", ha="center",
                    va="bottom", fontsize=13, color=INK)
        ratio = after[k] / before[k]
        ax.text(0.5, -0.2, f"x {ratio:.1f}", transform=ax.transAxes, ha="center",
                fontsize=14, color=TEAL if k != "R2" else AMBER, fontweight="bold")
    fig.suptitle("One wrong value (true 12 -> 30, prediction still 12): how far each metric moves",
                 fontsize=15, color=INK, y=0.99)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(FIG_DIR / "outlier-effect.png", dpi=150, facecolor=PAPER)
    plt.close(fig)


def fig_residual_interval(yte, p, q):
    lo, med, hi, cov, width = q
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.6), facecolor=PAPER)
    style(a1)
    res = yte - p
    rng = np.random.default_rng(SEED)
    sel = rng.choice(len(yte), 1500, replace=False)
    capped = yte[sel] >= 5.0
    a1.scatter(p[sel][~capped], res[sel][~capped], s=9, color=GREY, alpha=0.6, lw=0, label="test rows")
    a1.scatter(p[sel][capped], res[sel][capped], s=12, color=AMBER, alpha=0.9, lw=0,
               label="true value capped at 5.0")
    xs = np.linspace(0, 5.2, 50)
    a1.plot(xs, 5.0 - xs, color=AMBER, lw=1, ls="--")
    a1.axhline(0, color=TEAL, lw=1.5)
    a1.set_xlabel("predicted value (100,000 USD)", fontsize=13, color=INK)
    a1.set_ylabel("residual = true - predicted", fontsize=13, color=INK)
    a1.set_title("Residuals of gradient boosting (1,500 test rows)", fontsize=14, color=INK)
    a1.legend(frameon=False, fontsize=11, loc="lower left")

    style(a2)
    order = np.argsort(med)
    pick = order[:: max(1, len(order) // 400)]
    xi = np.arange(len(pick))
    a2.fill_between(xi, lo[pick], hi[pick], color=TEAL_LIGHT, alpha=0.8, lw=0,
                    label="predicted 5%-95% interval")
    a2.plot(xi, med[pick], color=TEAL, lw=1.4, label="predicted median (q = 0.5)")
    inside = (yte[pick] >= lo[pick]) & (yte[pick] <= hi[pick])
    a2.scatter(xi[inside], yte[pick][inside], s=8, color=INK_2, lw=0, label="true value, inside")
    a2.scatter(xi[~inside], yte[pick][~inside], s=14, color=AMBER, lw=0, label="true value, outside")
    a2.set_xlabel("test rows sorted by predicted median (every 10th row)", fontsize=13, color=INK)
    a2.set_ylabel("house value (100,000 USD)", fontsize=13, color=INK)
    a2.set_title(f"Quantile boosting: coverage {cov:.1%} (target 90%), mean width {width:.2f}",
                 fontsize=14, color=INK)
    a2.legend(frameon=False, fontsize=11, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "residuals-and-interval.png", dpi=150, facecolor=PAPER)
    plt.close(fig)


def fig_extrapolation(x_tr, y_tr, curves):
    fig, ax = plt.subplots(figsize=(11, 5.6), facecolor=PAPER)
    style(ax)
    ax.axvspan(10, 15, color=GRID, alpha=0.7, lw=0)
    ax.text(12.5, 2, "outside the training range", ha="center", fontsize=13, color=INK_2)
    ax.scatter(x_tr, y_tr, s=10, color=GREY, lw=0, label="training points")
    xs = np.linspace(0, 15, 2)
    ax.plot(xs, 2 * xs, color=INK, lw=1, ls=":", label="true line y = 2x")
    colors = {"LinearRegression": INK_2, "DecisionTree (depth 6)": AMBER,
              "RandomForest (300)": TEAL, "HistGradientBoosting": "#64748B"}
    styles = {"LinearRegression": "--", "DecisionTree (depth 6)": "-",
              "RandomForest (300)": "-", "HistGradientBoosting": "-."}
    for name, (xx, pp) in curves.items():
        ax.plot(xx, pp, color=colors[name], lw=2, ls=styles[name], label=name)
    ax.set_xlabel("x", fontsize=13, color=INK)
    ax.set_ylabel("prediction", fontsize=13, color=INK)
    ax.set_title("Tree models stay flat beyond the largest target they were trained on",
                 fontsize=14, color=INK)
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "extrapolation.png", dpi=150, facecolor=PAPER)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(exist_ok=True)
    say(f"scikit-learn {sklearn.__version__}, numpy {np.__version__}, seed {SEED}")
    before, after = part_toy()
    X, y, names = load()
    Xtr, Xte, ytr, yte, p_hgb = part_california(X, y)
    part_losses(Xtr, Xte, ytr, yte)
    q = part_quantiles(Xtr, Xte, ytr, yte)
    x_tr, y_tr, curves = part_extrapolation(X, y, names)
    fig_outlier(before, after)
    fig_residual_interval(yte, p_hgb, q)
    fig_extrapolation(x_tr, y_tr, curves)
    (HERE / "results.txt").write_text(out.getvalue(), encoding="utf-8")
    print("\nwrote results.txt and figures/")


if __name__ == "__main__":
    main()
