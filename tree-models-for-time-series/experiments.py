"""Experiments for the "tree models for time-series forecasting" blog post.

Every number the post calls "measured" comes from this script:

1. Hand example: a 10-point series turned into a lag-1 / lag-2 / rolling-mean-3
   table with ``shift`` before ``rolling``, asserted against hand values, plus
   the leaky rolling mean that includes the current value and its "perfect"
   error.
2. Bike Sharing (OpenML 42712) rebuilt on a complete hourly grid: direct and
   recursive forecasts for horizons 1-48 h against naive, seasonal naive and
   linear-regression baselines (MAE, MASE, skill vs seasonal naive).
3. The same leak as in (1) on Bike Sharing, and why no CV gap can catch it.
4. Validation of the 24-h direct model on the training period only:
   shuffled KFold, TimeSeriesSplit with different gaps, expanding vs sliding
   window, random-forest OOB, compared with the future hold-out.
5. What trees add: lags vs calendar vs (observed) weather on Bike Sharing,
   and many related synthetic series in one table vs one model per series.
6. The extrapolation limit on a seeded trending series (trend + weekly season
   + noise) and the standard fixes: differencing, detrending, a level-ratio
   target, a log target and LightGBM ``linear_tree``.

Run:  .venv/Scripts/python.exe tree-models-for-time-series/experiments.py
Output: tree-models-for-time-series/results.txt and PNG figures in
tree-models-for-time-series/figures/. Bike Sharing is cached under data_cache/
at the repository root. Every model is capped at 4 threads.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import lightgbm as lgb  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.datasets import fetch_openml  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import mean_absolute_error  # noqa: E402
from sklearn.model_selection import KFold, TimeSeriesSplit  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE = ROOT / "data_cache"
FIG_DIR = HERE / "figures"
THREADS = 4

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


def hgb(**kw) -> HistGradientBoostingRegressor:
    """The boosting model used throughout: fixed number of rounds, no random
    early-stopping split (which would shuffle time inside the training set)."""
    params = dict(max_iter=300, learning_rate=0.05, early_stopping=False, random_state=0)
    params.update(kw)
    return HistGradientBoostingRegressor(**params)


def mae(a, b) -> float:
    return float(mean_absolute_error(a, b))


# ---------------------------------------------------------------------------
# 1. Hand example
# ---------------------------------------------------------------------------

Y10 = [20.0, 23.0, 21.0, 25.0, 27.0, 24.0, 28.0, 31.0, 29.0, 33.0]


def hand_example() -> pd.DataFrame:
    header("1. Hand example: 10-point series -> supervised table")
    y = pd.Series(Y10, index=pd.RangeIndex(1, 11, name="t"), name="y")
    tab = pd.DataFrame({
        "y": y,
        "lag_1": y.shift(1),
        "lag_2": y.shift(2),
        "mean_3": y.shift(1).rolling(3).mean(),      # correct: shift BEFORE rolling
        "leaky_mean_3": y.rolling(3).mean(),          # includes y_t itself
    })
    say(tab.round(4).to_string())

    # Hand-computed values (t = 4 and t = 10)
    assert tab.loc[4, "lag_1"] == 21 and tab.loc[4, "lag_2"] == 23
    assert abs(tab.loc[4, "mean_3"] - (20 + 23 + 21) / 3) < 1e-12
    assert abs(tab.loc[4, "leaky_mean_3"] - (23 + 21 + 25) / 3) < 1e-12
    assert abs(tab.loc[10, "mean_3"] - (28 + 31 + 29) / 3) < 1e-12
    assert tab["mean_3"].isna().sum() == 3 and tab["leaky_mean_3"].isna().sum() == 2

    # Train on t = 4..8, test on t = 9..10 (the rows where every feature exists)
    tr, te = tab.loc[4:8], tab.loc[9:10]
    res = {}
    for name, cols in {"correct (lag_1, lag_2, mean_3)": ["lag_1", "lag_2", "mean_3"],
                       "leaky (lag_1, lag_2, leaky_mean_3)": ["lag_1", "lag_2", "leaky_mean_3"]}.items():
        lr = LinearRegression().fit(tr[cols], tr["y"])
        pred = lr.predict(te[cols])
        res[name] = mae(te["y"], pred)
        say(f"  {name:<38s} test MAE {res[name]:.6f}  predictions {np.round(pred, 4).tolist()}"
            f"  coef {np.round(lr.coef_, 4).tolist()} intercept {lr.intercept_:.4f}")
    say(f"  naive (y_t-1) on t = 9..10: MAE {mae(te['y'], te['lag_1']):.4f}")
    # The leaky feature makes y_t = 3 * leaky_mean_3 - lag_1 - lag_2 exact.
    assert res["leaky (lag_1, lag_2, leaky_mean_3)"] < 1e-9
    say("  asserted: leaky model test MAE < 1e-9 because y_t = 3*leaky_mean_3 - lag_1 - lag_2")
    return tab


# ---------------------------------------------------------------------------
# 2. Bike Sharing on a complete hourly grid
# ---------------------------------------------------------------------------

SEASONAL_LAGS = (24, 48, 168)
CAL = ["hour", "dow", "month", "workingday", "holiday"]
WEATHER = ["weather", "temp", "feel_temp", "humidity", "windspeed"]


def load_bike() -> dict:
    header("2. Bike Sharing (OpenML 42712) rebuilt as an hourly series")
    data = fetch_openml(data_id=42712, as_frame=True, data_home=str(CACHE), parser="liac-arff")
    df = data.frame.copy()
    wd = pd.to_numeric(df["weekday"]).astype(int).to_numpy()
    hr = pd.to_numeric(df["hour"]).astype(int).to_numpy()
    # The file has no date column. Rows are in time order and every calendar day
    # has at least one row, so a new day starts whenever "weekday" changes.
    day = np.cumsum(np.r_[0, (wd[1:] != wd[:-1]).astype(int)])
    ts = pd.Timestamp("2011-01-01") + pd.to_timedelta(day, unit="D") + pd.to_timedelta(hr, unit="h")
    ts = pd.DatetimeIndex(ts)
    assert ts.is_monotonic_increasing and ts.is_unique
    assert day.max() == 730
    # UCI weekday: 0 = Sunday. pandas dayofweek: 0 = Monday.
    assert (((ts.dayofweek + 1) % 7) == wd).all()
    n_rows = len(df)
    grid = pd.date_range("2011-01-01", periods=731 * 24, freq="h")
    y = pd.Series(df["count"].to_numpy(dtype=float), index=ts).reindex(grid)
    say(f"rows in file {n_rows}; hourly grid {len(grid)}; hours missing from the file {int(y.isna().sum())}")

    frame = pd.DataFrame(index=grid)
    frame["hour"] = grid.hour
    frame["dow"] = grid.dayofweek
    frame["month"] = grid.month
    daily = pd.DataFrame({"date": ts.normalize(),
                          "workingday": pd.to_numeric(df["workingday"].astype(str).replace(
                              {"True": 1, "False": 0})).to_numpy(),
                          "holiday": pd.to_numeric(df["holiday"].astype(str).replace(
                              {"True": 1, "False": 0})).to_numpy()}).groupby("date").first()
    frame["workingday"] = daily["workingday"].reindex(grid.normalize()).to_numpy()
    frame["holiday"] = daily["holiday"].reindex(grid.normalize()).to_numpy()
    wcode = df["weather"].astype("category").cat.codes.to_numpy()
    wx = pd.DataFrame({"weather": wcode, "temp": df["temp"].to_numpy(float),
                       "feel_temp": df["feel_temp"].to_numpy(float),
                       "humidity": df["humidity"].to_numpy(float),
                       "windspeed": df["windspeed"].to_numpy(float)}, index=ts).reindex(grid).ffill()
    frame = frame.join(wx)

    cut_row = int(n_rows * 0.8)  # same split as the cross-validation post
    t0 = ts[cut_row]
    say(f"future hold-out starts at {t0} (file row {cut_row}); "
        f"hold-out hours observed {int(y[grid >= t0].notna().sum())}")
    is_train = grid < t0
    scale24 = float(np.nanmean(np.abs(y[is_train] - y.shift(24)[is_train])))
    scale1 = float(np.nanmean(np.abs(y[is_train] - y.shift(1)[is_train])))
    say(f"MASE scale (training MAE of seasonal naive, m = 24): {scale24:.3f}; "
        f"(one-step naive, m = 1): {scale1:.3f}")
    say(f"mean count: training {np.nanmean(y[is_train]):.1f}, hold-out {np.nanmean(y[~is_train]):.1f}")
    return {"y": y, "frame": frame, "grid": grid, "t0": t0, "is_train": is_train,
            "scale24": scale24, "df": df, "ts": ts, "cut_row": cut_row}


def lag_features(y: pd.Series, h: int) -> pd.DataFrame:
    """Features for predicting y_t when the newest known value is y_{t-h}."""
    lags = sorted({h, h + 1, h + 2} | {s for s in SEASONAL_LAGS if s >= h})
    X = {f"lag_{k}": y.shift(k) for k in lags}
    past = y.shift(h)                              # shift BEFORE rolling
    X["mean_24"] = past.rolling(24, min_periods=12).mean()
    X["mean_168"] = past.rolling(168, min_periods=84).mean()
    return pd.DataFrame(X)


def seasonal_naive(y: pd.Series, h: int, m: int = 24) -> pd.Series:
    k = int(np.ceil(h / m))
    return y.shift(m * k)


def linear_model(lag_cols: list[str]):
    ct = ColumnTransformer([("cal", OneHotEncoder(handle_unknown="ignore"), ["hour", "dow", "month"]),
                            ("num", "passthrough", ["workingday", "holiday"] + lag_cols)])
    return make_pipeline(ct, LinearRegression())


def direct_and_recursive(b: dict) -> dict:
    header("3. Horizons 1-48 h: direct vs recursive vs baselines (hold-out, MAE in rentals/hour)")
    y, frame, is_train = b["y"], b["frame"], b["is_train"]
    horizons = [1, 2, 3, 6, 12, 24, 48]
    res = {"h": horizons, "rows": {}}

    # Recursive: one model for h = 1, iterated on its own predictions.
    X1 = lag_features(y, 1).join(frame[CAL])
    ok1 = y.notna() & X1.notna().all(axis=1)
    rec_model = hgb().fit(X1[ok1 & is_train], y[ok1 & is_train])
    lag_cols_1 = [c for c in X1.columns if c.startswith("lag_")]
    lag_k = [int(c.split("_")[1]) for c in lag_cols_1]

    yv = y.to_numpy()
    N, H, L = len(yv), max(horizons), 192
    t0_pos = int(np.searchsorted(b["grid"], b["t0"]))
    origins = np.arange(t0_pos - H - 1, N - 1)
    buf = np.full((len(origins), L + H), np.nan)
    for j, o in enumerate(origins):
        buf[j, :L] = yv[o - L + 1:o + 1]
    cal_ext = frame[CAL].to_numpy(float)
    cal_ext = np.vstack([cal_ext, np.full((H, len(CAL)), np.nan)])
    rec_pred = np.full((len(origins), H), np.nan)
    with threadpool_limits(THREADS):
        for k in range(1, H + 1):
            p = L - 1 + k
            feats = [buf[:, p - lk] for lk in lag_k]
            w24, w168 = buf[:, p - 24:p], buf[:, p - 168:p]
            with np.errstate(all="ignore"):
                m24 = np.where(np.sum(~np.isnan(w24), 1) >= 12, np.nanmean(w24, 1), np.nan)
                m168 = np.where(np.sum(~np.isnan(w168), 1) >= 84, np.nanmean(w168, 1), np.nan)
            Xk = np.column_stack(feats + [m24, m168, cal_ext[origins + k]])
            pred = rec_model.predict(pd.DataFrame(Xk, columns=list(X1.columns)))
            buf[:, p] = pred
            rec_pred[:, k - 1] = pred

    for h in horizons:
        Xh = lag_features(y, h)
        lag_cols = list(Xh.columns)
        Xh = Xh.join(frame[CAL])
        naive = y.shift(h)
        snaive = seasonal_naive(y, h, 24)
        wnaive = y.shift(168)
        ok = y.notna() & Xh.notna().all(axis=1) & naive.notna() & snaive.notna() & wnaive.notna()
        tr, te = ok & is_train, ok & ~is_train
        with threadpool_limits(THREADS):
            m = hgb().fit(Xh[tr], y[tr])
            p_dir = m.predict(Xh[te])
            lin = linear_model(lag_cols).fit(Xh[tr], y[tr])
            p_lin = lin.predict(Xh[te])
        te_pos = np.flatnonzero(te.to_numpy())
        p_rec = rec_pred[te_pos - h - origins[0], h - 1]
        yt = y[te].to_numpy()
        row = {
            "naive": mae(yt, naive[te]),
            "seasonal naive (24 h)": mae(yt, snaive[te]),
            "seasonal naive (168 h)": mae(yt, wnaive[te]),
            "linear regression, direct": mae(yt, p_lin),
            "HGB direct": mae(yt, p_dir),
            "HGB recursive": mae(yt, p_rec),
            "n": int(te.sum()),
        }
        if h == 1:
            assert np.allclose(p_rec, p_dir), "recursive step 1 must equal the direct h=1 model"
        res["rows"][h] = row
        if h == 24:
            res["h24"] = {"X": Xh, "ok": ok, "lag_cols": lag_cols, "pred": p_dir, "te": te}
    say("  (recursive h=1 predictions asserted equal to the direct h=1 model)")
    names = ["naive", "seasonal naive (24 h)", "seasonal naive (168 h)", "linear regression, direct",
             "HGB direct", "HGB recursive"]
    say("  h    n     " + "  ".join(f"{n[:14]:>14s}" for n in names))
    for h in horizons:
        r = res["rows"][h]
        say(f"  {h:<4d} {r['n']:<5d} " + "  ".join(f"{r[n]:14.2f}" for n in names))
    say("  MASE (scale = training MAE of the 24-h seasonal naive) and skill vs seasonal naive (24 h):")
    for h in horizons:
        r = res["rows"][h]
        sn = r["seasonal naive (24 h)"]
        say(f"  h={h:<3d} " + "  ".join(
            f"{n.split(',')[0]}: MASE {r[n] / b['scale24']:.3f} skill {1 - r[n] / sn:+.3f}"
            for n in ["naive", "seasonal naive (24 h)", "seasonal naive (168 h)", "HGB direct", "HGB recursive"]))
    return res


def leaky_rolling(b: dict) -> None:
    header("4. The leaky rolling mean on Bike Sharing (h = 1)")
    y, frame, is_train = b["y"], b["frame"], b["is_train"]
    X = lag_features(y, 1).join(frame[CAL])
    Xl = X.copy()
    Xl["leaky_mean_24"] = y.rolling(24, min_periods=12).mean()   # includes y_t
    ok = y.notna() & Xl.notna().all(axis=1)
    tr, te = ok & is_train, ok & ~is_train
    with threadpool_limits(THREADS):
        for name, XX in {"correct features": X, "+ leaky_mean_24 (includes y_t)": Xl}.items():
            m = hgb().fit(XX[tr], y[tr])
            ho = mae(y[te], m.predict(XX[te]))
            Xtr, ytr = XX[tr], y[tr]
            cvs = []
            for gap in (0, 192):
                cv = TimeSeriesSplit(n_splits=5, test_size=1000, gap=gap)
                s = [mae(ytr.iloc[b_], hgb().fit(Xtr.iloc[a_], ytr.iloc[a_]).predict(Xtr.iloc[b_]))
                     for a_, b_ in cv.split(Xtr)]
                cvs.append(np.mean(s))
            say(f"  {name:<34s} hold-out MAE {ho:6.2f}   TimeSeriesSplit(5, test_size=1000) CV MAE "
                f"gap=0: {cvs[0]:6.2f}  gap=192: {cvs[1]:6.2f}")


# ---------------------------------------------------------------------------
# 4. Validation on the training period (h = 24)
# ---------------------------------------------------------------------------

def validation(b: dict, d: dict) -> dict:
    header("5. Validating the 24-h direct model on the training period only")
    y = b["y"]
    h24 = d["h24"]
    X, ok, te = h24["X"], h24["ok"], h24["te"]
    tr = ok & b["is_train"]
    Xtr, ytr = X[tr], y[tr]
    truth = mae(y[te], h24["pred"])
    say(f"  training rows {len(ytr)}; truth = hold-out MAE of the same model {truth:.2f}")
    res = {"truth": truth}

    def run(cv, name):
        t = time.perf_counter()
        scores, sizes = [], []
        with threadpool_limits(THREADS):
            for a_, b_ in cv.split(Xtr):
                m = hgb().fit(Xtr.iloc[a_], ytr.iloc[a_])
                scores.append(mae(ytr.iloc[b_], m.predict(Xtr.iloc[b_])))
                sizes.append(len(a_))
        res[name] = float(np.mean(scores))
        say(f"  {name:<48s} CV MAE {np.mean(scores):6.2f}  (folds {len(scores)}, train sizes "
            f"{min(sizes)}-{max(sizes)}, {time.perf_counter() - t:.0f}s)")

    run(KFold(5, shuffle=True, random_state=0), "KFold(5, shuffle=True)")
    for gap in (0, 23, 24 + 168):
        run(TimeSeriesSplit(n_splits=5, test_size=1000, gap=gap),
            f"TimeSeriesSplit(5, test_size=1000, gap={gap})")
    run(TimeSeriesSplit(n_splits=5, test_size=1000, gap=23, max_train_size=24 * 7 * 8),
        "same, gap=23, max_train_size=1344 (8 weeks)")
    run(TimeSeriesSplit(n_splits=5, test_size=1000, gap=23, max_train_size=24 * 7 * 26),
        "same, gap=23, max_train_size=4368 (26 weeks)")
    # Small test folds make any train/test overlap a larger share of the score.
    for gap in (0, 23):
        run(TimeSeriesSplit(n_splits=40, test_size=24, gap=gap),
            f"TimeSeriesSplit(40, test_size=24, gap={gap})")

    # Random forest OOB on the same table
    with threadpool_limits(THREADS):
        rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.5,
                                   oob_score=True, n_jobs=THREADS, random_state=0).fit(Xtr, ytr)
        oob = mae(ytr, rf.oob_prediction_)
        rf_ho = mae(y[te], rf.predict(X[te]))
    res["rf_oob"], res["rf_holdout"] = oob, rf_ho
    say(f"  RandomForest(200, min_samples_leaf=5, max_features=0.5): OOB MAE {oob:.2f}, hold-out MAE {rf_ho:.2f}")
    return res


# ---------------------------------------------------------------------------
# 5. What trees add
# ---------------------------------------------------------------------------

def feature_sets(b: dict) -> dict:
    header("6. Bike Sharing, h = 24 direct: which inputs matter (hold-out)")
    y, frame, is_train = b["y"], b["frame"], b["is_train"]
    L = lag_features(y, 24)
    lag_cols = list(L.columns)
    sets = {
        "lags only": L,
        "lags + calendar": L.join(frame[CAL]),
        "lags + calendar + observed weather": L.join(frame[CAL + WEATHER]),
        "calendar + observed weather, no lags": frame[CAL + WEATHER],
    }
    ok_all = y.notna() & L.notna().all(axis=1)
    tr, te = ok_all & is_train, ok_all & ~is_train
    res = {}
    with threadpool_limits(THREADS):
        for name, XX in sets.items():
            p = hgb().fit(XX[tr], y[tr]).predict(XX[te])
            res[name] = (mae(y[te], p), float(np.mean(y[te] - p)))
            say(f"  HGB {name:<40s} MAE {res[name][0]:6.2f}  mean error (y - pred) {res[name][1]:+7.2f}")
        p = linear_model(lag_cols).fit(sets["lags + calendar"][tr], y[tr]).predict(sets["lags + calendar"][te])
        res["linear lags + calendar"] = (mae(y[te], p), float(np.mean(y[te] - p)))
        say(f"  LinearRegression lags + one-hot calendar        MAE {res['linear lags + calendar'][0]:6.2f}"
            f"  mean error {res['linear lags + calendar'][1]:+7.2f}")
        sn = seasonal_naive(y, 24)
        say(f"  seasonal naive (24 h) on the same rows          MAE {mae(y[te], sn[te]):6.2f}")
    say(f"  rows: train {int(tr.sum())}, hold-out {int(te.sum())}")
    return res


def global_vs_local() -> dict:
    header("7. Many related series: one global table vs one model per series (synthetic)")
    # Documented generator: 40 daily series sharing a weekly profile and a
    # non-linear heat effect, each with its own level and AR(1) noise.
    rng = np.random.default_rng(7)
    S, T = 40, 400
    t = np.arange(T)
    temp = 28 + 5 * np.sin(2 * np.pi * t / 365) + rng.normal(0, 1.5, T)
    weekly = np.array([1.0, 1.05, 1.05, 1.1, 1.25, 1.5, 0.7])
    heat = 1 - 0.35 * np.clip(temp - 30, 0, None) / 4
    rows = []
    for i in range(S):
        level = rng.uniform(20, 200)
        wk = weekly * (1 + rng.normal(0, 0.05, 7))
        e = np.zeros(T)
        for k in range(1, T):
            e[k] = 0.6 * e[k - 1] + rng.normal(0, 0.08)
        yi = level * wk[t % 7] * heat * np.exp(e)
        s = pd.Series(yi)
        rows.append(pd.DataFrame({
            "id": i, "t": t, "dow": t % 7, "temp": temp, "y": yi,
            "lag_1": s.shift(1), "lag_7": s.shift(7), "mean_7": s.shift(1).rolling(7).mean(),
            "scale": np.nan}))
    df = pd.concat(rows, ignore_index=True).dropna(subset=["lag_7", "mean_7"])
    train, test = df[df.t < 300].copy(), df[df.t >= 300].copy()
    # Per-series scale from training data only, used for the normalised global model and MASE.
    sc = train.groupby("id")["y"].mean()
    mase_scale = train.assign(d=(train.y - train.lag_7).abs()).groupby("id")["d"].mean()
    for part in (train, test):
        part["scale"] = part["id"].map(sc)
    feats = ["dow", "temp", "lag_1", "lag_7", "mean_7"]
    preds = {}
    with threadpool_limits(THREADS):
        g = hgb(categorical_features=[0]).fit(train[["id"] + feats], train.y)
        preds["global HGB (id as categorical), raw levels"] = g.predict(test[["id"] + feats])
        tn = train.copy()
        ten = test.copy()
        for c in ("y", "lag_1", "lag_7", "mean_7"):
            tn[c] = tn[c] / tn["scale"]
            ten[c] = ten[c] / ten["scale"]
        gn = hgb(categorical_features=[0]).fit(tn[["id"] + feats], tn.y)
        preds["global HGB, each series divided by its training mean"] = gn.predict(ten[["id"] + feats]) * test["scale"].to_numpy()
        loc = np.empty(len(test))
        for i in range(S):
            mtr, mte = train.id == i, (test.id == i).to_numpy()
            loc[mte] = hgb().fit(train.loc[mtr, feats], train.y[mtr]).predict(test.loc[mte, feats])
        preds["one HGB per series (293 rows each)"] = loc
        lin = LinearRegression().fit(tn[feats], tn.y)
        preds["global linear regression, normalised"] = lin.predict(ten[feats]) * test["scale"].to_numpy()
    preds["seasonal naive (lag 7)"] = test.lag_7.to_numpy()
    preds["naive (lag 1)"] = test.lag_1.to_numpy()
    res = {}
    for name, p in preds.items():
        err = pd.Series(np.abs(test.y.to_numpy() - p)).groupby(test.id.to_numpy()).mean()
        res[name] = float((err / mase_scale).mean())
        say(f"  {name:<56s} mean MASE over 40 series {res[name]:.3f}")
    say(f"  train rows {len(train)} (40 x {len(train) // 40}), test rows {len(test)}")
    return res


# ---------------------------------------------------------------------------
# 6. Extrapolation on a trending series
# ---------------------------------------------------------------------------

def trending_series(seed: int = 0, n: int = 1000) -> np.ndarray:
    """y_t = 50 + 0.1 t + 8 sin(2 pi t / 7) + N(0, 2^2): linear trend, weekly season, noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 50 + 0.1 * t + 8 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 2, n)


def extrapolation() -> dict:
    header("8. Extrapolation: trending series (n = 1000, train = first 700), one-step forecasts")
    yv = trending_series()
    n, cut = len(yv), 700
    y = pd.Series(yv)
    t = np.arange(n)
    dow = t % 7

    def lag_table(s: pd.Series) -> pd.DataFrame:
        X = pd.DataFrame({f"lag_{k}": s.shift(k) for k in range(1, 8)})
        X["mean_7"] = s.shift(1).rolling(7).mean()
        X["dow"] = dow
        return X

    first = 15
    tr_idx = np.arange(first, cut)
    te_idx = np.arange(cut, n)
    yt = yv[te_idx]
    say(f"  training y range {yv[:cut].min():.2f} .. {yv[:cut].max():.2f}; test y range "
        f"{yt.min():.2f} .. {yt.max():.2f}")
    preds = {}
    X = lag_table(y)
    with threadpool_limits(THREADS):
        preds["naive"] = yv[te_idx - 1]
        preds["seasonal naive (m = 7)"] = yv[te_idx - 7]
        lin_X = pd.get_dummies(X.drop(columns="dow").assign(dow=dow.astype(str)), columns=["dow"], dtype=float)
        preds["linear regression on lags (AR)"] = LinearRegression().fit(
            lin_X.iloc[tr_idx], yv[tr_idx]).predict(lin_X.iloc[te_idx])
        preds["HGB on levels"] = hgb().fit(X.iloc[tr_idx], yv[tr_idx]).predict(X.iloc[te_idx])
        preds["RandomForest on levels"] = RandomForestRegressor(
            n_estimators=300, n_jobs=THREADS, random_state=0).fit(X.iloc[tr_idx], yv[tr_idx]).predict(X.iloc[te_idx])

        # Fix 1: model the change d_t = y_t - y_{t-1}
        d = y.diff()
        Xd = pd.DataFrame({f"dlag_{k}": d.shift(k) for k in range(1, 8)})
        Xd["dow"] = dow
        dhat = hgb().fit(Xd.iloc[tr_idx], d.iloc[tr_idx]).predict(Xd.iloc[te_idx])
        preds["HGB on differences"] = yv[te_idx - 1] + dhat

        # Fix 2: detrend with a straight line fitted on training data, boost the residual
        trend = LinearRegression().fit(t[:cut].reshape(-1, 1), yv[:cut])
        tr_line = trend.predict(t.reshape(-1, 1))
        r = pd.Series(yv - tr_line)
        Xr = lag_table(r)
        rhat = hgb().fit(Xr.iloc[tr_idx], r.iloc[tr_idx]).predict(Xr.iloc[te_idx])
        preds["linear trend + HGB on residual"] = tr_line[te_idx] + rhat
        say(f"  fitted trend: intercept {trend.intercept_:.3f}, slope {trend.coef_[0]:.4f} (true 50, 0.1)")

        # Fix 3: divide target and lags by the mean of the previous 7 values
        lvl = y.shift(1).rolling(7).mean()
        Xn = X.drop(columns=["dow", "mean_7"]).div(lvl, axis=0)
        Xn["dow"] = dow
        yn = y / lvl
        preds["HGB on y / mean of last 7"] = hgb().fit(Xn.iloc[tr_idx], yn.iloc[tr_idx]).predict(
            Xn.iloc[te_idx]) * lvl.iloc[te_idx].to_numpy()

        # Not a fix: log target (still a level, still bounded)
        ly = np.log(y)
        Xl = lag_table(ly)
        preds["HGB on log(y)"] = np.exp(hgb().fit(Xl.iloc[tr_idx], ly.iloc[tr_idx]).predict(Xl.iloc[te_idx]))

        # LightGBM linear_tree: linear models in the leaves
        for name, lt in (("LightGBM on levels", False), ("LightGBM linear_tree=True on levels", True)):
            m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, linear_tree=lt,
                                  n_jobs=THREADS, random_state=0, verbose=-1)
            m.fit(X.iloc[tr_idx], yv[tr_idx])
            preds[name] = m.predict(X.iloc[te_idx])
            if lt:
                assert m.booster_.dump_model()["tree_info"][1]["tree_structure"].get("leaf_coeff") is not None \
                    or "leaf_coeff" in str(m.booster_.dump_model()["tree_info"][1])
                say("  checked: LightGBM 4.7.0 booster dump has leaf_coeff (linear leaves) when linear_tree=True")

    res = {}
    late = te_idx >= 900
    for name, p in preds.items():
        res[name] = (mae(yt, p), mae(yt[late], p[late]), float(np.max(p)))
        say(f"  {name:<40s} test MAE {res[name][0]:6.2f}  MAE on t >= 900 {res[name][1]:6.2f}  "
            f"max prediction {res[name][2]:7.2f}")
    say(f"  training max y {yv[:cut].max():.2f}; true values on t >= 900 average {yt[late].mean():.2f}")
    return {"y": yv, "cut": cut, "te_idx": te_idx, "preds": preds, "res": res}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure_lag_table(tab: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 3.9), dpi=150)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    rows = [("y", "y"), ("lag_1", "lag 1  = y.shift(1)"), ("lag_2", "lag 2  = y.shift(2)"),
            ("mean_3", "mean 3 = y.shift(1).rolling(3)"),
            ("leaky_mean_3", "leaky = y.rolling(3)")]
    target_t = 7
    for r, (col, label) in enumerate(rows):
        yy = len(rows) - 1 - r
        ax.text(-0.25, yy + 0.45, label, ha="right", va="center", fontsize=9.5, color=INK, family="monospace")
        for t in range(1, 11):
            v = tab.loc[t, col]
            face = PAPER
            if col == "y" and target_t - 3 <= t <= target_t - 1:
                face = TEAL_LIGHT
            if col == "y" and t == target_t:
                face = AMBER
            if t == target_t and col in ("lag_1", "lag_2", "mean_3"):
                face = TEAL_LIGHT
            if t == target_t and col == "leaky_mean_3":
                face = "#FCD9A0"
            ax.add_patch(Rectangle((t - 1, yy), 0.92, 0.9, fc=face, ec=GREY, lw=0.8))
            txt = "NaN" if pd.isna(v) else (f"{v:.0f}" if float(v).is_integer() else f"{v:.2f}")
            ax.text(t - 1 + 0.46, yy + 0.45, txt, ha="center", va="center", fontsize=9,
                    color=INK_2 if pd.isna(v) else INK)
    for t in range(1, 11):
        ax.text(t - 1 + 0.46, len(rows) + 0.2, f"t={t}", ha="center", fontsize=8.5, color=INK_2)
    # Window brackets on the y row
    ytop = len(rows) - 1
    ax.annotate("", xy=(target_t - 4 + 0.05, ytop - 0.12), xytext=(target_t - 1 - 0.13, ytop - 0.12),
                arrowprops=dict(arrowstyle="-", color=TEAL, lw=2.2))
    ax.text(10.3, 1.45, "row t = 7:\nmean 3 uses y4..y6 (known)\nleaky uses y5..y7 — y7 is the\nvalue being predicted",
            fontsize=9, color=INK, va="center")
    ax.set_xlim(-4.2, 13.6)
    ax.set_ylim(-0.3, len(rows) + 0.6)
    ax.axis("off")
    ax.legend(handles=[Patch(color=AMBER, label="target y_t"),
                       Patch(color=TEAL_LIGHT, label="values a row may use"),
                       Patch(color="#FCD9A0", label="feature that contains y_t (leak)")],
              loc="lower right", bbox_to_anchor=(1.0, -0.08), frameon=False, fontsize=8.5, labelcolor=INK, ncol=3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lag-table.png", facecolor=PAPER)
    plt.close(fig)


def figure_folds_and_horizon(d: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), dpi=150, gridspec_kw={"width_ratios": [1.25, 1]})
    fig.patch.set_facecolor(PAPER)
    ax = axes[0]
    ax.set_facecolor(PAPER)
    n, test_size, gap = 48, 6, 3
    labels = []
    schemes = [("expanding", None), ("sliding (max_train_size=18)", 18)]
    yrow = 0
    for title, mts in schemes:
        cv = TimeSeriesSplit(n_splits=4, test_size=test_size, gap=gap, max_train_size=mts)
        for k, (tr, te) in enumerate(cv.split(np.zeros(n))):
            for i in range(n):
                if i in set(tr):
                    c = TEAL_LIGHT
                elif i in set(te):
                    c = TEAL
                elif i < te[0] and i > tr[-1]:
                    c = AMBER
                else:
                    c = "#EFEBE3"
                ax.add_patch(Rectangle((i, -yrow), 0.9, 0.8, fc=c, ec="none"))
            labels.append((-yrow + 0.4, f"{title}, fold {k + 1}"))
            yrow += 1
        yrow += 0.6
    for yy, lab in labels:
        ax.text(-0.8, yy, lab, ha="right", va="center", fontsize=8.5, color=INK)
    ax.set_xlim(-17, n + 0.5)
    ax.set_ylim(-yrow + 0.4, 1.4)
    ax.axis("off")
    ax.text(0, 1.05, "TimeSeriesSplit(n_splits=4, test_size=6, gap=3) on 48 time steps", fontsize=10, color=INK)
    ax.legend(handles=[Patch(color=TEAL_LIGHT, label="train"), Patch(color=AMBER, label="gap (dropped)"),
                       Patch(color=TEAL, label="test"), Patch(color="#EFEBE3", label="unused")],
              loc="lower center", bbox_to_anchor=(0.62, -0.12), ncol=4, frameon=False, fontsize=8.5, labelcolor=INK)

    ax = axes[1]
    style_axes(ax)
    hs = d["h"]
    series = [("HGB direct", TEAL, "-", "o"), ("HGB recursive", TEAL, "--", "s"),
              ("seasonal naive (24 h)", INK_2, "-", "^"), ("naive", GREY, "-", "v"),
              ("linear regression, direct", AMBER, "-", "D")]
    for name, col, ls, mk in series:
        ax.plot(range(len(hs)), [d["rows"][h][name] for h in hs], color=col, ls=ls, marker=mk, ms=4.5,
                lw=1.8, label=name)
    ax.set_xticks(range(len(hs)), [str(h) for h in hs])
    ax.set_xlabel("forecast horizon h (hours)")
    ax.set_ylabel("hold-out MAE (rentals / hour)")
    ax.set_title("Bike Sharing: error grows with the horizon", fontsize=11)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "folds-and-horizon.png", facecolor=PAPER)
    plt.close(fig)


def figure_extrapolation(e: dict) -> None:
    yv, cut, te_idx, preds = e["y"], e["cut"], e["te_idx"], e["preds"]
    fig, ax = plt.subplots(figsize=(11, 4.4), dpi=150)
    fig.patch.set_facecolor(PAPER)
    style_axes(ax)
    t = np.arange(len(yv))
    ax.axvspan(cut, len(yv), color="#EFEBE3", lw=0)
    ax.plot(t, yv, color=GREY, lw=0.9, label="series (trend + weekly season + noise)")
    ax.axhline(yv[:cut].max(), color=INK_2, lw=0.8, ls=":")
    ax.text(30, yv[:cut].max() + 2, f"highest training value {yv[:cut].max():.1f}", fontsize=8.5, color=INK_2)
    ax.plot(te_idx, preds["HGB on levels"], color=AMBER, lw=1.4, label="HGB on levels (flattens)")
    ax.plot(te_idx, preds["linear trend + HGB on residual"], color=TEAL, lw=1.2,
            label="linear trend + HGB on residual")
    ax.text(cut + 8, yv.min() + 2, "test: one-step forecasts", fontsize=9, color=INK)
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.set_title("A tree cannot predict beyond the values it was trained on", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "trend-extrapolation.png", facecolor=PAPER)
    plt.close(fig)


def main() -> None:
    start = time.perf_counter()
    FIG_DIR.mkdir(exist_ok=True)
    say(f"scikit-learn {sklearn.__version__}, lightgbm {lgb.__version__}, numpy {np.__version__}, "
        f"pandas {pd.__version__}; threads per model {THREADS}")
    tab = hand_example()
    b = load_bike()
    d = direct_and_recursive(b)
    leaky_rolling(b)
    validation(b, d)
    feature_sets(b)
    global_vs_local()
    e = extrapolation()
    figure_lag_table(tab)
    figure_folds_and_horizon(d)
    figure_extrapolation(e)
    say()
    say(f"total runtime {time.perf_counter() - start:.0f}s")
    (HERE / "results.txt").write_text(out.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
