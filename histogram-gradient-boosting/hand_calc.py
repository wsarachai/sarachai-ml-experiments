"""Hand calculation for histogram-based split finding, checked against real libraries.

Data: the 6-row "study hours -> exam score" example from the XGBoost post,
extended to 8 rows, plus a second feature (hours of sleep) so the histogram
subtraction trick has something to do.

Setup used by the hand calculation:
    squared-error loss, one split (a stump), learning rate 1, lambda 0.
    g = F0 - y, h = 1, F0 = mean(y)

Run:  python hand_calc.py   (prints everything to stdout)
"""

import numpy as np
import xgboost as xgb
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper

hours = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
sleep = np.array([5, 7, 6, 8, 5, 7, 8, 6], dtype=float)
score = np.array([10, 12, 11, 40, 42, 41, 47, 45], dtype=float)
X = np.column_stack([hours, sleep])

print(f"scikit-learn {sklearn.__version__}, xgboost {xgb.__version__}")
print()

F0 = score.mean()
g = F0 - score
h = np.ones_like(g)
print(f"F0 = {F0}")
print(f"g  = {g.tolist()}   sum = {g.sum()}")
print()


def gain(GL, HL, GR, HR, lam=0.0):
    """Gain formula from the XGBoost paper (with the 1/2 factor)."""
    G, H = GL + GR, HL + HR
    return 0.5 * (GL**2 / (HL + lam) + GR**2 / (HR + lam) - G**2 / (H + lam))


# ---------------------------------------------------------------- exact search
print("== Exact search on 'hours' (every midpoint) ==")
order = np.argsort(hours)
best_exact = None
for k in range(1, len(hours)):
    left = order[:k]
    right = order[k:]
    thr = (hours[order[k - 1]] + hours[order[k]]) / 2
    GL, HL = g[left].sum(), h[left].sum()
    GR, HR = g[right].sum(), h[right].sum()
    gn = gain(GL, HL, GR, HR)
    print(f"  x < {thr:3.1f}: GL={GL:6.1f} HL={HL:.0f} GR={GR:6.1f} HR={HR:.0f} gain={gn:8.3f}")
    if best_exact is None or gn > best_exact[1]:
        best_exact = (thr, gn, GL, HL, GR, HR)
thr, gn, GL, HL, GR, HR = best_exact
print(f"  best exact: x < {thr}  gain={gn:.3f}  leaves {F0 - GL / HL:.4f} / {F0 - GR / HR:.4f}")
print()

# ---------------------------------------------------------------- binning
print("== Binning with scikit-learn's _BinMapper, max_bins=4 ==")
mapper = _BinMapper(n_bins=4 + 1, random_state=0).fit(X)  # +1 bin reserved for NaN
Xb = mapper.transform(X)
for j, name in enumerate(["hours", "sleep"]):
    print(f"  {name}: thresholds {mapper.bin_thresholds_[j].tolist()}  bins {Xb[:, j].tolist()}")
print()


def histogram(bins, rows, n_bins):
    G = np.zeros(n_bins)
    H = np.zeros(n_bins)
    for i in rows:
        G[bins[i]] += g[i]
        H[bins[i]] += h[i]
    return G, H


all_rows = np.arange(len(score))
print("== Root histograms (sum of g, sum of h per bin) ==")
root_hist = {}
for j, name in enumerate(["hours", "sleep"]):
    G, H = histogram(Xb[:, j], all_rows, 4)
    root_hist[j] = (G, H)
    print(f"  {name}: G={G.tolist()}  H={H.tolist()}")
print()

print("== Split search over bin boundaries (cumulative sums) ==")
best_hist = None
for j, name in enumerate(["hours", "sleep"]):
    G, H = root_hist[j]
    cg, ch = np.cumsum(G), np.cumsum(H)
    for b in range(3):
        gn = gain(cg[b], ch[b], G.sum() - cg[b], H.sum() - ch[b])
        print(f"  {name}: bins 0..{b} | {b + 1}..3  GL={cg[b]:6.1f} HL={ch[b]:.0f} gain={gn:8.3f}")
        if best_hist is None or gn > best_hist[2]:
            best_hist = (j, b, gn, cg[b], ch[b])
j, b, gn, GLb, HLb = best_hist
print(f"  best hist: feature {j} after bin {b}, threshold {mapper.bin_thresholds_[j][b]}, gain={gn:.3f}")
print(f"  leaves {F0 - GLb / HLb:.4f} / {F0 + GLb / (8 - HLb):.4f}")
print()

# ---------------------------------------------------------------- subtraction trick
print("== Histogram subtraction: right child = parent - left child ==")
left_rows = all_rows[Xb[:, 0] <= b]
right_rows = all_rows[Xb[:, 0] > b]
for jj, name in enumerate(["hours", "sleep"]):
    Gp, Hp = root_hist[jj]
    Gl, Hl = histogram(Xb[:, jj], left_rows, 4)
    Gr_sub, Hr_sub = Gp - Gl, Hp - Hl
    Gr_dir, Hr_dir = histogram(Xb[:, jj], right_rows, 4)
    print(f"  {name}: left (built) G={Gl.tolist()} H={Hl.tolist()}")
    print(f"  {name}: right (subtracted) G={Gr_sub.tolist()} H={Hr_sub.tolist()}")
    print(f"  {name}: right (built directly) G={Gr_dir.tolist()}  match={np.allclose(Gr_sub, Gr_dir)}")
print()

# ---------------------------------------------------------------- real libraries
print("== scikit-learn HistGradientBoostingRegressor, one stump ==")
for mb in (4, 255):
    m = HistGradientBoostingRegressor(
        max_iter=1, learning_rate=1.0, max_depth=1, min_samples_leaf=1,
        l2_regularization=0.0, max_bins=mb, early_stopping=False,
    ).fit(X, score)
    nodes = m._predictors[0][0].nodes
    root = nodes[0]
    print(f"  max_bins={mb}: baseline={m._baseline_prediction.ravel()[0]}, "
          f"feature={root['feature_idx']}, threshold={root['num_threshold']}, "
          f"root gain field={root['gain']:.4f}")
    print(f"    predictions {m.predict(X).tolist()}")
print()

print("== xgboost: tree_method exact vs hist (max_bin=4) ==")
d = xgb.DMatrix(X, label=score, feature_names=["hours", "sleep"])
for tm, extra in (("exact", {}), ("hist", {"max_bin": 4})):
    params = dict(objective="reg:squarederror", max_depth=1, eta=1.0,
                  reg_lambda=0.0, gamma=0.0, min_child_weight=0.0,
                  base_score=F0, tree_method=tm, **extra)
    bst = xgb.train(params, d, num_boost_round=1)
    print(f"  {tm}: {bst.get_dump(with_stats=True)[0].strip()}")
    print(f"    predictions {bst.predict(d).tolist()}")
print()

# ---------------------------------------------------------------- missing values
print("== Missing value: hours of row 4 set to NaN, max_bins=4 ==")
Xm = X.copy()
Xm[3, 0] = np.nan
mm = _BinMapper(n_bins=5, random_state=0).fit(Xm)
Xbm = mm.transform(Xm)
print(f"  hours thresholds {mm.bin_thresholds_[0].tolist()}  bins {Xbm[:, 0].tolist()} "
      f"(missing bin index {mm.missing_values_bin_idx_})")
G = np.zeros(5)
H = np.zeros(5)
for i in all_rows:
    G[Xbm[i, 0]] += g[i]
    H[Xbm[i, 0]] += h[i]
print(f"  histogram G={G.tolist()} H={H.tolist()}")
Gm, Hm = G[4], H[4]
for b in range(3):
    GL = G[: b + 1].sum()
    HL = H[: b + 1].sum()
    gr = gain(GL, HL, 0 - GL, 8 - HL)  # missing goes right
    gl = gain(GL + Gm, HL + Hm, 0 - GL - Gm, 8 - HL - Hm)  # missing goes left
    print(f"  after bin {b}: missing->right gain={gr:8.3f}  missing->left gain={gl:8.3f}")
m = HistGradientBoostingRegressor(
    max_iter=1, learning_rate=1.0, max_depth=1, min_samples_leaf=1,
    l2_regularization=0.0, max_bins=4, early_stopping=False,
).fit(Xm, score)
root = m._predictors[0][0].nodes[0]
print(f"  sklearn: feature={root['feature_idx']} threshold={root['num_threshold']} "
      f"missing_go_to_left={bool(root['missing_go_to_left'])} gain={root['gain']:.4f}")
print(f"    predictions {m.predict(Xm).tolist()}")

# ---------------------------------------------------------------- two extra checks
print()
print("== l2_regularization=1 (lambda in -G/(H+lambda)), max_bins=255 ==")
m = HistGradientBoostingRegressor(
    max_iter=1, learning_rate=1.0, max_depth=1, min_samples_leaf=1,
    l2_regularization=1.0, early_stopping=False,
).fit(X, score)
print(f"  predictions {m.predict(X).tolist()}  root gain field={m._predictors[0][0].nodes[0]['gain']:.4f}")
print("  hand: left 31 - 60/(3+1) = 16, right 31 + 60/(5+1) = 41")

print("== Default min_samples_leaf=20 on 8 rows ==")
m = HistGradientBoostingRegressor(
    max_iter=1, learning_rate=1.0, max_depth=1, early_stopping=False).fit(X, score)
print(f"  predictions {m.predict(X).tolist()}  (no split is possible)")
