"""Hand calculations in the "stacking-and-voting" post, redone and asserted.

1. Hard vs soft voting: 3 classifiers x 6 rows of P(class 1), using sklearn's
   hard-voting rule (argmax of bincount) and soft rule (mean probability).
2. Majority vote of 3 classifiers with accuracy p: 3p^2 - 2p^3 when errors are
   independent (exact enumeration + simulation), p when errors are perfectly
   correlated, and 1.0 when the errors never overlap.
3. Stacking by hand: 2-fold out-of-fold predictions of a straight line and a
   1-nearest-neighbour regressor on 6 rows, a least-squares meta-learner (no
   intercept) solved from the 2x2 normal equations, and the same meta-learner
   fitted on in-sample predictions (the leak), scored on 2 new rows.
4. The MSE-of-an-average formula with the measured California numbers (seed 0):
   RMSE and residual correlations rounded to 4 decimals as printed in the post,
   for LightGBM + ridge (2 models) and the three boosters (read from results.json).

Run:    .venv/Scripts/python.exe stacking-and-voting/hand_check.py
"""

from __future__ import annotations

import itertools

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor


def close(a, b, tol=5e-5):
    assert abs(a - b) <= tol, (a, b)


# ------------------------------------------------------------------ 1. hard vs soft
y = np.array([1, 0, 1, 1, 0, 1])
P = np.array([  # columns: C1, C2, C3 = P(class 1)
    [0.90, 0.80, 0.70],
    [0.20, 0.10, 0.40],
    [0.60, 0.55, 0.10],
    [0.45, 0.48, 0.95],
    [0.30, 0.65, 0.20],
    [0.70, 0.35, 0.80],
])
labels = (P >= 0.5).astype(int)
acc_each = (labels == y[:, None]).mean(0)
assert list(np.round(acc_each * 6).astype(int)) == [5, 3, 5]
# sklearn's hard rule: argmax(bincount(labels)); with 3 voters there is no tie
hard = np.array([np.argmax(np.bincount(r, minlength=2)) for r in labels])
soft_p = P.mean(1)
soft = (soft_p > 0.5).astype(int)  # argmax of [1-p, p]
print("soft averages:", np.round(soft_p, 4))
assert list(hard) == [1, 0, 1, 0, 0, 1]
assert list(soft) == [1, 0, 0, 1, 0, 1]
close(soft_p[2], 1.25 / 3)
close(soft_p[3], 1.88 / 3)
assert (hard == y).sum() == 5 and (soft == y).sum() == 5
print("hard vs soft: rows where they disagree:", [int(i) + 1 for i in np.where(hard != soft)[0]])

# soft voting with weights 1,1,1 on log-odds would differ; show what one over-confident model does:
# row 4 is decided by C3's 0.95 alone.
close((0.45 + 0.48 + 0.95) / 3, 0.6267, 5e-5)

# ------------------------------------------------------------------ 2. majority of three
def majority_independent(p):
    total = 0.0
    for outcome in itertools.product([0, 1], repeat=3):  # 1 = correct
        prob = np.prod([p if o else 1 - p for o in outcome])
        if sum(outcome) >= 2:
            total += prob
    return total


for p in [0.55, 0.6, 0.7, 0.8]:
    close(majority_independent(p), 3 * p**2 - 2 * p**3, 1e-12)
close(3 * 0.7**2 - 2 * 0.7**3, 0.784, 1e-12)
close(3 * 0.6**2 - 2 * 0.6**3, 0.648, 1e-12)
close(3 * 0.55**2 - 2 * 0.55**3, 0.57475, 1e-12)
close(3 * 0.45**2 - 2 * 0.45**3, 0.42525, 1e-12)

rng = np.random.default_rng(0)
n = 1_000_000
ind = rng.random((n, 3)) < 0.7  # independent errors
sim = (ind.sum(1) >= 2).mean()
print(f"simulated independent majority (p=0.7): {sim:.4f}")
close(sim, 0.784, 0.002)
same = np.repeat(rng.random((n, 1)) < 0.7, 3, axis=1)  # perfectly correlated: identical errors
close((same.sum(1) >= 2).mean(), same[:, 0].mean(), 1e-12)

# 10 rows, each classifier wrong on 3 different rows (accuracy 0.7 each, errors never overlap)
correct = np.ones((10, 3), dtype=int)
correct[0:3, 0] = 0
correct[3:6, 1] = 0
correct[6:9, 2] = 0
assert list(correct.mean(0)) == [0.7, 0.7, 0.7]
assert (correct.sum(1) >= 2).mean() == 1.0
print("majority of 3: independent 0.784, identical errors 0.7, disjoint errors 1.0")

# ------------------------------------------------------------------ 3. stacking by hand
x = np.array([1, 2, 4, 5, 7, 8], dtype=float)
yr = np.array([1, 3, 4, 3, 7, 9], dtype=float)
folds = [np.array([0, 2, 4]), np.array([1, 3, 5])]  # rows 1,3,5 and rows 2,4,6


def line(xt, yt):
    return LinearRegression().fit(xt[:, None], yt)


def nn1(xt, yt):
    return KNeighborsRegressor(n_neighbors=1).fit(xt[:, None], yt)


A = np.zeros(6)
B = np.zeros(6)
for k, held in enumerate(folds):
    fit_rows = folds[1 - k]
    A[held] = line(x[fit_rows], yr[fit_rows]).predict(x[held, None])
    B[held] = nn1(x[fit_rows], yr[fit_rows]).predict(x[held, None])
assert np.allclose(A, [1, 2, 4, 5, 7, 8])  # both fold lines are y = x
assert np.allclose(B, [3, 1, 3, 4, 9, 7])
G = np.array([[A @ A, A @ B], [A @ B, B @ B]])
r = np.array([A @ yr, B @ yr])
assert np.allclose(G, [[159, 156], [156, 165]]) and np.allclose(r, [159, 156])
det = G[0, 0] * G[1, 1] - G[0, 1] ** 2
close(det, 1899, 1e-9)
w = np.linalg.solve(G, r)
assert np.allclose(w, [1, 0])
print("OOF meta weights (line, 1-NN):", np.round(w, 6))

# in-sample: base models refitted on all 6 rows, predicting the same 6 rows
full_line, full_nn = line(x, yr), nn1(x, yr)
close(full_line.coef_[0], 1.0, 1e-12)
close(full_line.intercept_, 0.0, 1e-12)
Ain, Bin = full_line.predict(x[:, None]), full_nn.predict(x[:, None])
assert np.allclose(Bin, yr)  # 1-NN memorises the training rows
w_leak = np.linalg.lstsq(np.column_stack([Ain, Bin]), yr, rcond=None)[0]
assert np.allclose(w_leak, [0, 1])
print("in-sample meta weights (line, 1-NN):", np.round(w_leak, 6))

# two new rows
xt = np.array([3.4, 6.4])
yt = np.array([3.5, 6.0])
At, Bt = full_line.predict(xt[:, None]), full_nn.predict(xt[:, None])
assert np.allclose(At, [3.4, 6.4]) and np.allclose(Bt, [4, 7])
mse_oof = np.mean((yt - np.column_stack([At, Bt]) @ w) ** 2)
mse_leak = np.mean((yt - np.column_stack([At, Bt]) @ w_leak) ** 2)
close(mse_oof, 0.085, 1e-12)
close(mse_leak, 0.625, 1e-12)
print(f"new rows: MSE with OOF weights {mse_oof:.3f}, with in-sample weights {mse_leak:.3f}")

# ------------------------------------------------------------------ 4. MSE of an average, real numbers
import json  # noqa: E402
from pathlib import Path  # noqa: E402

G = json.loads((Path(__file__).resolve().parent / "results.json").read_text(encoding="utf-8"))["california"]["groups"]


def sd_corr(group):
    M = np.array(G[group]["identity_seed0"]["M"])
    s = np.sqrt(np.diag(M))
    return s, M / np.outer(s, s)


# two models: LightGBM (index 0) and ridge (index 2) of the diverse group
s, C = sd_corr("diverse")
close(s[0], 0.4344)
close(s[2], 0.7276)
close(C[0, 2], 0.5936)
s1, s2, r = 0.4344, 0.7276, 0.5936
mse2 = (s1**2 + s2**2 + 2 * r * s1 * s2) / 4
close(s1**2, 0.18870, 1e-5)
close(s2**2, 0.52940, 1e-5)
close(2 * r * s1 * s2, 0.37524, 1e-5)
close(mse2, 0.27334, 1e-5)
close(np.sqrt(mse2), 0.5228, 5e-5)
close(s2**2 + 2 * r * s1 * s2, 0.90464, 1e-5)  # average beats LightGBM only if this < 3 s1^2
close(3 * s1**2, 0.56611, 1e-5)
print(f"LightGBM + ridge average: RMSE {np.sqrt(mse2):.4f} (LightGBM alone {s1})")

# three boosters, equal weights
s, C = sd_corr("boosters")
assert list(np.round(s, 4)) == [0.4446, 0.4344, 0.4357], s
assert [round(C[0, 1], 4), round(C[0, 2], 4), round(C[1, 2], 4)] == [0.9331, 0.9422, 0.9241]
sr = np.array([0.4446, 0.4344, 0.4357])
Cr = np.array([[1, 0.9331, 0.9422], [0.9331, 1, 0.9241], [0.9422, 0.9241, 1]])
mse3 = float(np.sum(np.outer(sr, sr) * Cr) / 9)
close(float(np.sum(sr**2)), 0.57621, 1e-5)
close(mse3, 0.18350, 1e-5)
for (i, j), v in {(0, 1): 0.18021, (0, 2): 0.18252, (1, 2): 0.17490}.items():
    close(Cr[i, j] * sr[i] * sr[j], v, 1e-5)
close(np.sqrt(mse3), 0.4284, 5e-5)
close(np.sqrt(mse3), G["boosters"]["identity_seed0"]["rmse_avg"], 5e-5)
print(f"three boosters average: RMSE {np.sqrt(mse3):.4f} (best member {sr.min()})")
print("all hand calculations check out")
