"""Follow-up checks for results that experiments.py raised (fast, < 1 minute).

1. Why RandomForest / DecisionTree / LightGBM predictions moved under scaling:
   - is RandomForest deterministic on the same data?
   - do distinct values collapse after the float32 cast scikit-learn trees use?
   - LightGBM: does a pure shift (moves 0) change the model while a pure
     positive multiplication (keeps 0) does not?
2. HistGradientBoosting native categorical on the 4-category toy: with 2 rows
   per category no split was made; repeat with 20 rows per category.
3. TargetEncoder made deterministic with an explicit cv splitter (the 1.9 way):
   AUC on Adult for several cv seeds.

Run:    .venv/Scripts/python.exe data-preparation-for-trees/supplement.py
Output: data-preparation-for-trees/results_supplement.txt
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, TargetEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "data_cache" / "openml"
lines: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    lines.append(msg)


adult = fetch_openml("adult", version=2, as_frame=True, data_home=str(CACHE))
X_all = adult.data.copy()
y_all = (adult.target == ">50K").astype(int).to_numpy()
NUM = [c for c in X_all.columns if X_all[c].dtype.kind in "if"]
CAT = [c for c in X_all.columns if c not in NUM]
X_all[NUM] = X_all[NUM].astype(float)
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.25, stratify=y_all, random_state=42)
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-2, encoded_missing_value=-1)
C_tr, C_te = enc.fit_transform(X_tr[CAT]), enc.transform(X_te[CAT])
N_tr, N_te = X_tr[NUM].to_numpy(), X_te[NUM].to_numpy()
sc = StandardScaler().fit(N_tr)

log("1a. distinct training values per numeric column: float64 vs after the float32 cast")
for name, A in (("raw", N_tr), ("standardised", sc.transform(N_tr)), ("log1p", np.log1p(N_tr))):
    counts = [f"{c}: {len(np.unique(A[:, j]))}/{len(np.unique(A[:, j].astype(np.float32)))}"
              for j, c in enumerate(NUM)]
    log(f"  {name:<13} " + ", ".join(counts))

log("\n1b. RandomForest(random_state=0) fitted twice on identical raw data")
raw_tr, raw_te = np.hstack([N_tr, C_tr]), np.hstack([N_te, C_te])
p1 = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=4).fit(raw_tr, y_tr).predict_proba(raw_te)[:, 1]
p2 = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=4).fit(raw_tr, y_tr).predict_proba(raw_te)[:, 1]
log(f"  max |p1 - p2| = {np.max(np.abs(p1 - p2)):.2e}")
# scale only the integer-valued columns that stay exact in float32 (drop fnlwgt)
keep = [j for j, c in enumerate(NUM) if c != "fnlwgt"]
a_tr = np.hstack([N_tr[:, keep], C_tr])
a_te = np.hstack([N_te[:, keep], C_te])
s2 = StandardScaler().fit(N_tr[:, keep])
b_tr = np.hstack([s2.transform(N_tr[:, keep]), C_tr])
b_te = np.hstack([s2.transform(N_te[:, keep]), C_te])
q1 = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=4).fit(a_tr, y_tr).predict_proba(a_te)[:, 1]
q2 = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=4).fit(b_tr, y_tr).predict_proba(b_te)[:, 1]
log(f"  without fnlwgt, raw vs standardised: max |dp| test = {np.max(np.abs(q1 - q2)):.2e}, "
    f"rows changed = {int(np.sum(np.abs(q1 - q2) > 1e-12))}")
c_tr = np.hstack([N_tr[:, keep] * 2.0, C_tr])
c_te = np.hstack([N_te[:, keep] * 2.0, C_te])
q3 = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=4).fit(c_tr, y_tr).predict_proba(c_te)[:, 1]
log(f"  without fnlwgt, raw vs x*2 (exact in float): max |dp| test = {np.max(np.abs(q1 - q3)):.2e}")
mu, sd = N_tr[:, keep].mean(axis=0), N_tr[:, keep].std(axis=0)
for name, f in (("x - mean (centre only)", lambda A: A - mu), ("x / sd (scale only)", lambda A: A / sd),
                ("x + 1000", lambda A: A + 1000.0)):
    m = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=4).fit(
        np.hstack([f(N_tr[:, keep]), C_tr]), y_tr)
    tr_diff = np.max(np.abs(m.predict_proba(np.hstack([f(N_tr[:, keep]), C_tr]))[:, 1]
                            - RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=4)
                            .fit(a_tr, y_tr).predict_proba(a_tr)[:, 1]))
    q = m.predict_proba(np.hstack([f(N_te[:, keep]), C_te]))[:, 1]
    log(f"  without fnlwgt, raw vs {name:<24} max |dp| test {np.max(np.abs(q1 - q)):.2e}, train {tr_diff:.2e}, "
        f"rows changed {int(np.sum(np.abs(q1 - q) > 1e-12))}")

log("\n1c. LightGBM defaults: positive multiplication vs pure shift of the numeric columns")
base = LGBMClassifier(random_state=0, n_jobs=4, verbose=-1).fit(raw_tr, y_tr).predict_proba(raw_te)[:, 1]
for name, f in (("x * 2", lambda A: A * 2.0), ("x + 1000", lambda A: A + 1000.0), ("x - mean (centre only)",
                lambda A: A - N_tr.mean(axis=0))):
    m = LGBMClassifier(random_state=0, n_jobs=4, verbose=-1).fit(np.hstack([f(N_tr), C_tr]), y_tr)
    p = m.predict_proba(np.hstack([f(N_te), C_te]))[:, 1]
    log(f"  {name:<24} max |dp| test {np.max(np.abs(p - base)):.2e}, rows changed "
        f"{int(np.sum(np.abs(p - base) > 1e-12))}, AUC {roc_auc_score(y_te, p):.4f} (raw {roc_auc_score(y_te, base):.4f})")

log("\n2. HistGB native categorical, one depth-1 tree, y = 1 for A and C")
for reps in (2, 5, 10, 20):
    cats = np.repeat(list("ABCD"), reps)
    yy = np.array([1 if c in "AC" else 0 for c in cats])
    df = pd.DataFrame({"c": pd.Categorical(cats)})
    m = HistGradientBoostingClassifier(max_iter=1, max_depth=1, learning_rate=1.0, min_samples_leaf=1,
                                       early_stopping=False).fit(df, yy)
    log(f"  {reps:>2} rows per category: train accuracy {(m.predict(df) == yy).mean():.4f}")

log("\n3. TargetEncoder with an explicit cv splitter (deterministic), Adult test AUC")
models = {
    "RandomForest": lambda: RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=4),
    "HistGB": lambda: HistGradientBoostingClassifier(random_state=0),
    "XGBoost": lambda: XGBClassifier(random_state=0, n_jobs=4),
    "LightGBM": lambda: LGBMClassifier(random_state=0, n_jobs=4, verbose=-1),
}
for name, make in models.items():
    aucs = []
    for s in (0, 1, 2):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
        pipe = make_pipeline(ColumnTransformer([("cat", TargetEncoder(target_type="binary", cv=cv), CAT)],
                                               remainder="passthrough"), make())
        aucs.append(roc_auc_score(y_te, pipe.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]))
    log(f"  {name:<13} AUC for cv seeds 0,1,2: {[round(a, 4) for a in aucs]}")

log("\n4. Mean vs median of the columns that get NaN injected (complete training data)")
for c in ["age", "education-num", "hours-per-week", "capital-gain"]:
    v = X_tr[c].astype(float)
    log(f"  {c:<15} mean {v.mean():.3f} (share of real rows equal to it {(v == v.mean()).mean():.4f}); "
        f"median {v.median():g} (share of real rows equal to it {(v == v.median()).mean():.4f})")

(HERE / "results_supplement.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
