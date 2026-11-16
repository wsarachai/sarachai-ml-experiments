"""Why does HistGradientBoostingClassifier score so badly on Covertype 10k?

experiments.py found HistGB default at mean accuracy ~0.74 with log loss
~2.6 on the 7-class Covertype sample, far below the other boosters.  This
script re-creates the seed-0 split of the covertype block and looks at:
  * training and test log loss per iteration (does it diverge?)
  * the largest absolute raw score (does some class's score blow up?)
  * the effect of l2_regularization and of a lower learning rate
  * LightGBM with the same tree size for reference
It changes nothing in the comparison; it only explains a result.

Run:    .venv/Scripts/python.exe tree-models-comparison/diagnose_histgb.py
Output: tree-models-comparison/diagnose_histgb.txt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.datasets import fetch_covtype
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "data_cache"
lines: list[str] = []


def log(s=""):
    print(s, flush=True)
    lines.append(s)


d = fetch_covtype(data_home=str(CACHE), as_frame=True)
X, y = d.data.astype(float), d.target.to_numpy() - 1
idx = np.random.default_rng(0).permutation(len(y))       # same as covtype_splitter, seed 0
te, tr = idx[:20_000], idx[25_000:35_000]
Xtr, ytr, Xte, yte = X.iloc[tr], y[tr], X.iloc[te], y[te]
log(f"seed-0 split: train {len(tr)}, test {len(te)}")
log(f"training class counts: {np.bincount(ytr, minlength=7).tolist()}")

with threadpool_limits(4):
    def run(label, model):
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)
        msg = f"  {label:<44} test acc {accuracy_score(yte, p.argmax(1)):.4f}  test log loss {log_loss(yte, p, labels=list(range(7))):.4f}"
        if hasattr(model, "staged_predict_proba"):
            tr_ll, te_ll, raw = [], [], []
            for k, (pt, pe) in enumerate(zip(model.staged_predict_proba(Xtr), model.staged_predict_proba(Xte))):
                if k in (9, 24, 49, 99):
                    tr_ll.append(round(log_loss(ytr, pt, labels=list(range(7))), 4))
                    te_ll.append(round(log_loss(yte, pe, labels=list(range(7))), 4))
            raw = np.abs(model._raw_predict(Xte)).max()
            msg += f"\n      log loss at iter 10/25/50/100: train {tr_ll} test {te_ll}; max |raw score| {raw:.1f}"
        log(msg)

    log("\nHistGB, 100 iterations, no early stopping (the default at 10,000 rows)")
    run("default (lr 0.1, 31 leaves, l2 0)", HistGradientBoostingClassifier(random_state=0))
    run("l2_regularization=1", HistGradientBoostingClassifier(random_state=0, l2_regularization=1.0))
    run("learning_rate=0.05", HistGradientBoostingClassifier(random_state=0, learning_rate=0.05))
    run("min_samples_leaf=50", HistGradientBoostingClassifier(random_state=0, min_samples_leaf=50))
    log("\nLightGBM reference, same size")
    run("LightGBM default (lr 0.1, 31 leaves, lambda 0)",
        LGBMClassifier(random_state=0, n_jobs=4, verbose=-1))

(HERE / "diagnose_histgb.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
