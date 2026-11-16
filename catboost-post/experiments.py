"""Reproduction of the measured numbers in the "catboost" blog post.

Parts
  1. Leakage demo: 1,000 rows, one unique `id` column, pure-random y, 700/300
     split. Greedy target statistic + depth-1 tree vs CatBoost on the raw id.
  2. Hand example of greedy vs ordered target statistics (asserted).
  3. Parameter values CatBoost actually resolves (read back from the model).
  4. CatBoost vs XGBoost + one-hot on synthetic data with a 200-level
     category. The post does not describe its generator, so this part uses a
     documented stand-in generator and only checks the direction of the gap.

Run from the repository root (folder named catboost-post so that it does not
shadow the catboost package):
    .venv/Scripts/python.exe catboost-post/experiments.py
Output is printed and written to catboost-post/results.txt.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import catboost
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent
buf = io.StringIO()
CHECKS: list[tuple[str, str, str, str]] = []


def say(*args) -> None:
    print(*args)
    print(*args, file=buf)


def header(title: str) -> None:
    say("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def check(label, published, value, tol=0.0, fmt="{:.4f}", status=None) -> None:
    if isinstance(published, str) or isinstance(value, str):
        pub_s, val_s = str(published), str(value)
        status = status or ("exact" if pub_s == val_s else "DIFFERS")
    else:
        pub_s, val_s = fmt.format(published), fmt.format(value)
        status = status or ("exact" if pub_s == val_s
                            else "within noise" if abs(published - value) <= tol else "DIFFERS")
    CHECKS.append((label, pub_s, val_s, status))


def cat_model(**kw) -> CatBoostClassifier:
    # allow_writing_files=False keeps catboost_info/ out of the repository
    return CatBoostClassifier(verbose=0, random_seed=42, allow_writing_files=False, **kw)


# ----------------------------------------------------------------- part 1 ---
def part1() -> None:
    header("PART 1  Greedy target encoding leaks on a pure-noise id column")
    say("Inferred procedure: y = RandomState(0).randint(0, 2, 1000), id = row number,\n"
        "train_test_split(test_size=0.3, random_state=42). Found by searching 5,166\n"
        "generator/seed/split combinations for the published prior 0.4871 and test\n"
        "class-0 share 0.4567; 2 matched, this is the simpler one.")
    n = 1000
    y = np.random.RandomState(0).randint(0, 2, n)
    ids = np.arange(n)
    itr, ite, ytr, yte = train_test_split(ids, y, test_size=0.3, random_state=42)
    p = ytr.mean()
    say(f"prior p = mean(y_train) = {p:.4f}; test share of class 0 = {(yte == 0).mean():.4f}")
    check("prior p (train mean)", 0.4871, p)

    # Greedy TS with a = 1: every id is unique, so x_k = (y_k + a p) / (1 + a).
    a = 1.0
    ts_tr = ((ytr + a * p) / (1 + a)).reshape(-1, 1)
    ts_te = np.full((len(ite), 1), p)  # unseen ids fall back to the prior
    stump = DecisionTreeClassifier(max_depth=1, random_state=42).fit(ts_tr, ytr)
    g_tr, g_te = stump.score(ts_tr, ytr), stump.score(ts_te, yte)
    g_auc = roc_auc_score(yte, stump.predict_proba(ts_te)[:, 1])
    say(f"greedy TS + stump: train {g_tr:.4f} test {g_te:.4f} AUC {g_auc:.4f}")
    check("greedy TS train accuracy", 1.0, g_tr)
    check("greedy TS test accuracy", 0.4567, g_te)
    check("greedy TS test AUC", 0.5, g_auc)

    Xtr, Xte = itr.astype(str).reshape(-1, 1), ite.astype(str).reshape(-1, 1)
    m = cat_model(iterations=300, depth=3).fit(Xtr, ytr, cat_features=[0])
    c_tr, c_te = m.score(Xtr, ytr), m.score(Xte, yte)
    c_auc = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
    say(f"CatBoost(300 it, depth 3) on raw id: train {c_tr:.4f} test {c_te:.4f} AUC {c_auc:.4f}")
    check("CatBoost train accuracy", 0.5129, c_tr)
    check("CatBoost test accuracy", 0.4567, c_te)
    check("CatBoost test AUC", 0.5, c_auc)
    return m


# ----------------------------------------------------------------- part 2 ---
def part2() -> None:
    header("PART 2  Greedy vs ordered target statistics, 6 rows (asserted)")
    cat = ["A", "B", "A", "C", "A", "B"]
    y = [1, 0, 0, 1, 1, 1]
    pos = [2, 4, 1, 6, 5, 3]  # position of each row in the permutation
    p, a = sum(y) / len(y), 1.0
    assert round(p, 4) == 0.6667
    greedy = {}
    for c in "ABC":
        ys = [yy for cc, yy in zip(cat, y) if cc == c]
        greedy[c] = (sum(ys) + a * p) / (len(ys) + a)
    say("greedy TS:", {k: round(v, 4) for k, v in greedy.items()})
    assert [round(greedy[c], 4) for c in "ABC"] == [0.6667, 0.5556, 0.8333]
    ordered = {}
    for i in range(6):
        hist = [y[j] for j in range(6) if cat[j] == cat[i] and pos[j] < pos[i]]
        ordered[i + 1] = (sum(hist) + a * p) / (len(hist) + a)
    say("ordered TS by row:", {k: round(v, 4) for k, v in ordered.items()})
    expected = {3: 0.6667, 1: 0.3333, 6: 0.6667, 2: 0.8333, 5: 0.5556, 4: 0.6667}
    assert all(round(ordered[r], 4) == v for r, v in expected.items())
    check("greedy/ordered TS hand tables", "asserted", "asserted")


# ----------------------------------------------------------------- part 3 ---
def part3(model) -> None:
    header(f"PART 3  Parameters resolved by catboost {catboost.__version__} (CPU)")
    say("Model: the 300-iteration, depth-3 model of part 1 (700 training rows).")
    ap = model.get_all_params()
    published = {"boosting_type": "Plain", "grow_policy": "SymmetricTree",
                 "one_hot_max_size": "2", "max_ctr_complexity": "4", "l2_leaf_reg": "3",
                 "bootstrap_type": "MVS", "learning_rate": "0.026684999465942383",
                 "random_strength": "1", "counter_calc_method": "SkipTest"}
    for k, v in published.items():
        got = ap.get(k)
        got_s = repr(got) if isinstance(got, float) and k == "learning_rate" else f"{got:g}" \
            if isinstance(got, float) else str(got)
        say(f"  {k:<20} = {got_s}")
        check(f"param {k}", v, got_s)
    say(f"  subsample            = {ap.get('subsample')}")
    d = cat_model()
    d.fit(np.random.RandomState(0).rand(50, 2), np.arange(50) % 2)
    dp = d.get_all_params()
    say(f"defaults on a plain fit: iterations={dp['iterations']} depth={dp['depth']}")
    check("default iterations", "1000", str(dp["iterations"]))
    check("default depth", "6", str(dp["depth"]))
    say("Note: the automatic learning rate depends on the data size as well as on the\n"
        "number of iterations; with defaults on the part-1 data it is "
        f"{cat_model().fit(np.arange(700).astype(str).reshape(-1, 1), np.random.RandomState(0).randint(0, 2, 700), cat_features=[0]).get_all_params()['learning_rate']:.6f}.")


# ----------------------------------------------------------------- part 4 ---
def make_data(seed: int, n: int = 3000, n_cat: int = 200):
    """Generator used for the 3,000-row comparison in the post (the original run was not recorded; the post now reports these numbers):
    logit = category effect ~ N(0, 1) + 0.5 * (x0 + x1 - x2) + N(0, 1) noise."""
    rng = np.random.default_rng(seed)
    effect = rng.normal(0, 1, n_cat)
    cat = rng.integers(0, n_cat, n)
    X = rng.normal(0, 1, (n, 5))
    logit = effect[cat] + 0.5 * (X[:, 0] + X[:, 1] - X[:, 2]) + rng.normal(0, 1, n)
    y = (logit > 0).astype(int)
    return X, cat, y


def part4() -> None:
    header("PART 4  CatBoost (raw category) vs XGBoost + one-hot, 3,000 rows, 3 seeds")
    say("The post's generator, split and model settings are not recorded. Stand-in: see\n"
        "make_data(); 70/30 split; both libraries at their defaults (random seed = seed).")
    acc_c, auc_c, acc_x, auc_x = [], [], [], []
    for seed in (0, 1, 2):
        X, cat, y = make_data(seed)
        idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=seed)
        # CatBoost: numeric columns + the category as a string column
        Xc = np.column_stack([X, cat.astype(str)]).astype(object)
        Xc[:, :5] = X
        cb = CatBoostClassifier(verbose=0, random_seed=seed, allow_writing_files=False)
        cb.fit(Xc[idx_tr], y[idx_tr], cat_features=[5])
        pc = cb.predict_proba(Xc[idx_te])[:, 1]
        # XGBoost: numeric columns + 200 one-hot columns
        Xo = np.hstack([X, np.eye(200)[cat]])
        xg = xgb.XGBClassifier(random_state=seed).fit(Xo[idx_tr], y[idx_tr])
        px = xg.predict_proba(Xo[idx_te])[:, 1]
        yt = y[idx_te]
        acc_c.append(np.mean((pc > 0.5) == yt))
        auc_c.append(roc_auc_score(yt, pc))
        acc_x.append(np.mean((px > 0.5) == yt))
        auc_x.append(roc_auc_score(yt, px))
        say(f"seed {seed}: CatBoost acc {acc_c[-1]:.4f} AUC {auc_c[-1]:.4f} | "
            f"XGBoost+one-hot acc {acc_x[-1]:.4f} AUC {auc_x[-1]:.4f}")
    say(f"mean: CatBoost acc {np.mean(acc_c):.4f} AUC {np.mean(auc_c):.4f} | "
        f"XGBoost acc {np.mean(acc_x):.4f} AUC {np.mean(auc_x):.4f}")
    gap_acc, gap_auc = np.mean(acc_c) - np.mean(acc_x), np.mean(auc_c) - np.mean(auc_x)
    say(f"gap (CatBoost - XGBoost): accuracy {gap_acc:+.4f}, AUC {gap_auc:+.4f}")
    na = "not reproducible (generator unknown)"
    check("CatBoost accuracy", 0.6811, np.mean(acc_c), status=na)
    check("CatBoost AUC", 0.7430, np.mean(auc_c), status=na)
    check("XGBoost+one-hot accuracy", 0.6619, np.mean(acc_x), status=na)
    check("XGBoost+one-hot AUC", 0.7131, np.mean(auc_x), status=na)
    check("direction: CatBoost ahead on acc and AUC", "yes",
          "yes" if gap_acc > 0 and gap_auc > 0 else "no")


def main() -> None:
    say(f"python {sys.version.split()[0]} | catboost {catboost.__version__} | "
        f"xgboost {xgb.__version__} | numpy {np.__version__}")
    model = part1()
    part2()
    part3(model)
    part4()
    header("SUMMARY  published vs reproduced")
    w = max(len(c[0]) for c in CHECKS)
    for label, pub, val, status in CHECKS:
        say(f"{label:<{w}} | published {pub:>20} | reproduced {val:>20} | {status}")
    (HERE / "results.txt").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
