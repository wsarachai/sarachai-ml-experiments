"""Search for the label-flip procedure behind the post's 15%-noise table.

The post reports, for GradientBoostingClassifier(n_estimators=1000,
max_depth=3, random_state=42) on the shared breast-cancer split, six test
errors that do not depend on the checkpoint grid:

    nu=0.1  @10 = 0.0994   @1000 = 0.1345
    nu=0.01 @500 = 0.0760  @1000 = 0.0877
    nu=0.5  @1 = 0.0819    @1000 = 0.1404

This script tries 972 flip procedures (3 RNGs x 27 seeds x 4 selection methods x
3 scopes; flipping all rows is scored against both clean and noisy test labels,
1,620 evaluations). Each procedure is first scored on the two nu=0.1 values;
only those matching at least one are fitted for nu=0.01 and nu=0.5.

Run from the repository root (about 30 minutes on 18 cores):
    .venv/Scripts/python.exe gradient-boosting/noise_search.py
Output: gradient-boosting/noise_search.txt
"""

from __future__ import annotations

import itertools
import random
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
RATE = 0.15
TARGET = {"0.1@10": 0.0994, "0.1@1000": 0.1345, "0.01@500": 0.0760,
          "0.01@1000": 0.0877, "0.5@1": 0.0819, "0.5@1000": 0.1404}
SEEDS = list(range(0, 21)) + [42, 123, 1234, 2024, 2025, 2026]
KINDS = ["RS", "gen", "py"]  # np RandomState, np default_rng, python random.Random
METHODS = ["choice_round", "choice_floor", "mask", "perm_round"]
SCOPES = ["train_only", "all_strat_clean", "all_strat_noisy"]


def flip_idx(n, kind, seed, method):
    k_round, k_floor = int(round(RATE * n)), int(RATE * n)
    if kind == "py":
        r = random.Random(seed)
        if method == "choice_round":
            return np.array(r.sample(range(n), k_round))
        if method == "choice_floor":
            return np.array(r.sample(range(n), k_floor))
        if method == "mask":
            return np.where(np.array([r.random() for _ in range(n)]) < RATE)[0]
        p = list(range(n))
        r.shuffle(p)
        return np.array(p[:k_round])
    r = np.random.RandomState(seed) if kind == "RS" else np.random.default_rng(seed)
    if method == "choice_round":
        return r.choice(n, k_round, replace=False)
    if method == "choice_floor":
        return r.choice(n, k_floor, replace=False)
    if method == "mask":
        return np.where((r.rand(n) if kind == "RS" else r.random(n)) < RATE)[0]
    return r.permutation(n)[:k_round]


def staged(lr, Xtr, ytr, Xte, yte):
    m = GradientBoostingClassifier(n_estimators=1000, learning_rate=lr,
                                   random_state=42).fit(Xtr, ytr)
    return np.array([np.mean(p != yte) for p in m.staged_predict(Xte)])


def run(kind, seed, method, scope):
    d = load_breast_cancer()
    X, y = d.data, d.target
    if scope == "train_only":
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        ytr = ytr.copy()
        i = flip_idx(len(ytr), kind, seed, method)
        ytr[i] = 1 - ytr[i]
        tests = {"clean": yte}
    else:
        yn = y.copy()
        i = flip_idx(len(y), kind, seed, method)
        yn[i] = 1 - yn[i]
        strat = yn if scope == "all_strat_noisy" else y
        Xtr, Xte, ytr, yte_n, _, yte_c = train_test_split(
            X, yn, y, test_size=0.3, random_state=42, stratify=strat)
        tests = {"noisy": yte_n, "clean": yte_c}
    out = []
    for tname, yte in tests.items():
        e = staged(0.1, Xtr, ytr, Xte, yte)
        sig = {"0.1@10": round(float(e[9]), 4), "0.1@1000": round(float(e[999]), 4)}
        if any(sig[k] == TARGET[k] for k in sig):
            e1, e5 = staged(0.01, Xtr, ytr, Xte, yte), staged(0.5, Xtr, ytr, Xte, yte)
            sig.update({"0.01@500": round(float(e1[499]), 4), "0.01@1000": round(float(e1[999]), 4),
                        "0.5@1": round(float(e5[0]), 4), "0.5@1000": round(float(e5[999]), 4)})
        score = sum(sig.get(k) == v for k, v in TARGET.items())
        out.append(dict(kind=kind, seed=seed, method=method, scope=scope, test=tname,
                        score=int(score), sig=sig))
    return out


def summarize(flat) -> str:
    lines = [f"procedures: {len(KINDS) * len(SEEDS) * len(METHODS) * len(SCOPES)}; "
             f"evaluations: {len(flat)}", f"target: {TARGET}", ""]
    counts = {}
    for r in flat:
        counts[r["score"]] = counts.get(r["score"], 0) + 1
    lines.append("evaluations by number of matched values (of 6): "
                 + ", ".join(f"{k}: {counts[k]}" for k in sorted(counts, reverse=True)))
    lines.append("\ntop 10:")
    for r in sorted(flat, key=lambda r: -r["score"])[:10]:
        lines.append(f"  {r['score']}/6  {r['kind']:<3} seed={r['seed']:<5} {r['method']:<12} "
                     f"{r['scope']:<15} test={r['test']:<5} {r['sig']}")
    tr = [r for r in flat if r["scope"] == "train_only"]
    lines.append("\ndistribution over the train-only procedures:")
    for key in ("0.1@10", "0.1@1000"):
        v = np.array([r["sig"][key] for r in tr])
        lines.append(f"  {key}: mean {v.mean():.4f} sd {v.std():.4f} min {v.min():.4f} "
                     f"max {v.max():.4f}")
    return "\n".join(lines) + "\n"


def main() -> None:
    jobs = list(itertools.product(KINDS, SEEDS, METHODS, SCOPES))
    res = Parallel(n_jobs=-1)(delayed(run)(*j) for j in jobs)
    text = summarize([r for rr in res for r in rr])
    print(text)
    (HERE / "noise_search.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
