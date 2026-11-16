"""Small behaviour checks quoted in the "lightgbm" blog post.

1. Old-style boosting="goss" vs data_sample_strategy="goss" (what 4.x resolves).
2. A main parameter name and one of its aliases passed together: which one wins.

Run from the repository root (appends to results.txt):
    .venv/Scripts/python.exe lightgbm-post/param_checks.py >> lightgbm-post/results.txt 2>&1
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np


def resolved(booster: lgb.Booster, key: str) -> str:
    text = booster.model_to_string()
    i = text.find(f"[{key}: ")
    return text[i:text.find("]", i) + 1]


def main() -> None:
    print("\n" + "=" * 78)
    print(f"PART D  Parameter behaviour checks, lightgbm {lgb.__version__}")
    print("=" * 78, flush=True)
    rng = np.random.RandomState(0)
    X = rng.rand(500, 3)
    y = (X[:, 0] > 0.5).astype(int)

    for extra in ({"boosting": "goss"}, {"data_sample_strategy": "goss"}):
        print(f"\n  params {extra}  (LightGBM warnings, if any, are printed by the library above/below)",
              flush=True)
        b = lgb.train({"objective": "binary", "verbosity": 0, **extra}, lgb.Dataset(X, y), num_boost_round=2)
        print(f"  resolved: {resolved(b, 'boosting')} {resolved(b, 'data_sample_strategy')}", flush=True)

    X8 = np.arange(1, 9, dtype=float).reshape(-1, 1)
    y8 = np.array([20, 10, 20, 18, 38, 34, 16, 28], dtype=float)
    params = {"objective": "regression", "min_data_in_leaf": 1, "min_child_samples": 3,
              "min_data_in_bin": 1, "feature_pre_filter": False, "verbosity": 0}
    b = lgb.train(params, lgb.Dataset(X8, y8, params={"min_data_in_bin": 1, "feature_pre_filter": False}),
                  num_boost_round=1)
    print(f"\n  min_data_in_leaf=1 together with alias min_child_samples=3 -> {resolved(b, 'min_data_in_leaf')}",
          flush=True)


if __name__ == "__main__":
    main()
