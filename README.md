# sarachai-ml-experiments

Scripts that produce (or re-check) the measured numbers in the machine-learning
articles on [sarachai.com](https://sarachai.com/blog). Each article has its own
folder with an `experiments.py` and the captured output `results.txt`; some
folders also hold figure scripts and the rendered figures.

Some articles were written before this repository existed. For those, the
script infers the procedure from the article text, recomputes every number, and
ends `results.txt` with a summary table that marks each published value as
`exact`, `within noise` or `DIFFERS` (with the reproduced value). Comments in
each script say how a procedure was inferred and how many alternatives were
searched.

## Setup

Python 3.13 and [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.13 .venv
uv pip install -r requirements.txt
```

Run a script from the repository root, for example:

```bash
.venv/Scripts/python.exe random-forest/experiments.py    # Windows
.venv/bin/python random-forest/experiments.py            # Linux / macOS
```

Library versions are pinned in `requirements.txt` (scikit-learn 1.9.1,
XGBoost 3.4.1, LightGBM 4.7.0, CatBoost 1.2.10). Results with other versions
may differ in the last digits. Timings reported by some scripts depend on the
machine and the number of threads; accuracy numbers do not.

Large datasets downloaded by scikit-learn (e.g. Covertype) are cached in
`data_cache/`, which is not committed.

Folders whose names would shadow an installed package carry a `-post`
suffix (`xgboost-post/`, `catboost-post/`, `lightgbm-post/`).

## Articles

| Article slug | Folder | Article |
|---|---|---|
| `machine-learning` | [`machine-learning/`](machine-learning/) | https://sarachai.com/blog/machine-learning |
| `decision-tree` | [`decision-tree/`](decision-tree/) | https://sarachai.com/blog/decision-tree |
| `random-forest` | [`random-forest/`](random-forest/) | https://sarachai.com/blog/random-forest |
| `out-of-bag` | [`out-of-bag/`](out-of-bag/) | https://sarachai.com/blog/out-of-bag |
| `gradient-boosting` | [`gradient-boosting/`](gradient-boosting/) | https://sarachai.com/blog/gradient-boosting |
| `adaboost` | [`adaboost/`](adaboost/) | https://sarachai.com/blog/adaboost |
| `histogram-gradient-boosting` | [`histogram-gradient-boosting/`](histogram-gradient-boosting/) | https://sarachai.com/blog/histogram-gradient-boosting |
| `xgboost` | [`xgboost-post/`](xgboost-post/) | https://sarachai.com/blog/xgboost |
| `lightgbm` | [`lightgbm-post/`](lightgbm-post/) | https://sarachai.com/blog/lightgbm |
| `catboost` | [`catboost-post/`](catboost-post/) | https://sarachai.com/blog/catboost |
| `tree-models-for-time-series` | [`tree-models-for-time-series/`](tree-models-for-time-series/) | https://sarachai.com/blog/tree-models-for-time-series |
| `stacking-and-voting` | [`stacking-and-voting/`](stacking-and-voting/) | https://sarachai.com/blog/stacking-and-voting |
| `tree-models-comparison` | [`tree-models-comparison/`](tree-models-comparison/) | https://sarachai.com/blog/tree-models-comparison |
| `model-interpretation` | [`model-interpretation/`](model-interpretation/) | https://sarachai.com/blog/model-interpretation |
| `hyperparameter-tuning` | [`hyperparameter-tuning/`](hyperparameter-tuning/) | https://sarachai.com/blog/hyperparameter-tuning |
| `data-preparation-for-trees` | [`data-preparation-for-trees/`](data-preparation-for-trees/) | https://sarachai.com/blog/data-preparation-for-trees |
| `cross-validation` | [`cross-validation/`](cross-validation/) | https://sarachai.com/blog/cross-validation |
| `regression-metrics` | [`regression-metrics/`](regression-metrics/) | https://sarachai.com/blog/regression-metrics |
| `classification-metrics` | [`classification-metrics/`](classification-metrics/) | https://sarachai.com/blog/classification-metrics |

## Conventions

- The breast-cancer experiments share one split:
  `train_test_split(test_size=0.3, random_state=42, stratify=y)` (398 / 171
  rows) and model `random_state=42`; multi-seed tables use seeds `0..9` for
  both the split and the model unless a script says otherwise.
- Standard deviations over seeds are population SDs (`ddof=0`) unless stated.
- Scripts never write outside their own folder.
