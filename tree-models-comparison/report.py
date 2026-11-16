"""Build results.txt and results.json from raw/<block>.json (written by experiments.py).

Statistics
  mean +- SD over seeds, SD with ddof=1 (sample SD).
  "Is A different from B?" uses the per-seed PAIRED difference A - B on the
  same split: mean d, SD of d (ddof=1), standard error SE = SD / sqrt(n) and
  the 95% interval mean d +- t(0.975, n-1) * SE.  If the interval excludes 0
  the difference is marked "beyond seed noise", otherwise "within noise".
  No correction for the many comparisons is made; read single marginal
  calls with care.

Run:    .venv/Scripts/python.exe tree-models-comparison/report.py
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from scipy import stats

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
MODELS = ["RandomForest", "HistGB", "XGBoost", "LightGBM", "CatBoost"]
HIGHER = {"roc_auc": True, "avg_precision": True, "accuracy": True, "r2": True,
          "log_loss": False, "rmse": False, "mae": False}
METRICS = {"binary": ["roc_auc", "avg_precision", "log_loss", "accuracy"],
           "multiclass": ["accuracy", "log_loss"],
           "regression": ["rmse", "mae", "r2"]}
TITLES = {"adult": "Adult (binary, 8 categorical columns)", "california": "California housing (regression)",
          "covertype": "Covertype (7 classes, 10,000 training rows per seed)",
          "breast": "Breast cancer (binary, small data)"}

lines: list[str] = []


def log(s: str = "") -> None:
    print(s)
    lines.append(s)


def sd(x):
    return statistics.stdev(x) if len(x) > 1 else 0.0


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    m, s = statistics.mean(d), sd(d)
    se = s / math.sqrt(n)
    half = stats.t.ppf(0.975, n - 1) * se
    return {"mean_diff": m, "sd_diff": s, "se": se, "ci_half": half,
            "beyond_noise": bool(abs(m) > half and half >= 0 and m != 0)}


def summarise(block):
    runs = block["runs"]
    task = block["task"]
    out = {}
    for setting in ["default", "matched"]:
        out[setting] = {}
        per = {m: sorted([r for r in runs if r["model"] == m and r["setting"] == setting], key=lambda r: r["seed"])
               for m in MODELS}
        for m, rs in per.items():
            s = {"seeds": [r["seed"] for r in rs]}
            for k in METRICS[task]:
                v = [r[k] for r in rs]
                s[k] = {"mean": statistics.mean(v), "sd": sd(v), "per_seed": v}
            for k in ["fit_s", "predict_10k_s", "size_bytes", "trees"]:
                v = [r[k] for r in rs]
                s[k] = {"median": statistics.median(v), "min": min(v), "max": max(v), "per_seed": v}
            out[setting][m] = s
        # paired comparison of every model against the best mean, per metric
        comp = {}
        for k in METRICS[task]:
            hi = HIGHER[k]
            best = (max if hi else min)(MODELS, key=lambda m: out[setting][m][k]["mean"])
            comp[k] = {"best": best, "vs_best": {}}
            for m in MODELS:
                if m == best:
                    continue
                comp[k]["vs_best"][m] = paired(out[setting][m][k]["per_seed"], out[setting][best][k]["per_seed"])
        out[setting]["comparisons"] = comp
    # default vs matched within each model, first metric
    k0 = METRICS[task][0]
    out["matched_minus_default"] = {
        m: paired(out["matched"][m][k0]["per_seed"], out["default"][m][k0]["per_seed"]) for m in MODELS}
    out["matched_minus_default_metric"] = k0
    return out


def fmt(k, v):
    return f"{v:.4f}"


def print_block(name, block, summ):
    task = block["task"]
    log("=" * 100)
    log(f"{TITLES[name]}: {block['rows']} rows, {block['cols']} columns; split train/valid/test = "
        f"{block['split_sizes']}; seeds {block['seeds']}; matched max rounds {block['max_rounds']}")
    if "positive_rate" in block:
        log(f"positive rate {block['positive_rate']:.4f}")
    log(f"block wall time {block['block_seconds']:.0f} s")
    log("=" * 100)
    for setting in ["default", "matched"]:
        log(f"\n  setting = {setting}   (metric: mean +- SD over {len(block['seeds'])} seeds; "
            f"fit / predict-10k / size / trees: median over seeds)")
        mets = METRICS[task]
        head = "  " + f"{'model':<13}" + "".join(f"{k:>20}" for k in mets) + \
            f"{'fit s':>9}{'pred ms':>9}{'size MB':>9}{'trees':>7}"
        log(head)
        for m in MODELS:
            s = summ[setting][m]
            row = "  " + f"{m:<13}" + "".join(f"{s[k]['mean']:>11.4f} +-{s[k]['sd']:.4f}" for k in mets)
            row += f"{s['fit_s']['median']:>9.2f}{s['predict_10k_s']['median'] * 1000:>9.1f}" \
                   f"{s['size_bytes']['median'] / 1e6:>9.2f}{s['trees']['median']:>7.0f}"
            log(row)
        log("  per-seed fit seconds: " + "; ".join(
            f"{m} {[round(x, 2) for x in summ[setting][m]['fit_s']['per_seed']]}" for m in MODELS))
        log("  paired difference to the best model (model - best), 95% t interval over seeds:")
        for k in mets:
            c = summ[setting]["comparisons"][k]
            parts = []
            for m, p in c["vs_best"].items():
                parts.append(f"{m} {p['mean_diff']:+.4f} (+-{p['ci_half']:.4f}) "
                             f"{'BEYOND noise' if p['beyond_noise'] else 'within noise'}")
            log(f"    {k:<14} best = {c['best']}: " + "; ".join(parts))
        for k in mets:
            log(f"    per-seed {k}: " + "; ".join(
                f"{m} {[round(x, 4) for x in summ[setting][m][k]['per_seed']]}" for m in MODELS))
    k0 = summ["matched_minus_default_metric"]
    log(f"\n  matched - default, {k0} (paired over seeds):")
    for m, p in summ["matched_minus_default"].items():
        log(f"    {m:<13} {p['mean_diff']:+.4f} (+-{p['ci_half']:.4f}) "
            f"{'BEYOND noise' if p['beyond_noise'] else 'within noise'}")
    log()


def print_scaling(block):
    log("=" * 100)
    log(f"SCALING  Covertype 7 classes, seed 0, fixed test {block['test_rows']} rows, every booster "
        f"{block['fixed_rounds']} rounds, learning rate 0.1, ~32 leaves, no early stopping; RF 100 trees")
    log(f"fit time = median of 3 fits at 10,000 rows, a single fit above; block wall time "
        f"{block['block_seconds']:.0f} s")
    log("=" * 100)
    log(f"  {'rows':>8} {'model':<13}{'accuracy':>10}{'log loss':>10}{'fit s':>9}{'pred ms':>9}{'size MB':>9}  fit runs")
    for r in block["runs"]:
        log(f"  {r['n_train']:>8} {r['model']:<13}{r['accuracy']:>10.4f}{r['log_loss']:>10.4f}{r['fit_s']:>9.2f}"
            f"{r['predict_10k_s'] * 1000:>9.1f}{r['size_bytes'] / 1e6:>9.2f}  {[round(t, 2) for t in r['fit_runs']]}")
    log()


def main():
    blocks = {p.stem: json.loads(p.read_text()) for p in sorted(RAW.glob("*.json"))}
    env = next(iter(blocks.values()))["environment"]
    log(f"python {env['python']}, scikit-learn {env['sklearn']}, xgboost {env['xgboost']}, "
        f"lightgbm {env['lightgbm']}, catboost {env['catboost']}, numpy {env['numpy']}, pandas {env['pandas']}")
    log(f"{env['logical_cpus']} logical CPUs, {env['threads_per_model']} threads per model. Other jobs shared the "
        f"machine; timings are machine-dependent and indicative only.")
    log("SD = sample SD (ddof=1). 'BEYOND noise' = 95% paired t interval over seeds excludes 0.\n")
    summary = {"environment": env}
    for name in ["adult", "california", "covertype", "breast"]:
        if name in blocks:
            summ = summarise(blocks[name])
            print_block(name, blocks[name], summ)
            summary[name] = {"meta": {k: v for k, v in blocks[name].items() if k != "runs"}, "summary": summ}
    if "scaling" in blocks:
        print_scaling(blocks["scaling"])
        summary["scaling"] = blocks["scaling"]
    (HERE / "results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (HERE / "results.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
