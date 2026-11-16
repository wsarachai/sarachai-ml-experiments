"""Turn raw/*.json into results.txt and results.json for the post.

For every dataset and group of base learners: mean +- SD over 5 seeds of each
method, the paired difference against the best single model (chosen by 5-fold
OOF score on dev, never by the test set) with a 95% t interval, residual
correlations of the base models, ensemble weights / meta-learner coefficients,
the in-sample-meta "leak", timings, and the check that the hand-built
combiners equal sklearn's Voting* / Stacking*.

Run:    .venv/Scripts/python.exe stacking-and-voting/report.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
ALL = ["XGBoost", "LightGBM", "CatBoost", "RandomForest", "Linear", "kNN"]
lines: list[str] = []
summary: dict = {}


def out(s=""):
    lines.append(s)


def ms(v):
    return f"{np.mean(v):.4f} +- {np.std(v, ddof=1):.4f}"


def paired(a, b):
    """mean, sd, 95% CI of a - b over seeds."""
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    return d.mean(), d.std(ddof=1), d.mean() - t * se, d.mean() + t * se


for block in ["adult", "california"]:
    R = json.loads((RAW / f"{block}.json").read_text())
    task = R["task"]
    runs = R["runs"]
    metrics = ["roc_auc", "log_loss", "accuracy"] if task == "binary" else ["rmse"]
    lower_better = {"roc_auc": False, "log_loss": True, "accuracy": False, "rmse": True}
    S = summary[block] = {"task": task, "split_sizes": runs[0]["split_sizes"], "groups": {}}
    out("=" * 100)
    out(f"{block}  task={task}  rows={R['rows']}  split train/val/test={runs[0]['split_sizes']}  "
        f"block time {R['block_seconds']:.0f} s")
    out("=" * 100)

    out("\nSingle base models fitted on dev (train+val), test metric mean +- SD over 5 seeds; OOF = 5-fold on dev")
    S["single"] = {}
    for n in ALL:
        row = {k: [r["models"][n]["test"][k] for r in runs] for k in metrics}
        oof = {k: [r["models"][n]["oof"][k] for r in runs] for k in metrics}
        S["single"][n] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "per_seed": v}
                          for k, v in row.items()}
        fit = statistics.median(r["models"][n]["fit_dev_s"] for r in runs)
        pred = statistics.median(r["models"][n]["pred_10k_s"] for r in runs)
        S["single"][n]["fit_s"], S["single"][n]["pred_ms"] = fit, pred * 1000
        out(f"  {n:<12} " + "  ".join(f"{k} {ms(v)}" for k, v in row.items())
            + "  | OOF " + "  ".join(f"{k} {np.mean(v):.4f}" for k, v in oof.items())
            + f"  | fit {fit:.2f}s pred10k {pred * 1000:.0f}ms")

    C = np.mean([np.array(r["residual_corr"]) for r in runs], axis=0)
    S["residual_corr"] = {"models": ALL, "matrix": C.tolist()}
    out("\nResidual correlation on the test rows (y - prediction), mean over seeds")
    out("  " + " " * 12 + "".join(f"{n[:8]:>9}" for n in ALL))
    for i, n in enumerate(ALL):
        out(f"  {n:<12}" + "".join(f"{C[i, j]:9.3f}" for j in range(len(ALL))))

    for g, names in R["groups"].items():
        G = S["groups"][g] = {"models": names}
        out(f"\n--- group {g}: {names}")
        bests = [r["groups"][g]["best"]["model"] for r in runs]
        oracles = [r["groups"][g]["oracle"]["model"] for r in runs]
        out(f"  best single by OOF per seed: {bests}   best on test per seed: {oracles}")
        G["best_models"] = bests
        idx = [ALL.index(n) for n in names]
        sub = C[np.ix_(idx, idx)]
        off = sub[np.triu_indices(len(names), 1)]
        G["mean_pairwise_corr"] = float(off.mean())
        out(f"  mean pairwise residual correlation: {off.mean():.3f} (min {off.min():.3f}, max {off.max():.3f})")
        methods = ["best", "oracle", "average", "weighted", "stack", "stack_gbm", "blend", "leak"]
        if task == "regression":
            methods.insert(6, "stack_pt")
        if task == "binary":
            methods.insert(3, "hard")
            methods.insert(methods.index("stack_gbm"), "stack_logit")
        G["methods"] = {}
        for m in methods:
            ks = [k for k in metrics if k in runs[0]["groups"][g][m]]
            vals = {k: [r["groups"][g][m][k] for r in runs] for k in ks}
            G["methods"][m] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "per_seed": v}
                               for k, v in vals.items()}
            txt = "  ".join(f"{k} {ms(v)}" for k, v in vals.items())
            diffs = []
            if m not in ("best",):
                for k in ks:
                    base = [r["groups"][g]["best"][k] for r in runs]
                    d, sd, lo, hi = paired(vals[k], base)
                    G["methods"][m][k]["vs_best"] = {"mean": float(d), "sd": float(sd), "lo": float(lo),
                                                     "hi": float(hi), "per_seed": list(np.asarray(vals[k]) - base)}
                    better = (d < 0) if lower_better[k] else (d > 0)
                    beyond = lo > 0 or hi < 0
                    tag = ("BETTER" if better else "WORSE") if beyond else "noise"
                    diffs.append(f"{k} d={d:+.4f} [{lo:+.4f},{hi:+.4f}] {tag}")
            out(f"  {m:<10} {txt}")
            if diffs:
                out(f"  {'':<10}   vs best: " + "; ".join(diffs))

        # weights and coefficients
        W = np.mean([r["groups"][g]["weighted"]["weights"] for r in runs], axis=0)
        Cs = np.mean([r["groups"][g]["stack"]["coefs"] for r in runs], axis=0)
        Cb = np.mean([r["groups"][g]["blend"]["coefs"] for r in runs], axis=0)
        Cl = np.mean([r["groups"][g]["leak"]["coefs"] for r in runs], axis=0)
        G["weights"], G["stack_coefs"], G["blend_coefs"], G["leak_coefs"] = W.tolist(), Cs.tolist(), Cb.tolist(), Cl.tolist()
        out(f"  simplex weights (OOF), mean over seeds: " + ", ".join(f"{n} {w:.3f}" for n, w in zip(names, W)))
        out(f"  stack meta coefs (+intercept), mean: " + ", ".join(f"{c:.3f}" for c in Cs))
        if task == "binary":
            Cg = np.mean([r["groups"][g]["stack_logit"]["coefs"] for r in runs], axis=0)
            G["stack_logit_coefs"] = Cg.tolist()
            out(f"  stack_logit meta coefs (+intercept), mean: " + ", ".join(f"{c:.3f}" for c in Cg))
        out(f"  blend meta coefs (+intercept), mean: " + ", ".join(f"{c:.3f}" for c in Cb))
        out(f"  LEAK  meta coefs (+intercept), mean: " + ", ".join(f"{c:.3f}" for c in Cl))
        k0 = metrics[0] if task == "regression" else "log_loss"
        leak_train = [r["groups"][g]["leak"]["train"][k0] for r in runs]
        stack_train = [r["groups"][g]["stack"]["train_oof"][k0] for r in runs]
        G["leak_train"] = float(np.mean(leak_train))
        G["stack_train_oof"] = float(np.mean(stack_train))
        out(f"  {k0} on the rows the meta-learner was fitted on: leak (in-sample) {np.mean(leak_train):.4f}, "
            f"stack (OOF) {np.mean(stack_train):.4f}")
        for k in (["roc_auc", "log_loss"] if task == "binary" else ["rmse"]):
            d, sd, lo, hi = paired([r["groups"][g]["leak"][k] for r in runs], [r["groups"][g]["stack"][k] for r in runs])
            out(f"  leak - stack on test {k}: {d:+.4f} [{lo:+.4f},{hi:+.4f}]")
            G.setdefault("leak_vs_stack", {})[k] = [float(d), float(lo), float(hi)]

        # costs (median over seeds)
        def med(f):
            return statistics.median(f(r) for r in runs)
        mods = lambda r: [r["models"][n] for n in names]  # noqa: E731
        cost = {
            "best": (med(lambda r: r["models"][r["groups"][g]["best"]["model"]]["fit_dev_s"]),
                     med(lambda r: r["models"][r["groups"][g]["best"]["model"]]["pred_10k_s"])),
            "average": (med(lambda r: sum(m["fit_dev_s"] for m in mods(r))),
                        med(lambda r: sum(m["pred_10k_s"] for m in mods(r)))),
            "stack": (med(lambda r: sum(m["fit_dev_s"] + sum(m["fit_folds_s"]) for m in mods(r))
                          + r["groups"][g]["stack"]["meta_fit_s"]),
                      med(lambda r: sum(m["pred_10k_s"] for m in mods(r)))),
            "blend": (med(lambda r: sum(m["fit_train_s"] for m in mods(r))),
                      med(lambda r: sum(m["pred_10k_s"] for m in mods(r)))),
        }
        G["cost"] = {k: {"fit_s": a, "pred_ms": b * 1000} for k, (a, b) in cost.items()}
        out("  cost (median over seeds): " + "; ".join(f"{k} fit {a:.1f}s pred10k {b * 1000:.0f}ms"
                                                     for k, (a, b) in cost.items()))
        out("  (weighted = stack fit cost minus the meta fit; hard/average = average)")

    # the w'Mw identity: MSE of an average equals w' M w, M = mean products of residuals
    if task == "regression":
        r0 = runs[0]
        M = np.array(r0["residual_cov"])
        for g, names in R["groups"].items():
            idx = [ALL.index(n) for n in names]
            w = np.full(len(names), 1 / len(names))
            mse = w @ M[np.ix_(idx, idx)] @ w
            out(f"\n  identity check seed 0 {g}: sqrt(w'Mw) = {np.sqrt(mse):.6f}  vs  average RMSE "
                f"{r0['groups'][g]['average']['rmse']:.6f}")
            assert abs(np.sqrt(mse) - r0["groups"][g]["average"]["rmse"]) < 1e-9
            S["groups"][g]["identity_seed0"] = {"M": M[np.ix_(idx, idx)].tolist(), "rmse_avg": float(np.sqrt(mse))}

    ck = runs[0]["sklearn_check"]
    S["sklearn_check"] = ck
    out(f"\n  sklearn check (seed 0, diverse): {ck}")
    out("")

out("environment: " + json.dumps(json.loads((RAW / "adult.json").read_text())["environment"]))
(HERE / "results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
(HERE / "results.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
print("\n".join(lines))
