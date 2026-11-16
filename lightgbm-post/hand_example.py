"""Hand-checkable examples for the "lightgbm" blog post.

Part 1  Leaf-wise (best-first) vs level-wise (depth-wise) growth on an 8-row
        regression toy. Both trees are grown to 4 leaves with the gain formula
        of the XGBoost post (squared loss, h = 1, lambda = 0, eta = 1), then
        LightGBM itself is asked to grow the same two trees:
          leaf-wise  = num_leaves=4, max_depth=-1
          level-wise = num_leaves=4, max_depth=2 (the only 4-leaf tree of
                       depth 2 is the full one, so this reproduces level-wise)
Part 2  GOSS on 10 rows: keep the top-a rows by |g|, sample b*n of the rest,
        re-weight them by (1-a)/b. Enumerates every possible draw to show the
        weighted gradient sum is unbiased.
Part 3  EFB: bundling mutually exclusive sparse features with bin offsets.

Run from the repository root:
    .venv/Scripts/python.exe lightgbm-post/hand_example.py
Output is printed. results.txt is produced with:
    .venv/Scripts/python.exe lightgbm-post/hand_example.py  > lightgbm-post/results.txt
    .venv/Scripts/python.exe lightgbm-post/experiments.py  >> lightgbm-post/results.txt
    .venv/Scripts/python.exe lightgbm-post/param_checks.py >> lightgbm-post/results.txt 2>&1
"""

from __future__ import annotations

import itertools
import json
from fractions import Fraction

import lightgbm as lgb
import numpy as np

# ---------------------------------------------------------------------------
# Part 1: leaf-wise vs level-wise
# ---------------------------------------------------------------------------

X1 = np.arange(1, 9, dtype=float).reshape(-1, 1)  # x = 1..8
Y1 = np.array([20, 10, 20, 18, 38, 34, 16, 28], dtype=float)


def half_gain(gl: Fraction, nl: int, gr: Fraction, nr: int) -> Fraction:
    """XGBoost-paper gain with the 1/2 factor, h = 1 per row, lambda = 0."""
    g = gl + gr
    return Fraction(1, 2) * (gl * gl / nl + gr * gr / nr - g * g / (nl + nr))


def best_split(rows: list[int], g: list[Fraction]):
    """Scan every cut of a contiguous, x-sorted row set. Returns all cuts and the best."""
    table = []
    for k in range(1, len(rows)):
        left, right = rows[:k], rows[k:]
        gl = sum(g[i] for i in left)
        gr = sum(g[i] for i in right)
        table.append((k, gl, gr, half_gain(gl, len(left), gr, len(right))))
    best = max(table, key=lambda t: t[3])
    return table, best


def describe_cut(rows, k):
    # threshold between x of rows[k-1] and rows[k] (x = row index + 1)
    return f"x <= {rows[k-1] + 1}"


def grow(g, policy: str):
    leaves = [list(range(8))]
    log, total = [], Fraction(0)
    if policy == "level":
        for _depth in range(2):
            nxt = []
            for rows in leaves:
                _, (k, gl, gr, gain) = best_split(rows, g)
                log.append((rows, describe_cut(rows, k), gain))
                total += gain
                nxt += [rows[:k], rows[k:]]
            leaves = nxt
    else:
        while len(leaves) < 4:
            cands = []
            for j, rows in enumerate(leaves):
                if len(rows) > 1:
                    _, best = best_split(rows, g)
                    cands.append((best[3], j, best))
            gain, j, (k, gl, gr, _) = max(cands, key=lambda c: c[0])
            rows = leaves.pop(j)
            log.append((rows, describe_cut(rows, k), gain,
                        [(leaves_rows_str(r), str(best_split(r, g)[1][3]) if len(r) > 1 else "-")
                         for r in leaves[:j] + [rows] + leaves[j:]]))
            leaves[j:j] = [rows[:k], rows[k:]]
            total += gain
    return leaves, log, total


def leaves_rows_str(rows):
    return "rows " + ",".join(str(r + 1) for r in rows)


def part1():
    print("=" * 78)
    print("PART 1  Leaf-wise vs level-wise, 8 rows, squared loss, lambda=0, eta=1")
    print("=" * 78)
    f0 = Fraction(int(Y1.sum()), len(Y1))
    g = [f0 - Fraction(int(v)) for v in Y1]  # g = F0 - y, h = 1
    print(f"y = {Y1.astype(int).tolist()}  F0 = mean(y) = {f0}")
    print(f"g = F0 - y = {[int(v) for v in g]}  G = {sum(g)}  H = 8")
    loss0 = sum(v * v for v in g) / 2
    print(f"loss before the tree, sum 1/2 (F0-y)^2 = {loss0}")

    print("\nRoot: every cut (gain uses the 1/2 factor of the XGBoost paper)")
    table, best = best_split(list(range(8)), g)
    for k, gl, gr, gain in table:
        print(f"  x <= {k}:  G_L={int(gl):4d} n_L={k}  G_R={int(gr):4d} n_R={8-k}  gain={float(gain):9.3f}")

    for name, rows in [("left  (rows 1-4)", [0, 1, 2, 3]), ("right (rows 5-8)", [4, 5, 6, 7]),
                       ("rows 5-6", [4, 5]), ("rows 7-8", [6, 7])]:
        table, best = best_split(rows, g)
        print(f"\nCandidate cuts inside {name}:")
        for k, gl, gr, gain in table:
            print(f"  {describe_cut(rows, k)}:  G_L={int(gl):4d}  G_R={int(gr):4d}  gain={float(gain):8.3f}")

    results = {}
    for policy in ("leaf", "level"):
        leaves, log, total = grow(g, policy)
        print(f"\n{policy}-wise growth to 4 leaves:")
        for step, entry in enumerate(log, 1):
            print(f"  split {step}: {leaves_rows_str(entry[0])} at {entry[1]}  gain={entry[2]}")
            if policy == "leaf":
                print(f"           candidates before this split: {entry[3]}")
        preds = np.zeros(8)
        for rows in leaves:
            w = -sum(g[i] for i in rows) / len(rows)
            for i in rows:
                preds[i] = float(f0 + w)
            print(f"  leaf {leaves_rows_str(rows):12s} w = -G/H = {str(w):>4s}  prediction = {float(f0 + w)}")
        loss = sum((Fraction(p).limit_denominator(1000) - Fraction(int(y))) ** 2 for p, y in zip(preds, Y1)) / 2
        depth = 3 if policy == "leaf" else 2
        print(f"  total gain = {total}  loss after = {loss}  (check: {loss0} - {total} = {loss0 - total})  depth = {depth}")
        results[policy] = preds

    print("\nLightGBM check (lightgbm %s)" % lgb.__version__)
    base = dict(objective="regression", learning_rate=1.0, lambda_l2=0.0, lambda_l1=0.0,
                min_data_in_leaf=1, min_sum_hessian_in_leaf=0.0, min_data_in_bin=1,
                max_bin=255, min_gain_to_split=0.0, boost_from_average=True,
                feature_pre_filter=False, deterministic=True, num_threads=1, verbose=-1, seed=0)
    for policy, extra in [("leaf", dict(num_leaves=4, max_depth=-1)),
                          ("level", dict(num_leaves=4, max_depth=2))]:
        ds = lgb.Dataset(X1, Y1, params={"min_data_in_bin": 1, "max_bin": 255, "verbose": -1})
        booster = lgb.train({**base, **extra}, ds, num_boost_round=1)
        tree = booster.dump_model()["tree_info"][0]["tree_structure"]
        print(f"\n  {policy}-wise params {extra}")
        print_tree(tree, indent="    ")
        pred = booster.predict(X1)
        print(f"    predict = {np.round(pred, 6).tolist()}")
        print(f"    hand    = {results[policy].tolist()}")
        print(f"    max |difference| = {np.max(np.abs(pred - results[policy])):.2e}")
    print("  Note: LightGBM's split_gain has no 1/2 factor, so it should be 2x the hand gain.")
    print("  Note: with boost_from_average the first tree's leaf values include F0.")


def print_tree(node, indent=""):
    if "leaf_value" in node:
        print(f"{indent}leaf value={node['leaf_value']:.6f} count={node.get('leaf_count')}")
        return
    print(f"{indent}split x <= {node['threshold']}  split_gain={node['split_gain']:.6f} "
          f"count={node['internal_count']}")
    print_tree(node["left_child"], indent + "  ")
    print_tree(node["right_child"], indent + "  ")


# ---------------------------------------------------------------------------
# Part 2: GOSS
# ---------------------------------------------------------------------------

G2 = [Fraction(v) for v in ["-3.0", "2.5", "0.4", "-0.2", "0.3", "-0.5", "0.1", "0.6", "-0.4", "0.2"]]


def part2():
    print("\n" + "=" * 78)
    print("PART 2  GOSS on 10 rows, a = top_rate = 0.2, b = other_rate = 0.2")
    print("=" * 78)
    n, a, b = len(G2), Fraction(2, 10), Fraction(2, 10)
    top_k, other_k = int(n * a), int(n * b)
    factor = (1 - a) / b
    order = sorted(range(n), key=lambda i: -abs(G2[i]))
    top = order[:top_k]
    rest = order[top_k:]
    print(f"g = {[float(v) for v in G2]}")
    print(f"exact sum of g = {float(sum(G2))}")
    print(f"top {top_k} rows by |g|: {[i + 1 for i in top]}  (g = {[float(G2[i]) for i in top]})")
    print(f"rest ({len(rest)} rows) sum of g = {float(sum(G2[i] for i in rest))}")
    print(f"sample {other_k} of the {len(rest)} rest rows, weight (1-a)/b = {factor}")
    print(f"each rest row is drawn with probability {other_k}/{len(rest)} = {Fraction(other_k, len(rest))}, "
          f"so E[weight] = {Fraction(other_k, len(rest)) * factor}")
    sum_top = sum(G2[i] for i in top)
    ests = []
    for pick in itertools.combinations(rest, other_k):
        est = sum_top + factor * sum(G2[i] for i in pick)
        ests.append(est)
    print(f"all C({len(rest)},{other_k}) = {len(ests)} possible draws:")
    print(f"  mean of the GOSS estimate = {float(sum(ests) / len(ests))}  (exact {float(sum(G2))})")
    print(f"  min = {float(min(ests))}  max = {float(max(ests))}")
    example = (rest[0], rest[3])
    est = sum_top + factor * sum(G2[i] for i in example)
    print(f"example draw rows {[i + 1 for i in example]} (g = {[float(G2[i]) for i in example]}): "
          f"estimate = {float(sum_top)} + {factor} x ({float(sum(G2[i] for i in example))}) = {float(est)}")
    # Same thing without re-weighting: biased toward the large-gradient rows.
    no_w = [sum_top + sum(G2[i] for i in pick) for pick in itertools.combinations(rest, other_k)]
    print(f"  without re-weighting the mean would be {float(sum(no_w) / len(no_w))}")


# ---------------------------------------------------------------------------
# Part 3: EFB
# ---------------------------------------------------------------------------

def part3():
    print("\n" + "=" * 78)
    print("PART 3  EFB: three mutually exclusive sparse features, bins + offsets")
    print("=" * 78)
    # Bin index per feature; 0 means "zero / default bin". At most one feature is non-zero per row.
    A = [1, 0, 0, 2, 0, 0]  # A uses bins 1..2
    B = [0, 3, 0, 0, 1, 0]  # B uses bins 1..3
    C = [0, 0, 1, 0, 0, 0]  # C uses bin 1
    num_bin = {"A": 3, "B": 4, "C": 2}  # including bin 0
    conflicts = sum(1 for r in zip(A, B, C) if sum(v != 0 for v in r) > 1)
    print(f"rows where more than one feature is non-zero (conflicts): {conflicts}")
    offset_a, offset_b, offset_c = 0, num_bin["A"] - 1, num_bin["A"] - 1 + num_bin["B"] - 1
    print(f"offsets: A +{offset_a}, B +{offset_b}, C +{offset_c}")
    bundle = []
    for a, b_, c in zip(A, B, C):
        v = 0
        if a:
            v = a + offset_a
        if b_:
            v = b_ + offset_b
        if c:
            v = c + offset_c
        bundle.append(v)
    print("row  A  B  C  -> bundle bin")
    for i, (a, b_, c, v) in enumerate(zip(A, B, C, bundle), 1):
        print(f"{i:3d}  {a}  {b_}  {c}  -> {v}")
    print("bundle bin ranges: A -> 1..2, B -> 3..5, C -> 6; bin 0 = all three zero")


if __name__ == "__main__":
    print(f"lightgbm imported from: {'installed site-packages' if 'site-packages' in lgb.__file__ else 'a local folder (shadowed!)'}")
    part1()
    part2()
    part3()
