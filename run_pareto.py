"""Cost-quality Pareto driver for resample-or-reroute vs. baselines.

Usage:
    python run_pareto.py --bench math500 --trials 20            # main (perfect verifier, early stop)
    python run_pareto.py --bench gpqa   --trials 20
    python run_pareto.py --bench math500 --quality 0.6          # verifier-quality ablation

Outputs:
    results/pareto_<bench>_q<quality>.csv
    figures/pareto_<bench>_q<quality>.{pdf,png}   (if matplotlib is available)

Protocol: queries are split 50/50; per-model Beta priors and the best-accuracy
model are calibrated on the train half; all policies are evaluated on the test
half, averaged over `--trials` random draw orderings.
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np

from data import load_bench
from policies import POLICIES
from verifier import verifier_expected_correct


def compute_priors(b_train):
    """b_train: (Ntr, M, k) -> Beta(a0,b0) per model + argmax-accuracy model."""
    succ = b_train.sum(axis=(0, 2)).astype(float)
    total = b_train.shape[0] * b_train.shape[2]
    a0 = succ + 1.0
    b0 = (total - succ) + 1.0
    acc = b_train.mean(axis=(0, 2))
    return a0, b0, int(np.argmax(acc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="math500", help="math500 | gpqa")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--quality", type=float, default=1.0,
                    help="1.0 = perfect verifier + early stop (main); <1 = imperfect-verifier ablation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "results"))
    args = ap.parse_args()

    d = load_bench(args.bench)
    b, costs, N, qr = d["b"], d["costs"], d["N"], d["q_router"]
    perfect = args.quality >= 1.0

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(N)
    tr, te = idx[: N // 2], idx[N // 2:]
    a0, b0, best = compute_priors(b[tr])

    budgets = np.unique(np.round(np.linspace(costs.min(), 6 * costs.max(), 12))).astype(float)
    os.makedirs(args.outdir, exist_ok=True)

    rows = []
    for name, policy in POLICIES.items():
        for B in budgets:
            accs, cs = [], []
            for t in range(args.trials):
                trng = np.random.default_rng(args.seed * 100003 + t)
                for i in te:
                    ctx = {
                        "a0": a0, "b0": b0, "best_acc_model": best,
                        "q_router_i": float(qr[i]),
                        "stop_on_correct": perfect,
                    }
                    drawn, cost = policy(b[i], costs, float(B), ctx, trng)
                    if perfect:
                        accs.append(1.0 if any(c == 1 for c in drawn) else 0.0)
                    else:
                        accs.append(verifier_expected_correct(drawn, args.quality))
                    cs.append(cost)
            rows.append({"policy": name, "budget": float(B),
                         "mean_cost": float(np.mean(cs)),
                         "accuracy": float(np.mean(accs))})
            if name in ("single_route", "learned_router"):
                break  # single-commit policies are budget-independent points

    csvp = os.path.join(args.outdir, f"pareto_{args.bench}_q{args.quality}.csv")
    with open(csvp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["policy", "budget", "mean_cost", "accuracy"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csvp} ({len(rows)} rows)")
    _plot(rows, args)


def _plot(rows, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print("matplotlib unavailable, CSV only:", e)
        return
    figdir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(figdir, exist_ok=True)
    by = defaultdict(list)
    for r in rows:
        by[r["policy"]].append((r["mean_cost"], r["accuracy"]))
    plt.figure(figsize=(6.2, 4.3))
    for name, pts in sorted(by.items()):
        pts = sorted(pts)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        highlight = name.startswith("resample_or_reroute")
        plt.plot(xs, ys, ("-o" if highlight else "--s"),
                 lw=2.2 if highlight else 1.2, ms=5 if highlight else 4,
                 label=name, zorder=3 if highlight else 2)
    plt.xlabel("mean cost per query  (parameter-B proxy for \\$/token)")
    plt.ylabel("accuracy")
    plt.title(f"Cost-quality Pareto: {args.bench}  (verifier q={args.quality})")
    plt.legend(fontsize=7, frameon=True, framealpha=1.0)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(figdir, f"pareto_{args.bench}_q{args.quality}.pdf")
    plt.savefig(p)
    plt.savefig(p.replace(".pdf", ".png"), dpi=140)
    print(f"wrote {p} (+ .png)")


if __name__ == "__main__":
    main()
