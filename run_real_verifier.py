"""REAL (label-free) verifier instantiation: agreement-based verification.

Uses the full tensors (data_full/<bench>_full.npz, with per-draw answer strings
Y) to replay RoR and budget-aware best-of-K under a deployable verifier that a
real system can actually run:

  * a drawn sample is "verified" when its extracted answer AGREES with another
    drawn sample (answer-cluster of size >= A);
  * the query stops early once any cluster reaches size A (consensus reached)
    or the budget is exhausted;
  * the final answer is the plurality cluster; correctness is evaluated against
    gold exactly as the tensors' b was.

This is self-consistency turned into a verifier signal — no labels, no reward
model, imperfect by construction (clusters can split on surface form, and two
models can agree on the same WRONG answer).

Outputs: results/real_verifier.csv + ../paper/tab_realverifier.tex
Usage:   python run_real_verifier.py
"""
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(HERE, "..", "paper_canonical")
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

BENCHES = ["gsm8k", "math500", "gpqa"]
TRIALS = 20
SEED = 0
AGREE = 2                     # consensus threshold A
BUDGETS = [26.0, 58.0]        # low + mid per-query budgets (B; early stop spends less)


def model_cost(name):
    import re
    m = re.search(r"(\d+(?:\.\d+)?)\s*[Bb]\b", name)
    return float(m.group(1)) if m else 8.0


def load_full(bench):
    z = np.load(os.path.join(HERE, "data_full", f"{bench}_full.npz"), allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    costs = np.array([model_cost(m) for m in meta["models"]], float)
    return dict(b=z["b"], Y=z["Y"], costs=costs, N=meta["N"], M=meta["M"], k=meta["k"])


def norm_ans(a):
    return str(a).strip().lower()


def replay_query(bq, Yq, costs, budget, order, choose):
    """Generic agreement-verified loop. choose(state, affordable) -> model.
    Returns (correct, cost)."""
    M, k = bq.shape
    ptr = [0] * M
    clusters = {}                 # norm answer -> [count, is_correct]
    drawn_by_model = [[] for _ in range(M)]   # list of norm answers per model
    cost = 0.0
    state = {"drawn_by_model": drawn_by_model, "clusters": clusters}
    while True:
        best = max(clusters.values(), key=lambda v: v[0]) if clusters else None
        if best is not None and best[0] >= AGREE:
            break                                   # consensus reached
        affordable = [m for m in range(M) if ptr[m] < k and cost + costs[m] <= budget + 1e-9]
        if not affordable:
            break
        m = choose(state, affordable)
        d = order[m][ptr[m]]; ptr[m] += 1
        a = norm_ans(Yq[m, d]); c = int(bq[m, d])
        cost += costs[m]
        drawn_by_model[m].append(a)
        if a not in clusters:
            clusters[a] = [0, c]
        clusters[a][0] += 1
    if not clusters:
        return 0, cost
    top = max(clusters.values(), key=lambda v: v[0])
    return int(top[1]), cost


def ror_choose(costs, p_glob, s=2.0):
    """RoR scoring with agreement-verified successes."""
    M = len(costs)

    def choose(state, affordable):
        succ = np.zeros(M); n = np.zeros(M)
        cl = state["clusters"]
        for m in range(M):
            answers = state["drawn_by_model"][m]
            n[m] = len(answers)
            succ[m] = sum(1 for a in answers if cl.get(a, [0])[0] >= AGREE or
                          (cl.get(a, [0])[0] >= 2))     # verified = agreed with another draw
        phat = (s * p_glob + succ) / (s + n)
        score = phat / costs
        return max(affordable, key=lambda m: score[m])
    return choose


def bok_choose(costs, p_glob, budget):
    """Budget-aware best-of-K: pick one model maximizing expected best-of-K, only it."""
    K = np.floor(budget / costs).astype(int)
    exp_acc = np.where(K >= 1, 1.0 - (1.0 - p_glob) ** np.maximum(K, 1), -1.0)
    mm = int(np.argmax(exp_acc))

    def choose(state, affordable):
        return mm if mm in affordable else None
    return choose


def main():
    rng0 = np.random.default_rng(SEED)
    rows = []
    for bench in BENCHES:
        d = load_full(bench)
        b, Y, costs, N, M, k = d["b"], d["Y"], d["costs"], d["N"], d["M"], d["k"]
        idx = np.random.default_rng(SEED).permutation(N)
        tr, te = idx[: N // 2], idx[N // 2:]
        p_glob = b[tr].mean(axis=(0, 2))
        for B in BUDGETS:
            for name in ["RoR", "BoK"]:
                accs, cs = [], []
                for t in range(TRIALS):
                    trng = np.random.default_rng(SEED * 100003 + t)
                    order = [trng.permutation(k) for _ in range(M)]
                    ch = (ror_choose(costs, p_glob) if name == "RoR"
                          else bok_choose(costs, p_glob, B))
                    for i in te:
                        order = [trng.permutation(k) for _ in range(M)]
                        correct, cost = replay_query(b[i], Y[i], costs, B, order, ch)
                        accs.append(correct); cs.append(cost)
                rows.append(dict(bench=bench, budget=B, policy=name,
                                 accuracy=float(np.mean(accs)),
                                 mean_cost=float(np.mean(cs))))
                print(f"{bench} B={B:.0f} {name}: acc={np.mean(accs):.3f} cost={np.mean(cs):.1f}")
    with open(os.path.join(RES, "real_verifier.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bench", "budget", "policy", "accuracy", "mean_cost"])
        w.writeheader(); w.writerows(rows)

    # LaTeX table
    def cell(bench, B, pol, key):
        for r in rows:
            if r["bench"] == bench and r["budget"] == B and r["policy"] == pol:
                return r[key]
        return None
    lines = [
        "% AUTO-GENERATED by experiments/run_real_verifier.py -- do not hand-edit.",
        "\\begin{table}[t]\\centering",
        "\\caption{Agreement-verified replay (consensus threshold $A{=}2$; label-free"
        " self-consistency verifier): accuracy (mean spent cost) at two per-query"
        " budgets, mean over 20 draw orderings.}",
        "\\label{tab:realverifier}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{llcccc}\\toprule",
        " & & \\multicolumn{2}{c}{$B=26$} & \\multicolumn{2}{c}{$B=58$} \\\\",
        "Benchmark & Policy & acc & cost & acc & cost \\\\ \\midrule",
    ]
    disp = {"gsm8k": "GSM8K", "math500": "MATH-500", "gpqa": "GPQA"}
    for bench in BENCHES:
        for pol in ["RoR", "BoK"]:
            nm = "RoR (agreement)" if pol == "RoR" else "best-of-$K$ (agreement)"
            row = [disp[bench] if pol == "RoR" else "", nm]
            for B in BUDGETS:
                row += [f"{cell(bench,B,pol,'accuracy'):.3f}",
                        f"{cell(bench,B,pol,'mean_cost'):.1f}"]
            lines.append(" & ".join(row) + (" \\\\ \\midrule" if pol == "BoK" and bench != BENCHES[-1] else " \\\\"))
    lines += ["\\bottomrule\\end{tabular}}\\end{table}"]
    open(os.path.join(PAPER, "tab_realverifier.tex"), "w").write("\n".join(lines) + "\n")
    print("wrote tab_realverifier.tex")


if __name__ == "__main__":
    main()
