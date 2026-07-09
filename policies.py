"""Resample-or-reroute policy and baselines.

MODELING CHOICES (skeleton — document/refine for the paper):
 * Draws are simulated by sampling (without replacement) from the k pre-computed
   draws in the correctness tensor for that (query, model).
 * VERIFIER: the default main experiment uses a perfect verifier that detects a
   correct sample, so a policy STOPS as soon as it has drawn a correct answer
   (early stopping => lower average cost). The verifier-quality ablation
   (run_pareto.py --quality) replaces this with a parametric imperfect verifier.
 * The adaptive policy uses only OFFLINE per-model priors (from a train split)
   plus its own draw outcomes; it never peeks at test ground truth to plan.

Every policy has the signature:
    policy(bq, costs, budget, ctx, rng) -> (drawn_correct: list[int], cost: float)
where bq is the (M, k) correctness slice for one query and ctx carries priors.
"""
import numpy as np


def _shuffled_order(M, k, rng):
    """A random draw order (without replacement) for each model, this trial."""
    return [rng.permutation(k) for _ in range(M)]


def _run_sequential(bq, costs, budget, choose_next, rng, stop_on_correct=True):
    """Generic loop: repeatedly pick a model via choose_next(state) and draw once."""
    M, k = bq.shape
    order = _shuffled_order(M, k, rng)
    ptr = [0] * M
    drawn = []
    cost = 0.0
    state = {"ptr": ptr, "drawn_by_model": [[] for _ in range(M)]}
    while True:
        if stop_on_correct and any(c == 1 for c in drawn):
            break
        affordable = [m for m in range(M) if ptr[m] < k and cost + costs[m] <= budget + 1e-9]
        if not affordable:
            break
        m = choose_next(state, affordable)
        if m is None:
            break
        idx = order[m][ptr[m]]
        ptr[m] += 1
        c = int(bq[m, idx])
        cost += costs[m]
        drawn.append(c)
        state["drawn_by_model"][m].append(c)
    return drawn, cost


# ---------------------------------------------------------------- the method
def resample_or_reroute(bq, costs, budget, ctx, rng, ucb=False, prior_strength=2.0):
    """Greedy marginal correctness-per-cost with PER-QUERY Bayesian belief update.

    The model's offline global accuracy sets the prior MEAN, but we give it only
    a small pseudo-count (`prior_strength`) so this query's own outcomes move the
    posterior fast: one or two failures on the committed model drop its score
    below a fresh model's, triggering an early REROUTE; a run of successes keeps
    it RESAMPLING. A correct draw ends the query (perfect-verifier early stop).
    `ucb=True` adds an exploration bonus that favors probing untried models
    (value of information -> find the query's specialist sooner).
    """
    M = len(costs)
    p_glob = ctx["a0"] / (ctx["a0"] + ctx["b0"])          # offline accuracy = prior mean
    a_base = prior_strength * p_glob + 1e-6
    b_base = prior_strength * (1.0 - p_glob) + 1e-6
    t = {"n": 0}

    def choose(state, affordable):
        t["n"] += 1
        succ = np.array([sum(state["drawn_by_model"][m]) for m in range(M)], float)
        n_m = np.array([len(state["drawn_by_model"][m]) for m in range(M)], float)
        phat = (a_base + succ) / (a_base + b_base + n_m)   # per-query posterior mean
        score = phat / costs
        if ucb:
            score = score + np.sqrt(2.0 * np.log(t["n"] + 1.0) / (n_m + 1.0)) / costs
        return max(affordable, key=lambda m: score[m])

    return _run_sequential(bq, costs, budget, choose, rng,
                           stop_on_correct=ctx.get("stop_on_correct", True))


# ------------------------------------------------------------------ baselines
def single_route(bq, costs, budget, ctx, rng):
    """Commit to the best single model (by offline accuracy), draw once."""
    m = ctx["best_acc_model"]
    order = _shuffled_order(*bq.shape, rng=rng)
    c = int(bq[m, order[m][0]])
    return [c], float(costs[m])


def fixed_best_of_k(bq, costs, budget, ctx, rng):
    """Strongest FAIR single-model baseline: commit to the one model that
    maximizes expected best-of-K accuracy *at this budget* (offline priors),
    then resample only it (never reroute), early stop on correct.

    This is budget-aware, so at a small budget it picks a cheap, decent model
    (many draws) rather than an unaffordable big one -- a much stronger and
    fairer comparison than fixing the single most-accurate model.
    """
    phat = ctx["a0"] / (ctx["a0"] + ctx["b0"])
    K = np.floor(budget / costs).astype(int)                 # affordable draws per model
    exp_acc = 1.0 - (1.0 - phat) ** np.maximum(K, 0)         # expected best-of-K correctness
    exp_acc = np.where(K >= 1, exp_acc, -1.0)                # mask unaffordable models
    m = int(np.argmax(exp_acc))
    if K[m] < 1:
        return [], 0.0

    def choose(state, affordable):
        return m if m in affordable else None

    return _run_sequential(bq, costs, budget, choose, rng,
                           stop_on_correct=ctx.get("stop_on_correct", True))


def learned_router(bq, costs, budget, ctx, rng):
    """Learned-router baseline: one commit, correctness given by q_router[i].

    q_router is a per-query outcome supplied via ctx['q_router_i']; cost is one
    draw of the routed model (proxied by the best-accuracy model's cost).
    """
    return [int(round(ctx["q_router_i"]))], float(costs[ctx["best_acc_model"]])


def frugal_cascade(bq, costs, budget, ctx, rng):
    """FrugalGPT-style: escalate cheap -> expensive, one draw each, early stop."""
    order_by_cost = list(np.argsort(costs))
    seq = {"i": 0}

    def choose(state, affordable):
        while seq["i"] < len(order_by_cost):
            m = order_by_cost[seq["i"]]
            seq["i"] += 1
            if m in affordable:
                return m
        return None

    return _run_sequential(bq, costs, budget, choose, rng,
                           stop_on_correct=ctx.get("stop_on_correct", True))


def random_alloc(bq, costs, budget, ctx, rng):
    """Spend the budget on uniformly random models, one draw each, early stop."""
    def choose(state, affordable):
        return int(rng.choice(affordable))

    return _run_sequential(bq, costs, budget, choose, rng,
                           stop_on_correct=ctx.get("stop_on_correct", True))


def oracle_alloc(bq, costs, budget, ctx, rng):
    """Per-query ORACLE upper bound (reference ceiling; uses ground truth).

    With a perfect verifier and early stopping, the ideal is to spend as little
    as possible: pick the cheapest model that actually has a correct draw and is
    affordable, and succeed in one draw. Accuracy ceiling = fraction of queries
    where any affordable model can be correct; cost = that cheapest model.
    Labeled a ceiling, not a deployable policy.
    """
    order = np.argsort(costs)
    for m in order:
        if costs[m] > budget + 1e-9:
            continue
        if int(bq[m].max()) == 1:          # oracle: this model can be correct here
            return [1], float(costs[m])
    return [0], 0.0


POLICIES = {
    "resample_or_reroute": lambda *a, **k: resample_or_reroute(*a, ucb=False, **k),
    "resample_or_reroute_ucb": lambda *a, **k: resample_or_reroute(*a, ucb=True, **k),
    "oracle_alloc": oracle_alloc,          # reference ceiling (uses ground truth)
    "single_route": single_route,
    "fixed_best_of_k": fixed_best_of_k,
    "learned_router": learned_router,
    "frugal_cascade": frugal_cascade,
    "random_alloc": random_alloc,
}
