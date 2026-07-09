"""Load the multi-draw correctness tensors produced by routing-oracle-experiment.

Tensor format (correctness_slim.npz):
  b        : (N, M, k) int8   correctness of each draw  (N queries, M models, k draws)
  b_single : (N, M)    int8   the single T=0.2 draw
  q_router : (N,)      float  learned-router per-query outcome (in [0,1])
  meta     : json str  {"N","M","k","models":[...]}
"""
import json
import os
import re

import numpy as np

DEFAULT_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "..", "routing-oracle", "routing-oracle-experiment", "artifacts"
)


def model_cost(name: str) -> float:
    """Cost proxy = parameter count in billions parsed from the model name.

    This is a stand-in for real $/token pricing. Replace with a measured
    price/latency vector before the paper's final experiments.
    """
    m = re.search(r"(\d+(?:\.\d+)?)\s*[Bb]\b", name)
    if m:
        return float(m.group(1))
    return 8.0  # fallback for names without an explicit size


def load_bench(bench: str, root: str = None) -> dict:
    root = root or DEFAULT_ROOT
    path = os.path.join(root, bench, "correctness_slim.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Benchmarks with a slim tensor: math500, gpqa. "
            "(gsm8k currently ships decomposition.json only — regenerate its tensor to include it.)"
        )
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    costs = np.array([model_cost(m) for m in meta["models"]], dtype=float)
    return {
        "b": z["b"].astype(np.int8),                # (N, M, k)
        "b_single": z["b_single"].astype(np.int8),  # (N, M)
        "q_router": z["q_router"].astype(float),    # (N,)
        "models": list(meta["models"]),
        "costs": costs,                             # (M,)
        "N": int(meta["N"]),
        "M": int(meta["M"]),
        "k": int(meta["k"]),
        "bench": bench,
    }
