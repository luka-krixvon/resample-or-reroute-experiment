"""[FALLBACK / standalone] Generate a code (pass@k) correctness tensor.

PREFERRED: use routing-oracle-experiment/scripts/gen_code_tensor.py, which
reuses src.generate so the generation settings match math500/gpqa exactly.
Use THIS standalone version only if you don't want to reuse that pipeline.

Produces the SAME format as the math500/gpqa `correctness_slim.npz`, so
run_pareto.py consumes it unchanged.

RUN THIS ON A GPU VM (it does model inference) + in a SANDBOX (it executes
model-generated code). It is a scaffold: wire `generate_samples(...)` to your
serving stack (vLLM / TGI / transformers) and confirm the execution harness.

Output: <outdir>/<bench>/correctness_slim.npz with
    b        : (N, M, k) int8   1 iff sample passed all unit tests
    b_single : (N, M)    int8   the first draw (T=0.2 proxy)
    q_router : (N,)      float  0.0 placeholder (fill if you have a router)
    gold     : (N,)      object task_id per problem
    meta     : json str  {"N","M","k","models":[...]}

Correctness = the sample passes ALL of the problem's unit tests (execution),
i.e. per-draw pass@1; best-of-K over draws is exactly pass@k, matching Thm. 2(b).
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

# Same 11-model pool as the math/gpqa experiments (keep consistent).
MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "Qwen/Qwen2.5-14B-Instruct-AWQ",
    "Qwen/Qwen2.5-32B-Instruct-AWQ",
    "microsoft/phi-4",
    "allenai/OLMo-2-1124-7B-Instruct",
    "01-ai/Yi-1.5-9B-Chat",
    "ibm-granite/granite-3.3-8b-instruct",
    "google/gemma-2-9b-it",
    "meta-llama/Llama-3.1-8B-Instruct",
]


def load_problems(bench: str):
    """Return list of {task_id, prompt, tests, entry_point}. Uses EvalPlus.

    pip install evalplus ; benches: 'humanevalplus' | 'mbppplus'.
    """
    if bench == "humanevalplus":
        from evalplus.data import get_human_eval_plus  # type: ignore
        data = get_human_eval_plus()
    elif bench == "mbppplus":
        from evalplus.data import get_mbpp_plus  # type: ignore
        data = get_mbpp_plus()
    else:
        raise ValueError(f"unknown code bench {bench!r} (humanevalplus | mbppplus)")
    return [{"task_id": tid, "prompt": p["prompt"], "tests": p.get("test", ""),
             "entry_point": p.get("entry_point", "")} for tid, p in data.items()]


def generate_samples(model: str, prompt: str, k: int, temperature: float = 0.2):
    """TODO: plug in your serving stack. Return a list of k code completions (str).

    Mirror the math/gpqa generation settings (T=0.2, top_p=1.0, same max tokens).
    Example (vLLM):
        from vllm import LLM, SamplingParams
        out = llm.generate([prompt]*k, SamplingParams(n=1, temperature=temperature, top_p=1.0))
        return [o.outputs[0].text for o in out]
    """
    raise NotImplementedError("Wire generate_samples() to your model server.")


def passes_tests(completion: str, problem: dict, timeout: float = 10.0) -> int:
    """Execute completion + the problem's tests in a subprocess. 1 iff all pass.

    SANDBOX THIS. Prefer evalplus's own evaluator for correctness; this minimal
    runner is a placeholder.
    """
    program = problem["prompt"] + completion + "\n" + problem["tests"] + "\n"
    if problem.get("entry_point"):
        program += f"\ncheck({problem['entry_point']})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=timeout)
        return int(r.returncode == 0)
    except Exception:
        return 0
    finally:
        os.unlink(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="humanevalplus", help="humanevalplus | mbppplus")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--outdir", default=os.path.join(
        os.path.dirname(__file__), "..", "..", "routing-oracle", "routing-oracle-experiment", "artifacts"))
    ap.add_argument("--limit", type=int, default=None, help="cap #problems (debug)")
    args = ap.parse_args()

    probs = load_problems(args.bench)
    if args.limit:
        probs = probs[: args.limit]
    N, M, k = len(probs), len(MODELS), args.k
    b = np.zeros((N, M, k), dtype=np.int8)
    print(f"{args.bench}: N={N} problems x M={M} models x k={k} draws")

    for j, model in enumerate(MODELS):
        for i, prob in enumerate(probs):
            comps = generate_samples(model, prob["prompt"], k, args.temperature)
            for d, c in enumerate(comps[:k]):
                b[i, j, d] = passes_tests(c, prob)
        print(f"  [{j+1}/{M}] {model} done")

    out = os.path.join(args.outdir, args.bench)
    os.makedirs(out, exist_ok=True)
    meta = {"N": N, "M": M, "k": k, "models": MODELS}
    np.savez_compressed(
        os.path.join(out, "correctness_slim.npz"),
        b=b,
        b_single=b[:, :, 0].astype(np.int8),
        q_router=np.zeros(N, dtype=float),
        gold=np.array([p["task_id"] for p in probs], dtype=object),
        meta=json.dumps(meta),
    )
    print("wrote", os.path.join(out, "correctness_slim.npz"))


if __name__ == "__main__":
    main()
