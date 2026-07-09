# Resample or Reroute? Budget-Aware Test-Time Model Selection for LLMs

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)
![Serving](https://img.shields.io/badge/serving-vLLM-orange.svg)
![Hardware](https://img.shields.io/badge/generation-2%C3%97%20RTX%204090-lightgrey.svg)
![Replay](https://img.shields.io/badge/replay-CPU%20only-brightgreen.svg)
![No API key](https://img.shields.io/badge/API%20key-not%20required-brightgreen.svg)
<!-- preprint/venue badge — add once the RoR paper is posted/accepted:
     [![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX) -->

> Reproducibility companion for the paper **“Resample or Reroute? Budget-Aware
> Test-Time Model Selection for Large Language Models”** (preprint forthcoming).
> **Open-weight, CPU-only, no API key** — every experiment is an offline replay
> on precomputed multi-draw correctness tensors: no model inference, no closed
> endpoint.
>
> **The question in one line.** Given a fixed per-query cost budget and an
> *imperfect* verifier — the conditions a real serving system actually faces —
> should the system **resample** the model it already committed to, or
> **reroute** to a different (possibly costlier) one? This repo measures how a
> policy that decides this *per query* compares against the standard baselines,
> and is honest about where the advantage holds and where it doesn't.

## TL;DR
- **RoR** spends each unit of budget on whichever action — one more draw of the
  committed model, or the first draw of a new one — has the highest **estimated
  marginal correctness per unit cost**.
- Across **four benchmarks** (GSM8K, MATH-500, GPQA-Diamond, HumanEval+) with an
  **11-model open-weight pool**, RoR traces a **favorable cost–quality Pareto
  front** against single-route, one-commit-router, budget-aware best-of-$K$,
  cascade, and random allocation.
- The win is **regime-dependent** and **verifier-gated** — and we mark exactly
  where it shrinks, inverts, or is matched by simpler baselines (see *Results*).

## The question — why this exists (前因)
Model **routing** sends each query to the cheapest model that can answer it,
motivated by the large reported gap between a deployed router and a per-instance
*oracle* that, in hindsight, always picks a correct model. The companion
analysis ([arXiv:2607.03436](https://arxiv.org/abs/2607.03436)) shows two things:
(1) part of that gap is **single-draw label noise** that no single-commit router
can capture; and (2) the *recoverable* part can be reached **without any router**
— by test-time **resampling** (best-of-$K$ on one model) — but only under an
idealized oracle equipped with correctness labels and an unbounded budget.

A deployed system has neither. It has a **fixed per-query budget** and an
**imperfect verifier**. That leaves the operational choice this repo studies:
spend the next unit of budget **resampling** the committed model, or
**rerouting** to another?

## The policy — how RoR decides (方法)
For the current query, RoR keeps a running estimate of each model's success
probability, $\hat p = (s\bar p + w)/(s + n)$ (prior mean $\bar p$ from offline
calibration; $w$ verified-correct out of $n$ draws so far), and each step takes
the affordable action that maximizes $\hat p / \text{cost}$ — resample a used
model, or try a new one. Both actions are scored on **one axis**, which is what
makes them competing uses of a single budget. The behavior is grounded in the
**recoverability asymmetry** the companion proves (a UCB variant and a
non-deployable oracle-allocation ceiling are included for reference).

## Results
Accuracy at a matched mid budget (point nearest mean cost 26; cost =
parameter-size proxy; `acc` shown, with RoR's mean cost where informative):

| Policy | GSM8K | MATH-500 | GPQA | HumanEval+ |
|---|---|---|---|---|
| **Resample-or-Reroute (ours)** | **0.993** @9.2 | **0.887** @26.4 | **0.968** @25.9 | 0.952 @20.5 |
| budget-aware best-of-$K$ | 0.983 | 0.867 | 0.861 | 0.852 |
| cascade (FrugalGPT-style) | 0.992 | 0.847 | 0.926 | 0.952 |
| random allocation | 0.992 | 0.846 | 0.706 | 0.959 |
| single-route (best model) | 0.966 @32 | 0.776 @32 | 0.551 @8 | 0.858 @32 |
| oracle allocation (ceiling) | 1.000 | 0.944 | 1.000 | 0.988 |

**1 — A favorable Pareto front.** RoR dominates or matches every budget-scalable
baseline and overtakes the single-commit baselines once the budget affords a
second draw.

**2 — Where it wins depends on the pool.** On **saturated** GSM8K the win is
*cost*: 0.993 at 24–34% lower cost than cascade / best-of-$K$, and 3.5× cheaper
than single-routing the best model. On **heterogeneous** GPQA-Diamond, where the
pool's specialists genuinely differ, *rerouting* matters most — **+10.7 accuracy
points** over best-of-$K$ at matched cost. On **code** (HumanEval+) it matches
the best single model's fixed-budget accuracy at **~3.2× lower cost** in the
low-budget regime.

**3 — The gains are verifier-gated (the honest part).** They shrink as the
verifier degrades, and can **invert** on low-signal verifiers — e.g. self-
consistency on 4-way multiple choice (GPQA), where two wrong draws easily agree.
Where a real *partial* verifier exists — base-test suites for code, measured
false-accept ~1% — RoR stays essentially at its perfect-verifier ceiling. And on
near-saturated benchmarks under a reliable verifier, undirected baselines
(cascade, random) **converge to RoR at high budget** (see the HumanEval+ column
above): RoR's edge lives in the cost-constrained middle, not in a high-budget
ceiling. Robustness replays under a real provider price vector and a label-free
agreement verifier delineate where these conclusions carry over.

## Data
Replays read the correctness tensors produced by the sibling repo
**[routing-oracle-experiment](https://github.com/luka-krixvon/routing-oracle-experiment)**:
`artifacts/<bench>/correctness_slim.npz` for `bench ∈ {gsm8k, math500, gpqa, humanevalplus}`
(each an `(N, M, k)` int8 tensor: N queries × M=11 models × k=30 seed-aligned draws at T=0.2).
By default `data.py` looks in `../../routing-oracle/routing-oracle-experiment/artifacts`;
clone the two repos as siblings, or pass `--root <path>`.

## Generation environment
The correctness tensors were produced by the sibling repo's protocol — serving
the 11-model pool under `vLLM` at `T=0.2` on the hardware and NVIDIA CUDA stack
below (one model resident at a time; weights evicted between models). **The RoR
replay in _this_ repo is CPU-only** and needs none of it. Values are the audited
runtime recorded by `scripts/detect_environment.py` in the sibling repo.

| Component | Value |
|---|---|
| CPU | AMD EPYC 7J13, 24 vCPUs |
| RAM | 64 GB |
| GPU | 2× NVIDIA GeForce RTX 4090, 24 GB each (Ada Lovelace, cc 8.9) |
| OS | Ubuntu 24.04 LTS (kernel 6.8) |
| NVIDIA driver | 580.159.03 |
| CUDA toolkit / runtime | 13.0 (NVCC 13.2, NVRTC 13.0) |
| NVIDIA math libraries | cuBLAS 13.1, cuDNN 9.19, cuSPARSELt 0.8 |
| NVIDIA communication | NCCL 2.28.9, NVSHMEM 3.4.5 |
| CUDA attention kernels | CUTLASS 4.5, FlashInfer 0.6.12 (vLLM backend) |
| Framework | PyTorch 2.11.0, vLLM 0.23.0, Transformers 5.12.1 |

<sub>Also installed as CUDA-13 build dependencies (via `pip`, pulled by the PyTorch/vLLM build): cuFFT, cuRAND, cuSOLVER, cuSPARSE, cuPTI, nvJitLink, NVVM, NVTX, cuFile, nvidia-ml-py.</sub>

## Run
```bash
python run_pareto.py --bench humanevalplus            # cost–quality Pareto (verifier q via --quality 1.0/0.8/0.6)
python run_ablations.py                               # prior-strength / split sensitivity + ordering stability
python run_real_costs.py                              # replay under real OpenRouter output prices
python run_real_verifier.py                           # deployable verifier: agreement / self-consistency
python run_real_verifier_code.py                      # deployable verifier for code: base HumanEval tests
python run_latency.py                                 # sequential round-trip (latency) proxy
python make_paper_assets.py                           # regenerate the paper's main tables + figures
```
Outputs land in `results/*.csv` and `figures/*.pdf` (git-ignored); the LaTeX
tables/figures are written into the paper source tree.

## Repository layout
```
data.py                    # load the multi-draw correctness tensors (from the sibling repo)
policies.py                # RoR (greedy + UCB) and baselines: best-of-K, cascade, random, single-route, learned router, oracle
verifier.py                # parametric verifier model used by the replay
run_pareto.py              # cost–quality Pareto fronts
run_ablations.py           # sensitivity (prior s, train/test split) + stability
run_real_costs.py          # real-price ($/token) replay
run_real_verifier.py       # agreement-based (self-consistency) verifier replay
run_real_verifier_code.py  # base-test-suite verifier replay (code)
run_latency.py             # mean sequential rounds (round-trips) per query
make_paper_assets.py       # tables + figures for the paper
gen_code_tensor.py         # (helper) code-benchmark generation, GPU — mirrors the sibling repo's runner
```

## Map to the paper
| Paper element | Where |
|---|---|
| RoR policy + belief update ($\hat p=(s\bar p+w)/(s+n)$) | `policies.py` (`resample_or_reroute`) |
| Cost–quality Pareto (Table + Fig.) | `run_pareto.py`, `make_paper_assets.py` |
| Verifier-quality ablation (q=1.0/0.8/0.6) | `run_pareto.py --quality`, `make_paper_assets.py` |
| Sensitivity / stability | `run_ablations.py` |
| Real-price replay | `run_real_costs.py` |
| Real verifier — agreement (exact-match) / base tests (code) | `run_real_verifier.py` / `run_real_verifier_code.py` |
| Latency (round-trips) | `run_latency.py` |

## Citation
See [`CITATION.cff`](CITATION.cff). The RoR preprint is forthcoming; meanwhile
please cite the companion analysis this builds on:

```bibtex
@article{chen2026routinggap,
  title   = {How Much of the Routing Gap Is Real? Decomposing the Router-to-Oracle
             Gap into Reproducible Specialist Advantage and Single-Draw Label Noise},
  author  = {Chen, Teng-Ruei},
  journal = {arXiv preprint arXiv:2607.03436},
  year    = {2026},
  url     = {https://arxiv.org/abs/2607.03436}
}
```

## License
[MIT](LICENSE). © 2026 Teng-Ruei Chen, Chun-Cheng Lin.
