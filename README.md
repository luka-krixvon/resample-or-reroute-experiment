# Resample or Reroute? Budget-Aware Test-Time Model Selection for LLMs

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)
![Serving](https://img.shields.io/badge/serving-vLLM-orange.svg)
![Hardware](https://img.shields.io/badge/generation-2%C3%97%20RTX%204090-lightgrey.svg)
![Replay](https://img.shields.io/badge/replay-CPU%20only-brightgreen.svg)
![No API key](https://img.shields.io/badge/API%20key-not%20required-brightgreen.svg)
<!-- preprint/venue badge — add once the RoR paper is posted/accepted:
     [![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX) -->

> Replay code for **“Resample or Reroute? Budget-Aware Test-Time Model Selection
> for LLMs”** (preprint forthcoming). Given a per-query budget and an *imperfect*
> verifier, RoR spends each unit of budget on whichever action — **resampling** a
> committed model or **rerouting** to another — has the highest estimated marginal
> correctness per unit cost. Every experiment is an **offline CPU replay** on
> precomputed multi-draw correctness tensors: no model inference, no API key.
>
> This is the applied companion to *“How Much of the Routing Gap Is Real?”*
> ([arXiv:2607.03436](https://arxiv.org/abs/2607.03436)), which proves the
> recoverability asymmetry RoR exploits and releases the generation protocol.

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
