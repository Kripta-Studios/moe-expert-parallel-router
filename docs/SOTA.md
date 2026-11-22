# State of the art: MoE routing and expert parallelism (as of late 2026)

These are the sources this repo relies on. I read each one (or its abstract or README) while writing this. Numbers are quoted from the source. I have left out anything I could not verify.

## Models: how routing is done today

| Model | Routed experts / active | Shared | Score | Balancing | Source |
|---|---|---|---|---|---|
| DeepSeek-V3 (671B total, 37B active) | 256 / 8, node-limited to 4 nodes | 1 | sigmoid, top-k weights renormalised, scale 2.5 | **auxiliary-loss-free bias** plus a tiny sequence-wise aux loss | [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) |
| Qwen3-30B-A3B / 235B-A22B | 128 / 8 | 0 (Qwen3 removed the shared experts that Qwen2.5-MoE had) | softmax | **global-batch** load-balancing loss | [arXiv:2505.09388](https://arxiv.org/abs/2505.09388) |

- **Auxiliary-loss-free load balancing**, Wang, Gao, Zhao, Sun, Dai (DeepSeek-AI), 2024, [arXiv:2408.15664](https://arxiv.org/abs/2408.15664). Before top-k selection, an expert-wise bias is added to the routing scores. After every step that bias is nudged according to recent load. Because the bias only affects *which* experts are chosen and not the gate value, no interference gradient reaches the LM loss. The paper reports better perplexity and better balance than aux-loss baselines on 1B and 3B MoEs trained on 100B and 300B tokens. DeepSeek-V3 adopts it at 671B scale. `router.py` implements this rule.
- A later theoretical treatment of the same bias rule is in [arXiv:2512.03915](https://arxiv.org/abs/2512.03915). I only saw it in search results and did not use it beyond noting that it exists.

## Kernels: how expert compute is laid out

- **MegaBlocks**, Gale, Narayanan, Young, Zaharia, MLSys 2023, [arXiv:2211.15841](https://arxiv.org/abs/2211.15841). It recasts MoE compute as block-sparse matrix products, so experts can take variable-size groups. That makes "dropless" MoE possible, with no capacity factor, no dropped tokens and no padding. It reports up to 40% end-to-end training speed-up over Tutel. Code: [databricks/megablocks](https://github.com/databricks/megablocks). `padded_bmm` vs `grouped_mm` / `triton` in `experts.py` shows the padding-vs-grouping trade-off it addresses.
- Grouped GEMM today: PyTorch exposes `torch._grouped_mm` (variable-size groups with an `offs` tensor). It works on this machine (sm_120, torch 2.11) and is the `grouped_mm` backend here. DeepSeek's DeepGEMM provides FP8 grouped GEMMs for the same "contiguous" (prefill) and "masked" (decode) layouts. It is used in the SGLang deployment below.

## Communication: dispatch / combine

- **DeepEP** ([deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP)) is the reference all-to-all library for MoE. It has *normal* kernels for prefill and training, which forward NVLink to RDMA asymmetrically and support FP8 dispatch. It also has *low-latency* kernels for decode, which are pure RDMA and use a hook-based overlap that occupies no SMs. The v1.x README reports these figures on H800 (about 160 GB/s NVLink) with ConnectX-7 400 Gb/s (about 50 GB/s):
  - normal kernels (4096 tokens, hidden 7168, top-4 groups, top-8 experts, FP8 dispatch / BF16 combine): intranode EP8 153 / 158 GB/s (NVLink); internode EP16 43 / 43, EP32 58 / 57, EP64 51 / 50 GB/s (RDMA) for dispatch / combine.
  - low-latency kernels (128 tokens, hidden 7168, top-8): EP64 dispatch **173 us** (43 GB/s), combine **314 us** (46 GB/s). At EP8 the figures are 77 / 114 us. At EP256 they are 194 / 360 us.
  - The current README describes a V2 / V2.5 rewrite: an NCCL-based device-side backend replaces NVSHMEM, kernels are JIT-compiled, high-throughput and low-latency paths are merged into one `EPBuffer` API, and V1 has been removed. The benchmark tables above come from the [v1.2.1 README](https://github.com/deepseek-ai/DeepEP/blob/v1.2.1/README.md), which also lists EP16/32/128 low-latency rows (118/195, 155/273, 192/369 us).
  `cost_model.py` uses these achieved bandwidths and is calibrated against the EP64 low-latency figures.
- **Node-limited routing** (DeepSeek-V3 §2.1.2 / §3.2): each token goes to at most M=4 nodes. It crosses InfiniBand once per node and is then forwarded over NVLink. That bounds IB traffic, which is roughly 3.2× slower than NVLink per GPU. `rows_per_token()` quantifies this.

## Load balancing at serving time

- **EPLB** ([deepseek-ai/EPLB](https://github.com/deepseek-ai/EPLB)) is the "redundant experts" strategy from the DeepSeek-V3 paper. It replicates heavy experts, then packs the replicas onto GPUs with greedy heuristics. It has a *hierarchical* policy (node groups first, for prefill with smaller EP) and a *global* policy (for decode with large EP). `cost_model.eplb_greedy` is a simplified version of the global policy.
- NVIDIA's TensorRT-LLM large-scale EP blog ([Scaling Expert Parallelism in TensorRT-LLM, Part 1](https://nvidia.github.io/TensorRT-LLM/1.1.0rc2/blogs/tech_blog/blog4_Scaling_Expert_Parallelism_in_TensorRT-LLM.html)) observes that per-rank imbalance stays fairly stable over tens to hundreds of iterations. It argues for an *online* balancer.

## Overlap: dual-batch / two-batch overlap

- **DeepSeek profile-data** ([deepseek-ai/profile-data](https://github.com/deepseek-ai/profile-data)) publishes PyTorch-profiler traces. Prefill (EP32, TP1, 4K prompts, 16K tokens/GPU) splits the batch into two micro-batches so that one computes while the other is in all-to-all. Decode does the same with non-SM-occupying all-to-all. Training uses DualPipe.
- **SGLang**, [Large-scale EP on 96 H100 (LMSYS, May 2025)](https://lmsys.org/blog/2025-05-05-large-scale-ep/): prefill/decode disaggregation with DeepEP, DeepGEMM, EPLB and two-batch overlap (`--enable-two-batch-overlap`). It reports 52.3k input and 22.3k output tok/s per node for 2000-token inputs. The [GB200 NVL72 follow-ups](https://lmsys.org/blog/2025-09-25-gb200-part-2) report that on NVL72's much faster all-NVLink fabric, two-batch overlap matters less and they move to finer-grained overlap. `model_a2a.py` reproduces this qualitatively: the "single NVLink domain" curve has comm far below compute.

## What this repo does *not* cover
FP8 dispatch kernels, NVSHMEM/IBGDA, DualPipe for training, and expert-sharded/TP hybrids. It also does not cover routing-collapse and quality effects of balancing on real training runs (no training here).
