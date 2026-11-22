"""Analytic all-to-all cost model for a DeepSeek-V3-scale MoE layer, plus EPLB.

Nothing here runs on a cluster: it is arithmetic on published hardware numbers,
so every output of this module is MODELED, not measured.

Per-GPU dispatch time is modeled as  t = alpha + bytes / BW  on the bottleneck link.
Fabric bandwidths default to the *achieved* figures DeepEP reports on H800 +
ConnectX-7 (about 155 GB/s NVLink, about 50 GB/s RDMA per GPU), which already fold in
protocol overheads. alpha (fixed launch/sync cost) is an assumption, not a fit.

How many copies of a token cross the network depends on the routing structure:
  * per-slot     plain all-to-all-v: one copy per (token, expert) pair, k copies.
  * per-rank     dedup: one copy per distinct destination GPU (receiver fans out).
  * node-limited DeepSeek-V3 + DeepEP: a token may touch at most M nodes; it crosses
                 InfiniBand once per remote node, then is forwarded over NVLink to the
                 GPUs inside that node.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import torch


@dataclass(frozen=True)
class Fabric:
    name: str
    bw_gbps: float      # achievable per-GPU unidirectional GB/s
    alpha_us: float     # fixed cost per all-to-all (assumed)


NVLINK_H800 = Fabric("NVLink H800 (DeepEP-achieved)", 155.0, 10.0)
IB_CX7 = Fabric("InfiniBand CX7 400Gb/s (DeepEP-achieved)", 50.0, 10.0)
NVLINK_NVL72 = Fabric("NVLink5 NVL72 (~0.9 TB/s/dir spec, 70% assumed)", 630.0, 10.0)


@dataclass(frozen=True)
class EPSetup:
    n_experts: int = 256
    top_k: int = 8
    hidden: int = 7168
    expert_inter: int = 2048
    ep: int = 64
    gpus_per_node: int = 8
    max_nodes: int = 4            # node-limited routing M (DeepSeek-V3: 4)
    dispatch_bytes: float = 1 + 4 / 128   # FP8 + one fp32 scale per 128 elements
    combine_bytes: float = 2.0            # BF16

    @property
    def nodes(self) -> int:
        return self.ep // self.gpus_per_node


def expected_distinct(n_items: int, n_bins: int, k: int) -> float:
    """E[#bins hit] when k distinct items are drawn uniformly from n_items split
    evenly into n_bins bins: n_bins * (1 - C(n - n/b, k) / C(n, k))."""
    per = n_items // n_bins
    return n_bins * (1 - comb(n_items - per, k) / comb(n_items, k))


def rows_per_token(s: EPSetup) -> dict[str, float]:
    """Expected copies of one token's hidden vector that leave the source GPU."""
    gpus_hit = expected_distinct(s.n_experts, s.ep, s.top_k)
    remote_gpu_frac = 1 - 1 / s.ep
    # node-limited: choose M of the nodes, then k experts among those M nodes' experts
    m = min(s.max_nodes, s.nodes)
    nodes_hit = expected_distinct(s.n_experts * m // s.nodes, m, s.top_k)
    return {
        "per_slot": s.top_k * remote_gpu_frac,
        "per_rank": gpus_hit * remote_gpu_frac,
        "ib_node_limited": nodes_hit * (1 - 1 / s.nodes),  # remote nodes only
        "nvl_node_limited": gpus_hit,  # each touched GPU receives one NVLink copy (upper bound)
    }


def a2a_time_us(tokens_per_gpu: int, s: EPSetup, scenario: str, elem_bytes: float) -> float:
    """One all-to-all (dispatch if elem_bytes=s.dispatch_bytes, combine if 2.0)."""
    r = rows_per_token(s)
    row = s.hidden * elem_bytes
    if scenario == "single_nvlink_domain":  # e.g. GB200 NVL72: EP=64 all on NVLink
        return NVLINK_NVL72.alpha_us + tokens_per_gpu * r["per_rank"] * row / (NVLINK_NVL72.bw_gbps * 1e3)
    if scenario == "ib_per_slot":           # naive all-to-all over InfiniBand, k copies
        return IB_CX7.alpha_us + tokens_per_gpu * r["per_slot"] * row / (IB_CX7.bw_gbps * 1e3)
    if scenario == "ib_node_limited":       # DeepEP: IB once per node, NVLink fan-out, pipelined
        ib = tokens_per_gpu * r["ib_node_limited"] * row / (IB_CX7.bw_gbps * 1e3)
        nvl = tokens_per_gpu * r["nvl_node_limited"] * row / (NVLINK_H800.bw_gbps * 1e3)
        return IB_CX7.alpha_us + max(ib, nvl)
    raise ValueError(scenario)


SCENARIOS = ("single_nvlink_domain", "ib_per_slot", "ib_node_limited")


def expert_time_us(tokens_per_gpu: int, s: EPSetup, tflops: float = 1979.0 * 0.6,
                   hbm_tbps: float = 3.35 * 0.8, weight_bytes: float = 1.0) -> float:
    """Routed-expert compute per GPU with perfect balance: roofline max(FLOP time,
    weight-read time). Defaults: H800 dense FP8 1979 TFLOPS and 3.35 TB/s HBM, with
    assumed 60% / 80% efficiency. Decode is weight-bandwidth bound, prefill FLOP bound."""
    rows = tokens_per_gpu * s.top_k  # arriving rows per GPU on average
    flops = rows * 2 * 3 * s.hidden * s.expert_inter
    w = (s.n_experts // s.ep) * 3 * s.hidden * s.expert_inter * weight_bytes
    return max(flops / (tflops * 1e6), w / (hbm_tbps * 1e6))


def layer_time_us(tokens_per_gpu: int, s: EPSetup, scenario: str, overlap: bool) -> float:
    """Dispatch + experts + combine. With dual-batch overlap (two micro-batches, one
    computing while the other communicates) the steady state approaches
    max(comm, compute) instead of comm + compute."""
    comm = a2a_time_us(tokens_per_gpu, s, scenario, s.dispatch_bytes) + \
        a2a_time_us(tokens_per_gpu, s, scenario, s.combine_bytes)
    comp = expert_time_us(tokens_per_gpu, s)
    return max(comm, comp) if overlap else comm + comp


def eplb_greedy(load: torch.Tensor, n_slots: int, world: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Simplified EPLB (deepseek-ai/EPLB "global" policy): give the hottest experts
    extra replicas (each replica serves load/replicas), then pack replicas onto ranks
    largest-first, each into the least-loaded rank that still has a free slot.
    Returns (replicas per expert [E], load per rank [R])."""
    e = load.numel()
    assert n_slots >= e and n_slots % world == 0
    replicas = torch.ones(e)
    for _ in range(n_slots - e):
        replicas[torch.argmax(load / replicas)] += 1
    items = torch.repeat_interleave(load / replicas, replicas.long())
    items = items.sort(descending=True).values
    rank_load = torch.zeros(world)
    free = torch.full((world,), n_slots // world)
    for w in items:
        cand = torch.where(free > 0, rank_load, torch.full_like(rank_load, float("inf")))
        r = int(torch.argmin(cand))
        rank_load[r] += w
        free[r] -= 1
    return replicas, rank_load
