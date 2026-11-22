"""MoELayer = router + routed experts + shared expert (DeepSeek-V3 layout).

    y = SharedFFN(x) + sum_{e in topk(x)} g_e(x) * Expert_e(x)

The shared expert sees every token, so it can hold common knowledge while the
routed experts specialise. It is a plain dense MLP: under expert parallelism it is
replicated on every rank, and its compute can hide part of the all-to-all.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from moe_ep.config import MoEConfig
from moe_ep.dispatch import capacity, permute, unpermute
from moe_ep.experts import Backend, ExpertBank, swiglu
from moe_ep.router import RouterOutput, TopKRouter


class SharedExpert(nn.Module):
    def __init__(self, cfg: MoEConfig) -> None:
        super().__init__()
        inter = cfg.n_shared * cfg.expert_inter
        self.gate_up = nn.Linear(cfg.hidden, 2 * inter, bias=False)
        self.down = nn.Linear(inter, cfg.hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(swiglu(self.gate_up(x)))


@dataclass
class MoEStats:
    routed_counts: torch.Tensor  # [E] demand per expert
    kept_counts: torch.Tensor    # [E] after capacity dropping
    router: RouterOutput


class MoELayer(nn.Module):
    def __init__(self, cfg: MoEConfig, backend: Backend = "grouped_mm") -> None:
        super().__init__()
        self.cfg = cfg
        self.backend: Backend = backend
        self.router = TopKRouter(cfg)
        self.experts = ExpertBank(cfg)
        self.shared = SharedExpert(cfg) if cfg.n_shared > 0 else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, MoEStats]:
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        r = self.router(x)
        if self.backend == "naive":
            assert self.cfg.capacity_factor is None, "the naive path is dropless"
            y = self.experts.forward_naive(x, r.topk_idx, r.topk_weight)
            routed = kept = torch.bincount(r.topk_idx.flatten(), minlength=self.cfg.n_experts)
        else:
            cap = None
            if self.cfg.capacity_factor is not None:
                cap = capacity(x.shape[0], self.cfg.top_k, self.cfg.n_experts, self.cfg.capacity_factor)
            p = permute(x, r.topk_idx, r.topk_weight, self.cfg.n_experts, cap)
            y = unpermute(self.experts.forward_grouped(p.rows, p.counts, self.backend), p, x.shape[0])
            routed, kept = p.routed_counts, p.counts
        if self.shared is not None:
            y = y + self.shared(x)
        return y.reshape(shape), MoEStats(routed, kept, r)


def reference_moe(layer: MoELayer, x: torch.Tensor) -> torch.Tensor:
    """Dense, obviously-correct oracle for tests: run EVERY expert on EVERY token in
    fp32 and mask by the routing table. O(T*E) work, only for tiny CPU shapes."""
    x32 = x.float()
    r = layer.router(x32)
    w1, w2 = layer.experts.w_gate_up.float(), layer.experts.w_down.float()
    all_y = torch.einsum("tei,eih->teh", swiglu(torch.einsum("th,ehi->tei", x32, w1)), w2)  # [T, E, H]
    gates = torch.zeros(x.shape[0], layer.cfg.n_experts, device=x.device)
    gates.scatter_(1, r.topk_idx, r.topk_weight.float())
    y = (gates[..., None] * all_y).sum(1)
    if layer.shared is not None:
        s = layer.shared
        y = y + F.linear(swiglu(F.linear(x32, s.gate_up.weight.float())), s.down.weight.float())
    return y
