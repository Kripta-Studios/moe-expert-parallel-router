"""Token permutation ("dispatch") and its inverse ("combine") on one device.

After routing, every token t has k (expert, weight) pairs. Expert compute wants the
opposite view: for every expert e, the contiguous block of rows routed to it. We get
it by flattening the T*k slots, stable-sorting them by expert id and gathering the
token rows in that order. The per-expert row counts become the `offsets` of a
grouped GEMM. Combining is the transpose: scale each output row by its gate weight
and scatter-add it back onto its source token.

Capacity factor: fixed-shape systems (GShard, Switch, early TPU MoE) give each
expert a buffer of C = ceil(cf * T * k / E) rows. Slots beyond C are *dropped*:
the token simply loses that expert's contribution (the residual stream and the
shared expert still carry it). Dropless systems (MegaBlocks, DeepSeek-V3, Qwen3)
use cf=None and variable-size groups instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class Permuted:
    rows: torch.Tensor        # [N, H]  token rows grouped by expert (N <= T*k after drops)
    token_idx: torch.Tensor   # [N]     source token of each row
    weight: torch.Tensor      # [N]     gate weight of each row
    expert_idx: torch.Tensor  # [N]     expert of each row (non-decreasing)
    counts: torch.Tensor      # [E]     rows per expert after dropping (int64)
    routed_counts: torch.Tensor  # [E]  rows per expert BEFORE dropping (the true demand)

    @property
    def n_dropped(self) -> int:
        return int((self.routed_counts - self.counts).sum())


def capacity(n_tokens: int, top_k: int, n_experts: int, cf: float) -> int:
    return math.ceil(cf * n_tokens * top_k / n_experts)


def permute(x: torch.Tensor, topk_idx: torch.Tensor, topk_weight: torch.Tensor,
            n_experts: int, cap: int | None = None) -> Permuted:
    t, k = topk_idx.shape
    flat_e = topk_idx.reshape(-1)
    # Stable sort keeps token order inside an expert, so dropping keeps the EARLIEST
    # tokens (GShard-style positional priority). Production kernels compute the same
    # permutation with a histogram + prefix-sum instead of a full sort.
    order = torch.argsort(flat_e, stable=True)
    sorted_e = flat_e[order]
    routed_counts = torch.bincount(flat_e, minlength=n_experts)
    counts = routed_counts
    if cap is not None:
        starts = torch.cumsum(routed_counts, 0) - routed_counts
        pos_in_expert = torch.arange(order.numel(), device=x.device) - starts[sorted_e]
        keep = pos_in_expert < cap
        order, sorted_e = order[keep], sorted_e[keep]
        counts = routed_counts.clamp(max=cap)
    token_idx = order // k
    return Permuted(rows=x[token_idx], token_idx=token_idx,
                    weight=topk_weight.reshape(-1)[order], expert_idx=sorted_e,
                    counts=counts, routed_counts=routed_counts)


def unpermute(y_rows: torch.Tensor, p: Permuted, n_tokens: int) -> torch.Tensor:
    out = torch.zeros(n_tokens, y_rows.shape[-1], device=y_rows.device, dtype=torch.float32)
    # Accumulate in fp32: a token sums k bf16 contributions.
    out.index_add_(0, p.token_idx, y_rows.float() * p.weight.float()[:, None])
    return out.to(y_rows.dtype)
