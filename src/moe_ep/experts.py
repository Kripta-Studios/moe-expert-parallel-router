"""SwiGLU expert bank and the compute back-ends we compare.

Every routed expert is a small SwiGLU MLP:  y = (silu(x W_g) * (x W_u)) W_d.
Weights are stored *stacked* ([E, H, 2I] and [E, I, H]) so a grouped GEMM can
index them by expert id.

Back-ends (all numerically equivalent, they differ only in how work is laid out):

  naive        HF-Mixtral-style loop: for each expert, find its tokens with a mask
               (torch.where over the [T, k] routing table), gather, 2 GEMMs.
               E host syncs and 4E+ kernel launches of tiny GEMMs that starve the GPU.
  sorted_loop  permute once (dispatch.permute), then loop over contiguous slices.
               Removes the per-expert masking but still launches E small GEMMs.
  padded_bmm   pad every group to the largest count and run one batched matmul.
               Few launches, but wastes FLOPs on padding when load is skewed
               (this is what a capacity-factor buffer looks like in memory).
  grouped_mm   torch._grouped_mm: PyTorch's variable-size grouped GEMM (a fused CUTLASS
               kernel on sm_90; on sm_120 / torch 2.11 the profiler shows one GEMM per group).
  triton       our own grouped GEMM (triton_grouped_gemm.py).
"""

from __future__ import annotations

from typing import Callable, Literal

import torch
from torch import nn
from torch.nn import functional as F

from moe_ep.config import MoEConfig

Backend = Literal["naive", "sorted_loop", "padded_bmm", "grouped_mm", "triton"]
BACKENDS: tuple[Backend, ...] = ("naive", "sorted_loop", "padded_bmm", "grouped_mm", "triton")


def swiglu(h: torch.Tensor) -> torch.Tensor:
    g, u = h.chunk(2, dim=-1)
    return F.silu(g) * u


class ExpertBank(nn.Module):
    def __init__(self, cfg: MoEConfig, n_experts: int | None = None) -> None:
        super().__init__()
        e = n_experts if n_experts is not None else cfg.n_experts
        h, i = cfg.hidden, cfg.expert_inter
        self.w_gate_up = nn.Parameter(torch.randn(e, h, 2 * i) * h ** -0.5)
        self.w_down = nn.Parameter(torch.randn(e, i, h) * i ** -0.5)

    @property
    def n_experts(self) -> int:
        return self.w_gate_up.shape[0]

    def forward_grouped(self, rows: torch.Tensor, counts: torch.Tensor, backend: Backend) -> torch.Tensor:
        """rows: [N, H] grouped by expert (contiguous), counts: [E]. Returns [N, H]."""
        return GROUPED_IMPLS[backend](rows, counts, self.w_gate_up, self.w_down)

    def forward_naive(self, x: torch.Tensor, topk_idx: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        """The un-permuted baseline: works directly on the [T, k] routing table."""
        out = torch.zeros(x.shape[0], x.shape[1], device=x.device, dtype=torch.float32)
        for e in range(self.n_experts):
            tok, slot = torch.where(topk_idx == e)  # host sync per expert
            if tok.numel() == 0:
                continue
            y = swiglu(x[tok] @ self.w_gate_up[e]) @ self.w_down[e]
            out.index_add_(0, tok, y.float() * topk_weight[tok, slot, None].float())
        return out.to(x.dtype)


GroupedFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def _sorted_loop(rows: torch.Tensor, counts: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(rows)
    start = 0
    for e, c in enumerate(counts.tolist()):  # one host sync, then E slices
        if c:
            out[start:start + c] = swiglu(rows[start:start + c] @ w1[e]) @ w2[e]
            start += c
    return out


def _padded_bmm(rows: torch.Tensor, counts: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    e, h = w1.shape[0], rows.shape[1]
    counts = counts.to(rows.device)
    cap = int(counts.max())
    starts = torch.cumsum(counts, 0) - counts
    expert_of_row = torch.repeat_interleave(torch.arange(e, device=rows.device), counts)
    pos = torch.arange(rows.shape[0], device=rows.device) - starts[expert_of_row]
    buf = rows.new_zeros(e, cap, h)
    buf[expert_of_row, pos] = rows
    y = torch.bmm(swiglu(torch.bmm(buf, w1)), w2)  # [E, cap, H]: padding is computed and thrown away
    return y[expert_of_row, pos]


def _grouped_mm(rows: torch.Tensor, counts: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    offs = torch.cumsum(counts.to(device=rows.device, dtype=torch.int32), 0, dtype=torch.int32)
    h = torch._grouped_mm(rows, w1, offs=offs)
    return torch._grouped_mm(swiglu(h), w2, offs=offs)


def _triton(rows: torch.Tensor, counts: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    from moe_ep.triton_grouped_gemm import grouped_gemm  # lazy: CPU-only users never import triton

    return grouped_gemm(swiglu(grouped_gemm(rows, w1, counts)), w2, counts)


GROUPED_IMPLS: dict[str, GroupedFn] = {
    "sorted_loop": _sorted_loop,
    "padded_bmm": _padded_bmm,
    "grouped_mm": _grouped_mm,
    "triton": _triton,
}
