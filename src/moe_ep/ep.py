"""Expert parallelism: dispatch -> all-to-all -> expert compute -> all-to-all -> combine.

With EP = R ranks, rank r owns experts [r*E/R, (r+1)*E/R) and the tokens of its own
micro-batch. One MoE layer then needs two all-to-alls:

  1. dispatch: every rank sorts its (token, expert) slots by expert id. Because
     experts are placed contiguously, "sorted by expert" == "sorted by destination
     rank", so the send buffer is already split into R contiguous chunks.
  2. expert compute on the rows received (a grouped GEMM over the local experts).
  3. combine: send the outputs back along the reverse route (send/recv splits swap)
     and do the weighted scatter-add onto the source tokens.

The same rank-level code runs in two settings:
  * `ep_forward_rank` inside real processes with torch.distributed (gloo on CPU),
  * `simulate_ep`, R "virtual ranks" in one process, where the all-to-all is a list
    transpose. That lets us check correctness and count bytes on any device.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.distributed as dist

from moe_ep.dispatch import Permuted, permute, unpermute
from moe_ep.experts import GROUPED_IMPLS, Backend
from moe_ep.layer import MoELayer

# (tensor grouped by destination, rows per destination) -> (received tensor, rows per source)
AllToAll = Callable[[torch.Tensor, list[int]], tuple[torch.Tensor, list[int]]]


def gloo_all_to_all(t: torch.Tensor, send_splits: list[int]) -> tuple[torch.Tensor, list[int]]:
    """Variable-size all-to-all. First exchange the counts (a tiny fixed-size
    all-to-all) so every rank can allocate its receive buffer; production kernels
    (DeepEP) do this count exchange on the GPU to avoid the host round-trip."""
    send_cnt = torch.tensor(send_splits, dtype=torch.int64)
    recv_cnt = torch.empty_like(send_cnt)
    dist.all_to_all_single(recv_cnt, send_cnt)
    recv_splits = recv_cnt.tolist()
    out = t.new_empty((sum(recv_splits), *t.shape[1:]))
    dist.all_to_all_single(out, t.contiguous(), recv_splits, send_splits)
    return out, recv_splits


def expert_slice(rank: int, world: int, n_experts: int) -> tuple[int, int]:
    per = n_experts // world
    return rank * per, (rank + 1) * per


def local_expert_compute(layer: MoELayer, rows: torch.Tensor, global_eid: torch.Tensor,
                         rank: int, world: int, backend: Backend) -> torch.Tensor:
    """Rows arrive grouped by SOURCE rank (and by expert inside each source chunk),
    so we re-sort them by local expert before the grouped GEMM and undo it after."""
    e0, e1 = expert_slice(rank, world, layer.cfg.n_experts)
    local = global_eid - e0
    order = torch.argsort(local, stable=True)
    counts = torch.bincount(local, minlength=e1 - e0)
    w1, w2 = layer.experts.w_gate_up[e0:e1], layer.experts.w_down[e0:e1]
    y_sorted = GROUPED_IMPLS[backend](rows[order], counts, w1, w2)
    y = torch.empty_like(y_sorted)
    y[order] = y_sorted
    return y


def plan_dispatch(layer: MoELayer, x: torch.Tensor, world: int) -> tuple[Permuted, list[int]]:
    r = layer.router(x)
    p = permute(x, r.topk_idx, r.topk_weight, layer.cfg.n_experts)  # EP here is dropless
    return p, p.counts.reshape(world, -1).sum(1).tolist()


@torch.no_grad()
def ep_forward_rank(layer: MoELayer, x: torch.Tensor, rank: int, world: int,
                    a2a: AllToAll, backend: Backend = "sorted_loop") -> torch.Tensor:
    """What ONE rank executes. `layer` holds the full weights for simplicity, but the
    rank only ever touches its own expert slice (plus the replicated router/shared)."""
    p, send_splits = plan_dispatch(layer, x, world)
    recv_rows, recv_splits = a2a(p.rows, send_splits)          # dispatch
    recv_eid, _ = a2a(p.expert_idx, send_splits)                # metadata rides along
    y_recv = local_expert_compute(layer, recv_rows, recv_eid, rank, world, backend)
    y_back, _ = a2a(y_recv, recv_splits)                        # combine (reverse route)
    y = unpermute(y_back, p, x.shape[0])
    if layer.shared is not None:
        y = y + layer.shared(x)  # in production this overlaps with the all-to-all
    return y


def _list_all_to_all(tensors: list[torch.Tensor], splits: list[list[int]]) -> tuple[list[torch.Tensor], list[list[int]]]:
    world = len(tensors)
    chunks = [t.split(s) for t, s in zip(tensors, splits)]
    recv = [torch.cat([chunks[src][dst] for src in range(world)]) for dst in range(world)]
    recv_splits = [[splits[src][dst] for src in range(world)] for dst in range(world)]
    return recv, recv_splits


@torch.no_grad()
def simulate_ep(layer: MoELayer, x: torch.Tensor, world: int,
                backend: Backend = "sorted_loop") -> tuple[torch.Tensor, torch.Tensor]:
    """Run EP with `world` virtual ranks in one process. Tokens are split evenly
    across ranks (data parallel attention). Returns (output [T, H], rows matrix
    [src, dst] of the dispatch all-to-all)."""
    xs = x.chunk(world)
    plans = [plan_dispatch(layer, xr, world) for xr in xs]
    splits = [s for _, s in plans]
    recv_rows, recv_splits = _list_all_to_all([p.rows for p, _ in plans], splits)
    recv_eid, _ = _list_all_to_all([p.expert_idx for p, _ in plans], splits)
    y_recv = [local_expert_compute(layer, recv_rows[r], recv_eid[r], r, world, backend) for r in range(world)]
    y_back, _ = _list_all_to_all(y_recv, recv_splits)
    outs = []
    for r, (p, _) in enumerate(plans):
        y = unpermute(y_back[r], p, xs[r].shape[0])
        outs.append(y + layer.shared(xs[r]) if layer.shared is not None else y)
    return torch.cat(outs), torch.tensor(splits)
