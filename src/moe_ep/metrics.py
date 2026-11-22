"""Load-balance statistics and communication-volume accounting.

Why max/mean matters more than CV: under expert parallelism every rank waits for
the slowest one at the next all-to-all, so step time ~ max_rank_load and the
efficiency of the whole EP group is mean/max. CV (std/mean) summarises the spread
of the whole distribution and is what most papers plot.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def load_stats(counts: torch.Tensor) -> dict[str, float]:
    c = counts.double().cpu()
    mean = c.mean().item()
    return {
        "max_over_mean": (c.max().item() / mean) if mean > 0 else 0.0,
        "cv": (c.std(unbiased=False).item() / mean) if mean > 0 else 0.0,
        "min_over_mean": (c.min().item() / mean) if mean > 0 else 0.0,
        "idle_frac": (c == 0).double().mean().item(),
    }


def rank_loads(expert_counts: torch.Tensor, world: int) -> torch.Tensor:
    """Contiguous placement: rank r owns experts [r*E/R, (r+1)*E/R)."""
    return expert_counts.reshape(world, -1).sum(1)


@dataclass
class CommVolume:
    """Byte matrices [src_rank, dst_rank] for ONE direction (dispatch). Combine moves
    the same rows backwards (with its own element size)."""
    per_slot: torch.Tensor   # one copy per (token, expert) pair: plain all-to-all-v
    dedup: torch.Tensor      # one copy per (token, destination rank): DeepEP-style

    @staticmethod
    def _offdiag(m: torch.Tensor) -> int:
        return int(m.sum() - m.diagonal().sum())  # the diagonal stays on-device

    @property
    def network_bytes_per_slot(self) -> int:
        return self._offdiag(self.per_slot)

    @property
    def network_bytes_dedup(self) -> int:
        return self._offdiag(self.dedup)


def comm_volume(topk_idx_per_rank: list[torch.Tensor], n_experts: int, bytes_per_row: int) -> CommVolume:
    """topk_idx_per_rank[r]: [T_r, k] routing of the tokens that live on rank r."""
    world = len(topk_idx_per_rank)
    per_rank = n_experts // world
    slot = torch.zeros(world, world, dtype=torch.int64)
    dedup = torch.zeros(world, world, dtype=torch.int64)
    for src, idx in enumerate(topk_idx_per_rank):
        dst = (idx // per_rank).cpu()  # [T, k] destination rank of each slot
        slot[src] = torch.bincount(dst.flatten(), minlength=world)
        hit = torch.zeros(dst.shape[0], world, dtype=torch.bool).scatter_(1, dst, True)
        dedup[src] = hit.sum(0)
    return CommVolume(slot * bytes_per_row, dedup * bytes_per_row)
