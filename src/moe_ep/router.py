"""Top-k gating in the DeepSeek-V3 / Qwen3-MoE style.

Two ideas matter here:

1. *Selection* and *weighting* are decoupled. DeepSeek-V3 adds a per-expert bias b_e
   to the affinity s_e ONLY when choosing the top-k experts; the gate weight that
   multiplies the expert output still uses the un-biased s_e. Therefore the bias
   steers load without ever entering the gradient of the LM loss
   ("auxiliary-loss-free" balancing, Wang et al. 2024, arXiv:2408.15664).

2. Group-limited routing. Experts are split into G groups (one per node in
   DeepSeek-V3). Each token first picks its best `topk_groups` groups (scored by the
   sum of the top-2 affinities inside the group) and then picks top-k experts only
   inside them. This caps how many nodes a token's activations must travel to,
   which bounds the expensive inter-node (InfiniBand) part of the all-to-all.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from moe_ep.config import MoEConfig


@dataclass
class RouterOutput:
    topk_idx: torch.Tensor      # [T, k] int64 expert ids
    topk_weight: torch.Tensor   # [T, k] gate weights (from un-biased scores)
    scores: torch.Tensor        # [T, E] un-biased affinities (sigmoid or softmax)


class TopKRouter(nn.Module):
    def __init__(self, cfg: MoEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.weight = nn.Parameter(torch.empty(cfg.n_experts, cfg.hidden))
        nn.init.normal_(self.weight, std=cfg.hidden ** -0.5)
        # Buffer, not Parameter: it is updated by a rule, never by the optimiser.
        self.register_buffer("expert_bias", torch.zeros(cfg.n_experts))

    def _apply(self, fn, recurse=True):  # type: ignore[no-untyped-def]
        # model.to(bfloat16) would also cast the bias; bf16 has 8 mantissa bits, so
        # thousands of +-1e-3 updates would be rounded away. Keep it in fp32.
        super()._apply(fn, recurse)
        self.expert_bias = self.expert_bias.float()
        return self

    def forward(self, x: torch.Tensor) -> RouterOutput:
        cfg = self.cfg
        logits = torch.nn.functional.linear(x.float(), self.weight.float())  # router in fp32, as in DSV3
        scores = logits.sigmoid() if cfg.score_func == "sigmoid" else logits.softmax(dim=-1)

        choice = scores + self.expert_bias  # biased scores are used ONLY for selection
        if cfg.n_groups > 1:
            choice = self._mask_groups(choice)
        topk_idx = choice.topk(cfg.top_k, dim=-1).indices
        topk_weight = scores.gather(-1, topk_idx)
        if cfg.norm_topk:
            topk_weight = topk_weight / topk_weight.sum(-1, keepdim=True).clamp_min(1e-20)
        topk_weight = topk_weight * cfg.routed_scale
        return RouterOutput(topk_idx, topk_weight.to(x.dtype), scores)

    def _mask_groups(self, choice: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        t = choice.shape[0]
        grouped = choice.view(t, cfg.n_groups, -1)
        group_score = grouped.topk(min(2, grouped.shape[-1]), dim=-1).values.sum(-1)  # [T, G]
        keep = group_score.topk(cfg.topk_groups, dim=-1).indices
        mask = torch.zeros_like(group_score, dtype=torch.bool).scatter_(1, keep, True)
        return grouped.masked_fill(~mask[..., None], float("-inf")).view(t, -1)

    @torch.no_grad()
    def update_bias(self, tokens_per_expert: torch.Tensor) -> None:
        """Aux-loss-free update (DeepSeek-V3 Sec. 2.1.2): after each step,
        b_e -= gamma if expert e is overloaded, b_e += gamma if underloaded.
        Under expert parallelism `tokens_per_expert` must be the GLOBAL count
        (all-reduced across ranks), otherwise every rank drifts differently."""
        load = tokens_per_expert.float()
        self.expert_bias += self.cfg.bias_update_speed * torch.sign(load.mean() - load)


def switch_aux_loss(scores: torch.Tensor, topk_idx: torch.Tensor, n_experts: int) -> torch.Tensor:
    """Classic auxiliary balance loss (Switch/GShard form), for comparison only:
    L = E * sum_e f_e * P_e, with f_e = fraction of routed slots that went to e and
    P_e = mean normalised affinity. It equals 1 at perfectly uniform load and grows
    with imbalance; its gradient flows into the router and competes with the LM loss."""
    t, k = topk_idx.shape
    f = torch.bincount(topk_idx.flatten(), minlength=n_experts).float() / (t * k)
    p = (scores / scores.sum(-1, keepdim=True)).mean(0)
    return n_experts * (f * p).sum()
