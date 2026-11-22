"""Hyper-parameters of one MoE layer.

Defaults are a scaled-down DeepSeek-V3 layer. The two reference configs below are
taken from the public model configs (config.json on the Hugging Face Hub).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal


@dataclass(frozen=True)
class MoEConfig:
    hidden: int = 512             # d_model
    expert_inter: int = 256       # intermediate size of ONE routed expert (fine-grained experts are narrow)
    n_experts: int = 32           # routed experts
    top_k: int = 4                # experts activated per token
    n_shared: int = 1             # shared (always-on) experts; DeepSeek-V3 uses 1, Qwen3-MoE uses 0
    score_func: Literal["sigmoid", "softmax"] = "sigmoid"  # DeepSeek-V3: sigmoid, Qwen3: softmax
    n_groups: int = 1             # group-limited routing (DeepSeek-V3: 8 groups == 8 nodes)
    topk_groups: int = 1          # groups a token may use (DeepSeek-V3: 4)
    norm_topk: bool = True        # renormalise the k selected weights to sum to 1
    routed_scale: float = 1.0     # DeepSeek-V3 multiplies routed output by 2.5
    bias_update_speed: float = 1e-3  # gamma of the aux-loss-free bias update
    capacity_factor: float | None = None  # None = dropless; else C = ceil(cf * T * k / E)

    def __post_init__(self) -> None:
        assert self.n_experts % self.n_groups == 0, "experts must split evenly into groups"
        assert self.topk_groups <= self.n_groups
        assert self.top_k <= (self.n_experts // self.n_groups) * self.topk_groups


# Real configs, used by the analytic cost model (no weights are ever loaded).
DEEPSEEK_V3 = MoEConfig(hidden=7168, expert_inter=2048, n_experts=256, top_k=8, n_shared=1,
                        score_func="sigmoid", n_groups=8, topk_groups=4, routed_scale=2.5)
QWEN3_30B_A3B = MoEConfig(hidden=2048, expert_inter=768, n_experts=128, top_k=8, n_shared=0,
                          score_func="softmax", norm_topk=True)


def with_(cfg: MoEConfig, **kw: object) -> MoEConfig:
    """Small helper so benchmarks can derive variants of a frozen config."""
    return replace(cfg, **kw)  # type: ignore[arg-type]
