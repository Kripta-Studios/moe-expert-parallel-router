"""moe_ep: a didactic Mixture-of-Experts layer with expert parallelism.

Modules
-------
config     MoEConfig dataclass (DeepSeek-V3 / Qwen3-MoE style hyper-parameters)
router     top-k gating, group-limited routing, aux-loss-free bias balancing
dispatch   permute / unpermute tokens, capacity factor and token dropping
experts    SwiGLU expert bank with several compute back-ends (loop, grouped GEMM)
layer      MoELayer = router + routed experts + shared expert
metrics    load-balance statistics and communication-volume accounting
ep         expert parallelism: dispatch -> all-to-all -> expert compute -> combine
cost_model analytic all-to-all model (NVLink vs InfiniBand) for DeepSeek-V3 scale
"""

from moe_ep.config import MoEConfig
from moe_ep.layer import MoELayer

__all__ = ["MoEConfig", "MoELayer"]
