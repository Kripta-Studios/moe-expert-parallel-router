"""Correctness tests. CPU tests are tiny; GPU tests run only when CUDA is available."""

from __future__ import annotations

import pytest
import torch

from moe_ep.config import MoEConfig, with_
from moe_ep.dispatch import capacity, permute
from moe_ep.ep import simulate_ep
from moe_ep.experts import GROUPED_IMPLS
from moe_ep.layer import MoELayer, reference_moe
from moe_ep.metrics import comm_volume, load_stats

CFG = MoEConfig(hidden=32, expert_inter=16, n_experts=8, top_k=2, n_shared=1)
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _layer(cfg: MoEConfig = CFG, backend: str = "sorted_loop", seed: int = 0) -> MoELayer:
    torch.manual_seed(seed)
    return MoELayer(cfg, backend)  # type: ignore[arg-type]


@pytest.mark.parametrize("backend", ["naive", "sorted_loop", "padded_bmm"])
@pytest.mark.parametrize("score", ["sigmoid", "softmax"])
def test_backends_match_dense_reference(backend: str, score: str) -> None:
    layer = _layer(with_(CFG, score_func=score), backend)
    x = torch.randn(37, CFG.hidden)
    with torch.no_grad():
        y, _ = layer(x)
        torch.testing.assert_close(y, reference_moe(layer, x), rtol=1e-4, atol=1e-5)


def test_group_limited_routing_respects_groups() -> None:
    cfg = MoEConfig(hidden=32, expert_inter=16, n_experts=16, top_k=4, n_groups=4, topk_groups=2)
    layer = _layer(cfg)
    r = layer.router(torch.randn(200, 32))
    groups = r.topk_idx // (cfg.n_experts // cfg.n_groups)
    assert all(len(set(g.tolist())) <= cfg.topk_groups for g in groups)


def test_bias_steers_selection_but_not_weights() -> None:
    layer = _layer()
    x = torch.randn(64, CFG.hidden)
    layer.router.expert_bias[3] = 10.0  # force expert 3 into every token's top-k
    r = layer.router(x)
    assert (r.topk_idx == 3).any(-1).all()
    raw = r.scores.gather(-1, r.topk_idx)
    torch.testing.assert_close(r.topk_weight, raw / raw.sum(-1, keepdim=True))


def test_bias_update_reduces_imbalance() -> None:
    cfg = with_(CFG, n_experts=16, top_k=2, bias_update_speed=0.01)
    layer = _layer(cfg)
    x = torch.randn(512, cfg.hidden)
    with torch.no_grad():
        x = x + 8.0 * layer.router.weight[-1]  # tokens lean towards the last expert: skewed load
    first = load_stats(torch.bincount(layer.router(x).topk_idx.flatten(), minlength=16))["max_over_mean"]
    for _ in range(300):
        layer.router.update_bias(torch.bincount(layer.router(x).topk_idx.flatten(), minlength=16))
    last = load_stats(torch.bincount(layer.router(x).topk_idx.flatten(), minlength=16))["max_over_mean"]
    assert first > 4 and last < 1.5  # 8x -> ~1.3x; the sign rule keeps a small oscillation


def test_capacity_drops_exactly_the_overflow() -> None:
    torch.manual_seed(1)
    idx = torch.randint(0, 4, (50, 2))
    idx[:, 1] = (idx[:, 0] + 1) % 4  # distinct experts per token
    w = torch.rand(50, 2)
    cap = capacity(50, 2, 4, 1.0)
    p = permute(torch.randn(50, 8), idx, w, 4, cap)
    assert int(p.counts.max()) <= cap
    assert p.n_dropped == int((p.routed_counts - cap).clamp(min=0).sum())
    # the kept rows of each expert are its EARLIEST tokens
    for e in range(4):
        kept = p.token_idx[p.expert_idx == e]
        assert torch.equal(kept, (idx == e).any(-1).nonzero().flatten()[:cap])


def test_huge_capacity_equals_dropless() -> None:
    x = torch.randn(40, CFG.hidden)
    a, b = _layer(), _layer(with_(CFG, capacity_factor=100.0))
    with torch.no_grad():
        torch.testing.assert_close(a(x)[0], b(x)[0])


@pytest.mark.parametrize("world", [2, 4])
def test_simulated_ep_matches_single_device(world: int) -> None:
    layer = _layer()
    x = torch.randn(48, CFG.hidden)
    with torch.no_grad():
        y_ep, rows = simulate_ep(layer, x, world)
        torch.testing.assert_close(y_ep, layer(x)[0], rtol=1e-4, atol=1e-5)
    assert int(rows.sum()) == 48 * CFG.top_k


def test_comm_volume_dedup_never_exceeds_per_slot() -> None:
    layer = _layer(with_(CFG, n_experts=16, top_k=4))
    idx = [layer.router(torch.randn(30, CFG.hidden)).topk_idx for _ in range(4)]
    v = comm_volume(idx, 16, bytes_per_row=1)
    assert int(v.per_slot.sum()) == 4 * 30 * 4
    assert bool((v.dedup <= v.per_slot).all())


@cuda
@pytest.mark.parametrize("backend", ["grouped_mm", "triton", "padded_bmm"])
def test_gpu_grouped_backends(backend: str) -> None:
    torch.manual_seed(0)
    e, h, i = 8, 256, 192
    counts = torch.tensor([0, 1, 130, 7, 0, 64, 3, 300])  # empty, tiny and multi-tile groups
    rows = torch.randn(int(counts.sum()), h, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(e, h, 2 * i, device="cuda", dtype=torch.bfloat16) * h ** -0.5
    w2 = torch.randn(e, i, h, device="cuda", dtype=torch.bfloat16) * i ** -0.5
    ref = GROUPED_IMPLS["sorted_loop"](rows.float(), counts, w1.float(), w2.float())
    out = GROUPED_IMPLS[backend](rows, counts.cuda(), w1, w2)
    torch.testing.assert_close(out.float(), ref, rtol=3e-2, atol=3e-2)
