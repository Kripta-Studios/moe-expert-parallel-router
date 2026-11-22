"""Gloo all-to-all (2 CPU processes, tiny tensors) and cost-model sanity checks."""

from __future__ import annotations

import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from moe_ep.config import MoEConfig
from moe_ep.cost_model import EPSetup, eplb_greedy, expected_distinct, rows_per_token
from moe_ep.ep import ep_forward_rank, gloo_all_to_all
from moe_ep.layer import MoELayer

CFG = MoEConfig(hidden=32, expert_inter=16, n_experts=8, top_k=2, n_shared=1)


def _worker(rank: int, world: int, init_file: str, result_file: str) -> None:
    dist.init_process_group("gloo", init_method=f"file:///{init_file}", rank=rank, world_size=world)
    torch.set_num_threads(1)
    torch.manual_seed(0)  # identical weights and inputs on every rank
    layer = MoELayer(CFG, "sorted_loop")
    x = torch.randn(24, CFG.hidden)
    mine = x.chunk(world)[rank]
    y = ep_forward_rank(layer, mine, rank, world, gloo_all_to_all)
    with torch.no_grad():
        ok = torch.allclose(y, layer(mine)[0], rtol=1e-4, atol=1e-5)
    flag = torch.tensor([int(ok)])
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if rank == 0:
        with open(result_file, "w") as f:
            f.write(str(int(flag)))
    dist.destroy_process_group()


def test_gloo_ep_matches_local() -> None:
    with tempfile.TemporaryDirectory() as d:
        init, res = os.path.join(d, "init").replace("\\", "/"), os.path.join(d, "res")
        mp.spawn(_worker, args=(2, init, res), nprocs=2, join=True)
        with open(res) as f:
            assert f.read() == "1"


def test_expected_distinct_edge_cases() -> None:
    assert expected_distinct(256, 256, 8) == 8.0      # one expert per bin: every pick is a new bin
    assert abs(expected_distinct(256, 1, 8) - 1.0) < 1e-12


def test_node_limited_routing_cuts_ib_traffic() -> None:
    r = rows_per_token(EPSetup())
    assert r["ib_node_limited"] < r["per_rank"] < r["per_slot"]


def test_eplb_reduces_rank_imbalance() -> None:
    torch.manual_seed(0)
    load = torch.distributions.Pareto(1.0, 1.5).sample((64,))  # heavy-tailed expert popularity
    world = 8
    naive = load.reshape(world, -1).sum(1)
    replicas, balanced = eplb_greedy(load, 64 + 16, world)
    assert int(replicas.sum()) == 80
    assert abs(float(balanced.sum() - load.sum())) < 1e-3
    assert balanced.max() / balanced.mean() < naive.max() / naive.mean()
