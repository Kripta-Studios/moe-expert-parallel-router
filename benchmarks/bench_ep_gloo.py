"""Real expert parallelism across 4 CPU processes with torch.distributed (gloo).

Each process is one EP rank owning E/4 experts. We run dispatch -> all-to-all ->
expert compute -> all-to-all -> combine, check the result against the single-process
layer, and time each phase. Gloo over loopback TCP is orders of magnitude slower
than NVLink/IB, so the *times* only illustrate the structure; the byte counts are
exact and are what the analytic model (cost_model.py) scales up.

Run:  python benchmarks/bench_ep_gloo.py      (4 processes, tiny tensors, CPU only)
"""

from __future__ import annotations

import json
import os
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from common import save_json

from moe_ep.config import MoEConfig
from moe_ep.dispatch import unpermute
from moe_ep.ep import expert_slice, gloo_all_to_all, local_expert_compute, plan_dispatch
from moe_ep.layer import MoELayer
from moe_ep.metrics import comm_volume

WORLD, TOKENS_PER_RANK, ITERS = 4, 512, 10
CFG = MoEConfig(hidden=512, expert_inter=256, n_experts=32, top_k=4, n_shared=1)


def _timed(fn):  # type: ignore[no-untyped-def]
    dist.barrier()  # phases are separated by barriers so per-phase times are comparable
    t0 = time.perf_counter()
    out = fn()
    dist.barrier()
    return out, (time.perf_counter() - t0) * 1e3


@torch.no_grad()
def worker(rank: int, init_file: str, result_file: str) -> None:
    dist.init_process_group("gloo", init_method=f"file:///{init_file}", rank=rank, world_size=WORLD)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    layer = MoELayer(CFG, "sorted_loop")
    x_all = torch.randn(WORLD * TOKENS_PER_RANK, CFG.hidden)
    x = x_all.chunk(WORLD)[rank]
    phases: dict[str, list[float]] = {k: [] for k in ("route_permute", "dispatch_a2a", "experts", "combine_a2a", "unpermute_shared")}
    for _ in range(ITERS):
        (p, splits), t = _timed(lambda: plan_dispatch(layer, x, WORLD))
        phases["route_permute"].append(t)
        (recv, rsplits), t = _timed(lambda: gloo_all_to_all(p.rows, splits))
        phases["dispatch_a2a"].append(t)
        recv_eid, _ = gloo_all_to_all(p.expert_idx, splits)
        y_recv, t = _timed(lambda: local_expert_compute(layer, recv, recv_eid, rank, WORLD, "sorted_loop"))
        phases["experts"].append(t)
        (y_back, _), t = _timed(lambda: gloo_all_to_all(y_recv, rsplits))
        phases["combine_a2a"].append(t)
        y, t = _timed(lambda: unpermute(y_back, p, x.shape[0]) + layer.shared(x))
        phases["unpermute_shared"].append(t)
    ok = torch.tensor([int(torch.allclose(y, layer(x)[0], rtol=1e-4, atol=1e-5))])
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    send = torch.tensor(splits)
    gathered = [torch.zeros_like(send) for _ in range(WORLD)]
    dist.all_gather(gathered, send)
    idx = [layer.router(c).topk_idx for c in x_all.chunk(WORLD)]
    if rank == 0:
        rows = torch.stack(gathered)
        vol = comm_volume(idx, CFG.n_experts, bytes_per_row=CFG.hidden * 4)  # fp32 rows
        e0, e1 = expert_slice(rank, WORLD, CFG.n_experts)
        med = {k: round(sorted(v)[len(v) // 2], 3) for k, v in phases.items()}
        res = {"world": WORLD, "tokens_per_rank": TOKENS_PER_RANK, "config": CFG.__dict__, "correct": bool(ok.item()),
               "rank0_experts": [e0, e1], "median_ms_rank0": med, "dispatch_rows_matrix": rows.tolist(),
               "network_bytes_per_slot": vol.network_bytes_per_slot, "network_bytes_dedup": vol.network_bytes_dedup,
               "dedup_saving": round(1 - vol.network_bytes_dedup / vol.network_bytes_per_slot, 3)}
        print(res, flush=True)
        with open(result_file, "w") as f:
            json.dump(res, f)
    dist.destroy_process_group()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        init, res = os.path.join(d, "init").replace("\\", "/"), os.path.join(d, "res.json")
        mp.spawn(worker, args=(init, res), nprocs=WORLD, join=True)
        with open(res) as f:
            save_json("ep_gloo.json", json.load(f))
