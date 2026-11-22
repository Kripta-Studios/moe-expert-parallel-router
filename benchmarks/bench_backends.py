"""Naive per-expert loop vs sorted/grouped GEMM on one GPU.

Layer shape = one Qwen3-30B-A3B MoE layer (H=2048, I=768, E=128, top-8, no shared
expert), bf16, random router weights. Times the full routed path
(router -> permute -> experts -> combine) for each back-end and token count.

Run (through the shared GPU lock):
    python benchmarks/bench_backends.py
"""

from __future__ import annotations

import torch
from common import cuda_time_ms, save_json

from moe_ep.config import QWEN3_30B_A3B
from moe_ep.experts import BACKENDS
from moe_ep.layer import MoELayer

TOKENS = (16, 128, 512, 2048, 8192)
PAD_BUDGET_BYTES = 1.2e9  # skip padded_bmm when its padded buffers would blow the VRAM budget


def ceilings() -> dict[str, float]:
    """Measured in the same run: what a plain dense GEMM and a device copy reach on
    this (shared, laptop) GPU right now. Use these to read the MoE numbers."""
    a = torch.randn(8192, 2048, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(2048, 1536, device="cuda", dtype=torch.bfloat16)
    mm = cuda_time_ms(lambda: a @ b, iters=20)
    buf = torch.empty(256 * 2**20, device="cuda", dtype=torch.bfloat16)
    cp = cuda_time_ms(lambda: buf.clone(), iters=20)
    return {"dense_gemm_tflops": round(2 * 8192 * 2048 * 1536 / (mm * 1e9), 1),
            "copy_GBps": round(2 * buf.numel() * 2 / (cp * 1e6), 1)}


def kernel_names(fn) -> list[str]:  # type: ignore[no-untyped-def]
    """Which CUDA kernels a call actually launches (torch.profiler)."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    evs = sorted(prof.key_averages(), key=lambda e: -e.device_time_total)
    return [f"{e.key[:90]} x{e.count}" for e in evs[:4]]


def main() -> None:
    torch.manual_seed(0)
    cfg = QWEN3_30B_A3B
    layer = MoELayer(cfg).to("cuda", torch.bfloat16)
    flop_per_row = 2 * 3 * cfg.hidden * cfg.expert_inter
    rows = []
    ceil = ceilings()
    print("ceilings", ceil, flush=True)
    kernels = {}
    with torch.no_grad():
        for t in TOKENS:
            x = torch.randn(t, cfg.hidden, device="cuda", dtype=torch.bfloat16)
            _, st = layer(x)
            cap = int(st.routed_counts.max())
            ref = None
            for be in BACKENDS:
                if be == "padded_bmm" and cfg.n_experts * cap * 2 * cfg.expert_inter * 2 > PAD_BUDGET_BYTES:
                    rows.append({"tokens": t, "backend": be, "ms": None, "note": "skipped: padding > VRAM budget"})
                    continue
                layer.backend = be
                y = layer(x)[0]
                ref = y if ref is None else ref
                err = ((y.float() - ref.float()).norm() / ref.float().norm()).item()
                ms = cuda_time_ms(lambda: layer(x), iters=5 if be == "naive" else 20)
                tflops = t * cfg.top_k * flop_per_row / (ms * 1e9)
                rows.append({"tokens": t, "backend": be, "ms": round(ms, 4), "tflops": round(tflops, 2),
                             "rel_err_vs_naive": float(f"{err:.2e}"),
                             "pad_waste": round(cfg.n_experts * cap / (t * cfg.top_k), 2) if be == "padded_bmm" else None})
                print(rows[-1], flush=True)
                if t == 2048 and be in ("grouped_mm", "triton"):
                    kernels[be] = kernel_names(lambda: layer(x))
            del x
    torch.cuda.empty_cache()
    print("peak VRAM GB", torch.cuda.max_memory_allocated() / 1e9)
    save_json("backends.json", {"config": cfg.__dict__, "ceilings": ceil, "kernels_at_2048": kernels, "rows": rows,
                                "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)})


if __name__ == "__main__":
    main()
