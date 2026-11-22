"""A from-scratch Triton grouped GEMM:  C[rows of e] = A[rows of e] @ B[e]  for all e.

A is the permuted activation matrix [N, K] whose rows are grouped by expert
(group e occupies rows [start_e, start_e + count_e)). B is the stacked expert
weight [E, K, Nout]. One kernel launch covers every expert.

Scheduling trick: each expert needs ceil(count_e / BM) row-tiles. We build a
cumulative tile count per expert on the GPU (one cumsum) and every program finds
its expert by binary search over it, so the launch never syncs with the host.
The grid is sized by an upper bound, sum_e ceil(c_e/BM) <= ceil(N/BM) + E, and
surplus programs exit immediately. Rows past the end of a group are masked, so a
tile never mixes two experts (that is what MegaBlocks' block-sparse layout and
cuBLAS/CUTLASS grouped GEMMs also guarantee).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# BM (rows per tile) is fixed because the tile table is built for one BM; the
# autotuner explores the other tile dimensions and pipelining depth.
BLOCK_M = 64


@triton.autotune(
    configs=[
        triton.Config({"BN": 128, "BK": 64}, num_warps=4, num_stages=3),
        triton.Config({"BN": 64, "BK": 64}, num_warps=4, num_stages=4),
        triton.Config({"BN": 128, "BK": 32}, num_warps=4, num_stages=4),
        triton.Config({"BN": 256, "BK": 64}, num_warps=8, num_stages=3),
    ],
    key=["K", "NOUT", "BUCKET"],
)
@triton.jit
def _grouped_gemm_kernel(a_ptr, b_ptr, c_ptr, starts_ptr, counts_ptr, tile_cum_ptr,
                         E, K, NOUT, BUCKET,
                         s_am, s_ak, s_be, s_bk, s_bn, s_cm, s_cn,
                         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # Find the expert owning this row-tile: first e with tile_cum[e] > pid_m (binary search).
    lo = 0
    hi = E
    while lo < hi:
        mid = (lo + hi) // 2
        if tl.load(tile_cum_ptr + mid) > pid_m:
            hi = mid
        else:
            lo = mid + 1
    e = lo
    if e >= E:
        return  # surplus program from the upper-bound grid
    tiles_before = tl.load(tile_cum_ptr + e - 1, mask=e > 0, other=0)
    start = tl.load(starts_ptr + e)
    end = start + tl.load(counts_ptr + e)
    offs_m = start + (pid_m - tiles_before) * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m[:, None].to(tl.int64) * s_am + offs_k[None, :] * s_ak
    b_ptrs = b_ptr + e.to(tl.int64) * s_be + offs_k[:, None] * s_bk + offs_n[None, :] * s_bn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in range(0, tl.cdiv(K, BK)):
        k_left = K - kk * BK
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < end) & (offs_k[None, :] < k_left), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_left) & (offs_n[None, :] < NOUT), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK * s_ak
        b_ptrs += BK * s_bk
    c_ptrs = c_ptr + offs_m[:, None].to(tl.int64) * s_cm + offs_n[None, :] * s_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=(offs_m[:, None] < end) & (offs_n[None, :] < NOUT))


def grouped_gemm(a: torch.Tensor, b: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """a: [N, K] rows grouped by expert, b: [E, K, NOUT], counts: [E] (sum == N)."""
    n, k = a.shape
    e, _, nout = b.shape
    c = torch.empty(n, nout, device=a.device, dtype=a.dtype)
    if n == 0:
        return c
    counts = counts.to(device=a.device, dtype=torch.int32)
    starts = torch.cumsum(counts, 0, dtype=torch.int32) - counts
    tile_cum = torch.cumsum((counts + BLOCK_M - 1) // BLOCK_M, 0, dtype=torch.int32)
    bucket = 1 << max(n - 1, 0).bit_length()  # autotune per power-of-two row count

    def grid(meta: dict) -> tuple[int, int]:
        return (triton.cdiv(n, BLOCK_M) + e, triton.cdiv(nout, meta["BN"]))

    _grouped_gemm_kernel[grid](
        a, b, c, starts, counts, tile_cum, e, k, nout, bucket,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), b.stride(2), c.stride(0), c.stride(1),
        BM=BLOCK_M)
    return c
