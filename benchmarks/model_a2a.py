"""Analytic all-to-all model for DeepSeek-V3 (256 experts, top-8, EP=64). MODELED.

Prints a table for decode-like (128 tokens/GPU) and prefill-like (4096 tokens/GPU)
micro-batches, compares the model against DeepEP's published H800 numbers, and
plots layer time vs tokens/GPU with and without dual-batch overlap.

Run:  python benchmarks/model_a2a.py     (pure arithmetic, no GPU)
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from common import RESULTS, save_json  # noqa: E402

from moe_ep.cost_model import (IB_CX7, SCENARIOS, EPSetup, a2a_time_us, expert_time_us,  # noqa: E402
                               layer_time_us, rows_per_token)

# DeepEP README (v1.x, H800 + CX7 400 Gb/s, 128 tokens, hidden 7168, top-8, FP8 dispatch / BF16 combine)
DEEPEP_LOW_LATENCY_EP64_US = {"dispatch": 173.0, "combine": 314.0}


def main() -> None:
    s = EPSetup()
    rows = rows_per_token(s)
    table = []
    for tokens in (128, 4096):
        for sc in SCENARIOS:
            d = a2a_time_us(tokens, s, sc, s.dispatch_bytes)
            c = a2a_time_us(tokens, s, sc, s.combine_bytes)
            table.append({"tokens_per_gpu": tokens, "scenario": sc, "dispatch_us": round(d, 1), "combine_us": round(c, 1),
                          "experts_us": round(expert_time_us(tokens, s), 1),
                          "layer_serial_us": round(layer_time_us(tokens, s, sc, False), 1),
                          "layer_dbo_us": round(layer_time_us(tokens, s, sc, True), 1)})
    for r in table:
        print(r)

    # Calibration: DeepEP low-latency kernels send one copy per (token, expert) slot
    # over RDMA (no NVLink forwarding). Our per-slot model counts only remote copies,
    # so include all k slots here to match what DeepEP's bandwidth figure counts.
    ll_bytes = {"dispatch": 128 * s.top_k * s.hidden * s.dispatch_bytes, "combine": 128 * s.top_k * s.hidden * 2.0}
    calib = {k: {"deepep_measured_us": DEEPEP_LOW_LATENCY_EP64_US[k],
                 "model_us_at_50GBps": round(IB_CX7.alpha_us + v / (IB_CX7.bw_gbps * 1e3), 1),
                 "implied_GBps": round(v / (DEEPEP_LOW_LATENCY_EP64_US[k] * 1e3), 1)} for k, v in ll_bytes.items()}
    print(calib)

    sweep = [2 ** i for i in range(4, 14)]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for sc, color in zip(SCENARIOS, ("tab:green", "tab:red", "tab:blue")):
        ax.plot(sweep, [layer_time_us(t, s, sc, False) for t in sweep], color=color, label=f"{sc} (serial)")
        ax.plot(sweep, [layer_time_us(t, s, sc, True) for t in sweep], color=color, ls="--", label=f"{sc} (DBO)")
    ax.plot(sweep, [expert_time_us(t, s) for t in sweep], color="k", lw=0.8, ls=":", label="expert compute only")
    ax.set(xscale="log", yscale="log", xlabel="tokens per GPU per micro-batch", ylabel="MoE layer time (us)",
           title="MODELED: DeepSeek-V3 MoE layer, EP=64 (H800-class)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "a2a_model.png", dpi=110)
    save_json("a2a_model.json", {"MODELED": True, "setup": s.__dict__, "rows_per_token": rows,
                                 "table": table, "deepep_calibration": calib})


if __name__ == "__main__":
    main()
