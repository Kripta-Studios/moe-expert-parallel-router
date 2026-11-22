"""Load balancing on REAL hidden states: aux-loss-free bias, capacity dropping, EP.

A router with random weights is perfectly balanced on isotropic Gaussian inputs,
which hides the problem. Real LLM hidden states are strongly anisotropic, so we
route actual Qwen3-0.6B activations (layer 14, WikiText-2) and watch:

  1. the aux-loss-free bias update (gamma sweep) driving max/mean load towards 1,
  2. capacity factor vs fraction of dropped (token, expert) slots and output error,
  3. what imbalance costs under EP=8 virtual ranks: per-rank token load and measured
     grouped-GEMM time of the slowest rank on this GPU, plus EPLB replication.

Run (through the shared GPU lock):  python benchmarks/bench_balance.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from common import RESULTS, ROOT, cuda_time_ms, save_json  # noqa: E402

from moe_ep.config import MoEConfig  # noqa: E402
from moe_ep.cost_model import eplb_greedy  # noqa: E402
from moe_ep.dispatch import capacity  # noqa: E402
from moe_ep.experts import GROUPED_IMPLS  # noqa: E402
from moe_ep.layer import MoELayer  # noqa: E402
from moe_ep.metrics import load_stats, rank_loads  # noqa: E402

CACHE = ROOT / ".cache" / "qwen3_0.6b_layer14_wikitext.pt"
N_TOKENS, LAYER, BATCH, STEPS, WORLD = 32768, 14, 1024, 400, 8
CFG = MoEConfig(hidden=1024, expert_inter=256, n_experts=64, top_k=8, n_shared=1)


def hidden_states() -> torch.Tensor:
    """Extract (once) N_TOKENS hidden states; ~64 MB fp16 cache, never committed."""
    if CACHE.exists():
        return torch.load(CACHE)
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True).to("cuda").eval()
    text = "\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"][:6000] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0][:N_TOKENS]
    chunks = []
    with torch.no_grad():
        for c in ids.split(1024):
            out = model(c[None].cuda(), output_hidden_states=True)
            chunks.append(out.hidden_states[LAYER][0].half().cpu())
    del model
    torch.cuda.empty_cache()
    h = torch.cat(chunks)
    CACHE.parent.mkdir(exist_ok=True)
    torch.save(h, CACHE)
    return h


def rms(x: torch.Tensor) -> torch.Tensor:  # the MoE block sees RMS-normalised input
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


def counts_of(layer: MoELayer, x: torch.Tensor) -> torch.Tensor:
    return torch.bincount(layer.router(x).topk_idx.flatten(), minlength=CFG.n_experts)


def main() -> None:
    h = rms(hidden_states().float().cuda())
    train, test = h[:-4096], h[-4096:]
    out: dict = {"config": CFG.__dict__, "tokens": h.shape[0], "source": f"Qwen3-0.6B layer {LAYER}, WikiText-2"}

    # 1. aux-loss-free bias balancing, gamma sweep
    curves, final_bias = {}, None
    for gamma in (0.0, 1e-4, 1e-3, 1e-2):
        torch.manual_seed(0)
        layer = MoELayer(MoEConfig(**{**CFG.__dict__, "bias_update_speed": gamma})).cuda()
        curve = []
        for step in range(STEPS):
            i = (step * BATCH) % (train.shape[0] - BATCH)
            c = counts_of(layer, train[i:i + BATCH])
            curve.append(load_stats(c)["max_over_mean"])
            layer.router.update_bias(c)
        curves[str(gamma)] = curve
        out[f"test_after_gamma_{gamma}"] = load_stats(counts_of(layer, test))
        if gamma == 1e-3:
            final_bias = layer.router.expert_bias.clone()
    print({k: v for k, v in out.items() if k.startswith("test")}, flush=True)

    # same weights (seed 0), bias off vs the gamma=1e-3 bias
    torch.manual_seed(0)
    layer = MoELayer(CFG).cuda()
    c_raw, ep_raw = counts_of(layer, test), counts_of(layer, h)
    layer.router.expert_bias.copy_(final_bias)
    c_bal, ep_bal = counts_of(layer, test), counts_of(layer, h)

    # 2. capacity factor: drop rate and output error vs dropless, with and without balancing
    caps = []
    with torch.no_grad():
        for label, bias in (("unbalanced", torch.zeros_like(final_bias)), ("bias_balanced", final_bias)):
            layer.router.expert_bias.copy_(bias)
            layer.cfg = CFG
            ref = layer(test)[0]
            for cf in (1.0, 1.25, 1.5, 2.0, 3.0):
                layer.cfg = MoEConfig(**{**CFG.__dict__, "capacity_factor": cf})
                y, st = layer(test)
                drop = 1 - st.kept_counts.sum().item() / st.routed_counts.sum().item()
                caps.append({"routing": label, "cf": cf, "capacity": capacity(test.shape[0], CFG.top_k, CFG.n_experts, cf),
                             "dropped_slot_frac": round(drop, 4),
                             "rel_err_vs_dropless": round(((y - ref).norm() / ref.norm()).item(), 4)})
    layer.cfg = CFG
    out["capacity"] = caps
    print(caps, flush=True)

    # 3. EP=8 virtual ranks over all 32k tokens (262k slots, so GEMM time is
    #    dominated by FLOPs, not launch overhead): rank loads, measured per-rank
    #    grouped-GEMM time on this GPU (ranks run one after another), and EPLB.
    w1 = layer.experts.w_gate_up.detach().bfloat16()
    w2 = layer.experts.w_down.detach().bfloat16()
    per = CFG.n_experts // WORLD
    ep = {}
    for label, c in (("unbalanced", ep_raw), ("bias_balanced", ep_bal)):
        rank_ms = []
        for r in range(WORLD):
            cnt = c[r * per:(r + 1) * per]
            rows = torch.randn(int(cnt.sum()), CFG.hidden, device="cuda", dtype=torch.bfloat16)
            rank_ms.append(cuda_time_ms(lambda: GROUPED_IMPLS["triton"](rows, cnt, w1[r * per:(r + 1) * per],
                                                                             w2[r * per:(r + 1) * per]), iters=20))
        rl = rank_loads(c, WORLD).float()
        _, eplb = eplb_greedy(c.float().cpu(), CFG.n_experts + WORLD * 2, WORLD)  # 2 redundant slots per rank
        ep[label] = {"rank_load_max_over_mean": round((rl.max() / rl.mean()).item(), 3),
                     "eplb_rank_load_max_over_mean": round((eplb.max() / eplb.mean()).item(), 3),
                     "rank_ms": [round(m, 4) for m in rank_ms],
                     "slowest_over_mean_ms": round(max(rank_ms) / (sum(rank_ms) / WORLD), 3),
                     "expert_load": c.tolist()}
    out["ep8"] = ep
    print({k: {kk: vv for kk, vv in v.items() if kk != "expert_load"} for k, v in ep.items()}, flush=True)
    save_json("balance.json", {**out, "curves": curves})
    plot(curves, c_raw.cpu(), c_bal.cpu())


def plot(curves: dict, c_raw: torch.Tensor, c_bal: torch.Tensor) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8))
    for g, cur in curves.items():
        a.plot(cur, label=f"gamma={g}", lw=1.2)
    a.set(xlabel="update step (batch of 1024 tokens)", ylabel="max / mean expert load", yscale="log",
          title="Aux-loss-free bias balancing (Qwen3-0.6B states)")
    a.axhline(1, color="gray", lw=0.8, ls=":")
    a.legend(frameon=False)
    x = torch.arange(len(c_raw))
    b.bar(x - 0.2, c_raw, width=0.4, label="bias = 0")
    b.bar(x + 0.2, c_bal, width=0.4, label="bias after 400 steps (gamma=1e-3)")
    b.set(xlabel="expert id", ylabel="routed slots (held-out 4096 tokens)", title="Per-expert load")
    b.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(Path(RESULTS) / "balance.png", dpi=110)


if __name__ == "__main__":
    main()
