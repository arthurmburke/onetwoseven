# DFlash2 → ANE: go/no-go measurement

**Verdict: no-go for DFlash2.** The ANE is 1.3–2.1× *slower* than the M5 Max GPU
on the drafter's own math, and that is a best case — the benchmark omits the
hard parts, all of which cost the ANE more.

Hardware: M5 Max, 128 GB, macOS 26.6.2. coremltools 9.0, MLX 0.32.2.
Target `mlx-community/Qwen3.8-27B-8bit`, drafter `z-lab/Qwen3.8-27B-DFlash2`.

## Numbers

Per speculative round, block_size=8, native fp16 I/O:

| | Core ML `CPU_AND_NE` | MLX GPU | Core ML `CPU_ONLY` |
|---|---|---|---|
| dispatch floor (400 KB in) | **0.23 ms** | — | 0.30 ms |
| fc proj 25600→5120 | 1.93 ms | 0.75–0.94 ms | 18.29 ms |
| 1 decoder layer | 4.75 ms | ~2.3–3.1 ms | 39.14 ms |
| **est. per-round total** | **25.88 ms** | **19.66 ms** | — |

MLX numbers are noisy (5-layer stack ranged 12.5–19.7 ms across runs; Core ML
compilation in the same process perturbs it). The *ratio* moved between 1.32×
and 2.1×; **the sign never did.** The ANE was slower in every configuration.

`CPU_ONLY` at 39.14 ms vs `CPU_AND_NE` at 4.75 ms (8.2×) confirms the ANE is
actually executing the graph rather than falling back to CPU.

## Why — it's bandwidth, not compute

The fc projection streams 262 MB of fp16 weights. ANE does it in 1.93 ms
(~136 GB/s effective); the GPU does it in ~0.8 ms (~330 GB/s).

The drafter is **weight-streaming-bound**, and the ANE gets materially less
memory bandwidth than the M5 Max GPU. The ANE's matmul throughput advantage is
irrelevant when the bottleneck is reading 3.6 GB of weights, which is the same
reason decode generally belongs on the GPU.

## Dispatch overhead is NOT the blocker

Worth recording, because it contradicts the published figure we were working
from. FusionML reports ~20–24 ms fixed Core ML dispatch cost; we measure
**0.23 ms** for a 400 KB round trip on macOS 26 / M5 Max — close to the ~190 µs
firmware-level figure in the Orion paper.

So ANE drafting is not *structurally* dead. It is dead **for a drafter this
large**.

## What the benchmark leaves out (all of which make ANE worse)

The MIL graph is the matmul skeleton only. The real DFlash2 layer adds:

- `GroupedDynamicCausalConv` — input-dependent conv weights, very likely a CPU
  fallback under Core ML
- `CandidateSelector` — top-k over vocab 248320, plus a **Python sampler
  callback** invoked mid-graph (`sample_proposal`), which cannot cross into Core ML
- RMSNorm, RoPE, sliding-window mask (window 2048), persistent KV cache
  (needs Core ML stateful models)
- Dynamic `block_size` (3…8 per `_dflash_next_block_size`) → needs enumerated
  shapes, one compiled variant each

## The structural blocker, independent of speed

`DFlash2DraftModel.bind(target_model)` (dflash2.py:144) wires `embed_tokens`
and `lm_head` to **the target's** tensors. The drafter's 1.92 B params are
86.5% decoder layers, 6.8% `fc`, 6.7% selector — it ships no embedding table
and no LM head.

So the LM head is a 248320×5120 8-bit tensor living in MLX GPU memory. Moving
the drafter to the ANE means either duplicating it (+2.5 GB fp16, streamed per
dispatch) or splitting the graph — which puts **two** boundary crossings in
every round instead of one.

And `draft_block` consumes `hidden` from the target's verify pass
(dflash.py:350), so drafting cannot start until verification resolves. Even a
free ANE would not overlap with the GPU for a single stream.

## If pursuing further

The measurement points somewhere specific: at 0.23 ms dispatch, a drafter small
enough to stop being bandwidth-bound (~100–300 M params, not 1.92 B) could
plausibly beat the GPU. To also unlock real ANE/GPU concurrency it must be
**standalone** — drafting from token IDs, not conditioned on target hidden
states — which DFlash2 is not.

That is a different drafter, requiring training/distillation against Qwen3.8's
248320-token vocabulary. Not a conversion of this one.

## Reproduce

```bash
./.venv/bin/python ane_drafter/bench_dispatch.py --units CPU_AND_NE
./.venv/bin/python ane_drafter/bench_dispatch.py --units CPU_ONLY   # placement control
```

---

# Part 2: DFlash2 acceptance, and why the standalone drafter was dropped

Measured on the real stack (target + drafter loaded, greedy, 5 coding prompts,
192 max tokens, 319+ rounds per config).

| block | tok/s | acc/round | accept% | draft ms | round ms |
|---|---|---|---|---|---|
| **default (adaptive)** | **35.2** | 2.97 | 73.9% | 10.41 | **84.96** |
| 4  | 32.9 | 2.79 | 78.3% | 10.86 | 85.49 |
| 8  | 30.4 | 2.94 | 71.3% | 11.43 | 97.29 |
| 12 | 23.3 | 2.98 | 70.6% | 13.31 | 128.41 |
| 16 | 21.0 | 2.98 | 70.6% | 13.19 | 142.35 |

## Two corrections to Part 1

1. **Real draft cost is 10.41 ms**, not the ~19.7 ms synthetic estimate. The
   prize is half what we thought.
2. **"Verify is flat in block size so draft harder" is wrong.** Acceptance
   saturates at ~2.97 regardless of block size; larger blocks add verify cost
   for zero extra acceptance. The adaptive default is already optimal.

## Break-even

verify = 84.96 - 10.41 = **74.55 ms** (agrees with the 76.88 ms direct measurement).

Standalone-on-ANE round = 1.49 + 74.55 = 76.04 ms, so break-even acceptance is
2.97 x 76.04/84.96 = **2.66 acc/round = 89.5% of DFlash2's**.

A 126 M standalone drafter with no target hidden states will not reach 89.5% of
a 1.9 B hidden-state-conditioned drafter. At a plausible 2.2 acc/round it gives
28.9 tok/s -- an 18% regression. Ceiling with zero draft cost and unchanged
acceptance is +14%; the ANE's share of that is ~1%.

**Decision: standalone drafter not built. ANE line closed.**

## Ranked levers

| Lever | Est. gain |
|---|---|
| Target 8-bit -> 4-bit (halves 74.55 ms of weight streaming) | ~+65% |
| Better acceptance (2.97 -> 3.5) | +18% |
| Zero draft cost (unachievable ceiling) | +14% |
| ANE offload | ~+1% |

Draft is 12.3% of round time; verify is 87.7%. Quantization is the lever.

## Reproduce

```bash
./.venv/bin/python ane_drafter/bench_acceptance.py --blocks none,4,8,12,16
./.venv/bin/python ane_drafter/bench_target.py
```
