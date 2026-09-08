"""ANE viability gate for the standalone drafter architecture.

Run BEFORE any training. Builds the candidate drafter as a Core ML graph at
full size and times it against the same math in MLX. If the ANE does not win
here, training a drafter for it is wasted effort.

Architecture (standalone, non-autoregressive):
    local ids [B,T] -> embed -> L x (GQA attn + SwiGLU) -> head -> argmax
    returns [B,T] int32 local ids; caller maps local->global.

The whole block is one dispatch: position 0 is the anchor (last accepted
token), positions 1..T-1 are MASK, and all drafts are predicted in parallel.
"""

import argparse
import time

import numpy as np

VOCAB = 32768  # pruned; covers ~100% of measured coding-workload tokens
BLOCK = 8  # anchor + 7 drafts
REPEATS = 30
WARMUP = 5


def _median_ms(fn, repeats=REPEATS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2], ts[0], ts[-1]


def params_m(hidden, layers, inter, heads, kv, hd):
    per_layer = hidden * (heads * hd) + 2 * hidden * (kv * hd) + (heads * hd) * hidden + 3 * hidden * inter
    return (2 * VOCAB * hidden + layers * per_layer) / 1e6


def build_coreml(hidden, layers, inter, heads, kv, hd, units, int8=False):
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    f16 = lambda *s: (np.random.randn(*s) * 0.02).astype(np.float16)
    emb = f16(VOCAB, hidden)
    head = f16(hidden, VOCAB)
    W = [
        dict(
            q=f16(hidden, heads * hd), k=f16(hidden, kv * hd), v=f16(hidden, kv * hd),
            o=f16(heads * hd, hidden), g=f16(hidden, inter), u=f16(hidden, inter), d=f16(inter, hidden),
        )
        for _ in range(layers)
    ]

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, BLOCK), dtype=types.int32)],
        opset_version=ct.target.iOS18,
    )
    def prog(ids):
        x = mb.gather(x=emb, indices=ids, axis=0)
        for w in W:
            q = mb.matmul(x=x, y=w["q"])
            k = mb.tile(x=mb.matmul(x=x, y=w["k"]), reps=[1, 1, heads // kv])
            v = mb.tile(x=mb.matmul(x=x, y=w["v"]), reps=[1, 1, heads // kv])
            s = mb.softmax(x=mb.mul(x=mb.matmul(x=q, y=k, transpose_y=True), y=np.float16(hd**-0.5)), axis=-1)
            h = mb.add(x=x, y=mb.matmul(x=mb.matmul(x=s, y=v), y=w["o"]))
            g = mb.silu(x=mb.matmul(x=h, y=w["g"]))
            x = mb.add(x=h, y=mb.matmul(x=mb.mul(x=g, y=mb.matmul(x=h, y=w["u"])), y=w["d"]))
        logits = mb.matmul(x=x, y=head)
        # argmax in-graph: return 8 int32 ids, not 524KB of logits.
        return mb.reduce_argmax(x=logits, axis=-1, name="out")

    t0 = time.perf_counter()
    model = ct.convert(
        prog,
        compute_units=units,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
    )
    if int8:
        from coremltools.optimize.coreml import (
            OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights,
        )
        cfg = OptimizationConfig(
            global_config=OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
        )
        model = linear_quantize_weights(model, config=cfg)
    return model, time.perf_counter() - t0


def bench_mlx(hidden, layers, inter, heads, kv, hd):
    import mlx.core as mx

    f = lambda *s: (mx.random.normal(s) * 0.02).astype(mx.bfloat16)
    emb, head = f(VOCAB, hidden), f(hidden, VOCAB)
    W = [
        dict(q=f(hidden, heads * hd), k=f(hidden, kv * hd), v=f(hidden, kv * hd),
             o=f(heads * hd, hidden), g=f(hidden, inter), u=f(hidden, inter), d=f(inter, hidden))
        for _ in range(layers)
    ]
    ids = mx.array(np.random.randint(0, VOCAB, (1, BLOCK)).astype(np.int32))

    def run():
        x = emb[ids]
        for w in W:
            q, k, v = x @ w["q"], x @ w["k"], x @ w["v"]
            k = mx.repeat(k, heads // kv, axis=-1)
            v = mx.repeat(v, heads // kv, axis=-1)
            s = mx.softmax((q @ k.transpose(0, 2, 1)) * hd**-0.5, axis=-1)
            h = x + (s @ v) @ w["o"]
            x = h + (mx.sigmoid(h @ w["g"]) * (h @ w["g"]) * (h @ w["u"])) @ w["d"]
        mx.eval(mx.argmax(x @ head, axis=-1))

    return _median_ms(run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=768)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--inter", type=int, default=2048)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--kv", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--units", default="CPU_AND_NE")
    a = ap.parse_args()

    import coremltools as ct

    cfg = (a.hidden, a.layers, a.inter, a.heads, a.kv, a.head_dim)
    n = params_m(*cfg)
    print(f"drafter: hidden={a.hidden} layers={a.layers} inter={a.inter} "
          f"heads={a.heads} kv={a.kv} head_dim={a.head_dim}")
    print(f"  params {n:.0f}M   fp16 {n*2:.0f} MB   int8 {n:.0f} MB   vocab {VOCAB}\n")

    model, compile_s = build_coreml(*cfg, getattr(ct.ComputeUnit, a.units), int8=a.int8)
    ids = np.random.randint(0, VOCAB, (1, BLOCK)).astype(np.int32)
    med, lo, hi = _median_ms(lambda: model.predict({"ids": ids}))
    tag = "int8" if a.int8 else "fp16"
    print(f"  Core ML {a.units:12s} ({tag})  {med:7.2f} ms  (min {lo:5.2f} / max {hi:5.2f})  [compile {compile_s:.0f}s]")

    med_mlx, lo_mlx, hi_mlx = bench_mlx(*cfg)
    print(f"  MLX GPU                    (bf16)  {med_mlx:7.2f} ms  (min {lo_mlx:5.2f} / max {hi_mlx:5.2f})")

    ratio = med / med_mlx
    verdict = "ANE FASTER" if ratio < 1 else "ANE SLOWER"
    print(f"\n  ==> {verdict}  ratio {ratio:.2f}x")
    print(f"      (DFlash2 on GPU measured ~19.7 ms/round for comparison)")


if __name__ == "__main__":
    main()
