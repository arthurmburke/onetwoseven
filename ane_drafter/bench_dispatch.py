"""Go/no-go benchmark for running the DFlash2 drafter on the ANE.

Decomposes the per-round cost so we learn *why*, not just *whether*:

  1. dispatch floor  - trivial graph, real 400KB input. Isolates Core ML
                       round-trip + host fp16 copy from any real compute.
  2. one layer       - a single DFlash2 decoder layer at true dims
                       (hidden=5120, inter=17408). Times ANE compute.
  3. fc projection   - the [25600 -> 5120] target-hidden projection.
  4. MLX baseline    - the identical math on GPU, for comparison.

Per-round ANE estimate = dispatch + fc + 5 * layer, versus the MLX number.
The drafter's lm_head/embed are the TARGET's tensors (bind()), so they stay
on GPU either way and are excluded from both sides.
"""

import argparse
import time

import numpy as np

# --- DFlash2 real config (z-lab/Qwen3.8-27B-DFlash2) ---
HIDDEN = 5120
INTER = 17408
N_HEADS = 32
N_KV = 8
HEAD_DIM = 128
N_LAYERS = 5
N_TARGET_LAYERS = 5  # len(target_layer_ids) = [5,19,33,47,61]
CAPTURE_DIM = HIDDEN * N_TARGET_LAYERS  # 25600
BLOCK = 8  # config block_size

REPEATS = 30
WARMUP = 5


def _median_ms(fn, repeats=REPEATS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2], times[0], times[-1]


# ---------------------------------------------------------------- Core ML


def _build_and_time(prog, input_shape, label, compute_units):
    import coremltools as ct

    t0 = time.perf_counter()
    model = ct.convert(
        prog,
        compute_units=compute_units,
        compute_precision=ct.precision.FLOAT16,
        # iOS18+ is required for native fp16 I/O; anything lower silently
        # inserts fp32 cast ops around the graph and doubles transfer size.
        minimum_deployment_target=ct.target.iOS18,
    )
    compile_s = time.perf_counter() - t0

    x = np.random.randn(*input_shape).astype(np.float16)
    med, lo, hi = _median_ms(lambda: model.predict({"x": x}))
    print(
        f"  {label:28s} {med:8.2f} ms  (min {lo:6.2f} / max {hi:6.2f})"
        f"   [compile {compile_s:.1f}s]"
    )
    return med


def coreml_dispatch_floor(compute_units):
    """Trivial graph at the real input size: pure round-trip cost."""
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    shape = (1, BLOCK, CAPTURE_DIM)

    @mb.program(
        input_specs=[mb.TensorSpec(shape=shape, dtype=types.fp16)],
        opset_version=ct.target.iOS18,
    )
    def prog(x):
        return mb.mul(x=x, y=np.float16(1.0001), name="out")

    return _build_and_time(prog, shape, "dispatch floor (400KB in)", compute_units)


def coreml_fc(compute_units):
    """The [25600 -> 5120] target-hidden projection (fc.weight)."""
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    shape = (1, BLOCK, CAPTURE_DIM)
    w = np.random.randn(CAPTURE_DIM, HIDDEN).astype(np.float16) * 0.02

    @mb.program(
        input_specs=[mb.TensorSpec(shape=shape, dtype=types.fp16)],
        opset_version=ct.target.iOS18,
    )
    def prog(x):
        return mb.matmul(x=x, y=w, name="out")

    return _build_and_time(prog, shape, "fc proj 25600->5120", compute_units)


def coreml_layer(compute_units):
    """One DFlash2 decoder layer: GQA attention + SwiGLU MLP at true dims."""
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    shape = (1, BLOCK, HIDDEN)
    f16 = lambda *s: (np.random.randn(*s) * 0.02).astype(np.float16)
    wq, wk, wv = f16(HIDDEN, N_HEADS * HEAD_DIM), f16(HIDDEN, N_KV * HEAD_DIM), f16(HIDDEN, N_KV * HEAD_DIM)
    wo = f16(N_HEADS * HEAD_DIM, HIDDEN)
    wg, wu, wd = f16(HIDDEN, INTER), f16(HIDDEN, INTER), f16(INTER, HIDDEN)

    @mb.program(
        input_specs=[mb.TensorSpec(shape=shape, dtype=types.fp16)],
        opset_version=ct.target.iOS18,
    )
    def prog(x):
        q = mb.matmul(x=x, y=wq)
        k = mb.matmul(x=x, y=wk)
        v = mb.matmul(x=x, y=wv)
        # GQA: repeat kv heads to match q heads (4x), then scaled dot-product.
        k = mb.tile(x=k, reps=[1, 1, N_HEADS // N_KV])
        v = mb.tile(x=v, reps=[1, 1, N_HEADS // N_KV])
        scores = mb.matmul(x=q, y=k, transpose_y=True)
        scores = mb.mul(x=scores, y=np.float16(HEAD_DIM**-0.5))
        probs = mb.softmax(x=scores, axis=-1)
        ctx = mb.matmul(x=probs, y=v)
        attn = mb.matmul(x=ctx, y=wo)
        h = mb.add(x=x, y=attn)
        # SwiGLU MLP
        g = mb.silu(x=mb.matmul(x=h, y=wg))
        u = mb.matmul(x=h, y=wu)
        m = mb.matmul(x=mb.mul(x=g, y=u), y=wd)
        return mb.add(x=h, y=m, name="out")

    return _build_and_time(prog, shape, "1 decoder layer", compute_units)


# ------------------------------------------------------------------- MLX


def mlx_baseline():
    import mlx.core as mx

    print("\nMLX / GPU baseline (same math, bf16):")
    f = lambda *s: mx.random.normal(s).astype(mx.bfloat16) * 0.02
    fc_w = f(CAPTURE_DIM, HIDDEN)
    wq, wk, wv = f(HIDDEN, N_HEADS * HEAD_DIM), f(HIDDEN, N_KV * HEAD_DIM), f(HIDDEN, N_KV * HEAD_DIM)
    wo = f(N_HEADS * HEAD_DIM, HIDDEN)
    wg, wu, wd = f(HIDDEN, INTER), f(HIDDEN, INTER), f(INTER, HIDDEN)
    cap = mx.random.normal((1, BLOCK, CAPTURE_DIM)).astype(mx.bfloat16)

    def fc_only():
        mx.eval(cap @ fc_w)

    def one_layer():
        x = cap @ fc_w
        q, k, v = x @ wq, x @ wk, x @ wv
        k = mx.repeat(k, N_HEADS // N_KV, axis=-1)
        v = mx.repeat(v, N_HEADS // N_KV, axis=-1)
        s = mx.softmax((q @ k.transpose(0, 2, 1)) * HEAD_DIM**-0.5, axis=-1)
        h = x + (s @ v) @ wo
        h = h + (mx.sigmoid(h @ wg) * (h @ wg) * (h @ wu)) @ wd
        mx.eval(h)

    def full_stack():
        """fc + all 5 decoder layers: the whole ANE-able portion of a round."""
        x = cap @ fc_w
        for _ in range(N_LAYERS):
            q, k, v = x @ wq, x @ wk, x @ wv
            k = mx.repeat(k, N_HEADS // N_KV, axis=-1)
            v = mx.repeat(v, N_HEADS // N_KV, axis=-1)
            s = mx.softmax((q @ k.transpose(0, 2, 1)) * HEAD_DIM**-0.5, axis=-1)
            h = x + (s @ v) @ wo
            x = h + (mx.sigmoid(h @ wg) * (h @ wg) * (h @ wu)) @ wd
        mx.eval(x)

    results = {}
    for label, fn in (
        ("fc proj 25600->5120", fc_only),
        ("fc + 1 decoder layer", one_layer),
        ("fc + 5 layers (full)", full_stack),
    ):
        med, lo, hi = _median_ms(fn)
        results[label] = med
        print(f"  {label:28s} {med:8.2f} ms  (min {lo:6.2f} / max {hi:6.2f})")
    return results["fc + 5 layers (full)"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", default="CPU_AND_NE", choices=["CPU_AND_NE", "CPU_ONLY", "ALL"])
    ap.add_argument("--skip-layer", action="store_true", help="skip the slow full-layer build")
    args = ap.parse_args()

    import coremltools as ct

    units = getattr(ct.ComputeUnit, args.units)
    print(f"Core ML ({args.units}), block_size={BLOCK}, capture_dim={CAPTURE_DIM}")
    print(f"  input tensor: {BLOCK * CAPTURE_DIM * 2 / 1024:.0f} KB fp16 per round\n")

    floor = coreml_dispatch_floor(units)
    fc = coreml_fc(units)
    layer = coreml_layer(units) if not args.skip_layer else None

    mlx_full = mlx_baseline()

    if layer is not None:
        est = floor + fc + N_LAYERS * layer
        print(f"\nEstimated ANE per-round draft = {est:.2f} ms")
        print("  (dispatch floor + fc + 5 x layer; excludes lm_head/selector,")
        print("   which stay on GPU because bind() shares the target's tensors)")
        if mlx_full:
            ratio = est / mlx_full
            print(f"MLX GPU equivalent            = {mlx_full:.2f} ms")
            verdict = "ANE FASTER" if ratio < 1 else "ANE SLOWER"
            print(f"\n  ==> {verdict} by {abs(1 - ratio) * 100:.0f}%  (ratio {ratio:.2f}x)")


if __name__ == "__main__":
    main()
