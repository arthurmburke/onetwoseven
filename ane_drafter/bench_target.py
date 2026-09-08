"""Measure the target's verify cost — decides whether draft cost matters at all.

Speculative speedup = accepted_per_round / (t_draft + t_verify).

If t_verify >> t_draft, the drafter's *cost* is irrelevant and only its
acceptance rate matters — in which case DFlash2 (hidden-state conditioned,
higher acceptance) beats a cheap standalone drafter and the ANE work is moot.

This times a block-8 forward through Qwen3.8-27B-8bit, which is what one
verification step costs.
"""

import time

import mlx.core as mx

MODEL = "mlx-community/Qwen3.8-27B-8bit"
BLOCK = 8


def main():
    from mlx_vlm.utils import load

    print(f"loading {MODEL} (~29 GB) ...")
    t0 = time.perf_counter()
    model, processor = load(MODEL)
    print(f"  loaded in {time.perf_counter() - t0:.0f}s")

    lm = getattr(model, "language_model", model)
    ids = mx.array([[1234] * BLOCK])
    ids1 = mx.array([[1234]])

    def fwd(x):
        out = lm(x)
        logits = out.logits if hasattr(out, "logits") else out
        mx.eval(logits)

    for label, x in (("single token (decode)", ids1), (f"block of {BLOCK} (verify)", ids)):
        for _ in range(3):
            fwd(x)
        ts = []
        for _ in range(10):
            t0 = time.perf_counter()
            fwd(x)
            ts.append((time.perf_counter() - t0) * 1000)
        ts.sort()
        print(f"  {label:24s} {ts[len(ts)//2]:7.2f} ms  (min {ts[0]:.2f})")

    print("\nInterpretation: verify ~= single-token decode confirms the model is")
    print("bandwidth-bound, which is what makes speculation profitable at all.")


if __name__ == "__main__":
    main()
