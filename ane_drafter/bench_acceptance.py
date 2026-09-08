"""Measure DFlash2's real acceptance rate and per-round draft cost.

Answers the two numbers the ANE analysis left open:

  1. accepted tokens/round  - decides whether a cheap standalone drafter could
                              ever break even (it needs ~81% of this).
  2. real draft cost/round  - my synthetic estimate was 19.7 ms; this measures
                              the actual bound method.

Also sweeps block_size, because the target verify measured 58.79 ms for 1
token vs 76.88 ms for 8 -- only +31% for 8x the positions. If verify is that
flat, drafting harder is nearly free and block_size is the real lever.

Greedy (temp=0) throughout so acceptance is deterministic and reproducible.
"""

import argparse
import time

TARGET = "mlx-community/Qwen3.8-27B-8bit"
DRAFTER = "z-lab/Qwen3.8-27B-DFlash2"

# Representative of the actual workload: an OpenCode-style coding agent.
PROMPTS = [
    "Write a Python function that merges two sorted lists into one sorted list, with a docstring.",
    "Explain what a Python context manager is and show a minimal example implementing __enter__ and __exit__.",
    "Refactor this to remove the nested loop:\n\nfor i in range(len(a)):\n    for j in range(len(b)):\n        if a[i] == b[j]:\n            out.append(a[i])",
    "Write a TypeScript interface for a paginated API response containing a list of users, then a function that fetches page N.",
    "What does this shell command do, step by step: find . -name '*.tmp' -mtime +7 -delete",
]


def timed_draft_block(draft_model, acc):
    """Wrap the bound draft method(s) to accumulate real wall time per call.

    Greedy decoding dispatches to ``draft_block_greedy`` when present
    (dflash.py:346), so wrapping only ``draft_block`` silently measures nothing.
    """
    names = [n for n in ("draft_block", "draft_block_greedy") if hasattr(draft_model, n)]
    originals = {n: getattr(draft_model, n) for n in names}

    def make(original):
        def wrapped(*args, **kwargs):
            import mlx.core as mx

            t0 = time.perf_counter()
            out = original(*args, **kwargs)
            mx.eval(out)  # force materialization so the timing is real
            acc["ms"] += (time.perf_counter() - t0) * 1000.0
            acc["calls"] += 1
            return out

        return wrapped

    for n, orig in originals.items():
        setattr(draft_model, n, make(orig))
    return originals


def run_one(model, processor, draft_model, kind, prompt, max_tokens, block_size):
    import mlx.core as mx
    from mlx_vlm import stream_generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.speculative.common import speculative_stats_snapshot, speculative_stats_since

    acc = {"ms": 0.0, "calls": 0}
    originals = timed_draft_block(draft_model, acc)
    snap = speculative_stats_snapshot(draft_model)

    formatted = apply_chat_template(processor, model.config, prompt, num_images=0)
    kwargs = dict(draft_model=draft_model, draft_kind=kind, max_tokens=max_tokens, temperature=0.0)
    if block_size is not None:
        kwargs["draft_block_size"] = block_size

    t0 = time.perf_counter()
    n_tok = 0
    try:
        for res in stream_generate(model, processor, formatted, **kwargs):
            n_tok += 1
    finally:
        for _n, _o in originals.items():
            setattr(draft_model, _n, _o)
    wall = time.perf_counter() - t0

    rounds, accepted, drafted = speculative_stats_since(draft_model, snap)
    return dict(
        wall_s=wall, tokens=n_tok, rounds=rounds, accepted=accepted, drafted=drafted,
        draft_ms=acc["ms"], draft_calls=acc["calls"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--blocks", default="none,4,6,8,12,16",
                    help="comma list of draft_block_size values; 'none' = config default")
    a = ap.parse_args()

    from mlx_vlm.utils import load
    from mlx_vlm.speculative.drafters import load_drafter, validate_drafter_compatibility

    print(f"loading target {TARGET} ...")
    model, processor = load(TARGET)
    print(f"loading drafter {DRAFTER} ...")
    draft_model, kind = load_drafter(DRAFTER)
    validate_drafter_compatibility(model, draft_model, kind)
    if hasattr(draft_model, "bind"):
        draft_model.bind(getattr(model, "language_model", model))
    print(f"  drafter kind={kind}  config.block_size={draft_model.config.block_size}\n")

    print(f"{'block':>6} {'tok/s':>8} {'acc/round':>10} {'accept%':>8} "
          f"{'draft ms':>9} {'round ms':>9} {'rounds':>7}")
    print("-" * 62)

    for spec in a.blocks.split(","):
        bs = None if spec.strip() == "none" else int(spec)
        tot = dict(wall=0.0, tokens=0, rounds=0, accepted=0, drafted=0, draft_ms=0.0)
        for p in PROMPTS:
            try:
                r = run_one(model, processor, draft_model, kind, p, a.max_tokens, bs)
            except Exception as e:  # a block size the drafter refuses
                print(f"{spec:>6}  failed: {type(e).__name__}: {str(e)[:60]}")
                tot = None
                break
            if r["rounds"] is None:
                continue
            tot["wall"] += r["wall_s"]; tot["tokens"] += r["tokens"]
            tot["rounds"] += r["rounds"]; tot["accepted"] += r["accepted"]
            tot["drafted"] += r["drafted"]; tot["draft_ms"] += r["draft_ms"]
        if tot is None or not tot["rounds"]:
            continue

        rounds = tot["rounds"]
        tok_s = tot["tokens"] / tot["wall"]
        acc_per_round = (tot["accepted"] + rounds) / rounds  # +1 bonus token/round
        accept_pct = 100 * tot["accepted"] / tot["drafted"] if tot["drafted"] else 0.0
        draft_ms = tot["draft_ms"] / rounds
        round_ms = tot["wall"] * 1000 / rounds
        print(f"{spec:>6} {tok_s:>8.1f} {acc_per_round:>10.2f} {accept_pct:>7.1f}% "
              f"{draft_ms:>9.2f} {round_ms:>9.2f} {rounds:>7}")

    print("\nacc/round includes the target's bonus token (accepted drafts + 1).")
    print("Break-even for a standalone drafter = acc/round x (1.49 + verify)/(draft + verify).")


if __name__ == "__main__":
    main()
