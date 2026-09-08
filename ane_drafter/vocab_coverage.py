"""Pick the drafter's pruned vocabulary size from real workload text.

The drafter only needs to *propose* tokens; `_speculative_walk` compares its
proposals against the target's own choice, so a token missing from the pruned
head is simply never proposed (that round's draft is rejected there). Correctness
is unaffected — only acceptance rate. So the question is purely empirical:
what N covers enough of the token distribution to keep acceptance high?

Corpus is this project's own code and prose, which is the right bias: the
workload is a coding agent, not open-domain chat.
"""

import argparse
import collections
import json
import pathlib
import random

CODE_EXT = {".py", ".ts", ".tsx", ".js", ".jsx", ".json", ".md", ".sh", ".toml", ".yaml", ".yml", ".css"}
SKIP_DIRS = {"node_modules", ".git", ".next", "__pycache__", ".wrangler", "dist", ".venv/lib/python3.12/site-packages/nvidia"}
MAX_BYTES = 200_000


def gather_corpus(roots, limit_files, seed=0):
    files = []
    for root in roots:
        for p in pathlib.Path(root).rglob("*"):
            if p.suffix not in CODE_EXT or not p.is_file():
                continue
            if any(s in str(p) for s in SKIP_DIRS):
                continue
            files.append(p)
    random.Random(seed).shuffle(files)
    files = files[:limit_files]
    texts, total = [], 0
    for p in files:
        try:
            t = p.read_text(errors="ignore")[:MAX_BYTES]
        except OSError:
            continue
        texts.append(t)
        total += len(t)
    return texts, total, len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--files", type=int, default=600)
    ap.add_argument("--out", default="ane_drafter/vocab_map.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"tokenizer vocab: {len(tok):,}")

    texts, nbytes, nfiles = gather_corpus(
        ["/Users/arthurburke/Developer/studio"], args.files
    )
    print(f"corpus: {nfiles} files, {nbytes/1e6:.1f} MB")

    counter = collections.Counter()
    for t in texts:
        counter.update(tok(t, add_special_tokens=False)["input_ids"])
    total = sum(counter.values())
    print(f"tokens: {total:,}   distinct: {len(counter):,}\n")

    ranked = [tid for tid, _ in counter.most_common()]
    cum, running = {}, 0
    counts_sorted = [c for _, c in counter.most_common()]
    for i, c in enumerate(counts_sorted):
        running += c
        cum[i + 1] = running / total

    print(f"{'N':>8} {'coverage':>10} {'head MB fp16 @h=768':>22}")
    for n in (4096, 8192, 16384, 32768, 65536, 131072):
        if n <= len(ranked):
            print(f"{n:>8} {cum[n]*100:>9.3f}% {n*768*2/1e6:>21.0f}")
        else:
            print(f"{n:>8} {'100.000':>9}% {n*768*2/1e6:>21.0f}  (exceeds distinct)")

    # Always include special tokens so the drafter can propose them.
    special = [i for i in tok.all_special_ids if i is not None]
    chosen_n = 32768
    keep = list(dict.fromkeys(ranked[:chosen_n] + special))
    payload = {
        "pruned_size": len(keep),
        "coverage": cum.get(min(chosen_n, len(ranked)), 1.0),
        "local_to_global": keep,
        "corpus_tokens": total,
        "full_vocab": len(tok),
    }
    pathlib.Path(args.out).write_text(json.dumps(payload))
    print(f"\nwrote {args.out}: {len(keep):,} ids, coverage {payload['coverage']*100:.3f}%")


if __name__ == "__main__":
    main()
