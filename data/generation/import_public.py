"""Import CNN/DailyMail summaries as NON-synthetic examples (anti-leakage).

These human-written summaries break the dependence on LLM generators'
compression style. Used both for training mix and for the held-out test set
with a different style from the synthetic narrative data.

Usage:
    python -m data.generation.import_public --n 300 --split train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils import save_jsonl, set_seed  # noqa: E402

MAX_CONTEXT_CHARS = 2500  # keep contexts small for 0.5B + 4GB GPU


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--split", default="train", choices=["train", "validation", "test"])
    p.add_argument("--type", default="A", choices=["A", "B"])
    p.add_argument("--out", default=None)
    args = p.parse_args()

    set_seed(42)
    from datasets import load_dataset
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split=args.split,
                      streaming=True)

    examples = []
    for row in ds:
        article = row["article"].strip()
        if len(article) > MAX_CONTEXT_CHARS:
            continue
        examples.append({
            "type": args.type,
            "context": article,
            "filler": "",
            "target": row["highlights"].strip(),
            "meta": {"source": "cnn_dailymail", "generator": "human",
                     "distance": 0, "facts": [], "label": None},
        })
        if len(examples) >= args.n:
            break

    out = args.out or f"data/raw/public_cnndm_{args.split}.jsonl"
    save_jsonl(examples, out)
    print(f"Done: {len(examples)} examples -> {out}")


if __name__ == "__main__":
    main()
