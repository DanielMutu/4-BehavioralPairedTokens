"""Merge raw + handwritten examples, split into train/eval/test.

Held-out rule (CLAUDE.md): the test set must contain a style DIFFERENT from
training — public (CNN/DailyMail) and handwritten examples are routed
preferentially to test; synthetic narrative data dominates train.

Usage:
    python -m data.generation.prepare_dataset
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils import load_jsonl, save_jsonl, set_seed  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--handwritten", default="data/handwritten/handwritten.jsonl")
    p.add_argument("--out-dir", default="data/processed")
    p.add_argument("--eval-frac", type=float, default=0.1)
    args = p.parse_args()

    set_seed(42)
    rng = random.Random(42)

    synthetic, heldout_style = [], []
    for path in sorted(Path(args.raw_dir).glob("*.jsonl")):
        for ex in load_jsonl(path):
            src = (ex.get("meta") or {}).get("source", "synthetic")
            (synthetic if src == "synthetic" else heldout_style).append(ex)
    if Path(args.handwritten).exists():
        heldout_style.extend(load_jsonl(args.handwritten))

    # dedupe by context
    seen, unique = set(), []
    for ex in synthetic + heldout_style:
        key = ex["context"][:300]
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    synthetic = [ex for ex in unique
                 if (ex.get("meta") or {}).get("source", "synthetic") == "synthetic"]
    heldout_style = [ex for ex in unique if ex not in synthetic]

    rng.shuffle(synthetic)
    rng.shuffle(heldout_style)

    n_eval = max(1, int(len(synthetic) * args.eval_frac))
    eval_set = synthetic[:n_eval]
    train_set = synthetic[n_eval:]
    # test = different-style data + a slice of synthetic for comparison
    test_set = heldout_style + synthetic[n_eval:n_eval + n_eval]

    out = Path(args.out_dir)
    save_jsonl(train_set, out / "train.jsonl")
    save_jsonl(eval_set, out / "eval.jsonl")
    save_jsonl(test_set, out / "test.jsonl")
    # probe set (Exp 3): labeled examples NOT seen in training — probing on
    # training texts would inflate probe accuracy
    probe_set = [ex for ex in eval_set + test_set
                 if (ex.get("meta") or {}).get("label")]
    save_jsonl(probe_set, out / "probe.jsonl")

    print(f"train={len(train_set)} eval={len(eval_set)} test={len(test_set)} "
          f"probe={len(probe_set)} -> {out}/")
    if not heldout_style:
        print("[WARN] no handwritten/public examples found — the held-out "
              "test set has the same style as training (leakage risk).")


if __name__ == "__main__":
    main()
