"""Build disjoint v2 splits (train/eval/test/probe) + manifest from raw data.

v0 had a real leakage bug: `test_set = heldout_style + synthetic[n_eval:n_eval+n_eval]`
re-used the first slice of the train range, so 148 train contexts appeared in
both test and probe. v2 allocates *disjoint* synthetic segments, keys every
row by `content_id`, and refuses to write splits that share any content with
train (see src.data_contract.assert_disjoint).

Split layout (synthetic pool, deterministic shuffle):
    [ eval | test_in_style | train ]
Held-out-style rows (CNN/DailyMail + handwritten) go to test only.
probe = labeled rows from eval + test (a derived evaluation view, never train).

Usage:
    python -m data.generation.prepare_dataset
    python -m data.generation.prepare_dataset --check   # verify existing manifest
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data_contract import (  # noqa: E402
    ContractError,
    assert_disjoint,
    build_manifest,
    load_jsonl_validated,
    save_jsonl_atomic,
    seal_example,
    upgrade_legacy_example,
    verify_manifest,
    write_manifest,
)

SPLIT_ALGORITHM = "disjoint-segments-v2"


def dedupe_by_content(rows: list[dict]) -> tuple[list[dict], int]:
    seen: set[str] = set()
    unique: list[dict] = []
    for ex in rows:
        if ex["content_id"] in seen:
            continue
        seen.add(ex["content_id"])
        unique.append(ex)
    return unique, len(rows) - len(unique)


def build_splits(synthetic: list[dict], heldout_style: list[dict],
                 eval_frac: float, test_frac: float,
                 seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    synthetic = sorted(synthetic, key=lambda ex: ex["example_id"])  # order-independent input
    heldout_style = sorted(heldout_style, key=lambda ex: ex["example_id"])
    rng.shuffle(synthetic)
    rng.shuffle(heldout_style)

    n = len(synthetic)
    n_eval = max(1, int(n * eval_frac))
    n_test = max(1, int(n * test_frac))
    if n_eval + n_test >= n:
        raise ContractError(
            f"not enough synthetic rows ({n}) for eval_frac={eval_frac} "
            f"+ test_frac={test_frac}")

    eval_set = synthetic[:n_eval]
    test_in_style = synthetic[n_eval:n_eval + n_test]
    train_set = synthetic[n_eval + n_test:]
    test_set = heldout_style + test_in_style

    probe_set = [ex for ex in eval_set + test_set
                 if (ex.get("meta") or {}).get("label")]

    splits = {"train": train_set, "eval": eval_set,
              "test": test_set, "probe": probe_set}
    assert_disjoint(splits, protected="train")
    return splits


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--handwritten", default="data/handwritten/handwritten.jsonl")
    p.add_argument("--out-dir", default="data/processed")
    p.add_argument("--eval-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1,
                   help="fraction of synthetic routed to the in-style test slice")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--check", action="store_true",
                   help="verify the existing manifest instead of rebuilding")
    args = p.parse_args()

    out = Path(args.out_dir)
    manifest_path = out / "manifest.json"

    if args.check:
        manifest = verify_manifest(manifest_path)
        print(f"manifest OK: {manifest_path}")
        print(json.dumps(manifest["overlap"], indent=2))
        return

    inputs: dict[str, Path] = {}
    synthetic: list[dict] = []
    heldout_style: list[dict] = []
    for path in sorted(Path(args.raw_dir).glob("*.jsonl")):
        inputs[path.name] = path
        for ex in load_jsonl_validated(path, upgrade=True):
            src = ex["meta"]["source"]
            (synthetic if src == "synthetic" else heldout_style).append(ex)
    hw = Path(args.handwritten)
    if hw.exists():
        inputs[hw.name] = hw
        heldout_style.extend(load_jsonl_validated(hw, upgrade=True))

    synthetic, dup_syn = dedupe_by_content(synthetic)
    heldout_style, dup_ho = dedupe_by_content(heldout_style)
    # cross-pool duplicates count as held-out (they must never reach train)
    heldout_ids = {ex["content_id"] for ex in heldout_style}
    before = len(synthetic)
    synthetic = [ex for ex in synthetic if ex["content_id"] not in heldout_ids]
    dropped_cross = before - len(synthetic)

    splits = build_splits(synthetic, heldout_style,
                          args.eval_frac, args.test_frac, args.seed)

    split_paths: dict[str, Path] = {}
    for name, rows in splits.items():
        rows = [seal_example(dict(ex)) for ex in rows]
        path = out / f"{name}.jsonl"
        save_jsonl_atomic(rows, path)
        split_paths[name] = path

    manifest = build_manifest(
        splits=split_paths, inputs=inputs, seed=args.seed,
        split_algorithm=SPLIT_ALGORITHM,
        extra={"rejected": {"duplicate_synthetic": dup_syn,
                            "duplicate_heldout": dup_ho,
                            "synthetic_also_in_heldout": dropped_cross}})
    write_manifest(manifest, manifest_path)

    print(" ".join(f"{n}={len(r)}" for n, r in splits.items()), f"-> {out}/")
    print(f"manifest: {manifest_path}")
    print("overlap:", json.dumps(manifest["overlap"]))
    if not heldout_style:
        print("[WARN] no handwritten/public examples found — the held-out "
              "test set has the same style as training (leakage risk).")


if __name__ == "__main__":
    main()
