"""Exp 3 — Linear probing of [COMPRESS] hidden states, with mandatory controls.

A probe on [COMPRESS] is meaningful ONLY if all controls fail (CLAUDE.md):
  1. random token in context            -> must fail
  2. [RECALL] on the UNTRAINED model    -> must fail
  3. shuffled positions                 -> must fail
  4. last context token pre-[COMPRESS]  -> if it matches [COMPRESS], the token
     is copying, not compressing

Labeled examples need meta.label (e.g. sentiment/topic class).

Usage:
    python -m src.probe --data data/processed/probe.jsonl \
        --checkpoint results/checkpoints/run/best \
        --out results/exp3_probing/probe_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.config import COMPRESS_TOKEN, RECALL_TOKEN, TrainConfig
from src.dataset import render_prompt
from src.eval import load_for_eval
from src.utils import load_jsonl, resolve_device, set_seed

CONDITIONS = ["compress", "random_context", "recall_untrained",
              "shuffled_positions", "pre_compress"]


@torch.no_grad()
def extract_states(model, tokenizer, examples: list[dict], device,
                   mode: str, layer: int = -1, rng: np.random.Generator | None = None):
    """Hidden state at a position chosen by `mode`, plus probe labels."""
    rng = rng or np.random.default_rng(42)
    compress_id = tokenizer.convert_tokens_to_ids(COMPRESS_TOKEN)
    recall_id = tokenizer.convert_tokens_to_ids(RECALL_TOKEN)

    feats, labels, compress_positions = [], [], []
    for ex in tqdm(examples, desc=f"extract[{mode}]"):
        ids = tokenizer(render_prompt(ex), return_tensors="pt",
                        add_special_tokens=False)["input_ids"].to(device)
        row = ids[0].tolist()
        if compress_id not in row:
            continue
        c_pos = row.index(compress_id)
        compress_positions.append(c_pos)

        if mode in ("compress", "shuffled_positions"):
            pos = c_pos  # shuffled_positions permutes afterwards
        elif mode == "random_context":
            pos = int(rng.integers(0, max(c_pos, 1)))
        elif mode == "pre_compress":
            pos = max(c_pos - 1, 0)
        elif mode == "recall_untrained":
            pos = row.index(recall_id) if recall_id in row else len(row) - 1
        else:
            raise ValueError(mode)

        h = model(input_ids=ids, output_hidden_states=True).hidden_states[layer]
        feats.append(h[0, pos].float().cpu().numpy())
        labels.append(ex["meta"]["label"])

    X, y = np.stack(feats), np.array(labels)
    if mode == "shuffled_positions":
        # control 3: each example gets the [COMPRESS] state of ANOTHER example
        # (position/content pairing destroyed) — re-extract with permuted rows
        X = X[rng.permutation(len(X))]
    return X, y


def run_probe(X: np.ndarray, y: np.ndarray, seed: int = 42) -> dict:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                              random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_tr, y_tr)
    classes, counts = np.unique(y, return_counts=True)
    return {
        "accuracy": float(clf.score(X_te, y_te)),
        "majority_baseline": float(counts.max() / counts.sum()),
        "n": int(len(y)),
        "_clf": clf,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="labeled jsonl (meta.label)")
    p.add_argument("--checkpoint", required=True, help="trained adapter dir")
    p.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--layer", type=int, default=-1)
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument("--out", type=str, default="results/exp3_probing/probe_results.json")
    args = p.parse_args()

    set_seed(42)
    cfg = TrainConfig(model_name=args.model_name)
    device = resolve_device("auto")
    examples = [ex for ex in load_jsonl(args.data, max_examples=args.max_samples)
                if (ex.get("meta") or {}).get("label")]
    if len(examples) < 20:
        raise SystemExit(f"Only {len(examples)} labeled examples — need more.")

    trained, tokenizer = load_for_eval(args.checkpoint, cfg, device)
    untrained, _ = load_for_eval(None, cfg, device)  # base + tokens, no fine-tune

    results = {}
    probe_artifacts = None
    for mode in CONDITIONS:
        model = untrained if mode == "recall_untrained" else trained
        X, y = extract_states(model, tokenizer, examples, device, mode, args.layer)
        res = run_probe(X, y)
        clf = res.pop("_clf")
        results[mode] = res
        print(f"{mode:>20}: acc={res['accuracy']:.3f} "
              f"(majority={res['majority_baseline']:.3f}, n={res['n']})")
        if mode == "compress":
            probe_artifacts = clf

    # verdict per CLAUDE.md: compress must beat all controls clearly
    acc = {m: results[m]["accuracy"] for m in CONDITIONS}
    margin = 0.10
    results["verdict"] = {
        "compress_beats_controls": all(
            acc["compress"] >= acc[m] + margin
            for m in CONDITIONS if m != "compress"),
        "copying_suspect": acc["pre_compress"] >= acc["compress"] - 0.05,
        "margin_used": margin,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    # save probe direction for Exp 5 (causal intervention)
    np.savez(out.with_suffix(".npz"),
             coef=probe_artifacts.coef_, intercept=probe_artifacts.intercept_,
             classes=probe_artifacts.classes_, layer=args.layer)
    print(f"\nSaved: {out} and {out.with_suffix('.npz')}")
    print(json.dumps(results["verdict"], indent=2))


if __name__ == "__main__":
    main()
