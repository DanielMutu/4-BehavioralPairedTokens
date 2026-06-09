"""Exp 5 — Causal intervention on [COMPRESS] hidden states.

Pipeline:
 1. take the hidden state of [COMPRESS] for a text X during prefill
 2. shift it along the probe direction of a target concept (e.g. opposite
    sentiment), via a forward hook on a decoder layer
 3. check whether the [RECALL] generation is coherent with the shift

If it is, the hidden state CAUSES the recall content (not just correlates).

Note: intervene on a MID-stack layer — the last hidden_states entry is
post-final-norm and does not propagate to later positions through attention.
Train the Exp 3 probe on the same mid layer you intervene on.

Usage:
    python -m src.intervention --data data/processed/probe.jsonl \
        --checkpoint results/checkpoints/run/best \
        --probe results/exp3_probing/probe_results.npz \
        --target-class positive --alpha 4.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.config import COMPRESS_TOKEN, TrainConfig
from src.dataset import render_prompt
from src.eval import generate_recall, load_for_eval
from src.utils import load_jsonl, resolve_device, set_seed


def get_decoder_layers(model):
    m = model
    if hasattr(m, "get_base_model"):
        m = m.get_base_model()
    while hasattr(m, "model"):
        m = m.model
    if not hasattr(m, "layers"):
        raise AttributeError("cannot locate decoder layers on this model")
    return m.layers


def probe_direction(npz_path: str, target_class: str) -> tuple[np.ndarray, int]:
    """Unit vector pointing toward `target_class` in probe space, + layer idx."""
    data = np.load(npz_path, allow_pickle=True)
    coef, classes = data["coef"], [str(c) for c in data["classes"]]
    layer = int(data["layer"])
    if target_class not in classes:
        raise ValueError(f"target {target_class!r} not in probe classes {classes}")
    if coef.shape[0] == 1:  # binary: coef points toward classes[1]
        direction = coef[0] if target_class == classes[1] else -coef[0]
    else:
        direction = coef[classes.index(target_class)]
    return direction / np.linalg.norm(direction), layer


class CompressIntervention:
    """Forward hook that shifts the [COMPRESS] hidden state during prefill."""

    def __init__(self, direction: torch.Tensor, position: int, alpha: float):
        self.direction = direction
        self.position = position
        self.alpha = alpha
        self.fired = False

    def __call__(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        # prefill only: the compress position exists in the sequence once
        if not self.fired and hidden.shape[1] > self.position:
            h = hidden[:, self.position]
            shift = self.alpha * h.norm(dim=-1, keepdim=True) \
                * self.direction.to(hidden.dtype).to(hidden.device)
            hidden[:, self.position] = h + shift
            self.fired = True
        return output


def intervene_and_generate(model, tokenizer, example: dict, device,
                           direction: np.ndarray, hook_layer: int,
                           alpha: float) -> dict:
    prompt_ids = tokenizer(render_prompt(example), add_special_tokens=False)["input_ids"]
    compress_id = tokenizer.convert_tokens_to_ids(COMPRESS_TOKEN)
    if compress_id not in prompt_ids:
        return {}
    pos = prompt_ids.index(compress_id)

    baseline = generate_recall(model, tokenizer, example, device)

    layers = get_decoder_layers(model)
    # probe layer is a hidden_states index ([0]=embeddings) -> decoder idx - 1
    idx = hook_layer - 1 if hook_layer > 0 else len(layers) + hook_layer
    idx = max(0, min(idx, len(layers) - 1))
    hook = CompressIntervention(torch.from_numpy(direction).float(), pos, alpha)
    handle = layers[idx].register_forward_hook(hook)
    try:
        intervened = generate_recall(model, tokenizer, example, device)
    finally:
        handle.remove()

    return {
        "context": example["context"][:200],
        "original_label": (example.get("meta") or {}).get("label"),
        "baseline_recall": baseline,
        "intervened_recall": intervened,
        "changed": baseline != intervened,
        "hook_decoder_layer": idx,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--probe", required=True, help=".npz saved by src.probe")
    p.add_argument("--target-class", required=True)
    p.add_argument("--alpha", type=float, default=4.0,
                   help="shift magnitude, relative to the hidden-state norm")
    p.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--max-samples", type=int, default=30)
    p.add_argument("--out", type=str, default="results/exp5_intervention/results.json")
    args = p.parse_args()

    set_seed(42)
    cfg = TrainConfig(model_name=args.model_name)
    device = resolve_device("auto")
    model, tokenizer = load_for_eval(args.checkpoint, cfg, device)
    direction, layer = probe_direction(args.probe, args.target_class)

    examples = load_jsonl(args.data, max_examples=args.max_samples)
    # most informative cases: examples whose label is NOT the target class
    examples = [ex for ex in examples
                if (ex.get("meta") or {}).get("label") != args.target_class] or examples

    records = []
    for ex in tqdm(examples, desc="intervention"):
        rec = intervene_and_generate(model, tokenizer, ex, device,
                                     direction, layer, args.alpha)
        if rec:
            records.append(rec)

    summary = {
        "target_class": args.target_class,
        "alpha": args.alpha,
        "probe_layer": layer,
        "n": len(records),
        "changed_fraction": sum(r["changed"] for r in records) / max(len(records), 1),
        "records": records,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"changed {summary['changed_fraction']:.0%} of {summary['n']} recalls "
          f"-> {out}")
    print("Inspect the records manually: 'changed' alone does not prove the "
          "shift is coherent with the target concept.")


if __name__ == "__main__":
    main()
