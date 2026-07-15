"""Exp 1 — Stability baseline.

Does adding the behavioral tokens ([COMPRESS]/[RECALL]/[REASON]) + a LoRA
adapter degrade the base model's general capability? We compare the untouched
base model against the trained checkpoint on:

  - WikiText-2 perplexity      (primary: language-modeling ability)
  - HellaSwag, MMLU accuracy   (secondary: structured downstream tasks)

Pre-registered gate (see experiments/decisions.md, 2026-06-13):
  PASS iff
    - WikiText-2 perplexity relative increase <= 5%, AND
    - each downstream accuracy drops by <= 2.0 points (absolute).
  Both benchmarks scored by length-normalized completion log-likelihood, so
  the absolute numbers are cloze-style; only the base-vs-trained DELTA is the
  signal. A PASS clears Exp 2; a FAIL means the fine-tuning recipe hurt the
  model and must be fixed before measuring recall.

Usage (from project root, after training a checkpoint):
    python experiments/exp1_stability/run_exp1.py \
        --checkpoint results/checkpoints/exp1-stability/best \
        --downstream-samples 200 --wikitext-samples 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import TrainConfig  # noqa: E402
from src.eval import (  # noqa: E402
    eval_downstream,
    load_for_eval,
    wikitext_perplexity,
)
from src.utils import resolve_device, set_seed  # noqa: E402

# Pre-registered thresholds (do not tune after seeing results).
MAX_PPL_REL_INCREASE = 0.05   # WikiText-2 perplexity may rise at most 5%
MAX_DOWNSTREAM_DROP = 2.0     # each downstream accuracy may fall at most 2 pts


def evaluate_model(checkpoint, cfg, device, wikitext_samples, downstream_samples):
    model, tokenizer = load_for_eval(checkpoint, cfg, device)
    res = {"wikitext2_perplexity": wikitext_perplexity(
        model, tokenizer, device, max_samples=wikitext_samples)}
    res.update(eval_downstream(model, tokenizer, device, max_samples=downstream_samples))
    del model
    return res


def verdict(base: dict, trained: dict) -> dict:
    ppl_rel = (trained["wikitext2_perplexity"] - base["wikitext2_perplexity"]) \
        / base["wikitext2_perplexity"]
    hs_drop = (base["hellaswag_accuracy"] - trained["hellaswag_accuracy"]) * 100
    mmlu_drop = (base["mmlu_accuracy"] - trained["mmlu_accuracy"]) * 100

    ppl_ok = ppl_rel <= MAX_PPL_REL_INCREASE
    hs_ok = hs_drop <= MAX_DOWNSTREAM_DROP
    mmlu_ok = mmlu_drop <= MAX_DOWNSTREAM_DROP
    passed = ppl_ok and hs_ok and mmlu_ok
    return {
        "wikitext2_ppl_rel_increase": round(ppl_rel, 4),
        "hellaswag_drop_pts": round(hs_drop, 2),
        "mmlu_drop_pts": round(mmlu_drop, 2),
        "ppl_ok": ppl_ok, "hellaswag_ok": hs_ok, "mmlu_ok": mmlu_ok,
        "PASS": passed,
        "thresholds": {
            "max_ppl_rel_increase": MAX_PPL_REL_INCREASE,
            "max_downstream_drop_pts": MAX_DOWNSTREAM_DROP,
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True,
                   help="trained adapter dir, e.g. results/checkpoints/exp1-stability/best")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--wikitext-samples", type=int, default=100)
    p.add_argument("--downstream-samples", type=int, default=200)
    p.add_argument("--out", default="results/exp1_stability.json")
    args = p.parse_args()

    set_seed(42)
    cfg = TrainConfig(model_name=args.model_name)
    device = resolve_device("auto")

    print(">>> Evaluating BASE model (no adapter)")
    base = evaluate_model(None, cfg, device, args.wikitext_samples, args.downstream_samples)
    print(">>> Evaluating TRAINED model (+ adapter)")
    trained = evaluate_model(args.checkpoint, cfg, device,
                             args.wikitext_samples, args.downstream_samples)

    results = {
        "model": args.model_name,
        "checkpoint": args.checkpoint,
        "base": base,
        "trained": trained,
        "verdict": verdict(base, trained),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    print(json.dumps(results, indent=2))
    v = results["verdict"]
    print("\n--- EXP 1 VERDICT ---")
    print(f"WikiText-2 ppl: {base['wikitext2_perplexity']:.3f} -> "
          f"{trained['wikitext2_perplexity']:.3f} "
          f"({v['wikitext2_ppl_rel_increase']:+.1%})  [{'ok' if v['ppl_ok'] else 'FAIL'}]")
    print(f"HellaSwag: {base['hellaswag_accuracy']:.3f} -> "
          f"{trained['hellaswag_accuracy']:.3f}  (drop {v['hellaswag_drop_pts']:+.2f} pts) "
          f"[{'ok' if v['hellaswag_ok'] else 'FAIL'}]")
    print(f"MMLU: {base['mmlu_accuracy']:.3f} -> {trained['mmlu_accuracy']:.3f}  "
          f"(drop {v['mmlu_drop_pts']:+.2f} pts) [{'ok' if v['mmlu_ok'] else 'FAIL'}]")
    print(f"\n{'PASS — Exp 2 cleared.' if v['PASS'] else 'FAIL — fix the recipe before Exp 2.'}")
    print("Log the decision in experiments/decisions.md.")


if __name__ == "__main__":
    main()
