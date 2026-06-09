"""Exp 0 — Prompt-engineering baseline (GATING DECISION).

Before training anything: can the BASE model already do compress-then-recall
with an explicit prompt? Two-stage protocol:
  1. ask the base model for a very dense bullet summary of the text
  2. ask it to answer questions using ONLY the summary

Compared against the full-context upper bound. If the prompt baseline is
already comparable to what trained tokens could plausibly add, the project
loses value — DO NOT proceed (CLAUDE.md gating rule).

Usage (from project root):
    python experiments/exp0_prompt_baseline/run_exp0.py \
        --data data/processed/test.jsonl --max-samples 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval import fact_retrieval_accuracy, option_loglik  # noqa: E402
from src.utils import load_jsonl, resolve_device, set_seed  # noqa: E402

SUMMARIZE_PROMPT = (
    "{context}\n\n"
    "Summarize the text above in very dense bullet points, keeping every "
    "name, number and key fact:\n-"
)
ANSWER_PROMPT = (
    "Summary of a text:\n{summary}\n\n"
    "Based ONLY on the summary above, answer the question.\n"
    "Question: {question}\nAnswer: "
)
FULL_CONTEXT_PROMPT = "{context}\n\nQuestion: {question}\nAnswer: "


@torch.no_grad()
def generate(model, tokenizer, prompt: str, device, max_new_tokens: int = 160) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def mcq_pick(model, tokenizer, prompt: str, options: list[str], device) -> int:
    scores = [option_loglik(model, tokenizer, prompt, opt, device) for opt in options]
    return scores.index(max(scores))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--data", default="data/processed/test.jsonl")
    p.add_argument("--max-samples", type=int, default=50)
    p.add_argument("--out", default="results/exp0_results.json")
    args = p.parse_args()

    set_seed(42)
    device = resolve_device("auto")
    # plain base model, NO special tokens — this is the untouched baseline
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device).eval()

    examples = [ex for ex in load_jsonl(args.data, max_examples=args.max_samples * 3)
                if (ex.get("meta") or {}).get("options")][:args.max_samples]
    if not examples:
        raise SystemExit("No MCQ-annotated examples found — run dataset prep first.")

    fact_scores, mcq_summary, mcq_full, records = [], 0, 0, []
    for ex in tqdm(examples, desc="exp0"):
        meta = ex["meta"]
        summary = "-" + generate(
            model, tokenizer, SUMMARIZE_PROMPT.format(context=ex["context"]), device)

        f = fact_retrieval_accuracy(summary, meta.get("facts", []))
        if f == f:  # not NaN
            fact_scores.append(f)

        pick_s = mcq_pick(model, tokenizer,
                          ANSWER_PROMPT.format(summary=summary, question=meta["question"]),
                          meta["options"], device)
        pick_f = mcq_pick(model, tokenizer,
                          FULL_CONTEXT_PROMPT.format(context=ex["context"],
                                                     question=meta["question"]),
                          meta["options"], device)
        mcq_summary += int(pick_s == meta["answer_idx"])
        mcq_full += int(pick_f == meta["answer_idx"])
        records.append({"summary": summary, "fact_acc": f,
                        "mcq_from_summary_ok": pick_s == meta["answer_idx"],
                        "mcq_full_context_ok": pick_f == meta["answer_idx"]})

    n = len(examples)
    results = {
        "model": args.model_name,
        "n": n,
        "summary_fact_retrieval": sum(fact_scores) / max(len(fact_scores), 1),
        "mcq_from_summary": mcq_summary / n,
        "mcq_full_context_upper_bound": mcq_full / n,
        "chance_level": 0.25,
        "records": records,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(json.dumps({k: v for k, v in results.items() if k != "records"}, indent=2))
    print("\n--- GATING GUIDANCE ---")
    print("Compare these numbers with the trained-token runs (Exp 2).")
    print("If trained [COMPRESS]/[RECALL] cannot clearly beat 'mcq_from_summary'")
    print("and 'summary_fact_retrieval', the project premise fails: STOP HERE.")
    print("Log the decision in experiments/decisions.md.")


if __name__ == "__main__":
    main()
