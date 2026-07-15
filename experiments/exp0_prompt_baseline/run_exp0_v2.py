"""Exp 0 v2 — prompt-engineering baseline on the FULL pre-registered cohort.

Protocol (identical to v0, cohort and statistics per the 2026-07-15
pre-registration in decisions.md):
  1. base model summarizes the context (plain-text prompt, full context —
     the baseline is DELIBERATELY full-context: it measures what prompt
     engineering alone buys, no behavioral tokens involved);
  2. MCQ answered from the summary alone (option_loglik_full_context);
  3. MCQ answered from the full context (upper-bound reference);
  4. fact retrieval measured on the summary text.

Outputs:
  - per-example prediction records (JSONL) — the paired vectors Exp 2 needs;
  - cohort file (example_ids + dataset sha256) via CohortSelection, so Exp 2
    runs on EXACTLY the same items;
  - aggregate JSON with bootstrap CIs, per-source / per-distance breakdowns,
    and paired McNemar summary-vs-fullcontext.

Usage:
    python experiments/exp0_prompt_baseline/run_exp0_v2.py \
        --out results/exp0_v2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import TrainConfig  # noqa: E402
from src.data_contract import (  # noqa: E402
    CohortSelection,
    file_sha256,
    load_jsonl_validated,
)
from src.eval import (  # noqa: E402
    example_distance,
    fact_retrieval_accuracy,
    load_for_eval,
    option_loglik_full_context,
)
from src.stats import bootstrap_ci, mcnemar_exact, mean  # noqa: E402
from src.utils import resolve_device, set_seed  # noqa: E402

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
def generate(model, tokenizer, prompt: str, device, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def mcq_pick(model, tokenizer, prompt: str, options: list[str], device) -> int:
    scores = [option_loglik_full_context(model, tokenizer, prompt, opt, device)
              for opt in options]
    return scores.index(max(scores))


def summarize_records(records: list[dict]) -> dict:
    s_ok = [r["summary_correct"] for r in records]
    f_ok = [r["full_context_correct"] for r in records]
    facts = [r["summary_fact_retrieval"] for r in records
             if r["summary_fact_retrieval"] == r["summary_fact_retrieval"]]  # not NaN

    def by(key):
        groups: dict[str, list[dict]] = {}
        for r in records:
            groups.setdefault(str(r[key]), []).append(r)
        return {k: {
            "n": len(v),
            "mcq_from_summary": mean([r["summary_correct"] for r in v]),
            "mcq_full_context": mean([r["full_context_correct"] for r in v]),
        } for k, v in sorted(groups.items())}

    return {
        "n": len(records),
        "mcq_from_summary": bootstrap_ci([float(x) for x in s_ok]),
        "mcq_full_context": bootstrap_ci([float(x) for x in f_ok]),
        "summary_fact_retrieval": bootstrap_ci(facts),
        "mcnemar_summary_vs_full_context": mcnemar_exact(s_ok, f_ok),
        "by_source": by("source"),
        "by_distance": by("distance"),
        "by_type": by("type"),
        "chance_level": 0.25,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/processed/test.jsonl")
    p.add_argument("--out", default="results/exp0_v2.json")
    p.add_argument("--records", default="results/exp0_v2_records.jsonl")
    p.add_argument("--cohort", default="results/exp0_v2_cohort.json")
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--limit", type=int, default=None,
                   help="SMOKE ONLY: cap examples; refuses the default --out")
    args = p.parse_args()

    if args.limit and args.out == "results/exp0_v2.json":
        raise SystemExit("--limit is for smoke runs: pass a non-default --out")

    set_seed(42)
    cfg = TrainConfig()
    device = resolve_device("auto")

    examples = load_jsonl_validated(args.data)
    missing_mcq = [ex["example_id"] for ex in examples
                   if not (ex["meta"].get("options"))]
    if missing_mcq:
        raise SystemExit(f"{len(missing_mcq)} rows without MCQ — cohort must be "
                         f"fully annotated (e.g. {missing_mcq[:3]})")
    if args.limit:
        examples = examples[:args.limit]

    # Freeze the cohort BY ID — Exp 2 must run on exactly these items.
    cohort = CohortSelection(
        example_ids=[ex["example_id"] for ex in examples],
        dataset_sha256=file_sha256(args.data),
        description="Exp 0 v2 / Exp 2 verdict cohort (pre-registered "
                    "2026-07-15: all MCQ-annotated v2 test rows)")
    Path(args.cohort).parent.mkdir(parents=True, exist_ok=True)
    cohort.save(args.cohort)

    print(f"cohort: {len(examples)} examples, data sha {cohort.dataset_sha256[:12]}…",
          flush=True)
    model, tokenizer = load_for_eval(None, cfg, device)  # BASE model: the baseline

    records = []
    rec_path = Path(args.records)
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rec_path, "w", encoding="utf-8") as rec_file:
        for ex in tqdm(examples, desc="exp0-v2"):
            meta = ex["meta"]
            summary = generate(model, tokenizer,
                               SUMMARIZE_PROMPT.format(context=ex["context"]),
                               device, args.max_new_tokens)
            s_pick = mcq_pick(model, tokenizer,
                              ANSWER_PROMPT.format(summary=summary,
                                                   question=meta["question"]),
                              meta["options"], device)
            f_pick = mcq_pick(model, tokenizer,
                              FULL_CONTEXT_PROMPT.format(context=ex["context"],
                                                         question=meta["question"]),
                              meta["options"], device)
            record = {
                "example_id": ex["example_id"],
                "content_id": ex["content_id"],
                "source": meta["source"],
                "type": ex["type"],
                "distance": example_distance(ex),
                "answer_idx": meta["answer_idx"],
                "summary_pick": s_pick,
                "full_context_pick": f_pick,
                "summary_correct": s_pick == meta["answer_idx"],
                "full_context_correct": f_pick == meta["answer_idx"],
                "summary_fact_retrieval": fact_retrieval_accuracy(
                    summary, meta.get("facts", [])),
                "summary_text": summary,
            }
            records.append(record)
            rec_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            rec_file.flush()

    results = summarize_records(records)
    results["protocol"] = {
        "model": cfg.model_name, "checkpoint": "base",
        "attention_mode": "full_context (prompt baseline, by protocol)",
        "max_new_tokens": args.max_new_tokens,
        "cohort_file": args.cohort,
        "dataset_sha256": cohort.dataset_sha256,
        "records_file": str(rec_path),
        "seed": 42,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    concise = {k: results[k] for k in
               ("n", "mcq_from_summary", "mcq_full_context",
                "summary_fact_retrieval", "mcnemar_summary_vs_full_context")}
    print(json.dumps(concise, indent=2), flush=True)
    print(f"EXP0 V2 DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
