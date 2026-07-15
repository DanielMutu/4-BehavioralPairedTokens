"""Evaluation: perplexity (Exp 1), recall quality metrics (Exp 2).

Metrics:
- WikiText-2 perplexity — checks fine-tuning did not degrade the base model
- fact retrieval accuracy — predefined facts found in the generated recall
- multiple-choice QA — scored by option log-likelihood (most reliable)
- ROUGE — indicative only (noisy, per CLAUDE.md)

Usage:
    python -m src.eval --task wikitext [--checkpoint results/checkpoints/run/best]
    python -m src.eval --task recall --data data/processed/test.jsonl --checkpoint ...
    python -m src.eval --task mcq --data data/processed/test.jsonl --checkpoint ...
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch
from tqdm import tqdm

from src.config import TrainConfig
from src.dataset import render_prompt
from src.model import setup_model_and_tokenizer
from src.utils import load_jsonl, resolve_device, set_seed


def load_for_eval(checkpoint: str | None, cfg: TrainConfig, device):
    """Base model when checkpoint is None, otherwise base + trained adapter."""
    model, tokenizer = setup_model_and_tokenizer(cfg, with_lora=False)
    if checkpoint:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint)
    model.to(device).eval()
    return model, tokenizer


# ---------------------------------------------------------------- wikitext

@torch.no_grad()
def wikitext_perplexity(model, tokenizer, device, max_samples: int = 200,
                        block_size: int = 1024) -> float:
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]

    nll, n_tokens = 0.0, 0
    n_blocks = min(max_samples, len(ids) // block_size)
    for i in tqdm(range(n_blocks), desc="wikitext ppl"):
        block = ids[i * block_size:(i + 1) * block_size].unsqueeze(0).to(device)
        out = model(input_ids=block, labels=block)
        nll += out.loss.item() * (block.shape[1] - 1)
        n_tokens += block.shape[1] - 1
    return math.exp(nll / max(n_tokens, 1))


# ---------------------------------------------------------------- recall

@torch.no_grad()
def generate_recall(model, tokenizer, example: dict, device,
                    max_new_tokens: int = 160) -> str:
    prompt = render_prompt(example)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def fact_retrieval_accuracy(generated: str, facts: list[str]) -> float:
    """Fraction of predefined facts explicitly present in the generation."""
    if not facts:
        return float("nan")
    gen = _normalize(generated)
    return sum(_normalize(f) in gen for f in facts) / len(facts)


def rouge_l(generated: str, reference: str) -> float:
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return float("nan")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, generated)["rougeL"].fmeasure


def eval_recall(model, tokenizer, examples: list[dict], device) -> dict:
    fact_scores, rouge_scores, per_distance = [], [], {}
    for ex in tqdm(examples, desc="recall eval"):
        gen = generate_recall(model, tokenizer, ex, device)
        facts = (ex.get("meta") or {}).get("facts", [])
        f = fact_retrieval_accuracy(gen, facts)
        r = rouge_l(gen, ex["target"])
        if not math.isnan(f):
            fact_scores.append(f)
            dist = (ex.get("meta") or {}).get("distance", 0)
            per_distance.setdefault(dist, []).append(f)
        if not math.isnan(r):
            rouge_scores.append(r)

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    return {
        "fact_retrieval_accuracy": mean(fact_scores),
        "rouge_l": mean(rouge_scores),
        # Exp 2: recall quality as a function of COMPRESS->RECALL distance
        "fact_accuracy_by_distance": {str(d): mean(v) for d, v in sorted(per_distance.items())},
        "n_examples": len(examples),
    }


# ---------------------------------------------------------------- MCQ

@torch.no_grad()
def option_loglik(model, tokenizer, prompt: str, option: str, device) -> float:
    """Mean per-token log-likelihood of `option` given `prompt`."""
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    o_ids = tokenizer(option, add_special_tokens=False)["input_ids"]
    ids = torch.tensor([p_ids + o_ids], device=device)
    labels = ids.clone()
    labels[0, :len(p_ids)] = -100
    out = model(input_ids=ids, labels=labels)
    return -out.loss.item()


def eval_mcq(model, tokenizer, examples: list[dict], device) -> dict:
    """examples need meta.question, meta.options, meta.answer_idx."""
    correct, total = 0, 0
    for ex in tqdm(examples, desc="mcq eval"):
        meta = ex.get("meta") or {}
        if "options" not in meta:
            continue
        prompt = render_prompt(ex) + f"Question: {meta['question']}\nAnswer: "
        scores = [option_loglik(model, tokenizer, prompt, opt, device)
                  for opt in meta["options"]]
        correct += int(scores.index(max(scores)) == meta["answer_idx"])
        total += 1
    return {"mcq_accuracy": correct / total if total else float("nan"), "n_mcq": total}


# ------------------------------------------------------------ downstream (Exp 1)
# Stability check: adding the behavioral tokens + LoRA must not degrade general
# capability. Both benchmarks are scored by length-normalized completion
# log-likelihood (same protocol as option_loglik), so the absolute numbers are
# cloze-style and need not match published leaderboards — only the base-vs-trained
# DELTA matters. GSM8K is intentionally omitted: it is generative, sits near
# chance on a 0.5B base model (no headroom to detect degradation), and is slow on
# CPU. See experiments/decisions.md.


def _mc_accuracy(model, tokenizer, items, device, desc: str) -> float:
    """items: list of (prompt, options, answer_idx). Argmax mean-token loglik."""
    correct = 0
    for prompt, options, answer_idx in tqdm(items, desc=desc):
        scores = [option_loglik(model, tokenizer, prompt, opt, device) for opt in options]
        correct += int(scores.index(max(scores)) == answer_idx)
    return correct / len(items) if items else float("nan")


def eval_hellaswag(model, tokenizer, device, max_samples: int = 300) -> tuple[float, int]:
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation")
    items = []
    for ex in ds:
        if ex["label"] == "":  # unlabeled rows
            continue
        prompt = (ex["activity_label"] + ": " + ex["ctx"]).strip()
        items.append((prompt, [" " + e.strip() for e in ex["endings"]], int(ex["label"])))
        if len(items) >= max_samples:
            break
    return _mc_accuracy(model, tokenizer, items, device, "hellaswag"), len(items)


def eval_mmlu(model, tokenizer, device, max_samples: int = 300) -> tuple[float, int]:
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=42)
    items = []
    for ex in ds:
        prompt = f"{ex['question'].strip()}\nAnswer:"
        items.append((prompt, [" " + c.strip() for c in ex["choices"]], int(ex["answer"])))
        if len(items) >= max_samples:
            break
    return _mc_accuracy(model, tokenizer, items, device, "mmlu"), len(items)


def eval_downstream(model, tokenizer, device, max_samples: int = 300) -> dict:
    hs_acc, hs_n = eval_hellaswag(model, tokenizer, device, max_samples)
    mmlu_acc, mmlu_n = eval_mmlu(model, tokenizer, device, max_samples)
    return {
        "hellaswag_accuracy": hs_acc, "n_hellaswag": hs_n,
        "mmlu_accuracy": mmlu_acc, "n_mmlu": mmlu_n,
    }


# ---------------------------------------------------------------- CLI

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True,
                   choices=["wikitext", "recall", "mcq", "downstream"])
    p.add_argument("--checkpoint", type=str, default=None,
                   help="adapter dir; omit to evaluate the base model")
    p.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--data", type=str, help="jsonl for recall/mcq tasks")
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--out", type=str, default=None, help="results json path")
    args = p.parse_args()

    set_seed(42)
    cfg = TrainConfig(model_name=args.model_name)
    device = resolve_device("auto")
    model, tokenizer = load_for_eval(args.checkpoint, cfg, device)

    if args.task == "wikitext":
        results = {"wikitext2_perplexity": wikitext_perplexity(
            model, tokenizer, device, max_samples=args.max_samples)}
    elif args.task == "downstream":
        results = eval_downstream(model, tokenizer, device, max_samples=args.max_samples)
    else:
        examples = load_jsonl(args.data, max_examples=args.max_samples)
        fn = eval_recall if args.task == "recall" else eval_mcq
        results = fn(model, tokenizer, examples, device)

    results["checkpoint"] = args.checkpoint or "base"
    print(json.dumps(results, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
