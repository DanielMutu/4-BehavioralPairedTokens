"""Exp 2 — recall through the bottleneck vs the prompt baseline (pre-registered).

Verdict cohort, criteria and conditions were pre-registered on 2026-07-15/16
(decisions.md) BEFORE any result existed:

  gating conditions (8):
    1 full_context_base   — base model, plain prompt (reused from Exp 0 v2)
    2 prompt_summary      — the baseline to beat (reused from Exp 0 v2)
    3 token_unmasked      — trained ckpt, tokens in prompt, FREE attention
                            (quantifies the copying contribution)
    4 true_bottleneck     — trained ckpt, bottleneck mask  ← THE condition
    5 anchor_removed      — post-anchor queries lose the anchor key too
    6 anchor_shuffled     — anchor column patched from ANOTHER example
    7 token_untrained     — base model + mean-init tokens, bottleneck mask
    8 anchor_only         — fillers blind to the anchor; [RECALL]+ read it
  diagnostics (2, non-gating):
    9 anchor_mean         — anchor column replaced by norm-matched context mean
   10 forced_relay        — fillers read the anchor; [RECALL]+ blind to it

  primary:   (4) beats (2) on the same 541 items — McNemar p < 0.05, diff > 0
  secondary: paired-diff 95% bootstrap CI within [-3, +3] points
  mandatory: separate numbers for the out-of-style CNN/DailyMail partition

MCQ is the gating metric (fact retrieval deferred: reference-decoder
generation for 541x10 conditions is not feasible on CPU; declared in
decisions.md). Records are written per (condition, example) and flushed, so
the run is resumable with --resume.

Usage:
    python experiments/exp2_distance/run_exp2.py \
        --checkpoint results/checkpoints/exp1b-bottleneck-v2/best \
        --out results/exp2_results.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.bottleneck import (  # noqa: E402
    build_anchor_only_mask,
    build_anchor_removed_mask,
    build_bottleneck_mask,
    build_causal_mask,
    build_forced_relay_mask,
    validate_layout,
)
from src.config import TrainConfig  # noqa: E402
from src.data_contract import CohortSelection, load_jsonl_validated  # noqa: E402
from src.dataset import render_prompt  # noqa: E402
from src.eval import example_distance, load_for_eval  # noqa: E402
from src.patching import AnchorPatcher  # noqa: E402
from src.stats import bootstrap_ci, bootstrap_ci_paired_diff, mcnemar_exact, mean  # noqa: E402
from src.utils import resolve_device, set_seed  # noqa: E402

SEED = 42
TIE_BAND_PTS = 3.0        # pre-registered: parity if 95% CI(diff) ⊆ [-3, +3] pts
GATING = ["full_context_base", "prompt_summary", "token_unmasked",
          "true_bottleneck", "anchor_removed", "anchor_shuffled",
          "token_untrained", "anchor_only"]
DIAGNOSTIC = ["anchor_mean", "forced_relay"]


def mask_for(condition: str, attn2d, cpos, rpos, dtype):
    if condition in ("true_bottleneck", "token_untrained",
                     "anchor_shuffled", "anchor_mean"):
        return build_bottleneck_mask(attn2d, cpos, dtype=dtype)
    if condition == "token_unmasked":
        return build_causal_mask(attn2d, dtype=dtype)
    if condition == "anchor_removed":
        return build_anchor_removed_mask(attn2d, cpos, dtype=dtype)
    if condition == "anchor_only":
        return build_anchor_only_mask(attn2d, cpos, rpos, dtype=dtype)
    if condition == "forced_relay":
        return build_forced_relay_mask(attn2d, cpos, rpos, dtype=dtype)
    raise ValueError(condition)


@torch.no_grad()
def option_logliks_batched(model, tokenizer, prompt: str, options: list[str],
                           condition: str, device) -> list[float]:
    """Mean per-token loglik of every option in ONE batched forward.

    Same boundary construction as option_loglik_bottlenecked (longest common
    prefix between prompt-only and full tokenization, per option); per-row
    means computed manually because HF's .loss averages across the batch.
    4x fewer forwards than scoring options one by one — the dominant cost of
    the 541x8 run.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    rows, bounds = [], []
    for opt in options:
        full_ids = tokenizer(prompt + opt, add_special_tokens=False)["input_ids"]
        boundary = 0
        for a, b in zip(prompt_ids, full_ids, strict=False):
            if a != b:
                break
            boundary += 1
        rows.append(full_ids)
        bounds.append(boundary)

    width = max(len(r) for r in rows)
    pad = tokenizer.pad_token_id
    ids = torch.full((len(rows), width), pad, dtype=torch.long)
    labels = torch.full((len(rows), width), -100, dtype=torch.long)
    attn2d = torch.zeros((len(rows), width), dtype=torch.long)
    for i, (r, b) in enumerate(zip(rows, bounds, strict=True)):
        ids[i, :len(r)] = torch.tensor(r)
        labels[i, b:len(r)] = torch.tensor(r[b:])
        attn2d[i, :len(r)] = 1
    ids, labels, attn2d = ids.to(device), labels.to(device), attn2d.to(device)

    cpos, rpos = validate_layout(ids, tokenizer)
    dtype = model.get_input_embeddings().weight.dtype
    mask = mask_for(condition, attn2d, cpos, rpos, dtype)
    logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits

    logprobs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    gold = labels[:, 1:]
    supervised = gold != -100
    gathered = logprobs.gather(-1, gold.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    per_row = (gathered * supervised).sum(dim=1) / supervised.sum(dim=1).clamp_min(1)
    return per_row.tolist()


def mcq_prompt(ex: dict) -> str:
    return render_prompt(ex) + f"Question: {ex['meta']['question']}\nAnswer: "


@torch.no_grad()
def pick_for_condition(model, tokenizer, ex: dict, condition: str, device,
                       donor: dict | None) -> int:
    prompt = mcq_prompt(ex)
    patcher = None
    if condition in ("anchor_shuffled", "anchor_mean"):
        host_ids = tokenizer(prompt, return_tensors="pt",
                             add_special_tokens=False)["input_ids"].to(device)
        host_c, _ = validate_layout(host_ids, tokenizer, require_recall=False)
        dtype = model.get_input_embeddings().weight.dtype
        if condition == "anchor_shuffled":
            donor_ids = tokenizer(mcq_prompt(donor), return_tensors="pt",
                                  add_special_tokens=False)["input_ids"].to(device)
            donor_c, _ = validate_layout(donor_ids, tokenizer, require_recall=False)
            patcher = AnchorPatcher(model, int(donor_c))
            patcher.capture(lambda: model(
                input_ids=donor_ids,
                attention_mask=build_bottleneck_mask(
                    torch.ones_like(donor_ids), donor_c, dtype=dtype),
                use_cache=False))
        else:  # anchor_mean: context mean from the host's own clean forward
            patcher = AnchorPatcher(model, int(host_c))
            patcher.capture_context_mean(lambda: model(
                input_ids=host_ids,
                attention_mask=build_bottleneck_mask(
                    torch.ones_like(host_ids), host_c, dtype=dtype),
                use_cache=False), context_end=int(host_c))
        patcher.retarget(int(host_c))

    if patcher is not None:
        with patcher:
            scores = option_logliks_batched(model, tokenizer, prompt,
                                            ex["meta"]["options"], condition,
                                            device)
    else:
        scores = option_logliks_batched(model, tokenizer, prompt,
                                        ex["meta"]["options"], condition,
                                        device)
    return scores.index(max(scores))


def load_exp0_records(path: str) -> dict[str, dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()
            if line.strip()]
    return {r["example_id"]: r for r in rows}


def verdict_block(records_by_cond: dict[str, dict[str, dict]],
                  cohort_ids: list[str]) -> dict:
    def vec(cond: str, key: str = "correct", ids=None) -> list:
        ids = ids or cohort_ids
        return [records_by_cond[cond][i][key] for i in ids]

    bott = vec("true_bottleneck")
    base = vec("prompt_summary")
    diff = bootstrap_ci_paired_diff([float(x) for x in bott],
                                    [float(x) for x in base])
    mcn = mcnemar_exact(bott, base)
    primary_win = (diff["mean_diff"] > 0) and (mcn["p_value"] < 0.05)
    tie = (diff["ci_low"] >= -TIE_BAND_PTS / 100
           and diff["ci_high"] <= TIE_BAND_PTS / 100)

    # mandatory out-of-style partition (CNN/DailyMail)
    cnn_ids = [i for i in cohort_ids
               if records_by_cond["true_bottleneck"][i]["source"] == "cnn_dailymail"]
    oos = {
        "n": len(cnn_ids),
        "true_bottleneck_acc": mean([float(x) for x in vec("true_bottleneck", ids=cnn_ids)]),
        "prompt_summary_acc": mean([float(x) for x in vec("prompt_summary", ids=cnn_ids)]),
        "paired_diff": bootstrap_ci_paired_diff(
            [float(x) for x in vec("true_bottleneck", ids=cnn_ids)],
            [float(x) for x in vec("prompt_summary", ids=cnn_ids)]),
        "mcnemar": mcnemar_exact(vec("true_bottleneck", ids=cnn_ids),
                                 vec("prompt_summary", ids=cnn_ids)),
    }
    return {
        "primary_bottleneck_beats_baseline": primary_win,
        "secondary_tie_within_3pts": tie,
        "paired_diff_bottleneck_minus_baseline": diff,
        "mcnemar_bottleneck_vs_baseline": mcn,
        "out_of_style_cnn": oos,
        "tie_band_pts": TIE_BAND_PTS,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",
                   default="results/checkpoints/exp1b-bottleneck-v2/best")
    p.add_argument("--data", default="data/processed/test.jsonl")
    p.add_argument("--cohort", default="results/exp0_v2_cohort.json")
    p.add_argument("--exp0-records", default="results/exp0_v2_records.jsonl")
    p.add_argument("--out", default="results/exp2_results.json")
    p.add_argument("--records", default="results/exp2_records.jsonl")
    p.add_argument("--resume", action="store_true",
                   help="skip (condition, example_id) pairs already recorded")
    p.add_argument("--limit", type=int, default=None,
                   help="SMOKE ONLY: refuses the default --out")
    args = p.parse_args()
    if args.limit and args.out == "results/exp2_results.json":
        raise SystemExit("--limit is for smoke runs: pass a non-default --out")

    set_seed(SEED)
    device = resolve_device("auto")
    cfg = TrainConfig()

    cohort = CohortSelection.load(args.cohort)
    examples = cohort.select(load_jsonl_validated(args.data))
    if args.limit:
        examples = examples[:args.limit]
    by_id = {ex["example_id"]: ex for ex in examples}
    ids = [ex["example_id"] for ex in examples]

    # donor pairing for anchor_shuffled: seeded shuffle, shift by one
    donor_order = list(ids)
    random.Random(SEED).shuffle(donor_order)
    donor_of = {a: by_id[donor_order[(i + 1) % len(donor_order)]]
                for i, a in enumerate(donor_order)}

    exp0 = load_exp0_records(args.exp0_records)
    missing = [i for i in ids if i not in exp0]
    if missing:
        raise SystemExit(f"{len(missing)} cohort ids missing from Exp 0 records "
                         f"(e.g. {missing[:3]}) — run Exp 0 v2 first")

    records_by_cond: dict[str, dict[str, dict]] = {c: {} for c in GATING + DIAGNOSTIC}
    for i in ids:  # conditions 1-2 come from Exp 0 v2 (same items, same picks)
        r0 = exp0[i]
        common = {"example_id": i, "source": r0["source"],
                  "type": r0["type"], "distance": r0["distance"],
                  "answer_idx": r0["answer_idx"]}
        records_by_cond["full_context_base"][i] = {
            **common, "condition": "full_context_base",
            "pick": r0["full_context_pick"], "correct": r0["full_context_correct"]}
        records_by_cond["prompt_summary"][i] = {
            **common, "condition": "prompt_summary",
            "pick": r0["summary_pick"], "correct": r0["summary_correct"]}

    rec_path = Path(args.records)
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, str]] = set()
    if args.resume and rec_path.exists():
        for line in rec_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                records_by_cond[r["condition"]][r["example_id"]] = r
                done.add((r["condition"], r["example_id"]))
        print(f"resume: {len(done)} records already present", flush=True)

    trained, tokenizer = load_for_eval(args.checkpoint, cfg, device)
    untrained, _ = load_for_eval(None, cfg, device)

    live_conditions = [c for c in GATING + DIAGNOSTIC
                       if c not in ("full_context_base", "prompt_summary")]
    with open(rec_path, "a", encoding="utf-8") as rec_file:
        for cond in live_conditions:
            model = untrained if cond == "token_untrained" else trained
            todo = [ex for ex in examples if (cond, ex["example_id"]) not in done]
            for ex in tqdm(todo, desc=cond):
                pick = pick_for_condition(model, tokenizer, ex, cond, device,
                                          donor_of.get(ex["example_id"]))
                r = {"example_id": ex["example_id"],
                     "source": ex["meta"]["source"], "type": ex["type"],
                     "distance": example_distance(ex),
                     "answer_idx": ex["meta"]["answer_idx"],
                     "condition": cond, "pick": pick,
                     "correct": pick == ex["meta"]["answer_idx"]}
                records_by_cond[cond][ex["example_id"]] = r
                rec_file.write(json.dumps(r, ensure_ascii=False) + "\n")
                rec_file.flush()
            acc = mean([float(records_by_cond[cond][i]["correct"]) for i in ids])
            print(f"[{cond}] accuracy = {acc:.4f}", flush=True)

    results = {
        "n": len(ids),
        "conditions": {
            cond: {
                "accuracy": bootstrap_ci(
                    [float(records_by_cond[cond][i]["correct"]) for i in ids]),
                "by_source": {
                    src: mean([float(records_by_cond[cond][i]["correct"])
                               for i in ids
                               if records_by_cond[cond][i]["source"] == src])
                    for src in ("cnn_dailymail", "synthetic", "handwritten")},
                "by_distance": {
                    str(d): mean([float(records_by_cond[cond][i]["correct"])
                                  for i in ids
                                  if records_by_cond[cond][i]["distance"] == d])
                    for d in sorted({records_by_cond[cond][i]["distance"]
                                     for i in ids})},
                "gating": cond in GATING,
            } for cond in GATING + DIAGNOSTIC},
        "verdict": verdict_block(records_by_cond, ids),
        "protocol": {
            "checkpoint": args.checkpoint, "cohort_file": args.cohort,
            "cohort_sha256": cohort.dataset_sha256, "seed": SEED,
            "metric": "MCQ (gating); fact retrieval deferred (see decisions.md)",
            "records_file": str(rec_path),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    v = results["verdict"]
    print(json.dumps({"verdict": {
        "primary_win": v["primary_bottleneck_beats_baseline"],
        "tie_within_3pts": v["secondary_tie_within_3pts"],
        "diff": v["paired_diff_bottleneck_minus_baseline"]["mean_diff"],
        "mcnemar_p": v["mcnemar_bottleneck_vs_baseline"]["p_value"],
    }}, indent=2), flush=True)
    print(f"EXP2 DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
