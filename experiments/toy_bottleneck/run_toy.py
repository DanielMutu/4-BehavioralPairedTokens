"""Toy code-recall gate: does the model NEED [COMPRESS] as its memory channel?

Task: "The secret code is NNNN." [COMPRESS] filler [RECALL] -> NNNN
Train a small LoRA (bottleneck mask ON) on ~160 codes, evaluate on unseen
codes, then run the causal controls that decide the gate.

Pre-registered PASS criteria (fixed BEFORE seeing any result — constants
below, do not retouch):
  1. bottleneck accuracy on UNSEEN codes        >= 0.90
  2. untrained-token accuracy                   <= 0.05
  3. anchor-removed accuracy (key blocked)      <= 0.05
  4. activation patching, context_override_rate >= 0.90
     (patching A's anchor states into B's prompt must NOT yield B's code:
      the context beyond the anchor is unreachable)
  5. activation patching, swap_rate — informative, reported (target >= 0.5:
      patched generation yields A's code, i.e. the anchor DETERMINES recall)
  6. full-context accuracy — informative upper bound (>= bottleneck expected)

Usage:
    python experiments/toy_bottleneck/run_toy.py \
        --out results/toy_bottleneck.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.bottleneck import (  # noqa: E402
    build_bottleneck_mask,
    forward_bottlenecked,
    generate_bottlenecked,
    validate_layout,
)
from src.config import COMPRESS_TOKEN, RECALL_TOKEN, TrainConfig  # noqa: E402
from src.model import apply_lora, load_base_model, load_tokenizer  # noqa: E402

SEED = 42
N_TRAIN, N_EVAL = 160, 40
FILLER = "the weather report mentions scattered clouds and mild wind today"

# ---- pre-registered gate thresholds (do NOT change after seeing results) ----
MIN_BOTTLENECK_ACC = 0.90
MAX_UNTRAINED_ACC = 0.05
MAX_ANCHOR_REMOVED_ACC = 0.05
MIN_CONTEXT_OVERRIDE_RATE = 0.90


def render_prompt(code: str) -> str:
    return (f"The secret code is {code}.\n{COMPRESS_TOKEN}\n"
            f"{FILLER}\n{RECALL_TOKEN}\n")


def make_codes(rng: random.Random, n_train: int = N_TRAIN,
               n_eval: int = N_EVAL) -> tuple[list[str], list[str]]:
    codes = [f"{n:04d}" for n in rng.sample(range(10000), n_train + n_eval)]
    return codes[:n_train], codes[n_train:]


# ------------------------------------------------------------------ batching


def build_batch(tokenizer, codes: list[str], device) -> dict:
    rows, labels_rows = [], []
    for code in codes:
        p_ids = tokenizer(render_prompt(code), add_special_tokens=False)["input_ids"]
        t_ids = tokenizer(code, add_special_tokens=False)["input_ids"]
        t_ids = t_ids + [tokenizer.eos_token_id]
        rows.append(p_ids + t_ids)
        labels_rows.append([-100] * len(p_ids) + t_ids)
    width = max(len(r) for r in rows)
    pad = tokenizer.pad_token_id
    ids = torch.full((len(rows), width), pad, dtype=torch.long)
    labels = torch.full((len(rows), width), -100, dtype=torch.long)
    attn = torch.zeros((len(rows), width), dtype=torch.long)
    for i, (r, lr) in enumerate(zip(rows, labels_rows, strict=True)):
        ids[i, :len(r)] = torch.tensor(r)
        labels[i, :len(lr)] = torch.tensor(lr)
        attn[i, :len(r)] = 1
    ids, labels, attn = ids.to(device), labels.to(device), attn.to(device)
    cpos, _ = validate_layout(ids, tokenizer)
    return {"input_ids": ids, "labels": labels, "attention_mask": attn,
            "compress_pos": cpos}


# ------------------------------------------------------------------- training


def train(model, tokenizer, train_codes: list[str], eval_codes: list[str],
          device, epochs: int, bs: int, lr: float,
          mode: str = "compress_bottleneck") -> dict:
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    eval_batch = build_batch(tokenizer, eval_codes, device)
    rng = random.Random(SEED)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(train_codes)
        rng.shuffle(order)
        losses = []
        t0 = time.time()
        for i in range(0, len(order), bs):
            batch = build_batch(tokenizer, order[i:i + bs], device)
            out = forward_bottlenecked(model, batch["input_ids"],
                                       batch["attention_mask"],
                                       batch["compress_pos"],
                                       labels=batch["labels"],
                                       mode=mode)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad()
            losses.append(float(out.loss))
        tf_acc = teacher_forced_accuracy(model, eval_batch)
        history.append({"epoch": epoch, "loss": sum(losses) / len(losses),
                        "eval_tf_acc": tf_acc,
                        "seconds": round(time.time() - t0, 1)})
        print(f"epoch {epoch}: loss={history[-1]['loss']:.4f} "
              f"eval_tf_acc={tf_acc:.3f} ({history[-1]['seconds']}s)", flush=True)
        if tf_acc >= 0.98:
            print("early stop: teacher-forced eval accuracy >= 0.98", flush=True)
            break
    return {"history": history}


@torch.no_grad()
def teacher_forced_accuracy(model, batch: dict) -> float:
    model.eval()
    out = forward_bottlenecked(model, batch["input_ids"],
                               batch["attention_mask"], batch["compress_pos"])
    preds = out.logits[:, :-1].argmax(dim=-1)
    gold = batch["labels"][:, 1:]
    ok = 0
    for i in range(gold.shape[0]):
        sel = gold[i] != -100
        ok += int((preds[i][sel] == gold[i][sel]).all())
    return ok / gold.shape[0]


# ------------------------------------------------------------------- controls


@torch.no_grad()
def generation_accuracy(model, tokenizer, codes: list[str],
                        mode: str = "compress_bottleneck") -> float:
    model.eval()
    hits = 0
    for code in codes:
        text = generate_bottlenecked(model, tokenizer, render_prompt(code),
                                     max_new_tokens=8, mode=mode)
        hits += int(text.strip() == code)
    return hits / len(codes)


@torch.no_grad()
def anchor_removed_accuracy(model, tokenizer, codes: list[str], device) -> float:
    """Greedy decoding with the anchor KEY blocked for every post-anchor query:
    the recall loses its only route to the context."""
    model.eval()
    hits = 0
    for code in codes:
        ids = tokenizer(render_prompt(code), return_tensors="pt",
                        add_special_tokens=False)["input_ids"].to(device)
        cpos, _ = validate_layout(ids, tokenizer, require_recall=False)
        ci = int(cpos)
        generated = []
        for _ in range(8):
            mask = build_bottleneck_mask(torch.ones_like(ids), cpos,
                                         dtype=model.get_input_embeddings().weight.dtype)
            mask[0, 0, ci + 1:, :ci + 1] = torch.finfo(mask.dtype).min
            out = model(input_ids=ids, attention_mask=mask, use_cache=False)
            nxt = int(out.logits[0, -1].argmax())
            if nxt == tokenizer.eos_token_id:
                break
            generated.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
        hits += int(tokenizer.decode(generated, skip_special_tokens=True).strip() == code)
    return hits / len(codes)


from src.patching import AnchorPatcher  # noqa: E402  (shared with Exp 2)


@torch.no_grad()
def patching_rates(model, tokenizer, eval_codes: list[str],
                   device, n_pairs: int = 20) -> dict:
    """Patch A's anchor states (all depths) into B's prompt and generate.

    context_override_rate: output != B's code  (context is unreachable)
    swap_rate:             output == A's code  (the anchor determines recall)
    """
    model.eval()
    rng = random.Random(SEED + 1)
    pairs = []
    while len(pairs) < n_pairs:
        a, b = rng.sample(eval_codes, 2)
        pairs.append((a, b))

    override = swap = 0
    for a, b in pairs:
        ids_a = tokenizer(render_prompt(a), return_tensors="pt",
                          add_special_tokens=False)["input_ids"].to(device)
        ids_b = tokenizer(render_prompt(b), return_tensors="pt",
                          add_special_tokens=False)["input_ids"].to(device)
        assert ids_a.shape == ids_b.shape, "fixed template must tokenize equally"
        cpos, _ = validate_layout(ids_a, tokenizer, require_recall=False)

        patcher = AnchorPatcher(model, int(cpos))
        patcher.capture(lambda ids=ids_a, cp=cpos: forward_bottlenecked(
            model, ids, torch.ones_like(ids), cp))
        with patcher:
            text = generate_bottlenecked(model, tokenizer, render_prompt(b),
                                         max_new_tokens=8)
        override += int(text.strip() != b)
        swap += int(text.strip() == a)
    return {"context_override_rate": override / n_pairs,
            "swap_rate": swap / n_pairs, "n_pairs": n_pairs}


# ----------------------------------------------------------------------- main


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results/toy_bottleneck.json")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--mode", default="compress_bottleneck",
                   choices=["compress_bottleneck", "full_context"],
                   help="training attention mode (full_context = control run)")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-targets", default="q_proj,v_proj",
                   help="comma-separated LoRA target modules")
    p.add_argument("--skip-untrained", action="store_true",
                   help="skip the untrained control (already measured)")
    p.add_argument("--n-train", type=int, default=N_TRAIN)
    p.add_argument("--n-eval", type=int, default=N_EVAL)
    args = p.parse_args()

    torch.manual_seed(SEED)
    device = torch.device("cpu")
    rng = random.Random(SEED)
    train_codes, eval_codes = make_codes(rng, args.n_train, args.n_eval)

    cfg = TrainConfig(lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                      lora_dropout=0.1, seed=SEED,
                      lora_target_modules=args.lora_targets.split(","))
    tokenizer = load_tokenizer(cfg.model_name)

    untrained_acc = None
    if not args.skip_untrained:
        print("== untrained control (base model, mean-init tokens, no adapter) ==",
              flush=True)
        base = load_base_model(cfg.model_name, tokenizer, dtype=torch.float32)
        base.eval()
        untrained_acc = generation_accuracy(base, tokenizer, eval_codes)
        print(f"untrained_acc={untrained_acc:.3f}", flush=True)
        del base

    print(f"== training (mode={args.mode}, targets={cfg.lora_target_modules}, "
          f"r={cfg.lora_r}) ==", flush=True)
    model = load_base_model(cfg.model_name, tokenizer, dtype=torch.float32)
    model = apply_lora(model, tokenizer, cfg)
    t0 = time.time()
    train_info = train(model, tokenizer, train_codes, eval_codes, device,
                       epochs=args.epochs, bs=args.batch_size, lr=args.lr,
                       mode=args.mode)
    train_seconds = round(time.time() - t0, 1)

    print("== controls ==", flush=True)
    results = {
        "seed": SEED,
        "mode": args.mode,
        "n_train": args.n_train, "n_eval": args.n_eval,
        "config": {"lora_r": cfg.lora_r, "lora_alpha": cfg.lora_alpha,
                   "lora_dropout": cfg.lora_dropout,
                   "lora_targets": cfg.lora_target_modules, "lr": args.lr,
                   "batch_size": args.batch_size, "epochs_max": args.epochs},
        "train_seconds": train_seconds,
        "history": train_info["history"],
        "untrained_acc": untrained_acc,
        "bottleneck_acc": generation_accuracy(model, tokenizer, eval_codes),
        "bottleneck_acc_train_subset": generation_accuracy(
            model, tokenizer, train_codes[:20]),
        "full_context_acc": generation_accuracy(model, tokenizer, eval_codes,
                                                mode="full_context"),
        "anchor_removed_acc": anchor_removed_accuracy(model, tokenizer,
                                                      eval_codes, device),
        "patching": patching_rates(model, tokenizer, eval_codes, device),
    }

    if args.mode == "compress_bottleneck":
        gate = {
            "bottleneck_acc>=0.90": results["bottleneck_acc"] >= MIN_BOTTLENECK_ACC,
            "anchor_removed_acc<=0.05":
                results["anchor_removed_acc"] <= MAX_ANCHOR_REMOVED_ACC,
            "context_override_rate>=0.90":
                results["patching"]["context_override_rate"] >= MIN_CONTEXT_OVERRIDE_RATE,
        }
        if untrained_acc is not None:
            gate["untrained_acc<=0.05"] = untrained_acc <= MAX_UNTRAINED_ACC
        results["gate"] = gate
        results["PASS"] = all(gate.values())
    else:
        results["gate"] = None
        results["PASS"] = None  # control run: informative only

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k != "history"},
                     indent=2), flush=True)
    print(f"\nTOY GATE: {'PASS' if results['PASS'] else 'FAIL'}  -> {out}", flush=True)


if __name__ == "__main__":
    main()
