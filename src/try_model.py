"""Interactive playground for the trained COMPRESS/RECALL model.

Paste a text, the script wraps it in the training format
(context + [COMPRESS] + [RECALL]) and shows what the model generates.

Usage:
    python -m src.try_model                                    # base model (no adapter)
    python -m src.try_model --checkpoint results/checkpoints/exp1-stability/best
    python -m src.try_model --checkpoint ... --filler "some intermediate text"
"""

from __future__ import annotations

import argparse

import torch

from src.config import TrainConfig
from src.dataset import render_prompt
from src.eval import load_for_eval
from src.utils import resolve_device, set_seed


@torch.no_grad()
def generate(model, tokenizer, prompt: str, device, max_new_tokens: int = 200) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def read_multiline(banner: str) -> str | None:
    """Read until blank line. Returns None on EOF (stdin exhausted)."""
    print(banner)
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            return "\n".join(lines) if lines else None
        if line.strip() == "":
            break
        lines.append(line)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="LoRA adapter dir (omit for base model)")
    ap.add_argument("--filler", default="",
                    help="text inserted between [COMPRESS] and [RECALL] (distance test)")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    args = ap.parse_args()

    set_seed(42)
    cfg = TrainConfig()
    device = resolve_device(cfg.device)
    print(f"Loading {'base + adapter ' + args.checkpoint if args.checkpoint else 'BASE model'} "
          f"on {device} ...")
    model, tokenizer = load_for_eval(args.checkpoint, cfg, device)
    print("Ready. Empty line ends the text, Ctrl+C exits.\n")

    while True:
        try:
            context = read_multiline("--- Paste your text (empty line to finish): ---")
        except KeyboardInterrupt:
            print("\nbye")
            break
        if context is None:
            break
        if not context.strip():
            continue

        example = {"context": context, "filler": args.filler, "target": ""}
        prompt = render_prompt(example)
        print("\n[prompt sent to the model]")
        print(prompt)
        print("[model recall] ...", flush=True)
        text = generate(model, tokenizer, prompt, device, args.max_new_tokens)
        print(text.strip() or "(empty output)")
        print()


if __name__ == "__main__":
    main()
