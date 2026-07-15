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

from src.bottleneck import generate_bottlenecked
from src.config import TrainConfig
from src.dataset import render_prompt
from src.eval import checkpoint_attention_mode, load_for_eval
from src.utils import resolve_device, set_seed


@torch.no_grad()
def generate(model, tokenizer, prompt: str, device, max_new_tokens: int = 200,
             mode: str = "compress_bottleneck") -> str:
    """Playground generation through the SAME shared path as eval (P0 gate).

    Reference full-recomputation decoder: slower than model.generate, but what
    you see is exactly the regime the checkpoint is evaluated under.
    """
    return generate_bottlenecked(model, tokenizer, prompt,
                                 max_new_tokens=max_new_tokens, mode=mode,
                                 device=device)


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
    ap.add_argument("--attention-mode", default=None,
                    choices=["compress_bottleneck", "full_context"],
                    help="override; default = the mode saved in the checkpoint "
                         "config (v0 checkpoints without the field -> "
                         "full_context, the regime they were trained under)")
    args = ap.parse_args()

    set_seed(42)
    cfg = TrainConfig()
    device = resolve_device(cfg.device)
    print(f"Loading {'base + adapter ' + args.checkpoint if args.checkpoint else 'BASE model'} "
          f"on {device} ...")
    model, tokenizer = load_for_eval(args.checkpoint, cfg, device)
    # v0 checkpoints (e.g. exp1-stability) predate attention_mode: they were
    # trained full-context, so that is their honest playground regime too.
    mode = args.attention_mode or checkpoint_attention_mode(
        args.checkpoint, "full_context" if args.checkpoint else cfg.attention_mode)
    print(f"attention_mode = {mode}")
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
        text = generate(model, tokenizer, prompt, device, args.max_new_tokens,
                        mode=mode)
        print(text.strip() or "(empty output)")
        print()


if __name__ == "__main__":
    main()
