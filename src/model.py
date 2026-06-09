"""Model + tokenizer setup: behavioral special tokens and LoRA adapters.

Base model stays frozen; only LoRA adapters (q_proj, v_proj) and the
embedding rows of the new tokens are trained.
"""

from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import SPECIAL_TOKENS, TrainConfig


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def special_token_ids(tokenizer) -> dict[str, int]:
    return {t: tokenizer.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS}


def load_base_model(model_name: str, tokenizer, dtype: torch.dtype | None = None):
    """Load the base model and make room for the new tokens.

    Qwen2.5 checkpoints have an embedding matrix padded beyond the tokenizer
    vocab, so new token ids usually already fit; we only resize when needed,
    then mean-initialize the new rows (more stable than random init).
    """
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    n_rows = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > n_rows:
        model.resize_token_embeddings(len(tokenizer))
    _mean_init_new_tokens(model, tokenizer)
    return model


def _mean_init_new_tokens(model, tokenizer) -> None:
    new_ids = list(special_token_ids(tokenizer).values())
    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        known = torch.ones(emb.shape[0], dtype=torch.bool)
        known[new_ids] = False
        mean = emb[known].mean(dim=0)
        for tid in new_ids:
            emb[tid] = mean
        if not getattr(model.config, "tie_word_embeddings", False):
            head = model.get_output_embeddings().weight
            head_mean = head[known[: head.shape[0]]].mean(dim=0)
            for tid in new_ids:
                head[tid] = head_mean


def apply_lora(model, tokenizer, cfg: TrainConfig):
    """Wrap the model with LoRA; train only adapters + new token embeddings.

    peft>=0.15 supports `trainable_token_indices`, which trains only the new
    embedding rows instead of the full embedding matrix (~136M params on
    Qwen2.5-0.5B) — essential on the GTX 970 4GB. Falls back to
    modules_to_save on older peft.
    """
    new_ids = list(special_token_ids(tokenizer).values())
    common = dict(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        task_type="CAUSAL_LM",
    )
    try:
        lora_cfg = LoraConfig(trainable_token_indices={"embed_tokens": new_ids}, **common)
        model = get_peft_model(model, lora_cfg)
    except TypeError:  # old peft without trainable_token_indices
        modules_to_save = ["embed_tokens"]
        if not getattr(model.config, "tie_word_embeddings", False):
            modules_to_save.append("lm_head")
        lora_cfg = LoraConfig(modules_to_save=modules_to_save, **common)
        model = get_peft_model(model, lora_cfg)
    return model


def setup_model_and_tokenizer(cfg: TrainConfig, with_lora: bool = True):
    """One-call setup used by train/eval/probe scripts."""
    tokenizer = load_tokenizer(cfg.model_name)
    dtype = torch.float32 if cfg.device == "cpu" else None
    model = load_base_model(cfg.model_name, tokenizer, dtype=dtype)
    if with_lora:
        model = apply_lora(model, tokenizer, cfg)
    return model, tokenizer


def find_token_positions(input_ids: torch.Tensor, token_id: int) -> torch.Tensor:
    """First occurrence of `token_id` per row; -1 when absent.

    input_ids: (B, T) -> returns (B,) long tensor.
    """
    matches = input_ids == token_id
    has = matches.any(dim=1)
    pos = matches.float().argmax(dim=1).long()
    pos[~has] = -1
    return pos
