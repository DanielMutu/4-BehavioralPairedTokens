"""True [COMPRESS] attention bottleneck: 4D mask, shared forward, reference decoding.

The scientific invariant enforced here:

    After [COMPRESS] (position c), no query may attend to keys before c.
    - query q <= c : ordinary causal attention (k <= q)
    - query q >  c : k in [c, q] only
    Padding keys are always blocked.

So [COMPRESS] can aggregate the whole context, but filler/[RECALL]/target and
every generated token can reach the context ONLY through [COMPRESS]'s K/V.
This is what turns the token from a behavioral marker into a mandatory memory
channel; without it, recall may be plain context re-reading.

Implementation notes (transformers 5.10.2 / Qwen2):
- We pass a prepared additive 4D float mask (B, 1, T, T); the library's mask
  pipeline forwards 4D masks unchanged, and both eager and SDPA accept the
  additive-float convention (0 = allowed, dtype-min = blocked).
- Generation uses full recomputation (`use_cache=False`) and rebuilds the mask
  each step. Slow but exact — the correctness oracle any future KV-pruned
  fast path must match logit-for-logit.
"""

from __future__ import annotations

import torch

from src.config import COMPRESS_TOKEN, RECALL_TOKEN
from src.model import find_token_positions


class LayoutError(ValueError):
    """A sequence violates the strict COMPRESS/RECALL layout."""


# ------------------------------------------------------------------ validation


def validate_layout(input_ids: torch.Tensor, tokenizer,
                    require_recall: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Strict per-row checks; returns (compress_pos, recall_pos).

    Rejects rows where truncation (or malformed data) removed either paired
    token, where a token appears more than once, or where [RECALL] does not
    follow [COMPRESS]. Silent -1 positions in training would quietly disable
    the architecture for that row, so we fail loudly instead.
    """
    compress_id = tokenizer.convert_tokens_to_ids(COMPRESS_TOKEN)
    recall_id = tokenizer.convert_tokens_to_ids(RECALL_TOKEN)

    n_compress = (input_ids == compress_id).sum(dim=1)
    if (n_compress != 1).any():
        bad = (n_compress != 1).nonzero(as_tuple=True)[0].tolist()
        raise LayoutError(f"rows {bad}: expected exactly one {COMPRESS_TOKEN}, "
                          f"got counts {n_compress[bad].tolist()}")
    compress_pos = find_token_positions(input_ids, compress_id)

    recall_pos = find_token_positions(input_ids, recall_id)
    if require_recall:
        n_recall = (input_ids == recall_id).sum(dim=1)
        if (n_recall != 1).any():
            bad = (n_recall != 1).nonzero(as_tuple=True)[0].tolist()
            raise LayoutError(f"rows {bad}: expected exactly one {RECALL_TOKEN}, "
                              f"got counts {n_recall[bad].tolist()}")
        if (recall_pos <= compress_pos).any():
            bad = (recall_pos <= compress_pos).nonzero(as_tuple=True)[0].tolist()
            raise LayoutError(f"rows {bad}: {RECALL_TOKEN} must follow {COMPRESS_TOKEN}")
    return compress_pos, recall_pos


# ------------------------------------------------------------------- 4D mask


def build_bottleneck_mask(attention_mask_2d: torch.Tensor,
                          compress_pos: torch.Tensor,
                          dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Additive 4D mask (B, 1, T, T): 0 = attend, dtype-min = blocked.

    attention_mask_2d: (B, T) padding mask (1 = real token).
    compress_pos:      (B,) position of [COMPRESS] per row (>= 0).
    """
    if (compress_pos < 0).any():
        raise LayoutError("build_bottleneck_mask needs a valid compress_pos per row")
    bsz, seq = attention_mask_2d.shape
    device = attention_mask_2d.device
    q = torch.arange(seq, device=device).view(1, seq, 1)   # query index
    k = torch.arange(seq, device=device).view(1, 1, seq)   # key index
    c = compress_pos.to(device).view(bsz, 1, 1)

    causal = k <= q
    # queries strictly after [COMPRESS] may not look before it
    no_bypass = (q <= c) | (k >= c)
    key_real = attention_mask_2d.to(torch.bool).view(bsz, 1, seq)

    allowed = causal & no_bypass & key_real
    _reject_fully_blocked_rows(allowed)
    mask = torch.zeros(bsz, 1, seq, seq, dtype=dtype, device=device)
    mask.masked_fill_(~allowed.view(bsz, 1, seq, seq), torch.finfo(dtype).min)
    return mask


def build_anchor_removed_mask(attention_mask_2d: torch.Tensor,
                              compress_pos: torch.Tensor,
                              dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Exp 2 condition 'anchor removed': post-anchor queries lose even the
    anchor key — k in (c, q] only. Operationalizes 'anchor azzerato' by
    removing ACCESS (same construction validated in the toy gate)."""
    return _build_variant_mask(attention_mask_2d, compress_pos, None,
                               post_includes_anchor=False,
                               recall_includes_anchor=False, dtype=dtype)


def build_anchor_only_mask(attention_mask_2d: torch.Tensor,
                           compress_pos: torch.Tensor,
                           recall_pos: torch.Tensor,
                           dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Exp 2 condition 'anchor-only recall': fillers are BLIND to the anchor
    (k in (c, q] for c < q < r); [RECALL] and everything after read it
    (k in [c, q] for q >= r). Separates single-state persistence from
    relay through the filler chain."""
    return _build_variant_mask(attention_mask_2d, compress_pos, recall_pos,
                               post_includes_anchor=False,
                               recall_includes_anchor=True, dtype=dtype)


def build_forced_relay_mask(attention_mask_2d: torch.Tensor,
                            compress_pos: torch.Tensor,
                            recall_pos: torch.Tensor,
                            dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Exp 2 diagnostic 'forced relay': fillers CAN read the anchor
    (k in [c, q] for c < q < r) but [RECALL] and after cannot
    (k in (c, q] for q >= r) — information must survive the hop chain."""
    return _build_variant_mask(attention_mask_2d, compress_pos, recall_pos,
                               post_includes_anchor=True,
                               recall_includes_anchor=False, dtype=dtype)


def _build_variant_mask(attention_mask_2d: torch.Tensor,
                        compress_pos: torch.Tensor,
                        recall_pos: torch.Tensor | None,
                        post_includes_anchor: bool,
                        recall_includes_anchor: bool,
                        dtype: torch.dtype) -> torch.Tensor:
    """Shared builder for bottleneck variants. All variants keep the core
    invariant (post-anchor queries never see keys < c); they differ only in
    WHO may read the anchor key itself."""
    if (compress_pos < 0).any():
        raise LayoutError("variant masks need a valid compress_pos per row")
    if recall_pos is not None and (recall_pos <= compress_pos).any():
        raise LayoutError("variant masks need recall_pos > compress_pos per row")
    bsz, seq = attention_mask_2d.shape
    device = attention_mask_2d.device
    q = torch.arange(seq, device=device).view(1, seq, 1)
    k = torch.arange(seq, device=device).view(1, 1, seq)
    c = compress_pos.to(device).view(bsz, 1, 1)

    causal = k <= q
    key_real = attention_mask_2d.to(torch.bool).view(bsz, 1, seq)
    pre = q <= c                       # ordinary causal region
    sees_anchor_strict = k > c         # (c, q]
    sees_anchor_incl = k >= c          # [c, q]

    if recall_pos is None:
        post_rule = sees_anchor_incl if post_includes_anchor else sees_anchor_strict
        allowed = causal & (pre | post_rule) & key_real
    else:
        r = recall_pos.to(device).view(bsz, 1, 1)
        filler_zone = (q > c) & (q < r)
        recall_zone = q >= r
        filler_rule = sees_anchor_incl if post_includes_anchor else sees_anchor_strict
        recall_rule = sees_anchor_incl if recall_includes_anchor else sees_anchor_strict
        allowed = causal & (pre
                            | (filler_zone & filler_rule)
                            | (recall_zone & recall_rule)) & key_real
    _reject_fully_blocked_rows(allowed)
    mask = torch.zeros(bsz, 1, seq, seq, dtype=dtype, device=device)
    mask.masked_fill_(~allowed.view(bsz, 1, seq, seq), torch.finfo(dtype).min)
    return mask


def build_causal_mask(attention_mask_2d: torch.Tensor,
                      dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Ordinary causal 4D mask, same convention — the no-bottleneck control."""
    bsz, seq = attention_mask_2d.shape
    device = attention_mask_2d.device
    q = torch.arange(seq, device=device).view(1, seq, 1)
    k = torch.arange(seq, device=device).view(1, 1, seq)
    key_real = attention_mask_2d.to(torch.bool).view(bsz, 1, seq)
    allowed = (k <= q) & key_real
    _reject_fully_blocked_rows(allowed)
    mask = torch.zeros(bsz, 1, seq, seq, dtype=dtype, device=device)
    mask.masked_fill_(~allowed.view(bsz, 1, seq, seq), torch.finfo(dtype).min)
    return mask


def _reject_fully_blocked_rows(allowed: torch.Tensor) -> None:
    """A query row with zero allowed keys makes softmax(-inf everywhere) = NaN,
    and the NaN poisons real positions through later layers (0 * NaN = NaN).
    This happens with LEFT padding, which is therefore rejected loudly here:
    the pipeline requires right padding.
    """
    if not allowed.any(dim=-1).all():
        bad = (~allowed.any(dim=-1)).nonzero(as_tuple=True)
        raise LayoutError(
            f"mask has fully-blocked query rows (batch,query)={[t.tolist() for t in bad]}; "
            "use right padding — left padding is unsupported")


# ------------------------------------------------------------- shared forward


def forward_bottlenecked(model, input_ids: torch.Tensor,
                         attention_mask_2d: torch.Tensor,
                         compress_pos: torch.Tensor,
                         labels: torch.Tensor | None = None,
                         mode: str = "compress_bottleneck",
                         **kwargs):
    """The one forward path train/eval/probe must share.

    mode: "compress_bottleneck" (v2) or "full_context" (v0-style control).
    """
    dtype = model.get_input_embeddings().weight.dtype
    if mode == "compress_bottleneck":
        mask = build_bottleneck_mask(attention_mask_2d, compress_pos, dtype=dtype)
    elif mode == "full_context":
        mask = build_causal_mask(attention_mask_2d, dtype=dtype)
    else:
        raise ValueError(f"unknown attention mode: {mode!r}")
    return model(input_ids=input_ids, attention_mask=mask,
                 labels=labels, use_cache=False, **kwargs)


# -------------------------------------------------------- reference generation


@torch.no_grad()
def generate_bottlenecked(model, tokenizer, prompt: str,
                          max_new_tokens: int = 160,
                          mode: str = "compress_bottleneck",
                          device: torch.device | None = None,
                          stop_at_eos: bool = True) -> str:
    """Greedy reference decoder with full recomputation (`use_cache=False`).

    Exact by construction: every step rebuilds the strict mask over the whole
    sequence, so generated tokens can never read pre-[COMPRESS] keys through a
    stale KV cache. O(T^2) per step — acceptable for eval-scale runs; any
    faster cached implementation must reproduce these logits.
    """
    device = device or next(model.parameters()).device
    ids = tokenizer(prompt, return_tensors="pt",
                    add_special_tokens=False)["input_ids"].to(device)
    compress_pos, _ = validate_layout(ids, tokenizer, require_recall=False)
    eos_id = tokenizer.eos_token_id

    generated: list[int] = []
    for _ in range(max_new_tokens):
        attn2d = torch.ones_like(ids)
        out = forward_bottlenecked(model, ids, attn2d, compress_pos, mode=mode)
        next_id = int(out.logits[0, -1].argmax())
        if stop_at_eos and next_id == eos_id:
            break
        generated.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


@torch.no_grad()
def option_loglik_bottlenecked(model, tokenizer, prompt: str, option: str,
                               device: torch.device,
                               mode: str = "compress_bottleneck") -> float:
    """Mean per-token log-likelihood of `option` given `prompt`, under `mode`.

    Fixes the v0 boundary bug too: the full text is tokenized as one string,
    and the prompt/option boundary is recovered from the tokenized prompt
    prefix, so scoring matches what the model would actually see.
    """
    full = prompt + option
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    # longest common prefix between prompt-only and full tokenization
    boundary = 0
    # strict=False: prompt_ids is expected to be shorter than full_ids
    for a, b in zip(prompt_ids, full_ids, strict=False):
        if a != b:
            break
        boundary += 1
    if boundary == len(full_ids):
        raise LayoutError("option produced no continuation tokens")

    ids = torch.tensor([full_ids], device=device)
    labels = ids.clone()
    labels[0, :boundary] = -100
    compress_pos, _ = validate_layout(ids, tokenizer, require_recall=False)
    attn2d = torch.ones_like(ids)
    out = forward_bottlenecked(model, ids, attn2d, compress_pos,
                               labels=labels, mode=mode)
    return -float(out.loss)
