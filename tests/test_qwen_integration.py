"""Integration: the 4D additive mask must actually reach Qwen2's attention.

src/bottleneck.py *claims* that transformers 5.10.2 forwards a prepared
(B,1,T,T) float mask unchanged. Until these tests pass on the real model,
that claim is unverified — and if it were false, the bottleneck would be
silently OFF and nothing downstream (toy gate, Exp 1b, Exp 2) would be
interpretable.

Runs on the locally cached Qwen2.5-0.5B (no downloads). Excluded from CI
(`-m "not integration"`); run locally with:  uv run pytest -m integration
"""

from __future__ import annotations

import pytest
import torch

from src.bottleneck import (
    build_bottleneck_mask,
    build_causal_mask,
    forward_bottlenecked,
    generate_bottlenecked,
    validate_layout,
)
from src.config import COMPRESS_TOKEN, RECALL_TOKEN, TrainConfig
from src.model import load_base_model, load_tokenizer

pytestmark = pytest.mark.integration

CONTEXT_A = "The secret launch code is 7319 and the probe departs from Cape Canaveral."
SUFFIX = f"\n{COMPRESS_TOKEN}\nunrelated filler words here\n{RECALL_TOKEN}\n"


@pytest.fixture(scope="module")
def qwen():
    cfg = TrainConfig()
    tokenizer = load_tokenizer(cfg.model_name)
    model = load_base_model(cfg.model_name, tokenizer, dtype=torch.float32)
    model.eval()
    return model, tokenizer


def _encode(tokenizer, text: str) -> torch.Tensor:
    return tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]


def _logits(model, ids: torch.Tensor, mask) -> torch.Tensor:
    with torch.no_grad():
        return model(input_ids=ids, attention_mask=mask, use_cache=False).logits


class TestMaskIsHonored:
    def test_causal_4d_matches_default_2d(self, qwen):
        """Additive-float convention: our explicit causal 4D mask must give the
        same logits as the library's own causal path with a plain 2D mask."""
        model, tokenizer = qwen
        ids = _encode(tokenizer, CONTEXT_A + SUFFIX)
        attn2d = torch.ones_like(ids)
        ref = _logits(model, ids, attn2d)
        ours = _logits(model, ids, build_causal_mask(attn2d))
        assert torch.allclose(ref, ours, atol=1e-4), (
            "explicit causal 4D mask diverges from the default causal path — "
            "the additive 4D convention is NOT being honored")

    def test_bottleneck_actually_removes_edges(self, qwen):
        """If the library ignored our 4D mask and fell back to causal, the
        bottleneck and causal logits would be identical. They must differ."""
        model, tokenizer = qwen
        ids = _encode(tokenizer, CONTEXT_A + SUFFIX)
        attn2d = torch.ones_like(ids)
        c, _ = validate_layout(ids, tokenizer, require_recall=False)
        causal = _logits(model, ids, build_causal_mask(attn2d))
        bott = _logits(model, ids, build_bottleneck_mask(attn2d, c))
        assert not torch.allclose(causal[0, -1], bott[0, -1], atol=1e-4), (
            "bottleneck mask produced identical logits to causal — "
            "the custom 4D mask is being ignored")


def _edited_context_ids(ids: torch.Tensor, compress_pos: int,
                        tokenizer) -> torch.Tensor:
    """Same sequence with a few CONTEXT token ids replaced (id-level edit:
    immune to tokenization-length differences between two texts)."""
    other = tokenizer(" blue", add_special_tokens=False)["input_ids"][0]
    edited = ids.clone()
    positions = [2, 3, 4]
    assert max(positions) < compress_pos
    for p in positions:
        assert int(ids[0, p]) != other
        edited[0, p] = other
    return edited


class TestNoLeak:
    """The definitive pair: context reaches post-anchor positions through the
    anchor's K/V and through NOTHING else."""

    def test_context_reaches_recall_via_anchor(self, qwen):
        # positive control: with the anchor visible, editing the context must
        # change post-anchor logits (the legitimate route exists)
        model, tokenizer = qwen
        ids_a = _encode(tokenizer, CONTEXT_A + SUFFIX)
        c, _ = validate_layout(ids_a, tokenizer, require_recall=False)
        ids_b = _edited_context_ids(ids_a, int(c), tokenizer)
        attn2d = torch.ones_like(ids_a)
        out_a = _logits(model, ids_a, build_bottleneck_mask(attn2d, c))
        out_b = _logits(model, ids_b, build_bottleneck_mask(attn2d, c))
        assert not torch.allclose(out_a[0, -1], out_b[0, -1], atol=1e-4)

    def test_blocking_anchor_makes_context_unreachable(self, qwen):
        # negative control: block the anchor key as well -> post-anchor logits
        # must be EXACTLY invariant to any context edit (blocked keys get
        # softmax weight 0, contributing nothing)
        model, tokenizer = qwen
        ids_a = _encode(tokenizer, CONTEXT_A + SUFFIX)
        c, _ = validate_layout(ids_a, tokenizer, require_recall=False)
        ids_b = _edited_context_ids(ids_a, int(c), tokenizer)
        attn2d = torch.ones_like(ids_a)
        mask = build_bottleneck_mask(attn2d, c)
        blocked = mask.clone()
        ci = int(c)
        blocked[0, 0, ci + 1:, :ci + 1] = torch.finfo(mask.dtype).min
        out_a = _logits(model, ids_a, blocked)
        out_b = _logits(model, ids_b, blocked)
        post = slice(ci + 1, None)
        assert torch.allclose(out_a[0, post], out_b[0, post], atol=1e-5), (
            "post-anchor logits changed with the anchor blocked — "
            "context information is LEAKING around the bottleneck")


class TestBatchAndGeneration:
    def test_right_padded_batch_no_nan(self, qwen):
        model, tokenizer = qwen
        ids_a = _encode(tokenizer, CONTEXT_A + SUFFIX)
        short = _encode(tokenizer, "Code 42." + SUFFIX)
        pad = tokenizer.pad_token_id
        width = ids_a.shape[1]
        padded = torch.full((1, width), pad)
        padded[0, :short.shape[1]] = short[0]
        ids = torch.cat([ids_a, padded])
        attn2d = torch.zeros_like(ids)
        attn2d[0] = 1
        attn2d[1, :short.shape[1]] = 1
        c = torch.tensor([
            int(validate_layout(ids_a, tokenizer, require_recall=False)[0]),
            int(validate_layout(short, tokenizer, require_recall=False)[0]),
        ])
        with torch.no_grad():
            out = forward_bottlenecked(model, ids, attn2d, c)
        assert not torch.isnan(out.logits[0]).any()
        assert not torch.isnan(out.logits[1, :short.shape[1]]).any()

    def test_reference_decoder_smoke(self, qwen):
        model, tokenizer = qwen
        text = generate_bottlenecked(model, tokenizer, CONTEXT_A + SUFFIX,
                                     max_new_tokens=4)
        assert isinstance(text, str)
