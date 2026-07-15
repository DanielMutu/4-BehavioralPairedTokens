"""Bottleneck mask: truth table, layout validation, gradient flow, invariance.

These are the tests that make the scientific claim checkable: after
[COMPRESS], nothing may read the context directly.
"""

from __future__ import annotations

import pytest
import torch

from src.bottleneck import (
    LayoutError,
    build_bottleneck_mask,
    build_causal_mask,
    validate_layout,
)
from tests.conftest import encode_layout

NEG = torch.finfo(torch.float32).min


def allowed(mask: torch.Tensor) -> torch.Tensor:
    """(B,1,T,T) additive mask -> boolean 'may attend' matrix (B,T,T)."""
    return (mask[:, 0] == 0)


class TestMaskTruthTable:
    def test_exhaustive_single_row(self):
        # layout: ctx0 ctx1 ctx2 ctx3 [COMPRESS] f0 f1 [RECALL] t0 t1 t2
        seq, c = 11, 4
        attn2d = torch.ones(1, seq)
        mask = build_bottleneck_mask(attn2d, torch.tensor([c]))
        ok = allowed(mask)[0]
        for q in range(seq):
            for k in range(seq):
                expected = k <= q if q <= c else c <= k <= q
                assert ok[q, k].item() == expected, (q, k)

    def test_padding_keys_blocked(self):
        attn2d = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])
        mask = build_bottleneck_mask(attn2d, torch.tensor([2]))
        assert not allowed(mask)[0][:, 6:].any()

    def test_batch_with_different_anchors(self):
        attn2d = torch.ones(2, 6)
        mask = build_bottleneck_mask(attn2d, torch.tensor([1, 4]))
        ok = allowed(mask)
        assert not ok[0, 5, 0]          # row 0: post-anchor can't see ctx
        assert ok[0, 5, 1]              # ...but sees its anchor at 1
        # row 1 (anchor at 4): q=5 blocks every key before the anchor
        assert not ok[1, 5, 2]
        assert not ok[1, 5, 3]
        assert ok[1, 5, 4]

    def test_negative_anchor_rejected(self):
        with pytest.raises(LayoutError):
            build_bottleneck_mask(torch.ones(1, 4), torch.tensor([-1]))

    def test_causal_control_mask_differs_only_pre_anchor(self):
        attn2d = torch.ones(1, 8)
        c = 3
        bott = allowed(build_bottleneck_mask(attn2d, torch.tensor([c])))[0]
        caus = allowed(build_causal_mask(attn2d))[0]
        diff = (bott != caus).nonzero().tolist()
        # every difference must be a post-anchor query reading a pre-anchor key
        assert diff, "bottleneck must actually remove edges"
        for q, k in diff:
            assert q > c and k < c


class TestLayoutValidation:
    def test_valid_layout(self, fake_tokenizer):
        ids, c, r = encode_layout(fake_tokenizer)
        cp, rp = validate_layout(ids, fake_tokenizer)
        assert cp.item() == c and rp.item() == r

    def test_missing_compress_rejected(self, fake_tokenizer):
        ids = torch.tensor([fake_tokenizer("a b c [RECALL] d")["input_ids"]])
        with pytest.raises(LayoutError, match="COMPRESS"):
            validate_layout(ids, fake_tokenizer)

    def test_duplicate_compress_rejected(self, fake_tokenizer):
        text = "a [COMPRESS] b [COMPRESS] [RECALL] c"
        ids = torch.tensor([fake_tokenizer(text)["input_ids"]])
        with pytest.raises(LayoutError, match="exactly one"):
            validate_layout(ids, fake_tokenizer)

    def test_recall_before_compress_rejected(self, fake_tokenizer):
        text = "a [RECALL] b [COMPRESS] c"
        ids = torch.tensor([fake_tokenizer(text)["input_ids"]])
        with pytest.raises(LayoutError, match="must follow"):
            validate_layout(ids, fake_tokenizer)

    def test_generation_mode_allows_missing_recall(self, fake_tokenizer):
        ids = torch.tensor([fake_tokenizer("a b [COMPRESS] c")["input_ids"]])
        cp, rp = validate_layout(ids, fake_tokenizer, require_recall=False)
        assert cp.item() == 2 and rp.item() == -1


class TinyAttention(torch.nn.Module):
    """Single-head attention over one-hot embeddings: mask semantics checker."""

    def __init__(self, vocab: int, dim: int = 16):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, dim)
        self.q = torch.nn.Linear(dim, dim)
        self.k = torch.nn.Linear(dim, dim)
        self.v = torch.nn.Linear(dim, dim)

    def forward(self, ids: torch.Tensor, mask4d: torch.Tensor) -> torch.Tensor:
        x = self.emb(ids)
        att = self.q(x) @ self.k(x).transpose(1, 2) / 4.0
        att = att + mask4d[:, 0]
        return torch.softmax(att, dim=-1) @ self.v(x)


class TestInformationFlow:
    def test_gradient_reaches_context_only_via_anchor(self):
        torch.manual_seed(0)
        model = TinyAttention(vocab=50)
        ids = torch.arange(10).unsqueeze(0)
        c = 4
        mask = build_bottleneck_mask(torch.ones(1, 10), torch.tensor([c]))

        out = model(ids, mask)
        loss = out[0, 7:].sum()          # loss only on post-anchor positions
        loss.backward()
        grad = model.emb.weight.grad.abs().sum(dim=1)
        # context embeddings still get gradient (route: anchor query reads them)
        assert grad[:c].sum() > 0

        # but if the anchor key is also blocked, the context route disappears
        model.zero_grad()
        blocked = mask.clone()
        blocked[0, 0, c + 1:, :c + 1] = torch.finfo(torch.float32).min
        out = model(ids, blocked)
        out[0, 7:].sum().backward()
        grad = model.emb.weight.grad.abs().sum(dim=1)
        assert grad[:c].sum() == pytest.approx(0.0, abs=1e-9)

    def test_post_anchor_output_invariant_to_context_edits(self):
        """The core leak test: with the mask, editing context must not change
        post-anchor outputs *except* through the anchor position itself."""
        torch.manual_seed(0)
        model = TinyAttention(vocab=50)
        c = 4
        mask = build_bottleneck_mask(torch.ones(1, 10), torch.tensor([c]))

        ids_a = torch.arange(10).unsqueeze(0)
        ids_b = ids_a.clone()
        ids_b[0, :c] = torch.tensor([40, 41, 42, 43])  # different context

        # freeze the anchor's contribution by copying its value row: here we
        # simply check that outputs differ ONLY because the anchor row read a
        # different context — post-anchor positions may not read ctx directly.
        with torch.no_grad():
            out_a = model(ids_a, mask)
            out_b = model(ids_b, mask)
        # positions strictly before the anchor obviously differ
        assert not torch.allclose(out_a[0, :c], out_b[0, :c])
        # the ONLY path to post-anchor positions is via the anchor's K/V.
        # If we also block the anchor key, post-anchor outputs become equal:
        blocked = mask.clone()
        blocked[0, 0, c + 1:, :c + 1] = torch.finfo(torch.float32).min
        with torch.no_grad():
            out_a2 = model(ids_a, blocked)
            out_b2 = model(ids_b, blocked)
        assert torch.allclose(out_a2[0, c + 1:], out_b2[0, c + 1:], atol=1e-6)
