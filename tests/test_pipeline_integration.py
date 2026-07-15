"""P0 anti-regression gate: EVERY pipeline entry point must use the mask.

These tests fail if train / eval-generate / MCQ scoring / probe extraction
ever fall back to ordinary causal attention (or to no 4D mask at all). They
run on a spy model that records the attention_mask it receives, so the CI
lane needs no model downloads.

See decisions.md 2026-07-15 (external-review triage) for why this gate exists.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.bottleneck import LayoutError, build_bottleneck_mask, build_causal_mask
from src.config import COMPRESS_TOKEN, RECALL_TOKEN, TrainConfig
from src.dataset import BehavioralTokenDataset
from src.eval import eval_mcq, example_distance, generate_recall
from src.probe import extract_states
from src.train import forward_batch, optimizer_steps_per_epoch, should_step
from tests.conftest import make_example

VOCAB, DIM = 64, 8


class SpyModel(torch.nn.Module):
    """Records every attention_mask it is called with; returns HF-like output."""

    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(VOCAB, DIM)
        self.masks: list = []

    def get_input_embeddings(self):
        return self.emb

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
        self.masks.append(attention_mask)
        bsz, seq = input_ids.shape
        logits = torch.zeros(bsz, seq, VOCAB)
        logits[..., 0] = 1.0  # argmax = eos -> generation stops after 1 step
        hidden = tuple(torch.zeros(bsz, seq, DIM) for _ in range(3))
        loss = torch.zeros((), requires_grad=True) + 0.0
        return SimpleNamespace(loss=loss, logits=logits, hidden_states=hidden)


def layout_ids(tok) -> torch.Tensor:
    text = f"c0 c1 c2 {COMPRESS_TOKEN} f0 f1 {RECALL_TOKEN} t0 t1"
    return torch.tensor([tok(text)["input_ids"]])


def assert_is_bottleneck_mask(mask, attn2d, cpos):
    __tracebackhide__ = True
    assert mask is not None and mask.dim() == 4, (
        "entry point did NOT pass a 4D mask — ordinary causal attention "
        "regression (P0 gate)")
    expected = build_bottleneck_mask(attn2d, cpos, dtype=mask.dtype)
    assert torch.equal(mask, expected), "4D mask differs from the bottleneck mask"


class TestTrainPath:
    def test_forward_batch_uses_bottleneck_mask(self, fake_tokenizer):
        ids = layout_ids(fake_tokenizer)
        cpos = torch.tensor([3])
        batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                 "labels": ids.clone(), "compress_pos": cpos,
                 "recall_pos": torch.tensor([6])}
        spy = SpyModel()
        forward_batch(spy, batch, TrainConfig())
        assert_is_bottleneck_mask(spy.masks[0], batch["attention_mask"], cpos)

    def test_forward_batch_full_context_control(self, fake_tokenizer):
        ids = layout_ids(fake_tokenizer)
        batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                 "labels": ids.clone(), "compress_pos": torch.tensor([3]),
                 "recall_pos": torch.tensor([6])}
        spy = SpyModel()
        forward_batch(spy, batch, TrainConfig(attention_mode="full_context"))
        expected = build_causal_mask(batch["attention_mask"],
                                     dtype=spy.masks[0].dtype)
        assert torch.equal(spy.masks[0], expected)

    def test_grad_accum_flushes_final_window(self):
        # 5 batches, accum 2 -> step after batches 2, 4 AND the trailing 5th
        steps = [should_step(i, 5, 2) for i in range(5)]
        assert steps == [False, True, False, True, True]
        assert optimizer_steps_per_epoch(5, 2) == 3
        assert optimizer_steps_per_epoch(4, 2) == 2  # no phantom extra step


class TestEvalPaths:
    def test_generate_recall_uses_bottleneck_mask(self, fake_tokenizer):
        ex = make_example(1)
        spy = SpyModel()
        text = generate_recall(spy, fake_tokenizer, ex, torch.device("cpu"))
        assert text == ""  # spy emits eos immediately
        assert len(spy.masks) == 1
        mask = spy.masks[0]
        assert mask is not None and mask.dim() == 4
        # reconstruct the expected mask over the rendered prompt
        from src.bottleneck import validate_layout
        from src.dataset import render_prompt
        ids = fake_tokenizer(render_prompt(ex), return_tensors="pt")["input_ids"]
        cpos, _ = validate_layout(ids, fake_tokenizer, require_recall=False)
        assert_is_bottleneck_mask(mask, torch.ones_like(ids), cpos)

    def test_eval_mcq_uses_bottleneck_mask(self, fake_tokenizer):
        ex = make_example(2)
        spy = SpyModel()
        res = eval_mcq(spy, fake_tokenizer, [ex], torch.device("cpu"))
        assert res["n_mcq"] == 1
        assert len(spy.masks) == 4  # one scoring pass per option
        for mask in spy.masks:
            assert mask is not None and mask.dim() == 4, (
                "MCQ scoring bypassed the bottleneck mask")

    def test_example_distance_v2_contract(self):
        assert example_distance(make_example(1, ex_type="A")) == 0
        assert example_distance(make_example(2, ex_type="B")) == 8
        broken = make_example(3, ex_type="B")
        del broken["meta"]["distance_target_tokens"]
        with pytest.raises(KeyError):
            example_distance(broken)  # loud failure, never bucket-0


class TestProbePath:
    def test_extract_states_uses_bottleneck_mask(self, fake_tokenizer):
        examples = [make_example(i) for i in range(3)]
        spy = SpyModel()
        X, y = extract_states(spy, fake_tokenizer, examples,
                              torch.device("cpu"), mode="compress")
        assert X.shape[0] == 3 and len(y) == 3
        assert len(spy.masks) == 3
        for mask in spy.masks:
            assert mask is not None and mask.dim() == 4, (
                "probe extraction bypassed the bottleneck mask")


class TestDatasetLayoutGuard:
    def _write(self, tmp_path, rows):
        import json
        p = tmp_path / "data.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return str(p)

    def test_valid_row_yields_positions(self, tmp_path, fake_tokenizer):
        path = self._write(tmp_path, [make_example(1, ex_type="B")])
        ds = BehavioralTokenDataset(path, fake_tokenizer, TrainConfig())
        item = ds[0]
        batch = ds.collate([item])
        assert int(batch["compress_pos"][0]) >= 0
        assert int(batch["recall_pos"][0]) > int(batch["compress_pos"][0])

    def test_truncated_recall_fails_loudly(self, tmp_path, fake_tokenizer):
        ex = make_example(1, ex_type="B")
        path = self._write(tmp_path, [ex])
        # max_length small enough to cut [RECALL] off (prompt words > 10)
        ds = BehavioralTokenDataset(path, fake_tokenizer,
                                    TrainConfig(max_length=10))
        with pytest.raises(LayoutError, match="layout after tokenization"):
            ds[0]

    def test_truncated_target_fails_loudly(self, tmp_path, fake_tokenizer):
        ex = make_example(1, ex_type="A")
        path = self._write(tmp_path, [ex])
        prompt_len = len(fake_tokenizer(
            __import__("src.dataset", fromlist=["render_prompt"])
            .render_prompt(ex))["input_ids"])
        ds = BehavioralTokenDataset(path, fake_tokenizer,
                                    TrainConfig(max_length=prompt_len))
        with pytest.raises(LayoutError, match="no target tokens"):
            ds[0]
