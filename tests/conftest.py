"""Shared fixtures: contract-valid examples and a tiny fake tokenizer."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import COMPRESS_TOKEN, REASON_TOKEN, RECALL_TOKEN  # noqa: E402
from src.data_contract import seal_example  # noqa: E402


def make_example(i: int = 0, ex_type: str = "A", label: str | None = "positive",
                 source: str = "synthetic", **overrides) -> dict:
    ex = {
        "type": ex_type,
        "context": f"Fact number {i}: the launch happened on day {i} in city {i}.",
        "filler": "",
        "target": f"Launch {i}: day {i}, city {i}.",
        "meta": {
            "source": source,
            "generator": "test-gen",
            "label": label,
            "label_kind": "sentiment" if label in ("positive", "negative") else None,
            "facts": [f"day {i}", f"city {i}"],
            "question": f"Which day was launch {i}?",
            "options": [f"day {i}", "day 999", "never", "unknown"],
            "answer_idx": 0,
        },
    }
    if ex_type == "B":
        ex["filler"] = "meanwhile unrelated routine matters continued as usual"
        ex["meta"]["distance_target_tokens"] = 8
        ex["meta"]["filler_word_count"] = 7
    if ex_type == "C":
        ex["composition"] = [COMPRESS_TOKEN, REASON_TOKEN]
    ex.update(overrides)
    return seal_example(ex)


class FakeTokenizer:
    """Minimal whitespace tokenizer with real special-token semantics."""

    def __init__(self):
        self.vocab: dict[str, int] = {"<eos>": 0, COMPRESS_TOKEN: 1,
                                      RECALL_TOKEN: 2, REASON_TOKEN: 3}
        self.eos_token = "<eos>"
        self.eos_token_id = 0
        self.pad_token_id = 0

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocab.get(token, -1)

    def _id(self, word: str) -> int:
        if word not in self.vocab:
            self.vocab[word] = len(self.vocab)
        return self.vocab[word]

    def __call__(self, text: str, **_):
        # newline-separated render: split on whitespace keeps special tokens whole
        ids = [self._id(w) for w in text.split()]
        return {"input_ids": ids}


@pytest.fixture()
def fake_tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


def encode_layout(fake_tokenizer: FakeTokenizer, n_context: int = 4,
                  n_filler: int = 2, n_target: int = 3) -> tuple[torch.Tensor, int, int]:
    """Build ids = [ctx... COMPRESS filler... RECALL target...]; return (ids, c, r)."""
    words = [f"ctx{i}" for i in range(n_context)]
    words.append(COMPRESS_TOKEN)
    words += [f"fill{i}" for i in range(n_filler)]
    words.append(RECALL_TOKEN)
    words += [f"tgt{i}" for i in range(n_target)]
    ids = torch.tensor([fake_tokenizer(" ".join(words))["input_ids"]])
    return ids, n_context, n_context + 1 + n_filler
