"""Contract v2: schema, IDs, hashing, JSONL IO, disjointness, manifests."""

from __future__ import annotations

import json

import pytest

from src.data_contract import (
    ContractError,
    assert_disjoint,
    canonical_json,
    content_id,
    example_id,
    load_jsonl_validated,
    overlap_report,
    save_jsonl_atomic,
    seal_example,
    upgrade_legacy_example,
    validate_example,
    validate_mcq,
)
from tests.conftest import make_example


class TestIdentity:
    def test_content_id_ignores_whitespace_and_nfc(self):
        assert content_id("a  b\nc") == content_id("a b c")

    def test_example_id_stable_under_key_order(self):
        ex = make_example(1)
        shuffled = json.loads(canonical_json(ex))
        assert example_id(ex) == example_id(shuffled)

    def test_example_id_ignores_annotation_changes(self):
        ex = make_example(2)
        before = ex["example_id"]
        ex["meta"]["question"] = "different question?"
        ex["meta"]["facts"] = ["other"]
        assert example_id(ex) == before

    def test_example_id_changes_with_target(self):
        a, b = make_example(3), make_example(3)
        b["target"] = "something else entirely"
        assert example_id(a) != example_id(b)


class TestValidation:
    def test_valid_examples_pass(self):
        for ex_type in ("A", "B", "C"):
            validate_example(make_example(1, ex_type=ex_type))

    def test_missing_context_rejected(self):
        ex = make_example(1)
        ex["context"] = "  "
        with pytest.raises(ContractError, match="context"):
            validate_example(seal_example(ex) if False else ex)

    def test_type_b_requires_filler_and_distance(self):
        ex = make_example(1, ex_type="B")
        ex["meta"]["distance_target_tokens"] = None
        with pytest.raises(ContractError, match="distance_target_tokens"):
            validate_example(ex)

    def test_type_a_must_not_have_filler(self):
        ex = make_example(1)
        ex["filler"] = "spurious"
        with pytest.raises(ContractError, match="empty filler"):
            validate_example(ex)

    def test_type_c_requires_single_compress(self):
        from src.config import COMPRESS_TOKEN
        ex = make_example(1, ex_type="C")
        ex["composition"] = [COMPRESS_TOKEN, COMPRESS_TOKEN]
        with pytest.raises(ContractError, match="exactly once"):
            validate_example(ex)

    def test_label_requires_valid_kind(self):
        ex = make_example(1)
        ex["meta"]["label"] = "sciencey"
        with pytest.raises(ContractError, match="vocabulary"):
            validate_example(ex)

    def test_mcq_five_options_rejected(self):
        meta = {"question": "q?", "options": ["a", "b", "c", "d", "e"], "answer_idx": 1}
        with pytest.raises(ContractError, match="options"):
            validate_mcq(meta)

    def test_mcq_duplicate_options_rejected(self):
        meta = {"question": "q?", "options": ["a", "A", "c", "d"], "answer_idx": 0}
        with pytest.raises(ContractError, match="duplicate"):
            validate_mcq(meta)

    def test_tampered_example_id_rejected(self):
        ex = make_example(1)
        ex["example_id"] = "0" * 16
        with pytest.raises(ContractError, match="example_id"):
            validate_example(ex)


class TestLegacyUpgrade:
    def test_v0_word_distance_becomes_target_tokens(self):
        legacy = {
            "type": "B", "context": "ctx words here", "target": "tgt",
            "filler": "one two three",
            "meta": {"source": "synthetic", "generator": "g",
                     "label": "positive", "label_kind": "sentiment",
                     "distance": 3},
        }
        up = upgrade_legacy_example(legacy)
        assert up["schema_version"] == 2
        assert up["meta"]["distance_target_tokens"] == 3
        assert up["meta"]["filler_word_count"] == 3
        assert "distance" not in up["meta"]

    def test_v0_invalid_mcq_dropped_not_fatal(self):
        legacy = {
            "type": "A", "context": "ctx", "target": "tgt", "filler": "",
            "meta": {"source": "synthetic", "generator": "g", "label": None,
                     "question": "q?", "options": ["a", "b", "c", "d", "e"],
                     "answer_idx": 1},
        }
        up = upgrade_legacy_example(legacy)
        assert "options" not in up["meta"]

    def test_v0_label_without_kind_gets_inferred(self):
        # v0 rows carried `label` but no `label_kind`; vocabularies are
        # disjoint so the kind must be recovered, not crash the rebuild
        legacy = {
            "type": "A", "context": "ctx", "target": "tgt", "filler": "",
            "meta": {"source": "synthetic", "generator": "g",
                     "label": "science"},
        }
        up = upgrade_legacy_example(legacy)
        assert up["meta"]["label_kind"] == "topic"


class TestIOAndSplits:
    def test_jsonl_roundtrip_and_line_errors(self, tmp_path):
        rows = [make_example(i) for i in range(3)]
        path = tmp_path / "x.jsonl"
        save_jsonl_atomic(rows, path)
        assert load_jsonl_validated(path) == rows

        path.write_text(path.read_text() + "{broken\n")
        with pytest.raises(ContractError, match=r"x\.jsonl:4"):
            load_jsonl_validated(path)

    def test_assert_disjoint_catches_leak(self):
        shared = make_example(1)
        splits = {"train": [shared, make_example(2)],
                  "test": [shared, make_example(3)]}
        with pytest.raises(ContractError, match="leakage"):
            assert_disjoint(splits)

    def test_assert_disjoint_extra_pairs(self):
        shared = make_example(1)
        splits = {"train": [make_example(2)],
                  "eval": [shared], "test": [shared]}
        assert_disjoint(splits)  # train is clean -> passes without pairs
        with pytest.raises(ContractError, match="'eval' and 'test'"):
            assert_disjoint(splits, pairs=[("eval", "test")])

    def test_overlap_report_counts(self):
        a, b, c = make_example(1), make_example(2), make_example(3)
        report = overlap_report({"train": [a, b], "test": [b, c]})
        assert report["test/train"] == 1


class TestPreparedSplits:
    def test_build_splits_disjoint_and_deterministic(self):
        from data.generation.prepare_dataset import build_splits
        synthetic = [make_example(i) for i in range(40)]
        heldout = [make_example(100 + i, source="handwritten", label=None)
                   for i in range(6)]
        s1 = build_splits(list(synthetic), list(heldout), 0.1, 0.1, seed=42)
        s2 = build_splits(list(reversed(synthetic)), list(heldout), 0.1, 0.1, seed=42)

        assert overlap_report(s1)["test/train"] == 0
        assert overlap_report(s1)["probe/train"] == 0
        assert overlap_report(s1)["eval/train"] == 0
        # order-independent: same input set => same split membership
        for name in ("train", "eval", "test"):
            assert {e["example_id"] for e in s1[name]} == \
                   {e["example_id"] for e in s2[name]}
        # every synthetic row lands in exactly one of train/eval/test-in-style
        n_synth_in_test = sum(1 for e in s1["test"]
                              if e["meta"]["source"] == "synthetic")
        assert len(s1["train"]) + len(s1["eval"]) + n_synth_in_test == len(synthetic)
