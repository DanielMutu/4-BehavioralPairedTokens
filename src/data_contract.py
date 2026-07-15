"""Dataset contract v2: schema validation, canonical IDs, hashing, manifests.

Every dataset artifact (raw generation output, processed split, probe view)
must pass through this module so that:
  - every example carries a stable `example_id` / `content_id`;
  - malformed rows are rejected with a `path:line` error, never silently kept;
  - splits can be proven pairwise disjoint before being written;
  - each build emits a manifest with hashes, counts and provenance.

Schema (JSONL, one example per line):
{
  "schema_version": 2,
  "example_id":  "<sha256[:16] of canonical example>",   # filled by this module
  "content_id":  "<sha256[:16] of normalized context>",  # filled by this module
  "type": "A" | "B" | "C",
  "context": str,          # non-empty
  "filler": str,           # "" for type A/C
  "target": str,           # non-empty
  "composition": [tok, tok],   # type C only, exactly one [COMPRESS]
  "meta": {
      "source": "synthetic" | "cnn_dailymail" | "handwritten",
      "generator": str,
      "label": str | None,
      "label_kind": "sentiment" | "topic" | None,
      "distance_target_tokens": int | None,   # type B: requested token distance
      "distance_actual_tokens": int | None,   # measured on the training tokenizer
      "filler_word_count": int | None,
      "facts": [str, ...],
      "question": str | None,
      "options": [str x4] | None,
      "answer_idx": 0..3 | None,
      ...provenance fields (source_dataset, source_split, prompt_sha256, ...)
  }
}

Legacy v0 rows (no schema_version, `distance` in words) are upgraded by
`upgrade_legacy_example` so old raw files can be rebuilt under the v2 contract.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

SCHEMA_VERSION = 2

EXAMPLE_TYPES = ("A", "B", "C")
SOURCES = ("synthetic", "cnn_dailymail", "handwritten")
LABEL_KINDS: dict[str, tuple[str, ...]] = {
    "sentiment": ("positive", "negative"),
    "topic": ("science", "sports", "finance", "culture"),
}
N_MCQ_OPTIONS = 4

# Fields that define example identity. Annotations (facts/question/options)
# are deliberately excluded: re-annotating MCQs must not change example_id.
_IDENTITY_FIELDS = ("type", "context", "filler", "target", "composition")


class ContractError(ValueError):
    """A dataset row violates the v2 contract."""


# ------------------------------------------------------------- canonical text


def normalize_text(text: str) -> str:
    """Canonical form used for content identity: NFC + collapsed whitespace."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_id(context: str) -> str:
    return hashlib.sha256(normalize_text(context).encode("utf-8")).hexdigest()[:16]


def example_id(example: dict) -> str:
    identity = {k: example.get(k) for k in _IDENTITY_FIELDS}
    identity["context"] = normalize_text(identity["context"] or "")
    identity["target"] = normalize_text(identity["target"] or "")
    identity["filler"] = normalize_text(identity["filler"] or "")
    return hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()[:16]


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ validation


def validate_mcq(meta: dict, where: str = "") -> None:
    """Shared MCQ shape check (generalizes annotate_mcq.valid)."""
    options = meta.get("options")
    question = meta.get("question")
    answer_idx = meta.get("answer_idx")
    if options is None and question is None and answer_idx is None:
        return  # MCQ annotation entirely absent is allowed
    if not isinstance(question, str) or not question.strip():
        raise ContractError(f"{where}: MCQ present but question is empty")
    if (not isinstance(options, list) or len(options) != N_MCQ_OPTIONS
            or not all(isinstance(o, str) and o.strip() for o in options)):
        raise ContractError(f"{where}: options must be {N_MCQ_OPTIONS} non-empty strings")
    if len({o.strip().lower() for o in options}) != N_MCQ_OPTIONS:
        raise ContractError(f"{where}: duplicate MCQ options")
    if (isinstance(answer_idx, bool) or not isinstance(answer_idx, int)
            or not 0 <= answer_idx < N_MCQ_OPTIONS):
        raise ContractError(f"{where}: answer_idx must be an int in [0,{N_MCQ_OPTIONS - 1}]")


def validate_example(example: dict, where: str = "") -> None:
    """Raise ContractError unless `example` satisfies the v2 schema."""
    if example.get("schema_version") != SCHEMA_VERSION:
        raise ContractError(f"{where}: schema_version must be {SCHEMA_VERSION}")

    ex_type = example.get("type")
    if ex_type not in EXAMPLE_TYPES:
        raise ContractError(f"{where}: type must be one of {EXAMPLE_TYPES}, got {ex_type!r}")
    for field in ("context", "target"):
        if not isinstance(example.get(field), str) or not example[field].strip():
            raise ContractError(f"{where}: {field} must be a non-empty string")
    if not isinstance(example.get("filler"), str):
        raise ContractError(f"{where}: filler must be a string ('' when unused)")

    meta = example.get("meta")
    if not isinstance(meta, dict):
        raise ContractError(f"{where}: meta must be a dict")
    if meta.get("source") not in SOURCES:
        raise ContractError(f"{where}: meta.source must be one of {SOURCES}")

    label, kind = meta.get("label"), meta.get("label_kind")
    if label is not None:
        if kind not in LABEL_KINDS:
            raise ContractError(f"{where}: label {label!r} without valid label_kind")
        if label not in LABEL_KINDS[kind]:
            raise ContractError(f"{where}: label {label!r} not in {kind} vocabulary")

    if ex_type == "B":
        if not example["filler"].strip():
            raise ContractError(f"{where}: type B requires a non-empty filler")
        dist = meta.get("distance_target_tokens")
        if not isinstance(dist, int) or dist <= 0:
            raise ContractError(f"{where}: type B requires distance_target_tokens > 0")
    else:
        if example["filler"].strip():
            raise ContractError(f"{where}: type {ex_type} must have empty filler")

    composition = example.get("composition")
    if ex_type == "C":
        from src.config import COMPRESS_TOKEN, SPECIAL_TOKENS
        if (not isinstance(composition, list) or len(composition) < 2
                or any(tok not in SPECIAL_TOKENS for tok in composition)):
            raise ContractError(f"{where}: type C requires a composition of special tokens")
        if composition.count(COMPRESS_TOKEN) != 1:
            raise ContractError(f"{where}: composition must contain [COMPRESS] exactly once")
    elif composition:
        raise ContractError(f"{where}: composition is only allowed on type C")

    validate_mcq(meta, where)

    expected = example_id(example)
    if example.get("example_id") != expected:
        raise ContractError(f"{where}: example_id mismatch (expected {expected})")
    if example.get("content_id") != content_id(example["context"]):
        raise ContractError(f"{where}: content_id mismatch")


def seal_example(example: dict) -> dict:
    """Fill schema_version + IDs, then validate. Returns the same dict."""
    example["schema_version"] = SCHEMA_VERSION
    example["example_id"] = example_id(example)
    example["content_id"] = content_id(example["context"])
    validate_example(example)
    return example


def upgrade_legacy_example(example: dict, where: str = "") -> dict:
    """Upgrade a v0 row (no schema_version) to the v2 contract, in place."""
    if example.get("schema_version") == SCHEMA_VERSION:
        return example
    meta = example.setdefault("meta", {})
    meta.setdefault("source", "synthetic")
    if meta.get("source") == "public":  # early naming in v0 docs
        meta["source"] = "cnn_dailymail"
    # v0 stored a word count in `distance`; keep it as the *target* and
    # leave actual token distance to be measured by the dataset pipeline.
    if "distance" in meta and "distance_target_tokens" not in meta:
        legacy = meta.pop("distance")
        if example.get("type") == "B" and isinstance(legacy, int) and legacy > 0:
            meta["distance_target_tokens"] = legacy
            meta["filler_word_count"] = len((example.get("filler") or "").split())
    meta.pop("distance", None)  # never keep the ambiguous legacy key around
    label = meta.get("label")
    if label is None:
        meta["label_kind"] = None
    elif meta.get("label_kind") not in LABEL_KINDS:
        # v0 rows carried `label` without `label_kind`; the vocabularies are
        # disjoint, so the kind is recoverable from the value itself.
        for kind, vocab in LABEL_KINDS.items():
            if label in vocab:
                meta["label_kind"] = kind
                break
    # v0 kept invalid MCQ shapes (e.g. 5 options); drop the annotation rather
    # than rejecting the whole row — MCQs are eval-side metadata.
    try:
        validate_mcq(meta, where)
    except ContractError:
        for key in ("question", "options", "answer_idx"):
            meta.pop(key, None)
    return seal_example(example)


# --------------------------------------------------------------------- JSONL IO


def load_jsonl_validated(path: str | Path,
                         upgrade: bool = False,
                         max_examples: int | None = None) -> list[dict]:
    """Read + validate a JSONL dataset, with `path:line` errors."""
    out: list[dict] = []
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            where = f"{path}:{lineno}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ContractError(f"{where}: invalid JSON ({e})") from e
            if upgrade:
                row = upgrade_legacy_example(row, where)
            else:
                validate_example(row, where)
            out.append(row)
            if max_examples is not None and len(out) >= max_examples:
                break
    return out


def save_jsonl_atomic(examples: Iterable[dict], path: str | Path) -> str:
    """Atomic UTF-8 write; returns the file's sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return file_sha256(path)


# --------------------------------------------------------------- split checking


def assert_disjoint(splits: dict[str, list[dict]],
                    protected: str = "train",
                    pairs: Iterable[tuple[str, str]] = (),
                    key: Callable[[dict], str] = lambda ex: ex["content_id"]) -> None:
    """Fail if any non-train split shares a content_id with `protected`.

    `pairs` adds extra pairwise checks (e.g. [("eval", "test")]). probe is
    deliberately NOT in `pairs` by default: it is a derived view of eval+test.
    """
    def ids(name: str) -> set[str]:
        return {key(ex) for ex in splits[name]}

    checks = [(protected, other) for other in splits if other != protected]
    checks += [p for p in pairs if p[0] in splits and p[1] in splits]
    for a, b in checks:
        overlap = ids(a) & ids(b)
        if overlap:
            sample = sorted(overlap)[:5]
            raise ContractError(
                f"split leakage: {len(overlap)} content_ids shared between "
                f"'{a}' and '{b}' (e.g. {sample})")


def overlap_report(splits: dict[str, list[dict]]) -> dict[str, int]:
    names = sorted(splits)
    report = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ids_a = {ex["content_id"] for ex in splits[a]}
            ids_b = {ex["content_id"] for ex in splits[b]}
            report[f"{a}/{b}"] = len(ids_a & ids_b)
    return report


# -------------------------------------------------------------------- manifest


def _git_state() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:  # noqa: BLE001 — manifests must work outside git too
        return {"commit": None, "dirty": None}


def split_counts(rows: list[dict]) -> dict:
    def counter(getter: Callable[[dict], Any]) -> dict:
        counts: dict[str, int] = {}
        for ex in rows:
            k = str(getter(ex))
            counts[k] = counts.get(k, 0) + 1
        return dict(sorted(counts.items()))

    return {
        "n": len(rows),
        "by_type": counter(lambda ex: ex["type"]),
        "by_source": counter(lambda ex: ex["meta"]["source"]),
        "by_generator": counter(lambda ex: ex["meta"].get("generator")),
        "by_label_kind": counter(lambda ex: ex["meta"].get("label_kind")),
        "by_label": counter(lambda ex: ex["meta"].get("label")),
        "by_distance_target": counter(
            lambda ex: ex["meta"].get("distance_target_tokens")),
        "n_mcq": sum(1 for ex in rows if ex["meta"].get("options")),
    }


def build_manifest(*, splits: dict[str, Path], inputs: dict[str, Path],
                   seed: int, split_algorithm: str,
                   tokenizer_name: str | None = None,
                   extra: dict | None = None) -> dict:
    loaded = {name: load_jsonl_validated(p) for name, p in splits.items()}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": _git_state(),
        "seed": seed,
        "split_algorithm": split_algorithm,
        "tokenizer": tokenizer_name,
        "inputs": {n: {"path": str(p), "sha256": file_sha256(p)}
                   for n, p in inputs.items() if Path(p).exists()},
        "splits": {n: {"path": str(p), "sha256": file_sha256(p),
                       **split_counts(loaded[n])}
                   for n, p in splits.items()},
        "overlap": overlap_report(loaded),
    }
    assert_disjoint(loaded, protected="train")
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(manifest: dict, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return file_sha256(path)


def verify_manifest(manifest_path: str | Path) -> dict:
    """Re-hash every split listed in a manifest; raise on mismatch."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    for name, entry in manifest["splits"].items():
        actual = file_sha256(entry["path"])
        if actual != entry["sha256"]:
            raise ContractError(
                f"manifest mismatch for split '{name}': {entry['path']} has "
                f"sha256 {actual}, manifest says {entry['sha256']}")
    return manifest


@dataclasses.dataclass
class CohortSelection:
    """A frozen, ID-based cohort for scientific runs (no 'first N' slices)."""

    example_ids: list[str]
    dataset_sha256: str
    description: str = ""

    def select(self, rows: list[dict]) -> list[dict]:
        by_id = {ex["example_id"]: ex for ex in rows}
        missing = [i for i in self.example_ids if i not in by_id]
        if missing:
            raise ContractError(
                f"cohort references {len(missing)} missing example_ids "
                f"(e.g. {missing[:3]})")
        return [by_id[i] for i in self.example_ids]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(dataclasses.asdict(self), indent=2),
                              encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CohortSelection":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
