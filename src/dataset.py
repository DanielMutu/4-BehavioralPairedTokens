"""Dataset formatting and DataLoader for the three example types.

JSONL schema (data/processed/*.jsonl):
{
  "type": "A" | "B" | "C",
  "context": "...",          # source text to compress
  "filler": "...",           # text between [COMPRESS] and [RECALL] ("" for type A)
  "target": "...",           # expected output after [RECALL]
  "composition": ["[COMPRESS]", "[REASON]"],   # type C only, token order
  "meta": {                  # optional, used by eval/probing
      "facts": [...], "question": "...", "options": [...], "answer_idx": 0,
      "label": "positive",   # probe label (sentiment/topic)
      "generator": "...", "distance": 0, "source": "synthetic|handwritten|public"
  }
}

Rendered training text:
    {context}\n[COMPRESS]\n{filler?}\n[RECALL]\n{target}{eos}
Type C replaces [COMPRESS] with the composition sequence.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from src.bottleneck import LayoutError, validate_layout
from src.config import COMPRESS_TOKEN, RECALL_TOKEN, TrainConfig
from src.model import find_token_positions
from src.utils import load_jsonl


def render_prompt(example: dict) -> str:
    """Everything up to and including [RECALL] — the inference-time prompt."""
    head = "".join(example.get("composition") or [COMPRESS_TOKEN])
    parts = [example["context"].rstrip(), head]
    filler = (example.get("filler") or "").strip()
    if filler:
        parts.append(filler)
    parts.append(RECALL_TOKEN)
    return "\n".join(parts) + "\n"


def render_full(example: dict, eos_token: str) -> str:
    return render_prompt(example) + example["target"].strip() + eos_token


class BehavioralTokenDataset(Dataset):
    def __init__(self, path: str, tokenizer, cfg: TrainConfig):
        self.examples = load_jsonl(path, max_examples=cfg.max_examples)
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.compress_id = tokenizer.convert_tokens_to_ids(COMPRESS_TOKEN)
        self.recall_id = tokenizer.convert_tokens_to_ids(RECALL_TOKEN)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        prompt = render_prompt(ex)
        full = render_full(ex, self.tokenizer.eos_token)

        ids = self.tokenizer(full, truncation=True, max_length=self.cfg.max_length,
                             add_special_tokens=False)["input_ids"]
        # P0 invariant: truncation must NEVER silently remove a paired token or
        # the whole target — a row like that would train with the architecture
        # effectively disabled. Fail loudly with the example identity instead.
        try:
            _, recall_pos = validate_layout(
                torch.tensor([ids]), self.tokenizer, require_recall=True)
        except LayoutError as e:
            raise LayoutError(
                f"example {idx} (id={ex.get('example_id', '?')}) breaks the "
                f"COMPRESS/RECALL layout after tokenization to "
                f"{len(ids)}<=max_length={self.cfg.max_length} tokens: {e}") from e
        if int(recall_pos) >= len(ids) - 1:
            raise LayoutError(
                f"example {idx} (id={ex.get('example_id', '?')}): no target "
                f"tokens survive truncation (max_length={self.cfg.max_length})")
        labels = list(ids)
        if self.cfg.loss_on_target_only:
            n_prompt = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
            n_prompt = min(n_prompt, len(ids))
            labels[:n_prompt] = [-100] * n_prompt

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def collate(self, batch: list[dict]) -> dict:
        pad_id = self.tokenizer.pad_token_id
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attention = [], [], []
        for b in batch:
            n_pad = max_len - len(b["input_ids"])
            input_ids.append(torch.cat([b["input_ids"], torch.full((n_pad,), pad_id, dtype=torch.long)]))
            labels.append(torch.cat([b["labels"], torch.full((n_pad,), -100, dtype=torch.long)]))
            attention.append(torch.cat([torch.ones(len(b["input_ids"]), dtype=torch.long),
                                        torch.zeros(n_pad, dtype=torch.long)]))
        input_ids = torch.stack(input_ids)
        return {
            "input_ids": input_ids,
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention),
            # positions of the paired tokens, needed for consistency loss
            # and hidden-state variance monitoring
            "compress_pos": find_token_positions(input_ids, self.compress_id),
            "recall_pos": find_token_positions(input_ids, self.recall_id),
        }


def make_dataloader(path: str, tokenizer, cfg: TrainConfig, shuffle: bool) -> DataLoader:
    ds = BehavioralTokenDataset(path, tokenizer, cfg)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                      collate_fn=ds.collate)
