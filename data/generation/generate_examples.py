"""Generate synthetic training examples (types A, B, C) with MULTIPLE LLMs.

Anti-leakage (CLAUDE.md): generators are rotated round-robin so the model
cannot learn a single generator's compression style. Combine the output with
handwritten examples (data/handwritten/) and public data (import_public.py)
via prepare_dataset.py.

Usage (from project root):
    python -m data.generation.generate_examples --type A --n 500
    python -m data.generation.generate_examples --type B --n 500
    python -m data.generation.generate_examples --type C --n 500
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import COMPRESS_TOKEN, REASON_TOKEN  # noqa: E402
from src.utils import save_jsonl, set_seed  # noqa: E402

TYPE_B_DISTANCES = [3, 10, 20, 50, 100]

TOPICS = [
    "a scientific discovery", "a small town event", "a tech product launch",
    "a historical episode", "a sports match", "a cooking recipe gone wrong",
    "a financial news item", "a nature documentary scene", "a court case",
    "a space mission", "a medical study", "a music festival",
]

GEN_PROMPT = """Generate ONE training example for a text-compression task as a single JSON object, no markdown fences.

Write a short text (4-7 sentences) about {topic}, with a clear {label_kind} of "{label}".
Then produce:
- "context": the text
- "target": a very dense compression of the context (1-2 sentences, keep every key fact)
- "facts": 3-5 short factual strings stated verbatim-ish in the context (names, numbers, places)
- "question": one multiple-choice question about the context
- "options": exactly 4 answer options
- "answer_idx": index (0-3) of the correct option

Return ONLY the JSON object."""

FILLER_POOL = (
    "Meanwhile, unrelated routine matters continued as usual in the background. "
    "The weather that week was unremarkable, with mild temperatures and light wind. "
    "Several administrative notes were filed, none of which changed anything important. "
    "People went about their ordinary schedules without paying attention to the news. "
    "A few minor announcements were made elsewhere, all of them quickly forgotten. "
).split()

LABEL_SETS = {
    "sentiment": ["positive", "negative"],
    "topic": ["science", "sports", "finance", "culture"],
}


def call_generator(gen: dict, prompt: str, timeout: int = 120) -> str:
    headers = {"Content-Type": "application/json"}
    if gen.get("api_key_env"):
        key = os.environ.get(gen["api_key_env"])
        if not key:
            raise RuntimeError(f"env var {gen['api_key_env']} not set for {gen['name']}")
        headers["Authorization"] = f"Bearer {key}"
    resp = requests.post(
        f"{gen['base_url']}/chat/completions",
        headers=headers,
        json={"model": gen["model"], "temperature": 0.9,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def parse_json_reply(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1].removeprefix("json").strip()
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


def make_filler(n_words: int, rng: random.Random) -> str:
    words = [rng.choice(FILLER_POOL) for _ in range(n_words)]
    return " ".join(words)


def build_example(raw: dict, ex_type: str, generator: str,
                  label_kind: str, label: str, rng: random.Random) -> dict:
    meta = {
        "facts": raw.get("facts", []),
        "question": raw.get("question"),
        "options": raw.get("options"),
        "answer_idx": raw.get("answer_idx"),
        "label": label,
        "label_kind": label_kind,
        "generator": generator,
        "source": "synthetic",
        "distance": 0,
    }
    ex = {"type": ex_type, "context": raw["context"].strip(),
          "filler": "", "target": raw["target"].strip(), "meta": meta}
    if ex_type == "B":
        dist = rng.choice(TYPE_B_DISTANCES)
        ex["filler"] = make_filler(dist, rng)
        meta["distance"] = dist
    elif ex_type == "C":
        order = rng.choice([[COMPRESS_TOKEN, REASON_TOKEN],
                            [REASON_TOKEN, COMPRESS_TOKEN]])
        ex["composition"] = order
        # type C target: reasoning step + compressed answer
        ex["target"] = (f"Reasoning: the key information must be preserved. "
                        f"{raw['target'].strip()}")
    return ex


def generate_one(i: int, gen: dict, ex_type: str, label_kind: str) -> dict | None:
    # per-task rng: deterministic given the task index, safe across threads
    rng = random.Random(10_000 + i)
    label = rng.choice(LABEL_SETS[label_kind])
    prompt = GEN_PROMPT.format(topic=rng.choice(TOPICS),
                               label_kind=label_kind, label=label)
    try:
        raw = parse_json_reply(call_generator(gen, prompt))
        assert raw.get("context") and raw.get("target")
        return build_example(raw, ex_type, gen["name"], label_kind, label, rng)
    except Exception as e:  # noqa: BLE001 — generation is best-effort
        print(f"[skip] {gen['name']}: {type(e).__name__}: {e}")
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--type", required=True, choices=["A", "B", "C"])
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--label-kind", default="sentiment", choices=list(LABEL_SETS))
    p.add_argument("--generators", default=str(Path(__file__).parent / "generators.json"))
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    set_seed(42)
    generators = json.loads(Path(args.generators).read_text())["generators"]
    out_path = args.out or f"data/raw/generated_type{args.type}.jsonl"

    attempts = int(args.n * 1.5)  # over-provision for failed/invalid replies
    examples = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(generate_one, i, generators[i % len(generators)],
                               args.type, args.label_kind)
                   for i in range(attempts)]
        for fut in as_completed(futures):
            ex = fut.result()
            if ex is not None and len(examples) < args.n:
                examples.append(ex)
                if len(examples) % 25 == 0:
                    save_jsonl(examples, out_path)  # periodic flush
                    print(f"{len(examples)}/{args.n} examples...")
            if len(examples) >= args.n:
                for f in futures:
                    f.cancel()
                break

    save_jsonl(examples, out_path)
    by_gen = {}
    for ex in examples:
        by_gen[ex["meta"]["generator"]] = by_gen.get(ex["meta"]["generator"], 0) + 1
    print(f"Done: {len(examples)} examples -> {out_path} "
          f"(per generator: {by_gen})")


if __name__ == "__main__":
    main()
