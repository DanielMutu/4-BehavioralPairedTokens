"""Annotate CNN/DailyMail rows with MCQ options + facts for Exp 2 eval.

The held-out test split (out-of-style, anti-leakage rule) was imported
without MCQ annotations, so the MCQ metric currently only measures the
synthetic style. This script fills in question/options/answer_idx/facts
for rows that lack them, using the same multi-family generator round-robin
as generate_examples.py. Annotations are eval-only (never trained on), so
generator style cannot leak into the model.

Correct-answer position is shuffled deterministically per row: LLM
generators tend to put the right option first, and a positional bias
would corrupt the log-likelihood MCQ metric.

Usage (from project root):
    python -m data.generation.annotate_mcq --data data/processed/test.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.generation.generate_examples import call_generator, parse_json_reply  # noqa: E402
from src.utils import set_seed  # noqa: E402

MAX_ARTICLE_CHARS = 6000

ANNOTATE_PROMPT = """You will receive a news article excerpt. Create ONE multiple-choice question that tests memory of a specific fact stated in the article (a name, number, place, or event).

Return a single JSON object, no markdown fences:
- "question": the question
- "options": exactly 4 short answer options; exactly one is correct per the article
- "answer_idx": index (0-3) of the correct option
- "facts": 3-5 short factual strings stated in the article (names, numbers, places)

The wrong options must be plausible but clearly incorrect per the article.

Article:
{article}"""


def valid(raw: dict) -> bool:
    opts = raw.get("options")
    return (
        isinstance(raw.get("question"), str) and raw["question"].strip() != ""
        and isinstance(opts, list) and len(opts) == 4
        and all(isinstance(o, str) and o.strip() for o in opts)
        and len({o.strip().lower() for o in opts}) == 4
        and isinstance(raw.get("answer_idx"), int) and 0 <= raw["answer_idx"] <= 3
    )


def annotate_one(i: int, row: dict, generators: list[dict]) -> tuple[int, dict | None]:
    rng = random.Random(20_000 + i)  # deterministic per row, thread-safe
    prompt = ANNOTATE_PROMPT.format(article=row["context"][:MAX_ARTICLE_CHARS])
    for attempt in range(3):  # on failure, retry with the next family
        gen = generators[(i + attempt) % len(generators)]
        try:
            raw = parse_json_reply(call_generator(gen, prompt))
            if not valid(raw):
                raise ValueError("invalid annotation shape")
            # shuffle correct-answer position (see module docstring)
            perm = rng.sample(range(4), 4)
            options = [raw["options"][j].strip() for j in perm]
            answer_idx = perm.index(raw["answer_idx"])
            facts = [f for f in raw.get("facts", []) if isinstance(f, str)][:5]
            return i, {"question": raw["question"].strip(), "options": options,
                       "answer_idx": answer_idx, "facts": facts,
                       "mcq_annotator": gen["name"]}
        except Exception as e:  # noqa: BLE001 — best-effort, try next family
            print(f"[retry {attempt + 1}] row {i} via {gen['name']}: "
                  f"{type(e).__name__}: {e}")
    return i, None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/processed/test.jsonl")
    p.add_argument("--source", default="cnn_dailymail",
                   help="only annotate rows with this meta.source")
    p.add_argument("--generators", default=str(Path(__file__).parent / "generators.json"))
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()

    set_seed(42)
    generators = json.loads(Path(args.generators).read_text())["generators"]
    path = Path(args.data)
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

    todo = [i for i, r in enumerate(rows)
            if (r.get("meta") or {}).get("source") == args.source
            and not (r["meta"].get("options"))]
    print(f"{len(todo)} rows to annotate out of {len(rows)}")

    done = failed = 0
    by_gen: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(annotate_one, i, rows[i], generators) for i in todo]
        for fut in as_completed(futures):
            i, ann = fut.result()
            if ann is None:
                failed += 1
                continue
            meta = rows[i]["meta"]
            meta.update({k: ann[k] for k in
                         ("question", "options", "answer_idx", "mcq_annotator")})
            if not meta.get("facts"):
                meta["facts"] = ann["facts"]
            by_gen[ann["mcq_annotator"]] = by_gen.get(ann["mcq_annotator"], 0) + 1
            done += 1
            if done % 25 == 0:
                print(f"{done}/{len(todo)} annotated...")

    # atomic write: full file to tmp, then replace
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    tmp.replace(path)
    print(f"Done: {done} annotated, {failed} failed -> {path} "
          f"(per generator: {by_gen})")


if __name__ == "__main__":
    main()
