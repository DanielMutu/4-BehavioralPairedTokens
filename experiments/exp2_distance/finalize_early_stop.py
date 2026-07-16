"""Finalize Exp 2 from the records written so far (declared early stop).

The pre-registered PRIMARY verdict needs only conditions 2 (prompt_summary,
from Exp 0 v2) and 4 (true_bottleneck) — both complete. This script computes
the formal verdict with the SAME functions the full runner would have used,
reports every completed condition, and explicitly marks the not-run ones as
a declared deviation (see decisions.md 2026-07-17). The run stays resumable:
records are untouched.

Usage:
    python experiments/exp2_distance/finalize_early_stop.py \
        --out results/exp2_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.exp2_distance.run_exp2 import (  # noqa: E402
    DIAGNOSTIC,
    GATING,
    load_exp0_records,
    verdict_block,
)
from src.data_contract import CohortSelection  # noqa: E402
from src.stats import bootstrap_ci, mean  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", default="results/exp0_v2_cohort.json")
    p.add_argument("--exp0-records", default="results/exp0_v2_records.jsonl")
    p.add_argument("--records", default="results/exp2_records.jsonl")
    p.add_argument("--out", default="results/exp2_results.json")
    args = p.parse_args()

    cohort = CohortSelection.load(args.cohort)
    ids = cohort.example_ids

    exp0 = load_exp0_records(args.exp0_records)
    records_by_cond: dict[str, dict[str, dict]] = {}
    for i in ids:
        r0 = exp0[i]
        common = {"example_id": i, "source": r0["source"], "type": r0["type"],
                  "distance": r0["distance"], "answer_idx": r0["answer_idx"]}
        records_by_cond.setdefault("full_context_base", {})[i] = {
            **common, "correct": r0["full_context_correct"]}
        records_by_cond.setdefault("prompt_summary", {})[i] = {
            **common, "correct": r0["summary_correct"]}

    raw = [json.loads(line) for line in Path(args.records).read_text().splitlines()
           if line.strip()]
    counts = Counter(r["condition"] for r in raw)
    for r in raw:
        records_by_cond.setdefault(r["condition"], {})[r["example_id"]] = r

    complete = [c for c in GATING + DIAGNOSTIC
                if len(records_by_cond.get(c, {})) == len(ids)]
    not_run = [c for c in GATING + DIAGNOSTIC if c not in complete]
    print(f"complete: {complete}")
    print(f"not run / partial (declared early stop): "
          f"{[(c, counts.get(c, 0)) for c in not_run]}")

    results = {
        "n": len(ids),
        "early_stop": {
            "declared": True,
            "reason": "primary verdict already determined by conditions 2+4; "
                      "key causal control (anchor_removed) complete; remaining "
                      "conditions deferred (resumable via --resume). See "
                      "decisions.md 2026-07-17.",
            "conditions_not_run": {c: counts.get(c, 0) for c in not_run},
        },
        "conditions": {
            cond: {
                "accuracy": bootstrap_ci(
                    [float(records_by_cond[cond][i]["correct"]) for i in ids]),
                "by_source": {
                    src: mean([float(records_by_cond[cond][i]["correct"])
                               for i in ids
                               if records_by_cond[cond][i]["source"] == src])
                    for src in ("cnn_dailymail", "synthetic", "handwritten")},
                "by_distance": {
                    str(d): mean([float(records_by_cond[cond][i]["correct"])
                                  for i in ids
                                  if records_by_cond[cond][i]["distance"] == d])
                    for d in sorted({records_by_cond[cond][i]["distance"]
                                     for i in ids})},
                "gating": cond in GATING,
            } for cond in complete},
        "verdict": verdict_block(records_by_cond, ids),
        "protocol": {
            "checkpoint": "results/checkpoints/exp1b-bottleneck-v2/best",
            "cohort_file": args.cohort, "cohort_sha256": cohort.dataset_sha256,
            "seed": 42, "records_file": args.records,
            "metric": "MCQ (gating); fact retrieval deferred",
        },
    }
    out = Path(args.out)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    v = results["verdict"]
    print(json.dumps({
        "primary_win": v["primary_bottleneck_beats_baseline"],
        "tie_within_3pts": v["secondary_tie_within_3pts"],
        "paired_diff": v["paired_diff_bottleneck_minus_baseline"]["mean_diff"],
        "mcnemar_p": v["mcnemar_bottleneck_vs_baseline"]["p_value"],
        "out_of_style_diff": v["out_of_style_cnn"]["paired_diff"]["mean_diff"],
    }, indent=2))
    print(f"EXP2 FINALIZED (early stop) -> {out}")


if __name__ == "__main__":
    main()
