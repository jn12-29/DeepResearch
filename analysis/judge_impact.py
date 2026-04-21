#!/usr/bin/env python3
"""
Analyse the effect of different judge-model scoring passes on evaluation metrics.

Groups summary.jsonl records by (model, dataset, inference-files fingerprint).
Groups with >1 record represent the same inference output scored by different judges.
Outputs: analysis/judge_impact.md

Usage:
    python judge_impact.py [--input path/to/summary.jsonl]
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT_INPUT = HERE.parent / "summary.jsonl"
DEFAULT_OUTPUT = HERE / "judge_impact.md"

DATASET_LABEL = {
    "xbench-deepsearch": "DeepSearch-2510",
    "gaia": "GAIA 2023",
    "hle": "HLE-200",
}
MODEL_LABEL = {
    "Tongyi-DeepResearch-30B-A3B_sglang": "30B-FP16",
    "Tongyi-DeepResearch-30B-A3B-Int4-W4A16_sglang": "30B-Int4-W4A16",
}

# Metrics affected by judge scoring (vary across runs on same inference)
JUDGE_METRICS: list[tuple[str, str, str]] = [
    ("overall",    "avg_pass_at_3",   "Avg Pass@3"),
    ("overall",    "best_pass_at_1",  "Best Pass@1"),
    ("overall",    "pass_at_3",       "Pass@3"),
    ("individual", "Round1_Pass@1",   "Round 1 Pass@1"),
    ("individual", "Round2_Pass@1",   "Round 2 Pass@1"),
    ("individual", "Round3_Pass@1",   "Round 3 Pass@1"),
    ("statistics", "avg_tool_calls_per_question_correctly_solved",      "Actions/Q (correct)"),
    ("statistics", "avg_assistant_tokens_per_question_correctly_solved","Tokens/Q (correct)"),
]


def parse_model(files: dict) -> str:
    path = next(iter(files.values()), "")
    try:
        model_dir = path.split("/inference/outputs/")[1].split("/")[0]
    except IndexError:
        return path
    return MODEL_LABEL.get(model_dir, model_dir)


def files_key(files: dict) -> tuple:
    return tuple(sorted(files.values()))


def load_records(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_metric(rec: dict, section: str, key: str) -> float | None:
    return rec.get(section, {}).get(key)


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals)


def std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def fmt(v: float | None, decimals: int = 2) -> str:
    return f"{v:.{decimals}f}" if v is not None else "—"


def md_table(headers: list[str], rows: list[list]) -> str:
    widths = [len(h) for h in headers]
    str_rows = [[str(c) for c in row] for row in rows]
    for row in str_rows:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(c))

    def fmt_row(cells):
        return "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    return "\n".join([fmt_row(headers), sep] + [fmt_row(r) for r in str_rows])


def build_md(records: list[dict]) -> str:
    # Group by (model, dataset, files_key)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for rec in records:
        model = parse_model(rec.get("files", {}))
        dataset = DATASET_LABEL.get(rec.get("dataset", ""), rec.get("dataset", ""))
        fk = files_key(rec.get("files", {}))
        groups[(model, dataset, fk)].append(rec)

    # Only keep groups with >1 scoring run
    multi = {k: v for k, v in groups.items() if len(v) > 1}

    sections = ["# Judge Model Impact Analysis\n"]
    sections.append(
        "Groups with identical inference outputs scored by multiple judges. "
        "Statistics-only metrics (tool usage, termination) are inference-derived and "
        "therefore identical across runs — only scoring-dependent metrics are shown.\n"
    )

    if not multi:
        sections.append("_No groups with multiple scoring runs found._\n")
        return "\n".join(sections)

    # ── Summary table across all groups ──────────────────────────────────────
    sections.append("## Summary: Score Spread per (Model, Dataset)\n")
    summary_headers = ["Model", "Dataset", "#Runs", "Metric", "Min", "Max", "Range", "Mean", "Std"]
    summary_rows = []

    for (model, dataset, _), recs in sorted(multi.items()):
        n = len(recs)
        first_group = True
        for section, key, label in JUDGE_METRICS:
            vals = [v for v in (get_metric(r, section, key) for r in recs) if v is not None]
            if not vals:
                continue
            mn, mx = min(vals), max(vals)
            rng = mx - mn
            mu = mean(vals)
            sd = std(vals)
            decimals = 0 if "Tokens" in label else 2
            row = [
                model if first_group else "",
                dataset if first_group else "",
                str(n) if first_group else "",
                label,
                fmt(mn, decimals),
                fmt(mx, decimals),
                fmt(rng, decimals),
                fmt(mu, decimals),
                fmt(sd, decimals),
            ]
            summary_rows.append(row)
            first_group = False

    sections.append(md_table(summary_headers, summary_rows))

    # ── Per-group detail tables ───────────────────────────────────────────────
    sections.append("\n\n## Per-Group Detail\n")

    for (model, dataset, _), recs in sorted(multi.items()):
        n = len(recs)
        sections.append(f"### {model} / {dataset} ({n} scoring runs)\n")

        run_labels = [f"Run {i+1}" for i in range(n)]
        headers = ["Metric"] + run_labels + ["Mean", "Std", "Range"]
        rows = []

        for section, key, label in JUDGE_METRICS:
            vals = [get_metric(r, section, key) for r in recs]
            nums = [v for v in vals if v is not None]
            if not nums:
                continue
            decimals = 0 if "Tokens" in label else 2
            mu = mean(nums)
            sd = std(nums)
            rng = max(nums) - min(nums)
            row = (
                [label]
                + [fmt(v, decimals) for v in vals]
                + [fmt(mu, decimals), fmt(sd, decimals), fmt(rng, decimals)]
            )
            rows.append(row)

        sections.append(md_table(headers, rows))
        sections.append("\n")

    return "\n".join(sections)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()

    records = load_records(args.input)
    print(f"Loaded {len(records)} records from {args.input}")

    md = build_md(records)
    DEFAULT_OUTPUT.write_text(md)
    print(f"Written to {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
