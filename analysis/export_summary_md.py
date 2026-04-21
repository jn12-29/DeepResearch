#!/usr/bin/env python3
"""
Export /home/xh/DeepResearch/summary.jsonl to a Markdown report.

Output: analysis/summary_results.md (same directory as this script)

Usage:
    python export_summary_md.py [--input path/to/summary.jsonl]
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT_INPUT = HERE.parent / "summary.jsonl"
DEFAULT_OUTPUT = HERE / "summary_results.md"

DATASET_LABEL = {
    "xbench-deepsearch": "DeepSearch-2510",
    "gaia": "GAIA 2023",
    "hle": "HLE-200",
}

MODEL_LABEL = {
    "Tongyi-DeepResearch-30B-A3B_sglang": "30B-FP16",
    "Tongyi-DeepResearch-30B-A3B-Int4-W4A16_sglang": "30B-Int4-W4A16",
}


def parse_model(files: dict) -> str:
    path = next(iter(files.values()), "")
    try:
        model_dir = path.split("/inference/outputs/")[1].split("/")[0]
    except IndexError:
        return path
    return MODEL_LABEL.get(model_dir, model_dir)


def load_records(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def fmt_pct(v, decimals=2) -> str:
    return f"{v:.{decimals}f}%" if v is not None else "—"


def fmt_num(v, decimals=1) -> str:
    return f"{v:.{decimals}f}" if v is not None else "—"


def md_table(headers: list, rows: list) -> str:
    widths = [len(str(h)) for h in headers]
    str_rows = [[str(c) for c in row] for row in rows]
    for row in str_rows:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(c))

    def fmt_row(cells):
        return "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    return "\n".join([fmt_row(headers), sep] + [fmt_row(r) for r in str_rows])


def smart_dedup(data_headers: list[str], raw_rows: list[tuple]) -> tuple[list[str], list[list]]:
    """
    raw_rows: [(model, dataset, run, val, val, ...)]

    Per (model, dataset) group:
    - If all runs share identical data values → collapse to one row, run=None
    - If any run differs → keep all rows with their run index

    Returns (final_headers, final_rows).
    Run column is included only when at least one row has a run value.
    """
    from collections import defaultdict
    groups: dict[tuple, list] = defaultdict(list)
    for row in raw_rows:
        key = (row[0], row[1])
        groups[key].append(row)

    final_rows = []
    for key in sorted(groups):
        group = groups[key]
        vals = [r[3:] for r in group]
        if len(set(vals)) == 1:
            # All identical — emit single row, run=None
            model, dataset = key
            final_rows.append([model, dataset, None] + list(vals[0]))
        else:
            for r in group:
                final_rows.append(list(r))

    need_run = any(r[2] is not None for r in final_rows)
    if need_run:
        headers = ["Model", "Dataset", "Run"] + data_headers
        out_rows = [
            [r[0], r[1], r[2] if r[2] is not None else "—"] + r[3:]
            for r in final_rows
        ]
    else:
        headers = ["Model", "Dataset"] + data_headers
        out_rows = [[r[0], r[1]] + r[3:] for r in final_rows]

    return headers, out_rows


def build_md(records: list[dict]) -> str:
    # Assign run index per (model, dataset) group
    run_counter: dict[tuple, int] = {}
    annotated = []
    for rec in records:
        model = parse_model(rec.get("files", {}))
        dataset = DATASET_LABEL.get(rec.get("dataset", ""), rec.get("dataset", ""))
        key = (model, dataset)
        run_counter[key] = run_counter.get(key, 0) + 1
        annotated.append((model, dataset, run_counter[key], rec))

    annotated.sort(key=lambda x: (x[0], x[1], x[2]))

    sections = []

    # ── 1. Overall Performance ────────────────────────────────────────────────
    sections.append("## Overall Performance\n")
    raw = []
    for model, dataset, run, rec in annotated:
        ov = rec.get("overall", {})
        raw.append((model, dataset, run,
            fmt_pct(ov.get("avg_pass_at_3")),
            fmt_pct(ov.get("best_pass_at_1")),
            fmt_pct(ov.get("pass_at_3")),
        ))
    headers, rows = smart_dedup(["Avg Pass@3", "Best Pass@1", "Pass@3"], raw)
    sections.append(md_table(headers, rows))

    # ── 2. Per-Round Pass@1 ───────────────────────────────────────────────────
    sections.append("\n\n## Per-Round Pass@1\n")
    raw = []
    for model, dataset, run, rec in annotated:
        ind = rec.get("individual", {})
        raw.append((model, dataset, run,
            fmt_pct(ind.get("Round1_Pass@1")),
            fmt_pct(ind.get("Round2_Pass@1")),
            fmt_pct(ind.get("Round3_Pass@1")),
        ))
    headers, rows = smart_dedup(["Round 1", "Round 2", "Round 3"], raw)
    sections.append(md_table(headers, rows))

    # ── 3. Tool Usage Statistics ──────────────────────────────────────────────
    sections.append("\n\n## Tool Usage Statistics\n")
    raw = []
    for model, dataset, run, rec in annotated:
        st = rec.get("statistics", {})
        raw.append((model, dataset, run,
            fmt_num(st.get("avg_action")),
            fmt_num(st.get("avg_search_action")),
            fmt_num(st.get("avg_visit_action")),
            fmt_num(st.get("avg_other_action")),
            fmt_num(st.get("avg_ans_length"), decimals=0),
            fmt_num(st.get("avg_think_length"), decimals=0),
        ))
    headers, rows = smart_dedup(
        ["Avg Actions", "Search", "Visit", "Other", "Ans Tokens", "Think Tokens"], raw
    )
    sections.append(md_table(headers, rows))

    # ── 4. Token Efficiency ───────────────────────────────────────────────────
    sections.append("\n\n## Token Efficiency\n")
    raw = []
    for model, dataset, run, rec in annotated:
        st = rec.get("statistics", {})
        raw.append((model, dataset, run,
            fmt_num(st.get("avg_assistant_tokens_per_question"), decimals=0),
            fmt_num(st.get("avg_assistant_tokens_per_question_correctly_solved"), decimals=0),
            fmt_num(st.get("avg_assistant_tokens_per_message"), decimals=0),
            fmt_num(st.get("avg_tool_calls_per_question_correctly_solved")),
            fmt_num(st.get("num_invalid"), decimals=0),
        ))
    headers, rows = smart_dedup(
        ["Tokens/Q", "Tokens/Q (correct)", "Tokens/Msg", "Actions/Q (correct)", "Invalid"], raw
    )
    sections.append(md_table(headers, rows))

    # ── 5. Termination Reasons ────────────────────────────────────────────────
    sections.append("\n\n## Termination Reasons\n")
    all_reasons = sorted({
        r
        for _, _, _, rec in annotated
        for r in rec.get("statistics", {}).get("termination_freq", {})
    })
    raw = []
    for model, dataset, run, rec in annotated:
        tf = rec.get("statistics", {}).get("termination_freq", {})
        raw.append((model, dataset, run,
            *[fmt_pct(tf.get(r)) for r in all_reasons]
        ))
    headers, rows = smart_dedup(all_reasons, raw)
    sections.append(md_table(headers, rows))

    return "# Evaluation Results Summary\n\n" + "\n".join(sections) + "\n"


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
