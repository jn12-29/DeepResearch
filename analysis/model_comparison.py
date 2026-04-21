#!/usr/bin/env python3
"""
Cross-model comparison: 30B-FP16 vs 30B-Int4-W4A16.

For each dataset, compares:
  1. Macro: aggregate metrics from summary.jsonl (means across judge runs)
  2. Micro: per-question correctness overlap — unique wins, shared correct, shared wrong

Output: analysis/model_comparison.md

Usage:
    python model_comparison.py [--outputs-dir path] [--summary path]
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT_OUTPUTS = HERE.parent / "inference" / "outputs"
DEFAULT_SUMMARY = HERE.parent / "summary.jsonl"
DEFAULT_OUTPUT      = HERE / "model_comparison.md"
DEFAULT_JSON_DIR    = HERE / "model_comparison_micro"

MODEL_DIRS = {
    "FP16": "Tongyi-DeepResearch-30B-A3B_sglang",
    "Int4":  "Tongyi-DeepResearch-30B-A3B-Int4-W4A16_sglang",
}
MODEL_LABEL = {v: k for k, v in MODEL_DIRS.items()}
MODEL_LABEL.update({
    "Tongyi-DeepResearch-30B-A3B_sglang": "30B-FP16",
    "Tongyi-DeepResearch-30B-A3B-Int4-W4A16_sglang": "30B-Int4",
})

DATASET_LABEL = {
    "xbench-deepsearch": "DeepSearch-2510",
    "gaia":              "GAIA 2023",
    "hle":               "HLE-200",
}
DATASET_DIR = {
    "xbench-deepsearch": "DeepSearch-2510",
    "gaia":              "gaia_2023_validation",
    "hle":               "hle_text_200",
}

ITERS = ["iter1", "iter2", "iter3"]
TS_PATTERN = re.compile(r"^(.+)_(\d{8}_\d{6})$")

MACRO_METRICS = [
    ("overall",    "avg_pass_at_3",   "Avg Pass@3",          2),
    ("overall",    "best_pass_at_1",  "Best Pass@1",         2),
    ("overall",    "pass_at_3",       "Pass@3",              2),
    ("individual", "Round1_Pass@1",   "Round 1 Pass@1",      2),
    ("individual", "Round2_Pass@1",   "Round 2 Pass@1",      2),
    ("individual", "Round3_Pass@1",   "Round 3 Pass@1",      2),
    ("statistics", "avg_action",                              "Avg Actions/Q",       1),
    ("statistics", "avg_search_action",                       "Search Actions/Q",    1),
    ("statistics", "avg_visit_action",                        "Visit Actions/Q",     1),
    ("statistics", "avg_assistant_tokens_per_question",       "Tokens/Q",            0),
    ("statistics", "avg_tool_calls_per_question_correctly_solved", "Actions/Q (correct)", 1),
    ("statistics", "avg_assistant_tokens_per_question_correctly_solved", "Tokens/Q (correct)", 0),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def mean(vals):
    return sum(vals) / len(vals) if vals else None

def std(vals):
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))

def fmt(v, decimals=2, suffix=""):
    if v is None:
        return "—"
    return f"{v:.{decimals}f}{suffix}"

def delta_str(fp16, int4, decimals=2):
    if fp16 is None or int4 is None:
        return "—"
    d = fp16 - int4
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.{decimals}f}"

def trunc(s: str, n: int = 60) -> str:
    s = str(s).replace("\n", " ").strip()
    return s[:n] + "…" if len(s) > n else s

def md_table(headers, rows):
    widths = [len(str(h)) for h in headers]
    str_rows = [[str(c) for c in row] for row in rows]
    for row in str_rows:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(c))

    def fmt_row(cells):
        return "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    return "\n".join([fmt_row(headers), sep] + [fmt_row(r) for r in str_rows])


# ── Macro analysis (summary.jsonl) ───────────────────────────────────────────

def parse_model_from_path(path: str) -> str:
    try:
        return path.split("/inference/outputs/")[1].split("/")[0]
    except IndexError:
        return path

def load_summary(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def build_macro(records: list[dict]) -> dict:
    """Returns {dataset_key: {model_dir: [record, ...]}}"""
    grouped: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        files = rec.get("files", {})
        path = next(iter(files.values()), "")
        model_dir = parse_model_from_path(path)
        dataset = rec.get("dataset", "")
        grouped[dataset][model_dir].append(rec)
    return grouped


# ── Micro analysis (scored files) ────────────────────────────────────────────

def load_raw(path: Path) -> dict[str, dict]:
    """Load iter*.jsonl → {question_text: {messages, termination}}"""
    result = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            q = r.get("question", "")
            result[q] = {
                "messages":    r.get("messages", []),
                "termination": r.get("termination", ""),
            }
    return result


def load_scored(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            records.append({
                "question":   r.get("question", ""),
                "answer":     r.get("answer", ""),
                "prediction": r.get("prediction", ""),
                "is_correct": bool(r.get("is_correct", False)),
                "orig_idx":   idx,
            })
    return records

def load_eval_details(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            item = r.get("item", {})
            records.append({
                "question":   item.get("question", ""),
                "answer":     item.get("answer", ""),
                "prediction": item.get("prediction", ""),
                "is_correct": bool(r.get("acc", 0)),
                "orig_idx":   idx,
            })
    return records

def load_iter(scoring_dir: Path, iter_name: str):
    scored = scoring_dir / f"{iter_name}_scored.jsonl"
    if scored.exists():
        return load_scored(scored)
    details = scoring_dir / f"{iter_name}.eval_details.jsonl"
    if details.exists():
        return load_eval_details(details)
    return None

def find_base_dir(outputs_dir: Path, model_dir_name: str, dataset_dir_name: str) -> Path | None:
    p = outputs_dir / model_dir_name / dataset_dir_name
    return p if p.is_dir() else None

def align_two(recs_a: list[dict], recs_b: list[dict]) -> list[tuple[dict, dict]]:
    """Align two record lists by position (if same length) or question text."""
    if len(recs_a) == len(recs_b):
        return list(zip(recs_a, recs_b))
    idx_b = {r["question"]: r for r in recs_b}
    aligned = []
    for ra in recs_a:
        rb = idx_b.get(ra["question"])
        if rb is not None:
            aligned.append((ra, rb))
    return aligned

def analyse_micro_iter(outputs_dir: Path, iter_name: str, dataset_key: str) -> dict | None:
    """
    Compare FP16 base dir vs Int4 base dir for one iter.
    Returns None if either is missing.
    """
    fp16_dir = find_base_dir(outputs_dir, MODEL_DIRS["FP16"], DATASET_DIR[dataset_key])
    int4_dir  = find_base_dir(outputs_dir, MODEL_DIRS["Int4"],  DATASET_DIR[dataset_key])
    if fp16_dir is None or int4_dir is None:
        return None

    fp16_recs = load_iter(fp16_dir, iter_name)
    int4_recs  = load_iter(int4_dir,  iter_name)
    if fp16_recs is None or int4_recs is None:
        return None

    # Load raw messages; raw file lives alongside scored file
    fp16_raw_path = fp16_dir / f"{iter_name}.jsonl"
    int4_raw_path  = int4_dir  / f"{iter_name}.jsonl"
    fp16_raw = load_raw(fp16_raw_path) if fp16_raw_path.exists() else {}
    int4_raw  = load_raw(int4_raw_path)  if int4_raw_path.exists()  else {}

    def enrich(rec, raw_map):
        raw = raw_map.get(rec["question"], {})
        rec["messages"]    = raw.get("messages", [])
        rec["termination"] = raw.get("termination", "")
        return rec

    fp16_recs = [enrich(r, fp16_raw) for r in fp16_recs]
    int4_recs  = [enrich(r, int4_raw)  for r in int4_recs]

    aligned = align_two(fp16_recs, int4_recs)
    n = len(aligned)
    if n == 0:
        return None

    both_correct = []
    both_wrong   = []
    fp16_only    = []   # FP16 correct, Int4 wrong
    int4_only    = []   # Int4 correct, FP16 wrong

    for ra, rb in aligned:
        if ra["is_correct"] and rb["is_correct"]:
            both_correct.append((ra, rb))
        elif not ra["is_correct"] and not rb["is_correct"]:
            both_wrong.append((ra, rb))
        elif ra["is_correct"] and not rb["is_correct"]:
            fp16_only.append((ra, rb))
        else:
            int4_only.append((ra, rb))

    def pair_record(ra, rb, verdict):
        return {
            "orig_idx":           ra["orig_idx"],
            "verdict":            verdict,
            "question":           ra["question"],
            "answer":             ra["answer"],
            "fp16_correct":       ra["is_correct"],
            "int4_correct":       rb["is_correct"],
            "fp16_prediction":    ra["prediction"],
            "int4_prediction":    rb["prediction"],
            "fp16_termination":   ra.get("termination", ""),
            "int4_termination":   rb.get("termination", ""),
            "fp16_messages":      ra.get("messages", []),
            "int4_messages":      rb.get("messages", []),
        }

    questions = (
        [pair_record(ra, rb, "both_correct") for ra, rb in both_correct] +
        [pair_record(ra, rb, "both_wrong")   for ra, rb in both_wrong]   +
        [pair_record(ra, rb, "fp16_only")    for ra, rb in fp16_only]    +
        [pair_record(ra, rb, "int4_only")    for ra, rb in int4_only]
    )
    questions.sort(key=lambda r: r["orig_idx"])

    return {
        "n":            n,
        "both_correct": len(both_correct),
        "both_wrong":   len(both_wrong),
        "fp16_only":    len(fp16_only),
        "int4_only":    len(int4_only),
        "fp16_only_qs": fp16_only,
        "int4_only_qs": int4_only,
        "fp16_pass":    (len(both_correct) + len(fp16_only)) / n * 100,
        "int4_pass":    (len(both_correct) + len(int4_only)) / n * 100,
        "questions":    questions,
    }


# ── Markdown builder ──────────────────────────────────────────────────────────

def build_md(records: list[dict], outputs_dir: Path) -> str:
    macro = build_macro(records)
    sections = ["# Cross-Model Comparison: 30B-FP16 vs 30B-Int4\n"]
    sections.append("Each scoring run is listed as a separate column. Δ = mean(FP16) − mean(Int4).\n")

    # ── 1. Macro comparison ───────────────────────────────────────────────────
    sections.append("## Macro Comparison (Aggregate Metrics)\n")

    for dataset_key in ["xbench-deepsearch", "gaia", "hle"]:
        dataset_label = DATASET_LABEL[dataset_key]
        model_recs = macro.get(dataset_key, {})

        fp16_recs_list = model_recs.get(MODEL_DIRS["FP16"], [])
        int4_recs_list  = model_recs.get(MODEL_DIRS["Int4"],  [])

        n_fp16 = len(fp16_recs_list)
        n_int4 = len(int4_recs_list)

        sections.append(f"### {dataset_label}\n")

        fp16_cols = [f"FP16-R{i+1}" for i in range(n_fp16)]
        int4_cols = [f"Int4-R{i+1}" for i in range(n_int4)]
        headers = ["Metric"] + fp16_cols + int4_cols + ["Δ (abs)", "Δ (%)"]
        rows = []

        for section, key, label, decimals in MACRO_METRICS:
            fp16_vals = [r.get(section, {}).get(key) for r in fp16_recs_list]
            int4_vals  = [r.get(section, {}).get(key) for r in int4_recs_list]

            fp16_nums = [v for v in fp16_vals if v is not None]
            int4_nums  = [v for v in int4_vals  if v is not None]
            fp16_mean = mean(fp16_nums)
            int4_mean = mean(int4_nums)

            if fp16_mean is not None and int4_mean is not None and int4_mean != 0:
                rel = (fp16_mean - int4_mean) / abs(int4_mean) * 100
                rel_str = ("+" if rel >= 0 else "") + f"{rel:.1f}%"
            else:
                rel_str = "—"

            rows.append([
                label,
                *[fmt(v, decimals) for v in fp16_vals],
                *[fmt(v, decimals) for v in int4_vals],
                delta_str(fp16_mean, int4_mean, decimals),
                rel_str,
            ])

        sections.append(md_table(headers, rows))
        sections.append("\n")

    # ── 2. Micro comparison ───────────────────────────────────────────────────
    sections.append("\n## Micro Comparison (Per-Question Overlap)\n")
    sections.append(
        "Questions are aligned by position (same-length files) or question text. "
        "**FP16-only** = FP16 correct, Int4 wrong. **Int4-only** = Int4 correct, FP16 wrong.\n"
    )

    # Summary table
    micro_headers = [
        "Dataset", "Iter", "#Q aligned",
        "Both ✓", "Both ✗", "FP16-only", "Int4-only",
        "FP16 Pass@1", "Int4 Pass@1",
    ]
    micro_rows = []
    all_micro = {}  # (dataset_key, iter_name) -> result

    for dataset_key in ["xbench-deepsearch", "gaia", "hle"]:
        dataset_label = DATASET_LABEL[dataset_key]
        first = True
        for iter_name in ITERS:
            res = analyse_micro_iter(outputs_dir, iter_name, dataset_key)
            if res is None:
                continue
            all_micro[(dataset_key, iter_name)] = res
            n = res["n"]
            micro_rows.append([
                dataset_label if first else "",
                iter_name,
                n,
                f"{res['both_correct']} ({res['both_correct']/n*100:.1f}%)",
                f"{res['both_wrong']}   ({res['both_wrong']/n*100:.1f}%)",
                f"{res['fp16_only']}  ({res['fp16_only']/n*100:.1f}%)",
                f"{res['int4_only']}  ({res['int4_only']/n*100:.1f}%)",
                f"{res['fp16_pass']:.1f}%",
                f"{res['int4_pass']:.1f}%",
            ])
            first = False

    sections.append(md_table(micro_headers, micro_rows))

    # ── 3. Unique-win question detail ─────────────────────────────────────────
    sections.append("\n\n## Unique-Win Questions (Where Models Diverge)\n")
    sections.append(
        "Questions where only one model is correct. "
        "Q# = line number in the base scoring file.\n"
    )

    for dataset_key in ["xbench-deepsearch", "gaia", "hle"]:
        dataset_label = DATASET_LABEL[dataset_key]
        for iter_name in ITERS:
            res = all_micro.get((dataset_key, iter_name))
            if res is None:
                continue
            fp16_qs = res["fp16_only_qs"]
            int4_qs  = res["int4_only_qs"]
            if not fp16_qs and not int4_qs:
                continue

            sections.append(f"### {dataset_label} / {iter_name}\n")

            if fp16_qs:
                sections.append(f"**FP16-only correct** ({len(fp16_qs)} questions — Int4 missed these)\n")
                headers = ["Q#", "#", "Question", "Answer", "FP16 Prediction", "Int4 Prediction"]
                rows = []
                for i, (ra, rb) in enumerate(fp16_qs, 1):
                    rows.append([
                        ra["orig_idx"], i,
                        trunc(ra["question"], 70),
                        trunc(ra["answer"], 40),
                        trunc(ra["prediction"], 40),
                        trunc(rb["prediction"], 40),
                    ])
                sections.append(md_table(headers, rows))
                sections.append("\n")

            if int4_qs:
                sections.append(f"**Int4-only correct** ({len(int4_qs)} questions — FP16 missed these)\n")
                headers = ["Q#", "#", "Question", "Answer", "FP16 Prediction", "Int4 Prediction"]
                rows = []
                for i, (ra, rb) in enumerate(int4_qs, 1):
                    rows.append([
                        rb["orig_idx"], i,
                        trunc(rb["question"], 70),
                        trunc(rb["answer"], 40),
                        trunc(ra["prediction"], 40),
                        trunc(rb["prediction"], 40),
                    ])
                sections.append(md_table(headers, rows))
                sections.append("\n")

    return "\n".join(sections), all_micro


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--summary",     type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    records = load_summary(args.summary)
    print(f"Loaded {len(records)} summary records")

    md, all_micro = build_md(records, args.outputs_dir)
    DEFAULT_OUTPUT.write_text(md)
    print(f"Written to {DEFAULT_OUTPUT}")

    DEFAULT_JSON_DIR.mkdir(exist_ok=True)
    VERDICT_FILES = [
        ("both_correct", "both_correct.json"),
        ("both_wrong",   "both_wrong.json"),
        ("fp16_only",    "fp16_only.json"),
        ("int4_only",    "int4_only.json"),
    ]
    for (dataset_key, iter_name), res in sorted(all_micro.items(), key=lambda x: x[0]):
        label = DATASET_LABEL[dataset_key].replace(" ", "_")
        group_dir = DEFAULT_JSON_DIR / f"{label}_{iter_name}"
        group_dir.mkdir(exist_ok=True)

        by_verdict = {v: [] for v, _ in VERDICT_FILES}
        for q in res["questions"]:
            by_verdict[q["verdict"]].append(q)

        meta = {
            "dataset":      DATASET_LABEL[dataset_key],
            "iter":         iter_name,
            "n":            res["n"],
            "both_correct": res["both_correct"],
            "both_wrong":   res["both_wrong"],
            "fp16_only":    res["fp16_only"],
            "int4_only":    res["int4_only"],
            "fp16_pass":    round(res["fp16_pass"], 2),
            "int4_pass":    round(res["int4_pass"], 2),
        }

        for verdict, fname in VERDICT_FILES:
            path = group_dir / fname
            payload = {**meta, "verdict": verdict, "questions": by_verdict[verdict]}
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

        print(f"  {group_dir.name}/ ({res['both_correct']} both✓, {res['both_wrong']} both✗, {res['fp16_only']} fp16-only, {res['int4_only']} int4-only)")


if __name__ == "__main__":
    main()
