#!/usr/bin/env python3
"""
Micro-level judge-model impact analysis.

For each (model, dataset) group, finds all scoring passes (base dir + timestamped
re-score siblings), matches questions across passes, and reports:
  1. Per-iter agreement stats: how often judges agree / disagree
  2. Controversial questions: cases where different judges give different verdicts

Supported file formats:
  - iter*_scored.jsonl  (DeepSearch, GAIA): fields is_correct, judgement, question, prediction
  - iter*.eval_details.jsonl (HLE):         fields acc, item.{question,answer,prediction}

Output: analysis/judge_impact_micro.md

Usage:
    python judge_impact_micro.py [--outputs-dir path/to/inference/outputs]
"""

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT_OUTPUTS = HERE.parent / "inference" / "outputs"
DEFAULT_OUTPUT = HERE / "judge_impact_micro.md"

MODEL_LABEL = {
    "Tongyi-DeepResearch-30B-A3B_sglang": "30B-FP16",
    "Tongyi-DeepResearch-30B-A3B-Int4-W4A16_sglang": "30B-Int4-W4A16",
}
DATASET_LABEL = {
    "DeepSearch-2510": "DeepSearch-2510",
    "gaia_2023_validation": "GAIA 2023",
    "hle_text_200": "HLE-200",
}
ITERS = ["iter1", "iter2", "iter3"]
TS_PATTERN = re.compile(r"^(.+)_(\d{8}_\d{6})$")


# ── File loading ──────────────────────────────────────────────────────────────

def load_scored(path: Path) -> list[dict]:
    """Load iter*_scored.jsonl → list of {question, prediction, is_correct, judgement, orig_idx}"""
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
                "judgement":  str(r.get("judgement", "")),
                "orig_idx":   idx,
            })
    return records


def load_eval_details(path: Path) -> list[dict]:
    """Load iter*.eval_details.jsonl → list of {question, prediction, is_correct, judgement, orig_idx}"""
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
                "judgement":  "",  # no judgement text in eval_details
                "orig_idx":   idx,
            })
    return records


def load_iter(scoring_dir: Path, iter_name: str) -> list[dict] | None:
    """Try scored then eval_details format; return None if neither exists."""
    scored = scoring_dir / f"{iter_name}_scored.jsonl"
    if scored.exists():
        return load_scored(scored)
    details = scoring_dir / f"{iter_name}.eval_details.jsonl"
    if details.exists():
        return load_eval_details(details)
    return None


# ── Directory discovery ───────────────────────────────────────────────────────

def discover_groups(outputs_dir: Path) -> list[dict]:
    """
    Returns list of:
      {model, model_label, dataset, dataset_label, base_dir, scoring_dirs: [Path, ...]}
    scoring_dirs[0] = base_dir, rest are timestamped siblings sorted by timestamp.
    Only returns groups with ≥2 scoring passes.
    """
    groups = []
    for model_dir in sorted(outputs_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_label = MODEL_LABEL.get(model_dir.name, model_dir.name)

        # Collect all dataset subdirs
        by_base: dict[str, dict] = {}
        for sub in sorted(model_dir.iterdir()):
            if not sub.is_dir():
                continue
            m = TS_PATTERN.match(sub.name)
            if m:
                base_name, ts = m.group(1), m.group(2)
                by_base.setdefault(base_name, {"base": None, "siblings": []})
                by_base[base_name]["siblings"].append((ts, sub))
            else:
                by_base.setdefault(sub.name, {"base": None, "siblings": []})
                by_base[sub.name]["base"] = sub

        for dataset_name, info in by_base.items():
            base = info["base"]
            siblings = sorted(info["siblings"], key=lambda x: x[0])
            if base is None or not siblings:
                continue
            scoring_dirs = [base] + [s for _, s in siblings]
            groups.append({
                "model":         model_dir.name,
                "model_label":   model_label,
                "dataset":       dataset_name,
                "dataset_label": DATASET_LABEL.get(dataset_name, dataset_name),
                "base_dir":      base,
                "scoring_dirs":  scoring_dirs,
            })
    return groups


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyse_iter(scoring_dirs: list[Path], iter_name: str) -> dict | None:
    """
    Returns None if any scoring dir is missing the iter file.
    Otherwise returns:
      {
        n_passes, n_questions,
        n_all_correct, n_all_wrong, n_controversial,
        pass_rates: [float, ...],   # per scoring pass
        controversies: [{question, answer, prediction, verdicts:[bool,...], judgements:[str,...]}]
      }
    """
    passes = []
    for d in scoring_dirs:
        data = load_iter(d, iter_name)
        if data is None:
            return None
        passes.append(data)

    n_passes = len(passes)
    # Align by question — use index if counts match, else match by question text
    counts = [len(p) for p in passes]
    if len(set(counts)) == 1:
        # Same length: zip by position (faster, handles edge cases)
        aligned = list(zip(*passes))
    else:
        # Different lengths: match by question text using first pass as reference
        q_idx = [{r["question"]: r for r in p} for p in passes]
        ref_questions = [r["question"] for r in passes[0]]
        aligned = []
        for q in ref_questions:
            row = tuple(idx.get(q) for idx in q_idx)
            if all(r is not None for r in row):
                aligned.append(row)

    n_questions = len(aligned)
    n_all_correct = 0
    n_all_wrong = 0
    controversies = []

    for row in aligned:
        verdicts = [r["is_correct"] for r in row]
        if all(verdicts):
            n_all_correct += 1
        elif not any(verdicts):
            n_all_wrong += 1
        else:
            controversies.append({
                "orig_idx":    row[0]["orig_idx"],
                "question":    row[0]["question"],
                "answer":      row[0]["answer"],
                "prediction":  row[0]["prediction"],
                "predictions": [r["prediction"] for r in row],
                "verdicts":    verdicts,
                "judgements":  [r["judgement"] for r in row],
            })

    n_controversial = len(controversies)
    pass_rates = [
        sum(r["is_correct"] for r in p) / len(p) * 100 if p else 0.0
        for p in passes
    ]

    return {
        "n_passes":       n_passes,
        "n_questions":    n_questions,
        "n_all_correct":  n_all_correct,
        "n_all_wrong":    n_all_wrong,
        "n_controversial": n_controversial,
        "pass_rates":     pass_rates,
        "controversies":  controversies,
    }


# ── Markdown rendering ────────────────────────────────────────────────────────

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


def trunc(s: str, n: int = 80) -> str:
    s = s.replace("\n", " ").strip()
    return s[:n] + "…" if len(s) > n else s


def head_tail(s: str, head: int, tail: int) -> str:
    s = s.replace("\n", " ").strip()
    if len(s) <= head + tail:
        return s
    return s[:head] + "…" + s[-tail:]


def build_md(groups: list[dict]) -> str:
    sections = ["# Judge Model Impact — Micro Analysis\n"]
    sections.append(
        "Per-question comparison across scoring passes on identical inference outputs. "
        "\"Controversial\" = judges disagree on correctness for the same question.\n"
    )

    # ── 1. Aggregate stats table ──────────────────────────────────────────────
    sections.append("## Aggregate Agreement Statistics\n")
    agg_headers = [
        "Model", "Dataset", "Iter", "#Passes", "#Q",
        "All✓ %", "All✗ %", "Controversial %",
        *[f"Judge{i+1} Pass@1" for i in range(3)],
    ]
    agg_rows = []

    all_group_results = []  # for detail sections

    for g in groups:
        group_iters = []
        for iter_name in ITERS:
            res = analyse_iter(g["scoring_dirs"], iter_name)
            if res is None:
                continue
            group_iters.append((iter_name, res))

            n = res["n_questions"]
            rate_cells = [f"{r:.1f}%" for r in res["pass_rates"]]
            # Pad to 3 judges
            rate_cells += ["—"] * (3 - len(rate_cells))

            agg_rows.append([
                g["model_label"],
                g["dataset_label"],
                iter_name,
                res["n_passes"],
                n,
                f"{res['n_all_correct']/n*100:.1f}%",
                f"{res['n_all_wrong']/n*100:.1f}%",
                f"{res['n_controversial']/n*100:.1f}%",
                *rate_cells,
            ])
        all_group_results.append((g, group_iters))

    sections.append(md_table(agg_headers, agg_rows))

    # ── 2. Per-group controversy detail ──────────────────────────────────────
    sections.append("\n\n## Controversial Questions (Judges Disagree)\n")

    for g, group_iters in all_group_results:
        for iter_name, res in group_iters:
            controvs = res["controversies"]
            if not controvs:
                continue
            n_passes = res["n_passes"]
            judge_labels = [f"J{i+1}" for i in range(n_passes)]
            sections.append(
                f"### {g['model_label']} / {g['dataset_label']} / {iter_name} "
                f"({len(controvs)} controversial / {res['n_questions']} questions)\n"
            )
            judge_cols = []
            for lbl in judge_labels:
                judge_cols += [lbl, f"{lbl} Output"]
            headers = ["Q#", "#", "Question", "Answer", "Model Prediction"] + judge_cols
            rows = []
            for idx, c in enumerate(controvs, 1):
                judge_cells = []
                for v, j in zip(c["verdicts"], c["judgements"]):
                    judge_cells.append("✓" if v else "✗")
                    judge_cells.append(trunc(j, 120) if j else "—")
                rows.append([
                    c["orig_idx"],
                    idx,
                    trunc(c["question"], 60),
                    trunc(c["answer"], 40),
                    head_tail(c["prediction"], 20, 40),
                    *judge_cells,
                ])
            sections.append(md_table(headers, rows))
            sections.append("\n")

    return "\n".join(sections)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS)
    args = parser.parse_args()

    groups = discover_groups(args.outputs_dir)
    print(f"Discovered {len(groups)} (model, dataset) groups with ≥2 scoring passes:")
    for g in groups:
        print(f"  {g['model_label']} / {g['dataset_label']}: {len(g['scoring_dirs'])} passes")

    md = build_md(groups)
    DEFAULT_OUTPUT.write_text(md)
    print(f"\nWritten to {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
