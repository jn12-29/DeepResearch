#!/usr/bin/env python3
"""
HLE latency / timing analysis: FP16 vs Int4-W4A16 on test_hle.

Metrics per model (aggregated across iter1/iter2/iter3):
  - e2e latency (total, percentiles)
  - LLM decode time vs tool time breakdown
  - TTFT (time-to-first-token)
  - TPOT (time-per-output-token, ms)
  - Throughput: tokens/s
  - Per-round statistics
  - Tool usage timing breakdown by tool name

Output: analysis/hle_latency_analysis.md

Usage:
    python analysis/hle_latency_analysis.py [--outputs-dir path]
"""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT_OUTPUTS = HERE.parent / "inference" / "outputs"
DEFAULT_OUTPUT = HERE / "hle_latency_analysis.md"

MODELS = {
    "FP16": "Tongyi-DeepResearch-30B-A3B_sglang",
    "Int4": "Tongyi-DeepResearch-30B-A3B-Int4-W4A16_sglang",
}
HLE_DIR = "test_hle"
ITERS = ["iter1", "iter2", "iter3"]


# ── helpers ──────────────────────────────────────────────────────────────────

def pct(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    idx = (len(s) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def mean(values):
    return statistics.mean(values) if values else float("nan")


def fmt(v, decimals=2):
    if math.isnan(v):
        return "N/A"
    return f"{v:.{decimals}f}"


def load_records(outputs_dir: Path, model_dir: str) -> list[dict]:
    """Load all records from iter1/2/3.jsonl for a model's test_hle dir."""
    records = []
    base = outputs_dir / model_dir / HLE_DIR
    for it in ITERS:
        path = base / f"{it}.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def extract_latency_stats(records: list[dict]) -> dict:
    """Extract per-record latency fields and aggregate."""
    e2e, ttft, decode, tool, tpot, tokens, rounds_cnt = [], [], [], [], [], [], []
    tool_time_by_name: dict[str, list[float]] = defaultdict(list)
    tool_calls_by_name: dict[str, int] = defaultdict(int)
    per_round_decode, per_round_ttft, per_round_tokens = [], [], []
    decode_frac, tool_frac = [], []

    for r in records:
        lat = r.get("latency")
        if not lat:
            continue
        e2e_s = lat.get("e2e_seconds", 0)
        e2e.append(e2e_s)
        ttft.append(lat.get("total_ttft_seconds", 0))
        dec_s = lat.get("total_decode_seconds", 0)
        tl_s = lat.get("total_tool_seconds", 0)
        decode.append(dec_s)
        tool.append(tl_s)
        tpot.append(lat.get("mean_tpot_ms", 0))
        tok = lat.get("total_completion_tokens", 0)
        tokens.append(tok)
        rounds_cnt.append(lat.get("total_rounds", 1))

        if e2e_s > 0:
            decode_frac.append(dec_s / e2e_s)
            tool_frac.append(tl_s / e2e_s)

        for rnd in lat.get("rounds", []):
            per_round_decode.append(rnd.get("llm_decode_seconds", 0))
            per_round_ttft.append(rnd.get("llm_ttft_seconds", 0))
            per_round_tokens.append(rnd.get("completion_tokens", 0))
            tname = rnd.get("tool_name")
            ts = rnd.get("tool_seconds", 0)
            if tname:
                tool_time_by_name[tname].append(ts)
                tool_calls_by_name[tname] += 1

    # tokens/s derived from decode time
    throughput = [t / d for t, d in zip(tokens, decode) if d > 0]

    return {
        "n": len(e2e),
        "e2e": e2e,
        "ttft": ttft,
        "decode": decode,
        "tool": tool,
        "tpot": tpot,
        "tokens": tokens,
        "rounds": rounds_cnt,
        "decode_frac": decode_frac,
        "tool_frac": tool_frac,
        "throughput": throughput,
        "per_round_decode": per_round_decode,
        "per_round_ttft": per_round_ttft,
        "per_round_tokens": per_round_tokens,
        "tool_time_by_name": dict(tool_time_by_name),
        "tool_calls_by_name": dict(tool_calls_by_name),
    }


# ── report builders ──────────────────────────────────────────────────────────

def summary_table(stats_map: dict[str, dict]) -> str:
    labels = list(stats_map.keys())
    rows = [
        ("样本数 (questions × iters)", "n", None, 0),
        ("E2E 时延 均值 (s)", "e2e", mean, 1),
        ("E2E 时延 中位数 (s)", "e2e", lambda v: pct(v, 50), 1),
        ("E2E 时延 P90 (s)", "e2e", lambda v: pct(v, 90), 1),
        ("E2E 时延 P99 (s)", "e2e", lambda v: pct(v, 99), 1),
        ("TTFT 均值 (s)", "ttft", mean, 2),
        ("Decode 时间 均值 (s)", "decode", mean, 1),
        ("Tool 时间 均值 (s)", "tool", mean, 2),
        ("Decode 占比 均值 (%)", "decode_frac", lambda v: mean(v) * 100, 1),
        ("Tool 占比 均值 (%)", "tool_frac", lambda v: mean(v) * 100, 1),
        ("TPOT 均值 (ms)", "tpot", mean, 2),
        ("吞吐量 (tokens/s)", "throughput", mean, 2),
        ("生成 tokens 均值", "tokens", mean, 0),
        ("ReAct 轮数 均值", "rounds", mean, 2),
    ]

    col_w = 32
    header = f"| {'指标':<{col_w}} | " + " | ".join(f"{l:>12}" for l in labels) + " |"
    sep = f"| {'-'*col_w} | " + " | ".join(f"{'-'*12}" for _ in labels) + " |"
    lines = [header, sep]

    for label, key, fn, decimals in rows:
        cells = []
        for lbl in labels:
            s = stats_map[lbl]
            if fn is None:
                cells.append(f"{s[key]:>12}")
            else:
                v = fn(s[key])
                cells.append(f"{fmt(v, decimals):>12}")
        lines.append(f"| {label:<{col_w}} | " + " | ".join(cells) + " |")

    return "\n".join(lines)


def tool_breakdown_table(stats_map: dict[str, dict]) -> str:
    # collect all tool names
    all_tools: set[str] = set()
    for s in stats_map.values():
        all_tools.update(s["tool_time_by_name"].keys())
    all_tools = sorted(all_tools)

    labels = list(stats_map.keys())
    col_w = 20

    def hdr(title):
        h = f"| {'工具':<{col_w}} | " + " | ".join(
            f"{'调用次数':>8}({'avg_s':>6})" for _ in labels
        ) + " |"
        return f"**{title}**\n\n" + h

    # build header
    header = f"| {'工具':<{col_w}} |"
    for lbl in labels:
        header += f" {lbl+' 调用次数':>14} | {lbl+' 均值(s)':>14} |"
    sep = f"| {'-'*col_w} |" + f" {'-'*14} | {'-'*14} |" * len(labels)
    lines = [header, sep]

    for tool in all_tools:
        row = f"| {tool:<{col_w}} |"
        for lbl in labels:
            s = stats_map[lbl]
            cnt = s["tool_calls_by_name"].get(tool, 0)
            times = s["tool_time_by_name"].get(tool, [])
            avg = mean(times)
            row += f" {cnt:>14} | {fmt(avg, 2):>14} |"
        lines.append(row)

    return "\n".join(lines)


def latency_dist_table(stats_map: dict[str, dict]) -> str:
    """Latency distribution buckets."""
    labels = list(stats_map.keys())
    buckets = [(0, 300), (300, 600), (600, 900), (900, 1200), (1200, 1800), (1800, None)]

    col_w = 20
    header = f"| {'区间 (s)':<{col_w}} |" + "".join(f" {l+' 占比%':>14} |" for l in labels)
    sep = f"| {'-'*col_w} |" + f" {'-'*14} |" * len(labels)
    lines = [header, sep]

    for lo, hi in buckets:
        label_str = f"[{lo}, {hi if hi else '∞'})"
        row = f"| {label_str:<{col_w}} |"
        for lbl in labels:
            vals = stats_map[lbl]["e2e"]
            if not vals:
                row += f" {'N/A':>14} |"
                continue
            cnt = sum(1 for v in vals if v >= lo and (hi is None or v < hi))
            row += f" {cnt/len(vals)*100:>13.1f}% |"
        lines.append(row)

    return "\n".join(lines)


def per_round_table(stats_map: dict[str, dict]) -> str:
    labels = list(stats_map.keys())
    rows = [
        ("平均每轮 Decode (s)", "per_round_decode", mean, 2),
        ("平均每轮 TTFT (s)", "per_round_ttft", mean, 3),
        ("平均每轮生成 tokens", "per_round_tokens", mean, 1),
    ]
    col_w = 24
    header = f"| {'指标':<{col_w}} | " + " | ".join(f"{l:>12}" for l in labels) + " |"
    sep = f"| {'-'*col_w} | " + " | ".join(f"{'-'*12}" for _ in labels) + " |"
    lines = [header, sep]
    for label, key, fn, decimals in rows:
        cells = [f"{fmt(fn(stats_map[lbl][key]), decimals):>12}" for lbl in labels]
        lines.append(f"| {label:<{col_w}} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def delta_summary(stats_map: dict[str, dict]) -> str:
    """FP16 vs Int4: relative differences on key metrics."""
    if "FP16" not in stats_map or "Int4" not in stats_map:
        return ""
    fp16 = stats_map["FP16"]
    int4 = stats_map["Int4"]

    def rel(key, fn=mean):
        a = fn(fp16[key])
        b = fn(int4[key])
        if math.isnan(a) or math.isnan(b) or a == 0:
            return "N/A"
        return f"{(b - a) / a * 100:+.1f}%"

    lines = [
        "Int4 相对 FP16 的变化（正数 = Int4 更大）：",
        "",
        f"- E2E 时延: {rel('e2e')}",
        f"- Decode 时间: {rel('decode')}",
        f"- Tool 时间: {rel('tool')}",
        f"- TPOT: {rel('tpot')}",
        f"- 吞吐量 (tokens/s): {rel('throughput')}",
        f"- 生成 tokens: {rel('tokens')}",
        f"- ReAct 轮数: {rel('rounds')}",
    ]
    return "\n".join(lines)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", default=str(DEFAULT_OUTPUTS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)

    stats_map: dict[str, dict] = {}
    for label, model_dir in MODELS.items():
        records = load_records(outputs_dir, model_dir)
        print(f"[{label}] loaded {len(records)} records from {model_dir}/{HLE_DIR}")
        stats_map[label] = extract_latency_stats(records)

    md_lines = [
        "# HLE 时延分析：FP16 vs Int4-W4A16",
        "",
        f"> 数据目录：`inference/outputs/*/test_hle/`，合并 iter1/iter2/iter3",
        "",
        "## 1. 核心时延指标汇总",
        "",
        summary_table(stats_map),
        "",
        "## 2. E2E 时延分布",
        "",
        latency_dist_table(stats_map),
        "",
        "## 3. 逐轮（per-round）统计",
        "",
        per_round_table(stats_map),
        "",
        "## 4. 工具调用时延明细",
        "",
        tool_breakdown_table(stats_map),
        "",
        "## 5. FP16 vs Int4 对比小结",
        "",
        delta_summary(stats_map),
        "",
    ]

    out = Path(args.output)
    out.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Report written to {out}")


if __name__ == "__main__":
    main()
