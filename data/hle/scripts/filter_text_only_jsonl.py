#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Filter a full HLE JSONL down to text-only samples."
    )
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Filtered JSONL path")
    parser.add_argument(
        "--deepresearch-output",
        help="Optional DeepResearch-format JSONL path with only question/answer",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    deepresearch_output = Path(args.deepresearch_output) if args.deepresearch_output else None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if deepresearch_output:
        deepresearch_output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0

    with input_path.open("r", encoding="utf-8") as in_fp, output_path.open(
        "w", encoding="utf-8"
    ) as out_fp:
        dr_fp = (
            deepresearch_output.open("w", encoding="utf-8")
            if deepresearch_output
            else None
        )
        try:
            for line in in_fp:
                line = line.strip()
                if not line:
                    continue
                total += 1
                item = json.loads(line)
                if item.get("image"):
                    continue
                kept += 1
                out_fp.write(json.dumps(item, ensure_ascii=False) + "\n")
                if dr_fp:
                    dr_fp.write(
                        json.dumps(
                            {
                                "question": item.get("question", ""),
                                "answer": item.get("answer", ""),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        finally:
            if dr_fp:
                dr_fp.close()

    print(
        json.dumps(
            {
                "input_rows": total,
                "kept_rows": kept,
                "removed_rows": total - kept,
                "output": str(output_path),
                "deepresearch_output": (
                    str(deepresearch_output) if deepresearch_output else None
                ),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
