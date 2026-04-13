#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import pandas as pd


FULL_COLUMNS = [
    "id",
    "question",
    "image",
    "answer",
    "answer_type",
    "author_name",
    "rationale",
    "raw_subject",
    "category",
    "canary",
]


def normalize(value):
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def build_deepresearch_item(row, image_note):
    question = normalize(row["question"]) or ""
    image = normalize(row["image"]) or ""
    if image and image_note:
        question = (
            "[This HLE sample includes an image in the original dataset; "
            "see the full JSONL for the image field.]\n\n" + question
        )
    return {
        "question": question,
        "answer": normalize(row["answer"]) or "",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert an HLE parquet split to reusable JSONL files."
    )
    parser.add_argument("--input", required=True, help="Input parquet file path")
    parser.add_argument(
        "--full-output",
        required=True,
        help="Output JSONL path with the main HLE fields preserved",
    )
    parser.add_argument(
        "--deepresearch-output",
        required=True,
        help="Output JSONL path with only question/answer for DeepResearch",
    )
    parser.add_argument(
        "--image-note",
        action="store_true",
        help="Prefix DeepResearch questions that have images with a text note",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    full_output = Path(args.full_output)
    deepresearch_output = Path(args.deepresearch_output)

    df = pd.read_parquet(input_path)

    full_output.parent.mkdir(parents=True, exist_ok=True)
    deepresearch_output.parent.mkdir(parents=True, exist_ok=True)

    with (
        full_output.open("w", encoding="utf-8") as full_fp,
        deepresearch_output.open("w", encoding="utf-8") as dr_fp,
    ):
        for _, row in df.iterrows():
            full_item = {column: normalize(row[column]) for column in FULL_COLUMNS}
            full_fp.write(json.dumps(full_item, ensure_ascii=False) + "\n")

            dr_item = build_deepresearch_item(row, image_note=args.image_note)
            dr_fp.write(json.dumps(dr_item, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "rows": len(df),
                "input": str(input_path),
                "full_output": str(full_output),
                "deepresearch_output": str(deepresearch_output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
