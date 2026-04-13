# HLE Conversion Scripts

This directory contains two small utility scripts for converting the local HLE dataset into JSONL files that are easier to reuse in evaluation pipelines such as `DeepResearch/inference`.

## Scripts

### `parquet_to_jsonl.py`

Converts the original HLE parquet split into:

- a full JSONL that preserves the main benchmark fields
- a lightweight JSONL containing only `question` and `answer`

This is useful when you want:

- a faithful exported copy of the dataset in line-delimited JSON
- a `DeepResearch`-compatible file for direct inference runs

#### Arguments

- `--input`: input parquet file
- `--full-output`: output JSONL path with preserved HLE fields
- `--deepresearch-output`: output JSONL path with only `question` and `answer`
- `--image-note`: optional flag; if set, questions with non-empty `image` fields will be prefixed with a short text note in the DeepResearch-format output

#### Example

```bash
python parquet_to_jsonl.py \
  --input data/test-00000-of-00001.parquet \
  --full-output data/test_full.jsonl \
  --deepresearch-output data/test_deepresearch.jsonl \
  --image-note
```

#### Output shape

Full JSONL example:

```json
{
  "id": "...",
  "question": "...",
  "image": "...",
  "answer": "...",
  "answer_type": "...",
  "author_name": "...",
  "rationale": "...",
  "raw_subject": "...",
  "category": "...",
  "canary": "..."
}
```

DeepResearch JSONL example:

```json
{ "question": "...", "answer": "..." }
```

### `filter_text_only_jsonl.py`

Filters a full HLE JSONL file and removes any sample whose `image` field is non-empty.

This is useful when you want a pure text subset for text-only evaluation.

#### Arguments

- `--input`: input JSONL path, typically the full JSONL produced by `parquet_to_jsonl.py`
- `--output`: filtered JSONL path, preserving the original record structure
- `--deepresearch-output`: optional output path for a filtered `question`/`answer` JSONL

#### Example

```bash
python filter_text_only_jsonl.py \
  --input data/test_full.jsonl \
  --output data/test_text_only_full.jsonl \
  --deepresearch-output data/test_text_only_deepresearch.jsonl
```

## Recommended workflow

1. Export parquet to JSONL with `parquet_to_jsonl.py`
2. If needed, create a pure text subset with `filter_text_only_jsonl.py`
3. Use the filtered `test_text_only_deepresearch.jsonl` file as the input dataset for `DeepResearch/inference`

## Current generated files

Under `data`:

- `test_full.jsonl`: full export, all 2500 samples
- `test_deepresearch.jsonl`: all 2500 samples in `question`/`answer` format
- `test_text_only_full.jsonl`: text-only subset, 2158 samples
- `test_text_only_deepresearch.jsonl`: text-only subset in `question`/`answer` format, 2158 samples
