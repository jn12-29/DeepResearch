#!/bin/bash

# Load environment variables from .env file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: .env file not found at $ENV_FILE"
    echo "Please copy .env.example to .env and configure your settings:"
    echo "  cp .env.example .env"
    exit 1
fi

echo "Loading environment variables from .env file..."
set -a  # automatically export all variables
source "$ENV_FILE"
set +a  # stop automatically exporting

# =============================================================================
# Configuration
# =============================================================================
INPUT_FP="${1:-$OUTPUT_PATH}"
TOKENIZER_PATH="${2:-$MODEL_PATH}"
MAX_WORKERS="${MAX_WORKERS:-10}"

if [ -z "$INPUT_FP" ]; then
    echo "Error: input path not specified."
    echo "Usage: bash evaluation/run_hle.sh <input_folder_or_file> [tokenizer_path]"
    echo "  Or set OUTPUT_PATH in .env"
    exit 1
fi

if [ -z "$TOKENIZER_PATH" ]; then
    echo "Error: tokenizer path not specified."
    echo "Usage: bash evaluation/run_hle.sh <input_folder_or_file> <tokenizer_path>"
    echo "  Or set MODEL_PATH in .env"
    exit 1
fi

# =============================================================================
# Run evaluation
# =============================================================================
run_eval() {
    local jsonl_file="$1"
    echo "----------------------------------------------------------------------"
    echo "Evaluating: $jsonl_file"
    echo "----------------------------------------------------------------------"
    python evaluation/evaluate_hle_official.py \
        --input_fp "$jsonl_file" \
        --tokenizer_path "$TOKENIZER_PATH" \
        --max_workers "$MAX_WORKERS"
}

if [ -f "$INPUT_FP" ]; then
    # Single file
    run_eval "$INPUT_FP"
elif [ -d "$INPUT_FP" ]; then
    # Directory: iterate over all iter*.jsonl files in order
    shopt -s nullglob
    files=("$INPUT_FP"/iter*.jsonl)
    if [ ${#files[@]} -eq 0 ]; then
        echo "Error: no iter*.jsonl files found in $INPUT_FP"
        exit 1
    fi
    echo "Found ${#files[@]} file(s) to evaluate in $INPUT_FP"
    for f in "${files[@]}"; do
        run_eval "$f"
    done
else
    echo "Error: $INPUT_FP is not a valid file or directory"
    exit 1
fi

echo "======================================================================"
echo "All evaluations done. Results saved alongside each input .jsonl file."
echo "======================================================================"
