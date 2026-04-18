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
set -a
source "$ENV_FILE"
set +a

# =============================================================================
# Configuration
# =============================================================================
INPUT_FOLDER="${1:-}"
DATASET="${2:-gaia}"

if [ -z "$INPUT_FOLDER" ]; then
    echo "Error: input folder not specified."
    echo "Usage: bash evaluation/run_dr.sh <input_folder> [dataset] [extra_args...]"
    echo "  dataset choices: gaia | browsecomp_zh | browsecomp_en | browsecomp_en_full | webwalker | xbench-deepsearch"
    echo "  extra_args: --judge_model <model> --base_url <url> --max_workers <n> ..."
    exit 1
fi

# Consume the two positional args; everything else is forwarded to Python
[[ $# -ge 2 ]] && shift 2 || shift $#

# =============================================================================
# Run evaluation
# =============================================================================
echo "----------------------------------------------------------------------"
echo "Input folder : $INPUT_FOLDER"
echo "Dataset      : $DATASET"
echo "Extra args   : $*"
echo "----------------------------------------------------------------------"

python evaluation/evaluate_deepsearch_official.py \
    --input_folder "$INPUT_FOLDER" \
    --dataset "$DATASET" \
    "$@"
