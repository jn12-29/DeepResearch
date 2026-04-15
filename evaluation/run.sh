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

# python evaluation/evaluate_deepsearch_official.py --input_folder /home/xh/DeepResearch/inference/outputs/Tongyi-DeepResearch-30B-A3B_sglang/DeepSearch-2510 --dataset xbench-deepsearch

python evaluation/evaluate_hle_official.py --input_fp /home/xh/DeepResearch/inference/outputs/Tongyi-DeepResearch-30B-A3B_sglang/hle_text_200