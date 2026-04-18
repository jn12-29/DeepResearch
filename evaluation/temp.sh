export API_KEY=dummy

# DeepSearch 系列
python evaluation/evaluate_deepsearch_official.py \
    --input_folder /home/xh/DeepResearch/inference/outputs/Tongyi-DeepResearch-30B-A3B-Int4-W4A16_sglang/gaia_2023_validation \
    --dataset gaia \
    --judge_model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --base_url http://127.0.0.1:8000/v1 \
    --max_workers 32

python evaluation/evaluate_deepsearch_official.py \
    --input_folder /home/xh/DeepResearch/inference/outputs/Tongyi-DeepResearch-30B-A3B-Int4-W4A16_sglang/DeepSearch-2510 \
    --dataset xbench-deepsearch \
    --judge_model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --base_url http://127.0.0.1:8000/v1 \
    --max_workers 32

# HLE 目录（3轮 Pass@k）
python evaluation/evaluate_hle_official.py \
    --input_folder /home/xh/DeepResearch/inference/outputs/Tongyi-DeepResearch-30B-A3B-Int4-W4A16_sglang/hle_text_200 \
    --judge_model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --base_url http://127.0.0.1:8000/v1 \
    --max_workers 32


# DeepSearch 系列
python evaluation/evaluate_deepsearch_official.py \
    --input_folder /home/xh/DeepResearch/inference/outputs/Tongyi-DeepResearch-30B-A3B_sglang/gaia_2023_validation \
    --dataset gaia \
    --judge_model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --base_url http://127.0.0.1:8000/v1 \
    --max_workers 32

python evaluation/evaluate_deepsearch_official.py \
    --input_folder /home/xh/DeepResearch/inference/outputs/Tongyi-DeepResearch-30B-A3B_sglang/DeepSearch-2510 \
    --dataset xbench-deepsearch \
    --judge_model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --base_url http://127.0.0.1:8000/v1 \
    --max_workers 32

# HLE 目录（3轮 Pass@k）
python evaluation/evaluate_hle_official.py \
    --input_folder /home/xh/DeepResearch/inference/outputs/Tongyi-DeepResearch-30B-A3B_sglang/hle_text_200 \
    --judge_model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --base_url http://127.0.0.1:8000/v1 \
    --max_workers 32


API_KEY=sk-a8d558bc9f844894a9bfe6380e39540c
API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
python evaluation/evaluate_deepsearch_official.py \
    --input_folder /home/xh/DeepResearch/inference/outputs/Tongyi-DeepResearch-30B-A3B-Int4-W4A16_sglang/gaia_2023_validation \
    --dataset gaia \
    --base_url https://dashscope.aliyuncs.com/compatible-mode/v1 \
    --max_workers 32
