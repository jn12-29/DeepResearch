# convert GAIA parquet -> JSONL for DeepResearch inference
# outputs to data/gaia/converted/, copies attachments to inference/eval_data/file_corpus/
python data/gaia/scripts/convert_to_jsonl.py --split all

# then set in .env:
#   DATASET=/home/xh/DeepResearch/data/gaia/converted/gaia_2023_validation.jsonl
# and run:
#   bash inference/run_react_infer.sh

---

# run code sandbox
sudo docker run -it --rm -p 8081:8080 vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609

hf download gaia-benchmark/GAIA --repo-type dataset --local-dir ./gaia

# kill old vllm
kill -9 -$(lsof -t -i:6006)
ps aux | grep vllm
pkill -9 -f vllm
# download hle

modelscope download --dataset cais/hle

modelscope download --model okwinds/Tongyi-DeepResearch-30B-A3B-Int4-W4A16 --local_dir ./Tongyi-DeepResearch-30B-A3B-Int4-W4A16

cd SandboxFusion && conda activate sandbox && make run-online


---
方案一：tmux（推荐）

conda activate react_infer_env
bash inference/run_react_infer.sh

之后按 Ctrl+B 然后 D 来分离（detach），脚本继续在后台跑。
tmux detach

tmux new -s dr
tmux attach -t dr

tmux new -s sb
tmux attach -t sb

列出所有会话：
tmux ls

---
方案二：nohup + 日志文件

mkdir -p logs
nohup bash inference/run_react_infer.sh > logs/infer_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "PID: $!"

实时追踪日志：
tail -f logs/infer_*.log

---

CUDA_VISIBLE_DEVICES=6,7 python quant.py


auto-round \
  --model Tongyi-DeepResearch-30B-A3B \
  --scheme W4A16 \
  --iters 0 \
  --group_size 128 \
  --format auto_gptq \
  --output_dir Tongyi-DeepResearch-30B-A3B-RTN-W4A16