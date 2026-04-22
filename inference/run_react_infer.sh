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

# Validate critical variables
if [ "$MODEL_PATH" = "/your/model/path" ] || [ -z "$MODEL_PATH" ]; then
    echo "Error: MODEL_PATH not configured in .env file"
    exit 1
fi

######################################
### 1. start server           ###
######################################

# Data parallel configuration
# Adjust DP_SIZE and CUDA_VISIBLE_DEVICES to match the number of available GPUs.
# Each DP replica occupies one GPU (no tensor parallelism needed: model ~60GB < 143GB per L20X).
# Examples:
#   6 GPU:  DP_SIZE=6, CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
#   8 GPU:  DP_SIZE=8, CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#   2 GPU (test): DP_SIZE=2, CUDA_VISIBLE_DEVICES=0,1
DP_SIZE=${DP_SIZE:-1}
DP_GPUS=${DP_GPUS:-"0"}
VLLM_PORT=${VLLM_PORT:-6001}

check_vllm_server() {
    local port=$1
    local expected_model=$2

    # 1. 基本端口 / models 检查
    local model_info
    model_info=$(curl -s --max-time 5 http://localhost:$port/v1/models 2>/dev/null)
    if [ -z "$model_info" ]; then
        return 1
    fi

    local model_id
    model_id=$(echo "$model_info" | grep -o '"id":"[^"]*"' | head -1 | sed 's/"id":"//;s/"//')
    if [ -z "$model_id" ]; then
        return 1
    fi

    local expected_name
    expected_name=$(basename "$expected_model")
    if [[ "$model_id" != *"$expected_name"* ]]; then
        return 2
    fi

    # 2. 真正打一次 chat completion，确认能推理
    local resp
    resp=$(curl -s --max-time 30 -o /tmp/vllm_health_$port.json -w "%{http_code}" \
        http://localhost:$port/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$model_id\",
            \"messages\": [{\"role\": \"user\", \"content\": \"ping\"}],
            \"max_tokens\": 1,
            \"temperature\": 0
        }")

    if [ "$resp" != "200" ]; then
        echo "  -> port $VLLM_PORT /v1/models OK, but chat/completions returned HTTP $resp"
        echo "  -> response body: $(cat /tmp/vllm_health_$port.json 2>/dev/null | head -c 500)"
        return 3
    fi

    return 0
}

echo "Checking existing VLLM server on port $VLLM_PORT..."
check_vllm_server "$VLLM_PORT" "$MODEL_PATH"
check_result=$?
need_start=true
case $check_result in
    0) echo "Port $VLLM_PORT: VLLM healthy (inference OK), skipping start."
       need_start=false ;;
    2) echo "Error: Port $VLLM_PORT is running a different model, free the port manually."
       exit 1 ;;
    3) echo "Error: Port $VLLM_PORT has a broken VLLM (500 on inference). Kill it and restart."
       exit 1 ;;
    *) echo "Port $VLLM_PORT: no server found, will start." ;;
esac

if [ "$need_start" = "true" ]; then
    echo "Starting VLLM with data-parallel-size=$DP_SIZE on GPUs [$DP_GPUS], port $VLLM_PORT..."
    CUDA_VISIBLE_DEVICES=$DP_GPUS vllm serve $MODEL_PATH \
        --host 0.0.0.0 \
        --port $VLLM_PORT \
        --data-parallel-size $DP_SIZE \
        --data-parallel-backend mp \
        --disable-log-requests &
fi

#######################################################
### 2. Waiting for the server port to be ready  ###
######################################################

timeout=6000
start_time=$(date +%s)

echo "Waiting for VLLM data-parallel server to start..."

while true; do
    model_id=$(curl -s http://localhost:$VLLM_PORT/v1/models 2>/dev/null | grep -o '"id":"[^"]*"' | head -1)
    if [ -n "$model_id" ]; then
        echo "VLLM server (port $VLLM_PORT, DP=$DP_SIZE) is ready! Model: $model_id"
        break
    fi

    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    if [ $elapsed -gt $timeout ]; then
        echo -e "\nError: Server startup timeout after ${timeout} seconds"
        exit 1
    fi

    printf 'Waiting for server to start .....'
    sleep 10
done

echo "VLLM server is running successfully!"

#####################################
### 3. start infer               ####
#####################################

echo "==== start infer... ===="


cd "$( dirname -- "${BASH_SOURCE[0]}" )"

python -u run_multi_react.py --dataset "$DATASET" --output "$OUTPUT_PATH" --max_workers $MAX_WORKERS --model $MODEL_PATH --temperature $TEMPERATURE --presence_penalty $PRESENCE_PENALTY --total_splits ${WORLD_SIZE:-1} --worker_split $((${RANK:-0} + 1)) --roll_out_count $ROLLOUT_COUNT --port_list "$VLLM_PORT"
