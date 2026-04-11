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

# main_ports=(6005 6006)
# gpu_devices=(4 5)
main_ports=(6006)
gpu_devices=(5)

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
        echo "  -> port $port /v1/models OK, but chat/completions returned HTTP $resp"
        echo "  -> response body: $(cat /tmp/vllm_health_$port.json 2>/dev/null | head -c 500)"
        return 3
    fi

    return 0
}

declare -A need_start
for i in "${!main_ports[@]}"; do
    need_start[${main_ports[$i]}]=true
done

echo "Checking existing VLLM servers..."
for port in "${main_ports[@]}"; do
    check_vllm_server "$port" "$MODEL_PATH"
    result=$?
    case $result in
        0) echo "Port $port: VLLM healthy (inference OK), skipping..."
           need_start[$port]=false ;;
        2) echo "Error: Port $port is running a different model, free the port manually."
           exit 1 ;;
        3) echo "Error: Port $port has a broken VLLM (500 on inference). Kill it and restart."
           exit 1 ;;
        *) echo "Port $port: no server, will start"
           need_start[$port]=true ;;
    esac
done

if [ ${#main_ports[@]} -ne ${#gpu_devices[@]} ]; then
    echo "Error: main_ports and gpu_devices must have the same length"
    exit 1
fi

echo "Starting VLLM servers..."
for i in "${!main_ports[@]}"; do
    port=${main_ports[$i]}
    gpu=${gpu_devices[$i]}
    
    if [ "${need_start[$port]}" = "true" ]; then
        echo "Starting VLLM on port $port with GPU $gpu..."
        CUDA_VISIBLE_DEVICES=$gpu vllm serve $MODEL_PATH --host 0.0.0.0 --port $port --disable-log-requests &
    fi
done

#######################################################
### 2. Waiting for the server port to be ready  ###
######################################################

timeout=6000
start_time=$(date +%s)

echo "Mode: All ports used as main model"

declare -A server_status
for port in "${main_ports[@]}"; do
    server_status[$port]=false
done

echo "Waiting for servers to start..."

while true; do
    all_ready=true

    for port in "${main_ports[@]}"; do
        if [ "${server_status[$port]}" = "false" ]; then
            model_id=$(curl -s http://localhost:$port/v1/models 2>/dev/null | grep -o '"id":"[^"]*"' | head -1)
            if [ -n "$model_id" ]; then
                echo "Main model server (port $port) is ready! Model: $model_id"
                server_status[$port]=true
            else
                all_ready=false
            fi
        fi
    done

    if [ "$all_ready" = "true" ]; then
        echo "All servers are ready for inference!"
        break
    fi

    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    if [ $elapsed -gt $timeout ]; then
        echo -e "\nError: Server startup timeout after ${timeout} seconds"

        for port in "${main_ports[@]}"; do
            if [ "${server_status[$port]}" = "false" ]; then
                echo "Main model server (port $port) failed to start"
            fi
        done


        exit 1
    fi

    printf 'Waiting for servers to start .....'
    sleep 10
done

failed_servers=()
for port in "${main_ports[@]}"; do
    if [ "${server_status[$port]}" = "false" ]; then
        failed_servers+=($port)
    fi
done

if [ ${#failed_servers[@]} -gt 0 ]; then
    echo "Error: The following servers failed to start: ${failed_servers[*]}"
    exit 1
else
    echo "All required servers are running successfully!"
fi

#####################################
### 3. start infer               ####
#####################################

echo "==== start infer... ===="


cd "$( dirname -- "${BASH_SOURCE[0]}" )"

python -u run_multi_react.py --dataset "$DATASET" --output "$OUTPUT_PATH" --max_workers $MAX_WORKERS --model $MODEL_PATH --temperature $TEMPERATURE --presence_penalty $PRESENCE_PENALTY --total_splits ${WORLD_SIZE:-1} --worker_split $((${RANK:-0} + 1)) --roll_out_count $ROLLOUT_COUNT --port_list "${main_ports[*]}"
