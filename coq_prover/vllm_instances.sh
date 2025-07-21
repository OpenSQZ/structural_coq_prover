#!/bin/bash

PORTS=(8080 8081 8082 8083 8084 8085 8086 8087)
GPU_IDS=(0 1 2 3 4 5 6 7)
MODEL_PATH="path/to/your/model(should be the same as ft_model_path in config.json)"

if [[ $MODEL_PATH == *"7b"* ]]; then
  TP_SIZE=1
  INSTANCES=${#PORTS[@]}
elif [[ $MODEL_PATH == *"32b"* ]]; then
  TP_SIZE=2
  INSTANCES=4
else
  TP_SIZE=1
  INSTANCES=${#PORTS[@]}
fi

if [[ $MODEL_PATH == *"reorganize"* ]]; then
  MAX_MODEL_LEN=4096
else
  MAX_MODEL_LEN=21000
fi

if [[ $TP_SIZE -eq 2 ]]; then
  PORTS=(${PORTS[@]:0:4})
fi

for i in "${!PORTS[@]}"; do
  PORT=${PORTS[$i]}

  if [[ $TP_SIZE -eq 2 ]]; then
    GPU="${GPU_IDS[$i*2]},${GPU_IDS[$i*2+1]}"
  else
    GPU=${GPU_IDS[$i]}
  fi
  
  if ! nc -z localhost $PORT; then
    echo "VLLM instance at port $PORT (GPU $GPU) is not responding, restarting..."
    
    EXISTING_PID=$(lsof -i:$PORT -t)
    if [ ! -z "$EXISTING_PID" ]; then
      kill -9 $EXISTING_PID
    fi
    
    CUDA_VISIBLE_DEVICES=$GPU python -m vllm.entrypoints.openai.api_server \
      --model $MODEL_PATH \
      --port $PORT \
      --max-model-len $MAX_MODEL_LEN \
      --max-logprobs 100 \
      --enforce-eager \
      --served-model-name $(basename $MODEL_PATH) \
      --tensor-parallel-size $TP_SIZE &
        
    echo "VLLM instance at port $PORT restarted using GPU: $GPU tensor-parallel-size: $TP_SIZE max-model-len: $MAX_MODEL_LEN"
  fi
done
