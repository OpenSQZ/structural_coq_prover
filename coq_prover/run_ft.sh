export PYTHONPATH=.
export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_IB_TIMEOUT=21
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

MODEL_NAME="/hf_cache/hub/Qwen2.5-7B"
DATA_PATH="ft_data/prompt_tactic_test.jsonl"
EVAL_PATH="ft_data/prompt_tactic_test.jsonl"
CURRENT_DATETIME=$(date "+%Y-%m-%d-%H-%M-%S")

SAVE_PATH="./coq-prover/checkpoints/${CURRENT_DATETIME}"

deepspeed --force_multi --hostfile hostfile sft_train.py \
  --model_name_or_path "$MODEL_NAME" \
  --data_path "$DATA_PATH" \
  --eval_path "$EVAL_PATH" \
  --output_dir "$SAVE_PATH" \
  --num_train_epochs 5 \
  --model_max_length 20000 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --save_strategy "steps" \
  --save_steps 2000 \
  --save_total_limit 5 \
  --learning_rate 1e-5 \
  --weight_decay 0.05 \
  --lr_scheduler_type "linear" \
  --warmup_steps 2000 \
  --min_lr_rate 0.01 \
  --gradient_log_freq 50 \
  --offload_adam true \
  --offload_params true \
  --gradient_checkpointing true \
  --zero_stage 3 \
  --wandb_enabled true \
  --wandb_project_name "coq-prover-full" \
