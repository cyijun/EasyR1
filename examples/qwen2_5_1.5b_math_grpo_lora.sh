#!/bin/bash

set -x

export HF_ENDPOINT=https://hf-mirror.com

MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct  # replace it with your local file path if needed

LOG_FILE="checkpoints/easy_r1/qwen2_5_1.5b_math_grpo_lora/training.log"
mkdir -p "$(dirname "$LOG_FILE")"

python3 -m verl.trainer.main \
    config=examples/config_1.5b_math_grpo.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.lora.rank=64 \
    worker.actor.optim.lr=1e-5 \
    worker.rollout.disable_tqdm=true \
    trainer.experiment_name=qwen2_5_1.5b_math_grpo_lora \
    "$@" >> "$LOG_FILE" 2>&1
