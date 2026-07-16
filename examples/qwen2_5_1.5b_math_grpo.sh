#!/bin/bash

set -x

export HF_ENDPOINT=https://hf-mirror.com

MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct  # replace it with your local file path if needed

python3 -m verl.trainer.main \
    config=examples/config_1.5b_math_grpo.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    "$@"
