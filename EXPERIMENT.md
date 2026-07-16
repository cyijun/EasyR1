# EasyR1 LoRA GRPO 1.5B 数学推理实验记录

## 1. 项目简介

[EasyR1](https://github.com/hiyouga/EasyR1) 是一个基于 FSDP 和 vLLM 的 RL-based visual reasoning 训练框架，支持 GRPO、REINFORCE++ 等算法，原生支持 LoRA 微调。本实验使用单卡环境，以 `Qwen/Qwen2.5-1.5B-Instruct` 为基座，在 `hiyouga/math12k` 数据集上跑通 LoRA GRPO 流程，并与原模型在测试集上做推理对比。

## 2. 环境信息

- 仓库：`hiyouga/EasyR1`（后续 fork 到个人账号）
- 工作目录：`/home/cyijun/easyr1`
- 模型：`Qwen/Qwen2.5-1.5B-Instruct`
- 数据集：`hiyouga/math12k`（train / test / val 均来自该数据集）
- 算法：GRPO（`adv_estimator: grpo`）
- 硬件：单 GPU
- 镜像源：`HF_ENDPOINT=https://hf-mirror.com`

## 3. 训练配置

主要配置文件：`examples/config_1.5b_math_grpo.yaml`

关键超参数：

| 配置项 | 值 | 说明 |
|--------|-----|------|
| `worker.actor.model.lora.rank` | 64 | LoRA rank |
| `worker.actor.model.lora.alpha` | 64 | LoRA alpha |
| `worker.actor.model.lora.target_modules` | `all-linear` | 对所有线性层加 LoRA |
| `worker.actor.model.lora.exclude_modules` | `.*visual.*` | 不微调视觉层 |
| `worker.actor.optim.lr` | 1e-5 | Actor 学习率 |
| `worker.actor.global_batch_size` | 32 | 全局更新 batch size |
| `data.rollout_batch_size` | 128 | rollout batch size |
| `worker.rollout.n` | 5 | 每条 prompt 采样数 |
| `algorithm.kl_coef` | 1e-2 | KL penalty 系数 |
| `trainer.val_freq` | 5 | 每 5 步验证 |
| `trainer.save_freq` | 5 | 每 5 步保存 checkpoint |

启动脚本：`examples/qwen2_5_1.5b_math_grpo_lora.sh`

```bash
python3 -m verl.trainer.main \
    config=examples/config_1.5b_math_grpo.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.lora.rank=64 \
    worker.actor.optim.lr=1e-5 \
    worker.rollout.disable_tqdm=true \
    trainer.experiment_name=qwen2_5_1.5b_math_grpo_lora
```

> 注：脚本已加上日志重定向与 `worker.rollout.disable_tqdm=true`，避免 tqdm 刷屏并保留完整训练日志。

## 4. 训练过程

- 实验从 step 15 的 checkpoint 断点续训（`find_last_checkpoint=true`）。
- 续跑至 step 50，共保存 3 个 checkpoint：`global_step_40`、`global_step_45`、`global_step_50`。
- 验证集最佳表现出现在 **step 45**：val overall reward 0.5978，format 0.974，accuracy 0.556。
- step 50 的 val overall reward 为 0.5962，略有回落。

训练日志路径：`checkpoints/easy_r1/qwen2_5_1.5b_math_grpo_lora/training.log`

实验指标日志路径：`checkpoints/easy_r1/qwen2_5_1.5b_math_grpo_lora/experiment_log.jsonl`

## 5. 推理评估

### 5.1 评估脚本

使用 vLLM 对 base 模型与 LoRA checkpoint 做同条件推理对比：

```bash
python scripts/eval_inference_vllm.py \
  --model_path Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path checkpoints/easy_r1/qwen2_5_1.5b_math_grpo_lora/global_step_45 \
  --num_samples 500 \
  --output_dir checkpoints/easy_r1/qwen2_5_1.5b_math_grpo_lora/eval_full \
  --gpu_memory_utilization 0.85
```

脚本位置：`scripts/eval_inference_vllm.py`

评估指标沿用项目默认的 math reward function：

- `format`：输出是否符合 `<think>...</think>\boxed{...}` 格式
- `accuracy`：提取 `\boxed{}` 答案并与标准答案做数学等价判断
- `overall`：`0.9 * accuracy + 0.1 * format`

### 5.2 评估结果

在 `hiyouga/math12k` 测试集前 500 条上对比：

| Metric | Base | LoRA (step 45) | Δ |
|--------|------|----------------|---|
| Overall | 0.4242 | 0.5136 | **+0.0894** |
| Format | 0.3000 | 0.3840 | **+0.0840** |
| Accuracy | 0.4380 | 0.5280 | **+0.0900** |

逐样本 overall 分数变化：

- Improved：169/500（33.8%）
- Same：231/500（46.2%）
- Worsened：100/500（20.0%）

结果文件路径：

- Base：`checkpoints/easy_r1/qwen2_5_1.5b_math_grpo_lora/eval_full/base_results.jsonl`
- LoRA step 45：`checkpoints/easy_r1/qwen2_5_1.5b_math_grpo_lora/eval_full/lora_results.jsonl`

## 6. 结论

- 成功在单卡环境下跑通 EasyR1 的 LoRA GRPO 1.5B 数学推理流程。
- 使用 LoRA rank=64 微调 35 个 step（15→50）后，模型在 math12k 测试集上 overall reward 提升约 **8.9 个百分点**，accuracy 提升约 **9 个百分点**，format 符合率提升约 **8.4 个百分点**。
- 最佳 checkpoint 为 step 45，后续可作为该实验的推荐模型。

## 7. 说明

- 本实验未修改 EasyR1 核心训练代码，仅新增/调整了启动脚本、配置文件与评估脚本。
- checkpoints、日志、结果文件体积较大，未提交到 git，仅保留脚本与实验记录文档。
- 仓库原有测试套件（`tests/`）因当前环境缺少 `qwen_vl_utils`、`codetiming` 等依赖而无法完整收集，属于环境依赖问题，与本实验改动无关。
