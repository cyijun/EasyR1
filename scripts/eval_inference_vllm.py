#!/usr/bin/env python3
"""Inference evaluation with vLLM for base and LoRA models on math12k test."""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from datasets import load_dataset
from mathruler.grader import extract_boxed_content, grade_answer
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


FORMAT_PATTERN = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)


def format_reward(response: str) -> float:
    return 1.0 if re.fullmatch(FORMAT_PATTERN, response) else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(reward_inputs, format_weight: float = 0.1):
    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )
    return scores


def build_prompt(problem: str, template: str) -> str:
    return template.replace("{{ content | trim }}", problem.strip())


def evaluate_vllm(
    model_name: str,
    model_path: str,
    adapter_path: str | None,
    num_samples: int,
    output_file: str,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.9,
):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"[{model_name}] Loading vLLM model from {model_path}")
    llm_kwargs = dict(
        model=model_path,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_lora=adapter_path is not None,
        trust_remote_code=True,
        disable_log_stats=True,
    )
    if adapter_path is not None:
        # Training used LoRA rank=64
        llm_kwargs["max_lora_rank"] = 64
    llm = LLM(**llm_kwargs)

    ds = load_dataset("hiyouga/math12k", split="test")
    if num_samples > 0:
        ds = ds.select(range(min(num_samples, len(ds))))

    template_path = Path(__file__).parent.parent / "examples" / "format_prompt" / "math.jinja"
    template = template_path.read_text().strip()

    prompts = [build_prompt(p, template) for p in ds["problem"]]
    answers = ds["answer"]

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=2048,
        n=1,
    )

    kwargs = {"sampling_params": sampling_params}
    if adapter_path:
        kwargs["lora_request"] = LoRARequest("lora", 1, adapter_path)

    print(f"[{model_name}] Generating {len(prompts)} responses...")
    outputs = llm.generate(prompts, **kwargs)

    responses = []
    for out in outputs:
        text = out.outputs[0].text
        # vllm may include prompt; strip if so
        prompt = out.prompt
        if text.startswith(prompt):
            text = text[len(prompt):]
        responses.append(text.strip())

    reward_inputs = [{"response": r, "ground_truth": a} for r, a in zip(responses, answers)]
    scores = compute_score(reward_inputs, format_weight=0.1)

    results = []
    for problem, answer, response, score in zip(ds["problem"], answers, responses, scores):
        results.append(
            {
                "problem": problem,
                "ground_truth": answer,
                "response": response,
                "overall": score["overall"],
                "format": score["format"],
                "accuracy": score["accuracy"],
            }
        )

    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(results)
    avg_overall = sum(r["overall"] for r in results) / n
    avg_format = sum(r["format"] for r in results) / n
    avg_accuracy = sum(r["accuracy"] for r in results) / n
    print(f"\n[{model_name}] Results on {n} samples:")
    print(f"  Overall:  {avg_overall:.4f}")
    print(f"  Format:   {avg_format:.4f}")
    print(f"  Accuracy: {avg_accuracy:.4f}")
    print(f"  Saved to: {output_file}")

    del llm
    return results


def compare_results(base_file: str, lora_file: str):
    base_results = [json.loads(line) for line in open(base_file)]
    lora_results = [json.loads(line) for line in open(lora_file)]

    assert len(base_results) == len(lora_results)
    n = len(base_results)

    metrics = ["overall", "format", "accuracy"]
    print(f"\n=== Comparison on {n} test samples ===")
    print(f"{'Metric':<12} {'Base':>10} {'LoRA':>10} {'Δ':>10}")
    for m in metrics:
        base_avg = sum(r[m] for r in base_results) / n
        lora_avg = sum(r[m] for r in lora_results) / n
        delta = lora_avg - base_avg
        print(f"{m:<12} {base_avg:>10.4f} {lora_avg:>10.4f} {delta:>+10.4f}")

    improved = sum(1 for b, l in zip(base_results, lora_results) if l["overall"] > b["overall"])
    same = sum(1 for b, l in zip(base_results, lora_results) if l["overall"] == b["overall"])
    worsened = n - improved - same
    print(f"\nPer-sample overall score:")
    print(f"  Improved: {improved}/{n} ({100*improved/n:.1f}%)")
    print(f"  Same:     {same}/{n} ({100*same/n:.1f}%)")
    print(f"  Worsened: {worsened}/{n} ({100*worsened/n:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--output_dir", default="checkpoints/easy_r1/qwen2_5_1.5b_math_grpo_lora/eval")
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--skip_base", action="store_true")
    parser.add_argument("--skip_lora", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    base_file = os.path.join(args.output_dir, "base_results.jsonl")
    lora_file = os.path.join(args.output_dir, "lora_results.jsonl")

    if not args.skip_base:
        evaluate_vllm(
            "Base",
            args.model_path,
            adapter_path=None,
            num_samples=args.num_samples,
            output_file=base_file,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

    if not args.skip_lora and args.adapter_path:
        evaluate_vllm(
            "LoRA",
            args.model_path,
            adapter_path=args.adapter_path,
            num_samples=args.num_samples,
            output_file=lora_file,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

    if os.path.exists(base_file) and os.path.exists(lora_file):
        compare_results(base_file, lora_file)


if __name__ == "__main__":
    main()
