#!/usr/bin/env python3
"""Inference evaluation for base and LoRA models on math12k test set."""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from mathruler.grader import extract_boxed_content, grade_answer
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# Copied from examples/reward_function/math.py to avoid module name clash
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


def evaluate_model(
    model_name: str,
    model_path: str,
    adapter_path: str | None,
    tokenizer_path: str,
    num_samples: int,
    output_file: str,
    batch_size: int = 8,
    max_new_tokens: int = 2048,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{model_name}] Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[{model_name}] Loading base model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    if adapter_path:
        print(f"[{model_name}] Loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"[{model_name}] Merging LoRA weights")
        model = model.merge_and_unload()

    model.eval()

    ds = load_dataset("hiyouga/math12k", split="test")
    if num_samples > 0:
        ds = ds.select(range(min(num_samples, len(ds))))

    template_path = Path(__file__).parent.parent / "examples" / "format_prompt" / "math.jinja"
    template = template_path.read_text().strip()

    results = []
    for i in tqdm(range(0, len(ds), batch_size), desc=f"[{model_name}] Generating"):
        batch = ds[i : i + batch_size]
        problems = batch["problem"]
        answers = batch["answer"]
        prompts = [build_prompt(p, template) for p in problems]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.6,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # Remove prompt from generated text
        responses = []
        for prompt, text in zip(prompts, generated_texts):
            response = text[len(prompt) :].strip()
            responses.append(response)

        reward_inputs = [
            {"response": r, "ground_truth": a} for r, a in zip(responses, answers)
        ]
        scores = compute_score(reward_inputs, format_weight=0.1)

        for problem, answer, response, score in zip(
            problems, answers, responses, scores
        ):
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

    # Save results
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Print summary
    n = len(results)
    avg_overall = sum(r["overall"] for r in results) / n
    avg_format = sum(r["format"] for r in results) / n
    avg_accuracy = sum(r["accuracy"] for r in results) / n
    print(f"\n[{model_name}] Results on {n} samples:")
    print(f"  Overall:  {avg_overall:.4f}")
    print(f"  Format:   {avg_format:.4f}")
    print(f"  Accuracy: {avg_accuracy:.4f}")
    print(f"  Saved to: {output_file}")

    del model
    torch.cuda.empty_cache()
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

    # Per-sample improvement
    improved = sum(
        1
        for b, l in zip(base_results, lora_results)
        if l["overall"] > b["overall"]
    )
    same = sum(
        1
        for b, l in zip(base_results, lora_results)
        if l["overall"] == b["overall"]
    )
    worsened = n - improved - same
    print(f"\nPer-sample overall score:")
    print(f"  Improved: {improved}/{n} ({100*improved/n:.1f}%)")
    print(f"  Same:     {same}/{n} ({100*same/n:.1f}%)")
    print(f"  Worsened: {worsened}/{n} ({100*worsened/n:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", default="checkpoints/easy_r1/qwen2_5_1.5b_math_grpo_lora/eval")
    parser.add_argument("--skip_base", action="store_true")
    parser.add_argument("--skip_lora", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer_path = args.tokenizer_path or args.model_path

    base_file = os.path.join(args.output_dir, "base_results.jsonl")
    lora_file = os.path.join(args.output_dir, "lora_results.jsonl")

    if not args.skip_base:
        evaluate_model(
            "Base",
            args.model_path,
            adapter_path=None,
            tokenizer_path=tokenizer_path,
            num_samples=args.num_samples,
            output_file=base_file,
            batch_size=args.batch_size,
        )

    if not args.skip_lora and args.adapter_path:
        evaluate_model(
            "LoRA",
            args.model_path,
            adapter_path=args.adapter_path,
            tokenizer_path=tokenizer_path,
            num_samples=args.num_samples,
            output_file=lora_file,
            batch_size=args.batch_size,
        )

    if os.path.exists(base_file) and os.path.exists(lora_file):
        compare_results(base_file, lora_file)


if __name__ == "__main__":
    main()
