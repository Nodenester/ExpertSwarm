"""Evaluate a single trained expert on a held-out test set.

Measures:
- Accuracy: does the output match expected answers?
- Format compliance: does the output match the expected JSON schema?
- Latency: inference time per example
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import jsonlines
import torch
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

from ..generate.sft_generator import EXPERT_TOOLS


def check_format_compliance(output_text: str, expert_name: str) -> dict:
    """Check if the model output matches the expected JSON schema for the expert."""
    expected_schema = EXPERT_TOOLS.get(expert_name, {}).get("output_schema", {})
    if not expected_schema:
        return {"valid_json": False, "schema_match": False, "errors": ["Unknown expert"]}

    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError as e:
        return {"valid_json": False, "schema_match": False, "errors": [f"Invalid JSON: {e}"]}

    errors = []
    for key in expected_schema:
        if key not in parsed:
            errors.append(f"Missing key: {key}")

    return {
        "valid_json": True,
        "schema_match": len(errors) == 0,
        "errors": errors,
        "extra_keys": [k for k in parsed if k not in expected_schema],
    }


def evaluate(
    model_path: str,
    test_data_path: str,
    expert_name: str,
    max_examples: int = 100,
    max_seq_length: int = 2048,
):
    """Run evaluation on a test set."""
    print(f"=== Evaluating expert: {expert_name} ===")
    print(f"  Model: {model_path}")
    print(f"  Test data: {test_data_path}")

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    # Load test data
    test_examples = []
    with jsonlines.open(test_data_path) as reader:
        for item in reader:
            test_examples.append(item)
            if len(test_examples) >= max_examples:
                break

    print(f"  Test examples: {len(test_examples)}")

    # Evaluate
    results = {
        "total": len(test_examples),
        "format_valid_json": 0,
        "format_schema_match": 0,
        "latencies_ms": [],
        "errors": [],
    }

    for i, example in enumerate(test_examples):
        messages = example.get("messages", [])
        if not messages:
            continue

        # Use only the user message as input
        user_msg = messages[0] if messages else {"role": "user", "content": ""}
        input_messages = [user_msg]

        input_text = tokenizer.apply_chat_template(
            input_messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(
            input_text, return_tensors="pt", truncation=True,
            max_length=max_seq_length,
        ).to(model.device)

        # Generate with timing
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.1,
                do_sample=True,
            )
        latency_ms = (time.perf_counter() - start) * 1000

        # Decode
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Check format
        compliance = check_format_compliance(response, expert_name)
        if compliance["valid_json"]:
            results["format_valid_json"] += 1
        if compliance["schema_match"]:
            results["format_schema_match"] += 1
        if compliance["errors"]:
            results["errors"].append({
                "example_idx": i,
                "errors": compliance["errors"],
                "response_preview": response[:200],
            })

        results["latencies_ms"].append(latency_ms)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(test_examples)}] "
                  f"JSON valid: {results['format_valid_json']}/{i+1}, "
                  f"Schema match: {results['format_schema_match']}/{i+1}, "
                  f"Avg latency: {sum(results['latencies_ms'])/len(results['latencies_ms']):.0f}ms")

    # Summary
    n = results["total"]
    avg_latency = sum(results["latencies_ms"]) / len(results["latencies_ms"]) if results["latencies_ms"] else 0
    p50 = sorted(results["latencies_ms"])[len(results["latencies_ms"]) // 2] if results["latencies_ms"] else 0
    p99 = sorted(results["latencies_ms"])[int(len(results["latencies_ms"]) * 0.99)] if results["latencies_ms"] else 0

    summary = {
        "expert": expert_name,
        "model": model_path,
        "total_examples": n,
        "json_valid_rate": results["format_valid_json"] / n if n else 0,
        "schema_match_rate": results["format_schema_match"] / n if n else 0,
        "latency_avg_ms": avg_latency,
        "latency_p50_ms": p50,
        "latency_p99_ms": p99,
        "error_count": len(results["errors"]),
    }

    print(f"\n=== Results ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

    # Save detailed results
    output_dir = Path(model_path).parent / "eval_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "errors.json").write_text(json.dumps(results["errors"], indent=2))
    print(f"\n  Saved results to: {output_dir}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained swarm expert")
    parser.add_argument("--model", required=True, help="Path to trained model/adapter")
    parser.add_argument("--test-data", required=True, help="Path to test JSONL file")
    parser.add_argument("--expert", required=True, help="Expert name (for schema checking)")
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        test_data_path=args.test_data,
        expert_name=args.expert,
        max_examples=args.max_examples,
        max_seq_length=args.max_seq_length,
    )


if __name__ == "__main__":
    main()
