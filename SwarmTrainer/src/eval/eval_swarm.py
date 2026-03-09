"""End-to-end swarm evaluation.

Runs full swarm tasks through SwarmExecutor and measures:
- Task completion rate
- Output quality (via LLM judge)
- Total latency
- Comparison against baseline (big model direct)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx
import jsonlines
from openai import OpenAI

from .judge import Judge


def run_swarm_task(task: dict, executor_url: str, timeout_s: int = 120) -> dict:
    """Submit a task to SwarmExecutor and wait for completion."""
    client = httpx.Client(timeout=timeout_s)
    try:
        resp = client.post(
            f"{executor_url}/api/v1/execute",
            json={"task": task.get("task", task.get("input", "")), "config": task.get("config", {})},
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as e:
        return {"error": str(e), "status": "failed"}
    finally:
        client.close()


def run_baseline(task: dict, codegate_url: str, api_key: str, model: str) -> dict:
    """Run the same task directly through a big model for comparison."""
    client = OpenAI(base_url=f"{codegate_url}/v1", api_key=api_key)
    start = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": task.get("task", task.get("input", ""))}],
            temperature=0.3,
            max_tokens=4096,
        )
        latency = time.perf_counter() - start
        return {
            "output": resp.choices[0].message.content,
            "latency_s": latency,
            "status": "ok",
            "tokens": resp.usage.total_tokens if resp.usage else 0,
        }
    except Exception as e:
        return {"error": str(e), "status": "failed", "latency_s": time.perf_counter() - start}


def evaluate(
    test_tasks_path: str,
    executor_url: str,
    codegate_url: str,
    api_key: str,
    judge_model: str,
    baseline_model: str,
    max_tasks: int = 50,
    run_baseline_comparison: bool = True,
):
    """Run end-to-end evaluation."""
    print(f"=== End-to-End Swarm Evaluation ===")
    print(f"  Executor: {executor_url}")
    print(f"  Tasks: {test_tasks_path}")

    judge = Judge(codegate_url, api_key, judge_model)

    # Load tasks
    tasks = []
    with jsonlines.open(test_tasks_path) as reader:
        for item in reader:
            tasks.append(item)
            if len(tasks) >= max_tasks:
                break

    print(f"  Loaded {len(tasks)} tasks")

    swarm_results = []
    baseline_results = []

    for i, task in enumerate(tasks):
        print(f"\n--- Task {i+1}/{len(tasks)} ---")
        print(f"  {json.dumps(task, default=str)[:100]}...")

        # Run through swarm
        start = time.perf_counter()
        swarm_output = run_swarm_task(task, executor_url)
        swarm_latency = time.perf_counter() - start

        swarm_score = judge.score_output(task, swarm_output)
        swarm_results.append({
            "task_idx": i,
            "output": swarm_output,
            "score": swarm_score,
            "latency_s": swarm_latency,
        })
        print(f"  Swarm: score={swarm_score['composite']:.2f}, latency={swarm_latency:.1f}s")

        # Run baseline
        if run_baseline_comparison:
            baseline_output = run_baseline(task, codegate_url, api_key, baseline_model)
            baseline_score = judge.score_output(task, baseline_output)
            baseline_results.append({
                "task_idx": i,
                "output": baseline_output,
                "score": baseline_score,
                "latency_s": baseline_output.get("latency_s", 0),
            })
            print(f"  Baseline: score={baseline_score['composite']:.2f}, "
                  f"latency={baseline_output.get('latency_s', 0):.1f}s")

    # Summary
    import numpy as np

    swarm_scores = [r["score"]["composite"] for r in swarm_results]
    swarm_latencies = [r["latency_s"] for r in swarm_results]

    summary = {
        "num_tasks": len(tasks),
        "swarm": {
            "avg_score": float(np.mean(swarm_scores)),
            "std_score": float(np.std(swarm_scores)),
            "completion_rate": sum(1 for r in swarm_results if r["score"]["correctness"] > 0) / len(tasks),
            "avg_latency_s": float(np.mean(swarm_latencies)),
        },
    }

    if baseline_results:
        baseline_scores = [r["score"]["composite"] for r in baseline_results]
        baseline_latencies = [r["latency_s"] for r in baseline_results]
        summary["baseline"] = {
            "avg_score": float(np.mean(baseline_scores)),
            "std_score": float(np.std(baseline_scores)),
            "completion_rate": sum(1 for r in baseline_results if r["score"]["correctness"] > 0) / len(tasks),
            "avg_latency_s": float(np.mean(baseline_latencies)),
        }
        summary["comparison"] = {
            "score_delta": summary["swarm"]["avg_score"] - summary["baseline"]["avg_score"],
            "latency_ratio": summary["swarm"]["avg_latency_s"] / max(summary["baseline"]["avg_latency_s"], 0.001),
        }

    print(f"\n=== Summary ===")
    print(json.dumps(summary, indent=2))

    # Save
    output_dir = Path("eval_results") / "swarm"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "swarm_details.json").write_text(json.dumps(swarm_results, indent=2, default=str))
    if baseline_results:
        (output_dir / "baseline_details.json").write_text(json.dumps(baseline_results, indent=2, default=str))
    print(f"\n  Saved to: {output_dir}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="End-to-end swarm evaluation")
    parser.add_argument("--tasks", required=True, help="Path to test tasks JSONL")
    parser.add_argument("--executor-url", default=os.environ.get("SWARM_RUNNER_URL", "http://localhost:8100"))
    parser.add_argument("--codegate-url", default=os.environ.get("CODEGATE_URL", "http://localhost:9212"))
    parser.add_argument("--api-key", default=os.environ.get("CODEGATE_API_KEY", "cgk_xxx"))
    parser.add_argument("--judge-model", default=os.environ.get("TEACHER_MODEL", "claude-sonnet-4-20250514"))
    parser.add_argument("--baseline-model", default=os.environ.get("TEACHER_MODEL", "claude-sonnet-4-20250514"))
    parser.add_argument("--max-tasks", type=int, default=50)
    parser.add_argument("--no-baseline", action="store_true")
    args = parser.parse_args()

    evaluate(
        test_tasks_path=args.tasks,
        executor_url=args.executor_url,
        codegate_url=args.codegate_url,
        api_key=args.api_key,
        judge_model=args.judge_model,
        baseline_model=args.baseline_model,
        max_tasks=args.max_tasks,
        run_baseline_comparison=not args.no_baseline,
    )


if __name__ == "__main__":
    main()
