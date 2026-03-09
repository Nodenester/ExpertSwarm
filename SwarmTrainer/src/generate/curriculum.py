"""Curriculum builder — generates staged training tasks of increasing difficulty.

Stage 1: Basic format examples (SFT cold start) — teach the model output format
Stage 2: RL free exploration tasks — open-ended tasks for policy gradient
Stage 3: Auto-improvement data — model critiques and improves its own outputs
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import jsonlines
import yaml
from openai import OpenAI


STAGE_DESCRIPTIONS = {
    1: "Basic format compliance — output must match the expert's JSON schema exactly",
    2: "Free exploration — open-ended tasks requiring multi-step reasoning",
    3: "Self-improvement — critique and improve previous outputs",
}

# Difficulty tiers within each stage
DIFFICULTY_TIERS = {
    "easy": {
        "description": "Single-step, straightforward, no ambiguity",
        "max_steps": 1,
        "context_size": "small",
    },
    "medium": {
        "description": "2-3 steps, some ambiguity, requires domain knowledge",
        "max_steps": 3,
        "context_size": "medium",
    },
    "hard": {
        "description": "4+ steps, significant ambiguity, edge cases, multi-source",
        "max_steps": 6,
        "context_size": "large",
    },
}


def generate_stage1_tasks(expert_name: str, count: int, client: OpenAI, model: str) -> list[dict]:
    """Stage 1: Generate basic format-compliance training examples.

    These are simple tasks where the main goal is teaching the model
    to produce correctly-structured JSON output.
    """
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": f"""Generate {count} very simple, unambiguous tasks for the "{expert_name}" expert.
These tasks should be easy enough that a basic model can produce the correct output format.
The focus is on FORMAT COMPLIANCE — the model must learn the exact JSON schema.

For each task, return a JSON object:
{{
  "task": "description of the task",
  "difficulty": "easy",
  "stage": 1,
  "expert": "{expert_name}",
  "hints": "what format the output should take"
}}

Return a JSON array of these objects. No markdown fences.""",
            }
        ],
        temperature=0.8,
        max_tokens=4096,
    )
    text = resp.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(text)


def generate_stage2_tasks(expert_name: str, count: int, client: OpenAI, model: str) -> list[dict]:
    """Stage 2: Generate free exploration tasks for RL training.

    These are harder, open-ended tasks where the model must figure out
    the best approach. Used for GRPO policy optimization.
    """
    tier_dist = {"easy": 0.2, "medium": 0.5, "hard": 0.3}
    all_tasks = []

    for tier, fraction in tier_dist.items():
        tier_count = max(1, int(count * fraction))
        tier_info = DIFFICULTY_TIERS[tier]
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"""Generate {tier_count} {tier}-difficulty tasks for the "{expert_name}" expert.

Difficulty: {tier} — {tier_info['description']}
Max steps allowed: {tier_info['max_steps']}
Context size: {tier_info['context_size']}

These tasks are for RL training — they should be open-ended enough that
different approaches could work. Include a "gold_criteria" field describing
what makes a response good.

Return a JSON array of objects:
{{
  "task": "description",
  "difficulty": "{tier}",
  "stage": 2,
  "expert": "{expert_name}",
  "max_steps": {tier_info['max_steps']},
  "gold_criteria": "what makes a good response"
}}

No markdown fences.""",
                }
            ],
            temperature=1.0,
            max_tokens=4096,
        )
        text = resp.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        all_tasks.extend(json.loads(text))

    random.shuffle(all_tasks)
    return all_tasks


def generate_stage3_tasks(
    expert_name: str,
    existing_outputs_path: Path,
    count: int,
    client: OpenAI,
    model: str,
) -> list[dict]:
    """Stage 3: Generate self-improvement tasks.

    Takes existing model outputs and asks the teacher to create
    'critique + improve' pairs. The model learns to identify and fix
    weaknesses in its own outputs.
    """
    existing_outputs = []
    if existing_outputs_path.exists():
        with jsonlines.open(existing_outputs_path) as reader:
            for item in reader:
                existing_outputs.append(item)

    if not existing_outputs:
        print(f"  [WARN] No existing outputs found at {existing_outputs_path}, "
              f"generating synthetic examples for Stage 3", file=sys.stderr)
        # Fall back to generating synthetic examples to critique
        return generate_stage2_tasks(expert_name, count, client, model)

    tasks = []
    samples = random.sample(existing_outputs, min(count, len(existing_outputs)))

    for sample in samples:
        messages = sample.get("messages", [])
        if len(messages) < 2:
            continue
        original_input = messages[0].get("content", "")
        original_output = messages[1].get("content", "")

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"""You are creating self-improvement training data for the "{expert_name}" expert.

Here is an example the model previously produced:

INPUT: {original_input[:1000]}

OUTPUT: {original_output[:2000]}

Create a training example where the model must:
1. Identify weaknesses in this output
2. Produce an improved version

Return a JSON object:
{{
  "task": "the original input + instruction to improve",
  "difficulty": "medium",
  "stage": 3,
  "expert": "{expert_name}",
  "critique": "what was wrong with the original",
  "improved_output": "the better version"
}}

No markdown fences.""",
                }
            ],
            temperature=0.7,
            max_tokens=4096,
        )
        text = resp.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        try:
            tasks.append(json.loads(text))
        except json.JSONDecodeError:
            continue

    return tasks


def build_curriculum(
    expert_name: str,
    stages: list[int],
    count_per_stage: int,
    output_dir: Path,
    existing_outputs_path: Path | None,
    codegate_url: str,
    api_key: str,
    model: str,
):
    """Build a multi-stage curriculum for an expert."""
    client = OpenAI(base_url=f"{codegate_url}/v1", api_key=api_key)
    output_dir.mkdir(parents=True, exist_ok=True)

    for stage in stages:
        print(f"\n=== Stage {stage}: {STAGE_DESCRIPTIONS[stage]} ===")

        if stage == 1:
            tasks = generate_stage1_tasks(expert_name, count_per_stage, client, model)
        elif stage == 2:
            tasks = generate_stage2_tasks(expert_name, count_per_stage, client, model)
        elif stage == 3:
            ep = existing_outputs_path or (output_dir / f"stage1_{expert_name}.jsonl")
            tasks = generate_stage3_tasks(expert_name, ep, count_per_stage, client, model)
        else:
            print(f"  Unknown stage {stage}, skipping")
            continue

        out_path = output_dir / f"stage{stage}_{expert_name}.jsonl"
        with jsonlines.open(out_path, mode="w") as writer:
            for task in tasks:
                writer.write(task)

        print(f"  Generated {len(tasks)} tasks -> {out_path}")

    print(f"\nCurriculum complete for {expert_name}")


def main():
    parser = argparse.ArgumentParser(description="Build staged training curriculum for a swarm expert")
    parser.add_argument("--expert", required=True, help="Expert name")
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3],
                        help="Which stages to generate (1, 2, 3)")
    parser.add_argument("--count", type=int, default=50,
                        help="Number of tasks per stage")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: datasets/curriculum/)")
    parser.add_argument("--existing-outputs", type=str, default=None,
                        help="Path to existing model outputs for Stage 3")
    parser.add_argument("--codegate-url", type=str,
                        default=os.environ.get("CODEGATE_URL", "http://localhost:9212"))
    parser.add_argument("--api-key", type=str,
                        default=os.environ.get("CODEGATE_API_KEY", "cgk_xxx"))
    parser.add_argument("--model", type=str,
                        default=os.environ.get("TEACHER_MODEL", "claude-sonnet-4-20250514"))
    args = parser.parse_args()

    dataset_dir = Path(os.environ.get("DATASET_DIR", "datasets"))
    output_dir = Path(args.output_dir) if args.output_dir else dataset_dir / "curriculum"
    existing = Path(args.existing_outputs) if args.existing_outputs else None

    build_curriculum(
        expert_name=args.expert,
        stages=args.stages,
        count_per_stage=args.count,
        output_dir=output_dir,
        existing_outputs_path=existing,
        codegate_url=args.codegate_url,
        api_key=args.api_key,
        model=args.model,
    )


if __name__ == "__main__":
    main()
