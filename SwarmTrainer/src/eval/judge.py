"""LLM-as-judge — routes through CodeGate to a big model for scoring.

Scores three dimensions:
- Correctness (0/1): Did the output achieve the task goal?
- Quality (0-1): How good is the output overall?
- Efficiency (0-1): Step penalty — fewer steps = higher score
"""

import json
import os
import sys

from openai import OpenAI


class Judge:
    """LLM judge that scores task outputs using a teacher model via CodeGate."""

    def __init__(
        self,
        codegate_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.codegate_url = codegate_url or os.environ.get("CODEGATE_URL", "http://localhost:9212")
        self.api_key = api_key or os.environ.get("CODEGATE_API_KEY", "cgk_xxx")
        self.model = model or os.environ.get("TEACHER_MODEL", "claude-sonnet-4-20250514")
        self.client = OpenAI(base_url=f"{self.codegate_url}/v1", api_key=self.api_key)

    def score_output(self, task: dict, output: dict) -> dict:
        """Score a task output on correctness, quality, and efficiency.

        Args:
            task: The original task description
            output: The model's output (can be a full trajectory or single response)

        Returns:
            Dict with correctness, quality, efficiency, and composite scores
        """
        task_str = json.dumps(task, default=str)[:2000]
        output_str = json.dumps(output, default=str)[:4000]

        # Count steps if this is a trajectory
        steps = 1
        if isinstance(output, dict):
            if "steps" in output:
                steps = len(output["steps"])
            elif "trajectory" in output:
                steps = len(output["trajectory"])

        prompt = f"""You are a strict evaluator scoring an AI system's output on a task.

TASK:
{task_str}

OUTPUT ({steps} step(s)):
{output_str}

Score on exactly three dimensions. Be strict but fair.

1. **correctness** (integer, 0 or 1): Did the output actually accomplish the task goal?
   - 1 = the core objective was met
   - 0 = the objective was not met, output is wrong, or there was an error

2. **quality** (float, 0.0 to 1.0): Overall quality of the output.
   - 1.0 = exceptional, detailed, well-structured, no issues
   - 0.7-0.9 = good quality, minor issues
   - 0.4-0.6 = acceptable but significant room for improvement
   - 0.0-0.3 = poor quality, major issues

3. **efficiency** (float, 0.0 to 1.0): Was the work done in a reasonable number of steps?
   - 1.0 = optimal, minimal steps needed
   - 0.7-0.9 = slightly more steps than necessary
   - 0.4-0.6 = noticeably wasteful
   - 0.0-0.3 = extremely wasteful, many unnecessary steps

Return ONLY a JSON object with these three keys. No explanation, no markdown fences."""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
            )
            text = resp.choices[0].message.content.strip()

            # Handle markdown fences if model ignores instruction
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            scores = json.loads(text)

            # Validate and clamp
            correctness = int(scores.get("correctness", 0))
            correctness = max(0, min(1, correctness))

            quality = float(scores.get("quality", 0))
            quality = max(0.0, min(1.0, quality))

            efficiency = float(scores.get("efficiency", 0))
            efficiency = max(0.0, min(1.0, efficiency))

            # Composite score (weighted average)
            composite = 0.5 * correctness + 0.3 * quality + 0.2 * efficiency

            return {
                "correctness": correctness,
                "quality": quality,
                "efficiency": efficiency,
                "composite": composite,
            }

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"  [WARN] Judge parse error: {e}", file=sys.stderr)
            return {"correctness": 0, "quality": 0.0, "efficiency": 0.0, "composite": 0.0}

        except Exception as e:
            print(f"  [WARN] Judge API error: {e}", file=sys.stderr)
            return {"correctness": 0, "quality": 0.0, "efficiency": 0.0, "composite": 0.0}

    def score_batch(self, tasks: list[dict], outputs: list[dict]) -> list[dict]:
        """Score a batch of task outputs."""
        return [self.score_output(t, o) for t, o in zip(tasks, outputs)]

    def compare_outputs(self, task: dict, output_a: dict, output_b: dict) -> dict:
        """Compare two outputs for the same task (pairwise comparison).

        Returns which output is better and why.
        """
        task_str = json.dumps(task, default=str)[:2000]
        a_str = json.dumps(output_a, default=str)[:3000]
        b_str = json.dumps(output_b, default=str)[:3000]

        prompt = f"""Compare two AI outputs for the same task.

TASK:
{task_str}

OUTPUT A:
{a_str}

OUTPUT B:
{b_str}

Which output is better? Return a JSON object:
{{
  "winner": "A" or "B" or "tie",
  "reason": "brief explanation",
  "score_a": 0.0-1.0,
  "score_b": 0.0-1.0
}}"""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
            )
            text = resp.choices[0].message.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            return json.loads(text)
        except Exception as e:
            print(f"  [WARN] Comparison error: {e}", file=sys.stderr)
            return {"winner": "tie", "reason": f"Error: {e}", "score_a": 0.5, "score_b": 0.5}
