"""SFT data generator — creates training examples using a teacher model via CodeGate proxy.

Batched + parallel: each API call generates ~10 examples. 300 examples in ~30 API calls.
Writes to disk as batches complete so you can see progress.
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

import jsonlines
from openai import AsyncOpenAI

EXPERT_TOOLS = {
    "query_gen": {
        "role": "search_query_generator",
        "description": "Generate optimal search queries for a given research question.",
        "output_schema": {
            "queries": ["list of search query strings"],
            "reasoning": "why these queries will find relevant results",
        },
    },
    "fact_extractor": {
        "role": "fact_extractor",
        "description": "Extract structured facts from raw text content.",
        "output_schema": {
            "facts": [{"claim": "str", "source": "str", "confidence": "float 0-1"}],
            "summary": "brief summary of extracted information",
        },
    },
    "code_reader": {
        "role": "code_reader",
        "description": "Read and analyze source code, extracting structure and meaning.",
        "output_schema": {
            "language": "str",
            "purpose": "str",
            "functions": [{"name": "str", "signature": "str", "description": "str"}],
            "dependencies": ["list of imports/deps"],
            "issues": ["potential bugs or improvements"],
        },
    },
    "url_picker": {
        "role": "url_picker",
        "description": "Select the most relevant URLs from a list based on a research goal.",
        "output_schema": {
            "selected": [{"url": "str", "reason": "str", "priority": "int 1-5"}],
            "rejected_reason": "why other URLs were skipped",
        },
    },
    "css_selector": {
        "role": "css_selector",
        "description": "Generate CSS selectors to extract specific data from HTML.",
        "output_schema": {
            "selectors": [{"target": "str", "selector": "str", "expected_type": "str"}],
            "fallback_strategy": "what to do if selectors fail",
        },
    },
    "orchestrator_web": {
        "role": "web_orchestrator",
        "description": "Plan and coordinate a multi-step web research task.",
        "output_schema": {
            "plan": [{"step": "int", "action": "str", "expert": "str", "input": "str"}],
            "success_criteria": "how to know the task is complete",
            "max_steps": "int",
        },
    },
    "orchestrator_code": {
        "role": "code_orchestrator",
        "description": "Plan and coordinate a multi-step code analysis task.",
        "output_schema": {
            "plan": [{"step": "int", "action": "str", "expert": "str", "input": "str"}],
            "success_criteria": "how to know the task is complete",
            "max_steps": "int",
        },
    },
    "cross_ref": {
        "role": "cross_referencer",
        "description": "Cross-reference multiple sources to verify and consolidate facts.",
        "output_schema": {
            "verified_facts": [{"claim": "str", "sources": ["str"], "agreement": "float 0-1"}],
            "conflicts": [{"claim": "str", "source_a": "str", "source_b": "str"}],
            "confidence": "float 0-1",
        },
    },
    "compressor": {
        "role": "compressor",
        "description": "Compress verbose information into concise, high-signal summaries.",
        "output_schema": {
            "compressed": "str — concise output",
            "tokens_saved_pct": "float — estimated compression ratio",
            "key_points_preserved": ["list of preserved key points"],
        },
    },
    "synthesizer": {
        "role": "synthesizer",
        "description": "Synthesize information from multiple sources into a coherent answer.",
        "output_schema": {
            "answer": "str — final synthesized answer",
            "sources_used": ["list of source references"],
            "confidence": "float 0-1",
            "caveats": ["any important caveats or limitations"],
        },
    },
}


def load_template(expert_name: str, templates_dir: Path) -> str:
    template_file = templates_dir / f"{expert_name}.txt"
    if not template_file.exists():
        raise FileNotFoundError(f"Template not found: {template_file}")
    return template_file.read_text(encoding="utf-8")


async def api_call_with_retry(coro_factory, max_retries=5, base_delay=2.0):
    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except Exception as e:
            if "429" in str(e) and attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                await asyncio.sleep(delay)
            else:
                raise


async def generate_batch(
    expert_name: str,
    template: str,
    batch_size: int,
    batch_num: int,
    total_batches: int,
    client: AsyncOpenAI,
    model: str,
    semaphore: asyncio.Semaphore,
    output_path: Path,
    lock: asyncio.Lock,
    counter: dict,
) -> int:
    """Generate a batch of examples in ONE api call, write to disk immediately."""
    expert_info = EXPERT_TOOLS[expert_name]

    prompt = f"""You are generating training data for a small language model that will serve as the "{expert_info['role']}" expert in an AI swarm.

The expert's job: {expert_info['description']}

Expected output JSON schema:
{json.dumps(expert_info['output_schema'], indent=2)}

TEMPLATE INSTRUCTIONS:
{template}

Generate {batch_size} diverse, realistic training examples. Vary complexity from simple to hard.

Return a JSON array of objects, each with exactly two keys:
- "input": the user/orchestrator message this expert would receive (include realistic source text, data, or context)
- "output": the expert's ideal response as a JSON object conforming to the schema above

Make inputs detailed and realistic — include actual text passages, HTML snippets, data excerpts, etc. that the expert would process.
Return ONLY the JSON array, no markdown fences, no extra text."""

    async with semaphore:
        try:
            async def _call():
                return await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.9,
                    max_tokens=16384,
                )

            resp = await api_call_with_retry(_call)
            text = resp.choices[0].message.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            examples = json.loads(text)
            if not isinstance(examples, list):
                examples = [examples]

            # Format for training and write to disk immediately
            formatted = []
            for ex in examples:
                if "input" not in ex or "output" not in ex:
                    continue
                output = ex["output"]
                if isinstance(output, dict):
                    output = json.dumps(output)
                formatted.append({
                    "expert": expert_name,
                    "messages": [
                        {"role": "user", "content": ex["input"]},
                        {"role": "assistant", "content": output},
                    ],
                })

            # Write to disk under lock
            async with lock:
                with jsonlines.open(output_path, mode="a") as writer:
                    for f in formatted:
                        writer.write(f)
                counter["done"] += len(formatted)
                print(f"  Batch {batch_num}/{total_batches}: +{len(formatted)} examples (total: {counter['done']})", flush=True)

            return len(formatted)

        except json.JSONDecodeError as e:
            print(f"  Batch {batch_num}/{total_batches}: JSON parse error: {e}", flush=True)
            return 0
        except Exception as e:
            print(f"  Batch {batch_num}/{total_batches}: Error: {e}", flush=True)
            return 0


async def run_async(
    expert_name: str,
    num_examples: int,
    output_path: Path,
    templates_dir: Path,
    codegate_url: str,
    api_key: str,
    model: str,
    concurrency: int = 10,
    batch_size: int = 10,
):
    """Generate SFT dataset using batched parallel API calls."""
    client = AsyncOpenAI(base_url=f"{codegate_url}/v1", api_key=api_key)
    template = load_template(expert_name, templates_dir)

    total_batches = (num_examples + batch_size - 1) // batch_size
    print(f"Generating {num_examples} examples for {expert_name}")
    print(f"  {total_batches} batches of ~{batch_size}, concurrency={concurrency}")
    print(f"  Output: {output_path}", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Count existing examples if appending
    existing = 0
    if output_path.exists():
        with open(output_path) as f:
            existing = sum(1 for _ in f)
        print(f"  Appending to existing {existing} examples", flush=True)

    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    counter = {"done": 0}
    start = time.time()

    tasks = []
    for i in range(total_batches):
        remaining = num_examples - i * batch_size
        bs = min(batch_size, remaining)
        tasks.append(generate_batch(
            expert_name, template, bs, i + 1, total_batches,
            client, model, semaphore, output_path, lock, counter,
        ))

    await asyncio.gather(*tasks)

    elapsed = time.time() - start
    print(f"\nDone. {counter['done']} examples in {elapsed:.1f}s -> {output_path}", flush=True)


def run(
    expert_name: str,
    num_examples: int,
    output_path: Path,
    templates_dir: Path,
    codegate_url: str,
    api_key: str,
    model: str,
    concurrency: int = 10,
    batch_size: int = 10,
):
    asyncio.run(run_async(
        expert_name, num_examples, output_path, templates_dir,
        codegate_url, api_key, model, concurrency, batch_size,
    ))


def main():
    parser = argparse.ArgumentParser(description="Generate SFT training data for a swarm expert")
    parser.add_argument("--expert", required=True, choices=list(EXPERT_TOOLS.keys()),
                        help="Expert to generate data for")
    parser.add_argument("--num-examples", type=int, default=300,
                        help="Number of training examples to generate")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL path (default: datasets/<expert>.jsonl)")
    parser.add_argument("--templates-dir", type=str,
                        default=str(Path(__file__).parent / "templates"),
                        help="Directory containing prompt templates")
    parser.add_argument("--codegate-url", type=str,
                        default=os.environ.get("CODEGATE_URL", "http://localhost:9212"))
    parser.add_argument("--api-key", type=str,
                        default=os.environ.get("CODEGATE_API_KEY", "proxy-local-key"))
    parser.add_argument("--model", type=str,
                        default=os.environ.get("TEACHER_MODEL", "claude-sonnet-4-20250514"))
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Max parallel API calls")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Examples per API call")
    args = parser.parse_args()

    dataset_dir = Path(os.environ.get("DATASET_DIR", "datasets"))
    output_path = Path(args.output) if args.output else dataset_dir / f"{args.expert}.jsonl"

    run(
        expert_name=args.expert,
        num_examples=args.num_examples,
        output_path=output_path,
        templates_dir=Path(args.templates_dir),
        codegate_url=args.codegate_url,
        api_key=args.api_key,
        model=args.model,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
