"""
Tree-reduction compression pipeline.

Takes N agent results and compresses them in rounds (tree-reduction style):
- Round 1: Group results into batches of `group_size`, compress each group
- Round 2: Group compressed outputs, compress again
- Repeat until a single compressed result remains

This lets us handle 50+ agent results without blowing the token budget
of the final synthesizer agent.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from src.config import config
from src.executor.context_builder import compress_result

logger = logging.getLogger(__name__)


async def _call_compression_expert(
    texts: list[str],
    task_context: str,
    runner_url: str,
) -> str:
    """
    Call SwarmRunner with a compression LoRA to merge multiple texts into one.
    Falls back to local compression if the runner is unavailable.
    """
    combined = "\n\n---\n\n".join(texts)

    prompt = (
        f"You are a compression expert. Merge these {len(texts)} research results into a single "
        f"coherent summary. Preserve all key facts, numbers, and conclusions. Remove redundancy.\n\n"
        f"Task context: {task_context}\n\n"
        f"Results to merge:\n{combined}"
    )

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{runner_url}/v1/chat/completions",
                json={
                    "model": "compressor",
                    "messages": [
                        {"role": "system", "content": "You merge and compress research results. Be thorough but concise."},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 2000,
                    "temperature": 0.3,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
    except (httpx.HTTPError, KeyError, IndexError) as exc:
        logger.warning("Compression expert unavailable, falling back to local compression: %s", exc)
        return _local_compress(texts, max_tokens=2000)


def _local_compress(texts: list[str], max_tokens: int = 2000) -> str:
    """Fallback local compression without LLM — truncate and merge."""
    per_text_budget = max(200, max_tokens // max(len(texts), 1))
    compressed_parts = []
    for i, text in enumerate(texts):
        part = compress_result(text, per_text_budget)
        compressed_parts.append(f"[{i + 1}] {part}")
    return "\n\n".join(compressed_parts)


async def tree_compress(
    results: dict[str, Any],
    task_context: str = "",
    group_size: int = 5,
    max_levels: int = 3,
    use_llm: bool = True,
) -> str:
    """
    Tree-reduction compression pipeline.

    Args:
        results: Dict of agent_id -> result
        task_context: Original task description for context
        group_size: How many results per compression group
        max_levels: Maximum compression rounds
        use_llm: Whether to use SwarmRunner's compression expert

    Returns:
        Single compressed string combining all results.
    """
    # Stringify all results
    texts: list[str] = []
    for agent_id, result in results.items():
        if isinstance(result, str):
            text = result
        elif isinstance(result, (dict, list)):
            text = json.dumps(result, ensure_ascii=False, default=str)
        else:
            text = str(result)
        texts.append(f"[{agent_id}] {text}")

    if not texts:
        return ""

    if len(texts) == 1:
        return texts[0]

    runner_url = config.SWARM_RUNNER_URL

    for level in range(max_levels):
        if len(texts) <= 1:
            break

        logger.info(
            "Compression round %d: %d texts -> ~%d groups",
            level + 1,
            len(texts),
            (len(texts) + group_size - 1) // group_size,
        )

        # Split into groups
        groups = [texts[i : i + group_size] for i in range(0, len(texts), group_size)]

        # Compress each group (in parallel)
        import asyncio

        async def _compress_group(group: list[str]) -> str:
            if len(group) == 1:
                return group[0]
            if use_llm:
                return await _call_compression_expert(group, task_context, runner_url)
            return _local_compress(group)

        texts = await asyncio.gather(*[_compress_group(g) for g in groups])
        texts = list(texts)

    # Final merge if still multiple
    if len(texts) > 1:
        return _local_compress(texts, max_tokens=4000)

    return texts[0]
