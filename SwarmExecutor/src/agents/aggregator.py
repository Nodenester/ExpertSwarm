"""
Aggregator agents — cross-reference, compress, and map relationships.

Aggregators sit in later waves and combine results from multiple workers.
They deduplicate facts, find contradictions, and build a coherent picture
from scattered findings.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class AggregatorAgent(BaseAgent):
    """
    Cross-references and merges results from multiple upstream agents.

    Key responsibilities:
    - Deduplicate overlapping findings
    - Identify contradictions between sources
    - Map relationships between entities
    - Compress verbose results into dense summaries
    """

    def build_messages(self, context: str, dependency_results: dict[str, Any]) -> list[dict[str, Any]]:
        system = self.node.system_prompt or (
            "You are a research aggregator. You receive findings from multiple research agents "
            "and must:\n"
            "1. Identify and merge duplicate or overlapping information\n"
            "2. Flag any contradictions between sources\n"
            "3. Map relationships between entities and concepts\n"
            "4. Produce a unified, structured summary\n\n"
            "Output JSON with: facts (deduplicated), cross_references, themes, gaps, and summary."
        )

        # Build a structured view of all dependency results
        dep_sections = []
        for agent_id, result in dependency_results.items():
            if isinstance(result, dict):
                result_str = json.dumps(result, ensure_ascii=False, default=str)
            else:
                result_str = str(result)
            dep_sections.append(f"=== From {agent_id} ===\n{result_str}")

        dep_context = "\n\n".join(dep_sections) if dep_sections else "No upstream results."

        user_content = self.node.task_prompt
        if context:
            # Context builder already formatted the deps — use that
            user_content = f"Processed findings:\n{context}\n\n---\n\n{user_content}"
        else:
            user_content = f"Raw findings:\n{dep_context}\n\n---\n\n{user_content}"

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

    def parse_result(self, raw_output: str, dependency_results: dict[str, Any]) -> Any:
        """Parse aggregator output into structured aggregation result."""
        try:
            data = json.loads(raw_output)
            return data
        except json.JSONDecodeError:
            # Return as structured text even if not JSON
            return {
                "summary": raw_output,
                "facts": [],
                "cross_references": [],
                "themes": [],
                "gaps": [],
            }
