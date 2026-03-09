"""
Scout agents — query generation and pattern discovery.

Scouts are early-wave agents that generate search queries, identify patterns,
and produce exploration plans for workers to execute.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class ScoutAgent(BaseAgent):
    """
    Generates search queries, URL patterns, or code search patterns.

    Scout outputs are consumed by worker agents to do the actual
    searching/scraping/analysis. This two-step pattern keeps each
    agent's job small and focused.
    """

    def build_messages(self, context: str, dependency_results: dict[str, Any]) -> list[dict[str, Any]]:
        system = self.node.system_prompt or (
            "You are a research scout. Your job is to generate effective search queries "
            "and identify patterns for investigation. Output structured JSON with your "
            "queries and reasoning."
        )

        user_content = self.node.task_prompt
        if context:
            user_content = f"Prior findings:\n{context}\n\n---\n\nYour task:\n{user_content}"

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

    def parse_result(self, raw_output: str, dependency_results: dict[str, Any]) -> Any:
        """
        Parse scout output. Expected formats:
        - {"queries": [...], "patterns": [...]}
        - {"urls": [...], "selectors": [...]}
        - {"search_terms": [...], "file_patterns": [...]}
        """
        try:
            data = json.loads(raw_output)
            return data
        except json.JSONDecodeError:
            # Extract useful content even from unstructured output
            lines = [l.strip() for l in raw_output.strip().split("\n") if l.strip()]
            queries = []
            for line in lines:
                cleaned = line.lstrip("- *0123456789.)")
                if cleaned:
                    queries.append(cleaned.strip())
            return {"queries": queries, "raw": raw_output}
