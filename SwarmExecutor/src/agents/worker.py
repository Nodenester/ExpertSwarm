"""
Worker agents — the hands of the swarm.

Workers do the actual work: picking URLs, scraping pages, extracting facts,
reading files, analyzing code. They use tools and produce structured results.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class WorkerAgent(BaseAgent):
    """
    General-purpose worker agent. Specialization comes from:
    - The LoRA adapter (e.g., url-picker, css-selector, fact-extractor)
    - The tools assigned in the swarm config
    - The task prompt from the orchestrator

    Worker subtypes are distinguished by their lora_name and tools,
    not by subclassing. This keeps the system simple and extensible
    through swarm YAML configs rather than code changes.
    """

    def build_messages(self, context: str, dependency_results: dict[str, Any]) -> list[dict[str, Any]]:
        system = self.node.system_prompt or (
            "You are a research worker. Use your available tools to complete the assigned task. "
            "Be thorough but focused. Return structured results."
        )

        # Inject tool descriptions into the system prompt
        if self.tools:
            from src.tools.registry import tool_registry
            tool_list = []
            for tool_name in self.tools:
                entry = tool_registry.get_entry(tool_name)
                if entry:
                    tool_list.append(f"- {entry.name}: {entry.description}")
            if tool_list:
                system += "\n\nAvailable tools:\n" + "\n".join(tool_list)

        user_content = self.node.task_prompt
        if context:
            user_content = f"Context from previous agents:\n{context}\n\n---\n\nYour task:\n{user_content}"

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

    def parse_result(self, raw_output: str, dependency_results: dict[str, Any]) -> Any:
        """
        Parse worker output. Workers should produce structured JSON when
        an output_schema is set, otherwise raw text is fine.
        """
        if self.node.output_schema:
            try:
                return json.loads(raw_output)
            except json.JSONDecodeError:
                logger.warning("Worker %s did not produce valid JSON", self.agent_id)

        # Try JSON parsing as best-effort
        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            return {"content": raw_output}
