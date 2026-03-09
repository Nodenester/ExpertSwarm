"""
Synthesizer agents — final synthesis and answer generation.

Synthesizers are the last wave. They take aggregated, compressed findings
and produce the final answer/report/analysis for the user.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class SynthesizerAgent(BaseAgent):
    """
    Produces the final output from aggregated findings.

    The synthesizer receives already-compressed, cross-referenced data
    and turns it into a polished final answer. It should NOT need to
    do any research itself — all the work is done by earlier waves.
    """

    def build_messages(self, context: str, dependency_results: dict[str, Any]) -> list[dict[str, Any]]:
        system = self.node.system_prompt or (
            "You are a research synthesizer. You receive aggregated findings from a team of "
            "research agents and must produce a final, comprehensive answer. Your output should "
            "be well-structured, cite sources where possible, and directly address the original query.\n\n"
            "Be thorough but concise. Focus on actionable insights and clear conclusions."
        )

        user_content = self.node.task_prompt
        if context:
            user_content = f"Aggregated research findings:\n{context}\n\n---\n\n{user_content}"

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

    def parse_result(self, raw_output: str, dependency_results: dict[str, Any]) -> Any:
        """
        Parse synthesizer output. Can be structured JSON or plain text
        depending on the swarm config.
        """
        if self.node.output_schema:
            try:
                return json.loads(raw_output)
            except json.JSONDecodeError:
                pass

        # For synthesis, plain text is often the desired output
        return {"answer": raw_output, "agent_id": self.agent_id}
