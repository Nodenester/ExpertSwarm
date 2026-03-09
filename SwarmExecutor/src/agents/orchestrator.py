"""
Orchestrator agent — plans the DAG and controls execution flow.

The orchestrator is the first agent to run. It takes the user's query and
the swarm template, then produces an execution plan: which agents to run,
what tools they need, and how they depend on each other.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.agents.base import BaseAgent
from src.schemas.plan_schema import AgentPlanNode

logger = logging.getLogger(__name__)


class OrchestratorAgent(BaseAgent):
    """
    Plans the execution DAG based on the user query and swarm template.

    The orchestrator:
    1. Receives the user query + swarm template (available agents, tools)
    2. Decides which agents to activate and their dependencies
    3. Generates task prompts for each agent
    4. Returns a structured plan that the DAG executor can run
    """

    def build_messages(self, context: str, dependency_results: dict[str, Any]) -> list[dict[str, Any]]:
        system = self.node.system_prompt or (
            "You are a swarm orchestrator. Given a user query and available agent types, "
            "plan an execution DAG. Output a JSON object with an 'agents' array. Each agent "
            "needs: agent_id, role, lora_name, task_prompt, depends_on (list of agent_ids), "
            "and tools (list of tool names)."
        )

        user_content = self.node.task_prompt
        if context:
            user_content = f"Available context from previous steps:\n{context}\n\n---\n\n{user_content}"

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

    def parse_result(self, raw_output: str, dependency_results: dict[str, Any]) -> Any:
        """Parse the orchestrator's plan output into structured agent nodes."""
        try:
            data = json.loads(raw_output)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            import re
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_output, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                except json.JSONDecodeError:
                    logger.error("Orchestrator output is not valid JSON: %s", raw_output[:200])
                    return {"error": "Invalid plan output", "raw": raw_output}
            else:
                return {"error": "Invalid plan output", "raw": raw_output}

        # Validate and normalize the plan
        agents_data = data.get("agents", [])
        if isinstance(agents_data, list):
            plan = {}
            for agent_data in agents_data:
                agent_id = agent_data.get("agent_id", f"agent_{len(plan)}")
                plan[agent_id] = {
                    "agent_id": agent_id,
                    "role": agent_data.get("role", "worker"),
                    "lora_name": agent_data.get("lora_name", "default"),
                    "task_prompt": agent_data.get("task_prompt", ""),
                    "depends_on": agent_data.get("depends_on", []),
                    "tools": agent_data.get("tools", []),
                }
            return {"agents": plan}

        return data
