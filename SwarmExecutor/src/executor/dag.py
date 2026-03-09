"""
DAG builder and dependency resolution for swarm execution plans.

Builds a Directed Acyclic Graph from a list of agent plan nodes,
performs topological sort into execution waves, and tracks completion.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Any

from src.schemas.plan_schema import (
    AgentPlanNode,
    AgentStatus,
    ExecutionPlan,
    ExecutionWave,
)

logger = logging.getLogger(__name__)


class DAGBuilder:
    """
    Build and manage the execution DAG from a list of agents with dependencies.

    Usage:
        dag = DAGBuilder(agents)
        waves = dag.build_waves()
        # waves[0] = agents with no deps, waves[1] = depends only on wave 0, etc.
    """

    def __init__(self, agents: dict[str, AgentPlanNode]) -> None:
        self._agents = agents
        self._adjacency: dict[str, list[str]] = defaultdict(list)
        self._in_degree: dict[str, int] = {}
        self._status: dict[str, AgentStatus] = {}

        self._build_graph()

    def _build_graph(self) -> None:
        """Build adjacency list and in-degree map from agent dependencies."""
        for agent_id in self._agents:
            self._in_degree.setdefault(agent_id, 0)
            self._status[agent_id] = AgentStatus.PENDING

        for agent_id, agent in self._agents.items():
            for dep_id in agent.depends_on:
                if dep_id not in self._agents:
                    raise ValueError(
                        f"Agent {agent_id!r} depends on unknown agent {dep_id!r}. "
                        f"Available: {list(self._agents.keys())}"
                    )
                self._adjacency[dep_id].append(agent_id)
                self._in_degree[agent_id] = self._in_degree.get(agent_id, 0) + 1

    def detect_cycle(self) -> list[str] | None:
        """Return a cycle path if one exists, else None."""
        visited: set[str] = set()
        rec_stack: set[str] = set()
        path: list[str] = []

        def _dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self._adjacency.get(node, []):
                if neighbor not in visited:
                    if _dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    path.append(neighbor)
                    return True

            path.pop()
            rec_stack.discard(node)
            return False

        for agent_id in self._agents:
            if agent_id not in visited:
                if _dfs(agent_id):
                    # Extract just the cycle portion
                    cycle_start = path[-1]
                    cycle_idx = path.index(cycle_start)
                    return path[cycle_idx:]
        return None

    def build_waves(self) -> list[ExecutionWave]:
        """
        Topological sort into parallel execution waves using Kahn's algorithm.

        Wave 0: all agents with no dependencies (in-degree 0)
        Wave 1: agents whose deps are all in wave 0
        Wave N: agents whose deps are all in waves < N

        Raises ValueError if a cycle is detected.
        """
        cycle = self.detect_cycle()
        if cycle:
            raise ValueError(f"Cycle detected in agent DAG: {' -> '.join(cycle)}")

        in_degree = dict(self._in_degree)
        waves: list[ExecutionWave] = []

        # Start with all zero in-degree nodes
        queue = deque(aid for aid, deg in in_degree.items() if deg == 0)

        wave_index = 0
        while queue:
            wave_agents = list(queue)
            waves.append(ExecutionWave(wave_index=wave_index, agent_ids=wave_agents))

            next_queue: deque[str] = deque()
            for agent_id in wave_agents:
                for downstream in self._adjacency.get(agent_id, []):
                    in_degree[downstream] -= 1
                    if in_degree[downstream] == 0:
                        next_queue.append(downstream)

            queue = next_queue
            wave_index += 1

        # Verify all agents are scheduled
        scheduled = {aid for w in waves for aid in w.agent_ids}
        missing = set(self._agents.keys()) - scheduled
        if missing:
            raise ValueError(f"Could not schedule agents (possible cycle): {missing}")

        logger.info("Built %d execution waves for %d agents", len(waves), len(self._agents))
        return waves

    def mark_completed(self, agent_id: str) -> None:
        self._status[agent_id] = AgentStatus.COMPLETED

    def mark_failed(self, agent_id: str) -> None:
        self._status[agent_id] = AgentStatus.FAILED

    def mark_running(self, agent_id: str) -> None:
        self._status[agent_id] = AgentStatus.RUNNING

    def get_status(self, agent_id: str) -> AgentStatus:
        return self._status.get(agent_id, AgentStatus.PENDING)

    def all_statuses(self) -> dict[str, AgentStatus]:
        return dict(self._status)

    def get_ready_agents(self) -> list[str]:
        """Return agents whose dependencies are all completed."""
        ready = []
        for agent_id, agent in self._agents.items():
            if self._status[agent_id] != AgentStatus.PENDING:
                continue
            if all(self._status.get(dep) == AgentStatus.COMPLETED for dep in agent.depends_on):
                ready.append(agent_id)
        return ready

    def get_dependents(self, agent_id: str) -> list[str]:
        """Return agents that depend on the given agent."""
        return list(self._adjacency.get(agent_id, []))

    def get_dependencies(self, agent_id: str) -> list[str]:
        """Return agents that the given agent depends on."""
        agent = self._agents.get(agent_id)
        return list(agent.depends_on) if agent else []


def build_execution_plan(
    task_id: str,
    swarm_type: str,
    agents: dict[str, AgentPlanNode],
    metadata: dict[str, Any] | None = None,
) -> ExecutionPlan:
    """
    Build a complete execution plan with topologically sorted waves.

    This is the main entry point for DAG construction.
    """
    dag = DAGBuilder(agents)
    waves = dag.build_waves()

    return ExecutionPlan(
        task_id=task_id,
        swarm_type=swarm_type,
        agents=agents,
        waves=waves,
        metadata=metadata or {},
    )
