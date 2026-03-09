"""
Async wave executor — runs agents in parallel waves using asyncio.gather().

Executes the DAG plan wave by wave:
1. For each wave, launch all agents in parallel
2. Collect results from each agent
3. Feed results to dependent agents in subsequent waves
4. Handle failures gracefully (skip dependents of failed agents)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator

from src.agents.base import BaseAgent
from src.executor.dag import DAGBuilder
from src.schemas.plan_schema import (
    AgentPlanNode,
    AgentStatus,
    ExecutionPlan,
    TaskState,
    TaskStatus,
)

logger = logging.getLogger(__name__)


class AgentEvent:
    """Event emitted during execution for SSE streaming."""

    __slots__ = ("agent_id", "event_type", "data", "timestamp")

    def __init__(self, agent_id: str, event_type: str, data: Any = None):
        self.agent_id = agent_id
        self.event_type = event_type  # "started", "completed", "failed", "wave_start", "wave_end"
        self.data = data
        self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "event": self.event_type,
            "data": self.data,
            "timestamp": self.timestamp,
        }


class WaveExecutor:
    """
    Execute a DAG plan wave-by-wave with parallel agent execution.

    Usage:
        executor = WaveExecutor(plan, agent_factory)
        state = await executor.run()

        # Or stream events:
        async for event in executor.run_streaming():
            print(event.to_dict())
    """

    def __init__(
        self,
        plan: ExecutionPlan,
        agent_factory: AgentFactory,
        max_parallel: int = 50,
    ) -> None:
        self._plan = plan
        self._factory = agent_factory
        self._max_parallel = max_parallel
        self._dag = DAGBuilder(plan.agents)
        self._results: dict[str, Any] = {}
        self._events: list[AgentEvent] = []
        self._event_queue: asyncio.Queue[AgentEvent] = asyncio.Queue()

    def _emit(self, agent_id: str, event_type: str, data: Any = None) -> None:
        event = AgentEvent(agent_id, event_type, data)
        self._events.append(event)
        self._event_queue.put_nowait(event)

    async def _execute_agent(self, agent_node: AgentPlanNode) -> Any:
        """Execute a single agent with its dependency results as context."""
        agent_id = agent_node.agent_id
        self._dag.mark_running(agent_id)
        self._emit(agent_id, "started")

        # Gather dependency results for this agent
        dep_results = {}
        for dep_id in agent_node.depends_on:
            if dep_id in self._results:
                dep_results[dep_id] = self._results[dep_id]

        try:
            agent = self._factory.create(agent_node)
            result = await agent.run(dep_results)
            self._results[agent_id] = result
            self._dag.mark_completed(agent_id)
            self._emit(agent_id, "completed", data={"result_preview": str(result)[:500]})
            logger.info("Agent %s completed", agent_id)
            return result

        except Exception as exc:
            logger.error("Agent %s failed: %s", agent_id, exc, exc_info=True)
            self._dag.mark_failed(agent_id)
            self._results[agent_id] = {"error": str(exc)}
            self._emit(agent_id, "failed", data={"error": str(exc)})
            return None

    async def _execute_wave(self, wave_index: int, agent_ids: list[str]) -> None:
        """Execute all agents in a wave in parallel using asyncio.gather()."""
        self._emit("system", "wave_start", data={"wave": wave_index, "agents": agent_ids})
        logger.info("Starting wave %d with %d agents: %s", wave_index, len(agent_ids), agent_ids)

        # Filter out agents whose dependencies failed
        runnable = []
        for agent_id in agent_ids:
            agent_node = self._plan.agents[agent_id]
            deps_ok = all(
                self._dag.get_status(dep) == AgentStatus.COMPLETED
                for dep in agent_node.depends_on
            )
            if deps_ok:
                runnable.append(agent_node)
            else:
                logger.warning("Skipping %s — dependency failed", agent_id)
                self._dag.mark_failed(agent_id)
                self._emit(agent_id, "skipped", data={"reason": "dependency_failed"})

        # Execute in parallel, respecting max_parallel
        semaphore = asyncio.Semaphore(self._max_parallel)

        async def _bounded_execute(node: AgentPlanNode) -> Any:
            async with semaphore:
                return await self._execute_agent(node)

        await asyncio.gather(*[_bounded_execute(node) for node in runnable])
        self._emit("system", "wave_end", data={"wave": wave_index})

    async def run(self) -> TaskState:
        """Execute the full plan and return final state."""
        state = TaskState(
            task_id=self._plan.task_id,
            swarm_type=self._plan.swarm_type,
            status=TaskStatus.EXECUTING,
            plan=self._plan,
        )

        try:
            for wave in self._plan.waves:
                await self._execute_wave(wave.wave_index, wave.agent_ids)

            state.agent_statuses = self._dag.all_statuses()
            state.agent_results = self._results
            state.status = TaskStatus.COMPLETED

            # Check if any agents failed
            failed = [aid for aid, s in state.agent_statuses.items() if s == AgentStatus.FAILED]
            if failed:
                logger.warning("Task %s completed with %d failed agents: %s", state.task_id, len(failed), failed)

        except Exception as exc:
            logger.error("Execution failed: %s", exc, exc_info=True)
            state.status = TaskStatus.FAILED
            state.error = str(exc)

        return state

    async def run_streaming(self) -> AsyncIterator[AgentEvent]:
        """Execute the plan and yield events as they happen."""
        task = asyncio.create_task(self.run())

        while not task.done() or not self._event_queue.empty():
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=0.5)
                yield event
            except asyncio.TimeoutError:
                continue

        # Yield any remaining events
        while not self._event_queue.empty():
            yield self._event_queue.get_nowait()

        # Propagate any exception from the main task
        task.result()


class AgentFactory:
    """
    Creates agent instances from plan nodes. Uses the swarm config's
    agent role to pick the right agent class.
    """

    def __init__(self, agent_classes: dict[str, type[BaseAgent]] | None = None) -> None:
        from src.agents.aggregator import AggregatorAgent
        from src.agents.orchestrator import OrchestratorAgent
        from src.agents.scout import ScoutAgent
        from src.agents.synthesizer import SynthesizerAgent
        from src.agents.worker import WorkerAgent

        self._classes: dict[str, type[BaseAgent]] = agent_classes or {
            "orchestrator": OrchestratorAgent,
            "scout": ScoutAgent,
            "worker": WorkerAgent,
            "aggregator": AggregatorAgent,
            "synthesizer": SynthesizerAgent,
        }

    def create(self, node: AgentPlanNode) -> BaseAgent:
        cls = self._classes.get(node.role.value)
        if cls is None:
            raise ValueError(f"No agent class for role: {node.role.value}")
        return cls(node)
