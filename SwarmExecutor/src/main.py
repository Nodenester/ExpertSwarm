"""
SwarmExecutor — FastAPI server for DAG-based swarm orchestration.

Endpoints:
  POST /api/v1/swarm/{swarm_type}  — Submit a task to any swarm type
  GET  /api/v1/swarms              — List available swarm types
  GET  /api/v1/tasks/{id}          — Get task status + results
  GET  /api/v1/tasks/{id}/stream   — SSE stream of agent progress
  GET  /health                     — Health check
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from src.config import config
from src.executor.compression import tree_compress
from src.executor.dag import build_execution_plan, DAGBuilder
from src.executor.runner import AgentFactory, WaveExecutor
from src.schemas.plan_schema import TaskState, TaskStatus
from src.swarms.loader import discover_swarms, SwarmTemplate

# Import tool modules to trigger registration
import src.tools.scrapling_tools  # noqa: F401
import src.tools.filesystem_tools  # noqa: F401
import src.tools.code_analysis  # noqa: F401
import src.tools.hivemind_tools  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SwarmExecutor",
    description="DAG-based swarm orchestration engine with parallel agent execution",
    version="1.0.0",
)

# Global state
_swarms: dict[str, SwarmTemplate] = {}
_tasks: dict[str, TaskState] = {}
_agent_factory = AgentFactory()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SwarmRequest(BaseModel):
    query: str = Field(..., description="The task/query for the swarm to process")
    config_overrides: dict[str, Any] = Field(default_factory=dict, description="Override swarm config values")


class SwarmResponse(BaseModel):
    task_id: str
    swarm_type: str
    status: str


class SwarmListItem(BaseModel):
    name: str
    description: str
    version: str
    agents: list[str]


class TaskResponse(BaseModel):
    task_id: str
    swarm_type: str
    status: str
    agent_statuses: dict[str, str]
    final_result: Any = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup() -> None:
    global _swarms
    _swarms = discover_swarms()
    logger.info("SwarmExecutor started. %d swarm types loaded.", len(_swarms))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "healthy", "service": "swarm-executor"}


@app.get("/api/v1/swarms", response_model=list[SwarmListItem])
async def list_swarms() -> list[SwarmListItem]:
    """List all available swarm types loaded from YAML configs."""
    items = []
    for name, template in _swarms.items():
        items.append(SwarmListItem(
            name=template.name,
            description=template.description,
            version=template.version,
            agents=list(template.agent_templates.keys()),
        ))
    return items


@app.post("/api/v1/swarm/{swarm_type}", response_model=SwarmResponse)
async def submit_task(swarm_type: str, request: SwarmRequest) -> SwarmResponse:
    """Submit a task to a swarm type. Returns immediately with task_id."""
    template = _swarms.get(swarm_type)
    if template is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown swarm type: {swarm_type!r}. Available: {list(_swarms.keys())}",
        )

    task_id = str(uuid.uuid4())

    # Create execution plan from template
    plan = template.create_plan(task_id, request.query)

    # Initialize task state
    state = TaskState(
        task_id=task_id,
        swarm_type=swarm_type,
        status=TaskStatus.QUEUED,
        query=request.query,
        plan=plan,
    )
    _tasks[task_id] = state

    # Launch execution in background
    asyncio.create_task(_run_task(task_id, template))

    return SwarmResponse(task_id=task_id, swarm_type=swarm_type, status="queued")


@app.get("/api/v1/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str) -> TaskResponse:
    """Get task status and results."""
    state = _tasks.get(task_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    return TaskResponse(
        task_id=state.task_id,
        swarm_type=state.swarm_type,
        status=state.status.value,
        agent_statuses={k: v.value for k, v in state.agent_statuses.items()},
        final_result=state.final_result,
        error=state.error,
    )


@app.get("/api/v1/tasks/{task_id}/stream")
async def stream_task(task_id: str) -> EventSourceResponse:
    """SSE stream of agent progress events for a task."""
    state = _tasks.get(task_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    async def event_generator():
        # Yield current state first
        yield {
            "event": "status",
            "data": json.dumps({
                "task_id": task_id,
                "status": state.status.value,
                "agent_statuses": {k: v.value for k, v in state.agent_statuses.items()},
            }),
        }

        # Poll for updates
        last_status = state.status
        last_agent_count = len(state.agent_statuses)

        while state.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED):
            await asyncio.sleep(0.5)

            # Emit update if anything changed
            current_count = len(state.agent_statuses)
            if state.status != last_status or current_count != last_agent_count:
                yield {
                    "event": "update",
                    "data": json.dumps({
                        "task_id": task_id,
                        "status": state.status.value,
                        "agent_statuses": {k: v.value for k, v in state.agent_statuses.items()},
                    }),
                }
                last_status = state.status
                last_agent_count = current_count

        # Final result
        yield {
            "event": "complete",
            "data": json.dumps({
                "task_id": task_id,
                "status": state.status.value,
                "final_result": state.final_result,
                "error": state.error,
            }),
        }

    return EventSourceResponse(event_generator())


# ---------------------------------------------------------------------------
# Background execution
# ---------------------------------------------------------------------------

async def _run_task(task_id: str, template: SwarmTemplate) -> None:
    """Run a swarm task in the background."""
    state = _tasks[task_id]

    try:
        state.status = TaskStatus.EXECUTING
        plan = state.plan

        # Execute the DAG
        executor = WaveExecutor(plan, _agent_factory, max_parallel=config.MAX_PARALLEL_AGENTS)
        result_state = await executor.run()

        state.agent_statuses = result_state.agent_statuses
        state.agent_results = result_state.agent_results

        # Compress results if configured
        compression = template.compression_config
        if compression.get("use_llm") or len(result_state.agent_results) > 3:
            state.status = TaskStatus.COMPRESSING
            compressed = await tree_compress(
                result_state.agent_results,
                task_context=state.query,
                group_size=compression.get("group_size", 5),
                max_levels=compression.get("max_levels", 3),
                use_llm=compression.get("use_llm", True),
            )
            state.final_result = compressed
        else:
            state.final_result = result_state.agent_results

        state.status = TaskStatus.COMPLETED
        logger.info("Task %s completed", task_id)

    except Exception as exc:
        logger.error("Task %s failed: %s", task_id, exc, exc_info=True)
        state.status = TaskStatus.FAILED
        state.error = str(exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=config.SWARM_EXECUTOR_PORT,
        reload=False,
        log_level="info",
    )
