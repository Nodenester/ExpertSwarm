from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    ORCHESTRATOR = "orchestrator"
    SCOUT = "scout"
    WORKER = "worker"
    AGGREGATOR = "aggregator"
    SYNTHESIZER = "synthesizer"


class AgentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class AgentPlanNode(BaseModel):
    """A single agent in the execution plan / DAG."""

    agent_id: str = Field(..., description="Unique identifier for this agent instance")
    role: AgentRole
    lora_name: str = Field(..., description="LoRA adapter name on SwarmRunner")
    system_prompt: str = Field(default="", description="System prompt for the expert")
    task_prompt: str = Field(default="", description="Task-specific prompt template")
    depends_on: list[str] = Field(default_factory=list, description="Agent IDs this depends on")
    tools: list[str] = Field(default_factory=list, description="Tool names this agent can use")
    output_schema: str | None = Field(default=None, description="Pydantic schema name for structured output")
    context_prep: ContextPrepConfig | None = Field(default=None, description="How to prepare context from deps")
    max_steps: int = Field(default=5, description="Max tool-call steps for this agent")


class ContextFormat(str, Enum):
    BULLET_POINTS = "bullet_points"
    STRUCTURED = "structured"
    RAW = "raw"
    CODE_SIGNATURES = "code_signatures"


class ContextPrepConfig(BaseModel):
    """Configuration for how an agent's context is prepared from dependency results."""

    format: ContextFormat = ContextFormat.STRUCTURED
    max_tokens: int = Field(default=4000, description="Token budget for context")
    strip_html: bool = Field(default=False, description="Strip HTML noise from inputs")
    compress: bool = Field(default=True, description="Compress dependency results")
    include_fields: list[str] = Field(default_factory=list, description="Only include these fields from dep results")
    exclude_fields: list[str] = Field(default_factory=list, description="Exclude these fields from dep results")


# Rebuild AgentPlanNode to resolve forward ref
AgentPlanNode.model_rebuild()


class ExecutionWave(BaseModel):
    """A group of agents that can run in parallel."""

    wave_index: int
    agent_ids: list[str]


class ExecutionPlan(BaseModel):
    """Complete DAG execution plan."""

    task_id: str
    swarm_type: str
    agents: dict[str, AgentPlanNode]
    waves: list[ExecutionWave] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskStatus(str, Enum):
    QUEUED = "queued"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPRESSING = "compressing"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskState(BaseModel):
    """Top-level task tracking state."""

    task_id: str
    swarm_type: str
    status: TaskStatus = TaskStatus.QUEUED
    query: str = ""
    plan: ExecutionPlan | None = None
    agent_statuses: dict[str, AgentStatus] = Field(default_factory=dict)
    agent_results: dict[str, Any] = Field(default_factory=dict)
    final_result: Any = None
    error: str | None = None
