from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ToolCallStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class ToolCall(BaseModel):
    """A tool invocation request from an agent."""

    tool_name: str = Field(..., description="Registered tool name")
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Result of a tool execution."""

    tool_name: str
    status: ToolCallStatus = ToolCallStatus.SUCCESS
    result: Any = None
    error: str | None = None
    duration_ms: float = 0.0


class AgentMessage(BaseModel):
    """A single message in an agent's conversation."""

    role: str = Field(..., description="system, user, assistant, or tool")
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)


class AgentTurn(BaseModel):
    """One full turn of agent execution (think + act)."""

    step: int
    messages: list[AgentMessage] = Field(default_factory=list)
    tool_calls_made: list[ToolCall] = Field(default_factory=list)
    tool_results_received: list[ToolResult] = Field(default_factory=list)
    reasoning: str = Field(default="", description="Agent's reasoning for this step")


class AgentExecutionTrace(BaseModel):
    """Full trace of an agent's execution for debugging/training."""

    agent_id: str
    lora_name: str = ""
    turns: list[AgentTurn] = Field(default_factory=list)
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_duration_ms: float = 0.0
    final_output: Any = None
