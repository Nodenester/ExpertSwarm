"""
Base agent — foundation for all swarm agent types.

Handles:
- Calling SwarmRunner (the LoRA inference backend) via OpenAI-compatible API
- Context preparation from dependency results
- Tool execution loop (multi-step agent behavior)
- Result collection and formatting
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from src.config import config
from src.executor.context_builder import build_context
from src.schemas.plan_schema import AgentPlanNode, ContextPrepConfig
from src.schemas.tool_call_schema import (
    AgentExecutionTrace,
    AgentMessage,
    AgentTurn,
    ToolCall,
    ToolResult,
    ToolCallStatus,
)
from src.tools.registry import tool_registry

logger = logging.getLogger(__name__)


class BaseAgent:
    """
    Base class for all swarm agents.

    Subclasses override `build_messages()` to customize the conversation
    structure, and `parse_result()` to extract structured output.
    """

    def __init__(self, node: AgentPlanNode) -> None:
        self.node = node
        self.agent_id = node.agent_id
        self.lora_name = node.lora_name
        self.max_steps = node.max_steps
        self.tools = node.tools
        self.trace = AgentExecutionTrace(agent_id=node.agent_id, lora_name=node.lora_name)

    async def run(self, dependency_results: dict[str, Any]) -> Any:
        """
        Execute this agent's full lifecycle:
        1. Prepare context from dependency results
        2. Build initial messages
        3. Run the agent loop (call expert, execute tools, repeat)
        4. Parse and return the final result
        """
        context = self._prepare_context(dependency_results)
        messages = self.build_messages(context, dependency_results)

        # Agent loop — multi-step tool use
        for step in range(self.max_steps):
            turn = AgentTurn(step=step)

            response = await self.call_expert(messages)
            turn.messages.append(AgentMessage(role="assistant", content=response.get("content", "")))

            # Check for tool calls
            tool_calls = response.get("tool_calls", [])
            if not tool_calls:
                # No tool calls = agent is done
                self.trace.turns.append(turn)
                break

            # Execute tool calls
            tool_results = []
            for tc in tool_calls:
                tool_name = tc["function"]["name"]
                try:
                    arguments = json.loads(tc["function"]["arguments"])
                except (json.JSONDecodeError, KeyError):
                    arguments = {}

                call = ToolCall(tool_name=tool_name, arguments=arguments)
                turn.tool_calls_made.append(call)

                result = await self._execute_tool(tool_name, arguments)
                tool_results.append(result)
                turn.tool_results_received.append(result)

            self.trace.turns.append(turn)

            # Feed tool results back into conversation
            messages.append({"role": "assistant", "content": response.get("content", ""), "tool_calls": tool_calls})
            for tc, tr in zip(tool_calls, tool_results):
                result_content = json.dumps(tr.result, ensure_ascii=False, default=str) if tr.result is not None else tr.error or ""
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": result_content,
                })
        else:
            logger.warning("Agent %s hit max_steps limit (%d)", self.agent_id, self.max_steps)

        # Extract final result
        final_content = ""
        if self.trace.turns:
            last_turn = self.trace.turns[-1]
            if last_turn.messages:
                final_content = last_turn.messages[-1].content

        result = self.parse_result(final_content, dependency_results)
        self.trace.final_output = result
        return result

    async def call_expert(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Call SwarmRunner via OpenAI-compatible API with this agent's LoRA.

        Returns the assistant message dict with 'content' and optionally 'tool_calls'.
        """
        tools = tool_registry.get_definitions(self.tools) if self.tools else None

        request_body: dict[str, Any] = {
            "model": self.lora_name,
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0.7,
        }
        if tools:
            request_body["tools"] = tools
            request_body["tool_choice"] = "auto"

        if self.node.output_schema:
            request_body["response_format"] = {"type": "json_object"}

        start = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{config.SWARM_RUNNER_URL}/v1/chat/completions",
                    json=request_body,
                )
                resp.raise_for_status()
                data = resp.json()

            elapsed = (time.monotonic() - start) * 1000
            self.trace.total_duration_ms += elapsed

            usage = data.get("usage", {})
            self.trace.total_tokens_in += usage.get("prompt_tokens", 0)
            self.trace.total_tokens_out += usage.get("completion_tokens", 0)

            choice = data["choices"][0]["message"]
            return {
                "content": choice.get("content", ""),
                "tool_calls": choice.get("tool_calls", []),
            }

        except httpx.HTTPError as exc:
            logger.error("call_expert failed for %s (LoRA: %s): %s", self.agent_id, self.lora_name, exc)
            raise RuntimeError(f"SwarmRunner call failed: {exc}") from exc

    async def _execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a registered tool and return the result."""
        start = time.monotonic()
        try:
            if not tool_registry.has(tool_name):
                return ToolResult(
                    tool_name=tool_name,
                    status=ToolCallStatus.ERROR,
                    error=f"Unknown tool: {tool_name}",
                )

            result = await tool_registry.execute(tool_name, arguments)
            elapsed = (time.monotonic() - start) * 1000
            return ToolResult(
                tool_name=tool_name,
                status=ToolCallStatus.SUCCESS,
                result=result,
                duration_ms=elapsed,
            )

        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            logger.error("Tool %s failed: %s", tool_name, exc)
            return ToolResult(
                tool_name=tool_name,
                status=ToolCallStatus.ERROR,
                error=str(exc),
                duration_ms=elapsed,
            )

    def _prepare_context(self, dependency_results: dict[str, Any]) -> str:
        """Prepare context from dependency results using the agent's context_prep config."""
        if not dependency_results:
            return ""

        context_config = self.node.context_prep or ContextPrepConfig()
        return build_context(context_config, dependency_results)

    def build_messages(self, context: str, dependency_results: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Build the initial message list for the expert call.
        Subclasses override this to customize conversation structure.
        """
        messages = []

        if self.node.system_prompt:
            messages.append({"role": "system", "content": self.node.system_prompt})

        user_content = self.node.task_prompt
        if context:
            user_content = f"{context}\n\n---\n\n{user_content}"

        messages.append({"role": "user", "content": user_content})
        return messages

    def parse_result(self, raw_output: str, dependency_results: dict[str, Any]) -> Any:
        """
        Parse the final agent output. Subclasses override for structured parsing.
        Default: return raw string.
        """
        if self.node.output_schema:
            try:
                return json.loads(raw_output)
            except json.JSONDecodeError:
                logger.warning("Agent %s output is not valid JSON despite output_schema", self.agent_id)
        return raw_output
