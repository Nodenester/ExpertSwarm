from __future__ import annotations

import logging
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)

# Type for async tool functions: (arguments: dict) -> Any
ToolFunction = Callable[[dict[str, Any]], Awaitable[Any]]


class _ToolEntry:
    """Internal wrapper for a registered tool."""

    __slots__ = ("name", "description", "parameters", "fn")

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        fn: ToolFunction,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.fn = fn


class ToolRegistry:
    """
    Global tool registry. Tools register by name and swarms reference them.

    Usage:
        registry = ToolRegistry()

        @registry.register("web_search", description="...", parameters={...})
        async def web_search(args):
            ...

        # Or register programmatically:
        registry.add("my_tool", description="...", parameters={}, fn=my_async_fn)

        # Execute:
        result = await registry.execute("web_search", {"query": "hello"})

        # List available tools:
        tools = registry.list_tools()

        # Get OpenAI-format tool definitions for a subset:
        definitions = registry.get_definitions(["web_search", "scrape_url"])
    """

    def __init__(self) -> None:
        self._tools: dict[str, _ToolEntry] = {}

    def register(
        self,
        name: str,
        description: str = "",
        parameters: dict[str, Any] | None = None,
    ) -> Callable[[ToolFunction], ToolFunction]:
        """Decorator to register a tool function."""

        def decorator(fn: ToolFunction) -> ToolFunction:
            self.add(name, description=description or fn.__doc__ or "", parameters=parameters or {}, fn=fn)
            return fn

        return decorator

    def add(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        fn: ToolFunction,
    ) -> None:
        if name in self._tools:
            logger.warning("Overwriting existing tool: %s", name)
        self._tools[name] = _ToolEntry(
            name=name,
            description=description,
            parameters=parameters,
            fn=fn,
        )
        logger.info("Registered tool: %s", name)

    async def execute(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        entry = self._tools.get(name)
        if entry is None:
            raise KeyError(f"Tool not found: {name!r}. Available: {list(self._tools.keys())}")
        return await entry.fn(arguments or {})

    def has(self, name: str) -> bool:
        return name in self._tools

    def list_tools(self) -> list[str]:
        return sorted(self._tools.keys())

    def get_definitions(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        """Return OpenAI-compatible tool definitions for the given names (or all)."""
        subset = names or list(self._tools.keys())
        definitions = []
        for name in subset:
            entry = self._tools.get(name)
            if entry is None:
                continue
            definitions.append({
                "type": "function",
                "function": {
                    "name": entry.name,
                    "description": entry.description,
                    "parameters": entry.parameters if entry.parameters else {"type": "object", "properties": {}},
                },
            })
        return definitions

    def get_entry(self, name: str) -> _ToolEntry | None:
        return self._tools.get(name)


# Singleton instance
tool_registry = ToolRegistry()
