from __future__ import annotations

import logging
from typing import Any

import httpx

from src.config import config
from src.tools.registry import tool_registry

logger = logging.getLogger(__name__)

_http_client: httpx.AsyncClient | None = None


async def _get_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=30.0)
    return _http_client


@tool_registry.register(
    "store_fact",
    description="Store a fact in HiveMind's knowledge base for later recall by any agent.",
    parameters={
        "type": "object",
        "properties": {
            "namespace": {"type": "string", "description": "Namespace/topic for the fact"},
            "claim": {"type": "string", "description": "The factual claim to store"},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"], "default": "medium"},
            "source_url": {"type": "string", "description": "Source URL (optional)"},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Topic tags"},
        },
        "required": ["namespace", "claim"],
    },
)
async def store_fact(args: dict[str, Any]) -> dict[str, Any]:
    client = await _get_client()
    try:
        resp = await client.post(
            f"{config.HIVEMIND_URL}/api/v1/facts",
            json={
                "namespace": args["namespace"],
                "claim": args["claim"],
                "confidence": args.get("confidence", "medium"),
                "source_url": args.get("source_url"),
                "tags": args.get("tags", []),
            },
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as exc:
        logger.error("store_fact failed: %s", exc)
        return {"error": str(exc), "stored": False}


@tool_registry.register(
    "recall_facts",
    description="Recall facts from HiveMind matching a query. Uses semantic search.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query for facts"},
            "namespace": {"type": "string", "description": "Limit to this namespace (optional)"},
            "max_results": {"type": "integer", "default": 20},
            "min_confidence": {"type": "string", "enum": ["high", "medium", "low"], "default": "low"},
        },
        "required": ["query"],
    },
)
async def recall_facts(args: dict[str, Any]) -> dict[str, Any]:
    client = await _get_client()
    try:
        resp = await client.post(
            f"{config.HIVEMIND_URL}/api/v1/facts/search",
            json={
                "query": args["query"],
                "namespace": args.get("namespace"),
                "max_results": args.get("max_results", 20),
                "min_confidence": args.get("min_confidence", "low"),
            },
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as exc:
        logger.error("recall_facts failed: %s", exc)
        return {"error": str(exc), "facts": []}


@tool_registry.register(
    "knowledge_graph_query",
    description="Query HiveMind's knowledge graph for entity relationships.",
    parameters={
        "type": "object",
        "properties": {
            "entity": {"type": "string", "description": "Entity name to query relationships for"},
            "relationship_type": {"type": "string", "description": "Filter by relationship type (optional)"},
            "depth": {"type": "integer", "description": "Graph traversal depth", "default": 2},
            "max_results": {"type": "integer", "default": 30},
        },
        "required": ["entity"],
    },
)
async def knowledge_graph_query(args: dict[str, Any]) -> dict[str, Any]:
    client = await _get_client()
    try:
        resp = await client.post(
            f"{config.HIVEMIND_URL}/api/v1/graph/query",
            json={
                "entity": args["entity"],
                "relationship_type": args.get("relationship_type"),
                "depth": args.get("depth", 2),
                "max_results": args.get("max_results", 30),
            },
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as exc:
        logger.error("knowledge_graph_query failed: %s", exc)
        return {"error": str(exc), "nodes": [], "edges": []}
