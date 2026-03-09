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
        _http_client = httpx.AsyncClient(timeout=60.0, follow_redirects=True)
    return _http_client


@tool_registry.register(
    "web_search",
    description="Search the web for a query and return ranked results with titles, URLs, and snippets.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {"type": "integer", "description": "Max results to return", "default": 10},
        },
        "required": ["query"],
    },
)
async def web_search(args: dict[str, Any]) -> dict[str, Any]:
    query = args["query"]
    max_results = args.get("max_results", 10)

    client = await _get_client()
    try:
        resp = await client.post(
            f"{config.CODEGATE_URL}/api/v1/search",
            json={"query": query, "max_results": max_results},
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as exc:
        logger.error("web_search failed: %s", exc)
        return {"error": str(exc), "results": []}


@tool_registry.register(
    "scrape_url",
    description="Scrape a URL and return its text content, stripped of navigation and boilerplate.",
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to scrape"},
            "selector": {"type": "string", "description": "CSS selector to extract (optional)"},
            "max_length": {"type": "integer", "description": "Max chars to return", "default": 15000},
        },
        "required": ["url"],
    },
)
async def scrape_url(args: dict[str, Any]) -> dict[str, Any]:
    url = args["url"]
    selector = args.get("selector")
    max_length = args.get("max_length", 15000)

    client = await _get_client()
    try:
        resp = await client.post(
            f"{config.CODEGATE_URL}/api/v1/scrape",
            json={"url": url, "selector": selector},
        )
        resp.raise_for_status()
        data = resp.json()
        text = data.get("text", "")
        if len(text) > max_length:
            text = text[:max_length] + "\n... [truncated]"
        return {"url": url, "title": data.get("title", ""), "text": text}
    except httpx.HTTPError as exc:
        logger.error("scrape_url failed for %s: %s", url, exc)
        return {"url": url, "error": str(exc), "text": ""}


@tool_registry.register(
    "extract_css",
    description="Extract content matching a CSS selector from a URL. Returns a list of text matches.",
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to scrape"},
            "selector": {"type": "string", "description": "CSS selector to extract"},
            "max_matches": {"type": "integer", "description": "Max elements to return", "default": 50},
        },
        "required": ["url", "selector"],
    },
)
async def extract_css(args: dict[str, Any]) -> dict[str, Any]:
    url = args["url"]
    selector = args["selector"]
    max_matches = args.get("max_matches", 50)

    client = await _get_client()
    try:
        resp = await client.post(
            f"{config.CODEGATE_URL}/api/v1/extract",
            json={"url": url, "selector": selector, "max_matches": max_matches},
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as exc:
        logger.error("extract_css failed for %s %s: %s", url, selector, exc)
        return {"url": url, "selector": selector, "error": str(exc), "matches": []}
