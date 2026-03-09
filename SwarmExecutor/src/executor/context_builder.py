"""
Context Builder — THE critical piece of SwarmExecutor.

Reads an expert's context_prep config from the swarm YAML and transforms raw
dependency results into a clean, compressed, token-budgeted context block that
a small LoRA-tuned model can immediately act on.

The small model should NEVER have to figure out what's relevant. This module
does all the heavy lifting: stripping HTML noise, compressing verbose results,
extracting code signatures instead of full files, and formatting everything
into the exact shape the agent needs.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.schemas.plan_schema import ContextFormat, ContextPrepConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTML noise stripping
# ---------------------------------------------------------------------------

# Tags whose entire content (including children) is noise
_STRIP_TAGS = re.compile(
    r"<\s*(nav|footer|header|aside|script|style|noscript|iframe|svg|"
    r"form|button|input|select|textarea|label|fieldset|menu|"
    r"\.cookie|\.banner|\.popup|\.modal|\.sidebar|\.ad|\.advertisement)"
    r"[\s>].*?</\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)

# Common ad/tracking class patterns
_NOISE_CLASS = re.compile(
    r'class\s*=\s*"[^"]*(?:cookie|consent|gdpr|newsletter|subscribe|'
    r"popup|modal|overlay|sidebar|nav-|menu-|footer-|header-|"
    r"ad-|ads-|advert|sponsor|promo|social-share|share-button|"
    r'related-post|breadcrumb)[^"]*"',
    re.IGNORECASE,
)

# Whitespace normalization
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def strip_html_noise(text: str) -> str:
    """
    Remove navigation, footer, ads, cookie banners, scripts, and other
    non-content HTML. Works on both raw HTML and already-extracted text
    that may contain HTML fragments.
    """
    if not text:
        return ""

    # Remove script/style tags and content
    result = re.sub(r"<\s*(?:script|style)\b[^>]*>.*?</\s*(?:script|style)\s*>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Remove noise tags (nav, footer, aside, etc.)
    result = _STRIP_TAGS.sub("", result)

    # Remove elements with noise-related classes
    # Find tags with noisy class attributes and remove them + content up to closing tag
    result = re.sub(
        r"<\w+[^>]*" + _NOISE_CLASS.pattern + r"[^>]*>.*?</\w+>",
        "",
        result,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Strip remaining HTML tags but keep text
    result = re.sub(r"<[^>]+>", " ", result)

    # Decode common HTML entities
    for entity, char in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'), ("&#39;", "'"), ("&nbsp;", " ")]:
        result = result.replace(entity, char)

    # Normalize whitespace
    result = _MULTI_SPACE.sub(" ", result)
    result = _MULTI_NEWLINE.sub("\n\n", result)

    return result.strip()


# ---------------------------------------------------------------------------
# Result compression
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return len(text) // 4


def compress_result(result: Any, max_tokens: int = 2000) -> str:
    """
    Compress a single agent result to fit within a token budget.

    Strategy:
    1. If it's a dict, extract the most informative fields
    2. If it's a string, truncate intelligently at sentence boundaries
    3. If it's a list, take top items and summarize the rest
    """
    if result is None:
        return ""

    if isinstance(result, str):
        return _compress_string(result, max_tokens)

    if isinstance(result, list):
        return _compress_list(result, max_tokens)

    if isinstance(result, dict):
        return _compress_dict(result, max_tokens)

    # Fallback: stringify and truncate
    text = str(result)
    return _compress_string(text, max_tokens)


def _compress_string(text: str, max_tokens: int) -> str:
    if _estimate_tokens(text) <= max_tokens:
        return text

    # Truncate at sentence boundary
    char_budget = max_tokens * 4
    truncated = text[:char_budget]

    # Find last sentence end
    for end_char in [".\n", ". ", "!\n", "! ", "?\n", "? "]:
        last_end = truncated.rfind(end_char)
        if last_end > char_budget * 0.5:
            return truncated[: last_end + 1] + "\n[...truncated]"

    return truncated + "\n[...truncated]"


def _compress_list(items: list, max_tokens: int) -> str:
    if not items:
        return "[]"

    # Budget per item
    per_item_budget = max(200, max_tokens // max(len(items), 1))
    compressed_items = []
    total_tokens = 0

    for item in items:
        item_str = compress_result(item, per_item_budget)
        item_tokens = _estimate_tokens(item_str)
        if total_tokens + item_tokens > max_tokens:
            remaining = len(items) - len(compressed_items)
            compressed_items.append(f"[...{remaining} more items omitted]")
            break
        compressed_items.append(item_str)
        total_tokens += item_tokens

    return "\n---\n".join(compressed_items)


def _compress_dict(d: dict, max_tokens: int) -> str:
    if not d:
        return "{}"

    # Priority fields — these carry the most signal
    priority_keys = ["summary", "claim", "text", "content", "result", "answer", "facts", "findings", "analysis"]
    low_value_keys = {"metadata", "raw_html", "debug", "trace", "headers", "cookies", "timing"}

    # Extract high-value fields first
    high_value = {}
    rest = {}
    for key, val in d.items():
        if key in low_value_keys:
            continue
        elif key in priority_keys:
            high_value[key] = val
        else:
            rest[key] = val

    # Serialize high-value first, then fill with rest
    parts = []
    tokens_used = 0

    for key, val in {**high_value, **rest}.items():
        if isinstance(val, (dict, list)):
            val_str = json.dumps(val, ensure_ascii=False, default=str)
        else:
            val_str = str(val)

        val_tokens = _estimate_tokens(val_str)
        if tokens_used + val_tokens > max_tokens:
            # Compress this value to fit remaining budget
            remaining_budget = max_tokens - tokens_used
            if remaining_budget > 50:
                val_str = _compress_string(val_str, remaining_budget)
                parts.append(f"{key}: {val_str}")
            break

        parts.append(f"{key}: {val_str}")
        tokens_used += val_tokens

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Field filtering
# ---------------------------------------------------------------------------

def _filter_fields(result: Any, include: list[str], exclude: list[str]) -> Any:
    """Filter dict fields based on include/exclude lists."""
    if not isinstance(result, dict):
        return result

    if include:
        return {k: v for k, v in result.items() if k in include}
    if exclude:
        return {k: v for k, v in result.items() if k not in exclude}
    return result


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _format_bullet_points(dep_results: dict[str, str]) -> str:
    """Format compressed results as bullet points — most compact."""
    lines = []
    for agent_id, content in dep_results.items():
        lines.append(f"## {agent_id}")
        for line in content.split("\n"):
            line = line.strip()
            if line:
                lines.append(f"- {line}")
        lines.append("")
    return "\n".join(lines)


def _format_structured(dep_results: dict[str, str]) -> str:
    """Format as structured sections — balanced detail."""
    sections = []
    for agent_id, content in dep_results.items():
        sections.append(f"=== {agent_id} ===\n{content}")
    return "\n\n".join(sections)


def _format_raw(dep_results: dict[str, str]) -> str:
    """Minimal formatting — just concatenate."""
    return "\n\n".join(f"[{agent_id}]\n{content}" for agent_id, content in dep_results.items())


def _format_code_signatures(dep_results: dict[str, str]) -> str:
    """Format for code analysis — extract signatures, strip bodies."""
    sections = []
    for agent_id, content in dep_results.items():
        # Try to parse as JSON to extract signature-level info
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "signatures" in data:
                sig_text = "\n".join(data["signatures"])
                sections.append(f"=== {agent_id} ===\n{sig_text}")
                continue
        except (json.JSONDecodeError, TypeError):
            pass
        sections.append(f"=== {agent_id} ===\n{content}")
    return "\n\n".join(sections)


_FORMATTERS = {
    ContextFormat.BULLET_POINTS: _format_bullet_points,
    ContextFormat.STRUCTURED: _format_structured,
    ContextFormat.RAW: _format_raw,
    ContextFormat.CODE_SIGNATURES: _format_code_signatures,
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_context(
    context_config: ContextPrepConfig,
    dependency_results: dict[str, Any],
) -> str:
    """
    Build a prepared context string from dependency results according to
    the agent's context_prep configuration.

    This is the function that ensures a small LoRA model gets exactly
    what it needs in the right format and token budget.

    Steps:
    1. Filter fields (include/exclude from config)
    2. Strip HTML noise if configured
    3. Compress each dependency result to budget
    4. Format according to context_config.format
    5. Final trim to token budget
    """
    if not dependency_results:
        return ""

    max_tokens = context_config.max_tokens
    # Budget per dependency
    per_dep_budget = max(500, max_tokens // max(len(dependency_results), 1))

    compressed: dict[str, str] = {}

    for agent_id, raw_result in dependency_results.items():
        # Step 1: Filter fields
        filtered = _filter_fields(raw_result, context_config.include_fields, context_config.exclude_fields)

        # Step 2: Stringify
        if isinstance(filtered, str):
            text = filtered
        elif isinstance(filtered, (dict, list)):
            text = json.dumps(filtered, ensure_ascii=False, default=str)
        else:
            text = str(filtered)

        # Step 3: Strip HTML noise
        if context_config.strip_html:
            text = strip_html_noise(text)

        # Step 4: Compress to budget
        if context_config.compress:
            text = compress_result(text, per_dep_budget)

        compressed[agent_id] = text

    # Step 5: Format
    formatter = _FORMATTERS.get(context_config.format, _format_structured)
    formatted = formatter(compressed)

    # Step 6: Final trim
    if _estimate_tokens(formatted) > max_tokens:
        char_budget = max_tokens * 4
        formatted = formatted[:char_budget] + "\n[...context trimmed to fit token budget]"

    return formatted
