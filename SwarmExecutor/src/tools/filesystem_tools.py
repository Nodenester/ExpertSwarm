from __future__ import annotations

import fnmatch
import logging
import os
import re
from pathlib import Path
from typing import Any

from src.tools.registry import tool_registry

logger = logging.getLogger(__name__)

# Safety: restrict to allowed root directories
_ALLOWED_ROOTS: list[str] = ["/workspace", "/app", "/tmp"]


def _is_safe_path(path: str) -> bool:
    resolved = str(Path(path).resolve())
    return any(resolved.startswith(root) for root in _ALLOWED_ROOTS)


@tool_registry.register(
    "grep_files",
    description="Search file contents using a regex pattern. Returns matching lines with context.",
    parameters={
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern to search for"},
            "path": {"type": "string", "description": "Directory or file to search in"},
            "glob_filter": {"type": "string", "description": "Glob pattern to filter files (e.g. '*.py')", "default": "*"},
            "max_results": {"type": "integer", "description": "Max matching lines", "default": 50},
            "context_lines": {"type": "integer", "description": "Lines of context around each match", "default": 2},
        },
        "required": ["pattern", "path"],
    },
)
async def grep_files(args: dict[str, Any]) -> dict[str, Any]:
    pattern = args["pattern"]
    search_path = args["path"]
    glob_filter = args.get("glob_filter", "*")
    max_results = args.get("max_results", 50)
    context_lines = args.get("context_lines", 2)

    if not _is_safe_path(search_path):
        return {"error": f"Path not allowed: {search_path}", "matches": []}

    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        return {"error": f"Invalid regex: {exc}", "matches": []}

    matches = []
    search = Path(search_path)

    files = [search] if search.is_file() else sorted(search.rglob(glob_filter))

    for filepath in files:
        if not filepath.is_file() or filepath.stat().st_size > 2_000_000:
            continue
        try:
            lines = filepath.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue

        for i, line in enumerate(lines):
            if regex.search(line):
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                matches.append({
                    "file": str(filepath),
                    "line_number": i + 1,
                    "line": line.rstrip(),
                    "context_before": [l.rstrip() for l in lines[start:i]],
                    "context_after": [l.rstrip() for l in lines[i + 1 : end]],
                })
                if len(matches) >= max_results:
                    return {"pattern": pattern, "matches": matches, "truncated": True}

    return {"pattern": pattern, "matches": matches, "truncated": False}


@tool_registry.register(
    "glob_files",
    description="Find files matching a glob pattern. Returns file paths.",
    parameters={
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern (e.g. '**/*.py')"},
            "path": {"type": "string", "description": "Root directory to search"},
            "max_results": {"type": "integer", "default": 100},
        },
        "required": ["pattern", "path"],
    },
)
async def glob_files(args: dict[str, Any]) -> dict[str, Any]:
    pattern = args["pattern"]
    root = args["path"]
    max_results = args.get("max_results", 100)

    if not _is_safe_path(root):
        return {"error": f"Path not allowed: {root}", "files": []}

    root_path = Path(root)
    if not root_path.is_dir():
        return {"error": f"Not a directory: {root}", "files": []}

    files = []
    for filepath in sorted(root_path.rglob(pattern)):
        if filepath.is_file():
            files.append(str(filepath))
            if len(files) >= max_results:
                break

    return {"pattern": pattern, "root": root, "files": files, "count": len(files)}


@tool_registry.register(
    "read_file",
    description="Read a file and return its contents. Supports line range selection.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to read"},
            "start_line": {"type": "integer", "description": "Starting line (1-indexed)", "default": 1},
            "end_line": {"type": "integer", "description": "Ending line (inclusive)", "default": 500},
        },
        "required": ["path"],
    },
)
async def read_file(args: dict[str, Any]) -> dict[str, Any]:
    filepath = args["path"]
    start = args.get("start_line", 1)
    end = args.get("end_line", 500)

    if not _is_safe_path(filepath):
        return {"error": f"Path not allowed: {filepath}"}

    p = Path(filepath)
    if not p.is_file():
        return {"error": f"Not a file: {filepath}"}

    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return {"error": str(exc)}

    selected = lines[max(0, start - 1) : end]
    return {
        "path": filepath,
        "total_lines": len(lines),
        "start_line": start,
        "end_line": min(end, len(lines)),
        "content": "\n".join(selected),
    }


@tool_registry.register(
    "directory_tree",
    description="Get the directory tree structure. Returns directories and files with sizes.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Root directory"},
            "max_depth": {"type": "integer", "description": "Max depth to recurse", "default": 3},
            "include_hidden": {"type": "boolean", "default": False},
        },
        "required": ["path"],
    },
)
async def directory_tree(args: dict[str, Any]) -> dict[str, Any]:
    root = args["path"]
    max_depth = args.get("max_depth", 3)
    include_hidden = args.get("include_hidden", False)

    if not _is_safe_path(root):
        return {"error": f"Path not allowed: {root}"}

    root_path = Path(root)
    if not root_path.is_dir():
        return {"error": f"Not a directory: {root}"}

    total_files = 0
    total_dirs = 0

    def _build(p: Path, depth: int) -> list[dict[str, Any]]:
        nonlocal total_files, total_dirs
        if depth > max_depth:
            return []

        entries = []
        try:
            children = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except PermissionError:
            return []

        for child in children:
            if not include_hidden and child.name.startswith("."):
                continue
            if child.is_dir():
                total_dirs += 1
                entries.append({
                    "name": child.name,
                    "is_dir": True,
                    "children": _build(child, depth + 1),
                })
            else:
                total_files += 1
                try:
                    size = child.stat().st_size
                except OSError:
                    size = 0
                entries.append({"name": child.name, "is_dir": False, "size": size})

        return entries

    tree = _build(root_path, 1)
    return {
        "path": root,
        "entries": tree,
        "total_files": total_files,
        "total_dirs": total_dirs,
    }
