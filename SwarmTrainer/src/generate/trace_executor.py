"""Trace executor — takes generated traces with tool calls and executes them for real.

Supports: HTTP GET/POST, file read/write/grep, shell commands (sandboxed), and scraping.
Records real observations so traces contain grounded outputs.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import httpx
import jsonlines
from bs4 import BeautifulSoup


ALLOWED_TOOLS = {
    "http_get", "http_post", "scrape_url", "grep_files",
    "read_file", "write_file", "list_dir", "shell_exec",
    "search_web",
}

# Safety: limit domains for HTTP requests during trace execution
ALLOWED_DOMAINS = None  # Set to a list to restrict, None = allow all

MAX_RESPONSE_BYTES = 512_000  # 500KB max per HTTP response
SHELL_TIMEOUT_S = 30
HTTP_TIMEOUT_S = 30


class TraceExecutor:
    """Execute tool calls from a trace and record real observations."""

    def __init__(self, sandbox_dir: str | None = None, allow_shell: bool = False):
        self.sandbox_dir = Path(sandbox_dir) if sandbox_dir else Path(tempfile.mkdtemp(prefix="trace_"))
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        self.allow_shell = allow_shell
        self.http_client = httpx.Client(timeout=HTTP_TIMEOUT_S, follow_redirects=True)

    def execute_tool_call(self, tool_name: str, args: dict) -> dict:
        """Execute a single tool call and return the observation."""
        if tool_name not in ALLOWED_TOOLS:
            return {"error": f"Unknown tool: {tool_name}", "status": "error"}

        handler = getattr(self, f"_exec_{tool_name}", None)
        if not handler:
            return {"error": f"No handler for tool: {tool_name}", "status": "error"}

        try:
            result = handler(args)
            return {"output": result, "status": "ok"}
        except Exception as e:
            return {"error": str(e), "status": "error"}

    def _exec_http_get(self, args: dict) -> str:
        url = args["url"]
        self._check_url(url)
        headers = args.get("headers", {})
        resp = self.http_client.get(url, headers=headers)
        resp.raise_for_status()
        body = resp.text[:MAX_RESPONSE_BYTES]
        return json.dumps({
            "status_code": resp.status_code,
            "headers": dict(resp.headers),
            "body": body,
            "truncated": len(resp.text) > MAX_RESPONSE_BYTES,
        })

    def _exec_http_post(self, args: dict) -> str:
        url = args["url"]
        self._check_url(url)
        headers = args.get("headers", {"Content-Type": "application/json"})
        payload = args.get("body", args.get("data", {}))
        resp = self.http_client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        body = resp.text[:MAX_RESPONSE_BYTES]
        return json.dumps({
            "status_code": resp.status_code,
            "body": body,
            "truncated": len(resp.text) > MAX_RESPONSE_BYTES,
        })

    def _exec_scrape_url(self, args: dict) -> str:
        url = args["url"]
        self._check_url(url)
        selector = args.get("selector", None)
        resp = self.http_client.get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove script/style tags
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        if selector:
            elements = soup.select(selector)
            texts = [el.get_text(strip=True) for el in elements]
            return json.dumps({"selector": selector, "matches": len(texts), "texts": texts[:50]})
        else:
            text = soup.get_text(separator="\n", strip=True)
            lines = [l for l in text.split("\n") if l.strip()]
            return json.dumps({"text": "\n".join(lines[:200]), "total_lines": len(lines)})

    def _exec_grep_files(self, args: dict) -> str:
        pattern = args["pattern"]
        directory = self._safe_path(args.get("directory", "."))
        glob_pattern = args.get("glob", "**/*")
        max_results = args.get("max_results", 50)

        matches = []
        for fpath in directory.rglob(glob_pattern.lstrip("**/") if glob_pattern.startswith("**/") else glob_pattern):
            if not fpath.is_file():
                continue
            try:
                text = fpath.read_text(encoding="utf-8", errors="ignore")
                for i, line in enumerate(text.split("\n"), 1):
                    if re.search(pattern, line):
                        matches.append({
                            "file": str(fpath.relative_to(directory)),
                            "line": i,
                            "content": line.strip()[:200],
                        })
                        if len(matches) >= max_results:
                            break
            except (OSError, UnicodeDecodeError):
                continue
            if len(matches) >= max_results:
                break

        return json.dumps({"pattern": pattern, "matches": matches, "total": len(matches)})

    def _exec_read_file(self, args: dict) -> str:
        fpath = self._safe_path(args["path"])
        if not fpath.exists():
            return json.dumps({"error": f"File not found: {fpath}"})
        max_lines = args.get("max_lines", 500)
        lines = fpath.read_text(encoding="utf-8", errors="ignore").split("\n")[:max_lines]
        return json.dumps({
            "path": str(fpath),
            "content": "\n".join(lines),
            "total_lines": len(lines),
            "truncated": len(lines) >= max_lines,
        })

    def _exec_write_file(self, args: dict) -> str:
        fpath = self._safe_path(args["path"])
        fpath.parent.mkdir(parents=True, exist_ok=True)
        content = args["content"]
        fpath.write_text(content, encoding="utf-8")
        return json.dumps({"path": str(fpath), "bytes_written": len(content.encode("utf-8"))})

    def _exec_list_dir(self, args: dict) -> str:
        directory = self._safe_path(args.get("path", "."))
        if not directory.is_dir():
            return json.dumps({"error": f"Not a directory: {directory}"})
        entries = []
        for entry in sorted(directory.iterdir()):
            entries.append({
                "name": entry.name,
                "type": "dir" if entry.is_dir() else "file",
                "size": entry.stat().st_size if entry.is_file() else None,
            })
        return json.dumps({"path": str(directory), "entries": entries[:200]})

    def _exec_shell_exec(self, args: dict) -> str:
        if not self.allow_shell:
            return json.dumps({"error": "Shell execution disabled in this executor"})
        cmd = args["command"]
        # Basic safety: block dangerous commands
        dangerous = ["rm -rf /", "mkfs", "dd if=", "> /dev/", ":(){ :|:", "fork"]
        for d in dangerous:
            if d in cmd:
                return json.dumps({"error": f"Blocked dangerous command pattern: {d}"})
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=SHELL_TIMEOUT_S, cwd=str(self.sandbox_dir),
            )
            return json.dumps({
                "stdout": result.stdout[:MAX_RESPONSE_BYTES],
                "stderr": result.stderr[:10000],
                "returncode": result.returncode,
            })
        except subprocess.TimeoutExpired:
            return json.dumps({"error": f"Command timed out after {SHELL_TIMEOUT_S}s"})

    def _exec_search_web(self, args: dict) -> str:
        # Placeholder — in production, route through a search API
        query = args["query"]
        return json.dumps({
            "query": query,
            "results": [],
            "note": "Web search not available in sandbox mode. Replace with real search API.",
        })

    def _check_url(self, url: str):
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
        if ALLOWED_DOMAINS is not None and parsed.hostname not in ALLOWED_DOMAINS:
            raise ValueError(f"Domain not allowed: {parsed.hostname}")

    def _safe_path(self, path_str: str) -> Path:
        """Resolve path within sandbox to prevent directory traversal."""
        candidate = (self.sandbox_dir / path_str).resolve()
        sandbox_resolved = self.sandbox_dir.resolve()
        if not str(candidate).startswith(str(sandbox_resolved)):
            raise ValueError(f"Path escapes sandbox: {path_str}")
        return candidate

    def close(self):
        self.http_client.close()


def execute_trace(trace: dict, executor: TraceExecutor) -> dict:
    """Execute all tool calls in a trace and record observations.

    A trace is a list of steps, where each step may contain a tool_call.
    Returns the trace with observations filled in.
    """
    steps = trace.get("steps", [])
    executed_steps = []

    for step in steps:
        tool_call = step.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name", tool_call.get("tool"))
            tool_args = tool_call.get("args", tool_call.get("arguments", {}))
            if isinstance(tool_args, str):
                tool_args = json.loads(tool_args)
            observation = executor.execute_tool_call(tool_name, tool_args)
            step["observation"] = observation
        executed_steps.append(step)

    trace["steps"] = executed_steps
    trace["executed"] = True
    return trace


def process_file(input_path: Path, output_path: Path, executor: TraceExecutor):
    """Process a JSONL file of traces, executing tool calls in each."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    errors = 0

    with jsonlines.open(input_path) as reader, jsonlines.open(output_path, mode="w") as writer:
        for trace in reader:
            try:
                executed = execute_trace(trace, executor)
                writer.write(executed)
                processed += 1
            except Exception as e:
                print(f"  [ERROR] Failed to execute trace: {e}", file=sys.stderr)
                trace["execution_error"] = str(e)
                writer.write(trace)
                errors += 1

    print(f"Processed {processed} traces ({errors} errors) -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Execute tool calls in generated traces")
    parser.add_argument("--input", required=True, help="Input JSONL file with traces")
    parser.add_argument("--output", required=True, help="Output JSONL file with executed traces")
    parser.add_argument("--sandbox-dir", default=None, help="Sandbox directory for file operations")
    parser.add_argument("--allow-shell", action="store_true", help="Allow shell command execution")
    args = parser.parse_args()

    executor = TraceExecutor(sandbox_dir=args.sandbox_dir, allow_shell=args.allow_shell)
    try:
        process_file(Path(args.input), Path(args.output), executor)
    finally:
        executor.close()


if __name__ == "__main__":
    main()
