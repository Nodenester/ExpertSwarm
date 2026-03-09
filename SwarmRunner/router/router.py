"""
SwarmRunner Meta-Router

FastAPI application that receives OpenAI-compatible requests on port 8100,
routes them to the appropriate LoRA adapter based on task type or keyword
matching, and forwards the request to the vLLM backend.
"""

import json
import os
import re
from pathlib import Path
from typing import Any

import httpx
import yaml
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

app = FastAPI(title="SwarmRunner Meta-Router", version="0.1.0")

VLLM_PORT = os.environ.get("SWARM_VLLM_PORT", "8000")
VLLM_BASE_URL = os.environ.get("SWARM_VLLM_URL", f"http://localhost:{VLLM_PORT}")
BASE_MODEL = os.environ.get("SWARM_BASE_MODEL", "Qwen/Qwen3.5-0.8B-GPTQ-Int4")
SCHEMA_DIR = os.environ.get("SWARM_SCHEMA_DIR", "/schemas")

# Load router config
CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            raw = f.read()
        # Expand env vars in the YAML
        raw = os.path.expandvars(raw)
        return yaml.safe_load(raw)
    return {"routes": [], "default_model": BASE_MODEL, "schema_bindings": {}}


CONFIG = load_config()
ROUTES: list[dict] = CONFIG.get("routes", [])
SCHEMA_BINDINGS: dict[str, str] = CONFIG.get("schema_bindings", {})


def classify_request(body: dict) -> tuple[str | None, str | None]:
    """
    Determine which LoRA adapter to route to based on the request body.

    Returns (lora_name, route_name) or (None, None) for base model fallback.
    """
    # 1. Explicit task_type field
    task_type = body.get("task_type", "").lower()
    if task_type:
        for route in ROUTES:
            if route["name"] == task_type:
                return route["lora_name"], route["name"]

    # 2. Explicit model field that matches a LoRA name
    requested_model = body.get("model", "")
    for route in ROUTES:
        if requested_model == route["lora_name"]:
            return route["lora_name"], route["name"]

    # 2b. If model is explicitly set and not the base model / "auto",
    #     pass it through as-is (user is targeting a specific LoRA)
    if requested_model and requested_model not in ("auto", "", BASE_MODEL):
        return requested_model, None

    # 3. Keyword scan on last user message
    messages = body.get("messages", [])
    if messages:
        last_user_msgs = [m for m in messages if m.get("role") == "user"]
        if last_user_msgs:
            text = last_user_msgs[-1].get("content", "").lower()
            best_match = None
            best_score = 0
            for route in ROUTES:
                score = sum(1 for kw in route["keywords"] if re.search(r"\b" + re.escape(kw) + r"\b", text))
                if score > best_score:
                    best_score = score
                    best_match = route
            if best_match and best_score > 0:
                return best_match["lora_name"], best_match["name"]

    return None, None


def get_schema_for_route(route_name: str) -> dict | None:
    """Load the JSON schema bound to a route for guided decoding."""
    schema_file = SCHEMA_BINDINGS.get(route_name)
    if not schema_file:
        return None
    schema_path = Path(SCHEMA_DIR) / schema_file
    if schema_path.exists():
        with open(schema_path) as f:
            return json.load(f)
    return None


@app.get("/health")
async def health():
    return {"status": "ok", "routes": len(ROUTES)}


@app.get("/routes")
async def list_routes():
    return {
        "default_model": BASE_MODEL,
        "routes": [
            {
                "name": r["name"],
                "lora_name": r["lora_name"],
                "keywords": r["keywords"],
                "description": r.get("description", ""),
                "schema": SCHEMA_BINDINGS.get(r["name"]),
            }
            for r in ROUTES
        ],
    }


@app.api_route("/v1/{path:path}", methods=["GET", "POST"])
async def proxy_to_vllm(path: str, request: Request):
    """
    Forward any /v1/* request to the vLLM backend, rewriting the model
    field to the appropriate LoRA adapter when a route matches.
    """
    body_bytes = await request.body()

    # Only attempt routing on POST requests with JSON bodies
    body = {}
    lora_name = None
    route_name = None
    if request.method == "POST" and body_bytes:
        try:
            body = json.loads(body_bytes)
        except json.JSONDecodeError:
            pass

        lora_name, route_name = classify_request(body)

        # Rewrite model field
        if lora_name:
            body["model"] = lora_name
        elif "model" not in body or body.get("task_type"):
            # No route match and no explicit model -- use base
            body["model"] = BASE_MODEL

        # Remove our custom field before forwarding
        body.pop("task_type", None)

        # Inject guided decoding schema if the route has one and caller
        # hasn't already specified response_format
        if route_name and "response_format" not in body:
            schema = get_schema_for_route(route_name)
            if schema:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": route_name,
                        "schema": schema,
                        "strict": True,
                    },
                }

        body_bytes = json.dumps(body).encode()

    target_url = f"{VLLM_BASE_URL}/v1/{path}"

    # Forward headers (minus host)
    headers = dict(request.headers)
    headers.pop("host", None)
    headers["content-length"] = str(len(body_bytes))

    # Stream the response back
    is_stream = body.get("stream", False)

    async with httpx.AsyncClient(timeout=300.0) as client:
        if is_stream:
            req = client.build_request(
                request.method, target_url, content=body_bytes, headers=headers
            )
            resp = await client.send(req, stream=True)

            async def stream_response():
                async for chunk in resp.aiter_bytes():
                    yield chunk
                await resp.aclose()

            return StreamingResponse(
                stream_response(),
                status_code=resp.status_code,
                headers=dict(resp.headers),
            )
        else:
            resp = await client.request(
                request.method, target_url, content=body_bytes, headers=headers
            )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=dict(resp.headers),
            )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("SWARM_RUNNER_PORT", "8100"))
    uvicorn.run(app, host="0.0.0.0", port=port)
