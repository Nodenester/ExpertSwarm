# SwarmRunner

vLLM inference engine with multi-LoRA support and a FastAPI meta-router for the ExpertSwarm system.

## Architecture

```
Client Request (port 8100)
    |
    v
[Meta-Router]  -- keyword/rule-based routing --> selects LoRA adapter
    |
    v
[vLLM Engine] (port 8000)  -- serves base model + up to 64 concurrent LoRAs
```

- **vLLM** serves the base model with `--enable-lora`, handling concurrent LoRA adapters in GPU memory
- **Meta-Router** receives OpenAI-compatible requests, classifies them by task type, rewrites the `model` field to the correct LoRA name, and optionally injects JSON schema for guided decoding
- Falls back to the base model when no route matches

## Quick Start

```bash
# 1. Copy env file and set your HF token
cp .env.example .env
# edit .env with your HF_TOKEN

# 2. (Optional) Pre-download the base model
pip install huggingface-hub
python scripts/download_base.py

# 3. Start the stack
docker compose up --build

# 4. (Optional) Hot-load LoRA adapters
# Place adapter directories in ./loras/, then:
bash scripts/register_loras.sh
```

## Ports

| Service     | Port | Description                          |
|-------------|------|--------------------------------------|
| vLLM        | 8000 | Direct OpenAI-compatible API         |
| Meta-Router | 8100 | Routed API with LoRA selection       |

## Sending Requests

### Via Meta-Router (recommended)

The router auto-selects the LoRA based on message content or explicit `task_type`:

```bash
# Auto-routed by keywords
curl http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Plan how to research this topic"}],
    "max_tokens": 256
  }'

# Explicit task type
curl http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "planner",
    "messages": [{"role": "user", "content": "Break this into steps"}],
    "max_tokens": 256
  }'
```

### Direct to vLLM

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-0.8B-GPTQ-Int4",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 64
  }'
```

## Adding LoRA Adapters

1. Place your adapter directory in `./loras/`:
   ```
   loras/
     planner-lora/
       adapter_config.json
       adapter_model.safetensors
     coder-lora/
       adapter_config.json
       adapter_model.safetensors
   ```

2. Register with vLLM at runtime:
   ```bash
   bash scripts/register_loras.sh
   ```

3. Update `router/config.yaml` to add routing rules for the new adapter.

## JSON Schemas for Guided Decoding

Schemas in `config/schemas/` are automatically injected into requests when a route matches and `response_format` is not already set. Available schemas:

| Schema              | Route          | Purpose                    |
|---------------------|----------------|----------------------------|
| `plan.json`         | planner        | DAG plan output            |
| `facts.json`        | analyst        | Fact extraction            |
| `queries.json`      | researcher     | Search queries             |
| `urls.json`         | (manual)       | URL selection              |
| `selector.json`     | extractor      | CSS selectors              |
| `code_summary.json` | coder          | Code analysis              |
| `verification.json` | verifier       | Cross-reference            |
| `graph.json`        | graph_builder  | Relationship mapping       |
| `patterns.json`     | (manual)       | Code patterns              |

## Benchmarking

```bash
python scripts/benchmark.py --requests 50 --concurrency 8 --stream
```

## Environment Variables

| Variable                | Default                         | Description                     |
|-------------------------|----------------------------------|---------------------------------|
| `SWARM_RUNNER_PORT`     | `8100`                          | Meta-router port                |
| `SWARM_VLLM_PORT`      | `8000`                          | vLLM API port                   |
| `SWARM_BASE_MODEL`     | `Qwen/Qwen3.5-0.8B-GPTQ-Int4` | HuggingFace model ID            |
| `SWARM_MAX_LORAS`      | `64`                            | Max concurrent GPU LoRAs        |
| `SWARM_MAX_CPU_LORAS`  | `128`                           | Max LoRAs in CPU memory         |
| `SWARM_MAX_LORA_RANK`  | `64`                            | Maximum LoRA rank               |
| `SWARM_GPU_MEMORY_UTIL`| `0.90`                          | GPU memory utilization fraction |
| `SWARM_MAX_MODEL_LEN`  | `8192`                          | Max sequence length             |
| `SWARM_GUIDED_BACKEND` | `xgrammar`                      | Guided decoding backend         |
| `HF_TOKEN`             | (none)                          | HuggingFace token               |
