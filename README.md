# ExpertSwarm

> **Created: March 2, 2026** | **Status: Archived** | **License: MIT**

Universal framework for creating swarms of ultra-narrow LoRA experts on tiny base models. Define a swarm config (YAML), train experts (QLoRA on Qwen3.5-0.8B), deploy via vLLM multi-LoRA serving. New swarm types require zero framework changes.

Built by [NodeNestor](https://github.com/Nodenester).

---

## What It Does

ExpertSwarm turns one small language model (Qwen3.5-0.8B) into a swarm of 15 specialized agents by hot-loading task-specific LoRA adapters at inference time. A DAG-based orchestration engine dispatches tasks to the right expert, runs agents in parallel waves, and synthesizes results -- all on a single consumer GPU.

**Two built-in swarm types:**
- **WebStorm** -- Deep parallel web research using Scrapling (scrape, extract facts, cross-reference, synthesize)
- **CodeStorm** -- Parallel codebase analysis (grep, glob, AST parsing, relationship mapping)

Adding a new swarm type is just a YAML config + training data. No framework code changes needed.

## Architecture

```
              Clients
                |
                v
+-----------------------------+
|  SwarmExecutor (:8500)      |
|  DAG orchestration engine   |
|  - YAML-defined swarms      |
|  - Async wave execution     |
|  - Context builder          |
|  - Tool registry            |
+-------------+---------------+
              | OpenAI-compatible API
              v
+-----------------------------+
|  SwarmRunner (:8100)        |
|  Meta-router (FastAPI)      |
|  - Keyword/LoRA routing     |
|  - Schema injection         |
|  - Streaming passthrough    |
+-------------+---------------+
              |
              v
+-----------------------------+
|  vLLM (:8000)               |
|  - Multi-LoRA serving       |
|  - Constrained decoding     |
|  - Hot-load adapters        |
|  - GPU inference            |
+-----------------------------+

SwarmTrainer (offline)
  - SFT data generation via teacher model
  - QLoRA training (Unsloth)
  - GRPO reinforcement learning
  - Export -> hot-load into SwarmRunner
```

## Project Structure

| Directory | Purpose | Port |
|-----------|---------|------|
| `SwarmRunner/` | vLLM inference + FastAPI meta-router | 8000, 8100 |
| `SwarmExecutor/` | DAG-based swarm orchestration | 8500 |
| `SwarmTrainer/` | SFT data gen + QLoRA training + GRPO RL | offline |

## Expert Roster (15 LoRAs)

| # | Name | LoRA ID | Swarm |
|---|------|---------|-------|
| 0 | Meta Router | `router` | Both |
| 1 | Web Orchestrator | `web-orch` | Web |
| 2 | Code Orchestrator | `code-orch` | Code |
| 3 | Query Generator | `query-gen` | Web |
| 4 | Pattern Generator | `pattern-gen` | Code |
| 5 | URL Picker | `url-pick` | Web |
| 6 | CSS Selector | `css-sel` | Web |
| 7 | Fact Extractor | `fact-ext` | Web |
| 8 | Code Reader | `code-read` | Code |
| 9 | File Classifier | `file-cls` | Code |
| 10 | Cross-Reference | `cross-ref` | Both |
| 11 | Compressor | `compress` | Both |
| 12 | Relationship Mapper | `rel-map` | Code |
| 13 | Web Synthesizer | `web-synth` | Web |
| 14 | Code Synthesizer | `code-synth` | Code |

## Tech Stack

- **Base Model**: Qwen3.5-0.8B (GPTQ-Int4 for inference)
- **Training**: Unsloth QLoRA (rank-32), GRPO reinforcement learning
- **Serving**: vLLM with multi-LoRA, xgrammar constrained decoding
- **Orchestration**: FastAPI, asyncio DAG executor
- **Web Scraping**: Scrapling
- **Infrastructure**: Docker, NVIDIA Container Toolkit

## Quick Start

```bash
# 1. Configure
cp .env.example .env
# Edit .env -- set HF_TOKEN at minimum

# 2. Launch
docker compose up -d

# 3. Verify
curl http://localhost:8000/v1/models     # vLLM
curl http://localhost:8100/health         # Router
curl http://localhost:8500/health         # Executor

# 4. Submit a task
curl -X POST http://localhost:8500/api/v1/swarm/web-research \
  -H "Content-Type: application/json" \
  -d '{"goal": "Research solid-state battery breakthroughs in 2025-2026"}'
```

## Training

```bash
cd SwarmTrainer

# Generate SFT data
python -m src.generate.sft_generator --expert fact-ext --count 300

# Train
python -m src.train.train_expert --config configs/experts/fact-ext.yaml

# Export to SwarmRunner (hot-loads automatically)
python -m src.export.export_lora --adapter checkpoints/fact-ext/final --name fact-ext
```

## Hardware Requirements

- **Inference**: 16GB+ VRAM (tested on RTX 5060 Ti) for base model + 64 concurrent LoRAs
- **Training**: 16GB+ VRAM for QLoRA
- **RAM**: 64GB recommended

## License

MIT
