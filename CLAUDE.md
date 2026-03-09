# ExpertSwarm — Universal LoRA Expert Swarm Framework

## What This Is
A framework for creating swarms of ultra-narrow LoRA experts on tiny base models (Qwen3.5-0.8B).
15 specialized LoRA adapters share one base model via vLLM multi-LoRA serving.
New swarm types = YAML config + training data, zero framework changes.

## Repo Structure
```
E:\AgentingStuff\ExpertSwarm\
├── SwarmRunner/      — vLLM inference + FastAPI meta-router (ports 8000, 8100)
├── SwarmExecutor/    — DAG-based swarm orchestration engine (port 8500)
├── SwarmTrainer/     — SFT data gen + QLoRA training + GRPO RL
└── docker-compose.yml — Root integration (wires Runner + Executor)
```

## Key Architecture Decisions
- Base model: `Qwen/Qwen3.5-0.8B` (configurable via env var)
- QLoRA rank-32 on all attn+MLP layers, trained with Unsloth
- vLLM serves base + all LoRAs, hot-loads via `POST /v1/load_lora_adapter`
- Constrained decoding via xgrammar (JSON schemas per expert)
- DAG executor fires agents in parallel waves via asyncio.gather()
- Context builder does ALL heavy lifting so 0.8B models get laser-focused input

## Training Pipeline (3 stages)
1. **SFT Cold Start**: Teacher model (GLM 4.7 via CodeGate) generates format examples → QLoRA
2. **RL Free Exploration**: Orchestrators run real swarm tasks in sandbox, GLM judges results, GRPO++ updates
3. **Auto-Improvement**: Router auto-trains from routing patterns, successful trajectories become new SFT data

## External Services
- **CodeGate**: `http://localhost:9212`, key: `proxy-local-key`
  - Opus tier → GLM 5, Sonnet tier → GLM 4.7, Haiku tier → GLM 4.5 Air
  - Used for SFT data generation (teacher model) and RL judging
- **HiveMindDB**: Shared swarm memory, knowledge graph (when available)

## Hardware
- GPU 1: RTX 5060 Ti (16GB) — primary for training + inference
- GPU 2: RTX 4060 (8GB) — secondary
- 64GB RAM, Windows 11

## Expert Roster (15 LoRAs)
| LoRA ID | Role | Training | Swarm |
|---------|------|----------|-------|
| router | Meta-router (1-token classification) | SFT | Both |
| web-orch | Web orchestrator (plans DAG) | SFT → RL | Web |
| code-orch | Code orchestrator (plans DAG) | SFT → RL | Code |
| query-gen | Search query generator | SFT | Web |
| pattern-gen | Code pattern generator | SFT | Code |
| url-pick | URL selector | SFT | Web |
| css-sel | CSS selector generator | SFT | Web |
| fact-ext | Fact extractor from text | SFT | Web |
| code-read | Code reader/analyzer | SFT | Code |
| file-cls | File classifier | SFT | Code |
| cross-ref | Cross-reference verifier | SFT | Both |
| compress | Information compressor | SFT | Both |
| rel-map | Relationship mapper | SFT | Code |
| web-synth | Web research synthesizer | SFT | Web |
| code-synth | Code analysis synthesizer | SFT | Code |

## Commands
```bash
# Generate SFT data (parallel API calls, ~3-5 min for 300 examples)
cd SwarmTrainer
python -m src.generate.sft_generator --expert fact_extractor --num-examples 300 --concurrency 10

# Train a single expert
python -m src.train.train_expert --config configs/experts/fact-ext.yaml

# Export LoRA to SwarmRunner
python -m src.export.export_lora --adapter checkpoints/fact_extractor/final --name fact-ext

# Launch inference + orchestration
docker compose up -d

# Hot-load a new LoRA without restart
curl -X POST http://localhost:8000/v1/load_lora_adapter -d '{"lora_name":"fact-ext","lora_path":"/loras/fact-ext"}'
```

## TODO / Future Improvements

### Real-World Training Data
Current SFT data is fully synthetic (teacher model invents both input and output).
For production quality, use real data with teacher annotations:
1. Scrape real web pages (Wikipedia, news, arxiv, docs)
2. Feed real HTML/text to GLM 4.7: "extract facts from this real page"
3. GLM produces gold-standard output for real input
4. Training pair: (real messy webpage → GLM's clean extraction)
This teaches the model to handle actual HTML noise (ads, nav bars, cookie banners).

### Multimodal Training Data
Qwen3.5 supports vision. Future experts should handle:
- Images embedded in web pages (charts, diagrams, screenshots)
- PDF documents with mixed text/image content
- Code screenshots, architecture diagrams
- Training data should include image inputs alongside text
- Requires multimodal SFT examples: (image + text → structured output)

### Additional Swarm Types (just YAML + training data)
- **CodeWriter** — Agentic coding with research-backed context prep
- **DataStorm** — Parallel data analysis / report generation
- **CompStorm** — General computer research (file system, documents)
- **JobHunter** — Job research + application automation

### RL Sandbox Integration
- VerlTool (ICLR 2026) for agentic RL with real tool use
- SWE-MiniSandbox for container-free code agent RL
- Open-AgentRL for GRPO/PPO with multi-turn reasoning
