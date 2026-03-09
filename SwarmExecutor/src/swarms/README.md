# Swarm Configuration Guide

## Overview

Each swarm type is defined by a YAML config file in this directory. The config
specifies the agent templates, their tools, context preparation settings, and
the default execution DAG.

## Creating a New Swarm Type

1. Create a new YAML file (e.g., `my_swarm.yaml`) in this directory.
2. The file will be auto-discovered on startup.

## Config Schema

```yaml
name: my_swarm                    # Unique identifier (must match filename)
description: "What this swarm does"
version: "1.0"

# Tree-reduction compression settings
compression:
  group_size: 5                   # Results per compression group
  max_levels: 3                   # Max compression rounds
  use_llm: true                   # Use SwarmRunner's compressor LoRA

# Agent templates
agents:
  agent_name:                     # Unique agent template name
    role: worker                  # orchestrator|scout|worker|aggregator|synthesizer
    lora_name: my-lora            # LoRA adapter name on SwarmRunner
    system_prompt: >              # System prompt for the expert
      Your system prompt here.
    tools:                        # Tool names from the registry
      - web_search
      - scrape_url
    max_steps: 5                  # Max tool-call loops
    output_schema: fact_schema    # Pydantic schema name (optional, enables JSON mode)
    context_prep:                 # How to prepare context from dependencies
      format: structured          # bullet_points|structured|raw|code_signatures
      max_tokens: 4000            # Token budget for context
      strip_html: false           # Remove HTML noise from inputs
      compress: true              # Compress dependency results
      include_fields: []          # Only include these fields (empty = all)
      exclude_fields: []          # Exclude these fields

# Default DAG structure
default_dag:
  waves:
    - [planner]
    - [scout_a, scout_b]
    - [worker_a, worker_b, worker_c]
    - [aggregator]
    - [synthesizer]
  dependencies:
    scout_a: [planner]
    scout_b: [planner]
    worker_a: [scout_a]
    worker_b: [scout_b]
    worker_c: [scout_a, scout_b]
    aggregator: [worker_a, worker_b, worker_c]
    synthesizer: [aggregator]
```

## Agent Roles

| Role | Purpose | Typical Wave |
|------|---------|-------------|
| `orchestrator` | Plans the DAG, decomposes the query | First (wave 0) |
| `scout` | Generates queries, patterns, file lists | Early (wave 1) |
| `worker` | Executes tools: search, scrape, analyze | Middle (wave 2-3) |
| `aggregator` | Cross-references and merges results | Late (wave N-1) |
| `synthesizer` | Produces the final answer/report | Last (wave N) |

## Context Prep Formats

- **bullet_points**: Most compact. Each dependency result as bullet points under a heading.
- **structured**: Balanced. Labeled sections with full content.
- **raw**: Minimal processing. Just concatenated.
- **code_signatures**: For code analysis. Extracts function/class signatures, strips bodies.

## Available Tools

### Web Tools
- `web_search` — Search the web for a query
- `scrape_url` — Scrape a URL and extract text
- `extract_css` — Extract content matching a CSS selector

### Filesystem Tools
- `grep_files` — Search file contents with regex
- `glob_files` — Find files by glob pattern
- `read_file` — Read file contents
- `directory_tree` — Get directory structure

### Code Analysis Tools
- `analyze_python_file` — Parse Python file structure (AST)
- `extract_signatures` — Extract function/method signatures only

### HiveMind Tools
- `store_fact` — Store a fact for later recall
- `recall_facts` — Semantic search over stored facts
- `knowledge_graph_query` — Query entity relationships
