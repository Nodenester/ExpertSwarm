"""
Swarm YAML loader — loads swarm configs and instantiates agent templates + DAG.

Discovers all .yaml files in the swarm config directory and parses them into
usable swarm definitions with agent templates and default DAG plans.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from src.config import config
from src.schemas.plan_schema import (
    AgentPlanNode,
    AgentRole,
    ContextFormat,
    ContextPrepConfig,
    ExecutionPlan,
    ExecutionWave,
)

logger = logging.getLogger(__name__)


class SwarmTemplate:
    """A parsed swarm configuration ready for instantiation."""

    def __init__(
        self,
        name: str,
        description: str,
        version: str,
        agent_templates: dict[str, AgentPlanNode],
        default_waves: list[ExecutionWave],
        dependencies: dict[str, list[str]],
        compression_config: dict[str, Any],
    ) -> None:
        self.name = name
        self.description = description
        self.version = version
        self.agent_templates = agent_templates
        self.default_waves = default_waves
        self.dependencies = dependencies
        self.compression_config = compression_config

    def create_plan(self, task_id: str, query: str) -> ExecutionPlan:
        """
        Create an execution plan from this template.

        Instantiates agent templates with the given task query,
        injecting the query into each agent's task_prompt.
        """
        agents: dict[str, AgentPlanNode] = {}
        for template_id, template in self.agent_templates.items():
            # Clone the template and inject the query
            task_prompt = template.task_prompt
            if "{query}" in task_prompt:
                task_prompt = task_prompt.replace("{query}", query)
            elif not task_prompt:
                task_prompt = query

            agents[template_id] = AgentPlanNode(
                agent_id=template_id,
                role=template.role,
                lora_name=template.lora_name,
                system_prompt=template.system_prompt,
                task_prompt=task_prompt,
                depends_on=template.depends_on,
                tools=template.tools,
                output_schema=template.output_schema,
                context_prep=template.context_prep,
                max_steps=template.max_steps,
            )

        return ExecutionPlan(
            task_id=task_id,
            swarm_type=self.name,
            agents=agents,
            waves=self.default_waves,
            metadata={"query": query, "compression": self.compression_config},
        )


def _parse_context_prep(data: dict[str, Any] | None) -> ContextPrepConfig | None:
    """Parse context_prep config from YAML."""
    if not data:
        return None

    format_map = {
        "bullet_points": ContextFormat.BULLET_POINTS,
        "structured": ContextFormat.STRUCTURED,
        "raw": ContextFormat.RAW,
        "code_signatures": ContextFormat.CODE_SIGNATURES,
    }

    return ContextPrepConfig(
        format=format_map.get(data.get("format", "structured"), ContextFormat.STRUCTURED),
        max_tokens=data.get("max_tokens", 4000),
        strip_html=data.get("strip_html", False),
        compress=data.get("compress", True),
        include_fields=data.get("include_fields", []),
        exclude_fields=data.get("exclude_fields", []),
    )


def _parse_agent(agent_id: str, data: dict[str, Any], deps: list[str]) -> AgentPlanNode:
    """Parse a single agent template from YAML."""
    role_map = {
        "orchestrator": AgentRole.ORCHESTRATOR,
        "scout": AgentRole.SCOUT,
        "worker": AgentRole.WORKER,
        "aggregator": AgentRole.AGGREGATOR,
        "synthesizer": AgentRole.SYNTHESIZER,
    }

    return AgentPlanNode(
        agent_id=agent_id,
        role=role_map.get(data.get("role", "worker"), AgentRole.WORKER),
        lora_name=data.get("lora_name", "default"),
        system_prompt=data.get("system_prompt", ""),
        task_prompt=data.get("task_prompt", "{query}"),
        depends_on=deps,
        tools=data.get("tools", []),
        output_schema=data.get("output_schema"),
        context_prep=_parse_context_prep(data.get("context_prep")),
        max_steps=data.get("max_steps", 5),
    )


def load_swarm_config(filepath: str | Path) -> SwarmTemplate:
    """Load a single swarm YAML config file."""
    filepath = Path(filepath)
    logger.info("Loading swarm config: %s", filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    name = raw.get("name", filepath.stem)
    description = raw.get("description", "")
    version = raw.get("version", "1.0")
    compression_config = raw.get("compression", {"group_size": 5, "max_levels": 3, "use_llm": True})

    # Parse dependencies from default_dag
    dag_config = raw.get("default_dag", {})
    dep_map: dict[str, list[str]] = dag_config.get("dependencies", {})

    # Parse agent templates
    agent_templates: dict[str, AgentPlanNode] = {}
    for agent_id, agent_data in raw.get("agents", {}).items():
        deps = dep_map.get(agent_id, [])
        agent_templates[agent_id] = _parse_agent(agent_id, agent_data, deps)

    # Parse default waves
    waves_raw = dag_config.get("waves", [])
    default_waves: list[ExecutionWave] = []
    for i, wave_agents in enumerate(waves_raw):
        default_waves.append(ExecutionWave(wave_index=i, agent_ids=wave_agents))

    template = SwarmTemplate(
        name=name,
        description=description,
        version=version,
        agent_templates=agent_templates,
        default_waves=default_waves,
        dependencies=dep_map,
        compression_config=compression_config,
    )

    logger.info(
        "Loaded swarm %r: %d agents, %d waves",
        name,
        len(agent_templates),
        len(default_waves),
    )
    return template


def discover_swarms(config_dir: str | None = None) -> dict[str, SwarmTemplate]:
    """
    Discover and load all swarm YAML configs from the config directory.
    Returns a dict of swarm_name -> SwarmTemplate.
    """
    search_dir = Path(config_dir or config.SWARM_CONFIG_DIR)
    if not search_dir.is_dir():
        logger.warning("Swarm config directory not found: %s", search_dir)
        return {}

    swarms: dict[str, SwarmTemplate] = {}
    for yaml_file in sorted(search_dir.glob("*.yaml")):
        try:
            template = load_swarm_config(yaml_file)
            swarms[template.name] = template
            logger.info("Discovered swarm: %s", template.name)
        except Exception as exc:
            logger.error("Failed to load swarm config %s: %s", yaml_file, exc, exc_info=True)

    logger.info("Discovered %d swarm types: %s", len(swarms), list(swarms.keys()))
    return swarms
