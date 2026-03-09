from __future__ import annotations

import os


class Config:
    """Environment-based configuration for SwarmExecutor."""

    SWARM_EXECUTOR_PORT: int = int(os.getenv("SWARM_EXECUTOR_PORT", "8500"))
    SWARM_RUNNER_URL: str = os.getenv("SWARM_RUNNER_URL", "http://swarm-runner:8100")
    HIVEMIND_URL: str = os.getenv("HIVEMIND_URL", "http://hivemind:8100")
    CODEGATE_URL: str = os.getenv("CODEGATE_URL", "http://codegate:9212")
    MAX_PARALLEL_AGENTS: int = int(os.getenv("MAX_PARALLEL_AGENTS", "50"))
    MAX_AGENT_STEPS: int = int(os.getenv("MAX_AGENT_STEPS", "10"))
    SWARM_CONFIG_DIR: str = os.getenv("SWARM_CONFIG_DIR", "./src/swarms")


config = Config()
