"""Training configuration — loads hyperparameters from YAML config files."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LoraConfig:
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    # Model
    base_model: str = "Qwen/Qwen3.5-0.8B"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    dtype: str = "float16"

    # LoRA
    lora: LoraConfig = field(default_factory=LoraConfig)

    # Training
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    max_grad_norm: float = 1.0

    # Data
    dataset_path: str = ""
    eval_split: float = 0.1
    chat_template: str = "chatml"

    # Output
    output_dir: str = "checkpoints"
    save_steps: int = 100
    eval_steps: int = 100
    logging_steps: int = 10

    # Wandb
    wandb_project: str = "swarm-trainer"
    wandb_run_name: str = ""

    # Expert metadata
    expert_name: str = ""
    expert_role: str = ""


@dataclass
class RLConfig:
    # Model
    base_model: str = "Qwen/Qwen3.5-0.8B"
    adapter_path: str = ""
    max_seq_length: int = 2048
    load_in_4bit: bool = True

    # GRPO
    episodes: int = 100
    trajectories_per_task: int = 8
    max_steps_per_trajectory: int = 10
    clip_high: float = 0.2
    clip_low: float = 0.2
    no_kl: bool = True
    length_normalize: bool = True
    gamma: float = 1.0
    gae_lambda: float = 0.95

    # Training
    batch_size: int = 2
    learning_rate: float = 5e-5
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 4

    # Reward
    judge_model: str = "claude-sonnet-4-20250514"
    reward_weights: dict = field(default_factory=lambda: {
        "correctness": 0.5,
        "quality": 0.3,
        "efficiency": 0.2,
    })

    # Environment
    swarm_runner_url: str = "http://localhost:8100"
    codegate_url: str = "http://localhost:9212"
    codegate_api_key: str = "cgk_xxx"

    # Output
    output_dir: str = "checkpoints/rl"
    logging_steps: int = 1
    save_episodes: int = 10
    wandb_project: str = "swarm-trainer-rl"
    wandb_run_name: str = ""

    # Expert
    expert_name: str = ""


@dataclass
class RouterConfig:
    base_model: str = "Qwen/Qwen3.5-0.8B"
    max_seq_length: int = 512
    load_in_4bit: bool = True
    num_labels: int = 10
    label_names: list[str] = field(default_factory=list)
    dataset_path: str = ""
    num_epochs: int = 5
    batch_size: int = 8
    learning_rate: float = 2e-4
    lora: LoraConfig = field(default_factory=lambda: LoraConfig(rank=16, alpha=32))
    output_dir: str = "checkpoints/router"
    wandb_project: str = "swarm-trainer-router"


def _merge_dict_into_dataclass(dc: Any, data: dict) -> Any:
    """Recursively merge a dict into a dataclass instance."""
    for key, value in data.items():
        if not hasattr(dc, key):
            continue
        current = getattr(dc, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dict_into_dataclass(current, value)
        else:
            setattr(dc, key, value)
    return dc


def load_training_config(config_path: str | Path) -> TrainingConfig:
    """Load training config from a YAML file, with env var overrides."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    config = TrainingConfig()
    _merge_dict_into_dataclass(config, data)

    # Env var overrides
    if os.environ.get("BASE_MODEL"):
        config.base_model = os.environ["BASE_MODEL"]
    if os.environ.get("DATASET_DIR") and config.dataset_path:
        # Resolve relative paths against DATASET_DIR
        ds = Path(config.dataset_path)
        if not ds.is_absolute():
            config.dataset_path = str(Path(os.environ["DATASET_DIR"]) / ds)
    if os.environ.get("LORA_OUTPUT_DIR"):
        config.output_dir = os.environ["LORA_OUTPUT_DIR"]

    return config


def load_rl_config(config_path: str | Path) -> RLConfig:
    """Load RL config from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    config = RLConfig()
    _merge_dict_into_dataclass(config, data)

    if os.environ.get("BASE_MODEL"):
        config.base_model = os.environ["BASE_MODEL"]
    if os.environ.get("CODEGATE_URL"):
        config.codegate_url = os.environ["CODEGATE_URL"]
    if os.environ.get("CODEGATE_API_KEY"):
        config.codegate_api_key = os.environ["CODEGATE_API_KEY"]
    if os.environ.get("SWARM_RUNNER_URL"):
        config.swarm_runner_url = os.environ["SWARM_RUNNER_URL"]

    return config


def load_router_config(config_path: str | Path) -> RouterConfig:
    """Load router config from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    config = RouterConfig()
    _merge_dict_into_dataclass(config, data)

    if os.environ.get("BASE_MODEL"):
        config.base_model = os.environ["BASE_MODEL"]

    return config
