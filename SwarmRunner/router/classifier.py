"""
SwarmRunner LoRA-based Request Classifier (future implementation)

This module will replace keyword-based routing with a tiny LoRA fine-tuned
for single-token classification. The classifier LoRA takes an incoming
request and outputs one token representing the target route.

Architecture:
  - Input: The user's last message (truncated to 512 tokens)
  - Output: Single token from a constrained vocabulary matching route names
  - Model: A 1-token classifier LoRA loaded alongside the specialist LoRAs
  - Latency target: <10ms per classification (single forward pass)

Integration:
  The router will call classify() instead of keyword matching when
  SWARM_USE_CLASSIFIER=true is set.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

CLASSIFIER_LORA_NAME = os.environ.get("SWARM_CLASSIFIER_LORA", "classifier-lora")
CLASSIFIER_ENABLED = os.environ.get("SWARM_USE_CLASSIFIER", "false").lower() == "true"


async def classify(text: str, vllm_base_url: str, route_names: list[str]) -> str | None:
    """
    Classify a request using the classifier LoRA.

    Args:
        text: The user message text to classify.
        vllm_base_url: Base URL of the vLLM backend.
        route_names: Valid route names to constrain output to.

    Returns:
        The predicted route name, or None if classification fails.

    Raises:
        NotImplementedError: Until the classifier LoRA is trained and deployed.
    """
    raise NotImplementedError(
        "LoRA-based classifier not yet implemented. "
        "Set SWARM_USE_CLASSIFIER=false to use keyword routing."
    )
