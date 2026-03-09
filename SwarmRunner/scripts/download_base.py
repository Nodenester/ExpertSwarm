#!/usr/bin/env python3
"""
Download the base model from HuggingFace to the local model directory.

Usage:
    python scripts/download_base.py

Environment variables:
    SWARM_BASE_MODEL  - HuggingFace model ID (default: Qwen/Qwen3.5-0.8B-GPTQ-Int4)
    SWARM_MODEL_DIR   - Local directory to download into (default: ./models)
    HF_TOKEN          - HuggingFace API token (optional, for gated models)
"""

import os
import sys


def main():
    model_id = os.environ.get("SWARM_BASE_MODEL", "Qwen/Qwen3.5-0.8B-GPTQ-Int4")
    model_dir = os.environ.get("SWARM_MODEL_DIR", "./models")
    hf_token = os.environ.get("HF_TOKEN")

    print(f"Downloading model: {model_id}")
    print(f"Target directory:  {model_dir}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface-hub")
        sys.exit(1)

    os.makedirs(model_dir, exist_ok=True)

    local_path = snapshot_download(
        repo_id=model_id,
        local_dir=os.path.join(model_dir, model_id.replace("/", "--")),
        token=hf_token,
        resume_download=True,
    )

    print(f"Download complete: {local_path}")
    return local_path


if __name__ == "__main__":
    main()
