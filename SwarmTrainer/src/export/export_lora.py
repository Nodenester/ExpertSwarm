"""Export trained LoRA adapters to SwarmRunner.

Copies adapter files and optionally hot-loads them via the SwarmRunner API.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import httpx


ADAPTER_FILES = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]


def copy_adapter(src_dir: Path, dst_dir: Path, expert_name: str) -> Path:
    """Copy adapter files from training output to SwarmRunner loras directory."""
    target = dst_dir / expert_name
    target.mkdir(parents=True, exist_ok=True)

    copied = []
    for fname in ADAPTER_FILES:
        src = src_dir / fname
        if src.exists():
            shutil.copy2(src, target / fname)
            copied.append(fname)

    # Also copy any other .safetensors or .bin files (multi-shard adapters)
    for pattern in ("*.safetensors", "*.bin"):
        for f in src_dir.glob(pattern):
            if f.name not in copied:
                shutil.copy2(f, target / f.name)
                copied.append(f.name)

    if not copied:
        raise FileNotFoundError(f"No adapter files found in {src_dir}")

    print(f"  Copied {len(copied)} files to {target}")
    return target


def hot_load_adapter(
    runner_url: str,
    expert_name: str,
    adapter_path: str,
    timeout_s: int = 60,
) -> bool:
    """Tell SwarmRunner to hot-load a new/updated LoRA adapter."""
    client = httpx.Client(timeout=timeout_s)
    try:
        resp = client.post(
            f"{runner_url}/api/v1/loras/load",
            json={
                "name": expert_name,
                "path": adapter_path,
            },
        )
        resp.raise_for_status()
        result = resp.json()
        print(f"  Hot-load response: {json.dumps(result)}")
        return result.get("status") == "ok"
    except httpx.HTTPError as e:
        print(f"  [ERROR] Hot-load failed: {e}", file=sys.stderr)
        return False
    finally:
        client.close()


def verify_adapter(runner_url: str, expert_name: str) -> bool:
    """Verify the adapter is loaded and working in SwarmRunner."""
    client = httpx.Client(timeout=30)
    try:
        # Check loaded adapters
        resp = client.get(f"{runner_url}/api/v1/loras")
        resp.raise_for_status()
        loras = resp.json()
        loaded_names = [l.get("name") for l in loras.get("adapters", loras.get("loras", []))]

        if expert_name not in loaded_names:
            print(f"  [WARN] Adapter '{expert_name}' not in loaded list: {loaded_names}")
            return False

        # Quick inference test
        resp = client.post(
            f"{runner_url}/v1/chat/completions",
            json={
                "model": expert_name,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 10,
            },
        )
        resp.raise_for_status()
        result = resp.json()
        if result.get("choices"):
            print(f"  Verification passed: adapter responds correctly")
            return True
        else:
            print(f"  [WARN] Unexpected response: {result}")
            return False

    except httpx.HTTPError as e:
        print(f"  [ERROR] Verification failed: {e}", file=sys.stderr)
        return False
    finally:
        client.close()


def export(
    src_dir: str,
    expert_name: str,
    dst_dir: str | None = None,
    runner_url: str | None = None,
    hot_load: bool = False,
    verify: bool = False,
):
    """Export a trained LoRA adapter."""
    src = Path(src_dir)
    if not src.exists():
        raise FileNotFoundError(f"Source directory not found: {src}")

    print(f"=== Exporting LoRA: {expert_name} ===")
    print(f"  Source: {src}")

    # Copy to destination
    dst = Path(dst_dir) if dst_dir else Path(os.environ.get("LORA_OUTPUT_DIR", "loras"))
    target_path = copy_adapter(src, dst, expert_name)

    # Hot-load if requested
    if hot_load:
        url = runner_url or os.environ.get("SWARM_RUNNER_URL", "http://localhost:8100")
        print(f"  Hot-loading to {url}...")
        success = hot_load_adapter(url, expert_name, str(target_path))
        if not success:
            print(f"  [WARN] Hot-load failed, adapter copied but not loaded", file=sys.stderr)
            return

        # Verify
        if verify:
            print(f"  Verifying...")
            if verify_adapter(url, expert_name):
                print(f"  Export complete and verified!")
            else:
                print(f"  [WARN] Verification failed", file=sys.stderr)
    else:
        print(f"  Export complete (no hot-load). Adapter at: {target_path}")


def main():
    parser = argparse.ArgumentParser(description="Export trained LoRA to SwarmRunner")
    parser.add_argument("--src", required=True, help="Source adapter directory")
    parser.add_argument("--expert", required=True, help="Expert name")
    parser.add_argument("--dst", default=None, help="Destination loras directory")
    parser.add_argument("--runner-url", default=None, help="SwarmRunner URL for hot-load")
    parser.add_argument("--hot-load", action="store_true", help="Hot-load adapter into running SwarmRunner")
    parser.add_argument("--verify", action="store_true", help="Verify adapter works after loading")
    args = parser.parse_args()

    export(
        src_dir=args.src,
        expert_name=args.expert,
        dst_dir=args.dst,
        runner_url=args.runner_url,
        hot_load=args.hot_load,
        verify=args.verify,
    )


if __name__ == "__main__":
    main()
