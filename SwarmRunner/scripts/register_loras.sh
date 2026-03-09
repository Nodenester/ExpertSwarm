#!/usr/bin/env bash
#
# Hot-load LoRA adapters into a running vLLM instance.
#
# Scans SWARM_LORA_DIR for directories containing adapter_config.json
# and registers each one via the vLLM API.
#
# Usage:
#   ./scripts/register_loras.sh
#
# Environment variables:
#   SWARM_VLLM_PORT  - vLLM port (default: 8000)
#   SWARM_LORA_DIR   - Directory containing LoRA adapters (default: ./loras)

set -euo pipefail

VLLM_PORT="${SWARM_VLLM_PORT:-8000}"
LORA_DIR="${SWARM_LORA_DIR:-./loras}"
VLLM_URL="http://localhost:${VLLM_PORT}"

echo "Registering LoRAs from: ${LORA_DIR}"
echo "vLLM endpoint: ${VLLM_URL}"

# Wait for vLLM to be ready
echo "Waiting for vLLM health check..."
until curl -sf "${VLLM_URL}/health" > /dev/null 2>&1; do
    sleep 2
done
echo "vLLM is ready."

registered=0
failed=0

for adapter_dir in "${LORA_DIR}"/*/; do
    [ -d "${adapter_dir}" ] || continue

    # Check if this looks like a LoRA adapter
    if [ ! -f "${adapter_dir}/adapter_config.json" ] && [ ! -f "${adapter_dir}/adapter_model.safetensors" ]; then
        echo "SKIP: ${adapter_dir} (no adapter_config.json or adapter_model.safetensors)"
        continue
    fi

    # Use directory name as the LoRA name
    lora_name=$(basename "${adapter_dir}")
    lora_path=$(realpath "${adapter_dir}")

    echo "Registering LoRA: ${lora_name} from ${lora_path}"

    response=$(curl -sf -X POST "${VLLM_URL}/v1/load_lora_adapter" \
        -H "Content-Type: application/json" \
        -d "{\"lora_name\": \"${lora_name}\", \"lora_path\": \"${lora_path}\"}" \
        2>&1) || {
        echo "  FAILED: ${response}"
        failed=$((failed + 1))
        continue
    }

    echo "  OK: ${response}"
    registered=$((registered + 1))
done

echo ""
echo "Registration complete: ${registered} registered, ${failed} failed"
