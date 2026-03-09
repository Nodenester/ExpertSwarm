"""Quick inference test for a trained LoRA expert.

Usage:
    python -m src.eval.test_inference --adapter /checkpoints/fact_extractor/final
"""

import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def test_inference(adapter_path: str, base_model: str = None):
    """Load base model + LoRA adapter and run test prompts."""

    # Read base model from adapter config if not specified
    if not base_model:
        with open(f"{adapter_path}/adapter_config.json") as f:
            cfg = json.load(f)
        base_model = cfg.get("base_model_name_or_path", "Qwen/Qwen2.5-0.5B-Instruct")
        # Unsloth stores the unsloth variant; map back to original
        if "unsloth/" in base_model:
            base_model = "Qwen/Qwen2.5-0.5B-Instruct"

    print(f"Loading base model: {base_model}")
    print(f"Loading adapter: {adapter_path}")

    # Load base model in 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    print(f"Model loaded on: {model.device}")

    # Test prompts
    test_cases = [
        {
            "name": "News article extraction",
            "prompt": (
                "Extract all key facts from this news snippet.\n\n"
                "SpaceX successfully launched 23 Starlink satellites from Cape Canaveral "
                "on March 1, 2026. The Falcon 9 booster completed its 20th landing on "
                "the drone ship 'Just Read the Instructions' in the Atlantic Ocean. "
                "CEO Elon Musk confirmed the next launch is scheduled for March 5."
            ),
        },
        {
            "name": "Product spec extraction",
            "prompt": (
                "Extract specifications and pricing from this product listing.\n\n"
                "Samsung Galaxy S25 Ultra - $1,299.99\n"
                "Display: 6.9\" Dynamic AMOLED 2X, 3120x1440, 120Hz\n"
                "Processor: Snapdragon 8 Elite\n"
                "RAM: 12GB | Storage: 256GB/512GB/1TB\n"
                "Battery: 5000mAh, 45W wired charging\n"
                "Camera: 200MP main + 50MP ultrawide + 50MP 5x telephoto"
            ),
        },
        {
            "name": "Meeting notes extraction",
            "prompt": (
                "Extract action items and decisions from these meeting notes.\n\n"
                "Q1 Planning Meeting - Jan 15, 2026\n"
                "Attendees: Lisa Chen (PM), Mark Davis (Eng Lead), Sarah Kim (Design)\n"
                "Decision: Migrate from PostgreSQL to CockroachDB by end of Q2.\n"
                "Action: Mark to draft migration plan by Jan 22.\n"
                "Action: Sarah to redesign the dashboard — mockups due Feb 1.\n"
                "Budget approved: $45,000 for cloud infrastructure upgrades.\n"
                "Next meeting: Jan 29, 2026 at 2pm EST."
            ),
        },
    ]

    print(f"\n{'='*60}")
    print("INFERENCE TEST RESULTS")
    print(f"{'='*60}")

    for i, tc in enumerate(test_cases, 1):
        messages = [{"role": "user", "content": tc["prompt"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = outputs[0][prompt_len:]
        response = tokenizer.decode(generated, skip_special_tokens=True)

        print(f"\n--- Test {i}: {tc['name']} ---")
        print(f"Response ({len(generated)} tokens):\n")
        print(response[:2000])

        # Try to parse as JSON to check structure
        try:
            parsed = json.loads(response)
            n_facts = len(parsed.get("facts", []))
            has_summary = "summary" in parsed
            print(f"\n  -> Valid JSON! {n_facts} facts extracted, summary: {has_summary}")
        except json.JSONDecodeError:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(response[start:end])
                    n_facts = len(parsed.get("facts", []))
                    has_summary = "summary" in parsed
                    print(f"\n  -> JSON found (with prefix). {n_facts} facts, summary: {has_summary}")
                except json.JSONDecodeError:
                    print(f"\n  -> NOT valid JSON output")
            else:
                print(f"\n  -> NOT valid JSON output")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Test LoRA expert inference")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter directory")
    parser.add_argument("--base-model", default=None, help="Base model name (auto-detected from adapter config)")
    args = parser.parse_args()
    test_inference(args.adapter, args.base_model)


if __name__ == "__main__":
    main()
