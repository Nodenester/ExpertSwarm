"""QLoRA training for a single swarm expert.

Usage:
    python -m src.train.train_expert --config configs/experts/fact-ext.yaml
"""

import argparse
import os
import sys
from pathlib import Path

import unsloth  # Must be imported before transformers/trl/peft
import torch
import wandb
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

from .config import load_training_config


def load_and_prepare_data(config, tokenizer):
    """Load JSONL dataset, format with chat template, and split into train/eval."""
    ds_path = config.dataset_path
    if not ds_path:
        raise ValueError("dataset_path is required in config")

    dataset = load_dataset("json", data_files=ds_path, split="train")

    # Pre-format: apply chat template to create a "text" column
    def _format(example):
        messages = example.get("messages", [])
        if not messages:
            messages = [
                {"role": "user", "content": example.get("input", "")},
                {"role": "assistant", "content": example.get("output", "")},
            ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    dataset = dataset.map(_format, remove_columns=dataset.column_names)

    if config.eval_split > 0:
        split = dataset.train_test_split(test_size=config.eval_split, seed=42)
        return split["train"], split["test"]
    return dataset, None


def train(config_path: str):
    """Run QLoRA training for one expert."""
    config = load_training_config(config_path)

    print(f"=== Training expert: {config.expert_name} ===")
    print(f"  Base model: {config.base_model}")
    print(f"  Dataset: {config.dataset_path}")
    print(f"  LoRA rank: {config.lora.rank}, alpha: {config.lora.alpha}")
    print(f"  Epochs: {config.num_epochs}, batch: {config.batch_size}")

    # Initialize wandb
    run_name = config.wandb_run_name or f"sft-{config.expert_name}"
    wandb.init(project=config.wandb_project, name=run_name, config={
        "expert": config.expert_name,
        "base_model": config.base_model,
        "lora_rank": config.lora.rank,
        "lora_alpha": config.lora.alpha,
        "epochs": config.num_epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
    })

    # Load model with Unsloth (4-bit quantized)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        dtype=getattr(torch, config.dtype) if config.dtype != "auto" else None,
    )

    # Apply LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        bias=config.lora.bias,
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Print trainable parameter count
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Load data
    train_dataset, eval_dataset = load_and_prepare_data(config, tokenizer)
    print(f"  Train examples: {len(train_dataset)}")
    if eval_dataset:
        print(f"  Eval examples: {len(eval_dataset)}")

    # Output directory
    output_dir = Path(config.output_dir) / config.expert_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
        warmup_ratio=float(config.warmup_ratio),
        lr_scheduler_type=config.lr_scheduler,
        max_grad_norm=float(config.max_grad_norm),
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=config.eval_steps if eval_dataset else None,
        save_total_limit=3,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="wandb",
        run_name=run_name,
        seed=42,
        dataloader_num_workers=2,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_seq_length=config.max_seq_length,
        args=training_args,
        packing=False,
    )

    # Train
    print("\nStarting training...")
    train_result = trainer.train()

    # Log final metrics
    print(f"\n=== Training complete ===")
    print(f"  Train loss: {train_result.training_loss:.4f}")
    print(f"  Train runtime: {train_result.metrics['train_runtime']:.1f}s")
    print(f"  Samples/sec: {train_result.metrics['train_samples_per_second']:.1f}")

    # Save final adapter
    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"  Saved adapter to: {final_dir}")

    # Also save as merged 16-bit for optional full-weight export
    if os.environ.get("EXPORT_MERGED"):
        merged_dir = output_dir / "merged_16bit"
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
        print(f"  Saved merged model to: {merged_dir}")

    wandb.finish()
    print(f"\nDone. Expert '{config.expert_name}' trained successfully.")


def main():
    parser = argparse.ArgumentParser(description="Train a swarm expert with QLoRA")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
