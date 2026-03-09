"""Train the meta-router LoRA — a 1-token classification model.

The router maps user requests to LoRA expert names.
Input: user message -> Output: expert LoRA name (1-token classification).

Usage:
    python -m src.train.train_router --config configs/router.yaml
"""

import argparse
import json
import os
from pathlib import Path

import torch
import wandb
from datasets import load_dataset, Dataset
from transformers import (
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

from .config import load_router_config


def prepare_router_dataset(config) -> tuple:
    """Load and prepare router classification dataset.

    Expected JSONL format:
    {"input": "user message", "label": "expert_name"}
    """
    dataset = load_dataset("json", data_files=config.dataset_path, split="train")

    # Build label mapping
    if config.label_names:
        label2id = {name: i for i, name in enumerate(config.label_names)}
    else:
        unique_labels = sorted(set(dataset["label"]))
        label2id = {name: i for i, name in enumerate(unique_labels)}
        config.label_names = list(label2id.keys())
        config.num_labels = len(label2id)

    id2label = {v: k for k, v in label2id.items()}

    # Map string labels to integer IDs
    def map_labels(example):
        example["label"] = label2id[example["label"]]
        return example

    dataset = dataset.map(map_labels)

    # Split
    split = dataset.train_test_split(test_size=0.15, seed=42, stratify_by_column="label")
    return split["train"], split["test"], label2id, id2label


def train(config_path: str):
    """Train the meta-router as a sequence classification model with LoRA."""
    config = load_router_config(config_path)

    print(f"=== Training Meta-Router ===")
    print(f"  Base model: {config.base_model}")
    print(f"  Dataset: {config.dataset_path}")
    print(f"  Num labels: {config.num_labels}")

    # Load data
    train_dataset, eval_dataset, label2id, id2label = prepare_router_dataset(config)
    print(f"  Train examples: {len(train_dataset)}")
    print(f"  Eval examples: {len(eval_dataset)}")
    print(f"  Labels: {list(label2id.keys())}")

    # Initialize wandb
    wandb.init(project=config.wandb_project, name="router-train", config={
        "base_model": config.base_model,
        "num_labels": config.num_labels,
        "labels": config.label_names,
        "epochs": config.num_epochs,
    })

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model,
        num_labels=config.num_labels,
        id2label=id2label,
        label2id=label2id,
        torch_dtype=torch.float16,
        load_in_4bit=config.load_in_4bit,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # Apply LoRA
    peft_config = PeftLoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        bias=config.lora.bias,
        task_type=TaskType.SEQ_CLS,
        modules_to_save=["score"],  # Save the classification head
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Tokenize datasets
    def tokenize(examples):
        return tokenizer(
            examples["input"],
            truncation=True,
            max_length=config.max_seq_length,
            padding="max_length",
        )

    train_tokenized = train_dataset.map(tokenize, batched=True, remove_columns=["input"])
    eval_tokenized = eval_dataset.map(tokenize, batched=True, remove_columns=["input"])

    # Training arguments
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size * 2,
        learning_rate=config.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="wandb",
        seed=42,
    )

    # Compute metrics
    import numpy as np

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    # Train
    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        compute_metrics=compute_metrics,
    )

    print("\nStarting router training...")
    result = trainer.train()

    print(f"\n=== Router training complete ===")
    print(f"  Train loss: {result.training_loss:.4f}")

    # Evaluate
    eval_results = trainer.evaluate()
    print(f"  Eval accuracy: {eval_results.get('eval_accuracy', 'N/A')}")

    # Save
    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Save label mapping
    mapping = {"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}
    (final_dir / "label_mapping.json").write_text(json.dumps(mapping, indent=2))

    print(f"  Saved router to: {final_dir}")
    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train the meta-router LoRA")
    parser.add_argument("--config", required=True, help="Path to router YAML config")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
