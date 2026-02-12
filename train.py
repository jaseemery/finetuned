
import argparse
import logging
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_training_data(data_path: str, split: str = "train"):
    """Load training data from JSONL file."""
    dataset = load_dataset("json", data_files=data_path, split=split)
    return dataset


def create_bnb_config(use_4bit: bool = True):
    """Create BitsAndBytes quantization config."""
    if use_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    return BitsAndBytesConfig(
        load_in_8bit=True,
    )


def create_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    target_modules: list = None,
):
    """Create LoRA configuration."""
    if target_modules is None:
        # Common target modules for most models
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


def format_messages(example, system_prompt: str = None):
    """Format messages for training."""
    # Handle TRL format with messages
    if "messages" in example:
        return example

    # Handle instruction/response format
    if "instruction" in example and "output" in example:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": example["instruction"]})
        messages.append({"role": "assistant", "content": example["output"]})
        return {"messages": messages}

    return example


def train(
    model_id: str,
    train_data_path: str,
    val_data_path: str = None,
    output_dir: str = "./finetuned-model",
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048,
    use_4bit: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    use_flash_attention: bool = False,
    system_prompt: str = None,
):
    """
    Fine-tune an LLM with QLoRA on custom data.

    Args:
        model_id: Hugging Face model ID (e.g., "meta-llama/Llama-3.2-1B-Instruct")
        train_data_path: Path to training data JSONL
        val_data_path: Path to validation data JSONL (optional)
        output_dir: Directory to save the fine-tuned model
        num_epochs: Number of training epochs
        batch_size: Training batch size per device
        gradient_accumulation_steps: Steps to accumulate gradients
        learning_rate: Learning rate
        max_seq_length: Maximum sequence length
        use_4bit: Use 4-bit quantization (QLoRA)
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        use_flash_attention: Use Flash Attention 2
        system_prompt: Optional system prompt for instruction/output formatted data
    """
    logger.info(f"Loading model: {model_id}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Quantization config
    bnb_config = create_bnb_config(use_4bit=use_4bit)

    # Load model
    model_kwargs = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    lora_config = create_lora_config(r=lora_r, lora_alpha=lora_alpha)
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    # Load datasets
    logger.info(f"Loading training data from: {train_data_path}")
    train_dataset = load_training_data(train_data_path)
    train_dataset = train_dataset.map(lambda ex: format_messages(ex, system_prompt))

    val_dataset = None
    if val_data_path and Path(val_data_path).exists():
        logger.info(f"Loading validation data from: {val_data_path}")
        val_dataset = load_training_data(val_data_path)
        val_dataset = val_dataset.map(lambda ex: format_messages(ex, system_prompt))

    # Training configuration
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=100 if val_dataset else None,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        packing=False,  # Disabled - requires flash attention
        report_to="tensorboard",
        remove_unused_columns=False,
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save the final model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("Training complete!")
    return trainer


def main():
    parser = argparse.ArgumentParser(description="Fine-tune an LLM with QLoRA on custom data")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to training data JSONL",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to validation data JSONL (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./finetuned-model",
        help="Output directory for fine-tuned model",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="System prompt to prepend when data uses instruction/output format",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--use-8bit", action="store_true", help="Use 8-bit instead of 4-bit")
    parser.add_argument(
        "--flash-attention", action="store_true", help="Use Flash Attention 2"
    )

    args = parser.parse_args()

    train(
        model_id=args.model,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
        use_4bit=not args.use_8bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_flash_attention=args.flash_attention,
        system_prompt=args.system_prompt,
    )


if __name__ == "__main__":
    main()
