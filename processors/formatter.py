"""Format data for different fine-tuning frameworks."""

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .qa_generator import QAPair

logger = logging.getLogger(__name__)


@dataclass
class FormattedExample:
    """A formatted training example."""

    text: str
    metadata: dict


class DataFormatter:
    """Format Q&A pairs for different fine-tuning frameworks."""

    # Chat templates for different models
    TEMPLATES = {
        "llama3": {
            "system": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>",
            "user": "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
            "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>",
        },
        "mistral": {
            "system": "",
            "user": "[INST] {content} [/INST]",
            "assistant": "{content}</s>",
        },
        "chatml": {
            "system": "<|im_start|>system\n{system}<|im_end|>\n",
            "user": "<|im_start|>user\n{content}<|im_end|>\n",
            "assistant": "<|im_start|>assistant\n{content}<|im_end|>\n",
        },
        "alpaca": {
            "template": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{response}",
        },
    }

    DEFAULT_SYSTEM_PROMPT = """You are an expert on Temporal.io, a workflow orchestration platform. You provide accurate, helpful information about Temporal concepts, best practices, and code examples. When discussing code, you consider the specific SDK being used (Python, TypeScript, Go, or Java)."""

    def __init__(
        self,
        template_name: str = "llama3",
        system_prompt: Optional[str] = None,
        train_split: float = 0.9,
    ):
        """
        Initialize the formatter.

        Args:
            template_name: Name of the chat template to use
            system_prompt: Custom system prompt (optional)
            train_split: Fraction of data for training (rest is validation)
        """
        self.template_name = template_name
        self.template = self.TEMPLATES.get(template_name, self.TEMPLATES["llama3"])
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.train_split = train_split

    def format_single(self, qa: QAPair) -> FormattedExample:
        """Format a single Q&A pair."""
        if self.template_name == "alpaca":
            text = self.template["template"].format(
                instruction=qa.instruction, response=qa.response
            )
        else:
            parts = []

            # Add system prompt
            if self.template.get("system"):
                parts.append(
                    self.template["system"].format(system=self.system_prompt)
                )

            # Add user message
            parts.append(self.template["user"].format(content=qa.instruction))

            # Add assistant response
            parts.append(self.template["assistant"].format(content=qa.response))

            text = "".join(parts)

        return FormattedExample(
            text=text,
            metadata={
                "source_id": qa.source_id,
                "category": qa.category,
                "concepts": qa.concepts,
            },
        )

    def format_for_trl(self, qa: QAPair) -> dict:
        """Format for Hugging Face TRL SFTTrainer (messages format)."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": qa.instruction},
            {"role": "assistant", "content": qa.response},
        ]
        return {
            "messages": messages,
            "source_id": qa.source_id,
            "category": qa.category,
        }

    def format_for_torchtune(self, qa: QAPair) -> dict:
        """Format for torchtune instruct dataset."""
        return {
            "instruction": qa.instruction,
            "input": "",  # torchtune expects this field
            "output": qa.response,
            "source_id": qa.source_id,
        }

    def format_for_axolotl(self, qa: QAPair) -> dict:
        """Format for Axolotl sharegpt format."""
        return {
            "conversations": [
                {"from": "system", "value": self.system_prompt},
                {"from": "human", "value": qa.instruction},
                {"from": "gpt", "value": qa.response},
            ]
        }

    def process_and_save(
        self,
        qa_pairs: list[QAPair],
        output_dir: Path,
        formats: list[str] = None,
    ) -> dict:
        """
        Process Q&A pairs and save in multiple formats.

        Args:
            qa_pairs: List of Q&A pairs to format
            output_dir: Directory to save formatted data
            formats: List of formats to generate. Options:
                     'trl', 'torchtune', 'axolotl', 'raw', 'alpaca'

        Returns:
            Dict with paths to generated files
        """
        if formats is None:
            formats = ["trl", "torchtune", "raw"]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Shuffle and split data
        random.shuffle(qa_pairs)
        split_idx = int(len(qa_pairs) * self.train_split)
        train_pairs = qa_pairs[:split_idx]
        val_pairs = qa_pairs[split_idx:]

        output_paths = {}

        for fmt in formats:
            logger.info(f"Generating {fmt} format...")

            if fmt == "trl":
                train_data = [self.format_for_trl(qa) for qa in train_pairs]
                val_data = [self.format_for_trl(qa) for qa in val_pairs]
            elif fmt == "torchtune":
                train_data = [self.format_for_torchtune(qa) for qa in train_pairs]
                val_data = [self.format_for_torchtune(qa) for qa in val_pairs]
            elif fmt == "axolotl":
                train_data = [self.format_for_axolotl(qa) for qa in train_pairs]
                val_data = [self.format_for_axolotl(qa) for qa in val_pairs]
            elif fmt == "raw":
                train_data = [qa.to_dict() for qa in train_pairs]
                val_data = [qa.to_dict() for qa in val_pairs]
            elif fmt == "alpaca":
                train_data = [
                    {"instruction": qa.instruction, "input": "", "output": qa.response}
                    for qa in train_pairs
                ]
                val_data = [
                    {"instruction": qa.instruction, "input": "", "output": qa.response}
                    for qa in val_pairs
                ]
            else:
                logger.warning(f"Unknown format: {fmt}")
                continue

            # Save files
            fmt_dir = output_dir / fmt
            fmt_dir.mkdir(exist_ok=True)

            train_path = fmt_dir / "train.jsonl"
            val_path = fmt_dir / "validation.jsonl"

            self._save_jsonl(train_data, train_path)
            self._save_jsonl(val_data, val_path)

            output_paths[fmt] = {
                "train": str(train_path),
                "validation": str(val_path),
                "train_count": len(train_data),
                "val_count": len(val_data),
            }

        # Save dataset info
        info = {
            "total_examples": len(qa_pairs),
            "train_examples": len(train_pairs),
            "validation_examples": len(val_pairs),
            "formats": output_paths,
            "template": self.template_name,
        }

        info_path = output_dir / "dataset_info.json"
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

        logger.info(f"Dataset info saved to {info_path}")
        return output_paths

    def _save_jsonl(self, data: list[dict], path: Path):
        """Save data as JSONL file."""
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(data)} examples to {path}")

    def create_huggingface_dataset(
        self, qa_pairs: list[QAPair], output_dir: Path
    ) -> Path:
        """Create a dataset in Hugging Face datasets format."""
        try:
            from datasets import Dataset, DatasetDict
        except ImportError:
            logger.error("datasets package not installed")
            return None

        # Format data
        train_data = []
        val_data = []

        random.shuffle(qa_pairs)
        split_idx = int(len(qa_pairs) * self.train_split)

        for i, qa in enumerate(qa_pairs):
            formatted = self.format_for_trl(qa)
            if i < split_idx:
                train_data.append(formatted)
            else:
                val_data.append(formatted)

        # Create dataset
        dataset = DatasetDict(
            {
                "train": Dataset.from_list(train_data),
                "validation": Dataset.from_list(val_data),
            }
        )

        # Save to disk
        output_path = output_dir / "hf_dataset"
        dataset.save_to_disk(str(output_path))
        logger.info(f"Saved Hugging Face dataset to {output_path}")

        return output_path


def main():
    """Test the formatter."""
    from config import Config

    config = Config()

    # Create sample Q&A pairs
    sample_pairs = [
        QAPair(
            instruction="What is a Temporal Workflow?",
            response="A Temporal Workflow is a durable, reliable function execution that defines a sequence of steps. Workflows can run for extended periods and automatically recover from failures.",
            source_id="test:1",
            source_url="https://docs.temporal.io/workflows",
            category="concept",
            concepts=["workflow"],
        ),
        QAPair(
            instruction="How do I create an Activity in Python?",
            response='In Python, you define an Activity using the @activity.defn decorator:\n\n```python\nfrom temporalio import activity\n\n@activity.defn\nasync def my_activity(param: str) -> str:\n    return f"Processed: {param}"\n```',
            source_id="test:2",
            source_url="https://docs.temporal.io/develop/python/activities",
            category="code",
            concepts=["activity"],
        ),
    ]

    formatter = DataFormatter(template_name="llama3")

    # Test single formatting
    print("Formatted example (llama3):")
    example = formatter.format_single(sample_pairs[0])
    print(example.text[:500])
    print("\n" + "=" * 50 + "\n")

    # Test TRL format
    print("TRL format:")
    trl_example = formatter.format_for_trl(sample_pairs[0])
    print(json.dumps(trl_example, indent=2))


if __name__ == "__main__":
    main()
