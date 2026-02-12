# LLM Fine-Tuning Toolkit

A toolkit for fine-tuning open-source LLMs with QLoRA on custom data. Includes a data collection pipeline (originally built for Temporal.io) and a generic training script that works with any JSONL dataset.

## Project Structure

```
temporal/
├── config.py              # Pipeline configuration
├── pipeline.py            # Data collection orchestrator
├── train.py               # Generic QLoRA training script
├── test_model.py          # Model testing script
├── requirements.txt       # Dependencies
├── collectors/
│   ├── __init__.py
│   ├── base.py            # Base collector class
│   ├── docs_scraper.py    # Temporal docs scraper
│   └── github_collector.py # GitHub code collector
├── processors/
│   ├── __init__.py
│   ├── qa_generator.py    # Q&A pair generation
│   └── formatter.py       # Training data formatting
└── data/                  # Generated data (created by pipeline)
    ├── raw/               # Raw collected documents
    ├── processed/         # Processed Q&A pairs and training data
    └── cache/             # Request cache
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train on Your Own Data

The training script accepts any JSONL file. Your data should use one of these formats:

**Messages format** (used as-is):
```json
{"messages": [{"role": "user", "content": "What is Python?"}, {"role": "assistant", "content": "A programming language."}]}
```

**Instruction/output format** (auto-converted to messages):
```json
{"instruction": "What is Python?", "output": "A programming language."}
```

Then run:

```bash
# Basic training (defaults to Qwen3-1.7B)
python train.py --train-data ./my_data.jsonl

# With validation data
python train.py --train-data ./train.jsonl --val-data ./val.jsonl

# With a custom system prompt (for instruction/output format)
python train.py --train-data ./train.jsonl \
  --system-prompt "You are a helpful coding assistant."

# Use a different base model
python train.py --train-data ./train.jsonl \
  --model meta-llama/Llama-3.2-1B-Instruct

# Full custom configuration
python train.py \
  --train-data ./train.jsonl \
  --val-data ./val.jsonl \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --system-prompt "You are a domain expert." \
  --epochs 5 \
  --batch-size 2 \
  --lora-r 32 \
  --output-dir ./my-model
```

### Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--train-data` | *(required)* | Path to training data JSONL |
| `--val-data` | `None` | Path to validation data JSONL |
| `--model` | `Qwen/Qwen3-1.7B` | Hugging Face model ID |
| `--output-dir` | `./finetuned-model` | Where to save the model |
| `--system-prompt` | `None` | System prompt for instruction/output data |
| `--epochs` | `3` | Number of training epochs |
| `--batch-size` | `4` | Batch size per device |
| `--gradient-accumulation` | `4` | Gradient accumulation steps |
| `--lr` | `2e-4` | Learning rate |
| `--max-seq-length` | `2048` | Maximum sequence length |
| `--lora-r` | `16` | LoRA rank |
| `--lora-alpha` | `32` | LoRA alpha |
| `--use-8bit` | `False` | Use 8-bit instead of 4-bit quantization |
| `--flash-attention` | `False` | Use Flash Attention 2 |

## Data Collection Pipeline (Temporal.io)

The project also includes a data pipeline for collecting Temporal.io documentation and code to generate training data. This can serve as a reference for building your own collection pipelines.

### Set Up Environment Variables

Create a `.env` file for API keys:

```bash
# For higher GitHub rate limits
GITHUB_TOKEN=your_github_token

# For LLM-based Q&A generation (optional)
ANTHROPIC_API_KEY=your_anthropic_key
# OR
OPENAI_API_KEY=your_openai_key
```

### Run the Pipeline

```bash
# Full pipeline (collect docs + GitHub + generate Q&A)
python pipeline.py

# Skip documentation collection (if already cached)
python pipeline.py --skip-docs

# Skip GitHub collection
python pipeline.py --skip-github

# Limit GitHub files (useful for testing)
python pipeline.py --github-limit 50

# Use LLM for Q&A generation (higher quality, requires API key)
python pipeline.py --use-llm

# Specify output formats
python pipeline.py --formats trl torchtune alpaca
```

### Output Formats

| Format | Use With | File Structure |
|--------|----------|----------------|
| `trl` | Hugging Face TRL/SFTTrainer | `{"messages": [...]}` |
| `torchtune` | PyTorch torchtune | `{"instruction": ..., "output": ...}` |
| `alpaca` | Stanford Alpaca format | `{"instruction": ..., "output": ...}` |
| `axolotl` | Axolotl trainer | `{"conversations": [...]}` |

## QLoRA Settings

Default configuration uses QLoRA for memory efficiency:

```python
# 4-bit quantization
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# LoRA config
LoraConfig(
    r=16,           # Rank
    lora_alpha=32,  # Alpha
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
```

## Training Tips

1. **Start small**: Use a 1-2B model first to validate your data
2. **Monitor loss**: Training loss should decrease steadily
3. **Validation matters**: Keep 10% data for validation to catch overfitting
4. **Quality > Quantity**: 1,000 high-quality examples often beats 10,000 noisy ones

## Troubleshooting

### Out of Memory

- Reduce `--batch-size` to 1 or 2
- Increase `--gradient-accumulation` to compensate
- Use `--use-8bit` instead of 4-bit (slightly more memory but sometimes more stable)

### GitHub Rate Limits

- Set `GITHUB_TOKEN` environment variable
- Use `--github-limit` to collect fewer files
- Pipeline caches requests for 24 hours

### Poor Model Quality

- Increase training epochs
- Use `--use-llm` for higher quality Q&A pairs
- Review and clean the generated data manually
- Consider using a larger base model

## License

MIT License - Feel free to use and modify for your projects.
