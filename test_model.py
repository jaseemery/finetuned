"""Test a fine-tuned model."""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model(base_model_id: str, adapter_path: str):
    """Load the fine-tuned model."""
    print(f"Loading base model: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    question: str,
    system_prompt: str = None,
    max_new_tokens: int = 512,
):
    """Generate a response to a question."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


def main():
    parser = argparse.ArgumentParser(description="Test a fine-tuned model")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="Base model ID",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="./finetuned-model",
        help="Path to the fine-tuned adapter",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Optional system prompt",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max new tokens to generate",
    )

    args = parser.parse_args()

    model, tokenizer = load_model(args.model, args.adapter)

    print("\n" + "=" * 60)
    print("INTERACTIVE MODEL TEST (type 'quit' to exit)")
    print("=" * 60)

    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        response = generate_response(
            model, tokenizer, question,
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_tokens,
        )
        print(f"\nModel: {response}")


if __name__ == "__main__":
    main()
