"""Test the fine-tuned Temporal LLM."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_model(base_model_id: str, adapter_path: str):
    """Load the fine-tuned model."""
    print(f"Loading base model: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    return model, tokenizer


def generate_response(model, tokenizer, question: str, max_new_tokens: int = 512):
    """Generate a response to a question."""
    messages = [
        {"role": "system", "content": "You are an expert on Temporal.io, a workflow orchestration platform."},
        {"role": "user", "content": question},
    ]

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
    base_model = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_path = "./temporal-llm"

    model, tokenizer = load_model(base_model, adapter_path)

    # Test questions about Temporal
    questions = [
        "What is a Temporal Workflow?",
        "How do Activities differ from Workflows in Temporal?",
        "What is a Task Queue in Temporal?",
        "Why must Temporal Workflows be deterministic?",
    ]

    print("\n" + "=" * 60)
    print("TESTING FINE-TUNED TEMPORAL LLM")
    print("=" * 60)

    for question in questions:
        print(f"\nQuestion: {question}")
        print("-" * 40)
        response = generate_response(model, tokenizer, question)
        print(f"Answer: {response}")
        print()


if __name__ == "__main__":
    main()
