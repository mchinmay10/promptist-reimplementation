import torch
from models.tokenizer import load_tokenizer
from models.prompt_model import PromptOptimizer

TEST_PROMPTS = [
    "a cat sitting on a chair",
    "a futuristic city at night",
    "a red sports car",
    "a mountain landscape",
    "a portrait of a woman",
]


def load_model(model_path=None):
    tokenizer = load_tokenizer("gpt2")

    if model_path is None:
        print("Loading Base GPT-2")
        model = PromptOptimizer("gpt2", tokenizer)
    else:
        print(f"Loading SFT Model from {model_path}")
        model = PromptOptimizer(model_path, tokenizer)

    model.eval()
    return model


def evaluate(model, title):
    print(f"\n{title}")

    for prompt in TEST_PROMPTS:
        print("\nUser prompt:")
        print(prompt)

        print("\nOptimized output:")
        print(model.generate(prompt, max_new_tokens=60))

    # Testing
    print(model.generate("a beautiful photograph of"))


def main():
    # Base model
    base_model = load_model()
    evaluate(base_model, "BASE MODEL")

    # SFT model (last checkpoints)
    sft_model = load_model("checkpoints/sft/epoch_3")
    evaluate(sft_model, "SFT_MODEL")


if __name__ == "__main__":
    main()
