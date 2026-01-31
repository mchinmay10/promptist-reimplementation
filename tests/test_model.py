from models.tokenizer import load_tokenizer
from models.prompt_model import PromptOptimizer


def main():
    tokenizer = load_tokenizer("gpt2")
    model = PromptOptimizer("gpt2", tokenizer)

    model.eval()

    prompt = "a cat sitting on a chair"
    output = model.generate(prompt)

    print("Optimized Prompt:")
    print(output)


if __name__ == "__main__":
    main()
