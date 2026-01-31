from models.tokenizer import load_tokenizer
from data.sft_dataset import SFTDataset


def main():
    tokenizer = load_tokenizer("gpt2")

    dummy_data = [
        {
            "user_prompt": "a cat sitting on a chair",
            "optimized_prompt": "high quality photo of a cat sitting on a wooden chair",
        }
    ]

    dataset = SFTDataset(dummy_data, tokenizer)

    sample = dataset[0]

    print("Input IDs:")
    print(sample["input_ids"][:20])

    print("Labels:")
    print(sample["labels"][:20])

    print("\nDecoded optimized part:")
    label_tokens = sample["labels"]
    label_tokens = label_tokens[label_tokens != -100]
    label_tokens = label_tokens[label_tokens != tokenizer.pad_token_id]
    print(tokenizer.decode(label_tokens))


if __name__ == "__main__":
    main()
