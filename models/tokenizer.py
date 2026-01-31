from transformers import AutoTokenizer

SPECIAL_TOKENS = {
    "bos_token": "<|bos|>",
    "eos_token": "<|eos|>",
    "pad_token": "<|pad|>",
}


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # GPT-2 has no pad token by default
    tokenizer.add_special_tokens(SPECIAL_TOKENS)

    return tokenizer


def format_prompt(user_prompt: str) -> str:
    return f"User: {user_prompt}\nOptimized:"
