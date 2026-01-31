import torch
from torch.utils.data import Dataset
from typing import List, Dict

from models.tokenizer import format_prompt


class SFTDataset(Dataset):
    def __init__(self, data: List[Dict[str, str]], tokenizer, max_length: int = 256):
        """
        data: list of dicts with keys:
          - 'user_prompt'
          - 'optimized_prompt'
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        user_prompt = example["user_prompt"]
        optimized_prompt = example["optimized_prompt"]

        # Construct full training text
        prefix = format_prompt(user_prompt)
        full_text = prefix + " " + optimized_prompt

        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)

        # Find where optimized prompt starts
        prefix_tokens = self.tokenizer(
            prefix,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        prefix_len = int(prefix_tokens["attention_mask"].sum())

        # Gold standard label creation
        labels = torch.full_like(input_ids, -100)

        # Supervise only optimized prompt token
        if prefix_len < self.max_length:
            labels[prefix_len:] = input_ids[prefix_len:]

        # Mask padding token
        labels[attention_mask == 0] = -100

        # Safety check
        if torch.all(labels == -100):
            raise RuntimeError(
                "SFTDataset produced an example with ZERO supervised tokens.\n"
                "Check prefix length, max_length, or optimized_prompt content."
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "user_prompt": user_prompt,
            "optimized_prompt": optimized_prompt,
        }
