# Fake rollout loop

import torch
from models.tokenizer import load_tokenizer
from models.prompt_model import PromptOptimizer
from data.sft_dataset import SFTDataset
from ppo.ppo_trainer import PPOTrainer
from ppo.ppo_utils import compute_logprobs
from rewards.reward_fn import CombinedReward
from utils.dummy_image import make_dummy_image


def main():
    tokenizer = load_tokenizer("gpt2")
    model = PromptOptimizer("gpt2", tokenizer)

    data = [
        {
            "user_prompt": "a cat sitting on a chair",
            "optimized_prompt": "high quality photo of a cat sitting on a wooden chair",
        },
        {
            "user_prompt": "a futuristic city at night",
            "optimized_prompt": "cinematic night cityscape, neon lights, ultra detailed",
        },
    ]

    dataset = SFTDataset(data, tokenizer, max_length=64)
    batch = [dataset[i] for i in range(len(dataset))]

    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])

    reward_fn = CombinedReward()
    dummy_image = make_dummy_image()

    rewards = []
    for b in batch:
        r = reward_fn(
            image=dummy_image,
            user_prompt=b["user_prompt"],
            optimized_prompt=b.get("optimized_prompt", ""),
        )
        rewards.append(r)

    rewards = torch.tensor(rewards, dtype=torch.float32)

    # computing old logprobs
    with torch.no_grad():
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        old_logprobs = compute_logprobs(outputs.logits, labels)

    trainer = PPOTrainer(model)

    for step in range(5):
        loss = trainer.step(
            input_ids,
            attention_mask,
            labels,
            rewards,
            old_logprobs,
        )

        print(f"PPO step {step}: loss = {loss:.4f}")


if __name__ == "__main__":
    main()
