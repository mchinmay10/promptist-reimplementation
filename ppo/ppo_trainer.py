# Minimal PPO trainer
import torch
from torch.optim import AdamW
from ppo.ppo_utils import compute_logprobs, ppo_clip_loss


class PPOTrainer:
    def __init__(
        self,
        model,
        lr=1e-5,
        clip_eps=0.2,
        device=None,
    ):
        self.model = model
        self.clip_eps = clip_eps
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = AdamW(self.model.parameters(), lr=lr)

    def step(
        self,
        input_ids,
        attention_mask,
        labels,
        rewards,
        old_logprobs,
    ):
        """
        One PPO update step
        """
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits
        new_logprobs = compute_logprobs(logits, labels)

        # advantage = reward - baseline
        advantages = rewards - rewards.mean()
        advantages = advantages.unsqueeze(-1)

        loss = ppo_clip_loss(
            new_logprobs,
            old_logprobs,
            advantages,
            self.clip_eps,
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
