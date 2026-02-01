import torch

from diffusion.sd_pipeline import StableDiffusionPipelineWrapper
from rewards.reward_fn import CombinedReward
from models.prompt_model import PromptOptimizer
from models.tokenizer import load_tokenizer

USER_PROMPT = "a cat sitting on a chair"
MODEL_NAME = "gpt2"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = load_tokenizer(MODEL_NAME)

    # Load prompt policy
    policy = PromptOptimizer(
        model_name=MODEL_NAME,
        tokenizer=tokenizer,
    ).to(device)
    policy.train()

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-5)

    # Stable Diffusion
    sd = StableDiffusionPipelineWrapper(device=device)

    # Reward function
    reward_fn = CombinedReward(device=device)

    # PPO micro-step
    optimizer.zero_grad()

    optimized_prompt, logprob = policy.sample(USER_PROMPT)

    print("Optimized prompt:")
    print(optimized_prompt)

    image = sd.generate(
        optimized_prompt,
        num_steps=30,
        guidance_scale=7.5,
    )

    reward = reward_fn(
        image=image,
        user_prompt=USER_PROMPT,
        optimized_prompt=optimized_prompt,
    )

    print("Reward:", reward)

    # PPO loss (minimal)
    advantage = torch.tensor(reward, device=device)

    loss = -logprob * advantage
    print("PPO loss:", loss.item())

    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()

    print("Grad norm:", grad_norm.item())
    print("PPO micro-step complete.")


if __name__ == "__main__":
    main()
