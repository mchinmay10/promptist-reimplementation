from PIL import Image
from rewards.reward_fn import CombinedReward


def main():
    img = Image.open("tests/sample1.jpg").convert("RGB")

    user_prompt = "a cat sitting on a chair"

    reward_fn = CombinedReward()
    r = reward_fn(
        image=img,
        user_prompt=user_prompt,
        optimized_prompt="high quality photo of a cat sitting on a wooden chair",
    )

    print("Combined reward:", r)


if __name__ == "__main__":
    main()
