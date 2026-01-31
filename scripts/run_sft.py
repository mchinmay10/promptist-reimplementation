from models.tokenizer import load_tokenizer
from models.prompt_model import PromptOptimizer
from data.sft_dataset import SFTDataset
from training.sft_trainer import SFTTrainer


def main():
    tokenizer = load_tokenizer("gpt2")
    model = PromptOptimizer("gpt2", tokenizer)

    # Dummy Data for now (later to be replaced with DiffusionDB)
    train_data = [
        {
            "user_prompt": "a cat sitting on a chair",
            "optimized_prompt": "high quality photo of a cat sitting on a wooden chair, soft lighting",
        },
        {
            "user_prompt": "a mountain landscape",
            "optimized_prompt": "ultra detailed landscape photograph of mountains at sunrise, cinematic lighting",
        },
        {
            "user_prompt": "a red sports car",
            "optimized_prompt": (
                "high quality photo of a red sports car, ultra realistic, "
                "cinematic lighting, sharp focus, 85mm lens, glossy paint, "
                "studio background"
            ),
        },
        {
            "user_prompt": "a forest in autumn",
            "optimized_prompt": (
                "ultra detailed landscape photograph of an autumn forest, "
                "golden and red leaves, soft morning light, misty atmosphere, "
                "cinematic composition, high resolution"
            ),
        },
    ]

    dataset = SFTDataset(
        train_data,
        tokenizer,
        max_length=128,
    )

    trainer = SFTTrainer(
        model=model, dataset=dataset, batch_size=2, num_epochs=3, lr=1e-5
    )

    trainer.train()


if __name__ == "__main__":
    main()
