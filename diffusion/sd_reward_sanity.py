import torch
import os
from PIL import Image
from diffusers import StableDiffusionPipeline
import clip
import ImageReward as IR

MODEL_ID = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = "outputs/sanity"
PROMPT = "a cat sitting on a wooden chair, high quality photo"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda")
    print("CUDA available:", torch.cuda.is_available())

    # Load Stable Diffusion
    print("Loading Stable Diffusion...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)
    pipe.enable_attention_slicing()

    # Generate Image
    print("Generating image...")
    with torch.autocast("cuda"):
        image = pipe(
            PROMPT,
            num_inference_steps=30,
            guidance_scale=7.5,
        ).images[0]

    image_path = os.path.join(OUTPUT_DIR, "sd_reward_test.png")
    image.save(image_path)
    print("Image saved:", image_path)

    # Load clip
    print("Loading Clip...")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    image_input = preprocess(image).unsqueeze(0).to(device)
    text_features = clip.tokenize([PROMPT]).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_features)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        clip_score = (image_features @ text_features.T).item()

    print(f"CLIP similarity score: {clip_score:.4f}")

    # Load ImageReward
    print("Loading Imagereward...")
    ir_model = IR.load("ImageReward-v1.0", device=device)

    with torch.no_grad():
        ir_score = ir_model.score(PROMPT, image_path)

    print(f"ImageReward score: {ir_score:.4f}")

    print("Reward sanity check complete.")


if __name__ == "__main__":
    main()
