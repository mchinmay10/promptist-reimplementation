import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

MODEL_ID = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = "outputs/sanity"
PROMPT = "a cat sitting on a wooden chair, hight quality photo"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("CUDA available:", torch.cuda.is_available())
    device = torch.device("cuda")

    print("Loading Stable Diffusion...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        safety_checker=None,
    )  # Disabled for research

    pipe = pipe.to(device)
    pipe.enable_attention_slicing  # safer for Colab VRAM

    print("Generating image...")
    with torch.autocast("cuda"):
        result = pipe(
            PROMPT,
            num_inference_steps=30,
            guidance_scale=7.5,
        )

    image: Image.Image = result.images[0]
    out_path = os.path.join(OUTPUT_DIR, "sd_test.png")
    image.save(out_path)


if __name__ == "__main__":
    main()
