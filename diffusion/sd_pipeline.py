import torch
from diffusers import StableDiffusionPipeline


class StableDiffusionPipelineWrapper:
    def __init__(
        self,
        model_id="runwayml/stable-diffusion-v1-5",
        device=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )

        self.pipe.to(self.device)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        num_steps: int = 30,
        guidance_scale: float = 7.5,
    ):
        image = self.pipe(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
        ).images[0]

        return image
