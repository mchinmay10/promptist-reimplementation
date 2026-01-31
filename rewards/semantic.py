# Enforces semantic alignment with the original user prompt
from typing import Optional
import torch
from transformers import CLIPProcessor, CLIPModel


class SemanticReward:
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        self.model.eval()

    @torch.no_grad()
    def score(self, image, text: str) -> float:
        """
        Args:
          image: PIL.Image
          text: original user prompt

        Returns:
          float: raw CLIP similarity (logits_per_image)
        """
        inputs = self.processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True,
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        return float(outputs.logits_per_image.item())
