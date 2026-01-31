# Scores visual quality
from typing import Optional
import torch
import tempfile
import os
from PIL import Image

try:
    import ImageReward as IR
except ImportError as e:
    raise ImportError(
        "ImageReward is not installed. Install with: pip install image-reward"
    ) from e


class AestheticReward:
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = IR.load("ImageReward-v1.0").to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score(self, image: Image.Image, prompt: str) -> float:
        """
        Args:
          image: PIL.image
          prompt: str (kept for API symmetry; ImageReward may weakly use it)

        Returns:
          float: raw aesthetic score (typically ~[-2, +2])
        """
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            image.save(tmp.name)
            score = self.model.score(prompt, tmp.name)

        return float(score)
