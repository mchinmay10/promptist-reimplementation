# Normalize, weight, and combine rewards
from typing import Optional
from rewards.aesthetic import AestheticReward
from rewards.semantic import SemanticReward


def normalize(x: float, min_val: float, max_val: float) -> float:
    if x < min_val:
        x = min_val
    if x > max_val:
        x = max_val
    return (x - min_val) / (max_val - min_val)


class CombinedReward:
    def __init__(
        self,
        w_aesthetic: float = 0.5,
        w_semantic: float = 0.5,
        device: Optional[str] = None,
    ):
        assert (
            abs(w_aesthetic + w_semantic - 1.0) < 1e-6
        ), "Reward weights must sum to 1."

        self.w_a = w_aesthetic
        self.w_s = w_semantic

        self.aesthetic = AestheticReward(device=device)
        self.semantic = SemanticReward(device=device)

        self.aesthetic_min = -2.0
        self.aesthetic_max = 2.0

        self.semantic_min = 0.0
        self.semantic_max = 40.0

    def __call__(
        self,
        image,
        user_prompt: str,
        optimized_prompt: Optional[str] = None,
    ):
        """
        Args:
          image: PIL.Image
          user_prompt: original user intent
          optimized_prompt: kept for future extensions (unused in v1)

        Returns:
          float: combined normalized reward in [0, 1]
        """
        r_a_raw = self.aesthetic.score(image, user_prompt)
        r_s_raw = self.semantic.score(image, user_prompt)

        r_a = normalize(r_a_raw, self.aesthetic_min, self.aesthetic_max)
        r_s = normalize(r_s_raw, self.semantic_min, self.semantic_max)

        return self.w_a * r_a + self.w_s * r_s
