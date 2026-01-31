# Log-probs and PPO loss
import torch
import torch.nn.functional as F


def compute_logprobs(logits, labels):
    """
    Compute token-level log probs for selected tokens.
    labels: token ids, with -100 for ignored positions
    """

    log_probs = F.log_softmax(logits, dim=-1)
    labels = labels.clone()

    # masking ignored tokens
    mask = labels != -100
    labels[~mask] = 0

    selected = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)
    return selected * mask


def ppo_clip_loss(
    new_logprobs,
    old_logprobs,
    advantages,
    clip_eps=0.2,
):
    """
    PPO clipped objective (token-level, averaged)
    """
    ratio = torch.exp(new_logprobs - old_logprobs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    loss = -torch.min(unclipped, clipped)
    return loss.mean()
