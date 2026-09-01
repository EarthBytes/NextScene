"""InfoNCE / sampled-softmax loss for next-item embedding prediction."""

from __future__ import annotations

import torch
import torch.nn.functional as F

DEFAULT_TEMPERATURE = 0.07


def infonce_loss(
    predicted: torch.Tensor,
    positive: torch.Tensor,
    extra_negatives: torch.Tensor | None = None,
    per_sample_negatives: torch.Tensor | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
) -> torch.Tensor:
    """Contrastive loss with in-batch negatives and optional catalog samples.

    Args:
        predicted: L2-normalised predictions, shape [B, D].
        positive: L2-normalised target embeddings, shape [B, D].
        extra_negatives: Optional shared negatives, [K, D].
        per_sample_negatives: Optional per-sample negatives, [B, K, D].
        temperature: Softmax temperature (CLIP default 0.07).
    """
    if predicted.shape != positive.shape or predicted.dim() != 2:
        raise ValueError(
            f"predicted and positive must both be [B, D], got {tuple(predicted.shape)} "
            f"and {tuple(positive.shape)}"
        )
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    logits = predicted @ positive.transpose(0, 1)
    if extra_negatives is not None:
        if extra_negatives.dim() != 2:
            raise ValueError(
                f"extra_negatives must be [K, D], got {tuple(extra_negatives.shape)}"
            )
        logits = torch.cat([logits, predicted @ extra_negatives.transpose(0, 1)], dim=1)
    if per_sample_negatives is not None:
        if per_sample_negatives.dim() != 3:
            raise ValueError(
                f"per_sample_negatives must be [B, K, D], got {tuple(per_sample_negatives.shape)}"
            )
        per_sample_logits = torch.bmm(per_sample_negatives, predicted.unsqueeze(-1)).squeeze(-1)
        logits = torch.cat([logits, per_sample_logits], dim=1)

    logits = logits / temperature
    labels = torch.arange(predicted.size(0), device=predicted.device)
    return F.cross_entropy(logits, labels)
