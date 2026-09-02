import pytest

pytest.importorskip("torch")

import torch
import torch.nn.functional as F
from app.models_ml.contrastive_loss import infonce_loss


def test_aligned_vectors_beat_random():
    batch, dim = 8, 16
    positive = F.normalize(torch.randn(batch, dim), dim=-1)
    loss_aligned = infonce_loss(positive, positive)
    random_pred = F.normalize(torch.randn(batch, dim), dim=-1)
    loss_random = infonce_loss(random_pred, positive)
    assert loss_aligned < loss_random
    assert torch.isfinite(loss_aligned)
    assert torch.isfinite(loss_random)


def test_extra_negatives_change_logits_shape_via_loss():
    batch, dim, extra_k = 4, 8, 7
    predicted = F.normalize(torch.randn(batch, dim), dim=-1)
    positive = F.normalize(torch.randn(batch, dim), dim=-1)
    extra = F.normalize(torch.randn(extra_k, dim), dim=-1)
    loss = infonce_loss(predicted, positive, extra_negatives=extra)
    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_per_sample_negatives():
    batch, dim, extra_k = 3, 8, 5
    predicted = F.normalize(torch.randn(batch, dim), dim=-1)
    positive = predicted.clone()
    extra = F.normalize(torch.randn(batch, extra_k, dim), dim=-1)
    loss = infonce_loss(predicted, positive, per_sample_negatives=extra, temperature=0.07)
    assert torch.isfinite(loss)


def test_temperature_scales_loss():
    predicted = F.normalize(torch.randn(6, 8), dim=-1)
    positive = F.normalize(torch.randn(6, 8), dim=-1)
    sharp = infonce_loss(predicted, positive, temperature=0.05)
    soft = infonce_loss(predicted, positive, temperature=1.0)
    assert torch.isfinite(sharp) and torch.isfinite(soft)
