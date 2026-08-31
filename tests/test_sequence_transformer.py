import pytest

pytest.importorskip("torch")

import torch
import torch.nn.functional as F

from app.models.item_embedding import EMBEDDING_DIM
from app.models_ml.sequence_transformer import SequenceTransformer, SequenceTransformerConfig


def _tiny_config(**kwargs) -> SequenceTransformerConfig:
    defaults = dict(
        d_model=32,
        n_heads=4,
        n_layers=2,
        ff_dim=64,
        dropout=0.0,
        max_seq_len=8,
        embedding_dim=16,
    )
    defaults.update(kwargs)
    return SequenceTransformerConfig(**defaults)


def test_config_rejects_invalid_heads():
    with pytest.raises(ValueError, match="divisible"):
        SequenceTransformerConfig(d_model=255, n_heads=4)


def test_forward_shape_and_unit_norm():
    model = SequenceTransformer(_tiny_config())
    model.eval()
    batch_size, seq_len, dim = 4, 5, 16
    embeddings = F.normalize(torch.randn(batch_size, seq_len, dim), dim=-1)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[:, -2:] = False

    predicted = model(embeddings, mask)

    assert predicted.shape == (batch_size, dim)
    norms = predicted.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_forward_matches_production_embedding_dim():
    config = SequenceTransformerConfig(
        d_model=32,
        n_heads=4,
        n_layers=1,
        ff_dim=64,
        dropout=0.0,
        max_seq_len=6,
    )
    model = SequenceTransformer(config)
    embeddings = F.normalize(torch.randn(2, 4, EMBEDDING_DIM), dim=-1)
    mask = torch.ones(2, 4, dtype=torch.bool)
    predicted = model(embeddings, mask)
    assert predicted.shape == (2, EMBEDDING_DIM)
    assert config.embedding_dim == EMBEDDING_DIM


def test_padding_does_not_change_output():
    model = SequenceTransformer(_tiny_config())
    model.eval()
    embeddings = F.normalize(torch.randn(1, 6, 16), dim=-1)
    mask = torch.tensor([[True, True, True, False, False, False]])

    out_a = model(embeddings, mask)
    embeddings_b = embeddings.clone()
    embeddings_b[:, 3:] = F.normalize(torch.randn(1, 3, 16), dim=-1)
    out_b = model(embeddings_b, mask)

    assert torch.allclose(out_a, out_b, atol=1e-5)


def test_causal_mask_hides_future_tokens():
    model = SequenceTransformer(_tiny_config())
    model.eval()
    embeddings = F.normalize(torch.randn(1, 5, 16), dim=-1)
    mask = torch.ones(1, 5, dtype=torch.bool)

    hidden_a = model.encode(embeddings, mask)
    embeddings_b = embeddings.clone()
    embeddings_b[:, -1] = F.normalize(torch.randn(1, 16), dim=-1)
    hidden_b = model.encode(embeddings_b, mask)

    assert torch.allclose(hidden_a[:, :-1], hidden_b[:, :-1], atol=1e-5)
    assert not torch.allclose(hidden_a[:, -1], hidden_b[:, -1], atol=1e-5)


def test_rejects_overlong_sequence():
    model = SequenceTransformer(_tiny_config(max_seq_len=4))
    embeddings = torch.randn(1, 5, 16)
    mask = torch.ones(1, 5, dtype=torch.bool)
    with pytest.raises(ValueError, match="max_seq_len"):
        model(embeddings, mask)
