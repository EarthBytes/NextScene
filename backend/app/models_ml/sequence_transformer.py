"""Causal transformer encoder that predicts the next-item CLIP embedding."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import torch
import torch.nn.functional as F
from app.models.item_embedding import EMBEDDING_DIM
from torch import nn


@dataclass
class SequenceTransformerConfig:
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    ff_dim: int = 512
    dropout: float = 0.1
    max_seq_len: int = 50
    embedding_dim: int = EMBEDDING_DIM
    clip_model: str = "openai/clip-vit-base-patch32"

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.max_seq_len < 1:
            raise ValueError("max_seq_len must be >= 1")
        if self.embedding_dim < 1:
            raise ValueError("embedding_dim must be >= 1")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> SequenceTransformerConfig:
        known = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})


class SequenceTransformer(nn.Module):
    """SASRec-style causal encoder over frozen item embeddings.

    Input:  [B, T, embedding_dim] CLIP vectors with a boolean validity mask.
    Output: [B, embedding_dim] L2-normalised predicted next-item vector.
    """

    def __init__(
        self,
        config: SequenceTransformerConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if config is None:
            config = SequenceTransformerConfig(**kwargs)
        elif kwargs:
            raise TypeError("Pass either config or keyword arguments, not both")

        self.config = config
        self.input_proj = nn.Linear(config.embedding_dim, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)
        self.input_dropout = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
            enable_nested_tensor=False,
        )
        self.out_norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.embedding_dim)

    def encode(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if embeddings.dim() != 3:
            raise ValueError(f"Expected embeddings [B, T, D], got {tuple(embeddings.shape)}")
        if mask.shape != embeddings.shape[:2]:
            raise ValueError(
                f"Mask shape {tuple(mask.shape)} does not match embeddings {tuple(embeddings.shape)}"
            )

        _batch_size, seq_len, _dim = embeddings.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.config.max_seq_len}"
            )

        positions = torch.arange(seq_len, device=embeddings.device)
        hidden = self.input_proj(embeddings) + self.pos_embed(positions)
        hidden = self.input_dropout(hidden)

        attn_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=embeddings.device),
            diagonal=1,
        )
        key_padding_mask = ~mask.bool()
        return self.encoder(hidden, mask=attn_mask, src_key_padding_mask=key_padding_mask)

    def pool_last(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.bool().sum(dim=1).clamp(min=1)
        last_index = lengths - 1
        batch_index = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch_index, last_index]

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(embeddings, mask)
        last_hidden = self.pool_last(hidden, mask)
        predicted = self.head(self.out_norm(last_hidden))
        return F.normalize(predicted, p=2, dim=-1)
