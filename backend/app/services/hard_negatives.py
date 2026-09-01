"""Catalog hard-negative sampling for contrastive training."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from app.services.sequence_dataset import ItemEmbeddingTable


@dataclass
class SampledNegatives:
    shared: torch.Tensor | None = None
    per_sample: torch.Tensor | None = None


class NegativeSampler:
    """Mix random catalog negatives with in-catalog hard negatives (top-k by cosine)."""

    def __init__(
        self,
        embedding_table: ItemEmbeddingTable,
        *,
        random_negatives: int = 32,
        hard_negatives: int = 32,
        hard_search_k: int = 128,
    ) -> None:
        self.embedding_table = embedding_table
        self.random_negatives = random_negatives
        self.hard_negatives = hard_negatives
        self.hard_search_k = hard_search_k

    def sample(
        self,
        query_vectors: torch.Tensor | None = None,
        exclude_item_ids: torch.Tensor | None = None,
    ) -> SampledNegatives:
        shared = (
            self.embedding_table.sample_negatives(self.random_negatives)
            if self.random_negatives > 0
            else None
        )
        per_sample = None
        if self.hard_negatives > 0 and query_vectors is not None:
            per_sample = self._sample_hard(query_vectors, exclude_item_ids)
        return SampledNegatives(shared=shared, per_sample=per_sample)

    def _sample_hard(
        self,
        query_vectors: torch.Tensor,
        exclude_item_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        catalog = self.embedding_table.vectors
        scores = query_vectors @ catalog.transpose(0, 1)

        if exclude_item_ids is not None:
            exclude_index = self.embedding_table.indices_for(exclude_item_ids)
            valid = exclude_index >= 0
            if valid.any():
                batch_indices = torch.arange(scores.size(0), device=scores.device)[valid]
                scores[batch_indices, exclude_index[valid]] = -torch.inf

        top_k = min(max(self.hard_negatives, 1), scores.size(1))
        top_indices = scores.topk(top_k, dim=1).indices[:, : self.hard_negatives]
        return catalog[top_indices]
