"""Catalog retrieval via numpy cosine search or optional Faiss index."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

from app.config import settings
from app.services.embedding_table import ItemEmbeddingTable
from app.services.faiss_index import load_faiss_index, search_index


@dataclass(frozen=True)
class CatalogSearcher:
    """Search the item catalog by cosine similarity."""

    embedding_table: ItemEmbeddingTable
    mode: str = "numpy"
    faiss_index: object | None = None
    faiss_item_ids: np.ndarray | None = None

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        exclude_item_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        if self.mode == "faiss" and self.faiss_index is not None and self.faiss_item_ids is not None:
            return self._search_faiss(query_vector, top_k, exclude_item_ids)
        return search_embedding_catalog(
            self.embedding_table,
            query_vector,
            top_k=top_k,
            exclude_item_ids=exclude_item_ids,
        )

    def _search_faiss(
        self,
        query_vector: np.ndarray,
        top_k: int,
        exclude_item_ids: set[int] | None,
    ) -> list[tuple[int, float]]:
        assert self.faiss_index is not None
        assert self.faiss_item_ids is not None

        fetch_k = top_k
        if exclude_item_ids:
            fetch_k = min(top_k + len(exclude_item_ids), int(self.faiss_index.ntotal))

        raw = search_index(self.faiss_index, self.faiss_item_ids, query_vector, top_k=fetch_k)
        if not exclude_item_ids:
            return raw[:top_k]

        filtered = [(item_id, score) for item_id, score in raw if item_id not in exclude_item_ids]
        return filtered[:top_k]


def search_embedding_catalog(
    embedding_table: ItemEmbeddingTable,
    query_vector: np.ndarray,
    top_k: int,
    exclude_item_ids: set[int] | None = None,
) -> list[tuple[int, float]]:
    """Retrieve top-k items by cosine similarity (macOS-safe after torch)."""
    vectors = embedding_table.vectors
    item_ids = embedding_table.item_ids
    query = np.ascontiguousarray(query_vector.astype(np.float32).reshape(-1))
    norm = float(np.linalg.norm(query))
    if norm > 0:
        query /= norm
    scores = vectors @ query
    if exclude_item_ids:
        scores = scores.copy()
        scores[np.isin(item_ids, list(exclude_item_ids))] = -np.inf
    available = int(np.sum(np.isfinite(scores)))
    k = min(top_k, available)
    if k <= 0:
        return []
    top_indices = np.argpartition(-scores, k - 1)[:k]
    top_indices = top_indices[np.argsort(-scores[top_indices])]
    return [(int(item_ids[idx]), float(scores[idx])) for idx in top_indices]


def try_load_catalog_searcher(
    embedding_table: ItemEmbeddingTable,
    *,
    index_path: Path | None = None,
    use_faiss: bool | None = None,
) -> CatalogSearcher:
    """Build a catalog searcher, optionally loading a Faiss index."""
    use_faiss = settings.enable_faiss_serving if use_faiss is None else use_faiss
    index_path = index_path or Path(settings.faiss_index_path)

    if use_faiss and index_path.is_file():
        try:
            faiss_index, faiss_item_ids = load_faiss_index(index_path)
            return CatalogSearcher(
                embedding_table=embedding_table,
                mode="faiss",
                faiss_index=faiss_index,
                faiss_item_ids=faiss_item_ids,
            )
        except Exception as exc:
            logger.warning("Faiss index load failed (%s); using numpy search", exc)

    return CatalogSearcher(embedding_table=embedding_table, mode="numpy")
