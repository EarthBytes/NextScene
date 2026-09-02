"""Numpy embedding table and sequence helpers for API serving (no PyTorch)."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from app.models.item_embedding import EMBEDDING_DIM
from app.services.faiss_index import load_embeddings_from_db
from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

PAD_ITEM_ID = 0
MIN_INTERACTIONS = 3
DEFAULT_MIN_RATING = 3.5
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1


@dataclass(frozen=True)
class SequenceSample:
    user_id: int
    input_item_ids: tuple[int, ...]
    target_item_id: int


class ItemEmbeddingTable:
    """In-memory item_id → CLIP vector lookup (numpy, no torch required)."""

    def __init__(self, item_ids: np.ndarray, vectors: np.ndarray) -> None:
        if item_ids.shape[0] != vectors.shape[0]:
            raise ValueError("item_ids and vectors must have the same length")
        if vectors.ndim != 2:
            raise ValueError(f"Expected vectors [N, D], got {vectors.shape}")

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        vectors = (vectors / norms).astype(np.float32, copy=False)

        self.item_ids = np.ascontiguousarray(item_ids, dtype=np.int64)
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        max_id = int(item_ids.max()) if len(item_ids) else 0
        index_map = np.full(max_id + 1, -1, dtype=np.int64)
        index_map[self.item_ids] = np.arange(len(item_ids), dtype=np.int64)
        self.index_map = index_map

    def __len__(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def embedding_dim(self) -> int:
        return int(self.vectors.shape[1])

    def indices_for(self, item_ids: np.ndarray | Sequence[int]) -> np.ndarray:
        flat = np.asarray(item_ids, dtype=np.int64).reshape(-1)
        in_range = (flat >= 0) & (flat < self.index_map.size)
        index = np.full_like(flat, -1)
        if in_range.any():
            index[in_range] = self.index_map[flat[in_range]]
        return index.reshape(np.asarray(item_ids).shape)


def parse_context(context) -> dict:
    if context is None:
        return {}
    if isinstance(context, str):
        return json.loads(context)
    return dict(context)


def build_interaction_history(
    rows: Iterable[tuple[int, str, dict | str | None]],
    *,
    max_items: int = 50,
    min_rating: float | None = DEFAULT_MIN_RATING,
) -> list[int]:
    history: list[int] = []
    for item_id, interaction_type, context in rows:
        if min_rating is not None and interaction_type == "rating":
            parsed = parse_context(context)
            rating = parsed.get("rating")
            if rating is None or float(rating) < min_rating:
                continue
        if history and history[-1] == item_id:
            continue
        history.append(item_id)
    return history[-max_items:]


def build_user_sequences(
    interactions: Iterable[tuple[int, int, str, dict | None]],
    embedded_item_ids: set[int],
    min_rating: float | None = DEFAULT_MIN_RATING,
    min_interactions: int = MIN_INTERACTIONS,
) -> dict[int, list[int]]:
    sequences: dict[int, list[int]] = {}
    for user_id, item_id, interaction_type, context in interactions:
        if item_id not in embedded_item_ids:
            continue
        if min_rating is not None and interaction_type == "rating":
            parsed = parse_context(context)
            rating = parsed.get("rating")
            if rating is None or float(rating) < min_rating:
                continue
        seq = sequences.setdefault(user_id, [])
        if seq and seq[-1] == item_id:
            continue
        seq.append(item_id)

    return {
        user_id: seq
        for user_id, seq in sequences.items()
        if len(seq) >= min_interactions
    }


def load_embedding_table(session: Session) -> ItemEmbeddingTable:
    item_ids, vectors = load_embeddings_from_db(session)
    if vectors.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"Expected {EMBEDDING_DIM}-dim embeddings, got {vectors.shape[1]}")
    return ItemEmbeddingTable(item_ids, vectors)


def load_embedded_item_ids(session: Session) -> set[int]:
    rows = session.execute(
        text(
            """
            SELECT item_id
            FROM item_embeddings
            WHERE vector IS NOT NULL
            ORDER BY item_id
            """
        )
    )
    return {int(row.item_id) for row in rows}


def iter_interaction_rows(
    session: Session,
    user_ids: Sequence[int] | None = None,
) -> Iterable[tuple[int, int, str, dict]]:
    if user_ids is None:
        query = text(
            """
            SELECT user_id, item_id, type, context_json
            FROM interactions
            ORDER BY user_id, ts, interaction_id
            """
        )
        rows = session.execute(query)
    else:
        if not user_ids:
            return
        query = (
            text(
                """
                SELECT user_id, item_id, type, context_json
                FROM interactions
                WHERE user_id IN :user_ids
                ORDER BY user_id, ts, interaction_id
                """
            ).bindparams(bindparam("user_ids", expanding=True))
        )
        rows = session.execute(query, {"user_ids": list(user_ids)})
    for row in rows:
        yield int(row.user_id), int(row.item_id), str(row.type), parse_context(row.context_json)


def load_user_sequences(
    session: Session,
    embedded_item_ids: set[int],
    min_rating: float | None = DEFAULT_MIN_RATING,
    min_interactions: int = MIN_INTERACTIONS,
    user_ids: Sequence[int] | None = None,
) -> dict[int, list[int]]:
    return build_user_sequences(
        iter_interaction_rows(session, user_ids=user_ids),
        embedded_item_ids=embedded_item_ids,
        min_rating=min_rating,
        min_interactions=min_interactions,
    )
