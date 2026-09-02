"""Orchestrate transformer inference, catalog retrieval, and item metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.config import settings
from app.models.item import Item
from app.services.sequence_cache import cache_paths, load_sequence_cache
from app.services.sequence_dataset import (
    DEFAULT_MIN_RATING,
    MIN_INTERACTIONS,
    ItemEmbeddingTable,
    load_embedding_table,
)
from app.services.sequence_evaluation import (
    CONFIG_FILENAME,
    build_popularity_ranking,
)
from app.services.ranking_service import RankingModel, try_load_ranking_model
from app.services.sequence_inference import SequenceInference


@dataclass(frozen=True)
class Recommendation:
    item_id: int
    title: str | None
    score: float


def search_embedding_catalog(
    embedding_table: ItemEmbeddingTable,
    query_vector: np.ndarray,
    top_k: int,
    exclude_item_ids: set[int] | None = None,
) -> list[tuple[int, float]]:
    """Retrieve top-k items by cosine similarity without Faiss (macOS-safe after torch)."""
    vectors = embedding_table.vectors.detach().cpu().numpy()
    item_ids = embedding_table.item_ids.detach().cpu().numpy().astype(np.int64)
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


class RecommendationService:
    def __init__(
        self,
        inference: SequenceInference,
        embedding_table: ItemEmbeddingTable,
        popularity_ranking: list[int],
        model_version: str,
        *,
        ranker: RankingModel | None = None,
        candidate_pool_size: int = 50,
        min_interactions: int = MIN_INTERACTIONS,
        min_rating: float | None = DEFAULT_MIN_RATING,
    ) -> None:
        self.inference = inference
        self.embedding_table = embedding_table
        self.popularity_ranking = popularity_ranking
        self.model_version = model_version
        self.ranker = ranker
        self.candidate_pool_size = candidate_pool_size
        self.min_interactions = min_interactions
        self.min_rating = min_rating

    def recommend(self, session: Session, user_id: int, k: int) -> list[Recommendation]:
        history = load_user_history(
            session,
            user_id,
            max_items=self.inference.max_seq_len,
            min_rating=self.min_rating,
        )
        if len(history) < self.min_interactions:
            return self._popularity_recommendations(session, user_id, k)

        predicted = self.inference.predict_next_vector(history)
        seen_items = load_user_seen_items(session, user_id)
        pool_size = self.candidate_pool_size if self.ranker is not None else k
        candidates = search_embedding_catalog(
            self.embedding_table,
            predicted,
            top_k=pool_size,
            exclude_item_ids=seen_items,
        )
        if self.ranker is not None:
            candidates = self.ranker.rerank(
                session,
                history,
                candidates,
                self.popularity_ranking,
                self.embedding_table,
                top_k=k,
            )
        else:
            candidates = candidates[:k]
        return self._attach_titles(session, candidates)

    def _popularity_recommendations(
        self,
        session: Session,
        user_id: int,
        k: int,
    ) -> list[Recommendation]:
        seen_items = load_user_seen_items(session, user_id)
        filtered = [
            (item_id, float(len(self.popularity_ranking) - index))
            for index, item_id in enumerate(self.popularity_ranking)
            if item_id not in seen_items
        ][:k]
        return self._attach_titles(session, filtered)

    def _attach_titles(
        self,
        session: Session,
        candidates: list[tuple[int, float]],
    ) -> list[Recommendation]:
        if not candidates:
            return []
        item_ids = [item_id for item_id, _score in candidates]
        rows = session.execute(select(Item.item_id, Item.title).where(Item.item_id.in_(item_ids))).all()
        titles = {int(row.item_id): row.title for row in rows}
        return [
            Recommendation(item_id=item_id, title=titles.get(item_id), score=score)
            for item_id, score in candidates
        ]


def model_version_from_config(model_dir: Path) -> str:
    config_path = model_dir / CONFIG_FILENAME
    if not config_path.is_file():
        return model_dir.name
    config = json.loads(config_path.read_text())
    trained_at = config.get("trained_at", "")
    recall = config.get("best_val_recall_at_10")
    if trained_at and recall is not None:
        return f"{model_dir.name}@{trained_at[:10]}_r10={recall:.4f}"
    return model_dir.name


def load_popularity_ranking(session: Session, embedding_table: ItemEmbeddingTable) -> list[int]:
    cache_dir = Path(settings.sequences_cache_path)
    _, npz_path = cache_paths(cache_dir)
    if npz_path.is_file():
        sequences = load_sequence_cache(cache_dir)
        return build_popularity_ranking(sequences, embedding_table)

    rows = session.execute(
        text(
            """
            SELECT item_id
            FROM interactions
            GROUP BY item_id
            ORDER BY COUNT(*) DESC
            LIMIT 500
            """
        )
    )
    embedded = set(int(item_id) for item_id in embedding_table.item_ids.tolist())
    return [int(row.item_id) for row in rows if int(row.item_id) in embedded]


def load_user_seen_items(session: Session, user_id: int) -> set[int]:
    """All distinct items the user has interacted with (for recommendation filtering)."""
    rows = session.execute(
        text(
            """
            SELECT DISTINCT item_id
            FROM interactions
            WHERE user_id = :user_id
            """
        ),
        {"user_id": user_id},
    )
    return {int(row.item_id) for row in rows}


def load_user_history(
    session: Session,
    user_id: int,
    max_items: int = 50,
    min_rating: float | None = DEFAULT_MIN_RATING,
) -> list[int]:
    rows = session.execute(
        text(
            """
            SELECT item_id, type, context_json
            FROM interactions
            WHERE user_id = :user_id
            ORDER BY ts, interaction_id
            """
        ),
        {"user_id": user_id},
    )
    history: list[int] = []
    for row in rows:
        item_id = int(row.item_id)
        interaction_type = str(row.type)
        if min_rating is not None and interaction_type == "rating":
            context = row.context_json or {}
            if isinstance(context, str):
                context = json.loads(context)
            rating = context.get("rating")
            if rating is None or float(rating) < min_rating:
                continue
        if history and history[-1] == item_id:
            continue
        history.append(item_id)
    return history[-max_items:]


def load_recommendation_service(
    session: Session,
    *,
    model_dir: Path | None = None,
    inference_device: str = "cpu",
) -> RecommendationService:
    """Load model, embeddings, and popularity data for serving."""
    model_dir = model_dir or Path(settings.transformer_model_path)
    embedding_table = load_embedding_table(session)
    inference = SequenceInference.from_model_dir(
        model_dir,
        embedding_table,
        device=inference_device,
    )
    popularity_ranking = load_popularity_ranking(session, embedding_table)
    ranker = None
    candidate_pool_size = settings.ranking_candidate_pool_size
    if settings.enable_ranking:
        ranker = try_load_ranking_model(Path(settings.ranking_model_path))
        if ranker is not None:
            candidate_pool_size = ranker.candidate_pool_size

    model_version = model_version_from_config(model_dir)
    if ranker is not None:
        model_version = f"{model_version}+{ranker.model_version}"

    return RecommendationService(
        inference=inference,
        embedding_table=embedding_table,
        popularity_ranking=popularity_ranking,
        model_version=model_version,
        ranker=ranker,
        candidate_pool_size=candidate_pool_size,
    )


@dataclass
class ServingContext:
    service: RecommendationService | None
    popularity_ranking: list[int]
    model_version: str


def try_load_serving_context(session: Session) -> ServingContext:
    """Load the full serving stack, or popularity fallback if the model is missing."""
    try:
        service = load_recommendation_service(session)
        return ServingContext(
            service=service,
            popularity_ranking=service.popularity_ranking,
            model_version=service.model_version,
        )
    except FileNotFoundError:
        if not settings.enable_fallback_recs:
            raise
        embedding_table = load_embedding_table(session)
        popularity_ranking = load_popularity_ranking(session, embedding_table)
        return ServingContext(
            service=None,
            popularity_ranking=popularity_ranking,
            model_version="popularity-fallback",
        )


def popularity_recommendations(
    session: Session,
    user_id: int,
    k: int,
    popularity_ranking: list[int],
) -> list[Recommendation]:
    seen_items = load_user_seen_items(session, user_id)
    filtered = [
        (item_id, float(len(popularity_ranking) - index))
        for index, item_id in enumerate(popularity_ranking)
        if item_id not in seen_items
    ][:k]
    if not filtered:
        return []
    item_ids = [item_id for item_id, _score in filtered]
    rows = session.execute(select(Item.item_id, Item.title).where(Item.item_id.in_(item_ids))).all()
    titles = {int(row.item_id): row.title for row in rows}
    return [
        Recommendation(item_id=item_id, title=titles.get(item_id), score=score)
        for item_id, score in filtered
    ]
