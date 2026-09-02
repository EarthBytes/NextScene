"""Orchestrate transformer inference, catalog retrieval, and item metadata."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.config import settings
from app.models.item import Item
from app.models_ml.checkpoints import CONFIG_FILENAME
from app.services.catalog_search import CatalogSearcher, try_load_catalog_searcher
from app.services.sequence_cache import cache_paths, load_sequence_cache
from app.services.sequence_dataset import (
    DEFAULT_MIN_RATING,
    MIN_INTERACTIONS,
    ItemEmbeddingTable,
    build_interaction_history,
    load_embedding_table,
)
from app.services.sequence_evaluation import build_popularity_ranking
from app.services.ranking_service import RankingModel, try_load_ranking_model
from app.services.sequence_inference import SequenceInference
from app.services.user_cache import UserCache

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Recommendation:
    item_id: int
    title: str | None
    score: float


@dataclass
class RecommendationTiming:
    history_ms: float = 0.0
    inference_ms: float = 0.0
    retrieval_ms: float = 0.0
    ranking_ms: float = 0.0
    total_ms: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "history_ms": round(self.history_ms, 2),
            "inference_ms": round(self.inference_ms, 2),
            "retrieval_ms": round(self.retrieval_ms, 2),
            "ranking_ms": round(self.ranking_ms, 2),
            "total_ms": round(self.total_ms, 2),
        }


def filter_popularity_candidates(
    popularity_ranking: list[int],
    seen_items: set[int],
    k: int,
) -> list[tuple[int, float]]:
    return [
        (item_id, float(len(popularity_ranking) - index))
        for index, item_id in enumerate(popularity_ranking)
        if item_id not in seen_items
    ][:k]


def attach_titles(
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


class RecommendationService:
    def __init__(
        self,
        inference: SequenceInference,
        embedding_table: ItemEmbeddingTable,
        popularity_ranking: list[int],
        model_version: str,
        *,
        catalog_searcher: CatalogSearcher,
        ranker: RankingModel | None = None,
        candidate_pool_size: int = 50,
        min_interactions: int = MIN_INTERACTIONS,
        min_rating: float | None = DEFAULT_MIN_RATING,
        user_cache: UserCache | None = None,
    ) -> None:
        self.inference = inference
        self.embedding_table = embedding_table
        self.popularity_ranking = popularity_ranking
        self.model_version = model_version
        self.catalog_searcher = catalog_searcher
        self.ranker = ranker
        self.candidate_pool_size = candidate_pool_size
        self.min_interactions = min_interactions
        self.min_rating = min_rating
        self.user_cache = user_cache

    def recommend(
        self,
        session: Session,
        user_id: int,
        k: int,
    ) -> tuple[list[Recommendation], RecommendationTiming]:
        timing = RecommendationTiming()
        total_start = time.perf_counter()

        history_start = time.perf_counter()
        history, seen_items = self._load_user_data(session, user_id)
        timing.history_ms = (time.perf_counter() - history_start) * 1000

        if len(history) < self.min_interactions:
            results = self._popularity_recommendations(session, user_id, k, seen_items)
            timing.total_ms = (time.perf_counter() - total_start) * 1000
            return results, timing

        inference_start = time.perf_counter()
        predicted = self.inference.predict_next_vector(history)
        timing.inference_ms = (time.perf_counter() - inference_start) * 1000

        pool_size = self.candidate_pool_size if self.ranker is not None else k
        retrieval_start = time.perf_counter()
        candidates = self.catalog_searcher.search(
            predicted,
            top_k=pool_size,
            exclude_item_ids=seen_items,
        )
        timing.retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        if self.ranker is not None:
            ranking_start = time.perf_counter()
            candidates = self.ranker.rerank(
                session,
                history,
                candidates,
                self.popularity_ranking,
                self.embedding_table,
                top_k=k,
            )
            timing.ranking_ms = (time.perf_counter() - ranking_start) * 1000
        else:
            candidates = candidates[:k]

        results = attach_titles(session, candidates)
        timing.total_ms = (time.perf_counter() - total_start) * 1000
        return results, timing

    def _load_user_data(self, session: Session, user_id: int) -> tuple[list[int], set[int]]:
        if self.user_cache is not None:
            cached = self.user_cache.get(user_id)
            if cached is not None:
                return cached.history, cached.seen_items

        history = load_user_history(
            session,
            user_id,
            max_items=self.inference.max_seq_len,
            min_rating=self.min_rating,
        )
        seen_items = load_user_seen_items(session, user_id)
        if self.user_cache is not None:
            self.user_cache.set(user_id, history, seen_items)
        return history, seen_items

    def _popularity_recommendations(
        self,
        session: Session,
        user_id: int,
        k: int,
        seen_items: set[int] | None = None,
    ) -> list[Recommendation]:
        if seen_items is None:
            seen_items = load_user_seen_items(session, user_id)
        filtered = filter_popularity_candidates(self.popularity_ranking, seen_items, k)
        return attach_titles(session, filtered)


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
    interaction_rows = (
        (int(row.item_id), str(row.type), row.context_json)
        for row in rows
    )
    return build_interaction_history(
        interaction_rows,
        max_items=max_items,
        min_rating=min_rating,
    )


def load_recommendation_service(
    session: Session,
    *,
    model_dir: Path | None = None,
    inference_device: str = "cpu",
    user_cache: UserCache | None = None,
    embedding_table: ItemEmbeddingTable | None = None,
    catalog_searcher: CatalogSearcher | None = None,
    popularity_ranking: list[int] | None = None,
) -> RecommendationService:
    """Load model, embeddings, and popularity data for serving."""
    model_dir = model_dir or Path(settings.transformer_model_path)
    embedding_table = embedding_table or load_embedding_table(session)
    catalog_searcher = catalog_searcher or try_load_catalog_searcher(embedding_table)
    popularity_ranking = popularity_ranking or load_popularity_ranking(session, embedding_table)
    inference = SequenceInference.from_model_dir(
        model_dir,
        embedding_table,
        device=inference_device,
    )
    ranker = None
    candidate_pool_size = settings.ranking_candidate_pool_size
    if settings.enable_ranking:
        ranker = try_load_ranking_model(Path(settings.ranking_model_path))
        if ranker is not None:
            candidate_pool_size = ranker.candidate_pool_size

    model_version = model_version_from_config(model_dir)
    if ranker is not None:
        model_version = f"{model_version}+{ranker.model_version}"
    model_version = f"{model_version}+{catalog_searcher.mode}"

    cache = user_cache or UserCache(
        max_size=settings.user_cache_max_size,
        ttl_seconds=settings.user_cache_ttl_seconds,
    )

    return RecommendationService(
        inference=inference,
        embedding_table=embedding_table,
        popularity_ranking=popularity_ranking,
        model_version=model_version,
        catalog_searcher=catalog_searcher,
        ranker=ranker,
        candidate_pool_size=candidate_pool_size,
        user_cache=cache,
    )


@dataclass
class ServingContext:
    service: RecommendationService | None
    popularity_ranking: list[int]
    model_version: str
    catalog_searcher: CatalogSearcher | None = None
    user_cache: UserCache = field(default_factory=UserCache)
    retrieval_mode: str = "numpy"


def try_load_serving_context(session: Session) -> ServingContext:
    """Load the full serving stack, or popularity fallback if the model is missing."""
    user_cache = UserCache(
        max_size=settings.user_cache_max_size,
        ttl_seconds=settings.user_cache_ttl_seconds,
    )
    embedding_table = load_embedding_table(session)
    catalog_searcher = try_load_catalog_searcher(embedding_table)
    popularity_ranking = load_popularity_ranking(session, embedding_table)

    try:
        service = load_recommendation_service(
            session,
            user_cache=user_cache,
            embedding_table=embedding_table,
            catalog_searcher=catalog_searcher,
            popularity_ranking=popularity_ranking,
        )
        return ServingContext(
            service=service,
            popularity_ranking=popularity_ranking,
            model_version=service.model_version,
            catalog_searcher=catalog_searcher,
            user_cache=user_cache,
            retrieval_mode=catalog_searcher.mode,
        )
    except FileNotFoundError:
        if not settings.enable_fallback_recs:
            raise
        return ServingContext(
            service=None,
            popularity_ranking=popularity_ranking,
            model_version="popularity-fallback",
            catalog_searcher=catalog_searcher,
            user_cache=user_cache,
            retrieval_mode=catalog_searcher.mode,
        )


def warmup_serving_context(session: Session, serving: ServingContext) -> None:
    """Run dummy inference to warm model weights and caches."""
    if not settings.warmup_on_startup or serving.service is None:
        return

    try:
        history = load_user_history(
            session,
            user_id=1,
            max_items=serving.service.inference.max_seq_len,
            min_rating=serving.service.min_rating,
        )
        if len(history) >= serving.service.min_interactions:
            query = serving.service.inference.predict_next_vector(history)
            if serving.catalog_searcher is not None:
                serving.catalog_searcher.search(query, top_k=5)
        logger.info("Serving warmup completed (retrieval_mode=%s)", serving.retrieval_mode)
    except Exception as exc:
        logger.warning("Serving warmup skipped: %s", exc)


def popularity_recommendations(
    session: Session,
    user_id: int,
    k: int,
    popularity_ranking: list[int],
    user_cache: UserCache | None = None,
) -> list[Recommendation]:
    if user_cache is not None:
        cached = user_cache.get(user_id)
        if cached is not None:
            seen_items = cached.seen_items
        else:
            seen_items = load_user_seen_items(session, user_id)
            history = load_user_history(session, user_id)
            user_cache.set(user_id, history, seen_items)
    else:
        seen_items = load_user_seen_items(session, user_id)

    filtered = filter_popularity_candidates(popularity_ranking, seen_items, k)
    return attach_titles(session, filtered)
