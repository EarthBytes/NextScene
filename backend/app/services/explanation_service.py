"""Build recommendation explanations from ranker features and retrieval scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.item import Item
from app.services.catalog_search import CatalogSearcher
from app.services.ranking_service import (
    FEATURE_NAMES,
    RankingModel,
    build_feature_row,
    load_item_features,
)
from app.services.recommendation_service import (
    RecommendationService,
    load_user_history,
    load_user_seen_items,
)
from app.services.user_cache import UserCache


@dataclass(frozen=True)
class ExplanationResult:
    user_id: int
    item_id: int
    title: str | None
    retrieval_score: float | None
    rank_score: float | None
    features: dict[str, float]
    feature_importance: dict[str, float]
    history_length: int
    variant: str


def _load_cached_user_data(
    session: Session,
    user_id: int,
    user_cache: UserCache | None,
    *,
    max_items: int,
    min_rating: float | None,
) -> tuple[list[int], set[int]]:
    if user_cache is not None:
        cached = user_cache.get(user_id)
        if cached is not None:
            return cached.history, cached.seen_items

    history = load_user_history(session, user_id, max_items=max_items, min_rating=min_rating)
    seen_items = load_user_seen_items(session, user_id)
    if user_cache is not None:
        user_cache.set(user_id, history, seen_items)
    return history, seen_items


def _feature_importance(ranker: RankingModel) -> dict[str, float]:
    raw = ranker.booster.feature_importance(importance_type="gain")
    total = float(np.sum(raw)) or 1.0
    return {
        name: float(value) / total
        for name, value in zip(ranker.feature_names, raw, strict=True)
    }


def explain_recommendation(
    session: Session,
    service: RecommendationService,
    *,
    user_id: int,
    item_id: int,
    catalog_searcher: CatalogSearcher,
    user_cache: UserCache | None = None,
) -> ExplanationResult:
    history, seen_items = _load_cached_user_data(
        session,
        user_id,
        user_cache,
        max_items=service.inference.max_seq_len,
        min_rating=service.min_rating,
    )

    if len(history) < service.min_interactions:
        variant = "popularity"
        retrieval_score = None
        rank_score = None
        features: dict[str, float] = {}
        importance: dict[str, float] = {}
    else:
        variant = "generative"
        predicted = service.inference.predict_next_vector(history)
        candidates = catalog_searcher.search(
            predicted,
            top_k=max(service.candidate_pool_size, 50),
            exclude_item_ids=seen_items,
        )
        retrieval_map = dict(candidates)
        retrieval_score = retrieval_map.get(item_id)

        if service.ranker is not None and item_id in retrieval_map:
            item_features = load_item_features(session, list(set(history) | {item_id}))
            feature_row = build_feature_row(
                candidate_id=item_id,
                retrieval_score=retrieval_map[item_id],
                history=history,
                popularity_ranking=service.popularity_ranking,
                item_features=item_features,
                embedding_table=service.embedding_table,
            )
            features = {
                name: float(value)
                for name, value in zip(FEATURE_NAMES, feature_row, strict=True)
            }
            matrix = np.asarray([feature_row], dtype=np.float32)
            rank_score = float(service.ranker.predict_scores(matrix)[0])
            importance = _feature_importance(service.ranker)
        elif service.ranker is not None:
            features = {}
            rank_score = None
            importance = _feature_importance(service.ranker)
        else:
            features = {"retrieval_score": float(retrieval_score or 0.0)}
            rank_score = retrieval_score
            importance = {"retrieval_score": 1.0}

    title_row = session.execute(
        select(Item.title).where(Item.item_id == item_id)
    ).scalar_one_or_none()

    return ExplanationResult(
        user_id=user_id,
        item_id=item_id,
        title=title_row,
        retrieval_score=retrieval_score,
        rank_score=rank_score,
        features=features,
        feature_importance=importance,
        history_length=len(history),
        variant=variant,
    )
