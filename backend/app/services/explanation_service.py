"""Build recommendation explanations from ranker features and retrieval scores."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from app.models.item import Item
from app.services.catalog_search import CatalogSearcher
from app.services.item_service import extract_year
from app.services.library_service import load_user_library
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
from sqlalchemy import select
from sqlalchemy.orm import Session


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


@dataclass(frozen=True)
class NaturalExplanation:
    item_id: int
    title: str | None
    explanation: str
    related_titles: list[str]
    shared_genres: list[str]
    reasons: list[str]


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


def _format_title_list(titles: list[str]) -> str:
    if not titles:
        return ""
    if len(titles) == 1:
        return titles[0]
    if len(titles) == 2:
        return f"{titles[0]} and {titles[1]}"
    return f"{', '.join(titles[:-1])}, and {titles[-1]}"


def _genre_overlap_score(item_genres: set[str], candidate_genres: set[str]) -> int:
    return len(item_genres & candidate_genres)


def _format_genre_list(genres: list[str]) -> str:
    if not genres:
        return ""
    if len(genres) == 1:
        return genres[0]
    if len(genres) == 2:
        return f"{genres[0]} and {genres[1]}"
    return f"{', '.join(genres[:-1])}, and {genres[-1]}"


def _truncate(text: str, max_len: int = 100) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rsplit(" ", 1)[0] + "..."


def explain_recommendation_natural(
    session: Session,
    service: RecommendationService,
    *,
    user_id: int,
    item_id: int,
    catalog_searcher: CatalogSearcher,
    user_cache: UserCache | None = None,
) -> NaturalExplanation:
    del service, catalog_searcher, user_cache

    library = load_user_library(session, user_id)
    target = session.get(Item, item_id)
    target_title = target.title if target is not None else "this movie"
    target_genres = list(target.genres or []) if target is not None else []
    target_genre_set = set(target_genres)
    target_year = extract_year(target.metadata_json if target else None)

    if not library:
        return NaturalExplanation(
            item_id=item_id,
            title=target_title,
            explanation=(
                f"Add more films to your library so we can explain why "
                f"{target_title} might suit your taste."
            ),
            related_titles=[],
            shared_genres=[],
            reasons=[],
        )

    genre_counts: Counter[str] = Counter()
    library_years: list[int] = []
    for entry in library:
        for genre in entry.get("genres") or []:
            genre_counts[genre] += 1
        if entry.get("year"):
            library_years.append(int(entry["year"]))

    shared_genres = sorted(
        target_genre_set & set(genre_counts.keys()),
        key=lambda genre: (-genre_counts[genre], genre),
    )

    scored_library: list[tuple[dict, int, int]] = []
    for entry in library:
        entry_genres = set(entry.get("genres") or [])
        overlap = len(entry_genres & target_genre_set)
        entry_year = entry.get("year") or target_year or 2000
        year_gap = abs((target_year or entry_year) - entry_year) if target_year else 50
        scored_library.append((entry, overlap, year_gap))

    scored_library.sort(key=lambda row: (-row[1], row[2]))
    top_related = [entry for entry, _overlap, _gap in scored_library[:3]]
    related_titles = [entry["title"] for entry in top_related]

    reasons: list[str] = []
    library_genre_set = set(genre_counts.keys())

    if shared_genres:
        lead_genre = shared_genres[0]
        reasons.append(f"Matches your taste for {lead_genre}.")

    if top_related:
        anchor = top_related[0]["title"]
        anchor_genres = set(top_related[0].get("genres") or [])
        overlap_with_anchor = (anchor_genres & target_genre_set) & library_genre_set
        if overlap_with_anchor:
            reasons.append(
                f"Similar vibe to {anchor} ({_format_genre_list(sorted(overlap_with_anchor)[:2])})."
            )
        elif shared_genres:
            reasons.append(f"Pairs well with {anchor} from your library.")
        else:
            reasons.append(f"Fits alongside {anchor} in your library.")

    if not reasons:
        reasons.append("Fits the mix of films you've added so far.")

    reasons = [_truncate(reason, 90) for reason in reasons[:2]]

    if shared_genres and top_related:
        explanation = (
            f"{target_title} fits your {shared_genres[0]} picks — especially if you liked "
            f"{top_related[0]['title']}."
        )
    elif shared_genres:
        explanation = f"{target_title} lines up with your interest in {shared_genres[0]}."
    else:
        explanation = reasons[0]

    explanation = _truncate(explanation, 140)

    return NaturalExplanation(
        item_id=item_id,
        title=target_title,
        explanation=explanation,
        related_titles=related_titles,
        shared_genres=shared_genres,
        reasons=reasons,
    )
