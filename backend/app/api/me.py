"""Authenticated user endpoints: library, recommendations, explanations."""

from __future__ import annotations

from app.api.deps import get_current_user
from app.api.recommendations import INFERENCE_ERRORS
from app.db.session import get_db
from app.models.item import Item
from app.models.user import User
from app.services.explanation_service import explain_recommendation_natural
from app.services.interaction_service import log_interaction
from app.services.item_service import load_items_by_ids
from app.services.library_service import (
    DISMISSED_SOURCE,
    LIBRARY_SOURCE,
    WATCHLIST_SOURCE,
    get_movie_status,
    load_user_library,
    load_user_watchlist,
    remove_app_interactions,
)
from app.services.preference_service import set_preferred_genres, sync_preferred_genres
from app.services.recommendation_service import (
    attach_titles,
    filter_popularity_candidates,
    genre_weighted_popularity_candidates,
    library_genre_profile,
)
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

router = APIRouter()


class LibraryItem(BaseModel):
    item_id: int
    title: str
    genres: list[str] = Field(default_factory=list)
    year: int | None = None
    poster_url: str | None = None
    image_url: str | None = None
    added_at: str


class AddMovieRequest(BaseModel):
    item_id: int


class RecommendationItemClean(BaseModel):
    item_id: int
    title: str | None = None
    rank: int
    genres: list[str] = Field(default_factory=list)
    year: int | None = None
    poster_url: str | None = None
    image_url: str | None = None


class RecommendationsCleanResponse(BaseModel):
    recommendations: list[RecommendationItemClean]
    library_count: int
    needs_more_movies: bool


class ExplanationCleanResponse(BaseModel):
    item_id: int
    title: str | None = None
    explanation: str
    related_titles: list[str] = Field(default_factory=list)
    shared_genres: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class MovieStatusResponse(BaseModel):
    item_id: int
    in_library: bool
    in_watchlist: bool
    dismissed: bool
    rating: float | None = None


class RatingRequest(BaseModel):
    item_id: int
    rating: float = Field(..., ge=1, le=5)


class PreferencesResponse(BaseModel):
    preferred_genres: list[str] = Field(default_factory=list)


class PreferencesUpdateRequest(BaseModel):
    preferred_genres: list[str] = Field(default_factory=list)


@router.get("/me/movies", response_model=list[LibraryItem])
def get_my_movies(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return load_user_library(db, current_user.id)


@router.post("/me/movies", response_model=LibraryItem, status_code=201)
def add_movie(
    body: AddMovieRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return _log_app_interaction(
        request,
        db,
        current_user,
        body.item_id,
        interaction_type="view",
        source="library",
    )


@router.get("/me/recommendations", response_model=RecommendationsCleanResponse)
def get_my_recommendations(
    request: Request,
    k: int = 12,
    genres: list[str] | None = Query(None, description="Filter recommendations to these genres"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if k < 1 or k > 50:
        raise HTTPException(status_code=400, detail="k must be between 1 and 50")

    library = load_user_library(db, current_user.id)
    library_count = len(library)

    if library_count == 0:
        return RecommendationsCleanResponse(
            recommendations=[],
            library_count=0,
            needs_more_movies=True,
        )

    active_genres = genres if genres else sync_preferred_genres(db, current_user)
    raw_results = _library_recommendations(
        request,
        db,
        user_id=current_user.id,
        k=k,
        genres=active_genres or None,
        library=library,
    )
    recommendations = [
        RecommendationItemClean(
            item_id=rec["item_id"],
            title=rec["title"],
            rank=index + 1,
            genres=rec.get("genres", []),
            year=rec.get("year"),
            poster_url=rec.get("poster_url"),
            image_url=rec.get("image_url"),
        )
        for index, rec in enumerate(raw_results)
    ]

    return RecommendationsCleanResponse(
        recommendations=recommendations,
        library_count=library_count,
        needs_more_movies=library_count < 3,
    )


@router.get("/me/preferences", response_model=PreferencesResponse)
def get_preferences(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return PreferencesResponse(preferred_genres=sync_preferred_genres(db, current_user))


@router.put("/me/preferences", response_model=PreferencesResponse)
def update_preferences(
    body: PreferencesUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    preferred_genres = set_preferred_genres(db, current_user, body.preferred_genres)
    return PreferencesResponse(preferred_genres=preferred_genres)


@router.get("/me/explanations/{item_id}", response_model=ExplanationCleanResponse)
def get_my_explanation(
    item_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    serving = getattr(request.app.state, "serving", None)
    if serving is None or serving.service is None:
        error = getattr(request.app.state, "serving_error", "Recommendation service unavailable")
        raise HTTPException(status_code=503, detail=error)

    if serving.catalog_searcher is None:
        raise HTTPException(status_code=503, detail="Catalog searcher not available")

    try:
        result = explain_recommendation_natural(
            db,
            serving.service,
            user_id=current_user.id,
            item_id=item_id,
            catalog_searcher=serving.catalog_searcher,
            user_cache=serving.user_cache,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {exc}") from exc

    return ExplanationCleanResponse(
        item_id=result.item_id,
        title=result.title,
        explanation=result.explanation,
        related_titles=result.related_titles,
        shared_genres=result.shared_genres,
        reasons=result.reasons,
    )


@router.get("/me/movies/{item_id}/status", response_model=MovieStatusResponse)
def get_movie_status_endpoint(
    item_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if db.get(Item, item_id) is None:
        raise HTTPException(status_code=404, detail="Movie not found")
    return get_movie_status(db, current_user.id, item_id)


@router.get("/me/watchlist", response_model=list[LibraryItem])
def get_watchlist(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return load_user_watchlist(db, current_user.id)


@router.post("/me/watchlist", response_model=LibraryItem, status_code=201)
def add_to_watchlist(
    body: AddMovieRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return _log_app_interaction(
        request,
        db,
        current_user,
        body.item_id,
        interaction_type="click",
        source=WATCHLIST_SOURCE,
    )


@router.delete("/me/watchlist/{item_id}", response_model=MovieStatusResponse)
def remove_from_watchlist(
    item_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _remove_interaction(request, db, current_user.id, item_id, source=WATCHLIST_SOURCE, interaction_type="click")
    return get_movie_status(db, current_user.id, item_id)


@router.delete("/me/watched/{item_id}", response_model=MovieStatusResponse)
def remove_from_watched(
    item_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _remove_interaction(request, db, current_user.id, item_id, source=LIBRARY_SOURCE, interaction_type="view")
    return get_movie_status(db, current_user.id, item_id)


@router.delete("/me/ratings/{item_id}", response_model=MovieStatusResponse)
def clear_rating(
    item_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _remove_interaction(request, db, current_user.id, item_id, source=LIBRARY_SOURCE, interaction_type="rating")
    return get_movie_status(db, current_user.id, item_id)


def _remove_interaction(
    request: Request,
    db: Session,
    user_id: int,
    item_id: int,
    *,
    source: str,
    interaction_type: str,
) -> None:
    if db.get(Item, item_id) is None:
        raise HTTPException(status_code=404, detail="Movie not found")

    removed = remove_app_interactions(
        db,
        user_id,
        item_id,
        source=source,
        interaction_type=interaction_type,
    )
    if removed == 0:
        raise HTTPException(status_code=404, detail="Nothing to remove")

    serving = getattr(request.app.state, "serving", None)
    if serving is not None and serving.user_cache is not None:
        serving.user_cache.invalidate(user_id)
        serving.user_cache.invalidate(1_000_000_000 + user_id)


@router.post("/me/watched", response_model=LibraryItem, status_code=201)
def mark_watched(
    body: AddMovieRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return _log_app_interaction(
        request,
        db,
        current_user,
        body.item_id,
        interaction_type="view",
        source="library",
    )


@router.post("/me/ratings", response_model=MovieStatusResponse, status_code=201)
def rate_movie(
    body: RatingRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if db.get(Item, body.item_id) is None:
        raise HTTPException(status_code=404, detail="Movie not found")

    serving = getattr(request.app.state, "serving", None)
    user_cache = serving.user_cache if serving is not None else None

    log_interaction(
        db,
        user_id=current_user.id,
        item_id=body.item_id,
        interaction_type="rating",
        context_json={"source": "library", "rating": body.rating},
        user_cache=user_cache,
    )
    return get_movie_status(db, current_user.id, body.item_id)


@router.post("/me/dismissed", response_model=MovieStatusResponse, status_code=201)
def dismiss_recommendation(
    body: AddMovieRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _log_app_interaction(
        request,
        db,
        current_user,
        body.item_id,
        interaction_type="click",
        source=DISMISSED_SOURCE,
        response_kind="status",
    )
    return get_movie_status(db, current_user.id, body.item_id)


def _log_app_interaction(
    request: Request,
    db: Session,
    current_user: User,
    item_id: int,
    *,
    interaction_type: str,
    source: str,
    response_kind: str = "library",
) -> LibraryItem | dict:
    if db.get(Item, item_id) is None:
        raise HTTPException(status_code=404, detail="Movie not found")

    serving = getattr(request.app.state, "serving", None)
    user_cache = serving.user_cache if serving is not None else None

    try:
        log_interaction(
            db,
            user_id=current_user.id,
            item_id=item_id,
            interaction_type=interaction_type,
            context_json={"source": source},
            user_cache=user_cache,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if response_kind == "status":
        return get_movie_status(db, current_user.id, item_id)

    if source == WATCHLIST_SOURCE:
        watchlist = load_user_watchlist(db, current_user.id)
        added = next((entry for entry in watchlist if entry["item_id"] == item_id), None)
        if added is None:
            raise HTTPException(status_code=500, detail="Failed to update watchlist")
        return added

    library = load_user_library(db, current_user.id)
    added = next((entry for entry in library if entry["item_id"] == item_id), None)
    if added is None:
        raise HTTPException(status_code=500, detail="Failed to update library")
    return added


def _library_recommendations(
    request: Request,
    db: Session,
    *,
    user_id: int,
    k: int,
    genres: list[str] | None = None,
    library: list[dict] | None = None,
) -> list[dict]:
    serving = getattr(request.app.state, "serving", None)
    if serving is None:
        error = getattr(request.app.state, "serving_error", "serving context not loaded")
        raise HTTPException(status_code=503, detail=f"Recommendation service unavailable: {error}")

    if library is None:
        library = load_user_library(db, user_id)

    if serving.service is not None:
        try:
            results, _timing = serving.service.recommend(
                db,
                user_id=user_id,
                k=k,
                library_only=True,
                genres=genres,
                library_entries=library,
            )
        except INFERENCE_ERRORS:
            results = _library_popularity_results(
                serving, db, user_id, k, genres=genres, library=library
            )
    else:
        results = _library_popularity_results(
            serving, db, user_id, k, genres=genres, library=library
        )

    item_details = load_items_by_ids(db, [rec.item_id for rec in results])
    payload: list[dict] = []
    for rec in results:
        details = item_details.get(rec.item_id, {})
        payload.append(
            {
                "item_id": rec.item_id,
                "title": rec.title,
                "genres": details.get("genres", []),
                "year": details.get("year"),
                "poster_url": details.get("poster_url"),
                "image_url": details.get("image_url"),
            }
        )
    return payload


def _library_popularity_results(
    serving,
    db: Session,
    user_id: int,
    k: int,
    *,
    genres: list[str] | None = None,
    library: list[dict] | None = None,
):
    from app.services.library_service import load_excluded_recommendation_items

    seen_items = load_excluded_recommendation_items(db, user_id)
    if library is None:
        library = load_user_library(db, user_id)
    genre_profile = library_genre_profile(library)

    if genres or genre_profile:
        filtered = genre_weighted_popularity_candidates(
            db,
            serving.popularity_ranking,
            seen_items,
            genre_profile=genre_profile if not genres else None,
            filter_genres=genres,
            k=k,
        )
    else:
        filtered = filter_popularity_candidates(serving.popularity_ranking, seen_items, k)
    return attach_titles(db, filtered)
