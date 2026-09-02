"""Tests for genre-weighted recommendations."""

from collections import Counter
from unittest.mock import MagicMock

from app.services.recommendation_service import (
    filter_candidates_by_genres,
    genre_weighted_popularity_candidates,
    item_matches_genres,
)


def test_item_matches_genres_case_insensitive():
    assert item_matches_genres({"animation", "family"}, {"Animation"})
    assert not item_matches_genres({"action"}, {"Animation"})


def test_filter_candidates_by_genres():
    session = MagicMock()
    session.execute.return_value.all.return_value = [
        MagicMock(item_id=1, genres=["Animation"]),
        MagicMock(item_id=2, genres=["Action"]),
        MagicMock(item_id=3, genres=["Animation", "Comedy"]),
    ]

    candidates = [(1, 0.9), (2, 0.8), (3, 0.7)]
    filtered = filter_candidates_by_genres(session, candidates, ["Animation"], k=5)
    assert [item_id for item_id, _score in filtered] == [1, 3]


def test_genre_weighted_popularity_prefers_library_genres():
    session = MagicMock()
    session.execute.return_value.all.return_value = [
        MagicMock(item_id=100, genres=["Drama", "Romance"]),
        MagicMock(item_id=200, genres=["Animation", "Family"]),
        MagicMock(item_id=300, genres=["Animation"]),
    ]

    profile = Counter({"animation": 3, "family": 2})
    results = genre_weighted_popularity_candidates(
        session,
        popularity_ranking=[100, 200, 300],
        seen_items=set(),
        genre_profile=profile,
        k=2,
    )

    assert [item_id for item_id, _score in results] == [200, 300]
