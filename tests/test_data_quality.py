"""Data quality checks for catalog and preference invariants."""

from app.services.item_service import CANONICAL_GENRES, genres_overlap_filter
from app.services.preference_service import _filter_canonical


def test_canonical_genres_are_unique():
    assert len(CANONICAL_GENRES) == len(set(CANONICAL_GENRES))


def test_preference_filter_drops_invalid_genres():
    cleaned = _filter_canonical(["News", "Animation", "Adult"])
    assert cleaned == ["Animation"]


def test_genre_overlap_filter_builds_clause():
    clause = genres_overlap_filter(["Animation"])
    assert clause is not None
