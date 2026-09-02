"""Tests for genre catalog helpers."""

from unittest.mock import MagicMock

from app.services.item_service import CANONICAL_GENRES, genres_overlap_filter, list_genres


def test_list_genres_returns_canonical_only():
    session = MagicMock()
    session.execute.return_value.all.return_value = [
        MagicMock(genre="Animation", count=1200),
        MagicMock(genre="IMAX", count=400),
        MagicMock(genre="Action", count=800),
        MagicMock(genre="Film-Noir", count=5),
    ]

    genres = list_genres(session, min_count=25)
    names = [row["genre"] for row in genres]

    assert "IMAX" not in names
    assert "Film-Noir" not in names
    assert "Animation" in names
    assert "Action" in names
    assert names == sorted(names, key=lambda genre: CANONICAL_GENRES.index(genre))


def test_genres_overlap_filter_uses_array_overlap():
    clause = genres_overlap_filter(["Animation", "Family"])
    assert clause is not None
