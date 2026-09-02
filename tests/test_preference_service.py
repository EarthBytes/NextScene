"""Tests for user preference storage."""

from unittest.mock import MagicMock

from app.services.preference_service import (
    get_preferred_genres,
    set_preferred_genres,
    sync_preferred_genres,
)


def test_get_and_set_preferred_genres():
    user = MagicMock()
    user.profile_json = {}

    session = MagicMock()
    genres = set_preferred_genres(session, user, ["Animation", "animation", " Comedy "])
    assert genres == ["Animation", "Comedy"]
    assert user.profile_json["preferred_genres"] == ["Animation", "Comedy"]
    session.commit.assert_called_once()


def test_get_preferred_genres_defaults_empty():
    user = MagicMock()
    user.profile_json = {}
    assert get_preferred_genres(user) == []


def test_get_preferred_genres_filters_non_canonical():
    user = MagicMock()
    user.profile_json = {
        "preferred_genres": ["News", "Adult", "Animation", "Reality-TV"],
    }
    assert get_preferred_genres(user) == ["Animation"]


def test_sync_preferred_genres_clears_stale_values():
    user = MagicMock()
    user.profile_json = {"preferred_genres": ["News", "Comedy"]}
    session = MagicMock()

    cleaned = sync_preferred_genres(session, user)

    assert cleaned == ["Comedy"]
    assert user.profile_json["preferred_genres"] == ["Comedy"]
    session.commit.assert_called_once()
