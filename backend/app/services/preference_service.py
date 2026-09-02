"""User recommendation preferences stored in profile_json."""

from __future__ import annotations

from app.models.user import User
from app.services.item_service import CANONICAL_GENRES
from sqlalchemy.orm import Session

_CANONICAL_SET = set(CANONICAL_GENRES)
_CANONICAL_ORDER = {genre: index for index, genre in enumerate(CANONICAL_GENRES)}


def _filter_canonical(genres: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for genre in genres:
        label = str(genre).strip()
        if not label or label not in _CANONICAL_SET or label in seen:
            continue
        seen.add(label)
        cleaned.append(label)
    cleaned.sort(key=lambda genre: _CANONICAL_ORDER[genre])
    return cleaned


def get_preferred_genres(user: User) -> list[str]:
    profile = user.profile_json or {}
    genres = profile.get("preferred_genres")
    if not isinstance(genres, list):
        return []
    raw = [str(genre).strip() for genre in genres if str(genre).strip()]
    return _filter_canonical(raw)


def set_preferred_genres(session: Session, user: User, genres: list[str]) -> list[str]:
    cleaned = _filter_canonical(genres)

    profile = dict(user.profile_json or {})
    profile["preferred_genres"] = cleaned
    user.profile_json = profile
    session.commit()
    session.refresh(user)
    return cleaned


def sync_preferred_genres(session: Session, user: User) -> list[str]:
    """Return canonical preferences and drop any stale values stored earlier."""
    profile = user.profile_json or {}
    genres = profile.get("preferred_genres")
    if not isinstance(genres, list):
        return []

    raw = [str(genre).strip() for genre in genres if str(genre).strip()]
    cleaned = _filter_canonical(raw)
    if cleaned != raw:
        profile = dict(profile)
        profile["preferred_genres"] = cleaned
        user.profile_json = profile
        session.commit()
        session.refresh(user)
    return cleaned
