"""Tests that app library interactions are isolated from legacy MovieLens data."""

from datetime import UTC, datetime

import pytest
from app.db.session import SessionLocal
from app.models.interaction import Interaction
from app.models.item import Item
from app.services.explanation_service import explain_recommendation_natural
from app.services.library_service import load_library_history, load_user_library


@pytest.fixture
def db_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def _add_item(session, item_id: int, title: str, genres: list[str]) -> None:
    existing = session.get(Item, item_id)
    if existing is not None:
        return
    session.add(
        Item(
            item_id=item_id,
            title=title,
            genres=genres,
            metadata_json={},
        )
    )


def _add_interaction(session, user_id: int, item_id: int, *, source: str | None = None) -> None:
    context = {"source": source} if source else {}
    session.add(
        Interaction(
            user_id=user_id,
            item_id=item_id,
            ts=datetime.now(UTC),
            type="view" if source == "library" else "rating",
            context_json=context,
        )
    )


def test_library_history_ignores_movielens_interactions(db_session):
    user_id = 999_001
    _add_item(db_session, 999_100, "Star Wars (1977)", ["Action", "Sci-Fi"])
    _add_item(db_session, 999_200, "Godsend (2004)", ["Horror"])
    _add_item(db_session, 999_300, "French Connection, The (1971)", ["Action", "Crime"])

    _add_interaction(db_session, user_id, 999_200, source=None)
    _add_interaction(db_session, user_id, 999_300, source=None)
    _add_interaction(db_session, user_id, 999_100, source="library")
    db_session.commit()

    library = load_user_library(db_session, user_id)
    assert len(library) == 1
    assert library[0]["title"] == "Star Wars (1977)"

    history = load_library_history(db_session, user_id)
    assert history == [999_100]


def test_natural_explanation_uses_library_only(db_session):
    user_id = 999_007
    _add_item(db_session, 999_101, "Star Wars: Episode IV (1977)", ["Action", "Adventure", "Sci-Fi"])
    _add_item(db_session, 999_102, "Star Wars: Episode V (1980)", ["Action", "Adventure", "Sci-Fi"])
    _add_item(db_session, 999_103, "Star Wars: Episode VI (1983)", ["Action", "Adventure", "Sci-Fi"])
    _add_item(db_session, 999_900, "Godsend (2004)", ["Horror"])
    _add_item(db_session, 999_901, "French Connection, The (1971)", ["Action", "Crime"])

    for item_id in (999_101, 999_102, 999_103):
        _add_interaction(db_session, user_id, item_id, source="library")

    _add_interaction(db_session, user_id, 999_900, source=None)
    db_session.commit()

    result = explain_recommendation_natural(
        db_session,
        service=None,  # type: ignore[arg-type]
        user_id=user_id,
        item_id=999_901,
        catalog_searcher=None,  # type: ignore[arg-type]
    )

    assert "Star Wars" in result.explanation
    assert len(result.reasons) >= 1
    assert all("Godsend" not in title for title in result.related_titles)
    assert len(result.related_titles) == 3
    assert "Comedy" not in result.shared_genres


def test_natural_explanation_does_not_claim_library_genres_not_present(db_session):
    user_id = 999_008
    _add_item(db_session, 999_110, "Toy Story (1995)", ["Animation", "Children"])
    _add_item(db_session, 999_111, "Finding Nemo (2003)", ["Animation", "Adventure"])
    _add_item(db_session, 999_112, "Forrest Gump (1994)", ["Drama", "Romance", "Comedy"])

    for item_id in (999_110, 999_111):
        _add_interaction(db_session, user_id, item_id, source="library")
    db_session.commit()

    result = explain_recommendation_natural(
        db_session,
        service=None,  # type: ignore[arg-type]
        user_id=user_id,
        item_id=999_112,
        catalog_searcher=None,  # type: ignore[arg-type]
    )

    assert "Comedy" not in result.explanation
    assert all("Comedy" not in reason for reason in result.reasons)
