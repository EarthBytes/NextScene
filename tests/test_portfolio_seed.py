from pathlib import Path

import pytest
from app.services.portfolio_seed import (
    filter_tags_by_movies,
    rank_movie_ids_by_tag_count,
    run_portfolio_seed,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "movielens"


def test_rank_movie_ids_by_tag_count():
    ranked = rank_movie_ids_by_tag_count(FIXTURE_DIR / "tags.csv", limit=10)
    assert ranked[0] == 880001
    assert 880002 in ranked


def test_filter_tags_by_movies():
    tags = {1: ["a"], 2: ["b"], 3: ["c"]}
    assert filter_tags_by_movies(tags, frozenset({1, 3})) == {1: ["a"], 3: ["c"]}


@pytest.fixture
def db_session():
    from app.db.session import SessionLocal
    from sqlalchemy import text
    from sqlalchemy.exc import OperationalError

    session = SessionLocal()
    try:
        session.execute(text("SELECT 1"))
    except OperationalError:
        session.close()
        pytest.skip("PostgreSQL not available")
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def cleanup_portfolio_rows(db_session):
    yield
    from sqlalchemy import text

    ids = [880001, 880002]
    db_session.execute(
        text("DELETE FROM interactions WHERE item_id = ANY(:ids)"),
        {"ids": ids},
    )
    db_session.execute(
        text("DELETE FROM items WHERE item_id = ANY(:ids)"),
        {"ids": ids},
    )
    db_session.commit()


def test_run_portfolio_seed_fixture(db_session, cleanup_portfolio_rows):
    counts = run_portfolio_seed(
        db_session,
        FIXTURE_DIR,
        max_movies=10,
        clear=False,
        sample_ratings=10,
        batch_size=5,
    )

    assert counts["items"] == 2
    assert counts["links_updated"] == 2
    assert counts["rating_interactions"] == 3
