from pathlib import Path

import pytest
from app.db.session import SessionLocal
from app.services.movielens_ingest import format_imdb_id, parse_genres, run_ingestion
from sqlalchemy import text

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "movielens"
FIXTURE_ITEM_IDS = (880001, 880002)


def test_format_imdb_id():
    assert format_imdb_id("0114709") == "tt0114709"
    assert format_imdb_id(114709) == "tt0114709"


def test_parse_genres():
    assert parse_genres("Adventure|Comedy") == ["Adventure", "Comedy"]
    assert parse_genres("(no genres listed)") is None


@pytest.fixture
def db_session():
    session = SessionLocal()
    try:
        session.execute(text("SELECT 1"))
    except Exception:
        session.close()
        pytest.skip("PostgreSQL not available")
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def cleanup_fixture_rows(db_session):
    yield
    db_session.execute(
        text("DELETE FROM interactions WHERE item_id = ANY(:ids)"),
        {"ids": list(FIXTURE_ITEM_IDS)},
    )
    db_session.execute(
        text("DELETE FROM items WHERE item_id = ANY(:ids)"),
        {"ids": list(FIXTURE_ITEM_IDS)},
    )
    db_session.commit()


def test_run_ingestion_fixture(db_session, cleanup_fixture_rows):
    counts = run_ingestion(
        db_session,
        FIXTURE_DIR,
        batch_size=10,
        clear=False,
        skip_ratings=False,
    )

    assert counts["items"] == 2
    assert counts["links_updated"] == 2
    assert counts["tag_interactions"] == 3
    assert counts["rating_interactions"] == 3

    row = db_session.execute(
        text("SELECT imdb_id, metadata_json FROM items WHERE item_id = 880001")
    ).one()
    assert row.imdb_id == "tt0114709"
    assert "pixar" in row.metadata_json["tags"]

    interaction_types = db_session.execute(
        text(
            """
            SELECT type, COUNT(*) AS count
            FROM interactions
            WHERE item_id = ANY(:ids)
            GROUP BY type
            ORDER BY type
            """
        ),
        {"ids": list(FIXTURE_ITEM_IDS)},
    ).all()
    assert [(r.type, r.count) for r in interaction_types] == [("rating", 3), ("tag", 3)]
