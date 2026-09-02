"""Shared pytest fixtures."""

from __future__ import annotations

import pytest
from app.db.session import SessionLocal
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError


def _postgres_available() -> bool:
    session = SessionLocal()
    try:
        session.execute(text("SELECT 1"))
        session.execute(text("SELECT to_regclass('public.items')"))
        return True
    except SQLAlchemyError:
        return False
    finally:
        session.close()


@pytest.fixture
def db_session():
    """Database session for integration tests; skips when Postgres/schema is unavailable."""
    if not _postgres_available():
        pytest.skip("PostgreSQL not available")
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
