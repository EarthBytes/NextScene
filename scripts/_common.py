"""Shared helpers for CLI scripts."""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.exc import OperationalError


def require_database(session) -> None:
    try:
        session.execute(text("SELECT 1"))
    except OperationalError:
        print("Cannot connect to PostgreSQL on localhost:5432.")
        print("Start Docker Desktop, then run: docker compose up -d postgres")
        raise SystemExit(1)
