"""Fetch plot and poster metadata from TMDb."""

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from app.config import settings
from app.db.session import SessionLocal
from app.services.tmdb_metadata import count_remaining, run_tmdb_fetch


def check_database(session) -> None:
    try:
        session.execute(text("SELECT 1"))
    except OperationalError:
        print("Cannot connect to PostgreSQL on localhost:5432.")
        print("Start Docker Desktop, then run: docker compose up -d postgres")
        raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch plot and posters from TMDb")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max items to fetch (default: all needing metadata)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch even if description/image already exist",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.26,
        help="Seconds between requests (~4/sec, under TMDb rate limits)",
    )
    args = parser.parse_args()

    api_key = settings.tmdb_api_key
    if not api_key:
        print("Set TMDB_API_KEY in .env. Get a key at https://www.themoviedb.org/settings/api")
        return 1

    session = SessionLocal()
    try:
        check_database(session)
        print("Fetching TMDb metadata ...")
        counts = run_tmdb_fetch(
            session,
            api_key=api_key,
            limit=args.limit,
            force=args.force,
            delay_seconds=args.delay,
        )
        remaining = count_remaining(session)
    finally:
        session.close()

    for key, value in counts.items():
        print(f"  {key}: {value:,}")

    if not args.force:
        print(f"  remaining_without_metadata: {remaining:,}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
