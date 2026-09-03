"""Fetch plot and poster metadata from TMDb."""

import argparse
import os

import _bootstrap  # noqa: F401

from _common import require_database

from app.config import settings
from app.db.session import SessionLocal
from app.services.tmdb_metadata import count_remaining, run_tmdb_fetch


def resolve_tmdb_api_key() -> str:
    return (settings.tmdb_api_key or os.getenv("TMDB_API_KEY") or "").strip()


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
    parser.add_argument(
        "--commit-every",
        type=int,
        default=50,
        help="Commit DB updates every N movies (higher = faster over remote DB)",
    )
    args = parser.parse_args()

    api_key = resolve_tmdb_api_key()
    if not api_key:
        print("TMDB_API_KEY is missing or empty.")
        print("Add it to the repo-root .env file (not frontend/.env.local).")
        print("If you already set it, check for duplicate TMDB_API_KEY= lines;")
        print("an empty one at the bottom overrides the real key.")
        print("Get a key at https://www.themoviedb.org/settings/api")
        return 1

    session = SessionLocal()
    try:
        require_database(session)
        remaining_before = count_remaining(session)
        print(f"Fetching TMDb metadata ({remaining_before:,} movies missing posters/plots) ...")
        counts = run_tmdb_fetch(
            session,
            api_key=api_key,
            limit=args.limit,
            force=args.force,
            delay_seconds=args.delay,
            commit_every=args.commit_every,
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
