"""Fetch IMDb metadata via OMDb API and enrich items table."""

import argparse

import _bootstrap  # noqa: F401

from _common import require_database

from app.config import settings
from app.db.session import SessionLocal
from app.services.omdb_metadata import count_remaining, run_metadata_fetch


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch IMDb metadata via OMDb")
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max items to fetch this run (free tier: ~1,000/day)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch even if description/image already exist",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.25,
        help="Seconds to wait between API requests",
    )
    args = parser.parse_args()

    api_key = settings.omdb_api_key
    if not api_key:
        print("Set OMDB_API_KEY in .env. Get a key at http://www.omdbapi.com/")
        return 1

    session = SessionLocal()
    try:
        require_database(session)
        print(f"Fetching OMDb metadata (limit={args.limit}) ...")
        counts = run_metadata_fetch(
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

    if remaining > 0 and not args.force:
        print("Free OMDb tier allows ~1,000 requests/day — re-run daily until remaining is 0.")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
