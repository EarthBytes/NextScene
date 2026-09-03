"""Seed a small movie catalog for portfolio deploys (Render DB + Vercel UI)."""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from _common import require_database
from app.db.session import SessionLocal
from app.services.portfolio_seed import run_portfolio_seed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seed ~1.5k movies into PostgreSQL for Render/Vercel portfolio demo",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/movielens"),
        help="Directory with movies.csv, links.csv, tags.csv (and optional ratings.csv)",
    )
    parser.add_argument(
        "--max-movies",
        type=int,
        default=1_500,
        help="Number of movies to keep (ranked by tag coverage)",
    )
    parser.add_argument(
        "--sample-ratings",
        type=int,
        default=50_000,
        help="Max rating rows to ingest for popularity (0 to skip)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help="CSV chunk size for rating sample inserts",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Truncate items/interactions before seeding",
    )
    args = parser.parse_args()

    required = ["movies.csv", "links.csv", "tags.csv"]
    missing = [name for name in required if not (args.data_dir / name).exists()]
    if missing:
        print(f"Missing in {args.data_dir}: {', '.join(missing)}")
        print("Download MovieLens 20M CSVs locally (movies/links/tags only is enough).")
        return 1

    print(f"Portfolio seed → {args.data_dir.resolve()} (max_movies={args.max_movies:,})")
    print("Point DATABASE_URL at Render before running for production data.\n")

    session = SessionLocal()
    try:
        require_database(session)
        counts = run_portfolio_seed(
            session,
            args.data_dir,
            max_movies=args.max_movies,
            clear=args.clear,
            sample_ratings=args.sample_ratings,
            batch_size=args.batch_size,
        )
    finally:
        session.close()

    for key, value in counts.items():
        if key == "selected_movie_ids":
            continue
        print(f"  {key}: {value:,}")

    print("\nNext: fetch posters into the same database:")
    print("  DATABASE_URL='...render...' TMDB_API_KEY='...' \\")
    print("  PYTHONPATH=backend python scripts/fetch_tmdb_metadata.py")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
