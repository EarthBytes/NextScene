"""Load MovieLens 20M ratings, movies, and tags into PostgreSQL."""

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/ingest_movielens.py` from repo root.
BACKEND_ROOT = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from app.db.session import SessionLocal
from app.services.movielens_ingest import run_ingestion


def check_database(session) -> None:
    try:
        session.execute(text("SELECT 1"))
    except OperationalError:
        print("Cannot connect to PostgreSQL on localhost:5432.")
        print("Start Docker Desktop, then run: docker compose up -d postgres")
        raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest MovieLens 20M into the database")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/movielens"),
        help="Directory containing ratings.csv, movies.csv, tags.csv",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="Rows per batch for ratings/tags interaction inserts",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Truncate items/interactions before ingest",
    )
    parser.add_argument(
        "--skip-ratings",
        action="store_true",
        help="Skip ratings.csv (useful if only tags/movies are available)",
    )
    args = parser.parse_args()

    required = ["movies.csv", "tags.csv"]
    missing = [f for f in required if not (args.data_dir / f).exists()]
    if missing:
        print(f"Missing files in {args.data_dir}: {', '.join(missing)}")
        print("Download from https://grouplens.org/datasets/movielens/20m/")
        return 1

    ratings_path = args.data_dir / "ratings.csv"
    if not ratings_path.exists() and not args.skip_ratings:
        print("Warning: ratings.csv not found — ingesting movies/tags only.")
        print("Re-download MovieLens 20M for full interaction history.")
        args.skip_ratings = True

    print(f"Ingesting from {args.data_dir.resolve()} ...")
    session = SessionLocal()
    try:
        check_database(session)
        counts = run_ingestion(
            session,
            args.data_dir,
            batch_size=args.batch_size,
            clear=args.clear,
            skip_ratings=args.skip_ratings,
        )
    finally:
        session.close()

    for key, value in counts.items():
        print(f"  {key}: {value:,}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
