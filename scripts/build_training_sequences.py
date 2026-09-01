"""Pre-build cached user interaction sequences for fast transformer training."""

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
from app.services.sequence_cache import build_sequence_cache_from_db, cache_paths


def check_database(session) -> None:
    try:
        session.execute(text("SELECT 1"))
    except OperationalError:
        print("Cannot connect to PostgreSQL on localhost:5432.")
        print("Start Docker Desktop, then run: docker compose up -d postgres")
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cached user sequences for training")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(settings.sequences_cache_path),
        help="Cache directory (default: data/sequences)",
    )
    parser.add_argument(
        "--min-rating",
        type=float,
        default=3.5,
        help="Keep rating interactions at or above this value",
    )
    parser.add_argument(
        "--no-rating-filter",
        action="store_true",
        help="Include all rating interactions regardless of score",
    )
    parser.add_argument("--min-interactions", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    min_rating = None if args.no_rating_filter else args.min_rating

    session = SessionLocal()
    try:
        check_database(session)
        print(
            f"Building sequence cache at {args.output} "
            f"(min_rating={min_rating}, min_interactions={args.min_interactions}) ..."
        )
        sequences = build_sequence_cache_from_db(
            session,
            args.output,
            min_rating=min_rating,
            min_interactions=args.min_interactions,
        )
    finally:
        session.close()

    meta_path, npz_path = cache_paths(args.output)
    print(f"  users: {len(sequences):,}")
    print(f"  cache: {npz_path}")
    print(f"  meta: {meta_path}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
