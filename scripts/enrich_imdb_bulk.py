"""Enrich items from IMDb bulk datasets."""

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from app.db.session import SessionLocal
from app.services.imdb_bulk_enrich import (
    IMDB_DOWNLOAD_BASE,
    required_imdb_files,
    run_imdb_bulk_enrichment,
)


def check_database(session) -> None:
    try:
        session.execute(text("SELECT 1"))
    except OperationalError:
        print("Cannot connect to PostgreSQL on localhost:5432.")
        print("Start Docker Desktop, then run: docker compose up -d postgres")
        raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enrich items from IMDb bulk datasets (datasets.imdbws.com)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/imdb"),
        help="Directory containing downloaded .tsv.gz files",
    )
    parser.add_argument(
        "--skip-cast",
        action="store_true",
        help="Skip cast enrichment (avoids large principals/name files)",
    )
    args = parser.parse_args()

    missing = [
        f
        for f in required_imdb_files(include_cast=not args.skip_cast)
        if not (args.data_dir / f).exists()
    ]
    if missing:
        print(f"Missing files in {args.data_dir}: {', '.join(missing)}")
        print(f"Download from {IMDB_DOWNLOAD_BASE}/")
        for name in missing:
            print(f"  {IMDB_DOWNLOAD_BASE}/{name}")
        return 1

    session = SessionLocal()
    try:
        check_database(session)
        print(f"Enriching from {args.data_dir.resolve()} ...")
        counts = run_imdb_bulk_enrichment(
            session,
            args.data_dir,
            include_cast=not args.skip_cast,
        )
    finally:
        session.close()

    for key, value in counts.items():
        print(f"  {key}: {value:,}")

    print("Done. Next: python scripts/fetch_tmdb_metadata.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
