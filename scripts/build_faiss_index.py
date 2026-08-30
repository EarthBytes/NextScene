"""Build a Faiss index from item_embeddings vectors."""

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
from app.services.faiss_index import run_faiss_index_build


def check_database(session) -> None:
    try:
        session.execute(text("SELECT 1"))
    except OperationalError:
        print("Cannot connect to PostgreSQL on localhost:5432.")
        print("Start Docker Desktop, then run: docker compose up -d postgres")
        raise SystemExit(1)


def main() -> int:
    try:
        import faiss  # noqa: F401
    except ImportError:
        print("Faiss is required. Install with: pip install -r requirements-ml.txt")
        return 1

    parser = argparse.ArgumentParser(description="Build Faiss vector index")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(settings.faiss_index_path),
        help="Path to save the Faiss index (default: data/faiss/items.index)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip post-build nearest-neighbor sanity checks",
    )
    args = parser.parse_args()

    session = SessionLocal()
    try:
        check_database(session)
        print(f"Building Faiss index at {args.output.resolve()} ...")
        counts = run_faiss_index_build(
            session,
            index_path=args.output,
            validate=not args.no_validate,
        )
    finally:
        session.close()

    for key, value in counts.items():
        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")

    if counts.get("validation_mismatches", 0):
        print("Warning: validation found mismatches in nearest-neighbor self-checks.")
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
