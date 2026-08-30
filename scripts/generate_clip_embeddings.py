"""Generate fused CLIP embeddings for items and store in item_embeddings."""

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
from app.services.clip_embeddings import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_IMAGE_WORKERS,
    DEFAULT_MODEL,
    count_remaining_embeddings,
    resolve_device,
    run_clip_embedding_generation,
)


def check_database(session) -> None:
    try:
        session.execute(text("SELECT 1"))
    except OperationalError:
        print("Cannot connect to PostgreSQL on localhost:5432.")
        print("Start Docker Desktop, then run: docker compose up -d postgres")
        raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate CLIP embeddings for items")
    parser.add_argument(
        "--posters-dir",
        type=Path,
        default=Path(settings.posters_dir),
        help="Directory containing poster images (default: data/posters)",
    )
    parser.add_argument(
        "--model",
        default=settings.clip_model_name,
        help=f"CLIP model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"Items per batch (default: 128 on GPU/MPS, 64 on CPU)",
    )
    parser.add_argument(
        "--image-workers",
        type=int,
        default=DEFAULT_IMAGE_WORKERS,
        help=f"Parallel poster loaders per batch (default: {DEFAULT_IMAGE_WORKERS})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max items to embed (default: all needing embeddings)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed even if a vector already exists",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (default: cuda if available else cpu)",
    )
    args = parser.parse_args()
    resolved_device = resolve_device(args.device)

    session = SessionLocal()
    try:
        check_database(session)
        print(
            f"Generating CLIP embeddings "
            f"(model={args.model}, device={resolved_device}, "
            f"batch_size={args.batch_size or 'auto'}) ..."
        )
        counts = run_clip_embedding_generation(
            session,
            posters_dir=args.posters_dir,
            model_name=args.model,
            batch_size=args.batch_size,
            limit=args.limit,
            force=args.force,
            device=args.device,
            image_workers=args.image_workers,
        )
        remaining = count_remaining_embeddings(session)
    finally:
        session.close()

    for key, value in counts.items():
        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")

    if not args.force:
        print(f"  remaining_without_embedding: {remaining:,}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
