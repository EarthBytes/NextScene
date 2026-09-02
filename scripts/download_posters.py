"""Download poster images for items with image_url into data/posters/."""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from _common import require_database

from app.config import settings
from app.db.session import SessionLocal
from app.services.poster_download import DEFAULT_WORKERS, count_remaining, run_poster_download


def main() -> int:
    parser = argparse.ArgumentParser(description="Download poster images to data/posters/")
    parser.add_argument(
        "--posters-dir",
        type=Path,
        default=Path(settings.posters_dir),
        help="Directory to store poster images (default: data/posters)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max items to download (default: all needing posters)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if a local poster already exists",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Concurrent downloads (default: {DEFAULT_WORKERS})",
    )
    args = parser.parse_args()

    session = SessionLocal()
    try:
        require_database(session)
        print(f"Downloading posters to {args.posters_dir.resolve()} ({args.workers} workers) ...")
        counts = run_poster_download(
            session,
            posters_dir=args.posters_dir,
            limit=args.limit,
            force=args.force,
            workers=args.workers,
        )
        remaining = count_remaining(session, args.posters_dir)
    finally:
        session.close()

    for key, value in counts.items():
        print(f"  {key}: {value:,}")

    if not args.force:
        print(f"  remaining_without_poster: {remaining:,}")

    print("Done. Next: python scripts/generate_clip_embeddings.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
