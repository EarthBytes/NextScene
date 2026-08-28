"""Download poster images from item image_url into local storage."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.models.item import Item

POSTER_EXTENSIONS = ("jpg", "jpeg", "png", "webp")
CONTENT_TYPE_EXTENSIONS = {
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
}
DEFAULT_WORKERS = 64
DEFAULT_COMMIT_BATCH = 500


def poster_filename(item_id: int, ext: str = "jpg") -> str:
    return f"{item_id}.{ext.lstrip('.').lower()}"


def poster_path_for_item(posters_dir: Path, item_id: int, ext: str = "jpg") -> Path:
    return posters_dir / poster_filename(item_id, ext)


def infer_extension_from_url(url: str) -> str | None:
    path = urlparse(url).path.lower()
    for ext in POSTER_EXTENSIONS:
        if path.endswith(f".{ext}"):
            return "jpg" if ext == "jpeg" else ext
    return None


def infer_extension(content_type: str | None, url: str) -> str:
    if content_type:
        normalized = content_type.split(";", 1)[0].strip().lower()
        if normalized in CONTENT_TYPE_EXTENSIONS:
            return CONTENT_TYPE_EXTENSIONS[normalized]

    from_url = infer_extension_from_url(url)
    if from_url:
        return from_url

    return "jpg"


def find_existing_poster(posters_dir: Path, item_id: int) -> Path | None:
    for ext in POSTER_EXTENSIONS:
        path = poster_path_for_item(posters_dir, item_id, ext)
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def existing_poster_item_ids(posters_dir: Path) -> set[int]:
    if not posters_dir.is_dir():
        return set()

    item_ids: set[int] = set()
    for path in posters_dir.iterdir():
        if not path.is_file():
            continue
        ext = path.suffix.lower().lstrip(".")
        if ext not in POSTER_EXTENSIONS:
            continue
        if path.stat().st_size == 0:
            continue
        try:
            item_ids.add(int(path.stem))
        except ValueError:
            continue
    return item_ids


def has_local_poster(
    session: Session,
    posters_dir: Path,
    item_id: int,
) -> Path | None:
    item = session.get(Item, item_id)
    poster_path = (item.metadata_json or {}).get("poster_path") if item else None
    if poster_path:
        path = Path(poster_path)
        if path.is_file() and path.stat().st_size > 0:
            return path
    return find_existing_poster(posters_dir, item_id)


def items_with_poster_urls(
    session: Session,
    posters_dir: Path,
    limit: int,
    force: bool = False,
    existing_ids: set[int] | None = None,
) -> list[tuple[int, str]]:
    rows = session.execute(
        text(
            """
            SELECT item_id, image_url
            FROM items
            WHERE image_url IS NOT NULL
              AND TRIM(image_url) <> ''
            ORDER BY item_id
            """
        )
    ).all()

    on_disk = existing_ids if existing_ids is not None else existing_poster_item_ids(posters_dir)
    results: list[tuple[int, str]] = []
    for row in rows:
        if len(results) >= limit:
            break

        item_id = int(row.item_id)
        image_url = str(row.image_url).strip()
        if not image_url:
            continue
        if not force and item_id in on_disk:
            continue
        results.append((item_id, image_url))
    return results


def count_remaining(session: Session, posters_dir: Path) -> int:
    on_disk = existing_poster_item_ids(posters_dir)
    rows = session.execute(
        text(
            """
            SELECT item_id
            FROM items
            WHERE image_url IS NOT NULL
              AND TRIM(image_url) <> ''
            """
        )
    ).all()
    return sum(1 for row in rows if int(row.item_id) not in on_disk)


def _record_poster_path(item: Item, poster_path: Path) -> None:
    current = dict(item.metadata_json or {})
    current["poster_path"] = str(poster_path)
    item.metadata_json = current


def _commit_poster_paths(session: Session, updates: list[tuple[int, Path]]) -> None:
    for item_id, poster_path in updates:
        item = session.get(Item, item_id)
        if item is not None:
            _record_poster_path(item, poster_path)
    session.commit()


def download_poster(
    url: str,
    dest_path: Path,
    client: httpx.Client,
    max_retries: int = 3,
) -> bool:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    for attempt in range(max_retries):
        try:
            with client.stream("GET", url, follow_redirects=True) as response:
                if response.status_code == 404:
                    return False
                response.raise_for_status()

                ext = infer_extension(response.headers.get("content-type"), url)
                if dest_path.suffix.lower() != f".{ext}":
                    dest_path = dest_path.with_suffix(f".{ext}")
                    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")

                with temp_path.open("wb") as handle:
                    for chunk in response.iter_bytes():
                        handle.write(chunk)

            if temp_path.stat().st_size == 0:
                temp_path.unlink(missing_ok=True)
                return False

            temp_path.replace(dest_path)
            return True
        except httpx.HTTPError:
            temp_path.unlink(missing_ok=True)
            if attempt + 1 == max_retries:
                return False
            time.sleep(0.5 * (attempt + 1))

    return False


async def download_poster_async(
    url: str,
    dest_path: Path,
    client: httpx.AsyncClient,
    max_retries: int = 3,
) -> Path | None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    for attempt in range(max_retries):
        try:
            async with client.stream("GET", url, follow_redirects=True) as response:
                if response.status_code == 404:
                    return None
                response.raise_for_status()

                ext = infer_extension(response.headers.get("content-type"), url)
                if dest_path.suffix.lower() != f".{ext}":
                    dest_path = dest_path.with_suffix(f".{ext}")
                    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")

                with temp_path.open("wb") as handle:
                    async for chunk in response.aiter_bytes():
                        handle.write(chunk)

            if temp_path.stat().st_size == 0:
                temp_path.unlink(missing_ok=True)
                return None

            temp_path.replace(dest_path)
            return dest_path
        except httpx.HTTPError:
            temp_path.unlink(missing_ok=True)
            if attempt + 1 == max_retries:
                return None
            await asyncio.sleep(0.25 * (attempt + 1))

    return None


async def _download_batch(
    items: list[tuple[int, str]],
    posters_dir: Path,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    max_retries: int,
) -> list[tuple[int, Path | None]]:
    async def download_one(item_id: int, image_url: str) -> tuple[int, Path | None]:
        async with semaphore:
            ext = infer_extension_from_url(image_url) or "jpg"
            dest_path = poster_path_for_item(posters_dir, item_id, ext)
            saved_path = await download_poster_async(
                image_url,
                dest_path,
                client,
                max_retries=max_retries,
            )
            return item_id, saved_path

    return await asyncio.gather(*(download_one(item_id, url) for item_id, url in items))


async def _run_poster_download_async(
    session: Session,
    posters_dir: Path,
    items: list[tuple[int, str]],
    workers: int,
    max_retries: int,
    commit_batch: int,
) -> dict[str, int]:
    counts = {
        "queued": len(items),
        "downloaded": 0,
        "failed": 0,
    }
    pending_updates: list[tuple[int, Path]] = []

    limits = httpx.Limits(
        max_connections=workers,
        max_keepalive_connections=workers,
    )
    semaphore = asyncio.Semaphore(workers)

    async with httpx.AsyncClient(timeout=30.0, limits=limits) as client:
        chunk_size = max(workers * 4, 256)
        for offset in range(0, len(items), chunk_size):
            batch = items[offset : offset + chunk_size]
            results = await _download_batch(
                batch,
                posters_dir,
                client,
                semaphore,
                max_retries,
            )

            for item_id, saved_path in results:
                if saved_path is None:
                    counts["failed"] += 1
                else:
                    counts["downloaded"] += 1
                    pending_updates.append((item_id, saved_path))

            if len(pending_updates) >= commit_batch:
                _commit_poster_paths(session, pending_updates)
                pending_updates.clear()

    if pending_updates:
        _commit_poster_paths(session, pending_updates)

    return counts


def run_poster_download(
    session: Session,
    posters_dir: Path,
    limit: int | None = None,
    force: bool = False,
    max_retries: int = 3,
    workers: int = DEFAULT_WORKERS,
    commit_batch: int = DEFAULT_COMMIT_BATCH,
) -> dict[str, int]:
    fetch_limit = limit if limit is not None else 10_000_000
    posters_dir.mkdir(parents=True, exist_ok=True)

    on_disk = existing_poster_item_ids(posters_dir)
    skipped_existing = 0
    if not force:
        rows = session.execute(
            text(
                """
                SELECT item_id
                FROM items
                WHERE image_url IS NOT NULL
                  AND TRIM(image_url) <> ''
                """
            )
        ).all()
        missing_metadata: list[tuple[int, Path]] = []
        for row in rows:
            item_id = int(row.item_id)
            if item_id not in on_disk:
                continue
            poster_path = find_existing_poster(posters_dir, item_id)
            if poster_path is None:
                continue
            item = session.get(Item, item_id)
            if item is None:
                continue
            current_path = (item.metadata_json or {}).get("poster_path")
            if current_path != str(poster_path):
                missing_metadata.append((item_id, poster_path))
        if missing_metadata:
            _commit_poster_paths(session, missing_metadata)
        skipped_existing = len(on_disk)

    items = items_with_poster_urls(
        session,
        posters_dir=posters_dir,
        limit=fetch_limit,
        force=force,
        existing_ids=on_disk if not force else set(),
    )

    if not items:
        return {
            "queued": 0,
            "downloaded": 0,
            "skipped_existing": skipped_existing,
            "failed": 0,
        }

    counts = asyncio.run(
        _run_poster_download_async(
            session,
            posters_dir,
            items,
            workers=workers,
            max_retries=max_retries,
            commit_batch=commit_batch,
        )
    )
    counts["skipped_existing"] = skipped_existing
    return counts
