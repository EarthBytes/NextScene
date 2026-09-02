"""Fetch plot and poster metadata from TMDb."""

from __future__ import annotations

import time
from typing import Any

import httpx
from app.models.item import Item
from app.services.metadata_utils import (
    apply_item_metadata,
    clean_api_string,
    count_items_missing_metadata,
)
from sqlalchemy import text
from sqlalchemy.orm import Session

TMDB_API_BASE = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"


def parse_tmdb_movie(data: dict[str, Any]) -> dict[str, Any] | None:
    if not data or data.get("success") is False:
        return None

    overview = clean_api_string(data.get("overview"), null_values=frozenset())
    poster_path = clean_api_string(data.get("poster_path"), null_values=frozenset())
    image_url = f"{TMDB_IMAGE_BASE}{poster_path}" if poster_path else None

    metadata: dict[str, Any] = {}
    if clean_api_string(data.get("tagline"), null_values=frozenset()):
        metadata["tagline"] = data["tagline"]
    if clean_api_string(data.get("release_date"), null_values=frozenset()):
        metadata["release_date"] = data["release_date"]
    if data.get("vote_average") is not None:
        metadata["tmdb_vote_average"] = data["vote_average"]
    if data.get("vote_count") is not None:
        metadata["tmdb_vote_count"] = data["vote_count"]
    if clean_api_string(data.get("original_language"), null_values=frozenset()):
        metadata["original_language"] = data["original_language"]

    genres = data.get("genres")
    genre_names = None
    if isinstance(genres, list):
        genre_names = [g["name"] for g in genres if g.get("name")]

    return {
        "description": overview,
        "image_url": image_url,
        "genres": genre_names,
        "metadata_json": metadata,
    }


def get_tmdb_id(item: Item) -> int | None:
    meta = item.metadata_json or {}
    tmdb_id = meta.get("tmdb_id")
    if tmdb_id is None:
        return None
    return int(tmdb_id)


def items_needing_tmdb(session: Session, limit: int, force: bool = False) -> list[Item]:
    if force:
        query = """
            SELECT item_id
            FROM items
            WHERE metadata_json ? 'tmdb_id'
            ORDER BY item_id
            LIMIT :limit
        """
    else:
        query = """
            SELECT item_id
            FROM items
            WHERE metadata_json ? 'tmdb_id'
              AND (description IS NULL OR image_url IS NULL)
            ORDER BY item_id
            LIMIT :limit
        """

    item_ids = [
        int(row.item_id)
        for row in session.execute(text(query), {"limit": limit}).all()
    ]
    items = []
    for item_id in item_ids:
        item = session.get(Item, item_id)
        if item is not None:
            items.append(item)
    return items


def apply_tmdb_metadata(item: Item, metadata: dict[str, Any]) -> None:
    apply_item_metadata(item, metadata, overwrite_genres=False)


def fetch_tmdb_movie(
    tmdb_id: int,
    api_key: str,
    client: httpx.Client,
) -> dict[str, Any] | None:
    response = client.get(
        f"{TMDB_API_BASE}/movie/{tmdb_id}",
        params={"api_key": api_key},
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return parse_tmdb_movie(response.json())


def count_remaining(session: Session) -> int:
    return count_items_missing_metadata(
        session,
        "metadata_json ? 'tmdb_id' AND (description IS NULL OR image_url IS NULL)",
    )


def run_tmdb_fetch(
    session: Session,
    api_key: str,
    limit: int | None = None,
    force: bool = False,
    delay_seconds: float = 0.26,
    max_retries: int = 3,
) -> dict[str, int]:
    fetch_limit = limit if limit is not None else 10_000_000
    items = items_needing_tmdb(session, limit=fetch_limit, force=force)
    counts = {"queued": len(items), "updated": 0, "not_found": 0, "failed": 0}

    with httpx.Client(timeout=30.0) as client:
        for item in items:
            tmdb_id = get_tmdb_id(item)
            if tmdb_id is None:
                counts["failed"] += 1
                continue

            metadata = None
            failed = False
            for attempt in range(max_retries):
                try:
                    metadata = fetch_tmdb_movie(tmdb_id, api_key, client)
                    break
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == 404:
                        metadata = None
                        break
                    if attempt + 1 == max_retries:
                        failed = True
                    else:
                        time.sleep(delay_seconds * (attempt + 1))
                except httpx.HTTPError:
                    if attempt + 1 == max_retries:
                        failed = True
                    else:
                        time.sleep(delay_seconds * (attempt + 1))

            if failed:
                counts["failed"] += 1
            elif metadata is None:
                counts["not_found"] += 1
            else:
                apply_tmdb_metadata(item, metadata)
                session.commit()
                counts["updated"] += 1

            if delay_seconds > 0:
                time.sleep(delay_seconds)

    return counts
