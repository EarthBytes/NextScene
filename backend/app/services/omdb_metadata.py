"""Fetch and store IMDb metadata from the OMDb API."""

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

OMDB_API_URL = "https://www.omdbapi.com/"


def parse_omdb_response(data: dict[str, Any]) -> dict[str, Any] | None:
    if data.get("Response") != "True":
        return None

    metadata: dict[str, Any] = {
        "year": clean_api_string(data.get("Year")),
        "rated": clean_api_string(data.get("Rated")),
        "released": clean_api_string(data.get("Released")),
        "runtime": clean_api_string(data.get("Runtime")),
        "director": clean_api_string(data.get("Director")),
        "writer": clean_api_string(data.get("Writer")),
        "actors": clean_api_string(data.get("Actors")),
        "language": clean_api_string(data.get("Language")),
        "country": clean_api_string(data.get("Country")),
        "awards": clean_api_string(data.get("Awards")),
        "imdb_rating": clean_api_string(data.get("imdbRating")),
        "imdb_votes": clean_api_string(data.get("imdbVotes")),
        "metascore": clean_api_string(data.get("Metascore")),
        "box_office": clean_api_string(data.get("BoxOffice")),
        "omdb_type": clean_api_string(data.get("Type")),
        "omdb_ratings": data.get("Ratings"),
    }

    genre_text = clean_api_string(data.get("Genre"))
    genres = None
    if genre_text:
        genres = [g.strip() for g in genre_text.split(",") if g.strip()]

    return {
        "description": clean_api_string(data.get("Plot")),
        "image_url": clean_api_string(data.get("Poster")),
        "genres": genres,
        "metadata_json": {k: v for k, v in metadata.items() if v is not None},
    }


def fetch_omdb_metadata(
    imdb_id: str,
    api_key: str,
    client: httpx.Client,
) -> dict[str, Any] | None:
    response = client.get(OMDB_API_URL, params={"i": imdb_id, "apikey": api_key})
    response.raise_for_status()
    return parse_omdb_response(response.json())


def apply_metadata_to_item(session: Session, item_id: int, metadata: dict[str, Any]) -> bool:
    item = session.get(Item, item_id)
    if item is None:
        return False

    apply_item_metadata(item, metadata, overwrite_genres=True)
    return True


def items_needing_metadata(
    session: Session,
    limit: int,
    force: bool = False,
) -> list[tuple[int, str]]:
    if force:
        query = """
            SELECT item_id, imdb_id
            FROM items
            WHERE imdb_id IS NOT NULL
            ORDER BY item_id
            LIMIT :limit
        """
    else:
        query = """
            SELECT item_id, imdb_id
            FROM items
            WHERE imdb_id IS NOT NULL
              AND (description IS NULL OR image_url IS NULL)
            ORDER BY item_id
            LIMIT :limit
        """

    rows = session.execute(text(query), {"limit": limit}).all()
    return [(int(row.item_id), str(row.imdb_id)) for row in rows]


def count_remaining(session: Session) -> int:
    return count_items_missing_metadata(
        session,
        "imdb_id IS NOT NULL AND (description IS NULL OR image_url IS NULL)",
    )


def run_metadata_fetch(
    session: Session,
    api_key: str,
    limit: int = 100,
    force: bool = False,
    delay_seconds: float = 0.25,
    max_retries: int = 3,
) -> dict[str, int]:
    items = items_needing_metadata(session, limit=limit, force=force)
    counts = {"queued": len(items), "updated": 0, "not_found": 0, "failed": 0}

    with httpx.Client(timeout=30.0) as client:
        for item_id, imdb_id in items:
            metadata: dict[str, Any] | None = None
            failed = False

            for attempt in range(max_retries):
                try:
                    metadata = fetch_omdb_metadata(imdb_id, api_key, client)
                    break
                except httpx.HTTPError:
                    if attempt + 1 == max_retries:
                        failed = True
                    else:
                        time.sleep(delay_seconds * (attempt + 1))

            if failed:
                counts["failed"] += 1
            elif metadata is None:
                counts["not_found"] += 1
            elif apply_metadata_to_item(session, item_id, metadata):
                session.commit()
                counts["updated"] += 1
            else:
                counts["failed"] += 1

            if delay_seconds > 0:
                time.sleep(delay_seconds)

    return counts
