"""Fetch and store IMDb metadata from the OMDb API."""

from __future__ import annotations

import time
from typing import Any

import httpx
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.models.item import Item

OMDB_API_URL = "https://www.omdbapi.com/"


def _clean(value: str | None) -> str | None:
    if value is None or value == "N/A":
        return None
    stripped = value.strip()
    return stripped or None


def parse_omdb_response(data: dict[str, Any]) -> dict[str, Any] | None:
    if data.get("Response") != "True":
        return None

    metadata: dict[str, Any] = {
        "year": _clean(data.get("Year")),
        "rated": _clean(data.get("Rated")),
        "released": _clean(data.get("Released")),
        "runtime": _clean(data.get("Runtime")),
        "director": _clean(data.get("Director")),
        "writer": _clean(data.get("Writer")),
        "actors": _clean(data.get("Actors")),
        "language": _clean(data.get("Language")),
        "country": _clean(data.get("Country")),
        "awards": _clean(data.get("Awards")),
        "imdb_rating": _clean(data.get("imdbRating")),
        "imdb_votes": _clean(data.get("imdbVotes")),
        "metascore": _clean(data.get("Metascore")),
        "box_office": _clean(data.get("BoxOffice")),
        "omdb_type": _clean(data.get("Type")),
        "omdb_ratings": data.get("Ratings"),
    }

    genre_text = _clean(data.get("Genre"))
    genres = None
    if genre_text:
        genres = [g.strip() for g in genre_text.split(",") if g.strip()]

    return {
        "description": _clean(data.get("Plot")),
        "image_url": _clean(data.get("Poster")),
        "genres": genres,
        "metadata_json": {k: v for k, v in metadata.items() if v is not None},
    }


def fetch_omdb_metadata(
    imdb_id: str,
    api_key: str,
    client: httpx.Client | None = None,
) -> dict[str, Any] | None:
    owns_client = client is None
    if owns_client:
        client = httpx.Client(timeout=30.0)

    try:
        response = client.get(OMDB_API_URL, params={"i": imdb_id, "apikey": api_key})
        response.raise_for_status()
        return parse_omdb_response(response.json())
    finally:
        if owns_client:
            client.close()


def apply_metadata_to_item(session: Session, item_id: int, metadata: dict[str, Any]) -> bool:
    item = session.get(Item, item_id)
    if item is None:
        return False

    if metadata.get("description"):
        item.description = metadata["description"]
    if metadata.get("image_url"):
        item.image_url = metadata["image_url"]
    if metadata.get("genres"):
        item.genres = metadata["genres"]

    patch = metadata.get("metadata_json", {})
    if patch:
        current = dict(item.metadata_json or {})
        current.update(patch)
        item.metadata_json = current

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
                    response = client.get(
                        OMDB_API_URL,
                        params={"i": imdb_id, "apikey": api_key},
                    )
                    response.raise_for_status()
                    metadata = parse_omdb_response(response.json())
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
