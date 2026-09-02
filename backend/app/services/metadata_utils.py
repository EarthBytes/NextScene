"""Shared helpers for external metadata enrichment."""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.orm import Session


def clean_api_string(value: str | None, *, null_values: frozenset[str] = frozenset({"N/A"})) -> str | None:
    if value is None:
        return None
    stripped = str(value).strip()
    if not stripped or stripped in null_values:
        return None
    return stripped


def merge_metadata_json(item, patch: dict) -> None:
    if not patch:
        return
    current = dict(item.metadata_json or {})
    current.update(patch)
    item.metadata_json = current


def apply_item_metadata(
    item,
    metadata: dict,
    *,
    overwrite_genres: bool = False,
) -> None:
    if metadata.get("description"):
        item.description = metadata["description"]
    if metadata.get("image_url"):
        item.image_url = metadata["image_url"]
    genres = metadata.get("genres")
    if genres and (overwrite_genres or not item.genres):
        item.genres = genres
    merge_metadata_json(item, metadata.get("metadata_json", {}))


def count_items_missing_metadata(session: Session, where_clause: str) -> int:
    return session.execute(
        text(
            f"""
            SELECT COUNT(*)
            FROM items
            WHERE {where_clause}
            """
        )
    ).scalar_one()
