"""Small string parsing helpers."""

from __future__ import annotations


def parse_delimited_genres(
    value: str | None,
    delimiter: str,
    *,
    null_sentinels: frozenset[str] = frozenset(),
) -> list[str] | None:
    if not value or value in null_sentinels:
        return None
    genres = [part.strip() for part in value.split(delimiter) if part.strip()]
    return genres or None
