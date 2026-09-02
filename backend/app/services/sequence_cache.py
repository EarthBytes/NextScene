"""Persist and load per-user interaction sequences for fast training startup."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from app.services.sequence_dataset import (
    DEFAULT_MIN_RATING,
    MIN_INTERACTIONS,
    build_user_sequences,
    iter_interaction_rows,
    load_embedded_item_ids,
)
from sqlalchemy.orm import Session

META_FILENAME = "meta.json"
SEQUENCES_FILENAME = "sequences.npz"


def cache_paths(cache_dir: Path) -> tuple[Path, Path]:
    return cache_dir / META_FILENAME, cache_dir / SEQUENCES_FILENAME


def cache_meta_matches(
    meta_path: Path,
    *,
    min_rating: float | None,
    min_interactions: int,
) -> bool:
    if not meta_path.is_file():
        return False
    meta = json.loads(meta_path.read_text())
    return (
        meta.get("min_rating") == min_rating
        and meta.get("min_interactions") == min_interactions
        and meta.get("version") == 1
    )


def sequences_to_arrays(sequences: dict[int, list[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    user_ids = np.array(sorted(sequences), dtype=np.int64)
    offsets = np.zeros(len(user_ids) + 1, dtype=np.int64)
    items: list[int] = []
    for index, user_id in enumerate(user_ids):
        seq = sequences[int(user_id)]
        items.extend(seq)
        offsets[index + 1] = len(items)
    return user_ids, offsets, np.array(items, dtype=np.int32)


def sequences_from_arrays(
    user_ids: np.ndarray,
    offsets: np.ndarray,
    items: np.ndarray,
) -> dict[int, list[int]]:
    sequences: dict[int, list[int]] = {}
    for index, user_id in enumerate(user_ids):
        start = int(offsets[index])
        end = int(offsets[index + 1])
        sequences[int(user_id)] = [int(item_id) for item_id in items[start:end]]
    return sequences


def save_sequence_cache(
    cache_dir: Path,
    sequences: dict[int, list[int]],
    *,
    min_rating: float | None,
    min_interactions: int,
    embedded_item_count: int,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path, npz_path = cache_paths(cache_dir)
    user_ids, offsets, items = sequences_to_arrays(sequences)
    np.savez_compressed(npz_path, user_ids=user_ids, offsets=offsets, items=items)
    meta = {
        "version": 1,
        "built_at": datetime.now(UTC).isoformat(),
        "min_rating": min_rating,
        "min_interactions": min_interactions,
        "user_count": len(sequences),
        "interaction_count": int(items.size),
        "embedded_item_count": embedded_item_count,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return npz_path


def load_sequence_cache(cache_dir: Path) -> dict[int, list[int]]:
    _, npz_path = cache_paths(cache_dir)
    if not npz_path.is_file():
        raise FileNotFoundError(f"Sequence cache not found: {npz_path}")
    data = np.load(npz_path)
    return sequences_from_arrays(data["user_ids"], data["offsets"], data["items"])


def build_sequence_cache_from_db(
    session: Session,
    cache_dir: Path,
    *,
    min_rating: float | None = DEFAULT_MIN_RATING,
    min_interactions: int = MIN_INTERACTIONS,
) -> dict[int, list[int]]:
    embedded_ids = load_embedded_item_ids(session)
    sequences = build_user_sequences(
        iter_interaction_rows(session),
        embedded_item_ids=embedded_ids,
        min_rating=min_rating,
        min_interactions=min_interactions,
    )
    save_sequence_cache(
        cache_dir,
        sequences,
        min_rating=min_rating,
        min_interactions=min_interactions,
        embedded_item_count=len(embedded_ids),
    )
    return sequences


def load_or_build_sequences(
    session: Session,
    cache_dir: Path,
    *,
    min_rating: float | None = DEFAULT_MIN_RATING,
    min_interactions: int = MIN_INTERACTIONS,
    rebuild: bool = False,
) -> tuple[dict[int, list[int]], str]:
    meta_path, npz_path = cache_paths(cache_dir)
    if (
        not rebuild
        and npz_path.is_file()
        and cache_meta_matches(meta_path, min_rating=min_rating, min_interactions=min_interactions)
    ):
        return load_sequence_cache(cache_dir), "cache"

    sequences = build_sequence_cache_from_db(
        session,
        cache_dir,
        min_rating=min_rating,
        min_interactions=min_interactions,
    )
    return sequences, "database"
