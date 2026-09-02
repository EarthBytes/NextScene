"""Build and load Faiss indexes from item_embeddings."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from app.models.item_embedding import EMBEDDING_DIM
from sqlalchemy import text
from sqlalchemy.orm import Session

INDEX_FILENAME = "items.index"


def ids_path_for_index(index_path: Path) -> Path:
    return index_path.with_name(f"{index_path.stem}.ids.npy")


def meta_path_for_index(index_path: Path) -> Path:
    return index_path.with_name(f"{index_path.stem}.meta.json")


def parse_vector(value) -> list[float]:
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, np.ndarray):
        return value.astype(np.float32).tolist()
    return [float(component) for component in value]


def load_embeddings_from_db(session: Session) -> tuple[np.ndarray, np.ndarray]:
    rows = session.execute(
        text(
            """
            SELECT item_id, vector::text AS vector
            FROM item_embeddings
            WHERE vector IS NOT NULL
            ORDER BY item_id
            """
        )
    ).all()

    if not rows:
        raise ValueError("No embeddings found in item_embeddings")

    item_ids = np.array([int(row.item_id) for row in rows], dtype=np.int64)
    vectors = np.array([parse_vector(row.vector) for row in rows], dtype=np.float32)

    if vectors.ndim != 2 or vectors.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Expected vectors with shape (n, {EMBEDDING_DIM}), got {vectors.shape}"
        )

    return item_ids, vectors


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    import faiss

    normalized = vectors.astype(np.float32, copy=True)
    faiss.normalize_L2(normalized)
    return normalized


def build_faiss_index(vectors: np.ndarray):
    import app.ml_runtime  # noqa: F401
    import faiss

    if vectors.ndim != 2 or vectors.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Expected vectors with shape (n, {EMBEDDING_DIM}), got {vectors.shape}"
        )

    normalized = normalize_vectors(vectors)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(normalized)
    return index


def save_faiss_index(
    index,
    index_path: Path,
    item_ids: np.ndarray,
    metadata: dict | None = None,
) -> None:
    import faiss

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    np.save(ids_path_for_index(index_path), item_ids)

    meta = {
        "embedding_dim": EMBEDDING_DIM,
        "item_count": len(item_ids),
        "index_type": "IndexFlatIP",
        "built_at": datetime.now(UTC).isoformat(),
    }
    if metadata:
        meta.update(metadata)

    meta_path_for_index(index_path).write_text(json.dumps(meta, indent=2))


def load_faiss_index(index_path: Path):
    import app.ml_runtime  # noqa: F401
    import faiss

    ids_path = ids_path_for_index(index_path)
    if not index_path.is_file():
        raise FileNotFoundError(f"Faiss index not found: {index_path}")
    if not ids_path.is_file():
        raise FileNotFoundError(f"Faiss item id mapping not found: {ids_path}")

    index = faiss.read_index(str(index_path))
    item_ids = np.load(ids_path)
    return index, item_ids


def search_index(
    index,
    item_ids: np.ndarray,
    query_vector: np.ndarray,
    top_k: int = 10,
) -> list[tuple[int, float]]:
    import faiss

    query = np.ascontiguousarray(
        query_vector.astype(np.float32).reshape(1, -1),
        dtype=np.float32,
    )
    faiss.normalize_L2(query)
    scores, indices = index.search(query, top_k)

    results: list[tuple[int, float]] = []
    for position, score in zip(indices[0], scores[0]):
        if position < 0:
            continue
        results.append((int(item_ids[position]), float(score)))
    return results


def validate_index(index, item_ids: np.ndarray, vectors: np.ndarray, sample_size: int = 5) -> int:
    checks = min(sample_size, len(item_ids))
    mismatches = 0

    for offset in range(checks):
        expected_item_id = int(item_ids[offset])
        matches = search_index(index, item_ids, vectors[offset], top_k=1)
        if not matches or matches[0][0] != expected_item_id:
            mismatches += 1

    return mismatches


def run_faiss_index_build(
    session: Session,
    index_path: Path,
    validate: bool = True,
) -> dict[str, int | str]:
    item_ids, vectors = load_embeddings_from_db(session)
    index = build_faiss_index(vectors)
    save_faiss_index(index, index_path, item_ids)

    counts: dict[str, int | str] = {
        "items_indexed": len(item_ids),
        "embedding_dim": EMBEDDING_DIM,
        "index_path": str(index_path),
        "ids_path": str(ids_path_for_index(index_path)),
        "validation_mismatches": 0,
    }

    if validate:
        counts["validation_mismatches"] = validate_index(index, item_ids, vectors)

    return counts
