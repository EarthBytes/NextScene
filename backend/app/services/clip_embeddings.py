"""Generate fused CLIP text+image embeddings for items."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.models.item import Item
from app.models.item_embedding import EMBEDDING_DIM, ItemEmbedding
from app.services.poster_download import existing_poster_item_ids, find_existing_poster

DEFAULT_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_BATCH_SIZE = 128
DEFAULT_IMAGE_WORKERS = 8


@dataclass(frozen=True)
class ItemEmbeddingInput:
    item_id: int
    text: str
    poster_path: Path | None


def resolve_device(device: str | None = None) -> str:
    import torch

    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_batch_size(device: str) -> int:
    return 128 if device in ("cuda", "mps") else 64


def build_item_text(item: Item) -> str:
    parts: list[str] = []

    if item.title:
        parts.append(item.title.strip())

    if item.description:
        parts.append(item.description.strip())

    if item.genres:
        parts.append(", ".join(genre.strip() for genre in item.genres if genre.strip()))

    metadata = item.metadata_json or {}
    for key in ("actors", "director", "cast"):
        value = metadata.get(key)
        if value:
            parts.append(str(value).strip())

    if parts:
        return ". ".join(parts)

    return item.title.strip() or f"item {item.item_id}"


def build_poster_index(posters_dir: Path) -> dict[int, Path]:
    index: dict[int, Path] = {}
    if not posters_dir.is_dir():
        return index

    for item_id in existing_poster_item_ids(posters_dir):
        poster_path = find_existing_poster(posters_dir, item_id)
        if poster_path is not None:
            index[item_id] = poster_path
    return index


def resolve_poster_path(
    item: Item,
    posters_dir: Path,
    poster_index: dict[int, Path] | None = None,
) -> Path | None:
    metadata_path = (item.metadata_json or {}).get("poster_path")
    if metadata_path:
        path = Path(metadata_path)
        if path.is_file() and path.stat().st_size > 0:
            return path

    if poster_index is not None:
        return poster_index.get(item.item_id)

    return find_existing_poster(posters_dir, item.item_id)


def fuse_embeddings(
    text_vector: np.ndarray,
    image_vector: np.ndarray | None = None,
) -> list[float]:
    if image_vector is not None:
        fused = text_vector + image_vector
    else:
        fused = text_vector

    norm = float(np.linalg.norm(fused))
    if norm > 0:
        fused = fused / norm

    vector = fused.astype(np.float32)
    if vector.shape[0] != EMBEDDING_DIM:
        raise ValueError(f"Expected {EMBEDDING_DIM}-dim vector, got {vector.shape[0]}")

    return vector.tolist()


def items_needing_embeddings(
    session: Session,
    limit: int,
    force: bool = False,
) -> list[int]:
    if force:
        query = """
            SELECT item_id
            FROM items
            ORDER BY item_id
            LIMIT :limit
        """
    else:
        query = """
            SELECT i.item_id
            FROM items i
            LEFT JOIN item_embeddings e ON e.item_id = i.item_id
            WHERE e.item_id IS NULL OR e.vector IS NULL
            ORDER BY i.item_id
            LIMIT :limit
        """

    rows = session.execute(text(query), {"limit": limit}).all()
    return [int(row.item_id) for row in rows]


def count_remaining_embeddings(session: Session) -> int:
    return session.execute(
        text(
            """
            SELECT COUNT(*)
            FROM items i
            LEFT JOIN item_embeddings e ON e.item_id = i.item_id
            WHERE e.item_id IS NULL OR e.vector IS NULL
            """
        )
    ).scalar_one()


def _load_items(session: Session, item_ids: list[int]) -> list[Item]:
    if not item_ids:
        return []

    items = session.scalars(select(Item).where(Item.item_id.in_(item_ids))).all()
    by_id = {item.item_id: item for item in items}
    return [by_id[item_id] for item_id in item_ids if item_id in by_id]


def _prepare_batch(
    items: list[Item],
    posters_dir: Path,
    poster_index: dict[int, Path],
) -> list[ItemEmbeddingInput]:
    return [
        ItemEmbeddingInput(
            item_id=item.item_id,
            text=build_item_text(item),
            poster_path=resolve_poster_path(item, posters_dir, poster_index),
        )
        for item in items
    ]


def _extract_features(output):
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    return output


def _load_clip(model_name: str, device: str):
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, processor, torch


def _load_image(path: Path):
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB")


def _encode_batch(
    batch: list[ItemEmbeddingInput],
    model,
    processor,
    device: str,
    image_workers: int,
) -> list[tuple[int, list[float]]]:
    import torch

    texts = [entry.text for entry in batch]
    text_inputs = processor(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    text_inputs = {
        key: value.to(device, non_blocking=True) for key, value in text_inputs.items()
    }

    use_amp = device in ("cuda", "mps")
    with torch.inference_mode():
        with torch.autocast(device_type=device, enabled=use_amp):
            text_features = _extract_features(model.get_text_features(**text_inputs))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_entries = [
            (index, entry.poster_path)
            for index, entry in enumerate(batch)
            if entry.poster_path is not None
        ]

        if image_entries:
            paths = [path for _, path in image_entries]
            with ThreadPoolExecutor(max_workers=image_workers) as executor:
                images = list(executor.map(_load_image, paths))

            image_inputs = processor(images=images, return_tensors="pt")
            image_inputs = {
                key: value.to(device, non_blocking=True)
                for key, value in image_inputs.items()
            }

            with torch.autocast(device_type=device, enabled=use_amp):
                image_features = _extract_features(model.get_image_features(**image_inputs))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            has_image = torch.zeros(len(batch), dtype=torch.bool, device=device)
            image_features_full = torch.zeros_like(text_features)
            for (index, _), vector in zip(image_entries, image_features):
                has_image[index] = True
                image_features_full[index] = vector

            fused = text_features.clone()
            fused[has_image] = text_features[has_image] + image_features_full[has_image]
            fused = fused / fused.norm(dim=-1, keepdim=True)
        else:
            fused = text_features

    vectors = fused.float().cpu().numpy()
    return [(entry.item_id, vectors[index].tolist()) for index, entry in enumerate(batch)]


def upsert_embeddings(session: Session, embeddings: list[tuple[int, list[float]]]) -> None:
    if not embeddings:
        return

    rows = [{"item_id": item_id, "vector": vector} for item_id, vector in embeddings]
    stmt = insert(ItemEmbedding).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=[ItemEmbedding.item_id],
        set_={"vector": stmt.excluded.vector},
    )
    session.execute(stmt)
    session.commit()


def run_clip_embedding_generation(
    session: Session,
    posters_dir: Path,
    model_name: str = DEFAULT_MODEL,
    batch_size: int | None = None,
    limit: int | None = None,
    force: bool = False,
    device: str | None = None,
    image_workers: int = DEFAULT_IMAGE_WORKERS,
    show_progress: bool = True,
) -> dict[str, int | str]:
    import torch

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    fetch_limit = limit if limit is not None else 10_000_000
    item_ids = items_needing_embeddings(session, limit=fetch_limit, force=force)
    resolved_device = resolve_device(device)
    effective_batch_size = batch_size or default_batch_size(resolved_device)
    poster_index = build_poster_index(posters_dir)

    counts: dict[str, int | str] = {
        "queued": len(item_ids),
        "embedded": 0,
        "text_only": 0,
        "text_and_image": 0,
        "failed": 0,
        "device": resolved_device,
        "batch_size": effective_batch_size,
    }

    if not item_ids:
        return counts

    model, processor, _torch = _load_clip(model_name, resolved_device)
    batch_ranges = range(0, len(item_ids), effective_batch_size)
    logged_error = False
    if show_progress and tqdm is not None:
        batch_ranges = tqdm(
            batch_ranges,
            total=(len(item_ids) + effective_batch_size - 1) // effective_batch_size,
            desc="Embedding batches",
            unit="batch",
        )

    try:
        for offset in batch_ranges:
            batch_ids = item_ids[offset : offset + effective_batch_size]
            items = _load_items(session, batch_ids)
            batch_inputs = _prepare_batch(items, posters_dir, poster_index)

            try:
                embeddings = _encode_batch(
                    batch_inputs,
                    model,
                    processor,
                    resolved_device,
                    image_workers=image_workers,
                )
                upsert_embeddings(session, embeddings)
                counts["embedded"] += len(embeddings)

                for entry in batch_inputs:
                    if entry.poster_path is None:
                        counts["text_only"] += 1
                    else:
                        counts["text_and_image"] += 1
            except Exception as exc:
                session.rollback()
                counts["failed"] += len(batch_inputs)
                if not logged_error:
                    print(f"Batch failed: {exc}")
                    logged_error = True
    finally:
        del model, processor
        if resolved_device == "cuda":
            torch.cuda.empty_cache()

    return counts
