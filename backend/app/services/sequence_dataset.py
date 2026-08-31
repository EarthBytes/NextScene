"""Build padded user-history sequences for transformer training."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from sqlalchemy import text
from sqlalchemy.orm import Session
from torch.utils.data import DataLoader, Dataset

from app.models.item_embedding import EMBEDDING_DIM
from app.services.faiss_index import load_embeddings_from_db

PAD_ITEM_ID = 0
MIN_INTERACTIONS = 3
DEFAULT_MIN_RATING = 3.5
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1


@dataclass(frozen=True)
class SequenceSample:
    user_id: int
    input_item_ids: tuple[int, ...]
    target_item_id: int


class ItemEmbeddingTable:
    """In-memory item_id → CLIP vector lookup used during training."""

    def __init__(self, item_ids: np.ndarray, vectors: np.ndarray) -> None:
        if item_ids.shape[0] != vectors.shape[0]:
            raise ValueError("item_ids and vectors must have the same length")
        if vectors.ndim != 2:
            raise ValueError(f"Expected vectors [N, D], got {vectors.shape}")

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        vectors = (vectors / norms).astype(np.float32, copy=False)

        self.item_ids = torch.from_numpy(np.ascontiguousarray(item_ids, dtype=np.int64))
        self.vectors = torch.from_numpy(np.ascontiguousarray(vectors, dtype=np.float32))

        max_id = int(item_ids.max()) if len(item_ids) else 0
        index_map = torch.full((max_id + 1,), -1, dtype=torch.long)
        index_map[self.item_ids] = torch.arange(len(item_ids), dtype=torch.long)
        self.index_map = index_map

    def __len__(self) -> int:
        return int(self.vectors.size(0))

    @property
    def embedding_dim(self) -> int:
        return int(self.vectors.size(1))

    def to(self, device: torch.device | str) -> ItemEmbeddingTable:
        self.item_ids = self.item_ids.to(device)
        self.vectors = self.vectors.to(device)
        self.index_map = self.index_map.to(device)
        return self

    def indices_for(self, item_ids: torch.Tensor) -> torch.Tensor:
        flat = item_ids.reshape(-1)
        in_range = (flat >= 0) & (flat < self.index_map.numel())
        index = torch.full_like(flat, -1)
        if in_range.any():
            index[in_range] = self.index_map[flat[in_range]]
        return index.view(item_ids.shape)

    def lookup(self, item_ids: torch.Tensor) -> torch.Tensor:
        index = self.indices_for(item_ids)
        valid = index >= 0
        safe_index = index.clamp(min=0)
        vectors = self.vectors[safe_index]
        return vectors * valid.unsqueeze(-1).to(dtype=vectors.dtype)

    def sample_negatives(self, k: int) -> torch.Tensor:
        n = self.vectors.size(0)
        if n == 0:
            raise ValueError("Cannot sample negatives from an empty embedding table")
        index = torch.randint(0, n, (k,), device=self.vectors.device)
        return self.vectors[index]


class SequenceDataset(Dataset):
    def __init__(self, samples: Sequence[SequenceSample], max_seq_len: int) -> None:
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be >= 1")
        self.samples = list(samples)
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        history = sample.input_item_ids[-self.max_seq_len :]
        padded = torch.full((self.max_seq_len,), PAD_ITEM_ID, dtype=torch.long)
        mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        length = len(history)
        if length:
            padded[:length] = torch.tensor(history, dtype=torch.long)
            mask[:length] = True
        return {
            "input_item_ids": padded,
            "input_mask": mask,
            "target_item_id": torch.tensor(sample.target_item_id, dtype=torch.long),
        }


def _parse_context(context) -> dict:
    if context is None:
        return {}
    if isinstance(context, str):
        return json.loads(context)
    return dict(context)


def build_user_sequences(
    interactions: Iterable[tuple[int, int, str, dict | None]],
    embedded_item_ids: set[int],
    min_rating: float | None = DEFAULT_MIN_RATING,
    min_interactions: int = MIN_INTERACTIONS,
) -> dict[int, list[int]]:
    """Group ordered interactions into per-user item sequences.

    Interactions must already be sorted by (user_id, timestamp). Consecutive
    duplicate items are dropped. Rating events below ``min_rating`` are skipped
    when the threshold is set; tag/other events are kept.
    """
    sequences: dict[int, list[int]] = {}
    for user_id, item_id, interaction_type, context in interactions:
        if item_id not in embedded_item_ids:
            continue
        if min_rating is not None and interaction_type == "rating":
            parsed = _parse_context(context)
            rating = parsed.get("rating")
            if rating is None or float(rating) < min_rating:
                continue
        seq = sequences.setdefault(user_id, [])
        if seq and seq[-1] == item_id:
            continue
        seq.append(item_id)

    return {
        user_id: seq
        for user_id, seq in sequences.items()
        if len(seq) >= min_interactions
    }


def split_user_ids(
    user_ids: Sequence[int],
    seed: int = 42,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> tuple[list[int], list[int], list[int]]:
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(user_ids), dtype=np.int64)
    rng.shuffle(ids)
    n_train = int(len(ids) * train_ratio)
    n_val = int(len(ids) * val_ratio)
    train = ids[:n_train].tolist()
    val = ids[n_train : n_train + n_val].tolist()
    test = ids[n_train + n_val :].tolist()
    return train, val, test


def subsample_user_ids(user_ids: Sequence[int], max_users: int, seed: int = 42) -> list[int]:
    ids = sorted(user_ids)
    if max_users < 1 or len(ids) <= max_users:
        return ids
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.array(ids, dtype=np.int64), size=max_users, replace=False)
    return sorted(int(user_id) for user_id in chosen)


def build_training_windows(
    sequences: dict[int, list[int]],
    user_ids: Sequence[int],
    max_seq_len: int,
    min_history: int = 1,
) -> list[SequenceSample]:
    samples: list[SequenceSample] = []
    for user_id in user_ids:
        seq = sequences[user_id]
        for target_index in range(min_history, len(seq)):
            history = seq[max(0, target_index - max_seq_len) : target_index]
            samples.append(
                SequenceSample(
                    user_id=user_id,
                    input_item_ids=tuple(history),
                    target_item_id=seq[target_index],
                )
            )
    return samples


def build_eval_samples(
    sequences: dict[int, list[int]],
    user_ids: Sequence[int],
    max_seq_len: int,
    holdout: int = 1,
) -> list[SequenceSample]:
    if holdout < 1:
        raise ValueError("holdout must be >= 1")
    samples: list[SequenceSample] = []
    for user_id in user_ids:
        seq = sequences[user_id]
        if len(seq) < holdout + 1:
            continue
        history = seq[:-holdout][-max_seq_len:]
        if not history:
            continue
        samples.append(
            SequenceSample(
                user_id=user_id,
                input_item_ids=tuple(history),
                target_item_id=seq[-1],
            )
        )
    return samples


def lookup_input_embeddings(
    batch: dict[str, torch.Tensor],
    embedding_table: ItemEmbeddingTable,
) -> torch.Tensor:
    embeddings = embedding_table.lookup(batch["input_item_ids"])
    mask = batch["input_mask"].unsqueeze(-1).to(dtype=embeddings.dtype)
    return embeddings * mask


def create_dataloader(
    dataset: SequenceDataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )


def load_embedding_table(session: Session) -> ItemEmbeddingTable:
    item_ids, vectors = load_embeddings_from_db(session)
    if vectors.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"Expected {EMBEDDING_DIM}-dim embeddings, got {vectors.shape[1]}")
    return ItemEmbeddingTable(item_ids, vectors)


def iter_interaction_rows(session: Session) -> Iterable[tuple[int, int, str, dict]]:
    rows = session.execute(
        text(
            """
            SELECT user_id, item_id, type, context_json
            FROM interactions
            ORDER BY user_id, ts, interaction_id
            """
        )
    )
    for row in rows:
        yield int(row.user_id), int(row.item_id), str(row.type), _parse_context(row.context_json)


def load_user_sequences(
    session: Session,
    embedded_item_ids: set[int],
    min_rating: float | None = DEFAULT_MIN_RATING,
    min_interactions: int = MIN_INTERACTIONS,
) -> dict[int, list[int]]:
    return build_user_sequences(
        iter_interaction_rows(session),
        embedded_item_ids=embedded_item_ids,
        min_rating=min_rating,
        min_interactions=min_interactions,
    )
