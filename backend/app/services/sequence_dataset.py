"""Build padded user-history sequences for transformer training."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np
import torch
from app.services.embedding_table import (
    DEFAULT_MIN_RATING,
    MIN_INTERACTIONS,
    PAD_ITEM_ID,
    TRAIN_RATIO,
    VAL_RATIO,
    SequenceSample,
    build_interaction_history,
    build_user_sequences,
    iter_interaction_rows,
    load_embedded_item_ids,
    load_embedding_table as load_numpy_embedding_table,
    load_user_sequences,
    parse_context,
)
from sqlalchemy import text
from torch.utils.data import DataLoader, Dataset

__all__ = [
    "DEFAULT_MIN_RATING",
    "MIN_INTERACTIONS",
    "PAD_ITEM_ID",
    "TRAIN_RATIO",
    "VAL_RATIO",
    "ItemEmbeddingTable",
    "SequenceSample",
    "SequenceDataset",
    "SampledWindowDataset",
    "build_eval_samples",
    "build_interaction_history",
    "build_training_windows",
    "build_user_sequences",
    "create_dataloader",
    "iter_interaction_rows",
    "load_embedded_item_ids",
    "load_embedding_table",
    "load_sequences_for_user_subset",
    "load_user_sequences",
    "lookup_input_embeddings",
    "parse_context",
    "split_into_shards",
    "split_user_ids",
    "subsample_user_ids",
]


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


def _sample_to_batch_tensors(
    history: Sequence[int],
    target_item_id: int,
    max_seq_len: int,
) -> dict[str, torch.Tensor]:
    clipped = list(history[-max_seq_len:])
    padded = torch.full((max_seq_len,), PAD_ITEM_ID, dtype=torch.long)
    mask = torch.zeros(max_seq_len, dtype=torch.bool)
    length = len(clipped)
    if length:
        padded[:length] = torch.tensor(clipped, dtype=torch.long)
        mask[:length] = True
    return {
        "input_item_ids": padded,
        "input_mask": mask,
        "target_item_id": torch.tensor(target_item_id, dtype=torch.long),
    }


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
        return _sample_to_batch_tensors(sample.input_item_ids, sample.target_item_id, self.max_seq_len)


class SampledWindowDataset(Dataset):
    """Random sliding-window samples per user, regenerated each epoch/shard."""

    def __init__(
        self,
        sequences: dict[int, list[int]],
        user_ids: Sequence[int],
        max_seq_len: int,
        windows_per_user: int,
        seed: int,
        min_history: int = 1,
    ) -> None:
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be >= 1")
        if windows_per_user < 1:
            raise ValueError("windows_per_user must be >= 1")

        self.sequences = sequences
        self.user_ids = list(user_ids)
        self.max_seq_len = max_seq_len
        self.windows_per_user = windows_per_user
        self.min_history = min_history
        self.samples = self._build_samples(seed)

    def _build_samples(self, seed: int) -> list[tuple[int, int]]:
        rng = np.random.default_rng(seed)
        samples: list[tuple[int, int]] = []
        for user_id in self.user_ids:
            seq = self.sequences.get(user_id)
            if not seq or len(seq) <= self.min_history:
                continue
            valid_targets = list(range(self.min_history, len(seq)))
            if len(valid_targets) <= self.windows_per_user:
                chosen = valid_targets
            else:
                chosen = rng.choice(valid_targets, size=self.windows_per_user, replace=False).tolist()
            samples.extend((user_id, target_index) for target_index in chosen)
        rng.shuffle(samples)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        user_id, target_index = self.samples[index]
        seq = self.sequences[user_id]
        history = seq[max(0, target_index - self.max_seq_len) : target_index]
        return _sample_to_batch_tensors(history, seq[target_index], self.max_seq_len)


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


def split_into_shards(user_ids: Sequence[int], num_shards: int) -> list[list[int]]:
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    ids = list(user_ids)
    if not ids:
        return []
    shard_count = min(num_shards, len(ids))
    shard_size = math.ceil(len(ids) / shard_count)
    return [ids[index : index + shard_size] for index in range(0, len(ids), shard_size)]


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


def load_embedding_table(session):
    numpy_table = load_numpy_embedding_table(session)
    return ItemEmbeddingTable(numpy_table.item_ids, numpy_table.vectors)



def load_sequences_for_user_subset(
    session,
    embedded_item_ids: set[int],
    max_users: int,
    seed: int,
    min_rating: float | None = DEFAULT_MIN_RATING,
    min_interactions: int = MIN_INTERACTIONS,
    batch_size: int = 5000,
) -> dict[int, list[int]]:
    """Load sequences for a random user subset without scanning the full table."""
    candidate_ids = [
        int(row.user_id)
        for row in session.execute(text("SELECT DISTINCT user_id FROM interactions ORDER BY user_id"))
    ]
    rng = np.random.default_rng(seed)
    rng.shuffle(candidate_ids)

    sequences: dict[int, list[int]] = {}
    for start in range(0, len(candidate_ids), batch_size):
        batch = candidate_ids[start : start + batch_size]
        batch_sequences = load_user_sequences(
            session,
            embedded_item_ids=embedded_item_ids,
            min_rating=min_rating,
            min_interactions=min_interactions,
            user_ids=batch,
        )
        sequences.update(batch_sequences)
        if len(sequences) >= max_users:
            break

    if len(sequences) < max_users:
        return sequences

    chosen = subsample_user_ids(list(sequences), max_users, seed=seed)
    return {user_id: sequences[user_id] for user_id in chosen}
