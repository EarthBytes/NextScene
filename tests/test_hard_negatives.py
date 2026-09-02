import numpy as np
import pytest

pytest.importorskip("torch")

import torch
from app.ml_runtime import resolve_num_workers
from app.services.hard_negatives import NegativeSampler
from app.services.sequence_dataset import ItemEmbeddingTable


def test_negative_sampler_returns_shared_random_negatives():
    item_ids = np.arange(1, 11, dtype=np.int64)
    vectors = np.eye(10, 8, dtype=np.float32)
    table = ItemEmbeddingTable(item_ids, vectors)
    sampler = NegativeSampler(table, random_negatives=6, hard_negatives=0)
    sampled = sampler.sample()
    assert sampled.shared is not None
    assert sampled.shared.shape == (6, 8)
    assert sampled.per_sample is None


def test_negative_sampler_hard_negatives_are_per_sample():
    item_ids = np.arange(1, 6, dtype=np.int64)
    vectors = np.eye(5, 4, dtype=np.float32)
    table = ItemEmbeddingTable(item_ids, vectors)
    sampler = NegativeSampler(table, random_negatives=0, hard_negatives=2)
    query = table.lookup(torch.tensor([1, 2]))
    sampled = sampler.sample(query_vectors=query, exclude_item_ids=torch.tensor([1, 2]))
    assert sampled.per_sample is not None
    assert sampled.per_sample.shape == (2, 2, 4)


def test_resolve_num_workers_zero_on_mps():
    assert resolve_num_workers("mps", 4) == 0
