import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from app.services.sequence_dataset import (
    PAD_ITEM_ID,
    ItemEmbeddingTable,
    SampledWindowDataset,
    SequenceDataset,
    build_eval_samples,
    build_training_windows,
    build_user_sequences,
    lookup_input_embeddings,
    split_into_shards,
    split_user_ids,
    subsample_user_ids,
)


def test_build_user_sequences_filters_and_dedupes():
    embedded = {10, 11, 12, 13}
    interactions = [
        (1, 10, "rating", {"rating": 4.0}),
        (1, 10, "rating", {"rating": 5.0}),
        (1, 11, "rating", {"rating": 2.0}),
        (1, 12, "tag", {"tag": "funny"}),
        (1, 13, "rating", {"rating": 3.5}),
        (2, 10, "rating", {"rating": 5.0}),
        (2, 99, "rating", {"rating": 5.0}),
        (2, 11, "rating", {"rating": 4.0}),
    ]
    sequences = build_user_sequences(interactions, embedded, min_rating=3.5, min_interactions=3)
    assert sequences == {1: [10, 12, 13]}


def test_split_user_ids_are_disjoint_and_complete():
    user_ids = list(range(100))
    train, val, test = split_user_ids(user_ids, seed=0)
    assert len(train) == 80
    assert len(val) == 10
    assert len(test) == 10
    assert set(train) | set(val) | set(test) == set(user_ids)
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
    train_b, _, _ = split_user_ids(user_ids, seed=0)
    assert train == train_b


def test_subsample_user_ids_is_deterministic():
    chosen_a = subsample_user_ids(range(50), max_users=10, seed=7)
    chosen_b = subsample_user_ids(range(50), max_users=10, seed=7)
    assert chosen_a == chosen_b
    assert len(chosen_a) == 10


def test_training_windows_and_eval_holdout():
    sequences = {1: [1, 2, 3, 4, 5]}
    windows = build_training_windows(sequences, [1], max_seq_len=3)
    assert [(list(sample.input_item_ids), sample.target_item_id) for sample in windows] == [
        ([1], 2),
        ([1, 2], 3),
        ([1, 2, 3], 4),
        ([2, 3, 4], 5),
    ]
    eval_samples = build_eval_samples(sequences, [1], max_seq_len=3, holdout=1)
    assert len(eval_samples) == 1
    assert list(eval_samples[0].input_item_ids) == [2, 3, 4]
    assert eval_samples[0].target_item_id == 5


def test_dataset_padding_and_mask():
    sequences = {9: [4, 5, 6]}
    samples = build_training_windows(sequences, [9], max_seq_len=5)
    dataset = SequenceDataset(samples, max_seq_len=5)
    item = dataset[0]
    assert item["input_item_ids"].tolist() == [4, PAD_ITEM_ID, PAD_ITEM_ID, PAD_ITEM_ID, PAD_ITEM_ID]
    assert item["input_mask"].tolist() == [True, False, False, False, False]
    assert int(item["target_item_id"]) == 5


def test_embedding_table_lookup_and_padding():
    item_ids = np.array([10, 20, 30], dtype=np.int64)
    vectors = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    table = ItemEmbeddingTable(item_ids, vectors)
    ids = torch.tensor([[10, 20, 0], [30, 0, 0]])
    looked_up = table.lookup(ids)
    assert looked_up.shape == (2, 3, 3)
    assert torch.allclose(looked_up[0, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(looked_up[0, 2], torch.zeros(3))
    negatives = table.sample_negatives(5)
    assert negatives.shape == (5, 3)


def test_lookup_input_embeddings_applies_mask():
    item_ids = np.array([1, 2], dtype=np.int64)
    vectors = np.eye(2, 4, dtype=np.float32)
    table = ItemEmbeddingTable(item_ids, vectors)
    batch = {
        "input_item_ids": torch.tensor([[1, 2, 0]]),
        "input_mask": torch.tensor([[True, True, False]]),
    }
    embeddings = lookup_input_embeddings(batch, table)
    assert embeddings.shape[-1] == 4
    assert torch.allclose(embeddings[0, 2], torch.zeros(4))


def test_split_into_shards_covers_all_users():
    user_ids = list(range(10))
    shards = split_into_shards(user_ids, num_shards=4)
    assert len(shards) == 4
    assert sorted(user_id for shard in shards for user_id in shard) == user_ids


def test_split_train_users_into_batches_covers_all_users():
    from app.services.sequence_training import split_train_users_into_batches

    train_ids = list(range(100))
    batches = split_train_users_into_batches(train_ids, num_batches=5)
    assert len(batches) == 5
    assert sorted(user_id for batch in batches for user_id in batch) == train_ids


def test_sampled_window_dataset_is_reproducible():
    sequences = {1: [1, 2, 3, 4, 5], 2: [6, 7, 8, 9]}
    first = SampledWindowDataset(sequences, [1, 2], max_seq_len=3, windows_per_user=2, seed=11)
    second = SampledWindowDataset(sequences, [1, 2], max_seq_len=3, windows_per_user=2, seed=11)
    assert len(first) == len(second) == 4
    assert first[0]["target_item_id"] == second[0]["target_item_id"]
