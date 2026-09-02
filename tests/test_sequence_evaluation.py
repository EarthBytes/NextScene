
import numpy as np
import pytest

pytest.importorskip("torch")

from app.services.sequence_evaluation import (
    build_popularity_ranking,
    coverage,
    mrr,
    ndcg_at_k,
    prepare_eval_samples,
    recall_at_k,
    summarize_metrics,
)


def test_recall_mrr_and_ndcg():
    rankings = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    targets = [2, 4, 10]
    assert recall_at_k(rankings, targets, k=1) == pytest.approx(1 / 3)
    assert recall_at_k(rankings, targets, k=2) == pytest.approx(2 / 3)
    assert mrr(rankings, targets) == pytest.approx((0.5 + 1.0) / 3)
    assert ndcg_at_k(rankings, targets, k=3) == pytest.approx((1 / 1.585 + 1.0) / 3, rel=1e-2)


def test_coverage_counts_unique_recommendations():
    rankings = [[1, 2], [2, 3], [3, 4]]
    assert coverage(rankings, catalog_size=10) == 0.4


def test_prepare_eval_samples_respects_training_subset():
    sequences = {user_id: list(range(3, user_id + 8)) for user_id in range(1, 101)}
    config_data = {"users": 50, "seed": 7, "max_seq_len": 5}
    samples = prepare_eval_samples(sequences, config_data, split="test", holdout=1)
    assert len(samples) == 5
    assert all(len(sample.input_item_ids) <= 5 for sample in samples)


def test_build_popularity_ranking_prefers_frequent_items():
    sequences = {
        1: [10, 20, 20],
        2: [20, 30],
        3: [10, 10, 10],
    }
    item_ids = np.array([10, 20, 30, 40], dtype=np.int64)
    vectors = np.eye(4, dtype=np.float32)
    from app.services.sequence_dataset import ItemEmbeddingTable

    table = ItemEmbeddingTable(item_ids, vectors)
    ranking = build_popularity_ranking(sequences, table)
    assert ranking[:2] == [10, 20]


def test_summarize_metrics_keys():
    rankings = [[1, 2, 3], [4, 5, 6]]
    metrics = summarize_metrics(rankings, [2, 5], k_values=(10, 20), catalog_size=100)
    assert "recall@10" in metrics
    assert "recall@20" in metrics
    assert "mrr" in metrics
    assert "ndcg@10" in metrics
    assert "coverage" in metrics
