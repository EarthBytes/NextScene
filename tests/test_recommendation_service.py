import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.services.catalog_search import CatalogSearcher, search_embedding_catalog
from app.services.recommendation_service import (
    RecommendationService,
    load_user_history,
    load_user_seen_items,
    model_version_from_config,
)
from app.services.sequence_dataset import ItemEmbeddingTable


def test_model_version_from_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "trained_at": "2026-09-01T15:40:51.956706+00:00",
                "best_val_recall_at_10": 0.0177,
            }
        )
    )
    version = model_version_from_config(tmp_path)
    assert "transformer" in version or tmp_path.name in version
    assert "0.0177" in version


def test_load_user_history_filters_low_ratings_and_dedupes():
    class Row:
        def __init__(self, item_id, type_, context_json):
            self.item_id = item_id
            self.type = type_
            self.context_json = context_json

    class FakeResult:
        def __init__(self, rows):
            self._rows = rows

        def __iter__(self):
            return iter(self._rows)

    class FakeSession:
        def execute(self, *_args, **_kwargs):
            return FakeResult(
                [
                    Row(10, "rating", {"rating": 4.0}),
                    Row(10, "rating", {"rating": 5.0}),
                    Row(11, "rating", {"rating": 2.0}),
                    Row(12, "tag", {"tag": "funny"}),
                ]
            )

    history = load_user_history(FakeSession(), user_id=1, min_rating=3.5)
    assert history == [10, 12]


def test_load_user_seen_items_returns_all_distinct_items():
    class Row:
        def __init__(self, item_id):
            self.item_id = item_id

    class FakeResult:
        def __init__(self, rows):
            self._rows = rows

        def __iter__(self):
            return iter(self._rows)

    class FakeSession:
        def execute(self, *_args, **_kwargs):
            return FakeResult([Row(904), Row(42), Row(904)])

    seen = load_user_seen_items(FakeSession(), user_id=500)
    assert seen == {904, 42}


def test_search_embedding_catalog_excludes_seen_items():
    item_ids = np.array([1, 2, 3, 4], dtype=np.int64)
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )
    table = ItemEmbeddingTable(item_ids=item_ids, vectors=vectors)
    query = np.array([1.0, 0.0], dtype=np.float32)

    top = search_embedding_catalog(table, query, top_k=2, exclude_item_ids={1, 2})
    assert [item_id for item_id, _score in top] == [3, 4]


def test_recommend_uses_candidate_pool_when_ranker_present():
    class HistoryRow:
        def __init__(self, item_id, type_, context_json):
            self.item_id = item_id
            self.type = type_
            self.context_json = context_json

    class SeenRow:
        def __init__(self, item_id):
            self.item_id = item_id

    class FakeResult:
        def __init__(self, rows):
            self._rows = rows

        def __iter__(self):
            return iter(self._rows)

        def all(self):
            return self._rows

    history_rows = [HistoryRow(item_id, "rating", {"rating": 4.0}) for item_id in range(1, 6)]
    seen_rows = [SeenRow(item_id) for item_id in range(1, 6)]

    class FakeSession:
        def execute(self, statement, *_args, **_kwargs):
            sql = str(statement)
            if "DISTINCT item_id" in sql:
                return FakeResult(seen_rows)
            if "items.item_id" in sql:
                return FakeResult([])
            return FakeResult(history_rows)

    item_ids = np.array([100, 101, 102, 103, 104], dtype=np.int64)
    vectors = np.eye(5, dtype=np.float32)
    table = ItemEmbeddingTable(item_ids=item_ids, vectors=vectors)
    inference = MagicMock()
    inference.max_seq_len = 50
    inference.predict_next_vector.return_value = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    ranker = MagicMock()
    ranker.candidate_pool_size = 3
    ranker.rerank.return_value = [(100, 0.95), (101, 0.90)]

    catalog_searcher = CatalogSearcher(embedding_table=table)

    service = RecommendationService(
        inference=inference,
        embedding_table=table,
        popularity_ranking=[100, 101],
        model_version="test",
        catalog_searcher=catalog_searcher,
        ranker=ranker,
        candidate_pool_size=3,
        min_interactions=3,
    )

    results, _timing = service.recommend(FakeSession(), user_id=500, k=2)
    ranker.rerank.assert_called_once()
    assert len(results) == 2


def test_recommend_excludes_items_outside_sequence_window():
    class HistoryRow:
        def __init__(self, item_id, type_, context_json):
            self.item_id = item_id
            self.type = type_
            self.context_json = context_json

    class SeenRow:
        def __init__(self, item_id):
            self.item_id = item_id

    class FakeResult:
        def __init__(self, rows):
            self._rows = rows

        def __iter__(self):
            return iter(self._rows)

        def all(self):
            return self._rows

    history_rows = [HistoryRow(item_id, "rating", {"rating": 4.0}) for item_id in range(1, 61)]
    seen_rows = [SeenRow(item_id) for item_id in [904, *range(1, 61)]]

    class FakeSession:
        def execute(self, statement, *_args, **_kwargs):
            sql = str(statement)
            if "DISTINCT item_id" in sql:
                return FakeResult(seen_rows)
            if "items.item_id" in sql:
                return FakeResult([])
            return FakeResult(history_rows)

    item_ids = np.array([904, 100, 101], dtype=np.int64)
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
        ],
        dtype=np.float32,
    )
    table = ItemEmbeddingTable(item_ids=item_ids, vectors=vectors)
    inference = MagicMock()
    inference.max_seq_len = 50
    inference.predict_next_vector.return_value = np.array([1.0, 0.0], dtype=np.float32)

    catalog_searcher = CatalogSearcher(embedding_table=table)

    service = RecommendationService(
        inference=inference,
        embedding_table=table,
        popularity_ranking=[904, 100],
        model_version="test",
        catalog_searcher=catalog_searcher,
        min_interactions=3,
    )

    results, _timing = service.recommend(FakeSession(), user_id=500, k=2)
    assert all(rec.item_id != 904 for rec in results)
    assert len(results) == 2
