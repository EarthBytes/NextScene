import json
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("torch")

from app.services.ranking_service import (
    FEATURE_NAMES,
    assign_ab_variant,
    build_feature_matrix,
    genre_overlap_ratio,
    normalize_year,
    popularity_rank_norm,
)
from app.services.sequence_dataset import ItemEmbeddingTable


def test_popularity_rank_norm():
    ranking = [100, 200, 300]
    assert popularity_rank_norm(100, ranking) == pytest.approx(1.0)
    assert popularity_rank_norm(200, ranking) == pytest.approx(2 / 3, rel=1e-3)
    assert popularity_rank_norm(999, ranking) == 0.0


def test_genre_overlap_ratio():
    assert genre_overlap_ratio(("Action", "Comedy"), {"Action", "Drama"}) == pytest.approx(0.5)
    assert genre_overlap_ratio((), {"Action"}) == 0.0


def test_normalize_year():
    assert normalize_year(1970) == pytest.approx(0.0)
    assert normalize_year(2024) == pytest.approx(1.0)
    assert normalize_year(None) == pytest.approx(0.5)


def test_build_feature_matrix_shape():
    item_ids = np.array([1, 2, 3], dtype=np.int64)
    vectors = np.eye(3, dtype=np.float32)
    table = ItemEmbeddingTable(item_ids=item_ids, vectors=vectors)
    item_features = {
        1: type("F", (), {"genres": ("Action",), "year": 2000})(),
        2: type("F", (), {"genres": ("Comedy",), "year": 2010})(),
        3: type("F", (), {"genres": ("Drama",), "year": 1990})(),
    }
    matrix = build_feature_matrix(
        candidate_ids=[2, 3],
        retrieval_scores=[0.9, 0.8],
        history=[1],
        popularity_ranking=[2, 1, 3],
        item_features=item_features,
        embedding_table=table,
    )
    assert matrix.shape == (2, len(FEATURE_NAMES))


def test_assign_ab_variant_is_stable():
    first = assign_ab_variant(42, generative_fraction=0.5)
    second = assign_ab_variant(42, generative_fraction=0.5)
    assert first == second
    assert first in {"generative", "popularity"}


def test_assign_ab_variant_respects_fraction():
    variants = {assign_ab_variant(user_id, generative_fraction=0.0) for user_id in range(100)}
    assert variants == {"popularity"}
    variants = {assign_ab_variant(user_id, generative_fraction=1.0) for user_id in range(100)}
    assert variants == {"generative"}


def test_ranking_model_rerank_orders_by_score():
    from app.services.ranking_service import RankingModel, RankingModelConfig

    booster = MagicMock()
    booster.predict.return_value = np.array([0.2, 0.9, 0.5], dtype=np.float32)

    config = RankingModelConfig(
        feature_names=FEATURE_NAMES,
        trained_at="2026-09-02T00:00:00+00:00",
        candidate_pool_size=3,
        train_samples=3,
    )
    ranker = RankingModel(booster, config)

    item_ids = np.array([10, 20, 30], dtype=np.int64)
    vectors = np.eye(3, dtype=np.float32)
    table = ItemEmbeddingTable(item_ids=item_ids, vectors=vectors)

    class Row:
        def __init__(self, item_id, genres, metadata_json):
            self.item_id = item_id
            self.genres = genres
            self.metadata_json = metadata_json

    class FakeResult:
        def __init__(self, rows):
            self._rows = rows

        def all(self):
            return self._rows

    class FakeSession:
        def execute(self, *_args, **_kwargs):
            return FakeResult(
                [
                    Row(10, ["Action"], {"start_year": 2000}),
                    Row(20, ["Comedy"], {"start_year": 2010}),
                    Row(30, ["Drama"], {"start_year": 1990}),
                ]
            )

    reranked = ranker.rerank(
        FakeSession(),
        history=[10],
        candidates=[(20, 0.8), (30, 0.7), (10, 0.9)],
        popularity_ranking=[20, 10, 30],
        embedding_table=table,
        top_k=2,
    )
    assert len(reranked) == 2
    assert reranked[0][0] == 30
    assert reranked[0][1] >= reranked[1][1]


def test_try_load_ranking_model_missing(tmp_path):
    from app.services.ranking_service import try_load_ranking_model

    assert try_load_ranking_model(tmp_path) is None


def test_ranking_model_save_writes_artifacts(tmp_path):
    from app.services.ranking_service import RankingModel, RankingModelConfig

    booster = MagicMock()
    booster.save_model = MagicMock()

    config = RankingModelConfig(
        feature_names=FEATURE_NAMES,
        trained_at="2026-09-02T00:00:00+00:00",
        candidate_pool_size=50,
        train_samples=1,
        model_version="ranker-test",
    )
    ranker = RankingModel(booster, config)
    ranker.save(tmp_path)

    booster.save_model.assert_called_once_with(str(tmp_path / "model.txt"))
    config_data = json.loads((tmp_path / "config.json").read_text())
    assert config_data["model_version"] == "ranker-test"
