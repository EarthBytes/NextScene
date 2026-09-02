"""LightGBM ranking layer: feature extraction and candidate re-ranking."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from app.models.item import Item
from app.models_ml.checkpoints import CONFIG_FILENAME, RANKER_MODEL_FILENAME
from app.services.embedding_table import ItemEmbeddingTable, as_numpy_array, item_row_index
from sqlalchemy import select
from sqlalchemy.orm import Session

FEATURE_NAMES: tuple[str, ...] = (
    "retrieval_score",
    "popularity_rank_norm",
    "history_length",
    "genre_overlap",
    "in_recent_history",
    "avg_history_similarity",
    "item_year_norm",
)

RECENT_HISTORY_WINDOW = 5
YEAR_MIN = 1970
YEAR_MAX = 2024


@dataclass(frozen=True)
class ItemFeatures:
    genres: tuple[str, ...]
    year: int | None


@dataclass(frozen=True)
class RankingModelConfig:
    feature_names: tuple[str, ...]
    trained_at: str
    candidate_pool_size: int
    train_samples: int
    val_auc: float | None = None
    model_version: str = "ranker-v1"

    def to_dict(self) -> dict:
        return {
            "feature_names": list(self.feature_names),
            "trained_at": self.trained_at,
            "candidate_pool_size": self.candidate_pool_size,
            "train_samples": self.train_samples,
            "val_auc": self.val_auc,
            "model_version": self.model_version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RankingModelConfig:
        return cls(
            feature_names=tuple(data.get("feature_names") or FEATURE_NAMES),
            trained_at=str(data["trained_at"]),
            candidate_pool_size=int(data["candidate_pool_size"]),
            train_samples=int(data.get("train_samples", 0)),
            val_auc=data.get("val_auc"),
            model_version=str(data.get("model_version", "ranker-v1")),
        )


def normalize_year(year: int | None) -> float:
    if year is None:
        return 0.5
    return float(np.clip((year - YEAR_MIN) / max(YEAR_MAX - YEAR_MIN, 1), 0.0, 1.0))


def popularity_rank_norm(item_id: int, popularity_ranking: list[int]) -> float:
    if not popularity_ranking:
        return 0.0
    try:
        index = popularity_ranking.index(item_id)
    except ValueError:
        return 0.0
    return 1.0 - (index / len(popularity_ranking))


def genre_overlap_ratio(
    candidate_genres: tuple[str, ...],
    history_genres: set[str],
) -> float:
    if not candidate_genres:
        return 0.0
    overlap = len(set(candidate_genres) & history_genres)
    return overlap / len(candidate_genres)


def avg_history_similarity(
    candidate_id: int,
    history: list[int],
    embedding_table: ItemEmbeddingTable,
) -> float:
    if not history:
        return 0.0
    candidate_idx = item_row_index(embedding_table, candidate_id)
    if candidate_idx < 0:
        return 0.0
    candidate_vec = as_numpy_array(embedding_table.vectors[candidate_idx])
    similarities: list[float] = []
    for item_id in history:
        hist_idx = item_row_index(embedding_table, item_id)
        if hist_idx < 0:
            continue
        hist_vec = as_numpy_array(embedding_table.vectors[hist_idx])
        similarities.append(float(np.dot(candidate_vec, hist_vec)))
    return float(np.mean(similarities)) if similarities else 0.0


def build_feature_row(
    *,
    candidate_id: int,
    retrieval_score: float,
    history: list[int],
    popularity_ranking: list[int],
    item_features: dict[int, ItemFeatures],
    embedding_table: ItemEmbeddingTable,
) -> list[float]:
    history_genres: set[str] = set()
    for item_id in history:
        features = item_features.get(item_id)
        if features:
            history_genres.update(features.genres)

    candidate = item_features.get(candidate_id, ItemFeatures((), None))
    recent = set(history[-RECENT_HISTORY_WINDOW:])

    return [
        float(retrieval_score),
        popularity_rank_norm(candidate_id, popularity_ranking),
        float(len(history)),
        genre_overlap_ratio(candidate.genres, history_genres),
        1.0 if candidate_id in recent else 0.0,
        avg_history_similarity(candidate_id, history, embedding_table),
        normalize_year(candidate.year),
    ]


def build_feature_matrix(
    *,
    candidate_ids: list[int],
    retrieval_scores: list[float],
    history: list[int],
    popularity_ranking: list[int],
    item_features: dict[int, ItemFeatures],
    embedding_table: ItemEmbeddingTable,
) -> np.ndarray:
    rows = [
        build_feature_row(
            candidate_id=candidate_id,
            retrieval_score=score,
            history=history,
            popularity_ranking=popularity_ranking,
            item_features=item_features,
            embedding_table=embedding_table,
        )
        for candidate_id, score in zip(candidate_ids, retrieval_scores, strict=True)
    ]
    return np.asarray(rows, dtype=np.float32)


def load_item_features(session: Session, item_ids: list[int]) -> dict[int, ItemFeatures]:
    if not item_ids:
        return {}
    rows = session.execute(
        select(Item.item_id, Item.genres, Item.metadata_json).where(Item.item_id.in_(item_ids))
    ).all()
    features: dict[int, ItemFeatures] = {}
    for row in rows:
        metadata = row.metadata_json or {}
        year = metadata.get("start_year") or metadata.get("year")
        if year is not None:
            try:
                year = int(year)
            except (TypeError, ValueError):
                year = None
        genres = tuple(row.genres or ())
        features[int(row.item_id)] = ItemFeatures(genres=genres, year=year)
    return features


class RankingModel:
    """LightGBM model that re-ranks retrieval candidates."""

    def __init__(
        self,
        booster,
        config: RankingModelConfig,
        *,
        feature_names: tuple[str, ...] = FEATURE_NAMES,
    ) -> None:
        self.booster = booster
        self.config = config
        self.feature_names = feature_names

    @property
    def model_version(self) -> str:
        return self.config.model_version

    @property
    def candidate_pool_size(self) -> int:
        return self.config.candidate_pool_size

    def predict_scores(self, feature_matrix: np.ndarray) -> np.ndarray:
        return self.booster.predict(feature_matrix)

    def rerank(
        self,
        session: Session,
        history: list[int],
        candidates: list[tuple[int, float]],
        popularity_ranking: list[int],
        embedding_table: ItemEmbeddingTable,
        top_k: int,
    ) -> list[tuple[int, float]]:
        if not candidates:
            return []
        item_ids = [item_id for item_id, _score in candidates]
        retrieval_scores = [score for _item_id, score in candidates]
        needed_ids = list(set(item_ids) | set(history))
        item_features = load_item_features(session, needed_ids)
        matrix = build_feature_matrix(
            candidate_ids=item_ids,
            retrieval_scores=retrieval_scores,
            history=history,
            popularity_ranking=popularity_ranking,
            item_features=item_features,
            embedding_table=embedding_table,
        )
        scores = self.predict_scores(matrix)
        ranked = sorted(
            zip(item_ids, scores, strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [(item_id, float(score)) for item_id, score in ranked[:top_k]]

    @classmethod
    def from_model_dir(cls, model_dir: Path) -> RankingModel:
        model_path = model_dir / RANKER_MODEL_FILENAME
        config_path = model_dir / CONFIG_FILENAME
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing ranking model: {model_path}")
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing ranking config: {config_path}")

        import lightgbm as lgb

        config = RankingModelConfig.from_dict(json.loads(config_path.read_text()))
        booster = lgb.Booster(model_file=str(model_path))
        return cls(booster, config, feature_names=config.feature_names)

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / CONFIG_FILENAME).write_text(json.dumps(self.config.to_dict(), indent=2) + "\n")
        self.booster.save_model(str(model_dir / RANKER_MODEL_FILENAME))


def try_load_ranking_model(model_dir: Path) -> RankingModel | None:
    model_path = model_dir / RANKER_MODEL_FILENAME
    if not model_path.is_file():
        return None
    return RankingModel.from_model_dir(model_dir)


def train_ranking_model(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    *,
    val_matrix: np.ndarray | None = None,
    val_labels: np.ndarray | None = None,
    candidate_pool_size: int = 50,
    train_samples: int = 0,
) -> RankingModel:
    import lightgbm as lgb

    train_set = lgb.Dataset(feature_matrix, label=labels, feature_name=list(FEATURE_NAMES))
    valid_sets = [train_set]
    valid_names = ["train"]
    if val_matrix is not None and val_labels is not None:
        valid_sets.append(lgb.Dataset(val_matrix, label=val_labels, feature_name=list(FEATURE_NAMES)))
        valid_names.append("val")

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=200,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=[lgb.early_stopping(20, verbose=False)],
    )

    val_auc = None
    if val_matrix is not None:
        val_auc = float(booster.best_score.get("val", {}).get("auc", 0.0))

    config = RankingModelConfig(
        feature_names=FEATURE_NAMES,
        trained_at=datetime.now(UTC).isoformat(),
        candidate_pool_size=candidate_pool_size,
        train_samples=train_samples,
        val_auc=val_auc,
    )
    return RankingModel(booster, config, feature_names=FEATURE_NAMES)


# --- A/B experiment assignment ---

EXPERIMENT_NAME = "generative_vs_popularity"


def assign_ab_variant(user_id: int, generative_fraction: float = 0.5) -> str:
    """Stable hash-based assignment: generative (transformer) or popularity."""
    digest = hashlib.sha256(f"{EXPERIMENT_NAME}:{user_id}".encode()).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "generative" if bucket < generative_fraction else "popularity"
