"""Offline evaluation for the sequence transformer and baselines."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sqlalchemy.orm import Session

from app.ml_runtime import resolve_device, use_amp
from app.models_ml.checkpoints import BEST_FILENAME, CONFIG_FILENAME
from app.models_ml.sequence_transformer import SequenceTransformer, SequenceTransformerConfig
from app.services.sequence_dataset import (
    DEFAULT_MIN_RATING,
    ItemEmbeddingTable,
    SequenceDataset,
    SequenceSample,
    build_eval_samples,
    create_dataloader,
    load_embedding_table,
    lookup_input_embeddings,
    split_user_ids,
    subsample_user_ids,
)
from app.services.sequence_training import TrainingConfig


@dataclass
class EvalConfig:
    checkpoint_path: Path
    sequences_cache_dir: Path
    split: str = "test"
    k_values: tuple[int, ...] = (10, 20, 50)
    batch_size: int = 256
    seed: int = 42
    holdout: int = 1
    max_users: int | None = None
    device: str | None = None


def load_checkpoint_config(checkpoint_path: Path) -> dict:
    config_path = checkpoint_path.parent / CONFIG_FILENAME
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing model config: {config_path}")
    return json.loads(config_path.read_text())


def load_trained_model(
    checkpoint_path: Path,
    device: str | None = None,
) -> tuple[SequenceTransformer, dict]:
    config_data = load_checkpoint_config(checkpoint_path)
    model_config = SequenceTransformerConfig.from_dict(config_data)
    resolved_device = resolve_device(device)
    model = SequenceTransformer(model_config)
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.to(resolved_device)
    model.eval()
    return model, config_data


def prepare_eval_samples(
    sequences: dict[int, list[int]],
    config_data: dict,
    split: str,
    holdout: int,
) -> list[SequenceSample]:
    user_ids = list(sequences)
    max_users = config_data.get("users")
    seed = int(config_data.get("seed", 42))
    if max_users is not None and len(user_ids) > max_users:
        user_ids = subsample_user_ids(user_ids, max_users, seed)
        sequences = {user_id: sequences[user_id] for user_id in user_ids}

    train_ids, val_ids, test_ids = split_user_ids(user_ids, seed=seed)
    if split == "val":
        eval_user_ids = val_ids
    elif split == "test":
        eval_user_ids = test_ids
    else:
        raise ValueError(f"split must be 'val' or 'test', got {split!r}")

    max_seq_len = int(config_data.get("max_seq_len", 50))
    return build_eval_samples(sequences, eval_user_ids, max_seq_len, holdout=holdout)


def build_popularity_ranking(
    sequences: dict[int, list[int]],
    embedding_table: ItemEmbeddingTable,
    top_n: int | None = None,
) -> list[int]:
    counts: Counter[int] = Counter()
    for seq in sequences.values():
        counts.update(seq)
    ranked = [item_id for item_id, _count in counts.most_common()]
    embedded = set(int(item_id) for item_id in embedding_table.item_ids.tolist())
    ranked = [item_id for item_id in ranked if item_id in embedded]
    if top_n is not None:
        ranked = ranked[:top_n]
    return ranked


def rank_items(
    query_vectors: torch.Tensor,
    embedding_table: ItemEmbeddingTable,
    k: int,
    exclude_item_ids: Iterable[set[int]] | None = None,
) -> list[list[int]]:
    catalog = embedding_table.vectors
    item_ids = embedding_table.item_ids
    scores = query_vectors @ catalog.transpose(0, 1)
    if exclude_item_ids is not None:
        for row_index, excluded in enumerate(exclude_item_ids):
            if not excluded:
                continue
            exclude_index = embedding_table.indices_for(
                torch.tensor(sorted(excluded), device=scores.device)
            )
            valid = exclude_index[exclude_index >= 0]
            if valid.numel():
                scores[row_index, valid] = -torch.inf

    top_k = min(k, scores.size(1))
    top_indices = scores.topk(top_k, dim=1).indices.cpu().numpy()
    return [
        [int(item_ids[column]) for column in row]
        for row in top_indices
    ]


def recall_at_k(rankings: list[list[int]], targets: list[int], k: int) -> float:
    hits = 0
    for ranking, target in zip(rankings, targets, strict=True):
        if target in ranking[:k]:
            hits += 1
    return hits / max(len(targets), 1)


def mrr(rankings: list[list[int]], targets: list[int]) -> float:
    total = 0.0
    for ranking, target in zip(rankings, targets, strict=True):
        try:
            rank = ranking.index(target) + 1
            total += 1.0 / rank
        except ValueError:
            continue
    return total / max(len(targets), 1)


def ndcg_at_k(rankings: list[list[int]], targets: list[int], k: int) -> float:
    total = 0.0
    for ranking, target in zip(rankings, targets, strict=True):
        clipped = ranking[:k]
        if target not in clipped:
            continue
        rank = clipped.index(target) + 1
        total += 1.0 / math.log2(rank + 1)
    return total / max(len(targets), 1)


def coverage(rankings: list[list[int]], catalog_size: int) -> float:
    if catalog_size <= 0:
        return 0.0
    recommended = {item_id for ranking in rankings for item_id in ranking}
    return len(recommended) / catalog_size


def summarize_metrics(
    rankings: list[list[int]],
    targets: list[int],
    k_values: Iterable[int],
    catalog_size: int,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(rankings, targets, k)
        if k == 10:
            metrics["ndcg@10"] = ndcg_at_k(rankings, targets, k)
    metrics["mrr"] = mrr(rankings, targets)
    metrics["coverage"] = coverage(rankings, catalog_size)
    return metrics


def _history_sets(samples: list[SequenceSample]) -> list[set[int]]:
    return [set(sample.input_item_ids) for sample in samples]


def _targets(samples: list[SequenceSample]) -> list[int]:
    return [sample.target_item_id for sample in samples]


@torch.no_grad()
def evaluate_transformer_model(
    model: SequenceTransformer,
    samples: list[SequenceSample],
    embedding_table: ItemEmbeddingTable,
    max_seq_len: int,
    k: int,
    device: str,
    batch_size: int,
) -> list[list[int]]:
    if not samples:
        return []

    dataset = SequenceDataset(samples, max_seq_len=max_seq_len)
    loader = create_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    embedding_table = embedding_table.to(device)
    model = model.to(device)
    histories = _history_sets(samples)
    rankings: list[list[int]] = []
    offset = 0
    amp_enabled = use_amp(device)

    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(device_type=torch.device(device).type, enabled=amp_enabled):
            embeddings = lookup_input_embeddings(batch, embedding_table)
            predicted = model(embeddings, batch["input_mask"]).float()
        batch_size_actual = batch["target_item_id"].size(0)
        batch_histories = histories[offset : offset + batch_size_actual]
        rankings.extend(
            rank_items(predicted, embedding_table, k=k, exclude_item_ids=batch_histories)
        )
        offset += batch_size_actual
    return rankings


def evaluate_popularity_baseline(
    samples: list[SequenceSample],
    popularity_ranking: list[int],
    k: int,
) -> list[list[int]]:
    rankings: list[list[int]] = []
    for sample in samples:
        excluded = set(sample.input_item_ids)
        candidates = [item_id for item_id in popularity_ranking if item_id not in excluded]
        rankings.append(candidates[:k])
    return rankings


def evaluate_history_repeat_baseline(
    samples: list[SequenceSample],
    k: int,
) -> list[list[int]]:
    rankings: list[list[int]] = []
    for sample in samples:
        seen: list[int] = []
        for item_id in reversed(sample.input_item_ids):
            if item_id not in seen:
                seen.append(item_id)
        rankings.append(seen[:k])
    return rankings


@torch.no_grad()
def evaluate_avg_embedding_baseline(
    samples: list[SequenceSample],
    embedding_table: ItemEmbeddingTable,
    max_seq_len: int,
    k: int,
    device: str,
    batch_size: int,
) -> list[list[int]]:
    if not samples:
        return []

    dataset = SequenceDataset(samples, max_seq_len=max_seq_len)
    loader = create_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    embedding_table = embedding_table.to(device)
    histories = _history_sets(samples)
    rankings: list[list[int]] = []
    offset = 0

    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        embeddings = lookup_input_embeddings(batch, embedding_table)
        mask = batch["input_mask"].unsqueeze(-1).to(dtype=embeddings.dtype)
        summed = (embeddings * mask).sum(dim=1)
        lengths = batch["input_mask"].sum(dim=1, keepdim=True).clamp(min=1).to(dtype=embeddings.dtype)
        averaged = torch.nn.functional.normalize(summed / lengths, p=2, dim=-1)
        batch_size_actual = batch["target_item_id"].size(0)
        batch_histories = histories[offset : offset + batch_size_actual]
        rankings.extend(
            rank_items(averaged.float(), embedding_table, k=k, exclude_item_ids=batch_histories)
        )
        offset += batch_size_actual
    return rankings


def run_offline_evaluation(
    session: Session,
    eval_config: EvalConfig,
) -> dict:
    checkpoint_path = eval_config.checkpoint_path
    config_data = load_checkpoint_config(checkpoint_path)
    sequences = load_sequence_cache(eval_config.sequences_cache_dir)
    samples = prepare_eval_samples(
        sequences,
        config_data,
        split=eval_config.split,
        holdout=eval_config.holdout,
    )
    if not samples:
        raise ValueError(f"No {eval_config.split} samples available for evaluation")

    embedding_table = load_embedding_table(session)
    model, _ = load_trained_model(checkpoint_path, device=eval_config.device)
    resolved_device = resolve_device(eval_config.device)
    max_seq_len = int(config_data.get("max_seq_len", 50))
    max_k = max(eval_config.k_values)
    targets = _targets(samples)
    catalog_size = len(embedding_table)
    popularity_ranking = build_popularity_ranking(sequences, embedding_table)

    results: dict[str, dict] = {
        "metadata": {
            "checkpoint": str(checkpoint_path),
            "split": eval_config.split,
            "evaluated_at": datetime.now(UTC).isoformat(),
            "samples": len(samples),
            "catalog_items": catalog_size,
            "k_values": list(eval_config.k_values),
            "training_users": config_data.get("users"),
            "training_seed": config_data.get("seed"),
            "best_val_recall_at_10": config_data.get("best_val_recall_at_10"),
        },
        "models": {},
    }

    evaluators = {
        "transformer": lambda: evaluate_transformer_model(
            model,
            samples,
            embedding_table,
            max_seq_len=max_seq_len,
            k=max_k,
            device=resolved_device,
            batch_size=eval_config.batch_size,
        ),
        "popularity": lambda: evaluate_popularity_baseline(samples, popularity_ranking, max_k),
        "history_repeat": lambda: evaluate_history_repeat_baseline(samples, max_k),
        "avg_history_embedding": lambda: evaluate_avg_embedding_baseline(
            samples,
            embedding_table,
            max_seq_len=max_seq_len,
            k=max_k,
            device=resolved_device,
            batch_size=eval_config.batch_size,
        ),
    }

    for name, evaluator in evaluators.items():
        rankings = evaluator()
        results["models"][name] = summarize_metrics(
            rankings,
            targets,
            eval_config.k_values,
            catalog_size,
        )

    transformer_recall = results["models"]["transformer"].get("recall@10", 0.0)
    baseline_recall = results["models"]["avg_history_embedding"].get("recall@10", 0.0)
    results["metadata"]["beats_avg_embedding_baseline"] = transformer_recall > baseline_recall
    return results


def write_eval_report(report: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
