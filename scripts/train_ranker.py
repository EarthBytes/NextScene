"""Train a LightGBM ranker on transformer retrieval candidates."""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import app.ml_runtime  # noqa: F401

import numpy as np
import torch

from _common import require_database

from app.config import settings
from app.db.session import SessionLocal
from app.ml_runtime import resolve_device, use_amp
from app.models_ml.checkpoints import BEST_FILENAME
from app.services.catalog_search import search_embedding_catalog
from app.services.ranking_service import build_feature_matrix, load_item_features, train_ranking_model
from app.services.sequence_cache import load_sequence_cache
from app.services.sequence_dataset import (
    build_eval_samples,
    load_embedding_table,
    lookup_input_embeddings,
    split_user_ids,
    subsample_user_ids,
)
from app.services.sequence_evaluation import (
    build_popularity_ranking,
    load_checkpoint_config,
    load_trained_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGBM ranker on retrieval candidates")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(settings.transformer_model_path) / BEST_FILENAME,
        help="Transformer checkpoint used to generate candidates",
    )
    parser.add_argument(
        "--sequences-cache",
        type=Path,
        default=Path(settings.sequences_cache_path),
        help="Cached user sequences",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(settings.ranking_model_path),
        help="Output directory for ranker artifacts",
    )
    parser.add_argument("--candidate-pool", type=int, default=50, help="Candidates per sample")
    parser.add_argument("--max-users", type=int, default=None, help="Limit training users")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


@torch.no_grad()
def generate_training_rows(
    session,
    *,
    checkpoint_path: Path,
    sequences_cache: Path,
    candidate_pool: int,
    max_users: int | None,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    config_data = load_checkpoint_config(checkpoint_path)
    sequences = load_sequence_cache(sequences_cache)
    user_ids = list(sequences)
    seed = int(config_data.get("seed", 42))
    if max_users is not None and len(user_ids) > max_users:
        user_ids = subsample_user_ids(user_ids, max_users, seed)
        sequences = {user_id: sequences[user_id] for user_id in user_ids}

    train_ids, val_ids, _test_ids = split_user_ids(user_ids, seed=seed)
    max_seq_len = int(config_data.get("max_seq_len", 50))
    train_samples = build_eval_samples(sequences, train_ids, max_seq_len, holdout=1)
    val_samples = build_eval_samples(sequences, val_ids, max_seq_len, holdout=1)
    if not train_samples:
        raise ValueError("No training samples available for ranker training")

    embedding_table = load_embedding_table(session)
    model, _ = load_trained_model(checkpoint_path, device=device)
    embedding_table = embedding_table.to(device)
    model = model.to(device)
    popularity_ranking = build_popularity_ranking(sequences, embedding_table)

    def rows_for_samples(samples):
        feature_rows: list[list[float]] = []
        labels: list[int] = []

        for sample in samples:
            history = list(sample.input_item_ids)
            target_id = sample.target_item_id
            predicted_vec = _predict_single(model, history, embedding_table, max_seq_len, device)
            excluded = set(history)
            candidates = search_embedding_catalog(
                embedding_table,
                predicted_vec,
                top_k=candidate_pool,
                exclude_item_ids=excluded,
            )
            if target_id not in {item_id for item_id, _ in candidates}:
                target_vec = _item_vector(embedding_table, target_id)
                if target_vec is not None:
                    score = float(np.dot(predicted_vec, target_vec))
                    candidates.append((target_id, score))

            needed_ids = list({item_id for item_id, _ in candidates} | set(history))
            item_features = load_item_features(session, needed_ids)
            matrix = build_feature_matrix(
                candidate_ids=[item_id for item_id, _ in candidates],
                retrieval_scores=[score for _, score in candidates],
                history=history,
                popularity_ranking=popularity_ranking,
                item_features=item_features,
                embedding_table=embedding_table,
            )
            sample_labels = [1 if item_id == target_id else 0 for item_id, _ in candidates]
            feature_rows.extend(matrix.tolist())
            labels.extend(sample_labels)

        return np.asarray(feature_rows, dtype=np.float32), np.asarray(labels, dtype=np.int8)

    train_x, train_y = rows_for_samples(train_samples)
    val_x, val_y = rows_for_samples(val_samples)
    return train_x, train_y, val_x, val_y


@torch.no_grad()
def _predict_single(model, history, embedding_table, max_seq_len, device):
    clipped = history[-max_seq_len:]
    padded = torch.zeros(max_seq_len, dtype=torch.long, device=device)
    mask = torch.zeros(max_seq_len, dtype=torch.bool, device=device)
    length = len(clipped)
    if length:
        padded[:length] = torch.tensor(clipped, dtype=torch.long, device=device)
        mask[:length] = True
    batch = {"input_item_ids": padded.unsqueeze(0), "input_mask": mask.unsqueeze(0)}
    embeddings = lookup_input_embeddings(batch, embedding_table)
    amp_enabled = use_amp(device)
    with torch.autocast(device_type=torch.device(device).type, enabled=amp_enabled):
        predicted = model(embeddings, batch["input_mask"]).float()
    return predicted[0].detach().cpu().numpy()


def _item_vector(embedding_table, item_id: int) -> np.ndarray | None:
    idx = embedding_table.indices_for(torch.tensor([item_id]))[0].item()
    if idx < 0:
        return None
    return embedding_table.vectors[idx].detach().cpu().numpy()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    session = SessionLocal()
    try:
        require_database(session)
        print(f"Generating training data (candidate pool={args.candidate_pool})...")
        train_x, train_y, val_x, val_y = generate_training_rows(
            session,
            checkpoint_path=args.checkpoint,
            sequences_cache=args.sequences_cache,
            candidate_pool=args.candidate_pool,
            max_users=args.max_users,
            device=device,
        )
        print(f"Training rows: {len(train_y)} ({int(train_y.sum())} positives)")
        print(f"Validation rows: {len(val_y)} ({int(val_y.sum())} positives)")
        ranker = train_ranking_model(
            train_x,
            train_y,
            val_matrix=val_x,
            val_labels=val_y,
            candidate_pool_size=args.candidate_pool,
            train_samples=len(train_y),
        )
        ranker.save(args.output)
        print(f"Saved ranker to {args.output}")
        if ranker.config.val_auc is not None:
            print(f"Validation AUC: {ranker.config.val_auc:.4f}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
