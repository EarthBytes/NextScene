"""Training loop, evaluation, and checkpointing for SequenceTransformer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
from sqlalchemy.orm import Session
from torch.utils.data import DataLoader

from app.config import settings
from app.models_ml.contrastive_loss import DEFAULT_TEMPERATURE, infonce_loss
from app.models_ml.sequence_transformer import SequenceTransformer, SequenceTransformerConfig
from app.services.clip_embeddings import resolve_device
from app.services.sequence_dataset import (
    DEFAULT_MIN_RATING,
    ItemEmbeddingTable,
    SequenceDataset,
    build_eval_samples,
    build_training_windows,
    create_dataloader,
    load_embedding_table,
    load_user_sequences,
    lookup_input_embeddings,
    split_user_ids,
    subsample_user_ids,
)

WEIGHTS_FILENAME = "weights.pt"
BEST_FILENAME = "best.pt"
CONFIG_FILENAME = "config.json"
LOG_FILENAME = "training_log.json"


@dataclass
class TrainingConfig:
    epochs: int = 15
    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    negatives_per_sample: int = 64
    temperature: float = DEFAULT_TEMPERATURE
    early_stopping_patience: int = 3
    num_workers: int = 0
    seed: int = 42
    min_rating: float | None = DEFAULT_MIN_RATING
    min_interactions: int = 3
    max_users: int | None = None
    recall_k: int = 10
    holdout: int = 1


def set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device_type(device: torch.device | str) -> str:
    return torch.device(device).type


def _use_amp(device: torch.device | str) -> bool:
    return _device_type(device) in ("cuda", "mps")


def recall_at_k(
    predicted: torch.Tensor,
    target_item_ids: torch.Tensor,
    embedding_table: ItemEmbeddingTable,
    k: int = 10,
) -> torch.Tensor:
    scores = predicted @ embedding_table.vectors.transpose(0, 1)
    top_k = min(k, scores.size(1))
    top_indices = scores.topk(top_k, dim=1).indices
    target_index = embedding_table.indices_for(target_item_ids)
    hits = (top_indices == target_index.unsqueeze(1)).any(dim=1)
    valid = target_index >= 0
    return (hits & valid).float().mean()


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device | str) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _forward_loss(
    model: SequenceTransformer,
    batch: dict[str, torch.Tensor],
    embedding_table: ItemEmbeddingTable,
    negatives_per_sample: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings = lookup_input_embeddings(batch, embedding_table)
    predicted = model(embeddings, batch["input_mask"])
    positive = embedding_table.lookup(batch["target_item_id"])
    extra_negatives = embedding_table.sample_negatives(negatives_per_sample)
    loss = infonce_loss(
        predicted,
        positive,
        extra_negatives=extra_negatives,
        temperature=temperature,
    )
    return loss, predicted


def train_one_epoch(
    model: SequenceTransformer,
    loader: DataLoader,
    embedding_table: ItemEmbeddingTable,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    config: TrainingConfig,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    amp_enabled = _use_amp(device)
    device_type = _device_type(device)

    for batch in loader:
        batch = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device_type, enabled=amp_enabled):
            loss, _predicted = _forward_loss(
                model,
                batch,
                embedding_table,
                negatives_per_sample=config.negatives_per_sample,
                temperature=config.temperature,
            )

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = batch["target_item_id"].size(0)
        total_loss += float(loss.detach()) * batch_size
        total_items += batch_size

    return total_loss / max(total_items, 1)


@torch.no_grad()
def evaluate(
    model: SequenceTransformer,
    loader: DataLoader,
    embedding_table: ItemEmbeddingTable,
    device: torch.device | str,
    config: TrainingConfig,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_recall = 0.0
    total_items = 0
    amp_enabled = _use_amp(device)
    device_type = _device_type(device)

    for batch in loader:
        batch = _move_batch(batch, device)
        with torch.autocast(device_type=device_type, enabled=amp_enabled):
            loss, predicted = _forward_loss(
                model,
                batch,
                embedding_table,
                negatives_per_sample=config.negatives_per_sample,
                temperature=config.temperature,
            )
            recall = recall_at_k(
                predicted.float(),
                batch["target_item_id"],
                embedding_table,
                k=config.recall_k,
            )
        batch_size = batch["target_item_id"].size(0)
        total_loss += float(loss) * batch_size
        total_recall += float(recall) * batch_size
        total_items += batch_size

    denom = max(total_items, 1)
    return total_loss / denom, total_recall / denom


def save_checkpoint(
    path: Path,
    model: SequenceTransformer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_recall_at_10: float,
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_recall_at_10": best_val_recall_at_10,
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, path)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def build_output_config(
    model_config: SequenceTransformerConfig,
    training_config: TrainingConfig,
    extra: dict | None = None,
) -> dict:
    payload = model_config.to_dict()
    payload.update(
        {
            "batch_size": training_config.batch_size,
            "learning_rate": training_config.learning_rate,
            "weight_decay": training_config.weight_decay,
            "epochs": training_config.epochs,
            "negatives_per_sample": training_config.negatives_per_sample,
            "temperature": training_config.temperature,
            "min_rating": training_config.min_rating,
            "seed": training_config.seed,
            "trained_at": datetime.now(UTC).isoformat(),
            "best_val_recall_at_10": None,
        }
    )
    if extra:
        payload.update(extra)
    return payload


def train_sequence_transformer(
    model: SequenceTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    embedding_table: ItemEmbeddingTable,
    output_dir: Path,
    training_config: TrainingConfig,
    device: str | None = None,
    extra_config: dict | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_device = resolve_device(device)
    model = model.to(resolved_device)
    embedding_table.to(resolved_device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(training_config.epochs, 1),
    )
    scaler = (
        torch.amp.GradScaler(_device_type(resolved_device))
        if _device_type(resolved_device) == "cuda"
        else None
    )

    weights_path = output_dir / WEIGHTS_FILENAME
    best_path = output_dir / BEST_FILENAME
    config_path = output_dir / CONFIG_FILENAME
    log_path = output_dir / LOG_FILENAME

    best_recall = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict] = []
    output_config = build_output_config(model.config, training_config, extra=extra_config)

    for epoch in range(1, training_config.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            embedding_table,
            optimizer,
            resolved_device,
            training_config,
            scaler=scaler,
        )
        scheduler.step()

        val_loss = None
        val_recall = None
        if val_loader is not None and len(val_loader) > 0:
            val_loss, val_recall = evaluate(
                model,
                val_loader,
                embedding_table,
                resolved_device,
                training_config,
            )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_recall_at_10": val_recall,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)

        improved = val_recall is None or val_recall > best_recall
        if val_recall is not None and val_recall > best_recall:
            best_recall = val_recall
            best_epoch = epoch
            epochs_without_improvement = 0
        elif val_recall is None:
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            weights_path,
            model,
            optimizer,
            epoch,
            best_recall if best_recall >= 0 else 0.0,
            scaler=scaler,
        )
        if improved:
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                best_recall if best_recall >= 0 else 0.0,
                scaler=scaler,
            )

        output_config["best_val_recall_at_10"] = best_recall if best_recall >= 0 else None
        output_config["best_epoch"] = best_epoch
        write_json(config_path, output_config)
        write_json(
            log_path,
            {
                "epochs": history,
                "best_epoch": best_epoch,
                "best_val_recall_at_10": best_recall if best_recall >= 0 else None,
            },
        )

        val_part = ""
        if val_recall is not None:
            val_part = f"  val_loss={val_loss:.4f}  recall@{training_config.recall_k}={val_recall:.4f}"
        print(f"epoch {epoch:02d}  train_loss={train_loss:.4f}{val_part}")

        if (
            val_loader is not None
            and training_config.early_stopping_patience > 0
            and epochs_without_improvement >= training_config.early_stopping_patience
        ):
            print(
                f"Early stopping at epoch {epoch} "
                f"(best recall@{training_config.recall_k}={best_recall:.4f} at epoch {best_epoch})"
            )
            break

    return {
        "best_epoch": best_epoch,
        "best_val_recall_at_10": best_recall if best_recall >= 0 else None,
        "epochs_run": len(history),
        "weights_path": str(weights_path),
        "best_path": str(best_path),
        "device": resolved_device,
    }


def prepare_dataloaders(
    sequences: dict[int, list[int]],
    max_seq_len: int,
    training_config: TrainingConfig,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    user_ids = list(sequences)
    if training_config.max_users is not None:
        user_ids = subsample_user_ids(user_ids, training_config.max_users, training_config.seed)

    train_ids, val_ids, test_ids = split_user_ids(user_ids, seed=training_config.seed)
    train_samples = build_training_windows(sequences, train_ids, max_seq_len)
    val_samples = build_eval_samples(
        sequences, val_ids, max_seq_len, holdout=training_config.holdout
    )
    test_samples = build_eval_samples(
        sequences, test_ids, max_seq_len, holdout=training_config.holdout
    )

    train_loader = create_dataloader(
        SequenceDataset(train_samples, max_seq_len),
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
    )
    val_loader = create_dataloader(
        SequenceDataset(val_samples, max_seq_len),
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
    )
    test_loader = create_dataloader(
        SequenceDataset(test_samples, max_seq_len),
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
    )
    counts = {
        "users": len(user_ids),
        "train_users": len(train_ids),
        "val_users": len(val_ids),
        "test_users": len(test_ids),
        "train_windows": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
    }
    return train_loader, val_loader, test_loader, counts


def run_training(
    session: Session,
    output_dir: Path,
    model_config: SequenceTransformerConfig | None = None,
    training_config: TrainingConfig | None = None,
    device: str | None = None,
) -> dict:
    model_config = model_config or SequenceTransformerConfig(clip_model=settings.clip_model_name)
    training_config = training_config or TrainingConfig()
    set_seed(training_config.seed)

    embedding_table = load_embedding_table(session)
    embedded_ids = set(int(item_id) for item_id in embedding_table.item_ids.tolist())
    sequences = load_user_sequences(
        session,
        embedded_item_ids=embedded_ids,
        min_rating=training_config.min_rating,
        min_interactions=training_config.min_interactions,
    )
    if not sequences:
        raise ValueError("No user sequences available for training")

    train_loader, val_loader, _test_loader, counts = prepare_dataloaders(
        sequences,
        max_seq_len=model_config.max_seq_len,
        training_config=training_config,
    )
    if counts["train_windows"] == 0:
        raise ValueError("Train split produced no windows")

    model = SequenceTransformer(model_config)
    result = train_sequence_transformer(
        model,
        train_loader,
        val_loader if counts["val_samples"] else None,
        embedding_table,
        output_dir,
        training_config,
        device=device,
        extra_config=counts,
    )
    result.update(counts)
    result["catalog_items"] = len(embedding_table)
    return result
