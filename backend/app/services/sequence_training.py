"""Training loop, evaluation, and checkpointing for SequenceTransformer."""

from __future__ import annotations

import app.ml_runtime  # noqa: F401

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import torch
from sqlalchemy.orm import Session
from torch.utils.data import DataLoader

from app.config import settings
from app.ml_runtime import resolve_device, resolve_num_workers, use_amp
from app.models_ml.checkpoints import BEST_FILENAME, CONFIG_FILENAME, WEIGHTS_FILENAME
from app.models_ml.contrastive_loss import DEFAULT_TEMPERATURE, infonce_loss
from app.models_ml.sequence_transformer import SequenceTransformer, SequenceTransformerConfig
from app.services.hard_negatives import NegativeSampler
from app.services.sequence_cache import load_or_build_sequences
from app.services.sequence_dataset import (
    DEFAULT_MIN_RATING,
    ItemEmbeddingTable,
    SampledWindowDataset,
    SequenceDataset,
    build_eval_samples,
    create_dataloader,
    load_embedded_item_ids,
    load_embedding_table,
    load_sequences_for_user_subset,
    lookup_input_embeddings,
    split_into_shards,
    split_user_ids,
    subsample_user_ids,
)

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
    sequences_cache_dir: str = "data/sequences"
    rebuild_sequences: bool = False
    windows_per_user: int = 5
    num_shards: int = 4
    use_hard_negatives: bool = True
    hard_negatives_per_sample: int = 32
    random_negatives_per_sample: int = 32
    faiss_index_path: str | None = None
    user_batches: int = 1
    user_batch: int | None = None
    resume_checkpoint: Path | None = None


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


def build_negative_sampler(
    embedding_table: ItemEmbeddingTable,
    training_config: TrainingConfig,
) -> NegativeSampler | None:
    random_negatives = training_config.random_negatives_per_sample
    hard_negatives = training_config.hard_negatives_per_sample
    if not training_config.use_hard_negatives:
        hard_negatives = 0
        random_negatives = training_config.negatives_per_sample

    if random_negatives <= 0 and hard_negatives <= 0:
        return None

    return NegativeSampler(
        embedding_table,
        random_negatives=random_negatives,
        hard_negatives=hard_negatives,
    )


def _forward_loss(
    model: SequenceTransformer,
    batch: dict[str, torch.Tensor],
    embedding_table: ItemEmbeddingTable,
    negative_sampler: NegativeSampler | None,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings = lookup_input_embeddings(batch, embedding_table)
    predicted = model(embeddings, batch["input_mask"])
    positive = embedding_table.lookup(batch["target_item_id"])
    extra_negatives = None
    per_sample_negatives = None
    if negative_sampler is not None:
        sampled = negative_sampler.sample(
            query_vectors=predicted,
            exclude_item_ids=batch["target_item_id"],
        )
        extra_negatives = sampled.shared
        per_sample_negatives = sampled.per_sample
    loss = infonce_loss(
        predicted,
        positive,
        extra_negatives=extra_negatives,
        per_sample_negatives=per_sample_negatives,
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
    negative_sampler: NegativeSampler | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    amp_enabled = use_amp(device)
    device_type = _device_type(device)

    for batch in loader:
        batch = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device_type, enabled=amp_enabled):
            loss, _predicted = _forward_loss(
                model,
                batch,
                embedding_table,
                negative_sampler,
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
    negative_sampler: NegativeSampler | None = None,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_recall = 0.0
    total_items = 0
    amp_enabled = use_amp(device)
    device_type = _device_type(device)

    for batch in loader:
        batch = _move_batch(batch, device)
        with torch.autocast(device_type=device_type, enabled=amp_enabled):
            loss, predicted = _forward_loss(
                model,
                batch,
                embedding_table,
                negative_sampler,
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
            "windows_per_user": training_config.windows_per_user,
            "num_shards": training_config.num_shards,
            "use_hard_negatives": training_config.use_hard_negatives,
            "hard_negatives_per_sample": training_config.hard_negatives_per_sample,
            "random_negatives_per_sample": training_config.random_negatives_per_sample,
            "trained_at": datetime.now(UTC).isoformat(),
            "best_val_recall_at_10": None,
        }
    )
    if extra:
        payload.update(extra)
    return payload


def _epoch_seed(base_seed: int, epoch: int, shard_index: int) -> int:
    return base_seed + epoch * 10_000 + shard_index * 100


def load_checkpoint(
    path: Path,
    model: SequenceTransformer,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint.get("epoch", 0)), float(checkpoint.get("best_val_recall_at_10", -1.0))


def train_sequence_transformer(
    model: SequenceTransformer,
    sequences: dict[int, list[int]],
    train_shard_user_ids: list[list[int]],
    val_loader: DataLoader | None,
    embedding_table: ItemEmbeddingTable,
    output_dir: Path,
    training_config: TrainingConfig,
    max_seq_len: int,
    device: str | None = None,
    extra_config: dict | None = None,
    resume_checkpoint: Path | None = None,
    enable_early_stopping: bool = True,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_device = resolve_device(device)
    num_workers = resolve_num_workers(resolved_device, training_config.num_workers)
    if num_workers != training_config.num_workers:
        print(
            f"Using num_workers={num_workers} on {resolved_device} "
            f"(requested {training_config.num_workers}; macOS/MPS requires 0)"
        )

    negative_sampler = build_negative_sampler(embedding_table, training_config)
    model = model.to(resolved_device)
    embedding_table.to(resolved_device)
    if negative_sampler is not None:
        negative_sampler.embedding_table = embedding_table

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

    best_recall = -1.0
    best_epoch = 0
    if resume_checkpoint is not None and resume_checkpoint.is_file():
        resumed_epoch, best_recall = load_checkpoint(resume_checkpoint, model, optimizer, scaler)
        best_epoch = resumed_epoch
        print(f"Resumed from {resume_checkpoint} (epoch={resumed_epoch}, best_recall@10={best_recall:.4f})")

    weights_path = output_dir / WEIGHTS_FILENAME
    best_path = output_dir / BEST_FILENAME
    config_path = output_dir / CONFIG_FILENAME
    log_path = output_dir / LOG_FILENAME

    epochs_without_improvement = 0
    history: list[dict] = []
    output_config = build_output_config(model.config, training_config, extra=extra_config)
    patience = training_config.early_stopping_patience if enable_early_stopping else 0

    for epoch in range(1, training_config.epochs + 1):
        shard_losses: list[float] = []
        shard_sizes: list[int] = []
        for shard_index, shard_user_ids in enumerate(train_shard_user_ids):
            if not shard_user_ids:
                continue
            dataset = SampledWindowDataset(
                sequences,
                shard_user_ids,
                max_seq_len=max_seq_len,
                windows_per_user=training_config.windows_per_user,
                seed=_epoch_seed(training_config.seed, epoch, shard_index),
            )
            if len(dataset) == 0:
                continue
            loader = create_dataloader(
                dataset,
                batch_size=training_config.batch_size,
                shuffle=True,
                num_workers=num_workers,
            )
            shard_loss = train_one_epoch(
                model,
                loader,
                embedding_table,
                optimizer,
                resolved_device,
                training_config,
                negative_sampler=negative_sampler,
                scaler=scaler,
            )
            shard_losses.append(shard_loss)
            shard_sizes.append(len(dataset))

        if shard_sizes:
            train_loss = sum(loss * size for loss, size in zip(shard_losses, shard_sizes)) / sum(shard_sizes)
        else:
            train_loss = 0.0
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
                negative_sampler=negative_sampler,
            )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_recall_at_10": val_recall,
            "lr": optimizer.param_groups[0]["lr"],
            "train_samples": sum(shard_sizes),
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
        sample_part = f"  samples={sum(shard_sizes):,}" if shard_sizes else ""
        print(f"epoch {epoch:02d}  train_loss={train_loss:.4f}{val_part}{sample_part}")

        if (
            val_loader is not None
            and patience > 0
            and epochs_without_improvement >= patience
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


def prepare_eval_dataloaders(
    sequences: dict[int, list[int]],
    max_seq_len: int,
    training_config: TrainingConfig,
    device: str | None = None,
    train_user_ids: list[int] | None = None,
) -> tuple[list[list[int]], DataLoader, DataLoader, dict[str, int]]:
    user_ids = list(sequences)
    train_ids, val_ids, test_ids = split_user_ids(user_ids, seed=training_config.seed)
    active_train_ids = train_user_ids if train_user_ids is not None else train_ids
    val_samples = build_eval_samples(
        sequences, val_ids, max_seq_len, holdout=training_config.holdout
    )
    test_samples = build_eval_samples(
        sequences, test_ids, max_seq_len, holdout=training_config.holdout
    )
    num_workers = resolve_num_workers(device, training_config.num_workers)

    val_loader = create_dataloader(
        SequenceDataset(val_samples, max_seq_len),
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = create_dataloader(
        SequenceDataset(test_samples, max_seq_len),
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    train_shards = split_into_shards(active_train_ids, training_config.num_shards)
    counts = {
        "users": len(user_ids),
        "train_users": len(active_train_ids),
        "val_users": len(val_ids),
        "test_users": len(test_ids),
        "train_windows_per_epoch": len(active_train_ids) * training_config.windows_per_user,
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "num_shards": len(train_shards),
    }
    return train_shards, val_loader, test_loader, counts


def load_training_sequences(
    session: Session,
    training_config: TrainingConfig,
) -> tuple[dict[int, list[int]], str]:
    cache_dir = Path(training_config.sequences_cache_dir)
    if training_config.max_users is None:
        sequences, source = load_or_build_sequences(
            session,
            cache_dir,
            min_rating=training_config.min_rating,
            min_interactions=training_config.min_interactions,
            rebuild=training_config.rebuild_sequences,
        )
        return sequences, source

    if (
        not training_config.rebuild_sequences
        and cache_dir.joinpath("sequences.npz").is_file()
    ):
        from app.services.sequence_cache import cache_meta_matches, cache_paths, load_sequence_cache

        meta_path, _ = cache_paths(cache_dir)
        if cache_meta_matches(
            meta_path,
            min_rating=training_config.min_rating,
            min_interactions=training_config.min_interactions,
        ):
            sequences = load_sequence_cache(cache_dir)
            chosen = subsample_user_ids(list(sequences), training_config.max_users, training_config.seed)
            return {user_id: sequences[user_id] for user_id in chosen}, "cache"

    embedded_ids = load_embedded_item_ids(session)
    sequences = load_sequences_for_user_subset(
        session,
        embedded_ids,
        max_users=training_config.max_users,
        seed=training_config.seed,
        min_rating=training_config.min_rating,
        min_interactions=training_config.min_interactions,
    )
    return sequences, "database_subset"


def split_train_users_into_batches(train_user_ids: list[int], num_batches: int) -> list[list[int]]:
    if num_batches < 1:
        raise ValueError("num_batches must be >= 1")
    return split_into_shards(train_user_ids, num_batches)


def run_batched_training(
    session: Session,
    output_dir: Path,
    model_config: SequenceTransformerConfig,
    training_config: TrainingConfig,
    device: str | None = None,
) -> dict:
    embedding_table = load_embedding_table(session)
    sequences, sequence_source = load_training_sequences(session, training_config)
    if not sequences:
        raise ValueError("No user sequences available for training")

    all_user_ids = list(sequences)
    train_ids, _val_ids, _test_ids = split_user_ids(all_user_ids, seed=training_config.seed)
    user_batch_slices = split_train_users_into_batches(train_ids, training_config.user_batches)

    if training_config.user_batch is not None:
        if training_config.user_batch < 0 or training_config.user_batch >= len(user_batch_slices):
            raise ValueError(
                f"user_batch must be between 0 and {len(user_batch_slices) - 1}, "
                f"got {training_config.user_batch}"
            )
        batch_indices = [training_config.user_batch]
    else:
        batch_indices = list(range(len(user_batch_slices)))

    epochs_per_batch = max(1, training_config.epochs // training_config.user_batches)
    remainder = training_config.epochs % training_config.user_batches

    _, val_loader, _test_loader, base_counts = prepare_eval_dataloaders(
        sequences,
        max_seq_len=model_config.max_seq_len,
        training_config=training_config,
        device=device,
    )

    model = SequenceTransformer(model_config)
    batch_results: list[dict] = []
    cumulative_epochs = 0

    for position, batch_index in enumerate(batch_indices):
        batch_train_ids = user_batch_slices[batch_index]
        if not batch_train_ids:
            continue

        batch_epochs = epochs_per_batch + (1 if batch_index < remainder else 0)
        train_shards = split_into_shards(batch_train_ids, training_config.num_shards)
        is_last_batch = batch_index == training_config.user_batches - 1
        if training_config.resume_checkpoint is not None:
            resume_path = training_config.resume_checkpoint
            if not resume_path.is_file():
                resume_path = None
        elif batch_index > 0 and (output_dir / WEIGHTS_FILENAME).is_file():
            resume_path = output_dir / WEIGHTS_FILENAME
        else:
            resume_path = None

        batch_training_config = replace(training_config, epochs=batch_epochs)

        print(
            f"User batch {batch_index + 1}/{training_config.user_batches}: "
            f"{len(batch_train_ids):,} train users, {batch_epochs} epochs"
        )

        counts = {
            **base_counts,
            "train_users": len(batch_train_ids),
            "train_windows_per_epoch": len(batch_train_ids) * training_config.windows_per_user,
            "user_batch": batch_index,
            "user_batches": training_config.user_batches,
            "epochs_this_batch": batch_epochs,
        }

        result = train_sequence_transformer(
            model,
            sequences,
            train_shards,
            val_loader if base_counts["val_samples"] else None,
            embedding_table,
            output_dir,
            batch_training_config,
            max_seq_len=model_config.max_seq_len,
            device=device,
            extra_config={**counts, "sequence_source": sequence_source},
            resume_checkpoint=resume_path,
            enable_early_stopping=is_last_batch,
        )
        cumulative_epochs += int(result["epochs_run"])
        batch_results.append(result)
        training_config.resume_checkpoint = output_dir / WEIGHTS_FILENAME

    final = dict(batch_results[-1])
    final.update(base_counts)
    final["catalog_items"] = len(embedding_table)
    final["sequence_source"] = sequence_source
    final["user_batches"] = training_config.user_batches
    final["epochs_run"] = cumulative_epochs
    final["batch_results"] = batch_results
    return final


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

    if training_config.user_batches > 1:
        return run_batched_training(
            session,
            output_dir,
            model_config,
            training_config,
            device=device,
        )

    embedding_table = load_embedding_table(session)
    sequences, sequence_source = load_training_sequences(session, training_config)
    if not sequences:
        raise ValueError("No user sequences available for training")

    train_shards, val_loader, _test_loader, counts = prepare_eval_dataloaders(
        sequences,
        max_seq_len=model_config.max_seq_len,
        training_config=training_config,
        device=device,
    )
    if not any(shard for shard in train_shards):
        raise ValueError("Train split produced no users")

    model = SequenceTransformer(model_config)
    result = train_sequence_transformer(
        model,
        sequences,
        train_shards,
        val_loader if counts["val_samples"] else None,
        embedding_table,
        output_dir,
        training_config,
        max_seq_len=model_config.max_seq_len,
        device=device,
        extra_config={**counts, "sequence_source": sequence_source},
        resume_checkpoint=training_config.resume_checkpoint,
    )
    result.update(counts)
    result["catalog_items"] = len(embedding_table)
    result["sequence_source"] = sequence_source
    return result
