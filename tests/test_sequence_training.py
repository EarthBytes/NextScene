from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

import torch
import torch.nn.functional as F
from app.models_ml.sequence_transformer import SequenceTransformer, SequenceTransformerConfig
from app.services.sequence_dataset import (
    ItemEmbeddingTable,
    SequenceDataset,
    SequenceSample,
    create_dataloader,
    lookup_input_embeddings,
    split_into_shards,
)
from app.services.sequence_training import (
    TrainingConfig,
    recall_at_k,
    train_one_epoch,
    train_sequence_transformer,
)


def _table_and_loader(batch_size: int = 4):
    catalog_ids = np.arange(1, 21, dtype=np.int64)
    vectors = np.eye(20, 16, dtype=np.float32)
    table = ItemEmbeddingTable(catalog_ids, vectors)

    samples = [
        SequenceSample(
            user_id=i,
            input_item_ids=(1 + (i % 10), 2 + (i % 10)),
            target_item_id=3 + (i % 10),
        )
        for i in range(16)
    ]
    loader = create_dataloader(SequenceDataset(samples, max_seq_len=4), batch_size=batch_size, shuffle=False)
    config = SequenceTransformerConfig(
        d_model=32,
        n_heads=4,
        n_layers=2,
        ff_dim=64,
        dropout=0.0,
        max_seq_len=4,
        embedding_dim=16,
    )
    return table, loader, config


def test_single_training_step_runs():
    table, loader, config = _table_and_loader()
    model = SequenceTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    training_config = TrainingConfig(negatives_per_sample=8, batch_size=4, epochs=1)
    loss = train_one_epoch(
        model,
        loader,
        table,
        optimizer,
        device="cpu",
        config=training_config,
    )
    assert np.isfinite(loss)
    assert any(param.grad is not None for param in model.parameters() if param.requires_grad)


def test_recall_at_k_finds_exact_match():
    item_ids = np.array([10, 20, 30], dtype=np.int64)
    vectors = np.eye(3, 4, dtype=np.float32)
    table = ItemEmbeddingTable(item_ids, vectors)
    predicted = table.lookup(torch.tensor([20, 10]))
    recall = recall_at_k(predicted, torch.tensor([20, 10]), table, k=1)
    assert float(recall) == 1.0
    miss = recall_at_k(predicted, torch.tensor([30, 30]), table, k=1)
    assert float(miss) == 0.0


def test_training_loop_writes_checkpoints(tmp_path: Path):
    table, loader, config = _table_and_loader(batch_size=8)
    model = SequenceTransformer(config)
    training_config = TrainingConfig(
        epochs=2,
        batch_size=8,
        learning_rate=1e-3,
        negatives_per_sample=8,
        random_negatives_per_sample=8,
        hard_negatives_per_sample=0,
        use_hard_negatives=False,
        early_stopping_patience=5,
        windows_per_user=2,
        num_shards=1,
        num_workers=0,
    )
    sequences = {
        sample.user_id: list(sample.input_item_ids) + [sample.target_item_id]
        for sample in [
            SequenceSample(user_id=i, input_item_ids=(1 + (i % 10), 2 + (i % 10)), target_item_id=3 + (i % 10))
            for i in range(16)
        ]
    }
    train_shards = split_into_shards(list(sequences), num_shards=1)
    result = train_sequence_transformer(
        model,
        sequences,
        train_shards,
        loader,
        table,
        output_dir=tmp_path,
        training_config=training_config,
        max_seq_len=4,
        device="cpu",
    )
    assert result["epochs_run"] == 2
    assert (tmp_path / "weights.pt").is_file()
    assert (tmp_path / "best.pt").is_file()
    assert (tmp_path / "config.json").is_file()
    assert (tmp_path / "training_log.json").is_file()

    checkpoint = torch.load(tmp_path / "best.pt", map_location="cpu", weights_only=True)
    assert "model" in checkpoint
    assert checkpoint["epoch"] >= 1


def test_forward_loss_inputs_are_masked():
    table, loader, config = _table_and_loader()
    model = SequenceTransformer(config)
    model.eval()
    batch = next(iter(loader))
    embeddings = lookup_input_embeddings(batch, table)
    assert embeddings.shape[0] == batch["input_item_ids"].shape[0]
    pad_positions = ~batch["input_mask"]
    if pad_positions.any():
        assert torch.allclose(
            embeddings[pad_positions],
            torch.zeros(embeddings.size(-1)),
        )
    predicted = model(embeddings, batch["input_mask"])
    assert predicted.shape[-1] == 16
    assert torch.allclose(predicted.norm(dim=-1), torch.ones(predicted.size(0)), atol=1e-5)
    assert F.mse_loss(predicted, table.lookup(batch["target_item_id"])).item() >= 0
