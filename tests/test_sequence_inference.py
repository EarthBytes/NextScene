from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from app.models_ml.sequence_transformer import SequenceTransformer, SequenceTransformerConfig
from app.services.sequence_dataset import ItemEmbeddingTable
from app.services.sequence_inference import SequenceInference
from app.services.sequence_training import save_checkpoint


def _build_inference(tmp_path: Path) -> SequenceInference:
    item_ids = np.arange(1, 21, dtype=np.int64)
    vectors = np.eye(20, 16, dtype=np.float32)
    table = ItemEmbeddingTable(item_ids, vectors)
    config = SequenceTransformerConfig(
        d_model=32,
        n_heads=4,
        n_layers=2,
        ff_dim=64,
        dropout=0.0,
        max_seq_len=4,
        embedding_dim=16,
    )
    model = SequenceTransformer(config)
    checkpoint_path = tmp_path / "best.pt"
    save_checkpoint(checkpoint_path, model, torch.optim.AdamW(model.parameters(), lr=1e-3), 1, 0.0)
    (tmp_path / "config.json").write_text(
        '{"max_seq_len": 4, "embedding_dim": 16, "d_model": 32, "n_heads": 4, "n_layers": 2, "ff_dim": 64, "dropout": 0.1}\n'
    )
    return SequenceInference.from_checkpoint(checkpoint_path, table, device="cpu")


def test_predict_next_vector_shape_and_unit_norm(tmp_path: Path):
    inference = _build_inference(tmp_path)
    vector = inference.predict_next_vector([1, 2, 3])
    assert vector.shape == (16,)
    norm = float(np.linalg.norm(vector))
    assert norm == pytest.approx(1.0, abs=1e-5)


def test_predict_next_vector_requires_history(tmp_path: Path):
    inference = _build_inference(tmp_path)
    with pytest.raises(ValueError):
        inference.predict_next_vector([])
