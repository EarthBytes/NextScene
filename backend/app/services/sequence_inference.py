"""Load a trained sequence transformer and predict next-item vectors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from app.models_ml.sequence_transformer import SequenceTransformer
from app.services.clip_embeddings import resolve_device
from app.services.sequence_dataset import ItemEmbeddingTable, lookup_input_embeddings
from app.services.sequence_evaluation import load_trained_model
from app.services.sequence_training import _use_amp

BEST_FILENAME = "best.pt"


class SequenceInference:
    """Run next-item embedding inference from a user interaction history."""

    def __init__(
        self,
        model: SequenceTransformer,
        embedding_table: ItemEmbeddingTable,
        max_seq_len: int,
        device: str | None = None,
    ) -> None:
        self.model = model
        self.embedding_table = embedding_table
        self.max_seq_len = max_seq_len
        self.device = resolve_device(device or "cpu")
        self.model.to(self.device)
        self.embedding_table.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        embedding_table: ItemEmbeddingTable,
        device: str | None = None,
    ) -> SequenceInference:
        model, config = load_trained_model(checkpoint_path, device=device or "cpu")
        max_seq_len = int(config.get("max_seq_len", model.config.max_seq_len))
        return cls(model, embedding_table, max_seq_len=max_seq_len, device=device or "cpu")

    @classmethod
    def from_model_dir(
        cls,
        model_dir: Path,
        embedding_table: ItemEmbeddingTable,
        device: str | None = None,
    ) -> SequenceInference:
        best_path = model_dir / BEST_FILENAME
        if not best_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {best_path}")
        return cls.from_checkpoint(best_path, embedding_table, device=device)

    @torch.no_grad()
    def predict_next_vector(self, history_item_ids: list[int]) -> np.ndarray:
        if not history_item_ids:
            raise ValueError("history_item_ids must not be empty")

        batch = self._history_to_batch(history_item_ids)
        batch = {key: value.to(self.device) for key, value in batch.items()}
        amp_enabled = _use_amp(self.device)
        with torch.autocast(device_type=torch.device(self.device).type, enabled=amp_enabled):
            embeddings = lookup_input_embeddings(batch, self.embedding_table)
            predicted = self.model(embeddings, batch["input_mask"])
        return predicted[0].detach().float().cpu().numpy()

    def _history_to_batch(self, history_item_ids: list[int]) -> dict[str, torch.Tensor]:
        from app.services.sequence_dataset import PAD_ITEM_ID

        clipped = list(history_item_ids[-self.max_seq_len :])
        padded = torch.full((1, self.max_seq_len), PAD_ITEM_ID, dtype=torch.long)
        mask = torch.zeros(1, self.max_seq_len, dtype=torch.bool)
        length = len(clipped)
        if length:
            padded[0, :length] = torch.tensor(clipped, dtype=torch.long)
            mask[0, :length] = True
        return {
            "input_item_ids": padded,
            "input_mask": mask,
            "target_item_id": torch.tensor([clipped[-1]], dtype=torch.long),
        }
