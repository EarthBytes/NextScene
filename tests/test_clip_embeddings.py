from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from app.models.item_embedding import EMBEDDING_DIM
from app.services.clip_embeddings import (
    _extract_features,
    build_item_text,
    fuse_embeddings,
    resolve_device,
    resolve_poster_path,
)


def _make_item(**kwargs):
    defaults = {
        "item_id": 1,
        "title": "Toy Story",
        "description": "A toy cowboy's world is turned upside down.",
        "genres": ["Animation", "Comedy"],
        "metadata_json": {"actors": "Tom Hanks, Tim Allen"},
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_build_item_text_includes_metadata():
    text = build_item_text(_make_item())
    assert "Toy Story" in text
    assert "A toy cowboy" in text
    assert "Animation, Comedy" in text
    assert "Tom Hanks" in text


def test_build_item_text_falls_back_to_title():
    text = build_item_text(_make_item(description=None, genres=None, metadata_json={}))
    assert text == "Toy Story"


def test_fuse_embeddings_text_only():
    text_vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    text_vector[0] = 1.0

    fused = fuse_embeddings(text_vector)
    assert len(fused) == EMBEDDING_DIM
    assert pytest.approx(np.linalg.norm(fused), rel=1e-5) == 1.0
    assert fused[0] == pytest.approx(1.0)


def test_fuse_embeddings_text_and_image():
    text_vector = np.array([1.0, 0.0, 0.0] + [0.0] * (EMBEDDING_DIM - 3), dtype=np.float32)
    image_vector = np.array([0.0, 1.0, 0.0] + [0.0] * (EMBEDDING_DIM - 3), dtype=np.float32)

    fused = fuse_embeddings(text_vector, image_vector)
    assert len(fused) == EMBEDDING_DIM
    assert pytest.approx(np.linalg.norm(fused), rel=1e-5) == 1.0
    assert fused[0] == pytest.approx(fused[1])


def test_resolve_poster_path_prefers_metadata(tmp_path: Path):
    poster = tmp_path / "42.jpg"
    poster.write_bytes(b"poster")

    item = _make_item(
        item_id=42,
        metadata_json={"poster_path": str(poster)},
    )
    assert resolve_poster_path(item, tmp_path) == poster


def test_resolve_poster_path_uses_index(tmp_path: Path):
    poster = tmp_path / "42.jpg"
    poster.write_bytes(b"poster")

    item = _make_item(item_id=42, metadata_json={})
    assert resolve_poster_path(item, tmp_path, poster_index={42: poster}) == poster


def test_resolve_poster_path_falls_back_to_posters_dir(tmp_path: Path):
    poster = tmp_path / "42.jpg"
    poster.write_bytes(b"poster")

    item = _make_item(item_id=42, metadata_json={})
    assert resolve_poster_path(item, tmp_path) == poster


def test_extract_features_tensor():
    import torch

    tensor = torch.tensor([[1.0, 0.0]])
    assert torch.equal(_extract_features(tensor), tensor)


def test_extract_features_pooler_output():
    from types import SimpleNamespace

    import torch

    tensor = torch.tensor([[1.0, 0.0]])
    output = SimpleNamespace(pooler_output=tensor)
    assert torch.equal(_extract_features(output), tensor)


def test_resolve_device_prefers_mps(monkeypatch):
    torch = pytest.importorskip("torch")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device() == "mps"

