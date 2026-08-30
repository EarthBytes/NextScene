import numpy as np
import pytest
from pathlib import Path

from app.models.item_embedding import EMBEDDING_DIM
from app.services.faiss_index import (
    ids_path_for_index,
    meta_path_for_index,
    normalize_vectors,
    parse_vector,
    save_faiss_index,
)


def test_parse_vector_from_string():
    vector = parse_vector("[1.0, 0.0, 0.5]")
    assert vector == [1.0, 0.0, 0.5]


def test_ids_and_meta_paths():
    index_path = Path("data/faiss/items.index")
    assert ids_path_for_index(index_path) == Path("data/faiss/items.ids.npy")
    assert meta_path_for_index(index_path) == Path("data/faiss/items.meta.json")


def test_normalize_vectors_unit_length():
    pytest.importorskip("faiss")

    vectors = np.array([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]], dtype=np.float32)
    padded = np.zeros((2, EMBEDDING_DIM), dtype=np.float32)
    padded[:, :3] = vectors

    normalized = normalize_vectors(padded)
    norms = np.linalg.norm(normalized, axis=1)
    assert np.allclose(norms, 1.0)


def test_build_save_and_load(tmp_path: Path):
    pytest.importorskip("faiss")
    from app.services.faiss_index import build_faiss_index, load_faiss_index

    vectors = np.random.default_rng(0).standard_normal((8, EMBEDDING_DIM)).astype(np.float32)
    item_ids = np.arange(100, 108, dtype=np.int64)
    index = build_faiss_index(vectors)

    index_path = tmp_path / "items.index"
    save_faiss_index(index, index_path, item_ids)
    loaded_index, loaded_ids = load_faiss_index(index_path)

    assert np.array_equal(loaded_ids, item_ids)
    assert loaded_index.ntotal == 8
    assert index_path.is_file()
    assert ids_path_for_index(index_path).is_file()
    assert meta_path_for_index(index_path).is_file()
