import numpy as np
from app.services.catalog_search import CatalogSearcher, search_embedding_catalog
from app.services.sequence_dataset import ItemEmbeddingTable


def _table() -> ItemEmbeddingTable:
    item_ids = np.array([1, 2, 3, 4], dtype=np.int64)
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )
    return ItemEmbeddingTable(item_ids=item_ids, vectors=vectors)


def test_catalog_searcher_numpy_mode():
    table = _table()
    searcher = CatalogSearcher(embedding_table=table, mode="numpy")
    query = np.array([1.0, 0.0], dtype=np.float32)
    results = searcher.search(query, top_k=2, exclude_item_ids={1})
    assert [item_id for item_id, _score in results] == [2, 3]


def test_search_embedding_catalog_returns_sorted_scores():
    table = _table()
    query = np.array([1.0, 0.0], dtype=np.float32)
    results = search_embedding_catalog(table, query, top_k=3)
    scores = [score for _item_id, score in results]
    assert scores == sorted(scores, reverse=True)
