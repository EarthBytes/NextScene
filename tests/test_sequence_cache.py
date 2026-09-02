from pathlib import Path

from app.services.sequence_cache import (
    cache_meta_matches,
    cache_paths,
    load_sequence_cache,
    save_sequence_cache,
    sequences_from_arrays,
    sequences_to_arrays,
)


def test_sequences_round_trip_arrays():
    sequences = {1: [10, 11, 12], 5: [20, 21]}
    user_ids, offsets, items = sequences_to_arrays(sequences)
    restored = sequences_from_arrays(user_ids, offsets, items)
    assert restored == sequences


def test_save_and_load_sequence_cache(tmp_path: Path):
    sequences = {3: [1, 2, 3, 4], 7: [8, 9, 10]}
    save_sequence_cache(
        tmp_path,
        sequences,
        min_rating=3.5,
        min_interactions=3,
        embedded_item_count=100,
    )
    meta_path, npz_path = cache_paths(tmp_path)
    assert npz_path.is_file()
    assert cache_meta_matches(meta_path, min_rating=3.5, min_interactions=3)
    assert load_sequence_cache(tmp_path) == sequences


def test_cache_meta_rejects_mismatched_params(tmp_path: Path):
    save_sequence_cache(
        tmp_path,
        {1: [1, 2, 3]},
        min_rating=3.5,
        min_interactions=3,
        embedded_item_count=10,
    )
    meta_path, _ = cache_paths(tmp_path)
    assert not cache_meta_matches(meta_path, min_rating=4.0, min_interactions=3)
