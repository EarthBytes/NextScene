"""Small-catalog ingest for portfolio deploys (Render + Vercel)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd
from app.models.interaction import Interaction
from app.services.movielens_ingest import (
    aggregate_tags_by_movie,
    clamp_interaction_batch_size,
    clear_ingested_data,
    enrich_items_with_tags,
    ingest_links,
    ingest_movies,
    timestamp_to_datetime,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session


def rank_movie_ids_by_tag_count(tags_path: Path, limit: int) -> list[int]:
    """Pick well-tagged movies so search, genres, and popularity have signal."""
    counts: dict[int, int] = defaultdict(int)
    for chunk in pd.read_csv(tags_path, chunksize=100_000):
        for row in chunk.itertuples(index=False):
            counts[int(row.movieId)] += 1

    ranked = sorted(counts, key=counts.get, reverse=True)
    return ranked[:limit]


def filter_tags_by_movies(
    tags_by_movie: dict[int, list[str]], item_ids: frozenset[int]
) -> dict[int, list[str]]:
    return {movie_id: tags for movie_id, tags in tags_by_movie.items() if movie_id in item_ids}


def ingest_ratings_sample(
    session: Session,
    ratings_path: Path,
    item_ids: frozenset[int],
    max_rows: int,
    batch_size: int,
) -> int:
    """Ingest a capped ratings sample for popularity ranking (portfolio deploy)."""
    batch_size = clamp_interaction_batch_size(batch_size)
    total = 0
    print(f"  sampling up to {max_rows:,} ratings for portfolio movies ...", flush=True)

    for chunk in pd.read_csv(ratings_path, chunksize=batch_size):
        if total >= max_rows:
            break

        filtered = chunk[chunk["movieId"].isin(item_ids)]
        if filtered.empty:
            continue

        remaining = max_rows - total
        if len(filtered) > remaining:
            filtered = filtered.head(remaining)

        rows = [
            {
                "user_id": int(row.userId),
                "item_id": int(row.movieId),
                "ts": timestamp_to_datetime(row.timestamp),
                "type": "rating",
                "context_json": {"rating": float(row.rating)},
            }
            for row in filtered.itertuples(index=False)
        ]

        session.execute(insert(Interaction).values(rows))
        session.commit()
        total += len(rows)
        print(f"    ratings: {total:,}", flush=True)

    return total


def run_portfolio_seed(
    session: Session,
    data_dir: Path,
    *,
    max_movies: int = 1_500,
    clear: bool = False,
    sample_ratings: int = 50_000,
    batch_size: int = 10_000,
) -> dict[str, int | list[int]]:
    movies_path = data_dir / "movies.csv"
    links_path = data_dir / "links.csv"
    tags_path = data_dir / "tags.csv"
    ratings_path = data_dir / "ratings.csv"

    for path in (movies_path, links_path, tags_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}")

    if clear:
        clear_ingested_data(session)

    print(f"Selecting top {max_movies:,} movies by tag coverage ...", flush=True)
    selected_ids = rank_movie_ids_by_tag_count(tags_path, max_movies)
    item_ids = frozenset(selected_ids)
    print(f"  selected: {len(selected_ids):,} movies", flush=True)

    counts: dict[str, int | list[int]] = {"selected_movie_ids": selected_ids}

    print("  loading movies ...", flush=True)
    counts["items"] = ingest_movies(session, movies_path, item_ids=item_ids)
    print(f"    movies: {counts['items']:,}", flush=True)

    if links_path.exists():
        print("  updating imdb/tmdb links ...", flush=True)
        counts["links_updated"] = ingest_links(session, links_path, item_ids=item_ids)
        print(f"    links updated: {counts['links_updated']:,}", flush=True)

    print("  aggregating tags per movie ...", flush=True)
    tags_by_movie = filter_tags_by_movies(aggregate_tags_by_movie(tags_path), item_ids)
    print(f"    movies with tags: {len(tags_by_movie):,}", flush=True)

    print("  enriching items with tags ...", flush=True)
    counts["items_tag_enriched"] = enrich_items_with_tags(session, tags_by_movie)
    print(f"    items tag-enriched: {counts['items_tag_enriched']:,}", flush=True)

    counts["tag_interactions"] = 0
    counts["rating_interactions"] = 0

    if ratings_path.is_file() and sample_ratings > 0:
        counts["rating_interactions"] = ingest_ratings_sample(
            session,
            ratings_path,
            item_ids,
            max_rows=sample_ratings,
            batch_size=batch_size,
        )
    else:
        print("  skipping ratings sample (no ratings.csv or sample_ratings=0)", flush=True)

    return counts
