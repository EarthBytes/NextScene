"""MovieLens 20M ingestion into items and interactions tables."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from app.models.interaction import Interaction
from app.models.item import Item
from app.utils.text import parse_delimited_genres
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

# PostgreSQL rejects statements with more than 65,535 bind parameters.
_INTERACTION_FIELDS_PER_ROW = 5
MAX_INTERACTION_BATCH_SIZE = 65_535 // _INTERACTION_FIELDS_PER_ROW


def clamp_interaction_batch_size(batch_size: int) -> int:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if batch_size > MAX_INTERACTION_BATCH_SIZE:
        print(
            f"Warning: batch_size {batch_size:,} exceeds PostgreSQL limit; "
            f"using {MAX_INTERACTION_BATCH_SIZE:,} instead."
        )
        return MAX_INTERACTION_BATCH_SIZE
    return batch_size


def format_imdb_id(imdb_id: str | int) -> str:
    raw = str(imdb_id).strip()
    if raw.startswith("tt"):
        return raw
    return f"tt{raw.zfill(7)}"


def parse_genres(genres: str | None) -> list[str] | None:
    return parse_delimited_genres(
        genres,
        "|",
        null_sentinels=frozenset({"(no genres listed)"}),
    )


def timestamp_to_datetime(ts: int) -> datetime:
    return datetime.fromtimestamp(int(ts), tz=UTC)


def clear_ingested_data(session: Session) -> None:
    session.execute(
        text("TRUNCATE interactions, item_embeddings, items RESTART IDENTITY CASCADE")
    )
    session.commit()


def ingest_movies(
    session: Session,
    movies_path: Path,
    item_ids: frozenset[int] | set[int] | None = None,
) -> int:
    df = pd.read_csv(movies_path)
    if item_ids is not None:
        df = df[df["movieId"].isin(item_ids)]
    rows = [
        {
            "item_id": int(row.movieId),
            "title": str(row.title),
            "genres": parse_genres(row.genres),
            "metadata_json": {},
        }
        for row in df.itertuples(index=False)
    ]

    stmt = insert(Item).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["item_id"],
        set_={
            "title": stmt.excluded.title,
            "genres": stmt.excluded.genres,
        },
    )
    session.execute(stmt)
    session.commit()
    return len(rows)


def ingest_links(
    session: Session,
    links_path: Path,
    item_ids: frozenset[int] | set[int] | None = None,
) -> int:
    df = pd.read_csv(links_path)
    if item_ids is not None:
        df = df[df["movieId"].isin(item_ids)]
    updated = 0
    for row in df.itertuples(index=False):
        if pd.isna(row.imdbId):
            continue

        imdb_id = format_imdb_id(row.imdbId)
        metadata_patch: dict = {}
        if pd.notna(row.tmdbId):
            metadata_patch["tmdb_id"] = int(row.tmdbId)

        params: dict = {
            "imdb_id": imdb_id,
            "item_id": int(row.movieId),
        }
        if metadata_patch:
            params["metadata"] = json.dumps(metadata_patch)
            query = """
                UPDATE items
                SET imdb_id = :imdb_id,
                    metadata_json = metadata_json || CAST(:metadata AS jsonb),
                    updated_at = NOW()
                WHERE item_id = :item_id
            """
        else:
            query = """
                UPDATE items
                SET imdb_id = :imdb_id,
                    updated_at = NOW()
                WHERE item_id = :item_id
            """

        result = session.execute(text(query), params)
        updated += result.rowcount

    session.commit()
    return updated


def aggregate_tags_by_movie(tags_path: Path) -> dict[int, list[str]]:
    tag_sets: dict[int, set[str]] = defaultdict(set)
    for chunk in pd.read_csv(tags_path, chunksize=100_000):
        for row in chunk.itertuples(index=False):
            tag_sets[int(row.movieId)].add(str(row.tag).strip())

    return {movie_id: sorted(tags) for movie_id, tags in tag_sets.items()}


def enrich_items_with_tags(session: Session, tags_by_movie: dict[int, list[str]]) -> int:
    updated = 0
    for item_id, tags in tags_by_movie.items():
        result = session.execute(
            text(
                """
                UPDATE items
                SET metadata_json = metadata_json || CAST(:metadata AS jsonb),
                    updated_at = NOW()
                WHERE item_id = :item_id
                """
            ),
            {
                "metadata": json.dumps({"tags": tags}),
                "item_id": item_id,
            },
        )
        updated += result.rowcount

    session.commit()
    return updated


def ingest_interactions_from_csv(
    session: Session,
    csv_path: Path,
    interaction_type: str,
    batch_size: int,
    context_fn,
) -> int:
    batch_size = clamp_interaction_batch_size(batch_size)
    total = 0
    print(f"  ingesting {interaction_type} interactions from {csv_path.name} ...")
    for chunk in pd.read_csv(csv_path, chunksize=batch_size):
        rows = []
        for row in chunk.itertuples(index=False):
            rows.append(
                {
                    "user_id": int(row.userId),
                    "item_id": int(row.movieId),
                    "ts": timestamp_to_datetime(row.timestamp),
                    "type": interaction_type,
                    "context_json": context_fn(row),
                }
            )

        session.execute(insert(Interaction).values(rows))
        session.commit()
        total += len(rows)
        print(f"    {interaction_type}: {total:,} rows", flush=True)

    return total


def ingest_tags(session: Session, tags_path: Path, batch_size: int) -> int:
    return ingest_interactions_from_csv(
        session,
        tags_path,
        "tag",
        batch_size,
        lambda row: {"tag": str(row.tag).strip()},
    )


def ingest_ratings(session: Session, ratings_path: Path, batch_size: int) -> int:
    return ingest_interactions_from_csv(
        session,
        ratings_path,
        "rating",
        batch_size,
        lambda row: {"rating": float(row.rating)},
    )


def run_ingestion(
    session: Session,
    data_dir: Path,
    batch_size: int = 10_000,
    clear: bool = False,
    skip_ratings: bool = False,
) -> dict[str, int]:
    movies_path = data_dir / "movies.csv"
    links_path = data_dir / "links.csv"
    tags_path = data_dir / "tags.csv"
    ratings_path = data_dir / "ratings.csv"

    if clear:
        clear_ingested_data(session)

    counts: dict[str, int] = {}

    print("  loading movies ...", flush=True)
    counts["items"] = ingest_movies(session, movies_path)
    print(f"    movies: {counts['items']:,}", flush=True)

    if links_path.exists():
        print("  updating imdb/tmdb links (row-by-row, slow over remote DB) ...", flush=True)
        counts["links_updated"] = ingest_links(session, links_path)
        print(f"    links updated: {counts['links_updated']:,}", flush=True)

    print("  aggregating tags per movie ...", flush=True)
    tags_by_movie = aggregate_tags_by_movie(tags_path)
    print(f"    movies with tags: {len(tags_by_movie):,}", flush=True)
    print("  enriching items with aggregated tags (row-by-row, slow over remote DB) ...", flush=True)
    counts["items_tag_enriched"] = enrich_items_with_tags(session, tags_by_movie)
    print(f"    items tag-enriched: {counts['items_tag_enriched']:,}", flush=True)
    counts["tag_interactions"] = ingest_tags(session, tags_path, batch_size)

    if ratings_path.exists() and not skip_ratings:
        counts["rating_interactions"] = ingest_ratings(session, ratings_path, batch_size)
    elif not skip_ratings:
        counts["rating_interactions"] = 0

    return counts
