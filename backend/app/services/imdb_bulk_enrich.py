"""Enrich items from IMDb bulk datasets (datasets.imdbws.com)."""

from __future__ import annotations

import csv
import gzip
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.models.item import Item

IMDB_DOWNLOAD_BASE = "https://datasets.imdbws.com"


def iter_imdb_tsv(path: Path) -> Iterator[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            yield row


def load_imdb_ids(session: Session) -> set[str]:
    rows = session.execute(
        text("SELECT imdb_id FROM items WHERE imdb_id IS NOT NULL")
    ).all()
    return {row.imdb_id for row in rows}


def parse_imdb_genres(genres: str | None) -> list[str] | None:
    if not genres or genres == "\\N":
        return None
    return [g.strip() for g in genres.split(",") if g.strip()]


def enrich_from_title_basics(session: Session, path: Path, imdb_ids: set[str]) -> int:
    updated = 0
    for row in iter_imdb_tsv(path):
        tconst = row.get("tconst")
        if not tconst or tconst not in imdb_ids:
            continue

        metadata: dict = {}
        if row.get("primaryTitle") and row["primaryTitle"] != "\\N":
            metadata["primary_title"] = row["primaryTitle"]
        if row.get("originalTitle") and row["originalTitle"] != "\\N":
            metadata["original_title"] = row["originalTitle"]
        if row.get("startYear") and row["startYear"] != "\\N":
            metadata["start_year"] = row["startYear"]
        if row.get("runtimeMinutes") and row["runtimeMinutes"] != "\\N":
            metadata["runtime_minutes"] = int(row["runtimeMinutes"])
        if row.get("titleType") and row["titleType"] != "\\N":
            metadata["title_type"] = row["titleType"]

        genres = parse_imdb_genres(row.get("genres"))
        item = session.execute(select(Item).where(Item.imdb_id == tconst)).scalar_one_or_none()
        if item is None:
            continue

        if genres:
            item.genres = genres
        if metadata:
            current = dict(item.metadata_json or {})
            current.update(metadata)
            item.metadata_json = current

        updated += 1
        if updated % 1000 == 0:
            session.commit()

    session.commit()
    return updated


def enrich_from_title_ratings(session: Session, path: Path, imdb_ids: set[str]) -> int:
    updated = 0
    for row in iter_imdb_tsv(path):
        tconst = row.get("tconst")
        if not tconst or tconst not in imdb_ids:
            continue

        item = session.execute(select(Item).where(Item.imdb_id == tconst)).scalar_one_or_none()
        if item is None:
            continue

        metadata = dict(item.metadata_json or {})
        if row.get("averageRating") and row["averageRating"] != "\\N":
            metadata["imdb_average_rating"] = float(row["averageRating"])
        if row.get("numVotes") and row["numVotes"] != "\\N":
            metadata["imdb_num_votes"] = int(row["numVotes"])
        item.metadata_json = metadata
        updated += 1
        if updated % 1000 == 0:
            session.commit()

    session.commit()
    return updated


def enrich_cast(
    session: Session,
    principals_path: Path,
    names_path: Path,
    imdb_ids: set[str],
    max_actors: int = 5,
) -> int:
    cast_by_tconst: dict[str, list[tuple[int, str]]] = defaultdict(list)
    needed_nconsts: set[str] = set()

    for row in iter_imdb_tsv(principals_path):
        tconst = row.get("tconst")
        if not tconst or tconst not in imdb_ids:
            continue

        category = row.get("category", "")
        if category not in {"actor", "actress", "self"}:
            continue

        nconst = row.get("nconst")
        if not nconst or nconst == "\\N":
            continue

        ordering = int(row["ordering"]) if row.get("ordering") and row["ordering"] != "\\N" else 999
        cast_by_tconst[tconst].append((ordering, nconst))
        needed_nconsts.add(nconst)

    name_map: dict[str, str] = {}
    for row in iter_imdb_tsv(names_path):
        nconst = row.get("nconst")
        if nconst not in needed_nconsts:
            continue
        primary = row.get("primaryName")
        if primary and primary != "\\N":
            name_map[nconst] = primary

    updated = 0
    for tconst, entries in cast_by_tconst.items():
        entries.sort(key=lambda x: x[0])
        actor_names = []
        for _, nconst in entries[:max_actors]:
            name = name_map.get(nconst)
            if name:
                actor_names.append(name)
        if not actor_names:
            continue

        item = session.execute(select(Item).where(Item.imdb_id == tconst)).scalar_one_or_none()
        if item is None:
            continue

        metadata = dict(item.metadata_json or {})
        metadata["actors"] = ", ".join(actor_names)
        item.metadata_json = metadata
        updated += 1
        if updated % 1000 == 0:
            session.commit()

    session.commit()
    return updated


def run_imdb_bulk_enrichment(
    session: Session,
    data_dir: Path,
    include_cast: bool = True,
) -> dict[str, int]:
    imdb_ids = load_imdb_ids(session)
    counts = {"imdb_ids_in_db": len(imdb_ids)}

    basics_path = data_dir / "title.basics.tsv.gz"
    ratings_path = data_dir / "title.ratings.tsv.gz"
    principals_path = data_dir / "title.principals.tsv.gz"
    names_path = data_dir / "name.basics.tsv.gz"

    counts["basics_updated"] = enrich_from_title_basics(session, basics_path, imdb_ids)
    counts["ratings_updated"] = enrich_from_title_ratings(session, ratings_path, imdb_ids)

    if include_cast:
        counts["cast_updated"] = enrich_cast(session, principals_path, names_path, imdb_ids)
    else:
        counts["cast_updated"] = 0

    return counts


def required_imdb_files(include_cast: bool) -> list[str]:
    files = ["title.basics.tsv.gz", "title.ratings.tsv.gz"]
    if include_cast:
        files.extend(["title.principals.tsv.gz", "name.basics.tsv.gz"])
    return files
