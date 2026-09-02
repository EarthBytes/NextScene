"""Item serialization and catalog queries for the frontend API."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import case, func, select, text
from sqlalchemy.orm import Session

from app.config import settings
from app.models.interaction import Interaction
from app.models.item import Item
from app.services.poster_download import find_existing_poster

CANONICAL_GENRES = (
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
)


def extract_year(metadata_json: dict | None) -> int | None:
    if not metadata_json:
        return None
    for key in ("start_year", "year"):
        value = metadata_json.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    release_date = metadata_json.get("release_date")
    if release_date and isinstance(release_date, str) and len(release_date) >= 4:
        try:
            return int(release_date[:4])
        except ValueError:
            return None
    return None


def poster_url_for_item(item_id: int) -> str | None:
    posters_dir = Path(settings.posters_dir)
    existing = find_existing_poster(posters_dir, item_id)
    if existing is not None:
        return f"/posters/{existing.name}"
    return None


def resolve_poster_url(item: Item) -> str | None:
    local = poster_url_for_item(item.item_id)
    if local is not None:
        return local
    return item.image_url


def serialize_item(item: Item, *, include_description: bool = True) -> dict:
    metadata = item.metadata_json or {}
    poster_url = resolve_poster_url(item)
    payload = {
        "item_id": item.item_id,
        "title": item.title,
        "genres": item.genres or [],
        "year": extract_year(metadata),
        "image_url": item.image_url,
        "poster_url": poster_url,
        "imdb_id": item.imdb_id,
        "metadata_json": metadata,
    }
    if include_description:
        payload["description"] = item.description
    return payload


def load_items_by_ids(session: Session, item_ids: list[int]) -> dict[int, dict]:
    if not item_ids:
        return {}
    rows = session.execute(select(Item).where(Item.item_id.in_(item_ids))).scalars().all()
    return {item.item_id: serialize_item(item, include_description=False) for item in rows}


def get_item(session: Session, item_id: int) -> dict | None:
    item = session.get(Item, item_id)
    if item is None:
        return None
    return serialize_item(item)


def _normalize_genre_list(genres: list[str] | None) -> list[str]:
    if not genres:
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for genre in genres:
        label = str(genre).strip()
        if not label:
            continue
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(label)
    return cleaned


def genres_overlap_filter(genres: list[str]):
    normalized = _normalize_genre_list(genres)
    if not normalized:
        return None
    return Item.genres.overlap(normalized)


def search_items(
    session: Session,
    *,
    q: str | None = None,
    genre: str | None = None,
    genres: list[str] | None = None,
    tag: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> tuple[list[dict], int]:
    stmt = select(Item)
    count_stmt = select(func.count()).select_from(Item)

    if q:
        query = q.strip()
        pattern = f"%{query}%"
        prefix = f"{query}%"
        title_filter = Item.title.ilike(pattern)
        stmt = stmt.where(title_filter)
        count_stmt = count_stmt.where(title_filter)
        relevance = case(
            (func.lower(Item.title) == query.lower(), 0),
            (Item.title.ilike(prefix), 1),
            else_=2,
        )
        order_by = (relevance, Item.title)
    else:
        order_by = (Item.title,)

    genre_filters = _normalize_genre_list(genres or ([genre] if genre else []))
    genre_clause = genres_overlap_filter(genre_filters)
    if genre_clause is not None:
        stmt = stmt.where(genre_clause)
        count_stmt = count_stmt.where(genre_clause)

    if tag:
        tag_query = tag.strip().lower()
        if tag_query:
            tag_filter = text(
                "EXISTS ("
                "SELECT 1 FROM jsonb_array_elements_text(metadata_json->'tags') AS t "
                "WHERE lower(t::text) LIKE :tag_pattern"
                ")"
            ).bindparams(tag_pattern=f"%{tag_query}%")
            stmt = stmt.where(tag_filter)
            count_stmt = count_stmt.where(tag_filter)

    total = session.execute(count_stmt).scalar_one()
    rows = (
        session.execute(stmt.order_by(*order_by).offset(offset).limit(limit))
        .scalars()
        .all()
    )
    return [serialize_item(item, include_description=False) for item in rows], int(total)


def list_tags(session: Session, limit: int = 30, min_count: int = 25) -> list[dict]:
    rows = session.execute(
        text(
            """
            SELECT lower(tag) AS tag, COUNT(*) AS count
            FROM items, jsonb_array_elements_text(metadata_json->'tags') AS tag
            WHERE metadata_json ? 'tags'
            GROUP BY lower(tag)
            HAVING COUNT(*) >= :min_count
            ORDER BY count DESC
            LIMIT :limit
            """
        ),
        {"limit": limit, "min_count": min_count},
    ).all()
    return [{"tag": str(row.tag), "count": int(row.count)} for row in rows]


def list_genres(session: Session, limit: int = 50, min_count: int = 25) -> list[dict]:
    rows = session.execute(
        select(func.unnest(Item.genres).label("genre"), func.count().label("count"))
        .where(Item.genres.isnot(None))
        .group_by("genre")
    ).all()

    counts = {str(row.genre): int(row.count) for row in rows}
    order = {genre: index for index, genre in enumerate(CANONICAL_GENRES)}
    filtered = [
        {"genre": genre, "count": counts[genre]}
        for genre in CANONICAL_GENRES
        if genre in counts and counts[genre] >= min_count
    ]
    filtered.sort(key=lambda row: order[row["genre"]])
    return filtered[:limit]


def load_user_history(
    session: Session,
    user_id: int,
    *,
    limit: int = 50,
) -> list[dict]:
    rows = session.execute(
        select(Interaction, Item)
        .join(Item, Interaction.item_id == Item.item_id)
        .where(Interaction.user_id == user_id)
        .order_by(Interaction.ts.desc(), Interaction.interaction_id.desc())
        .limit(limit)
    ).all()

    history: list[dict] = []
    for interaction, item in rows:
        entry = serialize_item(item, include_description=False)
        entry.update(
            {
                "interaction_id": int(interaction.interaction_id),
                "type": interaction.type,
                "ts": interaction.ts.isoformat(),
                "context_json": interaction.context_json or {},
            }
        )
        history.append(entry)
    return history


def user_stats(session: Session, user_id: int) -> dict:
    interaction_count = session.execute(
        select(func.count()).select_from(Interaction).where(Interaction.user_id == user_id)
    ).scalar_one()
    rating_count = session.execute(
        select(func.count())
        .select_from(Interaction)
        .where(Interaction.user_id == user_id, Interaction.type == "rating")
    ).scalar_one()
    return {
        "user_id": user_id,
        "interaction_count": int(interaction_count),
        "rating_count": int(rating_count),
    }
