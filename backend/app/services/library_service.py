"""Library-scoped interaction queries for authenticated app users."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.interaction import Interaction
from app.models.item import Item
from app.services.item_service import serialize_item

LIBRARY_SOURCE = "library"
WATCHLIST_SOURCE = "watchlist"
DISMISSED_SOURCE = "dismissed"


def _source_clause(source: str):
    return Interaction.context_json.contains({"source": source})


def library_interaction_clause():
    return _source_clause(LIBRARY_SOURCE)


def load_library_history(session: Session, user_id: int, *, max_items: int = 50) -> list[int]:
    rows = session.execute(
        select(Interaction.item_id)
        .where(Interaction.user_id == user_id, library_interaction_clause())
        .order_by(Interaction.ts, Interaction.interaction_id)
    ).scalars().all()

    history: list[int] = []
    for item_id in rows:
        iid = int(item_id)
        if history and history[-1] == iid:
            continue
        history.append(iid)
    return history[-max_items:]


def load_library_seen_items(session: Session, user_id: int) -> set[int]:
    rows = session.execute(
        select(Interaction.item_id)
        .where(Interaction.user_id == user_id, library_interaction_clause())
        .distinct()
    ).scalars().all()
    return {int(item_id) for item_id in rows}


def load_dismissed_items(session: Session, user_id: int) -> set[int]:
    rows = session.execute(
        select(Interaction.item_id)
        .where(Interaction.user_id == user_id, _source_clause(DISMISSED_SOURCE))
        .distinct()
    ).scalars().all()
    return {int(item_id) for item_id in rows}


def load_excluded_recommendation_items(session: Session, user_id: int) -> set[int]:
    """Items to exclude from recommendations: library + dismissed."""
    return load_library_seen_items(session, user_id) | load_dismissed_items(session, user_id)


def load_user_library(session: Session, user_id: int) -> list[dict]:
    rows = session.execute(
        select(Interaction, Item)
        .join(Item, Interaction.item_id == Item.item_id)
        .where(
            Interaction.user_id == user_id,
            Interaction.type == "view",
            library_interaction_clause(),
        )
        .order_by(Interaction.ts.desc(), Interaction.interaction_id.desc())
    ).all()

    seen: set[int] = set()
    library: list[dict] = []
    for interaction, item in rows:
        if item.item_id in seen:
            continue
        seen.add(item.item_id)
        entry = serialize_item(item, include_description=False)
        entry["added_at"] = interaction.ts.isoformat()
        library.append(entry)
    return library


def load_user_watchlist(session: Session, user_id: int) -> list[dict]:
    rows = session.execute(
        select(Interaction, Item)
        .join(Item, Interaction.item_id == Item.item_id)
        .where(
            Interaction.user_id == user_id,
            Interaction.type == "click",
            _source_clause(WATCHLIST_SOURCE),
        )
        .order_by(Interaction.ts.desc(), Interaction.interaction_id.desc())
    ).all()

    seen: set[int] = set()
    watchlist: list[dict] = []
    for interaction, item in rows:
        if item.item_id in seen:
            continue
        seen.add(item.item_id)
        entry = serialize_item(item, include_description=False)
        entry["added_at"] = interaction.ts.isoformat()
        watchlist.append(entry)
    return watchlist


def get_movie_status(session: Session, user_id: int, item_id: int) -> dict:
    rows = session.execute(
        select(Interaction)
        .where(Interaction.user_id == user_id, Interaction.item_id == item_id)
        .order_by(Interaction.ts.desc(), Interaction.interaction_id.desc())
    ).scalars().all()

    in_library = False
    in_watchlist = False
    dismissed = False
    rating: float | None = None

    for interaction in rows:
        source = (interaction.context_json or {}).get("source")
        if source == LIBRARY_SOURCE and interaction.type == "view":
            in_library = True
        if source == WATCHLIST_SOURCE and interaction.type == "click":
            in_watchlist = True
        if source == DISMISSED_SOURCE and interaction.type == "click":
            dismissed = True
        if interaction.type == "rating" and rating is None and source == LIBRARY_SOURCE:
            raw = (interaction.context_json or {}).get("rating")
            if raw is not None:
                try:
                    rating = float(raw)
                except (TypeError, ValueError):
                    pass

    return {
        "item_id": item_id,
        "in_library": in_library,
        "in_watchlist": in_watchlist,
        "dismissed": dismissed,
        "rating": rating,
    }


def remove_app_interactions(
    session: Session,
    user_id: int,
    item_id: int,
    *,
    source: str,
    interaction_type: str | None = None,
) -> int:
    from sqlalchemy import delete

    conditions = [
        Interaction.user_id == user_id,
        Interaction.item_id == item_id,
        _source_clause(source),
    ]
    if interaction_type is not None:
        conditions.append(Interaction.type == interaction_type)

    result = session.execute(delete(Interaction).where(*conditions))
    session.commit()
    return int(result.rowcount or 0)
