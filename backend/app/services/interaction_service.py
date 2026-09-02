"""Log user interactions and invalidate recommendation caches."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.interaction import Interaction, InteractionType
from app.models.item import Item
from app.services.user_cache import UserCache


VALID_INTERACTION_TYPES = {item.value for item in InteractionType}


def log_interaction(
    session: Session,
    *,
    user_id: int,
    item_id: int,
    interaction_type: str,
    context_json: dict | None = None,
    user_cache: UserCache | None = None,
) -> Interaction:
    if interaction_type not in VALID_INTERACTION_TYPES:
        raise ValueError(
            f"Invalid interaction type '{interaction_type}'. "
            f"Must be one of: {', '.join(sorted(VALID_INTERACTION_TYPES))}"
        )

    item_exists = session.execute(
        select(Item.item_id).where(Item.item_id == item_id)
    ).scalar_one_or_none()
    if item_exists is None:
        raise LookupError(f"Item {item_id} not found")

    interaction = Interaction(
        user_id=user_id,
        item_id=item_id,
        ts=datetime.now(UTC),
        type=interaction_type,
        context_json=context_json or {},
    )
    session.add(interaction)
    session.commit()
    session.refresh(interaction)

    if user_cache is not None:
        user_cache.invalidate(user_id)
        user_cache.invalidate(1_000_000_000 + user_id)

    return interaction
