import enum
from datetime import datetime
from typing import TYPE_CHECKING

from app.db.base import Base
from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, Integer, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

if TYPE_CHECKING:
    from app.models.item import Item


class InteractionType(str, enum.Enum):
    VIEW = "view"
    RATING = "rating"
    TAG = "tag"
    CLICK = "click"
    PURCHASE = "purchase"


class Interaction(Base):
    __tablename__ = "interactions"
    __table_args__ = (Index("idx_interactions_user_ts", "user_id", "ts"),)

    interaction_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("items.item_id"), nullable=False, index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    type: Mapped[str] = mapped_column(Text, nullable=False)
    context_json: Mapped[dict] = mapped_column(JSONB, server_default="{}", nullable=False)

    item: Mapped["Item"] = relationship(back_populates="interactions")
