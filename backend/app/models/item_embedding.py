from datetime import datetime
from typing import TYPE_CHECKING

from app.db.base import Base
from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

if TYPE_CHECKING:
    from app.models.item import Item

EMBEDDING_DIM = 512


class ItemEmbedding(Base):
    __tablename__ = "item_embeddings"

    item_id: Mapped[int] = mapped_column(ForeignKey("items.item_id"), primary_key=True)
    vector: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    item: Mapped["Item"] = relationship(back_populates="embedding")
