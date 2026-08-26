from typing import Sequence, Union

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "001"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "signup_ts",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "profile_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "items",
        sa.Column("item_id", sa.Integer(), nullable=False),
        sa.Column("imdb_id", sa.Text(), nullable=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("genres", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("image_url", sa.Text(), nullable=True),
        sa.Column(
            "metadata_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("item_id"),
    )

    op.create_table(
        "interactions",
        sa.Column("interaction_id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("item_id", sa.Integer(), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("type", sa.Text(), nullable=False),
        sa.Column(
            "context_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["item_id"], ["items.item_id"]),
        sa.PrimaryKeyConstraint("interaction_id"),
        sa.CheckConstraint(
            "type IN ('view', 'rating', 'tag', 'click', 'purchase')",
            name="ck_interactions_type",
        ),
    )
    op.create_index("idx_interactions_user_ts", "interactions", ["user_id", "ts"], unique=False)
    op.create_index(op.f("ix_interactions_item_id"), "interactions", ["item_id"], unique=False)
    op.create_index(op.f("ix_interactions_user_id"), "interactions", ["user_id"], unique=False)

    op.create_table(
        "item_embeddings",
        sa.Column("item_id", sa.Integer(), nullable=False),
        sa.Column("vector", Vector(512), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["item_id"], ["items.item_id"]),
        sa.PrimaryKeyConstraint("item_id"),
    )
    op.create_index(
        "idx_item_embeddings_updated", "item_embeddings", ["updated_at"], unique=False
    )


def downgrade() -> None:
    op.drop_index("idx_item_embeddings_updated", table_name="item_embeddings")
    op.drop_table("item_embeddings")
    op.drop_index(op.f("ix_interactions_user_id"), table_name="interactions")
    op.drop_index(op.f("ix_interactions_item_id"), table_name="interactions")
    op.drop_index("idx_interactions_user_ts", table_name="interactions")
    op.drop_table("interactions")
    op.drop_table("items")
    op.drop_table("users")
