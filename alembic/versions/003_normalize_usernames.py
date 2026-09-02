"""Normalize usernames to lowercase and merge case-variant duplicates."""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "003"
down_revision: Union[str, Sequence[str], None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()

    # Preserve original casing as display_name before normalization.
    conn.execute(
        sa.text(
            """
            UPDATE users
            SET profile_json = profile_json || jsonb_build_object('display_name', username)
            WHERE username IS NOT NULL
              AND COALESCE(profile_json->>'display_name', '') = ''
            """
        )
    )

    # Move interactions from duplicate accounts onto the canonical account (lowest id).
    conn.execute(
        sa.text(
            """
            WITH ranked AS (
                SELECT
                    id,
                    MIN(id) OVER (PARTITION BY LOWER(username)) AS keep_id
                FROM users
                WHERE username IS NOT NULL
            )
            UPDATE interactions AS i
            SET user_id = r.keep_id
            FROM ranked AS r
            WHERE i.user_id = r.id
              AND r.id <> r.keep_id
            """
        )
    )

    # Remove duplicate accounts created with different casing.
    conn.execute(
        sa.text(
            """
            DELETE FROM users AS u
            WHERE username IS NOT NULL
              AND u.id NOT IN (
                  SELECT MIN(id)
                  FROM users
                  WHERE username IS NOT NULL
                  GROUP BY LOWER(username)
              )
            """
        )
    )

    conn.execute(
        sa.text(
            """
            UPDATE users
            SET username = LOWER(username)
            WHERE username IS NOT NULL
            """
        )
    )

    op.drop_index("uq_users_username", table_name="users")
    op.create_index(
        "uq_users_username_lower",
        "users",
        [sa.text("lower(username)")],
        unique=True,
        postgresql_where=sa.text("username IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("uq_users_username_lower", table_name="users")
    op.create_index("uq_users_username", "users", ["username"], unique=True)
