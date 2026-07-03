"""add user preferred_name + unit_system

Revision ID: c1f3a7b8e2d4
Revises: b09382574363
Create Date: 2026-06-23 13:10:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c1f3a7b8e2d4"
down_revision: Union[str, Sequence[str], None] = "b09382574363"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # App-owned profile fields, independent of the Google claims sync.
    op.add_column(
        "users", sa.Column("preferred_name", sa.String(length=255), nullable=True)
    )
    op.add_column(
        "users",
        sa.Column(
            "unit_system", sa.String(length=10), server_default="metric", nullable=False
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("users", "unit_system")
    op.drop_column("users", "preferred_name")
