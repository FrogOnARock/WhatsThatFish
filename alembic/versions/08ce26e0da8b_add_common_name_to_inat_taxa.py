"""add common_name to inat_taxa

Revision ID: 08ce26e0da8b
Revises: c1f3a7b8e2d4
Create Date: 2026-07-03 13:26:34.314650

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "08ce26e0da8b"
down_revision: Union[str, Sequence[str], None] = "c1f3a7b8e2d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Additive + nullable → existing rows backfill to NULL, no table rewrite,
    # no lock beyond a fast catalog update. Safe to run on Cloud SQL ahead of
    # the code that reads it (migrate-before-deploy).
    op.add_column(
        "inat_taxa",
        sa.Column("common_name", sa.String(length=255), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("inat_taxa", "common_name")
