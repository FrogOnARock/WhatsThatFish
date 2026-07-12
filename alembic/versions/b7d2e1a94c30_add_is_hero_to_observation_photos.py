"""add is_hero to observation_photos

Revision ID: b7d2e1a94c30
Revises: 08ce26e0da8b
Create Date: 2026-07-12 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "b7d2e1a94c30"
down_revision: Union[str, Sequence[str], None] = "08ce26e0da8b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Additive, NOT NULL with a server default → existing rows backfill to
    # false in a single fast catalog update (no per-row rewrite needed since PG
    # 11 for a constant default). Marks a photo as the user's chosen card image
    # for its effective species; at most one hero per (user, corrected_taxon_id),
    # enforced in the service, not by a DB constraint.
    op.add_column(
        "observation_photos",
        sa.Column(
            "is_hero",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("observation_photos", "is_hero")
