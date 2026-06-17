"""add val_topup to inat_classification_dataset

Revision ID: f1a2b3c4d5e6
Revises: 528c8d11d272
Create Date: 2026-06-15 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "f1a2b3c4d5e6"
down_revision: Union[str, Sequence[str], None] = "528c8d11d272"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "inat_classification_dataset",
        sa.Column("val_topup", sa.Boolean(), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("inat_classification_dataset", "val_topup")
