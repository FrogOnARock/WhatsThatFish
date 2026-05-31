"""add inat_obj_detection_dataset

Revision ID: b4f7e2a9c3d1
Revises: 7292239f728e
Create Date: 2026-05-30

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'b4f7e2a9c3d1'
down_revision: Union[str, Sequence[str], None] = '7292239f728e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'inat_obj_detection_dataset',
        sa.Column('photo_uuid', sa.String(36), sa.ForeignKey('inat_filtered_observations.photo_uuid'), primary_key=True),
        sa.Column('filename', sa.String(), nullable=True),
        sa.Column('uiqm', sa.Float(), nullable=True),
        sa.Column('train', sa.Boolean(), nullable=True),
        sa.Column('proposed_bbox', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('conf', sa.Float(), nullable=True),
        sa.Column('annotation', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.create_index('ix_inat_obj_detection_dataset_train', 'inat_obj_detection_dataset', ['train'])
    op.create_index('ix_inat_obj_detection_dataset_uiqm', 'inat_obj_detection_dataset', ['uiqm'])


def downgrade() -> None:
    op.drop_index('ix_inat_obj_detection_dataset_uiqm', table_name='inat_obj_detection_dataset')
    op.drop_index('ix_inat_obj_detection_dataset_train', table_name='inat_obj_detection_dataset')
    op.drop_table('inat_obj_detection_dataset')
