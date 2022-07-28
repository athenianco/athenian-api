"""Use JSONB

Revision ID: 58582455884b
Revises: 9172948308c9
Create Date: 2022-08-01 09:59:09.419832+00:00

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "58582455884b"
down_revision = "9172948308c9"
branch_labels = None
depends_on = None


def upgrade():
    if op.get_bind().dialect.name != "postgresql":
        return
    with op.batch_alter_table("account_features") as bop:
        bop.alter_column(
            "parameters",
            existing_type=sa.JSON(),
            type_=postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
            existing_nullable=True,
        )
    with op.batch_alter_table("features") as bop:
        bop.alter_column(
            "default_parameters",
            existing_type=sa.JSON(),
            type_=postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
            existing_nullable=False,
            existing_server_default=sa.text("'{}'::json"),
        )
    with op.batch_alter_table("logical_repositories") as bop:
        bop.alter_column(
            "prs",
            existing_type=postgresql.JSON(astext_type=sa.Text()),
            type_=postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
            existing_nullable=False,
            existing_server_default=sa.text("'{}'::json"),
        )
        bop.alter_column(
            "deployments",
            existing_type=postgresql.JSON(astext_type=sa.Text()),
            type_=postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
            existing_nullable=False,
            existing_server_default=sa.text("'{}'::json"),
        )
    with op.batch_alter_table("repository_sets") as bop:
        bop.alter_column(
            "items",
            existing_type=sa.JSON(),
            type_=postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
            existing_nullable=False,
        )
    with op.batch_alter_table("shares") as bop:
        bop.alter_column(
            "id",
            existing_type=sa.INTEGER(),
            type_=sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
            existing_nullable=False,
            autoincrement=True,
        )
    with op.batch_alter_table("team_goals") as bop:
        bop.alter_column(
            "target",
            existing_type=sa.JSON(),
            type_=postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
            existing_nullable=False,
        )
    with op.batch_alter_table("teams") as bop:
        bop.alter_column(
            "members",
            existing_type=sa.JSON(),
            type_=postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
            existing_nullable=False,
        )
    with op.batch_alter_table("work_types") as bop:
        bop.alter_column(
            "rules",
            existing_type=postgresql.JSON(astext_type=sa.Text()),
            type_=postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"),
            existing_nullable=False,
        )


def downgrade():
    if op.get_bind().dialect.name != "postgresql":
        return
    with op.batch_alter_table("work_types") as bop:
        bop.alter_column(
            "rules",
            existing_type=postgresql.JSONB(astext_type=sa.Text()).with_variant(
                sa.JSON(), "sqlite",
            ),
            type_=postgresql.JSON(astext_type=sa.Text()),
            existing_nullable=False,
        )

    with op.batch_alter_table("teams") as bop:
        bop.alter_column(
            "members",
            existing_type=postgresql.JSONB(astext_type=sa.Text()).with_variant(
                sa.JSON(), "sqlite",
            ),
            type_=sa.JSON(),
            existing_nullable=False,
        )
    with op.batch_alter_table("team_goals") as bop:
        bop.alter_column(
            "target",
            existing_type=postgresql.JSONB(astext_type=sa.Text()).with_variant(
                sa.JSON(), "sqlite",
            ),
            type_=sa.JSON(),
            existing_nullable=False,
        )
    with op.batch_alter_table("shares") as bop:
        bop.alter_column(
            "id",
            existing_type=sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
            type_=sa.INTEGER(),
            existing_nullable=False,
            autoincrement=True,
        )
    with op.batch_alter_table("repository_sets") as bop:
        bop.alter_column(
            "items",
            existing_type=postgresql.JSONB(astext_type=sa.Text()).with_variant(
                sa.JSON(), "sqlite",
            ),
            type_=sa.JSON(),
            existing_nullable=False,
        )
    with op.batch_alter_table("logical_repositories") as bop:
        bop.alter_column(
            "deployments",
            existing_type=postgresql.JSONB(astext_type=sa.Text()).with_variant(
                sa.JSON(), "sqlite",
            ),
            type_=postgresql.JSON(astext_type=sa.Text()),
            existing_nullable=False,
            existing_server_default=sa.text("'{}'::json"),
        )
        bop.alter_column(
            "prs",
            existing_type=postgresql.JSONB(astext_type=sa.Text()).with_variant(
                sa.JSON(), "sqlite",
            ),
            type_=postgresql.JSON(astext_type=sa.Text()),
            existing_nullable=False,
            existing_server_default=sa.text("'{}'::json"),
        )
    with op.batch_alter_table("features") as bop:
        bop.alter_column(
            "default_parameters",
            existing_type=postgresql.JSONB(astext_type=sa.Text()).with_variant(
                sa.JSON(), "sqlite",
            ),
            type_=sa.JSON(),
            existing_nullable=False,
            existing_server_default=sa.text("'{}'::json"),
        )
    with op.batch_alter_table("account_features") as bop:
        bop.alter_column(
            "parameters",
            existing_type=postgresql.JSONB(astext_type=sa.Text()).with_variant(
                sa.JSON(), "sqlite",
            ),
            type_=sa.JSON(),
            existing_nullable=True,
        )
