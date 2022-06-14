"""Change PK of deployment_notifications

Revision ID: 876b48cd3813
Revises: 5aa9506f1374
Create Date: 2021-07-01 13:10:34.345730+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "876b48cd3813"
down_revision = "5aa9506f1374"
branch_labels = None
depends_on = None


def _reset():
    name = "deployment_notifications"
    if op.get_bind().dialect.name == "postgresql":
        schema_arg = {"schema": "athenian"}
    else:
        name = "athenian." + name
        schema_arg = {}
    op.drop_table(name.replace("deployment_notifications", "deployed_components"), **schema_arg)
    op.drop_table(name.replace("deployment_notifications", "deployed_labels"), **schema_arg)
    op.drop_table(name, **schema_arg)


def upgrade():
    _reset()
    name = "deployment_notifications"
    if op.get_bind().dialect.name == "postgresql":
        schema_arg = {"schema": "athenian"}
        schema = "athenian."
    else:
        name = "athenian." + name
        schema_arg = {}
        schema = ""
    op.create_table(
        name,
        sa.Column("account_id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.Text(), primary_key=True, nullable=False),
        sa.Column("conclusion", sa.Text()),
        sa.Column("environment", sa.Text(), nullable=False),
        sa.Column("url", sa.Text()),
        sa.Column("started_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("finished_at", sa.TIMESTAMP(timezone=True)),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        **schema_arg,
    )
    op.create_table(
        name.replace("deployment_notifications", "deployed_labels"),
        sa.Column("account_id", sa.Integer(), primary_key=True),
        sa.Column("deployment_name", sa.Text(), primary_key=True),
        sa.Column("key", sa.Text(), primary_key=True),
        sa.Column("value", sa.JSON()),
        sa.ForeignKeyConstraint(
            ("account_id", "deployment_name"),
            (f"{schema}{name}.account_id", f"{schema}{name}.name"),
            name="fk_deployed_labels_deployment",
        ),
        **schema_arg,
    )
    op.create_table(
        name.replace("deployment_notifications", "deployed_components"),
        sa.Column("account_id", sa.Integer(), primary_key=True),
        sa.Column("deployment_name", sa.Text(), primary_key=True),
        sa.Column("repository_node_id", sa.Text(), primary_key=True),
        sa.Column("reference", sa.Text(), primary_key=True),
        sa.Column("resolved_commit_node_id", sa.Text()),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(
            ("account_id", "deployment_name"),
            (f"{schema}{name}.account_id", f"{schema}{name}.name"),
            name="fk_deployed_components_deployment",
        ),
        **schema_arg,
    )


def downgrade():
    _reset()
    name = "deployment_notifications"
    if op.get_bind().dialect.name == "postgresql":
        schema_arg = {"schema": "athenian"}
        schema = "athenian."
    else:
        name = "athenian." + name
        schema_arg = {}
        schema = ""
    op.create_table(
        name,
        sa.Column("account_id", sa.Integer(), primary_key=True),
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("conclusion", sa.Text()),
        sa.Column("environment", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("url", sa.Text()),
        sa.Column("started_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("finished_at", sa.TIMESTAMP(timezone=True)),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        **schema_arg,
    )
    op.create_table(
        name.replace("deployment_notifications", "deployed_labels"),
        sa.Column("account_id", sa.Integer(), primary_key=True),
        sa.Column("deployment_id", sa.BigInteger(), primary_key=True),
        sa.Column("key", sa.Text(), primary_key=True),
        sa.Column("value", sa.Text()),
        sa.ForeignKeyConstraint(
            ("account_id", "deployment_id"),
            (f"{schema}{name}.account_id", f"{schema}{name}.id"),
            name="fk_deployed_labels_deployment",
        ),
        **schema_arg,
    )
    op.create_table(
        name.replace("deployment_notifications", "deployed_components"),
        sa.Column("account_id", sa.Integer(), primary_key=True),
        sa.Column("deployment_id", sa.BigInteger(), primary_key=True),
        sa.Column("repository_node_id", sa.Text(), primary_key=True),
        sa.Column("reference", sa.Text(), primary_key=True),
        sa.Column("resolved_commit_node_id", sa.Text()),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(
            ("account_id", "deployment_id"),
            (f"{schema}{name}.account_id", f"{schema}{name}.id"),
            name="fk_deployed_components_deployment",
        ),
        **schema_arg,
    )
