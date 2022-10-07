"""Change PK of release_notifications

Revision ID: f0ae6bfc60b1
Revises: 192e84118ff5
Create Date: 2022-10-07 07:56:06.126673+00:00

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "f0ae6bfc60b1"
down_revision = "192e84118ff5"
branch_labels = None
depends_on = None


def upgrade():
    name = "release_notifications"
    if is_pg := op.get_bind().dialect.name == "postgresql":
        schema_arg = {"schema": "athenian"}
        q = ""
        text = "::text"
    else:
        name = "athenian." + name
        schema_arg = {}
        q = '"'
        text = ""
    op.execute(
        f"""
    UPDATE {q}athenian.release_notifications{q}
    SET name = repository_node_id{text} || '@' || commit_hash_prefix
    WHERE name is null;
    """,
    )
    with op.batch_alter_table(name, **schema_arg) as bop:
        if is_pg:
            bop.drop_constraint("release_notifications_pkey1", type_="primary")
        bop.create_primary_key(
            "pk_release_notifications",
            ["account_id", "repository_node_id", "name"],
        )


def downgrade():
    name = "release_notifications"
    if is_pg := op.get_bind().dialect.name == "postgresql":
        schema_arg = {"schema": "athenian"}
    else:
        name = "athenian." + name
        schema_arg = {}
    with op.batch_alter_table(name, **schema_arg) as bop:
        if is_pg:
            bop.drop_constraint("pk_release_notifications", type_="primary")
        bop.create_primary_key(
            "release_notifications_pkey1",
            ["account_id", "repository_node_id", "commit_hash_prefix"],
        )
