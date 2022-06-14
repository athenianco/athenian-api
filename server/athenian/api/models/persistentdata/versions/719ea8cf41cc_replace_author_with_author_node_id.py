"""Replace author with author_node_id

Revision ID: 719ea8cf41cc
Revises: 390cd86f2ad6
Create Date: 2021-04-16 09:47:18.258525+00:00

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "719ea8cf41cc"
down_revision = "390cd86f2ad6"
branch_labels = None
depends_on = None


def upgrade():
    name = "release_notifications"
    if op.get_bind().dialect.name == "postgresql":
        schema_arg = {"schema": "athenian"}
    else:
        name = "athenian." + name
        schema_arg = {}
    with op.batch_alter_table(name, **schema_arg) as bop:
        bop.alter_column("author", new_column_name="author_node_id")


def downgrade():
    name = "release_notifications"
    if op.get_bind().dialect.name == "postgresql":
        schema_arg = {"schema": "athenian"}
    else:
        name = "athenian." + name
        schema_arg = {}
    with op.batch_alter_table(name, **schema_arg) as bop:
        bop.alter_column("author_node_id", new_column_name="author")
