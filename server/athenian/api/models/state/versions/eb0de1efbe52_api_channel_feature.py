"""API channel feature

Revision ID: eb0de1efbe52
Revises: de3819e9b84d
Create Date: 2021-11-17 13:16:59.991522+00:00

"""
from alembic import op

# revision identifiers, used by Alembic.

revision = "eb0de1efbe52"
down_revision = "de3819e9b84d"
branch_labels = None
depends_on = None


def upgrade():
    if op.get_bind().dialect == "sqlite":
        op.execute("insert or ignore into features(name, component, enabled, default_parameters) "
                   "values ('api_channel', 'webapp', true, '\"stable\"');")
    else:
        op.execute("insert into features(name, component, enabled, default_parameters) "
                   "values ('api_channel', 'webapp', true, '\"stable\"') "
                   "on conflict do nothing;")


def downgrade():
    op.execute("delete from features where name = 'api_channel' and component = 'webapp';")
