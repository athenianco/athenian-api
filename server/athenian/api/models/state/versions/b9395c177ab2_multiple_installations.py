"""Multiple installations

Revision ID: b9395c177ab2
Revises: d4ad0ed074a5
Create Date: 2020-04-15 16:25:27.492418+00:00

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.orm import Session

# revision identifiers, used by Alembic.

revision = "b9395c177ab2"
down_revision = "d4ad0ed074a5"
branch_labels = None
depends_on = None


def upgrade():
    session = Session(bind=op.get_bind())
    op.create_table(
        "installations",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=False),
        sa.Column(
            "account_id",
            sa.Integer(),
            sa.ForeignKey("accounts.id", name="fk_installation_id_owner"),
            nullable=False,
        ),
    )
    for account in session.execute("select * from accounts"):
        if account.installation_id is not None:
            session.execute(
                "insert into installations(id, account_id) values(%d, %d)"
                % (account.installation_id, account.id)
            )
    session.commit()
    with op.batch_alter_table("accounts") as bop:
        bop.drop_column("installation_id")


def downgrade():
    with op.batch_alter_table("accounts") as bop:
        bop.add_column(sa.Column("installation_id", sa.BigInteger(), nullable=True))
        bop.create_unique_constraint("uq_installation_id", ["installation_id"])
    session = Session(bind=op.get_bind())
    for iid in session.execute("select * from installations"):
        session.execute(
            "update accounts set installation_id = %d where id = %d" % (iid.id, iid.account_id)
        )
    session.commit()
    op.drop_table("installations")
