"""Fill account secret

Revision ID: e85fd22de7fe
Revises: 4b00ea73d30a
Create Date: 2020-10-09 08:16:01.477981+00:00

"""
import base64
from datetime import datetime, timezone
import os
from random import randint
import struct
from typing import Tuple

from alembic import op
import pyffx
import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session

# revision identifiers, used by Alembic.
revision = "e85fd22de7fe"
down_revision = "4b00ea73d30a"
branch_labels = None
depends_on = None


ikey = os.getenv("ATHENIAN_INVITATION_KEY")


class Account(declarative_base()):
    """Group of users, some are admins and some are regular."""

    __tablename__ = "accounts"
    __table_args__ = {"sqlite_autoincrement": True}

    id = sa.Column(sa.Integer(), primary_key=True)
    secret_salt = sa.Column(sa.Integer())
    secret = sa.Column(sa.String(8))
    created_at = sa.Column(
        sa.TIMESTAMP(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=sa.sql.func.now(),
    )


def encode_slug(iid: int, salt: int) -> str:
    """Encode an invitation ID and some extra data to 8 chars."""
    part1 = struct.pack("!H", salt)  # 2 bytes
    part2 = struct.pack("!I", iid)[1:]  # 3 bytes
    binseq = (part1 + part2).hex()  # 5 bytes, 10 hex chars
    e = pyffx.String(ikey.encode(), alphabet="0123456789abcdef", length=len(binseq))
    encseq = e.encrypt(binseq)  # encrypted 5 bytes, 10 hex chars
    finseq = base64.b32encode(bytes.fromhex(encseq)).lower().decode()  # 8 base32 chars
    finseq = finseq.replace("o", "8").replace("l", "9")
    return finseq


def generate_account_secret(account_id: int) -> Tuple[int, str]:
    """Compose the account's salt and secret from its identifier."""
    salt = randint(0, (1 << 16) - 1)  # 0:65535 - 2 bytes
    secret = encode_slug(account_id, salt)
    return salt, secret


def upgrade():
    assert ikey is not None
    admin_backdoor = (1 << 24) - 1
    bind = op.get_bind()
    session = Session(bind=bind)
    for account in session.query(Account):
        if account.secret_salt is None:
            if account.id == admin_backdoor:
                account.secret_salt, account.secret = 0, "0" * 8
            else:
                account.secret_salt, account.secret = generate_account_secret(account.id)
            session.add(account)
    session.commit()
    with op.batch_alter_table("accounts") as bop:
        bop.alter_column("secret_salt", nullable=False)
        bop.alter_column("secret", nullable=False)


def downgrade():
    with op.batch_alter_table("accounts") as bop:
        bop.alter_column("secret_salt", nullable=True)
        bop.alter_column("secret", nullable=True)
