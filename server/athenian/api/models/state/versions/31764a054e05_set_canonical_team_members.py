"""Set canonical team members

Revision ID: 31764a054e05
Revises: 916dfb933702
Create Date: 2021-04-15 10:52:02.882010+00:00

"""
import ctypes
from datetime import datetime, timezone
import json

from alembic import op
import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
import xxhash

# revision identifiers, used by Alembic.
revision = "31764a054e05"
down_revision = "916dfb933702"
branch_labels = None
depends_on = None

bots = {
    "allcontributors",
    "atlantis-faire-staging",
    "azure-pipelines",
    "bors",
    "changeset-bot",
    "clubhouse",
    "codeclimate",
    "codecov",
    "codefactor-io",
    "commitlint",
    "cypress",
    "datacamp-inf-selfserve-local",
    "datacamp-inf-selfserve-ops",
    "datacamp-inf-selfserve-prod",
    "datacamp-inf-selfserve-staging",
    "deepcode-ci-bot",
    "dependabot",
    "dependabot-preview",
    "depfu",
    "faire-pr-bot-app",
    "gally-bot",
    "github-actions",
    "gogogithubapp",
    "greenkeeper",
    "guardrails",
    "height",
    "imgbot",
    "jira",
    "linc",
    "linear-app",
    "lingohub",
    "linux-foundation-easycla",
    "locale-translation",
    "mergequeue",
    "netlify",
    "probot-auto-merge",
    "pull-request-badge",
    "release-drafter",
    "release-please",
    "renovate",
    "sentry-internal-tools",
    "sentry-io",
    "slack-trop",
    "slack-trop-test",
    "slash-commands",
    "sonarcloud",
    "sourcelevel-bot",
    "stale",
    "stepsize",
    "swarmia",
    "sync-by-unito",
    "thehub-integration",
    "transifex-integration",
    "trybe-evaluation-feedback",
    "trybe-evaluation-feedback-staging",
    "vercel",
    "whitesource-bolt-for-github",
}


Base = declarative_base()


def make_team_cls(with_checksum: bool):
    class Team(Base):
        """Group of users part of the same team."""

        __tablename__ = "teams"

        id = sa.Column(sa.Integer(), primary_key=True)
        owner_id = sa.Column(sa.Integer(), nullable=False)
        parent_id = sa.Column(sa.Integer(), sa.ForeignKey("teams.id", name="fk_team_parent"))
        name = sa.Column(sa.String(256), nullable=False)
        members = sa.Column(sa.JSON(), nullable=False)
        members_count = sa.Column(sa.Integer(), nullable=False)
        if with_checksum:
            members_checksum = sa.Column(sa.BigInteger())
        created_at = sa.Column(
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            default=lambda: datetime.now(timezone.utc),
            server_default=sa.func.now(),
        )
        updated_at = sa.Column(
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            default=lambda: datetime.now(timezone.utc),
            server_default=sa.func.now(),
            onupdate=lambda ctx: datetime.now(timezone.utc),
        )

    return Team


def upgrade():
    with op.batch_alter_table("teams") as bop:
        bop.drop_column("members_checksum")
    session = Session(bind=op.get_bind())
    for obj in session.query(make_team_cls(False)):
        new_members = []
        dirty = False
        for user in obj.members:
            user = user.rsplit("/", 1)[1]
            if user in bots:
                dirty = True
                new_members.append("github.com/apps/" + user)
            else:
                new_members.append("github.com/" + user)
        if dirty:
            obj.members = new_members
            session.add(obj)
    session.commit()


def downgrade():
    with op.batch_alter_table("teams") as bop:
        bop.add_column(sa.Column("members_checksum", sa.BigInteger()))
    session = Session(bind=op.get_bind())
    for obj in session.query(make_team_cls(True)):
        new_members = ["github.com/" + user.rsplit("/", 1)[1] for user in obj.members]
        obj.members = new_members
        obj.members_checksum = ctypes.c_longlong(
            xxhash.xxh64_intdigest(json.dumps(new_members))
        ).value
        session.add(obj)
    session.commit()
    with op.batch_alter_table("teams") as bop:
        bop.alter_column("members_checksum", nullable=False)
