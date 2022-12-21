import argparse
from collections import defaultdict
import sys

import pandas as pd
from sqlalchemy import and_, create_engine
from sqlalchemy.orm import Session, load_only, sessionmaker
from tqdm import tqdm

from athenian.api.internal.jira import ALLOWED_USER_TYPES
from athenian.api.models.metadata.github import OrganizationMember, PushCommit, User as GitHubUser
from athenian.api.models.metadata.jira import User as JIRAUser
from athenian.api.models.state.models import AccountGitHubAccount, AccountJiraInstallation
from athenian.api.slogging import add_logging_args


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    add_logging_args(parser)
    parser.add_argument(
        "--metadata-db",
        default="postgresql://postgres:postgres@0.0.0.0:5432/metadata",
        help="Metadata (GitHub, JIRA, etc.) DB connection string in SQLAlchemy format.",
    )
    parser.add_argument(
        "--state-db",
        default="postgresql://postgres:postgres@0.0.0.0:5432/state",
        help=(
            "Server state (user settings, teams, etc.) DB connection string in SQLAlchemy format."
        ),
    )
    parser.add_argument("--account", type=int, help="Account number in state DB.")
    return parser.parse_args()


def main():
    """Compose and store github.pickle and jira.pickle."""
    args = parse_args()
    bar = tqdm(total=11)
    engine = create_engine(args.metadata_db)
    mdb: Session = sessionmaker(bind=engine)()
    bar.update(1)
    engine = create_engine(args.state_db)
    sdb: Session = sessionmaker(bind=engine)()
    bar.update(1)
    meta_id = (
        sdb.query(AccountGitHubAccount.id)
        .filter(AccountGitHubAccount.account_id == args.account)
        .scalar()
    )
    bar.update(1)
    user_ids = [
        r[0]
        for r in mdb.query(OrganizationMember.child_id)
        .filter(OrganizationMember.acc_id == meta_id)
        .all()
    ]
    bar.update(1)
    users = (
        mdb.query(GitHubUser)
        .filter(and_(GitHubUser.acc_id == meta_id, GitHubUser.node_id.in_(user_ids)))
        .all()
    )
    bar.update(1)
    authors = (
        mdb.query(PushCommit)
        .options(
            load_only(PushCommit.author_user_id, PushCommit.author_name, PushCommit.author_email),
        )
        .filter(and_(PushCommit.acc_id == meta_id, PushCommit.author_user_id.in_(user_ids)))
        .distinct()
        .all()
    )
    bar.update(1)
    committers = (
        mdb.query(PushCommit)
        .options(
            load_only(
                PushCommit.committer_user_id,
                PushCommit.committer_name,
                PushCommit.committer_email,
            ),
        )
        .filter(and_(PushCommit.acc_id == meta_id, PushCommit.committer_user_id.in_(user_ids)))
        .distinct()
        .all()
    )
    bar.update(1)
    signatures = defaultdict(set)
    for commit in committers:
        signatures[commit.committer_user].add((commit.committer_name, commit.committer_email))
    for commit in authors:
        signatures[commit.author_user].add((commit.author_name, commit.author_email))
    df = pd.DataFrame(
        [
            {
                "name": u.name,
                "login": u.login,
                "email": u.email,
                "signatures": signatures.get(u.node_id),
            }
            for u in users
        ],
    )
    df.to_pickle("github_%d.pickle" % args.account)
    bar.update(1)
    jira_id = (
        sdb.query(AccountJiraInstallation.id)
        .filter(AccountJiraInstallation.account_id == args.account)
        .scalar()
    )
    bar.update(1)
    jira_users = mdb.query(JIRAUser).filter(
        and_(JIRAUser.acc_id == jira_id, JIRAUser.type.in_(ALLOWED_USER_TYPES)),
    )
    bar.update(1)
    df = pd.DataFrame(
        [
            {
                "name": u.display_name,
            }
            for u in jira_users
        ],
    )
    df.to_pickle("jira_%d.pickle" % args.account)
    bar.update(1)
    bar.close()


if __name__ == "__main__":
    sys.exit(main())
