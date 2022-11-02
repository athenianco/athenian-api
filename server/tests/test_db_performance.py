from datetime import datetime

from morcilla.backends.asyncpg import PostgresConnection
from sqlalchemy import and_, case, exists, func, select, text

from athenian.api.models.metadata.github import (
    NodePullRequest,
    NodePullRequestCommit,
    PullRequest,
    PullRequestLabel,
)


def test_compile_performance_benchmark(benchmark):
    benchmark(_benchmark, PostgresConnection(None, None))


def _benchmark(conn: PostgresConnection) -> None:
    pr_node_ids = [1102354] * 1000
    conn._compile(
        select(
            [
                NodePullRequest.id,
                NodePullRequest.author_id,
                NodePullRequest.merged,
                NodePullRequest.created_at,
                NodePullRequest.closed_at,
            ],
        ).where(
            and_(NodePullRequest.acc_id.in_([1]), NodePullRequest.id.in_any_values(pr_node_ids)),
        ),
    )
    conn._compile(
        select(
            [
                NodePullRequestCommit.pull_request_id,
                func.count(NodePullRequestCommit.commit_id).label("count"),
            ],
        )
        .where(
            and_(
                NodePullRequestCommit.acc_id.in_([1]),
                NodePullRequestCommit.pull_request_id.in_any_values(pr_node_ids),
            ),
        )
        .group_by(NodePullRequestCommit.pull_request_id),
    )

    filters = [
        case(
            [(PullRequest.closed, PullRequest.closed_at)],
            else_=text("'3000-01-01'"),
        )
        >= datetime(2020, 1, 1),
        PullRequest.created_at < datetime(2022, 1, 1),
        PullRequest.acc_id.in_([1, 2]),
        PullRequest.repository_full_name.in_(["athenianco/athenian-api"] * 500),
        PullRequest.updated_at >= datetime(2020, 1, 1),
        PullRequest.updated_at.between(datetime(2021, 1, 1), datetime(2022, 1, 1)),
        PullRequest.node_id.notin_(pr_node_ids),
        PullRequest.user_login.in_(["vadim"] * 200),
        exists().where(
            and_(
                PullRequestLabel.acc_id == PullRequest.acc_id,
                PullRequestLabel.pull_request_node_id == PullRequest.node_id,
                func.lower(PullRequestLabel.name).in_(["one", "two", "three"]),
            ),
        ),
    ]
    conn._compile(select([PullRequest]).where(and_(*filters)))
