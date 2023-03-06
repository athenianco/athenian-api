import sqlalchemy as sa

from athenian.api.db import DatabaseLike
from athenian.api.models.state.models import Goal, TeamGoal
from athenian.api.tracing import sentry_span


@sentry_span
async def delete_empty_goals(account: int, sdb_conn: DatabaseLike) -> None:
    """Delete all account Goal-s having no more TeamGoal-s assigned."""
    delete_stmt = sa.delete(Goal).where(
        sa.and_(
            Goal.account_id == account,
            sa.not_(sa.exists().where(TeamGoal.goal_id == Goal.id)),
            # inefficient, generates a subquery:
            # Goal.id.not_in(sa.select(TeamGoal.goal_id).distinct()),
        ),
    )
    await sdb_conn.execute(delete_stmt)
