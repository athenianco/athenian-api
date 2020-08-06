import asyncio
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import marshal
from typing import Any, Collection, Dict, List, Optional

import aiomcache
import databases
import sentry_sdk
from sqlalchemy import and_, func, or_, select

from athenian.api.cache import cached
from athenian.api.models.metadata.github import (PullRequest, PullRequestComment,
                                                 PullRequestReview, PushCommit, Release, User)
from athenian.api.tracing import sentry_span


@sentry_span
async def mine_contributors(repos: Collection[str],
                            time_from: Optional[datetime],
                            time_to: Optional[datetime],
                            db: databases.Database,
                            cache: Optional[aiomcache.Client],
                            with_stats: Optional[bool] = True,
                            as_roles: List[str] = None) -> List[Dict[str, Any]]:
    """Discover developers who made any important action in the given repositories and \
    in the given time frame."""
    time_from = time_from or datetime(1970, 1, 1, tzinfo=timezone.utc)
    now = datetime.now(timezone.utc) + timedelta(days=1)
    time_to = time_to or datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    as_roles = as_roles or []

    return await _mine_contributors(repos, time_from, time_to, with_stats, as_roles, db, cache)


@sentry_span
@cached(
    exptime=5 * 60,
    serialize=marshal.dumps,
    deserialize=marshal.loads,
    key=lambda repos, time_from, time_to, with_stats, as_roles, **_: (
        ",".join(sorted(repos)), time_from.timestamp(), time_to.timestamp(), with_stats,
        sorted(as_roles)),
)
async def _mine_contributors(repos: Collection[str],
                             time_from: datetime,
                             time_to: datetime,
                             with_stats: bool,
                             as_roles: List[str],
                             db: databases.Database,
                             cache: Optional[aiomcache.Client]) -> List[Dict[str, Any]]:
    assert isinstance(time_from, datetime)
    assert isinstance(time_to, datetime)

    common_prs_where = and_(
        PullRequest.repository_full_name.in_(repos),
        PullRequest.hidden.is_(False),
        or_(PullRequest.created_at.between(time_from, time_to),
            and_(PullRequest.created_at < time_to,
                 PullRequest.closed_at.is_(None)),
            PullRequest.closed_at.between(time_from, time_to)),
    )

    @sentry_span
    async def fetch_author():
        # TODO(vmarkovtsev): load released PRs from the pdb
        return await db.fetch_all(
            select([PullRequest.user_login, func.count(PullRequest.user_login)])
            .where(common_prs_where)
            .group_by(PullRequest.user_login))

    @sentry_span
    async def fetch_reviewer():
        return await db.fetch_all(
            select([PullRequestReview.user_login, func.count(PullRequestReview.user_login)])
            .where(and_(PullRequestReview.repository_full_name.in_(repos),
                        PullRequestReview.submitted_at.between(time_from, time_to)))
            .group_by(PullRequestReview.user_login))

    @sentry_span
    async def fetch_commit_author():
        return await db.fetch_all(
            select([PushCommit.author_login, func.count(PushCommit.author_login)])
            .where(and_(PushCommit.repository_full_name.in_(repos),
                        PushCommit.committed_date.between(time_from, time_to)))
            .group_by(PushCommit.author_login))

    @sentry_span
    async def fetch_commit_committer():
        return await db.fetch_all(
            select([PushCommit.committer_login, func.count(PushCommit.committer_login)])
            .where(and_(PushCommit.repository_full_name.in_(repos),
                        PushCommit.committed_date.between(time_from, time_to)))
            .group_by(PushCommit.committer_login))

    @sentry_span
    async def fetch_commenter():
        return await db.fetch_all(
            select([PullRequestComment.user_login, func.count(PullRequestComment.user_login)])
            .where(and_(PullRequestComment.repository_full_name.in_(repos),
                        PullRequestComment.created_at.between(time_from, time_to),
                        ))
            .group_by(PullRequestComment.user_login))

    @sentry_span
    async def fetch_merger():
        return await db.fetch_all(
            select([PullRequest.merged_by_login, func.count(PullRequest.merged_by_login)])
            .where(common_prs_where)
            .group_by(PullRequest.merged_by_login))

    @sentry_span
    async def fetch_releaser():
        return await db.fetch_all(
            select([Release.author, func.count(Release.author)])
            .where(and_(Release.repository_full_name.in_(repos),
                        Release.published_at.between(time_from, time_to)))
            .group_by(Release.author))

    fetchers_mapping = {
        "author": fetch_author,
        "reviewer": fetch_reviewer,
        "commit_author": fetch_commit_author,
        "commit_committer": fetch_commit_committer,
        "commenter": fetch_commenter,
        "merger": fetch_merger,
        "releaser": fetch_releaser,
    }

    as_roles = as_roles or fetchers_mapping.keys()
    tasks = {k: v() for k, v in fetchers_mapping.items() if k in as_roles}
    data = await asyncio.gather(*tasks.values(), return_exceptions=True)
    stats = defaultdict(dict)
    for r, key in zip(data, tasks.keys()):
        if isinstance(r, Exception):
            raise r from None

        for row in r:
            stats[row[0]][key] = row[1]

    stats.pop(None, None)

    cols = [User.login, User.email, User.avatar_url, User.name]
    with sentry_sdk.start_span(op="SELECT FROM github_users_v2_compat"):
        user_details = await db.fetch_all(select(cols).where(User.login.in_(stats.keys())))

    contribs = []
    for ud in user_details:
        c = dict(ud)
        c["stats"] = stats[c[User.login.key]]
        if as_roles and sum(c["stats"].get(role, 0) for role in as_roles) == 0:
            continue

        if "author" in c["stats"]:
            # We could get rid of these re-mapping, maybe worth looking at it along with the
            # definition of `DeveloperUpdates`
            c["stats"]["prs"] = c["stats"].pop("author")

        if not with_stats:
            c.pop("stats")

        contribs.append(c)

    return contribs
