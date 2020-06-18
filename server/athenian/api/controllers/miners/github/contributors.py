import asyncio
import marshal
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Collection, List, Optional

import aiomcache
import databases
import sentry_sdk
from sqlalchemy import and_, func, or_, select

from athenian.api.cache import cached
from athenian.api.models.metadata.github import (PullRequest,
                                                 PullRequestComment,
                                                 PullRequestReview, PushCommit,
                                                 Release, User)
from athenian.api.tracing import sentry_span


@sentry_span
async def mine_contributors(repos: Collection[str],
                            time_from: Optional[datetime],
                            time_to: Optional[datetime],
                            db: databases.Database,
                            cache: Optional[aiomcache.Client],
                            with_stats: Optional[bool] = True,
                            as_roles: List[str] = None) -> List[dict]:
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
        ",".join(repos), time_from.timestamp(), time_to.timestamp(), with_stats,
        sorted(as_roles)),
)
async def _mine_contributors(repos: Collection[str],
                             time_from: datetime,
                             time_to: datetime,
                             with_stats: bool,
                             as_roles: List[str],
                             db: databases.Database,
                             cache: Optional[aiomcache.Client]) -> List[dict]:
    assert isinstance(time_from, datetime)
    assert isinstance(time_to, datetime)

    common_prs_where = and_(
        PullRequest.repository_full_name.in_(repos),
        PullRequest.hidden.is_(False),
        or_(PullRequest.created_at.between(time_from, time_to),
            and_(PullRequest.created_at < time_to,
                 PullRequest.closed_at.is_(None)),
            PullRequest.closed_at.between(time_from, time_to),
            PullRequest.updated_at.between(time_from, time_to)),
    )

    @sentry_span
    async def fetch_authors():
        return await db.fetch_all(
            select([PullRequest.user_login, func.count(PullRequest.user_login)])
            .where(common_prs_where)
            .group_by(PullRequest.user_login))

    @sentry_span
    async def fetch_reviewers():
        return await db.fetch_all(
            select([PullRequestReview.user_login, func.count(PullRequestReview.user_login)])
            .where(and_(PullRequestReview.repository_full_name.in_(repos),
                        PullRequestReview.submitted_at.between(time_from, time_to)))
            .group_by(PullRequestReview.user_login))

    @sentry_span
    async def fetch_commit_authors():
        return await db.fetch_all(
            select([PushCommit.author_login, func.count(PushCommit.author_login)])
            .where(and_(PushCommit.repository_full_name.in_(repos),
                        PushCommit.committed_date.between(time_from, time_to)))
            .group_by(PushCommit.author_login))

    @sentry_span
    async def fetch_commit_committers():
        return await db.fetch_all(
            select([PushCommit.committer_login, func.count(PushCommit.committer_login)])
            .where(and_(PushCommit.repository_full_name.in_(repos),
                        PushCommit.committed_date.between(time_from, time_to)))
            .group_by(PushCommit.committer_login))

    @sentry_span
    async def fetch_commenters():
        return await db.fetch_all(
            select([PullRequestComment.user_login, func.count(PullRequestComment.user_login)])
            .where(and_(PullRequestComment.repository_full_name.in_(repos),
                        PullRequestComment.created_at.between(time_from, time_to),
                        ))
            .group_by(PullRequestComment.user_login))

    @sentry_span
    async def fetch_mergers():
        return await db.fetch_all(
            select([PullRequest.merged_by_login, func.count(PullRequest.merged_by_login)])
            .where(common_prs_where)
            .group_by(PullRequest.merged_by_login))

    @sentry_span
    async def fetch_releasers():
        return await db.fetch_all(
            select([Release.author, func.count(Release.author)])
            .where(and_(Release.repository_full_name.in_(repos),
                        Release.published_at.between(time_from, time_to)))
            .group_by(Release.author))

    def strip_stats(contribs):
        for c in contribs:
            c.pop("stats")

    data = await asyncio.gather(
        fetch_authors(), fetch_reviewers(),
        fetch_commit_authors(), fetch_commit_committers(),
        fetch_commenters(), fetch_mergers(), fetch_releasers())

    stats = defaultdict(dict)
    for rows, key in zip(data, ("prs", "reviewer", "commit_author", "commit_committer",
                                "commenter", "merger", "releaser")):
        for row in rows:
            stats[row[0]][key] = row[1]

    stats.pop(None, None)

    cols = [User.login, User.email, User.avatar_url, User.name]
    with sentry_sdk.start_span(op="SELECT FROM github_users_v2_compat"):
        user_details = await db.fetch_all(select(cols).where(User.login.in_(stats.keys())))

    contribs = []
    for ud in user_details:
        c = dict(ud)
        c["stats"] = stats[c[User.login.key]]
        contribs.append(c)

    filtered_contribs = _filter_contributors_by_roles(contribs, as_roles)
    if not with_stats:
        strip_stats(filtered_contribs)

    return filtered_contribs


def _filter_contributors_by_roles(contribs: List[dict], as_roles: List[str]) -> List[dict]:
    if not as_roles:
        return contribs

    filtered_contribs = []
    # We could get rid of these re-mapping, maybe worth looking at it along with the
    # definition of `DeveloperUpdates`
    role_mapping = {"author": "prs"}
    for c in contribs:
        has_active_role = sum(c["stats"].get(role_mapping.get(role, role), 0)
                              for role in as_roles) > 0
        if not has_active_role:
            continue

        filtered_contribs.append(c)

    return filtered_contribs
