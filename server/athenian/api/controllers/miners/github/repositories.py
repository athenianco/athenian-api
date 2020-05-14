import asyncio
from datetime import datetime
from itertools import chain
import marshal
from typing import Collection, List, Optional

import aiomcache
import databases
from sqlalchemy import and_, distinct, or_, select

from athenian.api.cache import cached
from athenian.api.models.metadata import PREFIXES
from athenian.api.models.metadata.github import PullRequest, PullRequestComment, \
    PullRequestReview, PushCommit, Release, Repository


@cached(
    exptime=5 * 60,
    serialize=marshal.dumps,
    deserialize=marshal.loads,
    key=lambda repos, time_from, time_to, **_: (
        ",".join(repos), time_from.timestamp(), time_to.timestamp()),
)
async def mine_repositories(repos: Collection[str],
                            time_from: datetime,
                            time_to: datetime,
                            db: databases.Database,
                            cache: Optional[aiomcache.Client],
                            ) -> List[str]:
    """Discover repositories from the given set which were updated in the given time frame."""
    assert isinstance(time_from, datetime)
    assert isinstance(time_to, datetime)

    async def fetch_prs():
        return await db.fetch_all(
            select([distinct(PullRequest.repository_full_name)])
            .where(and_(PullRequest.repository_full_name.in_(repos),
                        PullRequest.hidden.is_(False),
                        or_(PullRequest.created_at.between(time_from, time_to),
                            and_(PullRequest.created_at < time_to,
                                 PullRequest.closed_at.is_(None)),
                            PullRequest.closed_at.between(time_from, time_to),
                            PullRequest.updated_at.between(time_from, time_to)))))

    async def fetch_comments():
        return await db.fetch_all(
            select([distinct(PullRequestComment.repository_full_name)])
            .where(and_(PullRequestComment.repository_full_name.in_(repos),
                        PullRequestComment.created_at.between(time_from, time_to),
                        )))

    async def fetch_push_commits():
        return await db.fetch_all(
            select([distinct(PushCommit.repository_full_name)])
            .where(and_(PushCommit.repository_full_name.in_(repos),
                        PushCommit.committed_date.between(time_from, time_to),
                        )))

    async def fetch_reviews():
        return await db.fetch_all(
            select([distinct(PullRequestReview.repository_full_name)])
            .where(and_(PullRequestReview.repository_full_name.in_(repos),
                        PullRequestReview.submitted_at.between(time_from, time_to),
                        )))

    async def fetch_releases():
        return await db.fetch_all(
            select([distinct(Release.repository_full_name)])
            .where(and_(Release.repository_full_name.in_(repos),
                        Release.published_at.between(time_from, time_to),
                        )))

    repos = set(r[0] for r in chain.from_iterable(await asyncio.gather(
        fetch_prs(), fetch_comments(), fetch_push_commits(), fetch_reviews(), fetch_releases())))
    repos = await db.fetch_all(select([Repository.full_name])
                               .where(and_(Repository.archived.is_(False),
                                           Repository.disabled.is_(False),
                                           Repository.full_name.in_(repos)))
                               .order_by(Repository.full_name))
    prefix = PREFIXES["github"]
    repos = [prefix + r[0] for r in repos]
    return repos
