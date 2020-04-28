import asyncio
from datetime import datetime
from itertools import chain
import marshal
from typing import Collection, List, Optional

import aiomcache
import databases
from sqlalchemy import and_, distinct, or_, select

from athenian.api.cache import cached
from athenian.api.models.metadata.github import PullRequest, PullRequestComment, \
    PullRequestReview, PushCommit, Release


@cached(
    exptime=5 * 60,
    serialize=marshal.dumps,
    deserialize=marshal.loads,
    key=lambda repos, date_from, date_to, **_: (
        ",".join(repos), date_from.timestamp(), date_to.timestamp()),
)
async def mine_repositories(repos: Collection[str],
                            date_from: datetime,
                            date_to: datetime,
                            db: databases.Database,
                            cache: Optional[aiomcache.Client],
                            ) -> List[str]:
    """Discover repositories from the given set which were updated in the given time frame."""
    assert isinstance(date_from, datetime)
    assert isinstance(date_to, datetime)

    async def fetch_prs():
        return await db.fetch_all(
            select([distinct(PullRequest.repository_full_name)])
            .where(and_(PullRequest.repository_full_name.in_(repos), or_(
                        PullRequest.created_at.between(date_from, date_to),
                        PullRequest.closed_at.between(date_from, date_to),
                        PullRequest.updated_at.between(date_from, date_to),
                        ))))

    async def fetch_comments():
        return await db.fetch_all(
            select([distinct(PullRequestComment.repository_full_name)])
            .where(and_(PullRequestComment.repository_full_name.in_(repos),
                        PullRequestComment.created_at.between(date_from, date_to),
                        )))

    async def fetch_push_commits():
        return await db.fetch_all(
            select([distinct(PushCommit.repository_full_name)])
            .where(and_(PushCommit.repository_full_name.in_(repos),
                        PushCommit.committed_date.between(date_from, date_to),
                        )))

    async def fetch_reviews():
        return await db.fetch_all(
            select([distinct(PullRequestReview.repository_full_name)])
            .where(and_(PullRequestReview.repository_full_name.in_(repos),
                        PullRequestReview.submitted_at.between(date_from, date_to),
                        )))

    async def fetch_releases():
        return await db.fetch_all(
            select([distinct(Release.repository_full_name)])
            .where(and_(Release.repository_full_name.in_(repos),
                        Release.published_at.between(date_from, date_to),
                        )))

    repos = sorted({"github.com/" + r[0] for r in chain.from_iterable(await asyncio.gather(
        fetch_prs(), fetch_comments(), fetch_push_commits(), fetch_reviews(),
        fetch_releases()))})
    return repos
