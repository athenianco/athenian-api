import asyncio
from itertools import chain
from typing import List, Union

from aiohttp import web
import databases.core
from sqlalchemy import and_, distinct, or_, select

from athenian.api import FriendlyJson, ResponseError
from athenian.api.controllers.reposet import resolve_reposet
from athenian.api.models.metadata.github import PullRequest, PullRequestCommit, \
    PullRequestReview, PushCommit
from athenian.api.models.state.models import RepositorySet, UserAccount
from athenian.api.models.web import ForbiddenError
from athenian.api.models.web.filter_items_request import FilterItemsRequest
from athenian.api.request import AthenianWebRequest


async def filter_contributors(request: AthenianWebRequest, body: dict) -> web.Response:
    """Find developers that made an action within the given timeframe."""
    filt = FilterItemsRequest.from_dict(body)
    try:
        repos = await _resolve_repos(request.sdb, filt, request.uid)
    except ResponseError as e:
        return e.response
    async with request.mdb.connection() as conn:
        u_pr = conn.fetch_all(
            select([distinct(PullRequest.user_login)])
            .where(and_(PullRequest.repository_fullname.in_(repos), or_(
                        PullRequest.created_at.between(filt.date_from, filt.date_to),
                        PullRequest.closed_at.between(filt.date_from, filt.date_to),
                        PullRequest.updated_at.between(filt.date_from, filt.date_to),
                        ))))
        u_pr_commits = conn.fetch_all(
            select([PullRequestCommit.author_login, PullRequestCommit.commiter_login])
            .where(and_(PullRequestCommit.repository_fullname.in_(repos),
                        PullRequestCommit.commit_date.between(filt.date_from, filt.date_to),
                        ))
            .distinct(PullRequestCommit.author_login, PullRequestCommit.commiter_login))
        u_push_commits = conn.fetch_all(
            select([PushCommit.author_login, PushCommit.committer_login])
            .where(and_(PushCommit.repository_fullname.in_(repos),
                        PushCommit.timestamp.between(filt.date_from, filt.date_to),
                        ))
            .distinct(PushCommit.author_login, PushCommit.committer_login))
        u_reviews = conn.fetch_all(
            select([distinct(PullRequestReview.user_login)])
            .where(and_(PullRequestReview.repository_fullname.in_(repos),
                        PullRequestReview.submitted_at.between(filt.date_from, filt.date_to),
                        )))
        rows = await asyncio.gather(u_pr, u_reviews, u_pr_commits, u_push_commits)
        singles = 2
    users = set(r[0] for r in chain.from_iterable(rows[:singles]))
    for r in chain.from_iterable(rows[singles:]):
        users.update(r)  # r[0] and r[1]
    users = ["github.com/" + u for u in sorted(users) if u]
    return web.json_response(users, dumps=FriendlyJson.dumps)


async def filter_repositories(request: AthenianWebRequest, body: dict) -> web.Response:
    """Find repositories that were updated within the given timeframe."""
    filt = FilterItemsRequest.from_dict(body)
    try:
        repos = await _resolve_repos(request.sdb, filt, request.uid)
    except ResponseError as e:
        return e.response
    async with request.mdb.connection() as conn:
        r_pr = conn.fetch_all(
            select([distinct(PullRequest.repository_fullname)])
            .where(and_(PullRequest.repository_fullname.in_(repos), or_(
                        PullRequest.created_at.between(filt.date_from, filt.date_to),
                        PullRequest.closed_at.between(filt.date_from, filt.date_to),
                        PullRequest.updated_at.between(filt.date_from, filt.date_to),
                        ))))
        r_pr_commits = conn.fetch_all(
            select([distinct(PullRequestCommit.repository_fullname)])
            .where(and_(PullRequestCommit.repository_fullname.in_(repos),
                        PullRequestCommit.commit_date.between(filt.date_from, filt.date_to),
                        )))
        r_push_commits = conn.fetch_all(
            select([distinct(PushCommit.repository_fullname)])
            .where(and_(PushCommit.repository_fullname.in_(repos),
                        PushCommit.timestamp.between(filt.date_from, filt.date_to),
                        )))
        r_reviews = conn.fetch_all(
            select([distinct(PullRequestReview.repository_fullname)])
            .where(and_(PullRequestReview.repository_fullname.in_(repos),
                        PullRequestReview.submitted_at.between(filt.date_from, filt.date_to),
                        )))
        repos = sorted({"github.com/" + r[0] for r in chain.from_iterable(await asyncio.gather(
            r_pr, r_pr_commits, r_push_commits, r_reviews))})
    return web.json_response(repos, dumps=FriendlyJson.dumps)


async def _resolve_repos(conn: Union[databases.core.Connection, databases.Database],
                         filt: FilterItemsRequest,
                         uid: str,
                         ) -> List[str]:
    status = await conn.fetch_one(
        select([UserAccount.is_admin]).where(and_(UserAccount.user_id == uid,
                                                  UserAccount.account_id == filt.account)))
    if status is None:
        raise ResponseError(ForbiddenError(
            detail="User %s is forbidden to access account %d" % (uid, filt.account)))
    if not filt._in:
        earliest_rs = await conn.fetch_one(
            select([RepositorySet.id])
            .where(RepositorySet.owner == filt.account)
            .order_by(RepositorySet.created_at))
        filt._in = ["{%d}" % earliest_rs[RepositorySet.id.key]]
    repos = set(chain.from_iterable(
        await asyncio.gather(*[resolve_reposet(r, ".in[%d]" % i, conn, uid, filt.account)
                               for i, r in enumerate(filt._in)])))
    prefix = "github.com/"
    repos = [r[r.startswith(prefix) and len(prefix):] for r in repos]
    return repos
