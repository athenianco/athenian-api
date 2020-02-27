import asyncio
from collections import defaultdict
from itertools import chain
from typing import List, Optional, Union

from aiohttp import web
import aiomcache
import databases.core
from sqlalchemy import and_, distinct, or_, select

from athenian.api.controllers.features.entries import PR_ENTRIES
from athenian.api.controllers.miners.access_classes import access_classes
from athenian.api.controllers.miners.pull_request_list_item import ParticipationKind, \
    PullRequestListItem, Stage
from athenian.api.controllers.reposet import resolve_reposet
from athenian.api.controllers.reposet_controller import load_account_reposets
from athenian.api.models.metadata.github import PullRequest, PullRequestCommit, \
    PullRequestReview, PushCommit, User
from athenian.api.models.state.models import RepositorySet, UserAccount
from athenian.api.models.web import ForbiddenError
from athenian.api.models.web.filter_contribs_or_repos_request import FilterContribsOrReposRequest
from athenian.api.models.web.filter_pull_requests_request import FilterPullRequestsRequest
from athenian.api.models.web.included_native_user import IncludedNativeUser
from athenian.api.models.web.included_native_users import IncludedNativeUsers
from athenian.api.models.web.pull_request import PullRequest as WebPullRequest
from athenian.api.models.web.pull_request_participant import PullRequestParticipant
from athenian.api.models.web.pull_request_set import PullRequestSet
from athenian.api.request import AthenianWebRequest
from athenian.api.response import FriendlyJson, response, ResponseError


async def filter_contributors(request: AthenianWebRequest, body: dict) -> web.Response:
    """Find developers that made an action within the given timeframe."""
    filt = FilterContribsOrReposRequest.from_dict(body)
    async with request.mdb.connection() as conn:
        try:
            repos = await _resolve_repos(
                filt, request.uid, request.native_uid, request.sdb, conn, request.cache)
        except ResponseError as e:
            return e.response
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
    filt = FilterContribsOrReposRequest.from_dict(body)
    async with request.mdb.connection() as conn:
        try:
            repos = await _resolve_repos(
                filt, request.uid, request.native_uid, request.sdb, conn, request.cache)
        except ResponseError as e:
            return e.response
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


async def _resolve_repos(filt: Union[FilterContribsOrReposRequest, FilterPullRequestsRequest],
                         uid: str,
                         native_uid: str,
                         sdb_conn: Union[databases.core.Connection, databases.Database],
                         mdb_conn: Union[databases.core.Connection, databases.Database],
                         cache: Optional[aiomcache.Client],
                         ) -> List[str]:
    status = await sdb_conn.fetch_one(
        select([UserAccount.is_admin]).where(and_(UserAccount.user_id == uid,
                                                  UserAccount.account_id == filt.account)))
    if status is None:
        raise ResponseError(ForbiddenError(
            detail="User %s is forbidden to access account %d" % (uid, filt.account)))
    check_access = True
    if not filt.in_:
        rss = await load_account_reposets(
            filt.account, native_uid, [RepositorySet.id], sdb_conn, mdb_conn, cache)
        filt.in_ = ["{%d}" % rss[0][RepositorySet.id.key]]
        check_access = False
    repos = set(chain.from_iterable(
        await asyncio.gather(*[resolve_reposet(r, ".in[%d]" % i, sdb_conn, uid, filt.account)
                               for i, r in enumerate(filt.in_)])))
    prefix = "github.com/"
    repos = [r[r.startswith(prefix) and len(prefix):] for r in repos]
    if check_access:
        checker = await access_classes["github"](filt.account, sdb_conn, mdb_conn, cache).load()
        denied = await checker.check(set(repos))
        if denied:
            raise ResponseError(ForbiddenError(
                detail="the following repositories are access denied for %s: %s" %
                       ("github.com/", denied),
            ))
    return repos


async def filter_prs(request: AthenianWebRequest, body: dict) -> web.Response:
    """List pull requests that satisfy the query."""
    filt = FilterPullRequestsRequest.from_dict(body)
    try:
        repos = await _resolve_repos(
            filt, request.uid, request.native_uid, request.sdb, request.mdb, request.cache)
    except ResponseError as e:
        return e.response
    stages = set(getattr(Stage, s.upper()) for s in filt.stages) if filt.stages else None
    if not stages:
        stages = set(Stage)
    participants = {getattr(ParticipationKind, k.upper()): v
                    for k, v in body.get("with", {}).items()}
    prs = await PR_ENTRIES["github"](
        filt.date_from, filt.date_to, repos, stages, participants, request.mdb, request.cache)
    web_prs = sorted(_web_pr_from_struct(pr) for pr in prs)
    users = {u.split("/", 1)[1] for u in
             chain.from_iterable(chain.from_iterable(pr.participants.values()) for pr in prs)}
    avatars = await request.mdb.fetch_all(
        select([User.login, User.avatar_url]).where(User.login.in_(users)))
    model = PullRequestSet(include=IncludedNativeUsers(users={
        "github.com/" + r[User.login.key]: IncludedNativeUser(avatar=r[User.avatar_url.key])
        for r in avatars
    }), data=web_prs)
    return response(model)


def _web_pr_from_struct(pr: PullRequestListItem) -> WebPullRequest:
    props = vars(pr).copy()
    props["stage"] = pr.stage.name.lower()
    participants = defaultdict(list)
    for pk, pids in sorted(pr.participants.items()):
        pkweb = pk.name.lower()
        for pid in pids:
            participants[pid].append(pkweb)
    props["participants"] = sorted(PullRequestParticipant(*p) for p in participants.items())
    return WebPullRequest(**props)


async def filter_commits(request: AthenianWebRequest, body: dict) -> web.Response:
    """Find commits that match the specified query and group them with the required time \
    granularity."""
    return None
