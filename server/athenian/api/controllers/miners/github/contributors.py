import asyncio
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from itertools import chain
import marshal
from typing import Any, Collection, Dict, List, Optional

import aiomcache
import databases
import sentry_sdk
from sqlalchemy import and_, func, or_, select

from athenian.api.cache import cached
from athenian.api.controllers.miners.github.branches import extract_branches
from athenian.api.controllers.miners.github.release import load_releases
from athenian.api.controllers.settings import ReleaseMatchSetting
from athenian.api.models.metadata.github import NodeCommit, NodeRepository, PullRequest, \
    PullRequestComment, PullRequestReview, Release, User
from athenian.api.models.precomputed.models import GitHubDonePullRequestFacts
from athenian.api.tracing import sentry_span


@sentry_span
@cached(
    exptime=5 * 60,
    serialize=marshal.dumps,
    deserialize=marshal.loads,
    key=lambda repos, time_from, time_to, with_stats, user_roles, release_settings, **_: (
        ",".join(sorted(repos)),
        time_from.timestamp() if time_from is not None else "null",
        time_to.timestamp() if time_to is not None else "null",
        with_stats, sorted(user_roles), release_settings,
    ),
)
async def mine_contributors(repos: Collection[str],
                            time_from: Optional[datetime],
                            time_to: Optional[datetime],
                            with_stats: bool,
                            user_roles: List[str],
                            release_settings: Dict[str, ReleaseMatchSetting],
                            mdb: databases.Database,
                            pdb: databases.Database,
                            cache: Optional[aiomcache.Client]) -> List[Dict[str, Any]]:
    """Discover developers who made any important action in the given repositories and \
    in the given time frame."""
    assert (time_from is None) == (time_to is None)
    if has_times := (time_from is not None):
        assert isinstance(time_from, datetime)
        assert time_from.tzinfo is not None
        assert isinstance(time_to, datetime)
        assert time_to.tzinfo is not None

    common_prs_where = and_(
        PullRequest.repository_full_name.in_(repos),
        PullRequest.hidden.is_(False),
        or_(PullRequest.created_at.between(time_from, time_to),
            and_(PullRequest.created_at < time_to,
                 PullRequest.closed_at.is_(None)),
            PullRequest.closed_at.between(time_from, time_to)) if has_times else True,
    )
    tasks = [
        extract_branches(repos, mdb, cache),
        mdb.fetch_all(select([NodeRepository.node_id])
                      .where(NodeRepository.name_with_owner.in_(repos))),
    ]
    branches, repo_rows = await asyncio.gather(*tasks, return_exceptions=True)
    for r in (branches, repo_rows):
        if isinstance(r, Exception):
            raise r from None
    branches, default_branches = branches
    repo_nodes = [r[0] for r in repo_rows]

    @sentry_span
    async def fetch_author():
        ghdprf = GitHubDonePullRequestFacts
        format_version = ghdprf.__table__.columns[ghdprf.format_version.key].default.arg
        tasks = [
            pdb.fetch_all(select([ghdprf.author, ghdprf.pr_node_id])
                          .where(and_(ghdprf.format_version == format_version,
                                      ghdprf.repository_full_name.in_(repos),
                                      ghdprf.pr_done_at.between(time_from, time_to)
                                      if has_times else True))
                          .distinct()),
            mdb.fetch_all(select([PullRequest.user_login, PullRequest.node_id])
                          .where(common_prs_where)),
        ]
        released, main = await asyncio.gather(*tasks, return_exceptions=True)
        for r in (released, main):
            if isinstance(r, Exception):
                raise r from None
        return {
            "author": Counter(user for user, _ in set(chain(
                ((r[0], r[1]) for r in released),
                ((r[0], r[1]) for r in main)))
            ).items(),
        }

    @sentry_span
    async def fetch_reviewer():
        return {
            "reviewer": await mdb.fetch_all(
                select([PullRequestReview.user_login, func.count(PullRequestReview.user_login)])
                .where(and_(PullRequestReview.repository_full_name.in_(repos),
                            PullRequestReview.submitted_at.between(time_from, time_to)
                            if has_times else True))
                .group_by(PullRequestReview.user_login)),
        }

    @sentry_span
    async def fetch_commit_user():
        tasks = [
            mdb.fetch_all(
                select([NodeCommit.author_user, func.count(NodeCommit.author_user)])
                .where(and_(NodeCommit.repository.in_(repo_nodes),
                            NodeCommit.committed_date.between(time_from, time_to)
                            if has_times else True,
                            NodeCommit.author_user.isnot(None)))
                .group_by(NodeCommit.author_user)),
            mdb.fetch_all(
                select([NodeCommit.committer_user, func.count(NodeCommit.committer_user)])
                .where(and_(NodeCommit.repository.in_(repo_nodes),
                            NodeCommit.committed_date.between(time_from, time_to)
                            if has_times else True,
                            NodeCommit.committer_user.isnot(None)))
                .group_by(NodeCommit.committer_user)),
        ]
        authors, committers = await asyncio.gather(*tasks, return_exceptions=True)
        for r in (authors, committers):
            if isinstance(r, Exception):
                raise r from None
        user_ids = set(r[0] for r in authors).union(r[0] for r in committers)
        logins = await mdb.fetch_all(select([User.node_id, User.login])
                                     .where(User.node_id.in_(user_ids)))
        logins = {r[0]: r[1] for r in logins}
        return {
            "commit_committer": [(logins[r[0]], r[1]) for r in committers],
            "commit_author": [(logins[r[0]], r[1]) for r in authors],
        }

    @sentry_span
    async def fetch_commenter():
        return {
            "commenter": await mdb.fetch_all(
                select([PullRequestComment.user_login, func.count(PullRequestComment.user_login)])
                .where(and_(PullRequestComment.repository_full_name.in_(repos),
                            PullRequestComment.created_at.between(time_from, time_to)
                            if has_times else True,
                            ))
                .group_by(PullRequestComment.user_login)),
        }

    @sentry_span
    async def fetch_merger():
        return {
            "merger": await mdb.fetch_all(
                select([PullRequest.merged_by_login, func.count(PullRequest.merged_by_login)])
                .where(common_prs_where)
                .group_by(PullRequest.merged_by_login)),
        }

    @sentry_span
    async def fetch_releaser():
        if has_times:
            rt_from, rt_to = time_from, time_to
        else:
            rt_from = datetime(1970, 1, 1, tzinfo=timezone.utc)
            now = datetime.now(timezone.utc) + timedelta(days=1)
            rt_to = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        releases, _ = await load_releases(repos, branches, default_branches, rt_from, rt_to,
                                          release_settings, mdb, pdb, cache)
        counts = releases[Release.author.key].value_counts()
        return {
            "releaser": zip(counts.index.values, counts.values),
        }

    fetchers_mapping = {
        "author": fetch_author,
        "reviewer": fetch_reviewer,
        "commit_committer": fetch_commit_user,
        "commit_author": fetch_commit_user,
        "commenter": fetch_commenter,
        "merger": fetch_merger,
        "releaser": fetch_releaser,
    }

    user_roles = user_roles or fetchers_mapping.keys()
    tasks = set(fetchers_mapping[role] for role in user_roles)
    data = await asyncio.gather(*(t() for t in tasks), return_exceptions=True)
    stats = defaultdict(dict)
    for dikt in data:
        if isinstance(dikt, Exception):
            raise dikt from None
        for key, rows in dikt.items():
            for row in rows:
                stats[row[0]][key] = row[1]

    stats.pop(None, None)

    cols = [User.login, User.email, User.avatar_url, User.name]
    with sentry_sdk.start_span(op="SELECT FROM github_users_v2_compat"):
        user_details = await mdb.fetch_all(select(cols).where(User.login.in_(stats.keys())))

    contribs = []
    for ud in user_details:
        c = dict(ud)
        c["stats"] = stats[c[User.login.key]]
        if user_roles and sum(c["stats"].get(role, 0) for role in user_roles) == 0:
            continue

        if "author" in c["stats"]:
            # We could get rid of these re-mapping, maybe worth looking at it along with the
            # definition of `DeveloperUpdates`
            c["stats"]["prs"] = c["stats"].pop("author")

        if not with_stats:
            c.pop("stats")

        contribs.append(c)

    return contribs
