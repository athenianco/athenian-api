from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from itertools import chain
import logging
import marshal
from typing import Any, Collection, Dict, List, Optional, Tuple

import aiomcache
import databases
import sentry_sdk
from sqlalchemy import and_, func, not_, select, union
from sqlalchemy.sql.functions import coalesce

from athenian.api.async_utils import gather
from athenian.api.cache import cached
from athenian.api.controllers.miners.github.bots import Bots
from athenian.api.controllers.miners.github.branches import extract_branches
from athenian.api.controllers.miners.github.release_load import load_releases
from athenian.api.controllers.settings import ReleaseMatchSetting
from athenian.api.models.metadata.github import NodeCommit, NodeRepository, OrganizationMember, \
    PullRequest, PullRequestComment, PullRequestReview, PushCommit, Release, User
from athenian.api.models.precomputed.models import GitHubDonePullRequestFacts
from athenian.api.models.state.models import Team
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
                            account: int,
                            meta_ids: Tuple[int, ...],
                            mdb: databases.Database,
                            pdb: databases.Database,
                            rdb: databases.Database,
                            cache: Optional[aiomcache.Client],
                            force_fresh_releases: bool = False) -> List[Dict[str, Any]]:
    """Discover developers who made any important action in the given repositories and \
    in the given time frame."""
    assert (time_from is None) == (time_to is None)
    if has_times := (time_from is not None):
        assert isinstance(time_from, datetime)
        assert time_from.tzinfo is not None
        assert isinstance(time_to, datetime)
        assert time_to.tzinfo is not None

    common_prs_where = lambda: [  # noqa(E731)
        PullRequest.repository_full_name.in_(repos),
        PullRequest.hidden.is_(False),
        PullRequest.acc_id.in_(meta_ids),
    ]
    tasks = [
        extract_branches(repos, meta_ids, mdb, cache),
        mdb.fetch_all(select([NodeRepository.node_id])
                      .where(and_(NodeRepository.name_with_owner.in_(repos),
                                  NodeRepository.acc_id.in_(meta_ids)))),
    ]
    (branches, default_branches), repo_rows = await gather(*tasks)
    repo_nodes = [r[0] for r in repo_rows]

    @sentry_span
    async def fetch_author():
        ghdprf = GitHubDonePullRequestFacts
        format_version = ghdprf.__table__.columns[ghdprf.format_version.key].default.arg
        if has_times:
            prs_opts = [
                PullRequest.created_at.between(time_from, time_to),
                and_(PullRequest.created_at < time_to,
                     not_(coalesce(PullRequest.closed, False))),
                PullRequest.closed_at.between(time_from, time_to),
            ]
        else:
            prs_opts = [True]
        tasks = [
            pdb.fetch_all(select([ghdprf.author, ghdprf.pr_node_id])
                          .where(and_(ghdprf.format_version == format_version,
                                      ghdprf.repository_full_name.in_(repos),
                                      ghdprf.acc_id == account,
                                      ghdprf.pr_done_at.between(time_from, time_to)
                                      if has_times else True))
                          .distinct()),
            mdb.fetch_all(union(*(select([PullRequest.user_login, PullRequest.node_id])
                                  .where(and_(*common_prs_where(), prs_opt))
                                  for prs_opt in prs_opts))),
        ]
        released, main = await gather(*tasks)
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
                            PullRequestReview.acc_id.in_(meta_ids),
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
                            NodeCommit.acc_id.in_(meta_ids),
                            NodeCommit.committed_date.between(time_from, time_to)
                            if has_times else True,
                            NodeCommit.author_user.isnot(None)))
                .group_by(NodeCommit.author_user)),
            mdb.fetch_all(
                select([NodeCommit.committer_user, func.count(NodeCommit.committer_user)])
                .where(and_(NodeCommit.repository.in_(repo_nodes),
                            NodeCommit.acc_id.in_(meta_ids),
                            NodeCommit.committed_date.between(time_from, time_to)
                            if has_times else True,
                            NodeCommit.committer_user.isnot(None)))
                .group_by(NodeCommit.committer_user)),
        ]
        authors, committers = await gather(*tasks)
        user_ids = set(r[0] for r in authors).union(r[0] for r in committers)
        logins = await mdb.fetch_all(select([User.node_id, User.login])
                                     .where(and_(User.node_id.in_(user_ids),
                                                 User.acc_id.in_(meta_ids))))
        logins = {r[0]: r[1] for r in logins}
        return {
            "commit_committer": [(logins[r[0]], r[1]) for r in committers if r[0] in logins],
            "commit_author": [(logins[r[0]], r[1]) for r in authors if r[0] in logins],
        }

    @sentry_span
    async def fetch_commenter():
        return {
            "commenter": await mdb.fetch_all(
                select([PullRequestComment.user_login, func.count(PullRequestComment.user_login)])
                .where(and_(PullRequestComment.repository_full_name.in_(repos),
                            PullRequestComment.acc_id.in_(meta_ids),
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
                .where(and_(*common_prs_where(),
                            PullRequest.closed,
                            PullRequest.merged_at.between(time_from, time_to)
                            if has_times else PullRequest.merged,
                            ))
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
        releases, _ = await load_releases(
            repos, branches, default_branches, rt_from, rt_to,
            release_settings, account, meta_ids, mdb, pdb, rdb, cache,
            force_fresh=force_fresh_releases)
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
    data = await gather(*(t() for t in tasks))
    stats = defaultdict(dict)
    for dikt in data:
        for key, rows in dikt.items():
            for row in rows:
                stats[row[0]][key] = row[1]

    stats.pop(None, None)

    cols = [User.login, User.email, User.avatar_url, User.name, User.node_id]
    with sentry_sdk.start_span(op="SELECT FROM github.api_users"):
        user_details = await mdb.fetch_all(
            select(cols)
            .where(and_(User.login.in_(stats.keys()), User.acc_id.in_(meta_ids))))

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


async def load_organization_members(account: int,
                                    meta_ids: Tuple[int, ...],
                                    mdb: databases.Database,
                                    sdb: databases.Database,
                                    log: logging.Logger,
                                    ) -> Tuple[Dict[str, set], Dict[str, set], Dict[str, str]]:
    """
    Fetch the mapping from account's GitHub organization member IDs to their signatures.

    :return: 1. Map from user node IDs to full names. \
             2. Map from user node IDs to emails. \
             3. Map from user node IDs to logins.
    """
    user_ids = [
        r[0] for r in await mdb.fetch_all(select([OrganizationMember.child_id])
                                          .where(OrganizationMember.acc_id.in_(meta_ids)))
    ]
    log.info("Discovered %d organization members", len(user_ids))
    tasks = [
        mdb.fetch_all(select([User.node_id, User.name, User.login, User.email])
                      .where(and_(User.acc_id.in_(meta_ids),
                                  User.node_id.in_(user_ids),
                                  User.name.isnot(None)))),
        Bots()(mdb),
        sdb.fetch_val(select([Team.members]).where(and_(
            Team.owner_id == account,
            Team.name == Team.BOTS,
        ))),
    ]
    user_rows, bots, team_bots = await gather(*tasks)
    if team_bots:
        bots = bots.copy()
        bots.update(u.split("/", 1)[1] for u in team_bots)
    log.info("Detailed %d GitHub users", len(user_rows))
    bot_ids = set()
    new_user_rows = []
    for row in user_rows:
        if row[User.login.key] in bots:
            bot_ids.add(row[User.node_id.key])
        else:
            new_user_rows.append(row)
    user_rows = new_user_rows
    user_ids = [uid for uid in user_ids if uid not in bot_ids]
    log.info("Excluded %d bots", len(bot_ids))
    if not user_ids:
        return {}, {}, {}
    signature_rows = await mdb.fetch_all(union(
        select([PushCommit.author_user, PushCommit.author_name, PushCommit.author_email])
        .where(and_(PushCommit.acc_id.in_(meta_ids),
                    PushCommit.author_user.in_(user_ids)))
        .distinct(),
        select([PushCommit.committer_user, PushCommit.committer_name, PushCommit.author_email])
        .where(and_(PushCommit.acc_id.in_(meta_ids),
                    PushCommit.committer_user.in_(user_ids)))
        .distinct(),
    ))
    log.info("Loaded %d signatures", len(signature_rows))
    github_names = defaultdict(set)
    github_emails = defaultdict(set)
    for row in signature_rows:
        node_id = row[PushCommit.author_user.key]
        if name := row[PushCommit.author_name.key]:
            github_names[node_id].add(name)
        if email := row[PushCommit.author_email.key]:
            github_emails[node_id].add(email)
    github_logins = {}
    for row in user_rows:
        node_id = row[User.node_id.key]
        github_logins[node_id] = row[User.login.key]
        if name := row[User.name.key]:
            github_names[node_id].add(name)
        if email := row[User.email.key]:
            github_emails[node_id].add(email)
    log.info("GitHub set size: %d", len(github_names))
    return github_names, github_emails, github_logins
