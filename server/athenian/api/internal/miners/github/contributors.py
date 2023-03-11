from collections import Counter, defaultdict
from datetime import datetime
from functools import reduce
from itertools import chain
import logging
import operator
import pickle
from typing import Any, Collection, Optional

import aiomcache
import morcilla
import sentry_sdk
from sqlalchemy import and_, false, func, not_, or_, select, union, union_all
from sqlalchemy.sql.functions import coalesce

from athenian.api.async_utils import gather, read_sql_query
from athenian.api.cache import cached, short_term_exptime
from athenian.api.internal.miners.github.bots import bots as fetch_bots
from athenian.api.internal.miners.github.branches import BranchMiner
from athenian.api.internal.miners.github.release_load import (
    ReleaseLoader,
    group_repos_by_release_match,
    match_groups_to_sql,
)
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.settings import LogicalRepositorySettings, ReleaseSettings
from athenian.api.models.metadata.github import (
    NodeCommit,
    NodeRepository,
    OrganizationMember,
    PullRequest,
    PullRequestComment,
    PullRequestReview,
    Release,
    User,
)
from athenian.api.models.precomputed.models import (
    GitHubDonePullRequestFacts,
    GitHubRelease as PrecomputedRelease,
)
from athenian.api.tracing import sentry_span


@sentry_span
@cached(
    exptime=short_term_exptime,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda repos, time_from, time_to, with_stats, user_roles, release_settings, logical_settings, **_: (  # noqa
        ",".join(sorted(repos)),
        time_from.timestamp() if time_from is not None else "null",
        time_to.timestamp() if time_to is not None else "null",
        with_stats,
        sorted(user_roles),
        release_settings,
        logical_settings,
    ),
)
async def mine_contributors(
    repos: Collection[str],
    time_from: Optional[datetime],
    time_to: Optional[datetime],
    with_stats: bool,
    user_roles: list[str],
    release_settings: ReleaseSettings,
    logical_settings: LogicalRepositorySettings,
    prefixer: Prefixer,
    account: int,
    meta_ids: tuple[int, ...],
    mdb: morcilla.Database,
    pdb: morcilla.Database,
    rdb: morcilla.Database,
    cache: Optional[aiomcache.Client],
    force_fresh_releases: bool = False,
) -> list[dict[str, Any]]:
    """Discover developers who made any important action in the given repositories and \
    in the given time frame."""
    assert (time_from is None) == (time_to is None)
    if has_times := (time_from is not None):
        assert isinstance(time_from, datetime)
        assert time_from.tzinfo is not None
        assert isinstance(time_to, datetime)
        assert time_to.tzinfo is not None

    common_prs_where = lambda: [  # noqa(E731)
        PullRequest.acc_id.in_(meta_ids),
        PullRequest.repository_full_name.in_(repos),
    ]
    (branches, default_branches), repo_id_df = await gather(
        BranchMiner.load_branches(repos, prefixer, account, meta_ids, mdb, pdb, cache),
        read_sql_query(
            select(NodeRepository.node_id).where(
                NodeRepository.acc_id.in_(meta_ids), NodeRepository.name_with_owner.in_(repos),
            ),
            mdb,
            [NodeRepository.node_id],
        ),
    )
    repo_nodes = repo_id_df[NodeRepository.node_id.name].values

    @sentry_span
    async def fetch_author():
        ghdprf = GitHubDonePullRequestFacts
        format_version = ghdprf.__table__.columns[ghdprf.format_version.key].default.arg
        if has_times:
            prs_opts = [
                (
                    PullRequest.created_at.between(time_from, time_to),
                    "IndexScan(pr github_node_pull_request_main)",
                ),
                (
                    and_(
                        PullRequest.created_at < time_to,
                        not_(coalesce(PullRequest.closed, False)),
                    ),
                    "IndexScan(pr node_pullrequest_open_created_at)",
                ),
                (
                    and_(PullRequest.closed, PullRequest.closed_at.between(time_from, time_to)),
                    "IndexScan(pr github_node_pull_request_repo_closed_at)",
                ),
            ]
        else:
            prs_opts = [(True, "")]
        released, *main = await gather(
            pdb.fetch_all(
                select(ghdprf.author, func.count(ghdprf.pr_node_id))
                .where(
                    ghdprf.acc_id == account,
                    ghdprf.format_version == format_version,
                    ghdprf.repository_full_name.in_(repos),
                    ghdprf.pr_done_at.between(time_from, time_to) if has_times else True,
                )
                .group_by(ghdprf.author),
            ),
            *(
                mdb.fetch_all(
                    select(PullRequest.user_login, func.count(PullRequest.node_id))
                    .where(*common_prs_where(), prs_opt)
                    .group_by(PullRequest.user_login)
                    .with_statement_hint("Rows(repo pr *100)")
                    .with_statement_hint(hint_opt),
                )
                for prs_opt, hint_opt in prs_opts
            ),
        )
        user_node_to_login_get = prefixer.user_node_to_login.get
        sum_stats = reduce(
            operator.add,
            (
                Counter({user_node_to_login_get(r[0]): r[1] for r in released}),
                *(Counter(dict(m)) for m in main),
            ),
        )
        return {"author": sum_stats.items()}

    @sentry_span
    async def fetch_reviewer():
        return {
            "reviewer": await mdb.fetch_all(
                select(PullRequestReview.user_login, func.count(PullRequestReview.user_login))
                .where(
                    PullRequestReview.acc_id.in_(meta_ids),
                    PullRequestReview.repository_full_name.in_(repos),
                    PullRequestReview.submitted_at.between(time_from, time_to)
                    if has_times
                    else True,
                )
                .group_by(PullRequestReview.user_login)
                .with_statement_hint("Rows(repo prr *100)"),
            ),
        }

    @sentry_span
    async def fetch_commit_user():
        tasks = [
            mdb.fetch_all(
                select(NodeCommit.author_user_id, func.count(NodeCommit.author_user_id))
                .where(
                    NodeCommit.acc_id.in_(meta_ids),
                    NodeCommit.repository_id.in_(repo_nodes),
                    NodeCommit.committed_date.between(time_from, time_to) if has_times else True,
                    NodeCommit.author_user_id.isnot(None),
                )
                .group_by(NodeCommit.author_user_id),
            ),
            mdb.fetch_all(
                select(NodeCommit.committer_user_id, func.count(NodeCommit.committer_user_id))
                .where(
                    NodeCommit.acc_id.in_(meta_ids),
                    NodeCommit.repository_id.in_(repo_nodes),
                    NodeCommit.committed_date.between(time_from, time_to) if has_times else True,
                    NodeCommit.committer_user_id.isnot(None),
                )
                .group_by(NodeCommit.committer_user_id),
            ),
        ]
        authors, committers = await gather(*tasks)
        user_node_to_login = prefixer.user_node_to_login.get
        return {
            "commit_committer": [(user_node_to_login(r[0]), r[1]) for r in committers],
            "commit_author": [(user_node_to_login(r[0]), r[1]) for r in authors],
        }

    @sentry_span
    async def fetch_commenter():
        return {
            "commenter": await mdb.fetch_all(
                select(PullRequestComment.user_login, func.count(PullRequestComment.user_login))
                .where(
                    PullRequestComment.acc_id.in_(meta_ids),
                    PullRequestComment.repository_full_name.in_(repos),
                    PullRequestComment.created_at.between(time_from, time_to)
                    if has_times
                    else True,
                )
                .group_by(PullRequestComment.user_login)
                .with_statement_hint("Rows(repo ic *200)"),
            ),
        }

    @sentry_span
    async def fetch_merger():
        return {
            "merger": await mdb.fetch_all(
                select(PullRequest.merged_by_login, func.count(PullRequest.merged_by_login))
                .where(
                    *common_prs_where(),
                    PullRequest.closed,
                    PullRequest.merged_at.between(time_from, time_to)
                    if has_times
                    else PullRequest.merged,
                )
                .group_by(PullRequest.merged_by_login)
                .with_statement_hint("Rows(repo pr *100)"),
            ),
        }

    @sentry_span
    async def fetch_releaser():
        if has_times:
            rt_from, rt_to = time_from, time_to
            releases, _ = await ReleaseLoader.load_releases(
                repos,
                branches,
                default_branches,
                rt_from,
                rt_to,
                release_settings,
                logical_settings,
                prefixer,
                account,
                meta_ids,
                mdb,
                pdb,
                rdb,
                cache,
                force_fresh=force_fresh_releases,
            )
            counts = releases[Release.author.name].value_counts()
            counts = list(zip(counts.index.values, counts.values))
        else:
            # we may load 200,000 releases here, must optimize and sacrifice precision
            prel = PrecomputedRelease
            or_items, _ = match_groups_to_sql(
                group_repos_by_release_match(repos, default_branches, release_settings)[0],
                prel,
                True,
                prefixer,
            )
            if pdb.url.dialect == "sqlite":
                query = (
                    select(prel.author_node_id, func.count(prel.node_id))
                    .where(prel.acc_id == account, or_(*or_items) if or_items else false())
                    .group_by(prel.author_node_id)
                )
                rows = await pdb.fetch_all(query)
            else:
                queries = [
                    select(prel.author_node_id, func.count(prel.node_id))
                    .where(prel.acc_id == account, item)
                    .group_by(prel.author_node_id)
                    for item in or_items
                ]
                batch_size = 10
                rows = chain.from_iterable(
                    await gather(
                        *(
                            pdb.fetch_all(union_all(*queries[batch : batch + batch_size]))
                            for batch in range(0, len(queries), batch_size)
                        ),
                    ),
                )
            user_node_to_login = prefixer.user_node_to_login.get
            counts = [(user_node_to_login(r[0]), r[1]) for r in rows]
        return {
            "releaser": counts,
        }

    @sentry_span
    async def fetch_organization_member():
        if time_from is None:
            # only load org members if we request the entire lifetime
            user_node_ids = (
                await read_sql_query(
                    select(OrganizationMember.child_id).where(
                        OrganizationMember.acc_id.in_(meta_ids),
                    ),
                    mdb,
                    [OrganizationMember.child_id],
                )
            )[OrganizationMember.child_id.name].values
        else:
            user_node_ids = []
        user_node_to_login = prefixer.user_node_to_login.get
        return {
            "member": [(user_node_to_login(u), 1) for u in user_node_ids],
        }

    fetchers_mapping = {
        "author": fetch_author,
        "reviewer": fetch_reviewer,
        "commit_committer": fetch_commit_user,
        "commit_author": fetch_commit_user,
        "commenter": fetch_commenter,
        "merger": fetch_merger,
        "releaser": fetch_releaser,
        "member": fetch_organization_member,
    }

    user_roles = user_roles or fetchers_mapping.keys()
    tasks = {fetchers_mapping[role] for role in user_roles}
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
            select(*cols).where(User.acc_id.in_(meta_ids), User.login.in_(stats.keys())),
        )

    contribs = []
    for ud in user_details:
        c = dict(ud)
        try:
            user_stats = stats[(login := c[User.login.name])]
        except KeyError:
            # may happen on multiple GitHub installations with the same user
            continue
        del stats[login]
        if user_roles and sum(user_stats.get(role, 0) for role in user_roles) == 0:
            continue

        if with_stats:
            if "author" in user_stats:
                # We could get rid of these re-mapping, maybe worth looking at it along with the
                # definition of `DeveloperUpdates`
                user_stats["prs"] = user_stats.pop("author")
            c["stats"] = user_stats

        contribs.append(c)

    return contribs


async def load_organization_members(
    account: int,
    meta_ids: tuple[int, ...],
    mdb: morcilla.Database,
    sdb: morcilla.Database,
    log: logging.Logger,
    cache: Optional[aiomcache.Client],
) -> tuple[dict[str, set], dict[str, set], dict[str, str]]:
    """
    Fetch the mapping from account's GitHub organization member IDs to their signatures.

    :return: 1. Map from user node IDs to full names. \
             2. Map from user node IDs to emails. \
             3. Map from user node IDs to prefixed logins.
    """
    user_ids = (
        await read_sql_query(
            select(OrganizationMember.child_id).where(OrganizationMember.acc_id.in_(meta_ids)),
            mdb,
            [OrganizationMember.child_id],
        )
    )[OrganizationMember.child_id.name].values
    log.info("Discovered %d organization members", len(user_ids))
    user_rows, bots = await gather(
        mdb.fetch_all(
            select(User.node_id, User.name, User.login, User.html_url, User.email).where(
                User.acc_id.in_(meta_ids), User.node_id.in_(user_ids), User.login.isnot(None),
            ),
        ),
        fetch_bots(account, meta_ids, mdb, sdb, cache),
    )
    log.info("Detailed %d GitHub users", len(user_rows))
    bot_ids = set()
    new_user_rows = []
    for row in user_rows:
        if row[User.login.name] in bots:
            bot_ids.add(row[User.node_id.name])
        else:
            new_user_rows.append(row)
    user_rows = new_user_rows
    user_ids = [uid for uid in user_ids if uid not in bot_ids]
    log.info("Excluded %d bots", len(bot_ids))
    if not user_ids:
        return {}, {}, {}
    # this may show up in Cloud Insights, but we cannot optimize the query much
    # the used indexes are rare and have to be read from disk
    signature_rows = await mdb.fetch_all(
        union(
            select(NodeCommit.author_user_id, NodeCommit.author_name, NodeCommit.author_email)
            .where(NodeCommit.acc_id.in_(meta_ids), NodeCommit.author_user_id.in_(user_ids))
            .distinct(),
            select(
                NodeCommit.committer_user_id, NodeCommit.committer_name, NodeCommit.author_email,
            )
            .where(NodeCommit.acc_id.in_(meta_ids), NodeCommit.committer_user_id.in_(user_ids))
            .distinct(),
        ),
    )
    log.info("Loaded %d signatures", len(signature_rows))
    github_names = defaultdict(set)
    github_emails = defaultdict(set)
    for row in signature_rows:
        node_id = row[NodeCommit.author_user_id.name]
        if name := row[NodeCommit.author_name.name]:
            github_names[node_id].add(name)
        if email := row[NodeCommit.author_email.name]:
            github_emails[node_id].add(email)
    github_prefixed_logins = {}
    for row in user_rows:
        node_id = row[User.node_id.name]
        github_prefixed_logins[node_id] = row[User.html_url.name].split("://", 1)[1]
        if name := row[User.name.name]:
            github_names[node_id].add(name)
        elif node_id not in github_names:
            github_names[node_id].add(row[User.login.name])
        if email := row[User.email.name]:
            github_emails[node_id].add(email)
    log.info("GitHub set size: %d", len(github_names))
    return github_names, github_emails, github_prefixed_logins
