from datetime import datetime
from itertools import chain
import marshal
from typing import Collection, Dict, List, Optional, Tuple

import aiomcache
import databases
import sentry_sdk
from sqlalchemy import and_, distinct, join, or_, select, union
from sqlalchemy.sql.functions import coalesce

from athenian.api.async_utils import gather
from athenian.api.cache import cached
from athenian.api.controllers.miners.filters import LabelFilter
from athenian.api.controllers.miners.github.branches import extract_branches
from athenian.api.controllers.miners.github.precomputed_prs import \
    discover_inactive_merged_unreleased_prs
from athenian.api.controllers.settings import ReleaseMatch, ReleaseMatchSetting
from athenian.api.models.metadata import PREFIXES
from athenian.api.models.metadata.github import NodeCommit, NodeRepository, PullRequest, \
    PullRequestComment, PullRequestReview, PushCommit, Repository
from athenian.api.models.precomputed.models import GitHubDonePullRequestFacts, \
    GitHubMergedPullRequestFacts, GitHubOpenPullRequestFacts, GitHubRelease
from athenian.api.tracing import sentry_span


@sentry_span
@cached(
    exptime=5 * 60,
    serialize=marshal.dumps,
    deserialize=marshal.loads,
    key=lambda repos, time_from, time_to, **_: (
        ",".join(sorted(repos)), time_from.timestamp(), time_to.timestamp()),
)
async def mine_repositories(accounts: Tuple[int, ...],
                            repos: Collection[str],
                            time_from: datetime,
                            time_to: datetime,
                            exclude_inactive: bool,
                            release_settings: Dict[str, ReleaseMatchSetting],
                            mdb: databases.Database,
                            pdb: databases.Database,
                            cache: Optional[aiomcache.Client],
                            ) -> List[str]:
    """Discover repositories from the given set which were updated in the given time frame."""
    assert isinstance(time_from, datetime)
    assert isinstance(time_to, datetime)
    prefix = PREFIXES["github"]
    with sentry_sdk.start_span(op="mine_repositories/nodes", description=str(len(repos))):
        rows = await mdb.fetch_all(select([NodeRepository.id])
                                   .where(NodeRepository.name_with_owner.in_(repos)))
        repo_ids = [r[0] for r in rows]

    @sentry_span
    async def fetch_active_prs():
        query_released = \
            select([distinct(GitHubDonePullRequestFacts.repository_full_name)]) \
            .where(and_(GitHubDonePullRequestFacts.repository_full_name.in_(repos),
                        or_(GitHubDonePullRequestFacts.pr_done_at.between(time_from, time_to),
                            GitHubDonePullRequestFacts.pr_created_at.between(time_from, time_to),
                            )))
        query_merged = \
            select([distinct(GitHubMergedPullRequestFacts.repository_full_name)]) \
            .where(and_(GitHubMergedPullRequestFacts.repository_full_name.in_(repos),
                        GitHubMergedPullRequestFacts.merged_at.between(time_from, time_to)))
        query_open = \
            select([distinct(GitHubOpenPullRequestFacts.repository_full_name)]) \
            .where(and_(GitHubOpenPullRequestFacts.repository_full_name.in_(repos),
                        or_(GitHubOpenPullRequestFacts.pr_updated_at.between(time_from, time_to),
                            GitHubOpenPullRequestFacts.pr_created_at.between(time_from, time_to),
                            )))
        return await pdb.fetch_all(union(query_released, query_merged, query_open))

    @sentry_span
    async def fetch_inactive_open_prs():
        return await mdb.fetch_all(
            select([distinct(PullRequest.repository_full_name)])
            .where(and_(PullRequest.repository_node_id.in_(repo_ids),
                        PullRequest.hidden.is_(False),
                        PullRequest.created_at < time_from,
                        coalesce(PullRequest.closed, False).is_(False))))

    @sentry_span
    async def fetch_inactive_merged_prs():
        _, default_branches = await extract_branches(repos, mdb, cache)
        _, inactive_repos = await discover_inactive_merged_unreleased_prs(
            time_from, time_to, repos, {}, LabelFilter.empty(), default_branches,
            release_settings, pdb, cache)
        return [(r,) for r in set(inactive_repos)]

    @sentry_span
    async def fetch_commits_comments_reviews():
        query_comments = \
            select([distinct(PullRequestComment.repository_full_name)]) \
            .where(and_(PullRequestComment.repository_node_id.in_(repo_ids),
                        PullRequestComment.created_at.between(time_from, time_to),
                        ))
        query_commits = \
            select([distinct(NodeRepository.name_with_owner)
                   .label(PushCommit.repository_full_name.key)]) \
            .select_from(join(NodeCommit, NodeRepository,
                              NodeCommit.repository == NodeRepository.id)) \
            .where(and_(NodeCommit.repository.in_(repo_ids),
                        NodeCommit.committed_date.between(time_from, time_to),
                        ))
        query_reviews = \
            select([distinct(PullRequestReview.repository_full_name)]) \
            .where(and_(PullRequestReview.repository_node_id.in_(repo_ids),
                        PullRequestReview.submitted_at.between(time_from, time_to),
                        ))
        return await mdb.fetch_all(union(query_comments, query_commits, query_reviews))

    @sentry_span
    async def fetch_releases():
        # we don't care about branch releases at all because they will bubble up in
        # fetch_commits()
        match_groups = {}
        for repo in repos:
            rms = release_settings[prefix + repo]
            if rms.match in (ReleaseMatch.tag, ReleaseMatch.tag_or_branch):
                match_groups.setdefault(rms.tags, []).append(repo)
        if not match_groups:
            # We experienced a huge fuck-up without this condition.
            return []
        or_items = or_(*(and_(GitHubRelease.release_match == "tag|" + m,
                              GitHubRelease.repository_full_name.in_(r))
                         for m, r in match_groups.items()))
        return await pdb.fetch_all(
            select([distinct(GitHubRelease.repository_full_name)])
            .where(and_(or_items, GitHubRelease.published_at.between(time_from, time_to))))

    tasks = [
        fetch_commits_comments_reviews(),
        fetch_active_prs(),
        fetch_releases(),
    ]
    if not exclude_inactive:
        tasks = [fetch_inactive_open_prs(), fetch_inactive_merged_prs()] + tasks

    results = await gather(*tasks)
    repos = set(r[0] for r in chain.from_iterable(results))
    with sentry_sdk.start_span(op="SELECT FROM github_repositories_v2_compat"):
        repos = await mdb.fetch_all(select([Repository.full_name])
                                    .where(and_(Repository.archived.is_(False),
                                                Repository.disabled.is_(False),
                                                Repository.full_name.in_(repos)))
                                    .order_by(Repository.full_name))
    repos = [prefix + r[0] for r in repos]
    return repos
