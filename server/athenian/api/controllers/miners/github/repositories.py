import asyncio
from datetime import datetime
from itertools import chain
import marshal
from typing import Collection, Dict, List, Optional

import aiomcache
import databases
import sentry_sdk
from sqlalchemy import and_, distinct, join, or_, select, union

from athenian.api.cache import cached
from athenian.api.controllers.miners.filters import LabelFilter
from athenian.api.controllers.miners.github.branches import dummy_branches_df, extract_branches
from athenian.api.controllers.miners.github.precomputed_prs import \
    discover_inactive_merged_unreleased_prs
from athenian.api.controllers.miners.github.release import load_releases
from athenian.api.controllers.settings import ReleaseMatchSetting
from athenian.api.models.metadata import PREFIXES
from athenian.api.models.metadata.github import NodeCommit, NodeRepository, PullRequest, \
    PullRequestComment, PullRequestReview, PushCommit, Release, Repository
from athenian.api.models.precomputed.models import GitHubDonePullRequestFacts, \
    GitHubMergedPullRequestFacts, GitHubOpenPullRequestFacts
from athenian.api.tracing import sentry_span


@sentry_span
@cached(
    exptime=5 * 60,
    serialize=marshal.dumps,
    deserialize=marshal.loads,
    key=lambda repos, time_from, time_to, **_: (
        ",".join(sorted(repos)), time_from.timestamp(), time_to.timestamp()),
)
async def mine_repositories(repos: Collection[str],
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
            .where(and_(PullRequest.repository_full_name.in_(repos),
                        PullRequest.hidden.is_(False),
                        PullRequest.created_at < time_from,
                        PullRequest.closed_at.is_(None))))

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
            .where(and_(PullRequestComment.repository_full_name.in_(repos),
                        PullRequestComment.created_at.between(time_from, time_to),
                        ))
        query_commits = \
            select([distinct(NodeRepository.name_with_owner)
                   .label(PushCommit.repository_full_name.key)]) \
            .select_from(join(NodeCommit, NodeRepository,
                              NodeCommit.repository == NodeRepository.id)) \
            .where(and_(NodeRepository.name_with_owner.in_(repos),
                        NodeCommit.committed_date.between(time_from, time_to),
                        ))
        query_reviews = \
            select([distinct(PullRequestReview.repository_full_name)]) \
            .where(and_(PullRequestReview.repository_full_name.in_(repos),
                        PullRequestReview.submitted_at.between(time_from, time_to),
                        ))
        return await mdb.fetch_all(union(query_comments, query_commits, query_reviews))

    @sentry_span
    async def fetch_releases():
        # we don't care about branch releases at all because they will bubble up in
        # fetch_commits()
        releases, _ = await load_releases(repos, dummy_branches_df(), {r: "!" for r in repos},
                                          time_from, time_to, release_settings, mdb, pdb, cache)
        return [(r,) for r in releases[Release.repository_full_name.key].unique()]

    tasks = [
        fetch_commits_comments_reviews(),
        fetch_releases(),
        fetch_active_prs(),
    ]
    if not exclude_inactive:
        tasks = [fetch_inactive_open_prs(), fetch_inactive_merged_prs()] + tasks

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in results:
        if isinstance(r, Exception):
            raise r from None
    repos = set(r[0] for r in chain.from_iterable(results))
    with sentry_sdk.start_span(op="SELECT FROM github_repositories_v2_compat"):
        repos = await mdb.fetch_all(select([Repository.full_name])
                                    .where(and_(Repository.archived.is_(False),
                                                Repository.disabled.is_(False),
                                                Repository.full_name.in_(repos)))
                                    .order_by(Repository.full_name))
    prefix = PREFIXES["github"]
    repos = [prefix + r[0] for r in repos]
    return repos
