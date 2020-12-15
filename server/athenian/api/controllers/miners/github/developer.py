from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from itertools import chain
import pickle
from typing import Collection, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import aiomcache
import databases
import numpy as np
import pandas as pd
from sqlalchemy import and_, distinct, func, join, or_, select
from sqlalchemy.orm.attributes import InstrumentedAttribute

from athenian.api.async_utils import gather, read_sql_query
from athenian.api.cache import cached
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.branches import extract_branches
from athenian.api.controllers.miners.github.pull_request import ReviewResolution
from athenian.api.controllers.miners.github.release_load import load_releases
from athenian.api.controllers.miners.jira.issue import generate_jira_prs_query
from athenian.api.controllers.settings import ReleaseMatchSetting
from athenian.api.models.metadata.github import PullRequest, PullRequestComment, \
    PullRequestLabel, PullRequestReview, PullRequestReviewComment, PushCommit, Release, \
    Repository, User
from athenian.api.tracing import sentry_span
from athenian.api.typing_utils import dataclass


class DeveloperTopic(Enum):
    """Possible developer statistics kinds."""

    commits_pushed = "dev-commits-pushed"
    lines_changed = "dev-lines-changed"
    prs_created = "dev-prs-created"
    prs_reviewed = "dev-prs-reviewed"
    prs_merged = "dev-prs-merged"
    releases = "dev-releases"
    reviews = "dev-reviews"
    review_approvals = "dev-review-approvals"
    review_rejections = "dev-review-rejections"
    review_neutrals = "dev-review-neutrals"
    pr_comments = "dev-pr-comments"
    regular_pr_comments = "dev-regular-pr-comments"
    review_pr_comments = "dev-review-pr-comments"


@dataclass(slots=True, frozen=True)
class DeveloperStats:
    """Calculated statistics about developer activities."""

    commits_pushed: int = 0
    lines_changed: int = 0
    prs_created: int = 0
    prs_reviewed: int = 0
    prs_merged: int = 0
    releases: int = 0
    reviews: int = 0
    review_approvals: int = 0
    review_rejections: int = 0
    review_neutrals: int = 0
    pr_comments: int = 0
    regular_pr_comments: int = 0
    review_pr_comments: int = 0


#                          repo             dev     metric
StatsByRepoByDev = Dict[Optional[str], Dict[str, Dict[str, int]]]


def _set_stats(topic: DeveloperTopic, stats: pd.Series, repogroups: bool,
               stats_by_repo_by_dev: StatsByRepoByDev):
    topic = topic.name
    if not repogroups:
        output = stats_by_repo_by_dev[None]
        for dev, n in zip(stats.index.values, stats.values):
            output[dev][topic] = n
    else:
        for (repo, dev), n in zip(stats.index.values, stats.values):
            stats_by_repo_by_dev[repo][dev][topic] = n


@sentry_span
async def _set_commits(stats_by_repo_by_dev: StatsByRepoByDev,
                       topics: Set[str],
                       time_from: datetime,
                       time_to: datetime,
                       dev_ids: Dict[str, str],
                       repo_ids: Dict[str, str],
                       repogroups: bool,
                       labels: LabelFilter,
                       jira: JIRAFilter,
                       release_settings: Dict[str, ReleaseMatchSetting],
                       meta_ids: Tuple[int, ...],
                       mdb: databases.Database,
                       pdb: databases.Database,
                       cache: Optional[aiomcache.Client]) -> None:
    commits = await _fetch_developer_commits(
        dev_ids.values(), repo_ids.values(), repogroups, time_from, time_to, meta_ids, mdb, cache)
    if DeveloperTopic.commits_pushed in topics:
        topic = DeveloperTopic.commits_pushed.name
        if not repogroups:
            output = stats_by_repo_by_dev[None]
            for dev, count in zip(commits.index.values, commits["count"].values):
                output[dev][topic] = count
        else:
            for (repo, dev), count in zip(commits.index.values, commits["count"].values):
                stats_by_repo_by_dev[repo][dev][topic] = count
    if DeveloperTopic.lines_changed in topics:
        topic = DeveloperTopic.lines_changed.name
        if not repogroups:
            output = stats_by_repo_by_dev[None]
            for dev, lines in zip(commits.index.values, commits["lines"].values):
                output[dev][topic] = lines
        else:
            for (repo, dev), lines in zip(commits.index.values, commits["lines"].values):
                stats_by_repo_by_dev[repo][dev][topic] = lines


@sentry_span
async def _set_prs_created(stats_by_repo_by_dev: StatsByRepoByDev,
                           topics: Set[str],
                           time_from: datetime,
                           time_to: datetime,
                           dev_ids: Dict[str, str],
                           repo_ids: Dict[str, str],
                           repogroups: bool,
                           labels: LabelFilter,
                           jira: JIRAFilter,
                           release_settings: Dict[str, ReleaseMatchSetting],
                           meta_ids: Tuple[int, ...],
                           mdb: databases.Database,
                           pdb: databases.Database,
                           cache: Optional[aiomcache.Client]) -> None:
    prs = await _fetch_developer_created_prs(
        dev_ids.values(), repo_ids.values(), repogroups, labels, jira,
        time_from, time_to, meta_ids, mdb, cache)
    _set_stats(DeveloperTopic.prs_created, prs["count"], repogroups, stats_by_repo_by_dev)


@sentry_span
async def _set_prs_reviewed(stats_by_repo_by_dev: StatsByRepoByDev,
                            topics: Set[str],
                            time_from: datetime,
                            time_to: datetime,
                            dev_ids: Dict[str, str],
                            repo_ids: Dict[str, str],
                            repogroups: bool,
                            labels: LabelFilter,
                            jira: JIRAFilter,
                            release_settings: Dict[str, ReleaseMatchSetting],
                            meta_ids: Tuple[int, ...],
                            mdb: databases.Database,
                            pdb: databases.Database,
                            cache: Optional[aiomcache.Client]) -> None:
    prs = await _fetch_developer_reviewed_prs(
        dev_ids.values(), repo_ids.values(), repogroups, labels, jira,
        time_from, time_to, meta_ids, mdb, cache)
    _set_stats(DeveloperTopic.prs_reviewed, prs["count"], repogroups, stats_by_repo_by_dev)


@sentry_span
async def _set_prs_merged(stats_by_repo_by_dev: StatsByRepoByDev,
                          topics: Set[str],
                          time_from: datetime,
                          time_to: datetime,
                          dev_ids: Dict[str, str],
                          repo_ids: Dict[str, str],
                          repogroups: bool,
                          labels: LabelFilter,
                          jira: JIRAFilter,
                          release_settings: Dict[str, ReleaseMatchSetting],
                          meta_ids: Tuple[int, ...],
                          mdb: databases.Database,
                          pdb: databases.Database,
                          cache: Optional[aiomcache.Client]) -> None:
    prs = await _fetch_developer_merged_prs(
        dev_ids.values(), repo_ids.values(), repogroups, labels, jira,
        time_from, time_to, meta_ids, mdb, cache)
    _set_stats(DeveloperTopic.prs_merged, prs["count"], repogroups, stats_by_repo_by_dev)


@sentry_span
async def _set_releases(stats_by_repo_by_dev: StatsByRepoByDev,
                        topics: Set[str],
                        time_from: datetime,
                        time_to: datetime,
                        dev_ids: Dict[str, str],
                        repo_ids: Dict[str, str],
                        repogroups: bool,
                        labels: LabelFilter,
                        jira: JIRAFilter,
                        release_settings: Dict[str, ReleaseMatchSetting],
                        meta_ids: Tuple[int, ...],
                        mdb: databases.Database,
                        pdb: databases.Database,
                        cache: Optional[aiomcache.Client]) -> None:
    branches, default_branches = await extract_branches(repo_ids, meta_ids, mdb, cache)
    releases, _ = await load_releases(
        repo_ids, branches, default_branches, time_from, time_to,
        release_settings, meta_ids, mdb, pdb, cache)
    topic = DeveloperTopic.releases.name
    included_releases = np.nonzero(np.in1d(releases[Release.author.key].values, list(dev_ids)))[0]
    if not repogroups:
        stats = releases[Release.author.key].take(included_releases).value_counts()
        output = stats_by_repo_by_dev[None]
        for author, count in zip(stats.index.values, stats.values):
            output[dev_ids[author]][topic] = count
    else:
        stats = releases[[Release.repository_node_id.key, Release.author.key]] \
            .take(included_releases).value_counts()
        for (repo_id, author), count in zip(stats.index.values, stats.values):
            stats_by_repo_by_dev[repo_id][dev_ids[author]][topic] = count


@sentry_span
async def _set_reviews(stats_by_repo_by_dev: StatsByRepoByDev,
                       topics: Set[str],
                       time_from: datetime,
                       time_to: datetime,
                       dev_ids: Dict[str, str],
                       repo_ids: Dict[str, str],
                       repogroups: bool,
                       labels: LabelFilter,
                       jira: JIRAFilter,
                       release_settings: Dict[str, ReleaseMatchSetting],
                       meta_ids: Tuple[int, ...],
                       mdb: databases.Database,
                       pdb: databases.Database,
                       cache: Optional[aiomcache.Client]) -> None:
    reviews = await _fetch_developer_reviews(
        dev_ids.values(), repo_ids.values(), repogroups, labels, jira,
        time_from, time_to, meta_ids, mdb, cache)
    if reviews.empty:
        return
    if DeveloperTopic.reviews in topics:
        topic = DeveloperTopic.reviews.name
        if not repogroups:
            output = stats_by_repo_by_dev[None]
            for dev, n in reviews.groupby(level=0, sort=False).sum()["count"].items():
                output[dev][topic] = n
        else:
            for (repo, dev), n in reviews.groupby(level=[0, 1], sort=False).sum()["count"].items():
                stats_by_repo_by_dev[repo][dev][topic] = n

    def set_stats_by_state(topic: DeveloperTopic, rr: ReviewResolution):
        topic = topic.name
        try:
            if not repogroups:
                output = stats_by_repo_by_dev[None]
                for dev, n in reviews.xs(
                        rr.value, level=PullRequestReview.state.key)["count"].items():
                    output[dev][topic] = n
            else:
                for (repo, dev), n in reviews.xs(
                        rr.value, level=PullRequestReview.state.key)["count"].items():
                    stats_by_repo_by_dev[repo][dev][topic] = n
        except KeyError:
            pass

    for topic, rr in ((DeveloperTopic.review_approvals, ReviewResolution.APPROVED),
                      (DeveloperTopic.review_neutrals, ReviewResolution.COMMENTED),
                      (DeveloperTopic.review_rejections, ReviewResolution.CHANGES_REQUESTED)):
        if topic in topics:
            set_stats_by_state(topic, rr)


@sentry_span
async def _set_pr_comments(stats_by_repo_by_dev: StatsByRepoByDev,
                           topics: Set[str],
                           time_from: datetime,
                           time_to: datetime,
                           dev_ids: Dict[str, str],
                           repo_ids: Dict[str, str],
                           repogroups: bool,
                           labels: LabelFilter,
                           jira: JIRAFilter,
                           release_settings: Dict[str, ReleaseMatchSetting],
                           meta_ids: Tuple[int, ...],
                           mdb: databases.Database,
                           pdb: databases.Database,
                           cache: Optional[aiomcache.Client]) -> None:
    if DeveloperTopic.review_pr_comments in topics or DeveloperTopic.pr_comments in topics:
        review_comments = await _fetch_developer_review_comments(
            dev_ids.values(), repo_ids.values(), repogroups,
            labels, jira, time_from, time_to, meta_ids, mdb, cache)
        if DeveloperTopic.review_pr_comments in topics:
            _set_stats(DeveloperTopic.review_pr_comments, review_comments["count"],
                       repogroups, stats_by_repo_by_dev)
    if DeveloperTopic.regular_pr_comments in topics or DeveloperTopic.pr_comments in topics:
        regular_pr_comments = await _fetch_developer_regular_pr_comments(
            dev_ids.values(), repo_ids.values(), repogroups,
            labels, jira, time_from, time_to, mdb, cache)
        if DeveloperTopic.regular_pr_comments in topics:
            _set_stats(DeveloperTopic.regular_pr_comments, regular_pr_comments["count"],
                       repogroups, stats_by_repo_by_dev)
    if DeveloperTopic.pr_comments in topics:
        review_index = review_comments["count"].index
        regular_index = regular_pr_comments["count"].index
        joint_index = review_index.union(regular_index)
        joint = pd.Series(data=[0] * len(joint_index), index=joint_index)
        joint[review_index.difference(regular_index)] = review_comments["count"]
        joint[regular_index.difference(review_index)] = regular_pr_comments["count"]
        common_index = review_index.intersection(regular_index)
        joint[common_index] = \
            review_comments["count"][common_index] + regular_pr_comments["count"][common_index]
        _set_stats(DeveloperTopic.pr_comments, joint, repogroups, stats_by_repo_by_dev)


processors = [
    ({DeveloperTopic.commits_pushed, DeveloperTopic.lines_changed}, _set_commits),
    ({DeveloperTopic.prs_created}, _set_prs_created),
    ({DeveloperTopic.prs_reviewed}, _set_prs_reviewed),
    ({DeveloperTopic.prs_merged}, _set_prs_merged),
    ({DeveloperTopic.releases}, _set_releases),
    ({DeveloperTopic.reviews, DeveloperTopic.review_approvals, DeveloperTopic.review_neutrals,
      DeveloperTopic.review_rejections}, _set_reviews),
    ({DeveloperTopic.pr_comments, DeveloperTopic.regular_pr_comments,
      DeveloperTopic.review_pr_comments}, _set_pr_comments),
]


@sentry_span
async def calc_developer_metrics_github(devs: Sequence[str],
                                        repos: Sequence[Collection[str]],
                                        time_from: datetime,
                                        time_to: datetime,
                                        topics: Set[DeveloperTopic],
                                        labels: LabelFilter,
                                        jira: JIRAFilter,
                                        release_settings: Dict[str, ReleaseMatchSetting],
                                        meta_ids: Tuple[int, ...],
                                        mdb: databases.Database,
                                        pdb: databases.Database,
                                        cache: Optional[aiomcache.Client],
                                        ) -> List[List[DeveloperStats]]:
    """Calculate various statistics about developer activities.

    :return: List with calculated stats, the order matches `devs`.
    """
    zerotd = timedelta(0)
    assert isinstance(time_from, datetime) and time_from.tzinfo.utcoffset(time_from) == zerotd
    assert isinstance(time_to, datetime) and time_to.tzinfo.utcoffset(time_to) == zerotd
    dev_ids_map, reverse_dev_ids_map, repo_ids_map = await _fetch_node_ids(devs, repos, mdb)
    stats_by_repo_by_dev = defaultdict(lambda: defaultdict(dict))
    tasks = []
    for key, setter in processors:
        if key.intersection(topics):
            tasks.append(setter(stats_by_repo_by_dev, topics, time_from, time_to,
                                dev_ids_map, repo_ids_map, len(repos) > 1,
                                labels, jira, release_settings, meta_ids, mdb, pdb, cache))
    await gather(*tasks)
    return _convert_stats(stats_by_repo_by_dev, devs, repos, repo_ids_map, reverse_dev_ids_map)


@sentry_span
async def _fetch_node_ids(devs: Collection[str],
                          repos: Sequence[Collection[str]],
                          mdb: databases.Database,
                          ) -> Tuple[Dict[str, str], Dict[str, int], Dict[str, str]]:
    all_repos = set(chain.from_iterable(repos))
    tasks = [
        mdb.fetch_all(select([Repository.node_id, Repository.full_name])
                      .where(Repository.full_name.in_(all_repos))),
        mdb.fetch_all(select([User.node_id, User.login])
                      .where(User.login.in_(devs))),
    ]
    repo_ids, dev_ids = await gather(*tasks)
    repo_ids_map = {r[1]: r[0] for r in repo_ids}
    dev_ids_map = {r[1]: r[0] for r in dev_ids}
    reverse_dev_ids_map = {dev_ids_map[dev]: i for i, dev in enumerate(devs) if dev in dev_ids_map}
    return dev_ids_map, reverse_dev_ids_map, repo_ids_map


@sentry_span
def _convert_stats(stats_by_repo_by_dev: StatsByRepoByDev,
                   devs: Sequence[str],
                   repos: Sequence[Collection[str]],
                   repo_ids_map: Dict[str, str],
                   reverse_dev_ids_map: Dict[str, int],
                   ) -> List[List[DeveloperStats]]:
    if len(repos) > 1:
        result = []
        for group in repos:
            agg_stats_by_dev = [defaultdict(int) for _ in devs]
            for repo in group:
                for dev_id, stats in stats_by_repo_by_dev[repo_ids_map[repo]].items():
                    dev_index = reverse_dev_ids_map[dev_id]
                    for k, v in stats.items():
                        agg_stats_by_dev[dev_index][k] += v
            result.append([DeveloperStats(**stats) for stats in agg_stats_by_dev])
    else:
        stats_by_dev = [{}] * len(devs)
        for dev_id, stats in stats_by_repo_by_dev[None].items():
            stats_by_dev[reverse_dev_ids_map[dev_id]] = stats
        result = [[DeveloperStats(**stats) for stats in stats_by_dev]]
    return result


CACHE_EXPIRATION_TIME = 5 * 60  # 5 min


@cached(
    exptime=CACHE_EXPIRATION_TIME,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda devs, repos, repogroups, time_from, time_to, **_: (
        ",".join(devs), ",".join(sorted(repos)), time_from.toordinal(), time_to.toordinal(),
        repogroups,
    ),
)
async def _fetch_developer_commits(devs: Iterable[str],
                                   repos: Iterable[str],
                                   repogroups: bool,
                                   time_from: datetime,
                                   time_to: datetime,
                                   meta_ids: Tuple[int, ...],
                                   mdb: databases.Database,
                                   cache: Optional[aiomcache.Client],
                                   ) -> pd.DataFrame:
    columns = [func.sum(PushCommit.additions + PushCommit.deletions).label("lines"),
               func.count(PushCommit.node_id).label("count"),
               PushCommit.author_user]
    group_by = [PushCommit.author_user]
    if repogroups:
        columns.insert(0, PushCommit.repository_node_id)
        group_by.insert(0, PushCommit.repository_node_id)
    query = select(columns).where(and_(
        PushCommit.committed_date.between(time_from, time_to),
        PushCommit.author_user.in_(devs),
        PushCommit.repository_node_id.in_(repos),
        PushCommit.acc_id.in_(meta_ids),
    )).group_by(*group_by)
    df = await read_sql_query(
        query, mdb, [c.key for c in columns], index=[c.key for c in group_by])
    df.fillna(0, inplace=True, downcast="infer")
    return df


async def _fetch_developer_timestamp_prs(attr_filter: InstrumentedAttribute,
                                         attr_user: InstrumentedAttribute,
                                         devs: Iterable[str],
                                         repos: Iterable[str],
                                         repogroups: bool,
                                         labels: LabelFilter,
                                         jira: JIRAFilter,
                                         time_from: datetime,
                                         time_to: datetime,
                                         meta_ids: Tuple[int, ...],
                                         mdb: databases.Database,
                                         cache: Optional[aiomcache.Client],
                                         ) -> pd.DataFrame:
    selected = [attr_user, func.count(PullRequest.node_id).label("count")]
    group_by = [attr_user]
    if repogroups:
        selected.insert(0, PullRequest.repository_node_id)
        group_by.insert(0, PullRequest.repository_node_id)
    filters = [
        attr_filter.between(time_from, time_to),
        attr_user.in_(devs),
        PullRequest.repository_node_id.in_(repos),
    ]
    if labels:
        filters.extend([
            func.lower(PullRequestLabel.name).in_(labels.include) if labels.include else True,
            or_(func.lower(PullRequestLabel.name).notin_(labels.exclude),
                PullRequestLabel.name.is_(None)) if labels.exclude else True,
        ])
        seed = join(
            PullRequest, PullRequestLabel,
            PullRequest.node_id == PullRequestLabel.pull_request_node_id,
            isouter=not labels.include,
        )
        if jira:
            query = await generate_jira_prs_query(
                filters, jira, mdb, cache, columns=selected, seed=seed)
        else:
            query = select(selected).select_from(seed).where(and_(*filters))
    elif jira:
        query = await generate_jira_prs_query(
            filters, jira, mdb, cache, columns=selected)
    else:
        query = select(selected).where(and_(*filters))
    df = await read_sql_query(
        query.group_by(*group_by), mdb, [c.key for c in selected], index=[c.key for c in group_by])
    df.fillna(0, inplace=True, downcast="infer")
    return df


@cached(
    exptime=CACHE_EXPIRATION_TIME,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda devs, repos, repogroups, labels, time_from, time_to, **_: (
        ",".join(devs), ",".join(sorted(repos)), time_from.toordinal(), time_to.toordinal(),
        labels, repogroups,
    ),
)
async def _fetch_developer_created_prs(devs: Iterable[str],
                                       repos: Iterable[str],
                                       repogroups: bool,
                                       labels: LabelFilter,
                                       jira: JIRAFilter,
                                       time_from: datetime,
                                       time_to: datetime,
                                       meta_ids: Tuple[int, ...],
                                       mdb: databases.Database,
                                       cache: Optional[aiomcache.Client],
                                       ) -> pd.DataFrame:
    return await _fetch_developer_timestamp_prs(
        PullRequest.created_at, PullRequest.user_node_id,
        devs, repos, repogroups, labels, jira, time_from, time_to, meta_ids, mdb, cache,
    )


@cached(
    exptime=CACHE_EXPIRATION_TIME,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda devs, repos, repogroups, time_from, time_to, labels, **_: (
        ",".join(devs), ",".join(sorted(repos)), time_from.toordinal(), time_to.toordinal(),
        labels, repogroups,
    ),
)
async def _fetch_developer_merged_prs(devs: Iterable[str],
                                      repos: Iterable[str],
                                      repogroups: bool,
                                      labels: LabelFilter,
                                      jira: JIRAFilter,
                                      time_from: datetime,
                                      time_to: datetime,
                                      meta_ids: Tuple[int, ...],
                                      mdb: databases.Database,
                                      cache: Optional[aiomcache.Client],
                                      ) -> pd.DataFrame:
    return await _fetch_developer_timestamp_prs(
        PullRequest.merged_at, PullRequest.merged_by,
        devs, repos, repogroups, labels, jira, time_from, time_to, meta_ids, mdb, cache,
    )


async def _fetch_developer_review_common(selected: List[InstrumentedAttribute],
                                         group_by: List[InstrumentedAttribute],
                                         devs: Iterable[str],
                                         repos: Iterable[str],
                                         repogroups: bool,
                                         labels: LabelFilter,
                                         jira: JIRAFilter,
                                         time_from: datetime,
                                         time_to: datetime,
                                         meta_ids: Tuple[int, ...],
                                         mdb: databases.Database,
                                         cache: Optional[aiomcache.Client],
                                         ) -> pd.DataFrame:
    if repogroups:
        selected.insert(0, PullRequestReview.repository_node_id)
        group_by.insert(0, PullRequestReview.repository_node_id)
    filters = [
        PullRequestReview.submitted_at.between(time_from, time_to),
        PullRequestReview.user_node_id.in_(devs),
        PullRequestReview.repository_node_id.in_(repos),
    ]
    if labels:
        filters.extend([
            func.lower(PullRequestLabel.name).in_(labels.include) if labels.include else True,
            or_(func.lower(PullRequestLabel.name).notin_(labels.exclude),
                PullRequestLabel.name.is_(None)) if labels.exclude else True,
        ])
        seed = join(
            PullRequestReview, PullRequestLabel,
            PullRequestReview.pull_request_node_id == PullRequestLabel.pull_request_node_id,
            isouter=not labels.include,
        )
        if jira:
            query = await generate_jira_prs_query(
                filters, jira, mdb, cache, columns=selected, seed=seed,
                on=(PullRequestReview.pull_request_node_id, PullRequestReview.acc_id))
        else:
            query = select(selected).select_from(seed).where(and_(*filters))
    elif jira:
        query = await generate_jira_prs_query(
            filters, jira, mdb, cache, columns=selected, seed=PullRequestReview,
            on=(PullRequestReview.pull_request_node_id, PullRequestReview.acc_id))
    else:
        query = select(selected).where(and_(*filters))
    df = await read_sql_query(
        query.group_by(*group_by), mdb, [c.key for c in selected], index=[c.key for c in group_by])
    df.fillna(0, inplace=True, downcast="infer")
    return df


@cached(
    exptime=CACHE_EXPIRATION_TIME,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda devs, repos, repogroups, time_from, time_to, labels, **_: (
        ",".join(devs), ",".join(sorted(repos)), time_from.toordinal(), time_to.toordinal(),
        labels, repogroups,
    ),
)
async def _fetch_developer_reviewed_prs(devs: Iterable[str],
                                        repos: Iterable[str],
                                        repogroups: bool,
                                        labels: LabelFilter,
                                        jira: JIRAFilter,
                                        time_from: datetime,
                                        time_to: datetime,
                                        meta_ids: Tuple[int, ...],
                                        mdb: databases.Database,
                                        cache: Optional[aiomcache.Client],
                                        ) -> pd.DataFrame:
    selected = [PullRequestReview.user_node_id,
                func.count(distinct(PullRequestReview.pull_request_node_id)).label("count")]
    group_by = [PullRequestReview.user_node_id]
    return await _fetch_developer_review_common(
        selected, group_by, devs, repos, repogroups, labels, jira, time_from, time_to,
        meta_ids, mdb, cache)


@cached(
    exptime=CACHE_EXPIRATION_TIME,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda devs, repos, repogroups, time_from, time_to, labels, **_: (
        ",".join(devs), ",".join(sorted(repos)), time_from.toordinal(), time_to.toordinal(),
        labels, repogroups,
    ),
)
async def _fetch_developer_reviews(devs: Iterable[str],
                                   repos: Iterable[str],
                                   repogroups: bool,
                                   labels: LabelFilter,
                                   jira: JIRAFilter,
                                   time_from: datetime,
                                   time_to: datetime,
                                   meta_ids: Tuple[int, ...],
                                   mdb: databases.Database,
                                   cache: Optional[aiomcache.Client],
                                   ) -> pd.DataFrame:
    selected = [PullRequestReview.user_node_id, PullRequestReview.state,
                func.count(PullRequestReview.node_id).label("count")]
    group_by = [PullRequestReview.user_node_id, PullRequestReview.state]
    return await _fetch_developer_review_common(
        selected, group_by, devs, repos, repogroups, labels, jira, time_from, time_to,
        meta_ids, mdb, cache)


@cached(
    exptime=CACHE_EXPIRATION_TIME,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda devs, repos, repogroups, time_from, time_to, labels, **_: (
        ",".join(devs), ",".join(sorted(repos)), time_from.toordinal(), time_to.toordinal(),
        labels, repogroups,
    ),
)
async def _fetch_developer_review_comments(devs: Iterable[str],
                                           repos: Iterable[str],
                                           repogroups: bool,
                                           labels: LabelFilter,
                                           jira: JIRAFilter,
                                           time_from: datetime,
                                           time_to: datetime,
                                           meta_ids: Tuple[int, ...],
                                           mdb: databases.Database,
                                           cache: Optional[aiomcache.Client],
                                           ) -> pd.DataFrame:
    selected = [PullRequestReviewComment.user_node_id,
                func.count(PullRequestReviewComment.node_id).label("count")]
    group_by = [PullRequestReviewComment.user_node_id]
    if repogroups:
        selected.insert(0, PullRequestReviewComment.repository_node_id)
        group_by.insert(0, PullRequestReviewComment.repository_node_id)
    filters = [
        PullRequestReviewComment.created_at.between(time_from, time_to),
        PullRequestReviewComment.user_node_id.in_(devs),
        PullRequestReviewComment.repository_node_id.in_(repos),
    ]
    if labels:
        filters.extend([
            func.lower(PullRequestLabel.name).in_(labels.include) if labels.include else True,
            or_(func.lower(PullRequestLabel.name).notin_(labels.exclude),
                PullRequestLabel.name.is_(None)) if labels.exclude else True,
        ])
        seed = join(
            PullRequestReviewComment, PullRequestLabel,
            PullRequestReviewComment.pull_request_node_id == PullRequestLabel.pull_request_node_id,
            isouter=not labels.include,
        )
        if jira:
            query = await generate_jira_prs_query(
                filters, jira, mdb, cache, columns=selected, seed=seed,
                on=(PullRequestReviewComment.pull_request_node_id,
                    PullRequestReviewComment.acc_id))
        else:
            query = select(selected).select_from(seed).where(and_(*filters))
    elif jira:
        query = await generate_jira_prs_query(
            filters, jira, mdb, cache, columns=selected, seed=PullRequestReviewComment,
            on=(PullRequestReviewComment.pull_request_node_id, PullRequestReviewComment.acc_id))
    else:
        query = select(selected).where(and_(*filters))
    df = await read_sql_query(
        query.group_by(*group_by), mdb, [c.key for c in selected], index=[c.key for c in group_by])
    df.fillna(0, inplace=True, downcast="infer")
    return df


@cached(
    exptime=CACHE_EXPIRATION_TIME,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda devs, repos, repogroups, time_from, time_to, labels, **_: (
        ",".join(devs), ",".join(sorted(repos)), time_from.toordinal(), time_to.toordinal(),
        labels, repogroups,
    ),
)
async def _fetch_developer_regular_pr_comments(devs: Iterable[str],
                                               repos: Iterable[str],
                                               repogroups: bool,
                                               labels: LabelFilter,
                                               jira: JIRAFilter,
                                               time_from: datetime,
                                               time_to: datetime,
                                               mdb: databases.Database,
                                               cache: Optional[aiomcache.Client],
                                               ) -> pd.DataFrame:
    selected = [PullRequestComment.user_node_id,
                func.count(PullRequestComment.user_node_id).label("count")]
    group_by = [PullRequestComment.user_node_id]
    if repogroups:
        selected.insert(0, PullRequestComment.repository_node_id)
        group_by.insert(0, PullRequestComment.repository_node_id)
    filters = [
        PullRequestComment.created_at.between(time_from, time_to),
        PullRequestComment.user_node_id.in_(devs),
        PullRequestComment.repository_node_id.in_(repos),
    ]
    if labels:
        filters.extend([
            func.lower(PullRequestLabel.name).in_(labels.include) if labels.include else True,
            or_(func.lower(PullRequestLabel.name).notin_(labels.exclude),
                PullRequestLabel.name.is_(None)) if labels.exclude else True,
        ])
        seed = join(
            PullRequestComment, PullRequestLabel,
            PullRequestComment.pull_request_node_id == PullRequestLabel.pull_request_node_id,
            isouter=not labels.include,
        )
        if jira:
            query = await generate_jira_prs_query(
                filters, jira, mdb, cache, columns=selected, seed=seed,
                on=(PullRequestComment.pull_request_node_id, PullRequestComment.acc_id))
        else:
            query = select(selected).select_from(seed).where(and_(*filters))
    elif jira:
        query = await generate_jira_prs_query(
            filters, jira, mdb, cache, columns=selected, seed=PullRequestComment,
            on=(PullRequestComment.pull_request_node_id, PullRequestComment.acc_id))
    else:
        query = select(selected).where(and_(*filters))
    df = await read_sql_query(
        query.group_by(*group_by), mdb, [c.key for c in selected], index=[c.key for c in group_by])
    df.fillna(0, inplace=True, downcast="infer")
    return df
