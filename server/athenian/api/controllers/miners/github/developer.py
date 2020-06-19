import asyncio
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import pickle
from typing import Collection, Dict, List, Optional, Sequence, Set, Union

import aiomcache
import databases
import pandas as pd
from sqlalchemy import and_, distinct, func, join, select

from athenian.api.async_read_sql_query import read_sql_query
from athenian.api.cache import cached
from athenian.api.controllers.miners.github.pull_request import ReviewResolution
from athenian.api.models.metadata.github import PullRequest, PullRequestComment, \
    PullRequestLabel, PullRequestReview, PullRequestReviewComment, PushCommit, Release
from athenian.api.tracing import sentry_span


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


@dataclass(frozen=True)
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


@sentry_span
async def _set_commits(stats_by_dev: Dict[str, Dict[str, Union[int, float]]],
                       topics: Set[str],
                       devs: Sequence[str],
                       repos: Collection[str],
                       labels: Set[str],
                       time_from: datetime,
                       time_to: datetime,
                       conn: databases.core.Connection,
                       cache: Optional[aiomcache.Client]) -> None:
    commits = await _fetch_developer_commits(devs, repos, time_from, time_to, conn, cache)
    commits_by_dev = commits.groupby(PushCommit.author_login.key, sort=False)
    if DeveloperTopic.commits_pushed in topics:
        topic = DeveloperTopic.commits_pushed.name
        for dev, dev_commits in commits_by_dev.count()[PushCommit.additions.key].items():
            stats_by_dev[dev][topic] = dev_commits
    if DeveloperTopic.lines_changed in topics:
        ads = commits_by_dev.sum()
        lines_by_dev = ads[PushCommit.additions.key] + ads[PushCommit.deletions.key]
        topic = DeveloperTopic.lines_changed.name
        for dev, dev_lines in lines_by_dev.items():
            stats_by_dev[dev][topic] = dev_lines


@sentry_span
async def _set_prs_created(stats_by_dev: Dict[str, Dict[str, Union[int, float]]],
                           topics: Set[str],
                           devs: Sequence[str],
                           repos: Collection[str],
                           labels: Set[str],
                           time_from: datetime,
                           time_to: datetime,
                           conn: databases.core.Connection,
                           cache: Optional[aiomcache.Client]) -> None:
    prs = await _fetch_developer_created_prs(devs, repos, labels, time_from, time_to, conn, cache)
    topic = DeveloperTopic.prs_created.name
    for dev, n in prs["created_count"].items():
        stats_by_dev[dev][topic] = n


@sentry_span
async def _set_prs_reviewed(stats_by_dev: Dict[str, Dict[str, Union[int, float]]],
                            topics: Set[str],
                            devs: Sequence[str],
                            repos: Collection[str],
                            labels: Set[str],
                            time_from: datetime,
                            time_to: datetime,
                            conn: databases.core.Connection,
                            cache: Optional[aiomcache.Client]) -> None:
    prs = await _fetch_developer_reviewed_prs(devs, repos, labels, time_from, time_to, conn, cache)
    topic = DeveloperTopic.prs_reviewed.name
    for dev, n in prs["reviewed_count"].items():
        stats_by_dev[dev][topic] = n


@sentry_span
async def _set_prs_merged(stats_by_dev: Dict[str, Dict[str, Union[int, float]]],
                          topics: Set[str],
                          devs: Sequence[str],
                          repos: Collection[str],
                          labels: Set[str],
                          time_from: datetime,
                          time_to: datetime,
                          conn: databases.core.Connection,
                          cache: Optional[aiomcache.Client]) -> None:
    prs = await _fetch_developer_merged_prs(devs, repos, labels, time_from, time_to, conn, cache)
    topic = DeveloperTopic.prs_merged.name
    for dev, n in prs["merged_count"].items():
        stats_by_dev[dev][topic] = n


@sentry_span
async def _set_releases(stats_by_dev: Dict[str, Dict[str, Union[int, float]]],
                        topics: Set[str],
                        devs: Sequence[str],
                        repos: Collection[str],
                        labels: Set[str],
                        time_from: datetime,
                        time_to: datetime,
                        conn: databases.core.Connection,
                        cache: Optional[aiomcache.Client]) -> None:
    prs = await _fetch_developer_releases(devs, repos, time_from, time_to, conn, cache)
    topic = DeveloperTopic.releases.name
    for dev, n in prs["released_count"].items():
        stats_by_dev[dev][topic] = n


@sentry_span
async def _set_reviews(stats_by_dev: Dict[str, Dict[str, Union[int, float]]],
                       topics: Set[str],
                       devs: Sequence[str],
                       repos: Collection[str],
                       labels: Set[str],
                       time_from: datetime,
                       time_to: datetime,
                       conn: databases.core.Connection,
                       cache: Optional[aiomcache.Client]) -> None:
    reviews = await _fetch_developer_reviews(devs, repos, labels, time_from, time_to, conn, cache)
    if reviews.empty:
        return
    if DeveloperTopic.reviews in topics:
        topic = DeveloperTopic.reviews.name
        for dev, n in (reviews
                       .groupby(level=0, sort=False)
                       .sum()["reviews_count"]).items():
            stats_by_dev[dev][topic] = n
    if DeveloperTopic.review_approvals in topics:
        topic = DeveloperTopic.review_approvals.name
        try:
            for dev, n in reviews.xs(ReviewResolution.APPROVED.value,
                                     level=PullRequestReview.state.key)["reviews_count"].items():
                stats_by_dev[dev][topic] = n
        except KeyError:
            pass
    if DeveloperTopic.review_neutrals in topics:
        topic = DeveloperTopic.review_neutrals.name
        try:
            for dev, n in reviews.xs(ReviewResolution.COMMENTED.value,
                                     level=PullRequestReview.state.key)["reviews_count"].items():
                stats_by_dev[dev][topic] = n
        except KeyError:
            pass
    if DeveloperTopic.review_rejections in topics:
        topic = DeveloperTopic.review_rejections.name
        try:
            for dev, n in reviews.xs(ReviewResolution.CHANGES_REQUESTED.value,
                                     level=PullRequestReview.state.key)["reviews_count"].items():
                stats_by_dev[dev][topic] = n
        except KeyError:
            pass


@sentry_span
async def _set_pr_comments(stats_by_dev: Dict[str, Dict[str, Union[int, float]]],
                           topics: Set[str],
                           devs: Sequence[str],
                           repos: Collection[str],
                           labels: Set[str],
                           time_from: datetime,
                           time_to: datetime,
                           conn: databases.core.Connection,
                           cache: Optional[aiomcache.Client]) -> None:
    if DeveloperTopic.review_pr_comments in topics or DeveloperTopic.pr_comments in topics:
        review_comments = await _fetch_developer_review_comments(
            devs, repos, labels, time_from, time_to, conn, cache)
        if DeveloperTopic.review_pr_comments in topics:
            topic = DeveloperTopic.review_pr_comments.name
            for dev, n in review_comments["comments_count"].items():
                stats_by_dev[dev][topic] = n
    if DeveloperTopic.regular_pr_comments in topics or DeveloperTopic.pr_comments in topics:
        regular_pr_comments = await _fetch_developer_regular_pr_comments(
            devs, repos, labels, time_from, time_to, conn, cache)
        if DeveloperTopic.regular_pr_comments in topics:
            topic = DeveloperTopic.regular_pr_comments.name
            for dev, n in regular_pr_comments["comments_count"].items():
                stats_by_dev[dev][topic] = n
    if DeveloperTopic.pr_comments in topics:
        topic = DeveloperTopic.pr_comments.name
        for dev, n in (review_comments["comments_count"] +
                       regular_pr_comments["comments_count"]).items():
            if n == n:  # can be NaN
                stats_by_dev[dev][topic] = n


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
async def calc_developer_metrics(devs: Sequence[str],
                                 repos: Collection[str],
                                 topics: Set[DeveloperTopic],
                                 labels: Set[str],
                                 time_from: datetime,
                                 time_to: datetime,
                                 db: databases.Database,
                                 cache: Optional[aiomcache.Client],
                                 ) -> List[DeveloperStats]:
    """Calculate various statistics about developer activities.

    :return: List with calculated stats, the order matches `devs`.
    """
    zerotd = timedelta(0)
    assert isinstance(time_from, datetime) and time_from.tzinfo.utcoffset(time_from) == zerotd
    assert isinstance(time_to, datetime) and time_to.tzinfo.utcoffset(time_to) == zerotd
    stats_by_dev = defaultdict(dict)
    tasks = []
    for key, setter in processors:
        if key.intersection(topics):
            tasks.append(setter(
                stats_by_dev, topics, devs, repos, labels, time_from, time_to, db, cache))
    errors = await asyncio.gather(*tasks, return_exceptions=True)
    for err in errors:
        if isinstance(err, Exception):
            raise err from None
    return [DeveloperStats(**stats_by_dev[dev]) for dev in devs]


CACHE_EXPIRATION_TIME = 5 * 60  # 5 min


@cached(
    exptime=CACHE_EXPIRATION_TIME,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda devs, repos, time_from, time_to, **_: (
        ",".join(devs), ",".join(sorted(repos)), time_from.toordinal(), time_to.toordinal()),
)
async def _fetch_developer_commits(devs: Sequence[str],
                                   repos: Collection[str],
                                   time_from: datetime,
                                   time_to: datetime,
                                   db: databases.core.Connection,
                                   cache: Optional[aiomcache.Client],
                                   ) -> pd.DataFrame:
    columns = [PushCommit.additions, PushCommit.deletions, PushCommit.author_login]
    return await read_sql_query(
        select(columns).where(and_(
            PushCommit.committed_date.between(time_from, time_to),
            PushCommit.author_login.in_(devs),
            PushCommit.repository_full_name.in_(repos),
        )),
        db, columns)


@cached(
    exptime=CACHE_EXPIRATION_TIME,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda devs, repos, time_from, time_to, **_: (
        ",".join(devs), ",".join(sorted(repos)), time_from.toordinal(), time_to.toordinal()),
)
async def _fetch_developer_created_prs(devs: Sequence[str],
                                       repos: Collection[str],
                                       labels: Set[str],
                                       time_from: datetime,
                                       time_to: datetime,
                                       db: databases.core.Connection,
                                       cache: Optional[aiomcache.Client],
                                       ) -> pd.DataFrame:
    query = select([PullRequest.user_login, func.count(PullRequest.created_at)])
    if labels:
        query = (
            query.select_from(join(
                PullRequest, PullRequestLabel,
                PullRequest.node_id == PullRequestLabel.pull_request_node_id,
            )).where(and_(
                PullRequest.created_at.between(time_from, time_to),
                PullRequest.user_login.in_(devs),
                PullRequest.repository_full_name.in_(repos),
                PullRequestLabel.name.in_(labels),
            ))
        )
    else:
        query = (
            query.where(and_(
                PullRequest.created_at.between(time_from, time_to),
                PullRequest.user_login.in_(devs),
                PullRequest.repository_full_name.in_(repos),
            ))
        )
    df = await read_sql_query(
        query.group_by(PullRequest.user_login),
        db, [PullRequest.user_login.key, "created_count"],
        index=PullRequest.user_login.key)
    df.fillna(0, inplace=True, downcast="infer")
    return df


@cached(
    exptime=CACHE_EXPIRATION_TIME,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda devs, repos, time_from, time_to, **_: (
        ",".join(devs), ",".join(sorted(repos)), time_from.toordinal(), time_to.toordinal()),
)
async def _fetch_developer_merged_prs(devs: Sequence[str],
                                      repos: Collection[str],
                                      labels: Set[str],
                                      time_from: datetime,
                                      time_to: datetime,
                                      db: databases.core.Connection,
                                      cache: Optional[aiomcache.Client],
                                      ) -> pd.DataFrame:
    query = select([PullRequest.merged_by_login, func.count(PullRequest.merged_at)])
    if labels:
        query = (
            query.select_from(join(
                PullRequest, PullRequestLabel,
                PullRequest.node_id == PullRequestLabel.pull_request_node_id,
            )).where(and_(
                PullRequest.merged_at.between(time_from, time_to),
                PullRequest.merged_by_login.in_(devs),
                PullRequest.repository_full_name.in_(repos),
                PullRequestLabel.name.in_(labels),
            ))
        )
    else:
        query = (
            query.where(and_(
                PullRequest.merged_at.between(time_from, time_to),
                PullRequest.merged_by_login.in_(devs),
                PullRequest.repository_full_name.in_(repos),
            ))
        )
    df = await read_sql_query(
        query.group_by(PullRequest.merged_by_login),
        db, [PullRequest.merged_by_login.key, "merged_count"],
        index=PullRequest.merged_by_login.key)
    df.fillna(0, inplace=True, downcast="infer")
    return df


@cached(
    exptime=CACHE_EXPIRATION_TIME,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda devs, repos, time_from, time_to, **_: (
        ",".join(devs), ",".join(sorted(repos)), time_from.toordinal(), time_to.toordinal()),
)
async def _fetch_developer_releases(devs: Sequence[str],
                                    repos: Collection[str],
                                    time_from: datetime,
                                    time_to: datetime,
                                    db: databases.core.Connection,
                                    cache: Optional[aiomcache.Client],
                                    ) -> pd.DataFrame:
    df = await read_sql_query(
        select([Release.author, func.count(Release.published_at)]).where(and_(
            Release.published_at.between(time_from, time_to),
            Release.author.in_(devs),
            Release.repository_full_name.in_(repos),
        )).group_by(Release.author),
        db, [Release.author.key, "released_count"], index=Release.author.key)
    df.fillna(0, inplace=True, downcast="infer")
    return df


@cached(
    exptime=CACHE_EXPIRATION_TIME,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda devs, repos, time_from, time_to, **_: (
        ",".join(devs), ",".join(sorted(repos)), time_from.toordinal(), time_to.toordinal()),
)
async def _fetch_developer_reviewed_prs(devs: Sequence[str],
                                        repos: Collection[str],
                                        labels: Set[str],
                                        time_from: datetime,
                                        time_to: datetime,
                                        db: databases.core.Connection,
                                        cache: Optional[aiomcache.Client],
                                        ) -> pd.DataFrame:
    query = select([PullRequestReview.user_login,
                    func.count(distinct(PullRequestReview.pull_request_node_id))])
    if labels:
        query = (
            query.select_from(join(
                PullRequestReview, PullRequestLabel,
                PullRequestReview.pull_request_node_id == PullRequestLabel.pull_request_node_id))
            .where(and_(
                PullRequestReview.submitted_at.between(time_from, time_to),
                PullRequestReview.user_login.in_(devs),
                PullRequestReview.repository_full_name.in_(repos),
                PullRequestLabel.name.in_(labels),
            ))
        )
    else:
        query = (
            query.where(and_(
                PullRequestReview.submitted_at.between(time_from, time_to),
                PullRequestReview.user_login.in_(devs),
                PullRequestReview.repository_full_name.in_(repos),
            ))
        )
    df = await read_sql_query(
        query.group_by(PullRequestReview.user_login),
        db, [PullRequestReview.user_login.key, "reviewed_count"],
        index=PullRequestReview.user_login.key)
    df.fillna(0, inplace=True, downcast="infer")
    return df


@cached(
    exptime=CACHE_EXPIRATION_TIME,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda devs, repos, time_from, time_to, **_: (
        ",".join(devs), ",".join(sorted(repos)), time_from.toordinal(), time_to.toordinal()),
)
async def _fetch_developer_reviews(devs: Sequence[str],
                                   repos: Collection[str],
                                   labels: Set[str],
                                   time_from: datetime,
                                   time_to: datetime,
                                   db: databases.core.Connection,
                                   cache: Optional[aiomcache.Client],
                                   ) -> pd.DataFrame:
    query = select([PullRequestReview.user_login, PullRequestReview.state,
                    func.count(PullRequestReview.submitted_at)])
    if labels:
        query = (
            query.select_from(join(
                PullRequestReview, PullRequestLabel,
                PullRequestReview.pull_request_node_id == PullRequestLabel.pull_request_node_id))
            .where(and_(
                PullRequestReview.submitted_at.between(time_from, time_to),
                PullRequestReview.user_login.in_(devs),
                PullRequestReview.repository_full_name.in_(repos),
                PullRequestLabel.name.in_(labels),
            ))
        )
    else:
        query = (
            query.where(and_(
                PullRequestReview.submitted_at.between(time_from, time_to),
                PullRequestReview.user_login.in_(devs),
                PullRequestReview.repository_full_name.in_(repos),
            ))
        )
    df = await read_sql_query(
        query.group_by(PullRequestReview.user_login, PullRequestReview.state),
        db, [PullRequestReview.user_login.key, PullRequestReview.state.key, "reviews_count"],
        index=[PullRequestReview.user_login.key, PullRequestReview.state.key])
    df.fillna(0, inplace=True, downcast="infer")
    return df


@cached(
    exptime=CACHE_EXPIRATION_TIME,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda devs, repos, time_from, time_to, **_: (
        ",".join(devs), ",".join(sorted(repos)), time_from.toordinal(), time_to.toordinal()),
)
async def _fetch_developer_review_comments(devs: Sequence[str],
                                           repos: Collection[str],
                                           labels: Set[str],
                                           time_from: datetime,
                                           time_to: datetime,
                                           db: databases.core.Connection,
                                           cache: Optional[aiomcache.Client],
                                           ) -> pd.DataFrame:
    query = select([PullRequestReviewComment.user_login,
                    func.count(PullRequestReviewComment.created_at)])
    if labels:
        query = (
            query.select_from(join(
                PullRequestReviewComment, PullRequestLabel,
                PullRequestReviewComment.pull_request_node_id == PullRequestLabel.pull_request_node_id,  # noqa
            )).where(and_(
                PullRequestReviewComment.created_at.between(time_from, time_to),
                PullRequestReviewComment.user_login.in_(devs),
                PullRequestReviewComment.repository_full_name.in_(repos),
                PullRequestLabel.name.in_(labels),
            ))
        )
    else:
        query = (
            query.where(and_(
                PullRequestReviewComment.created_at.between(time_from, time_to),
                PullRequestReviewComment.user_login.in_(devs),
                PullRequestReviewComment.repository_full_name.in_(repos),
            ))
        )
    df = await read_sql_query(
        query.group_by(PullRequestReviewComment.user_login),
        db, [PullRequestReviewComment.user_login.key, "comments_count"],
        index=PullRequestReviewComment.user_login.key)
    df.fillna(0, inplace=True, downcast="infer")
    return df


@cached(
    exptime=CACHE_EXPIRATION_TIME,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda devs, repos, time_from, time_to, **_: (
        ",".join(devs), ",".join(sorted(repos)), time_from.toordinal(), time_to.toordinal()),
)
async def _fetch_developer_regular_pr_comments(devs: Sequence[str],
                                               repos: Collection[str],
                                               labels: Set[str],
                                               time_from: datetime,
                                               time_to: datetime,
                                               db: databases.core.Connection,
                                               cache: Optional[aiomcache.Client],
                                               ) -> pd.DataFrame:
    query = select([PullRequestComment.user_login, func.count(PullRequestComment.created_at)])
    if labels:
        query = (
            query.select_from(join(
                PullRequestComment, PullRequestLabel,
                PullRequestComment.pull_request_node_id == PullRequestLabel.pull_request_node_id))
            .where(and_(
                PullRequestComment.created_at.between(time_from, time_to),
                PullRequestComment.user_login.in_(devs),
                PullRequestComment.repository_full_name.in_(repos),
                PullRequestLabel.name.in_(labels),
            ))
        )
    else:
        query = (
            query.where(and_(
                PullRequestComment.created_at.between(time_from, time_to),
                PullRequestComment.user_login.in_(devs),
                PullRequestComment.repository_full_name.in_(repos),
            ))
        )
    df = await read_sql_query(
        query.group_by(PullRequestComment.user_login),
        db, [PullRequestComment.user_login.key, "comments_count"],
        index=PullRequestComment.user_login.key)
    df.fillna(0, inplace=True, downcast="infer")
    return df
