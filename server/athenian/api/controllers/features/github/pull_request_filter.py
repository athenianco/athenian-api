import asyncio
from datetime import datetime, timedelta, timezone
from functools import partial
import logging
import pickle
from typing import Callable, Dict, Generator, Iterable, List, Optional, Set, Tuple, Union

import aiomcache
import databases
import numpy as np
import pandas as pd
import sentry_sdk
from sqlalchemy import and_, or_, select

from athenian.api import metadata
from athenian.api.async_read_sql_query import read_sql_query
from athenian.api.cache import cached, CancelCache
from athenian.api.controllers.datetime_utils import coarsen_time_interval
from athenian.api.controllers.features.github.pull_request_metrics import \
    MergingTimeCalculator, ReleaseTimeCalculator, ReviewTimeCalculator, \
    WorkInProgressTimeCalculator
from athenian.api.controllers.miners.github.branches import extract_branches
from athenian.api.controllers.miners.github.precomputed_prs import discover_unreleased_prs, \
    load_precomputed_done_times_filters, load_precomputed_done_times_reponums, \
    store_precomputed_done_times
from athenian.api.controllers.miners.github.pull_request import dtmin, ImpossiblePullRequest, \
    PullRequestMiner, PullRequestTimesMiner, ReviewResolution
from athenian.api.controllers.miners.github.release import dummy_releases_df, load_releases
from athenian.api.controllers.miners.types import Label, MinedPullRequest, Participants, \
    Property, PullRequestListItem, PullRequestTimes
from athenian.api.controllers.settings import ReleaseMatchSetting
from athenian.api.db import set_pdb_hits, set_pdb_misses
from athenian.api.models.metadata import PREFIXES
from athenian.api.models.metadata.github import PullRequest, PullRequestCommit, PullRequestLabel, \
    PullRequestReview, PullRequestReviewComment, Release
from athenian.api.tracing import sentry_span


class PullRequestListMiner:
    """Collect various PR metadata for displaying PRs on the frontend."""

    _prefix = PREFIXES["github"]
    log = logging.getLogger("%s.PullRequestListMiner" % metadata.__version__)

    def __init__(self,
                 prs_time_machine: Iterable[MinedPullRequest],
                 prs_today: Iterable[MinedPullRequest],
                 precomputed_times: Dict[str, PullRequestTimes],
                 properties: Set[Property],
                 time_from: datetime):
        """Initialize a new instance of `PullRequestListMiner`."""
        self._prs_time_machine = prs_time_machine
        self._prs_today = prs_today
        self._precomputed_times = precomputed_times
        self._times_miner = PullRequestTimesMiner()
        self._properties = properties
        self._calcs = {
            "wip": (WorkInProgressTimeCalculator(), Property.WIP),
            "review": (ReviewTimeCalculator(), Property.REVIEWING),
            "merge": (MergingTimeCalculator(), Property.MERGING),
            "release": (ReleaseTimeCalculator(), Property.RELEASING),
        }
        self._no_time_from = datetime(year=1970, month=1, day=1, tzinfo=timezone.utc)
        assert isinstance(time_from, datetime)
        self._time_from = time_from
        self._now = datetime.now(tz=timezone.utc)
        self._precomputed_hits = self._precomputed_misses = 0
        self._calculated_released_times = []

    @property
    def precomputed_hits(self) -> int:
        """Return the number of used precomputed PullRequestTimes."""
        return self._precomputed_hits

    @property
    def precomputed_misses(self) -> int:
        """Return the number of times PullRequestTimes was calculated from scratch."""
        return self._precomputed_misses

    @property
    def calculated_released_times(self) -> List[Tuple[MinedPullRequest, PullRequestTimes]]:
        """Return the pairs of calculated mined and released PR times."""
        return self._calculated_released_times

    @classmethod
    def _collect_properties(cls,
                            times: PullRequestTimes,
                            pr: MinedPullRequest,
                            time_from: datetime,
                            ) -> Set[Property]:
        author = pr.pr[PullRequest.user_login.key]
        props = set()
        if times.released or (times.closed and not times.merged):
            props.add(Property.DONE)
        elif times.merged:
            props.add(Property.RELEASING)
        elif times.approved:
            props.add(Property.MERGING)
        elif times.first_review_request:
            props.add(Property.REVIEWING)
        else:
            props.add(Property.WIP)
        if times.created.best > time_from:
            props.add(Property.CREATED)
        if (pr.commits[PullRequestCommit.committed_date.key] > time_from).any():
            props.add(Property.COMMIT_HAPPENED)
        review_submitted_ats = pr.reviews[PullRequestReview.submitted_at.key]
        if ((review_submitted_ats > time_from)
                & (pr.reviews[PullRequestReview.user_login.key] != author)).any():
            props.add(Property.REVIEW_HAPPENED)
        if times.first_review_request.value is not None and \
                times.first_review_request.value > time_from:
            props.add(Property.REVIEW_REQUEST_HAPPENED)
        if times.approved and times.approved.best > time_from:
            props.add(Property.APPROVE_HAPPENED)
        if times.merged and times.merged.best > time_from:
            props.add(Property.MERGE_HAPPENED)
        if not times.merged and times.closed and times.closed.best > time_from:
            props.add(Property.REJECTION_HAPPENED)
        if times.released and times.released.best > time_from:
            props.add(Property.RELEASE_HAPPENED)
        review_states = pr.reviews[PullRequestReview.state.key]
        if ((review_states.values == ReviewResolution.CHANGES_REQUESTED.value)
                & (review_submitted_ats > time_from).values).any():
            props.add(Property.CHANGES_REQUEST_HAPPENED)
        return props

    def _compile(self,
                 pr_time_machine: MinedPullRequest,
                 times_time_machine: PullRequestTimes,
                 pr_today: MinedPullRequest,
                 times_today: Union[PullRequestTimes, Callable[[], PullRequestTimes]],
                 ) -> Optional[PullRequestListItem]:
        """
        Match the PR to the required participants and properties.

        :param pr_time_machine: PR's metadata as of `time_to`.
        :param pr_today: Today's version of the PR's metadata.
        :param times_time_machine: Facts about the PR corresponding to [`time_from`, `time_to`].
        :param times_today: Facts about the PR as of datetime.now().
        """
        assert pr_time_machine.pr[PullRequest.node_id.key] == pr_today.pr[PullRequest.node_id.key]
        props_time_machine = self._collect_properties(
            times_time_machine, pr_time_machine, self._time_from)
        if not self._properties.intersection(props_time_machine):
            return None
        if callable(times_today):
            times_today = times_today()
            if times_today.released:
                self._calculated_released_times.append((pr_today, times_today))
        props_today = self._collect_properties(times_today, pr_today, self._no_time_from)
        author = pr_today.pr[PullRequest.user_id.key]
        external_reviews_mask = pr_today.reviews[PullRequestReview.user_id.key].values != author
        first_review = dtmin(pr_today.reviews[PullRequestReview.created_at.key].take(
            np.where(external_reviews_mask)[0]).min())
        review_comments = (
            pr_today.review_comments[PullRequestReviewComment.user_id.key].values != author
        ).sum()
        delta_comments = len(pr_today.review_comments) - review_comments
        reviews = external_reviews_mask.sum()
        stage_timings = {}
        no_time_from = self._no_time_from
        now = self._now
        for k, (calc, prop) in self._calcs.items():
            kwargs = {} if k != "review" else {"allow_unclosed": True}
            if prop in props_today:
                kwargs["override_event_time"] = now - timedelta(seconds=1)  # < time_max
            stage_timings[k] = calc.analyze(times_today, no_time_from, now, **kwargs)
        for p in range(Property.WIP, Property.DONE + 1):
            p = Property(p)
            if p in props_time_machine:
                props_today.add(p)
            else:
                try:
                    props_today.remove(p)
                except KeyError:
                    pass
        updated_at = pr_today.pr[PullRequest.updated_at.key]
        assert updated_at == updated_at
        if pr_today.labels.empty:
            labels = None
        else:
            labels = [
                Label(name=name, description=description, color=color)
                for name, description, color in zip(
                    pr_today.labels[PullRequestLabel.name.key].values,
                    pr_today.labels[PullRequestLabel.description.key].values,
                    pr_today.labels[PullRequestLabel.color.key].values,
                )]
        return PullRequestListItem(
            repository=self._prefix + pr_today.pr[PullRequest.repository_full_name.key],
            number=pr_today.pr[PullRequest.number.key],
            title=pr_today.pr[PullRequest.title.key],
            size_added=pr_today.pr[PullRequest.additions.key],
            size_removed=pr_today.pr[PullRequest.deletions.key],
            files_changed=pr_today.pr[PullRequest.changed_files.key],
            created=pr_today.pr[PullRequest.created_at.key],
            updated=updated_at,
            closed=times_today.closed.best,
            comments=len(pr_today.comments) + delta_comments,
            commits=len(pr_today.commits),
            review_requested=times_today.first_review_request.value,
            first_review=first_review,
            approved=times_today.approved.best,
            review_comments=review_comments,
            reviews=reviews,
            merged=times_today.merged.best,
            released=times_today.released.best,
            release_url=pr_today.release[Release.url.key],
            properties=props_today,
            stage_timings=stage_timings,
            participants=pr_today.participants(),
            labels=labels,
        )

    def __iter__(self) -> Generator[PullRequestListItem, None, None]:
        """Iterate over the individual pull requests."""
        evals = 0
        for pr_time_machine, pr_today in zip(self._prs_time_machine, self._prs_today):
            try:
                times_time_machine = times_today = \
                    self._precomputed_times[pr_today.pr[PullRequest.node_id.key]]
            except KeyError:
                times_time_machine = self._times_miner(pr_time_machine)
                times_today = partial(self._times_miner, pr_today)
                evals += 1
            try:
                item = self._compile(pr_time_machine, times_time_machine, pr_today, times_today)
            except ImpossiblePullRequest:
                continue
            if item is not None:
                yield item
        self._precomputed_hits = len(self._prs_today) - evals
        self._precomputed_misses = evals


@sentry_span
async def _refresh_prs_to_today(prs_time_machine: List[MinedPullRequest],
                                time_to: datetime,
                                repos: Set[str],
                                branches: pd.DataFrame,
                                default_branches: Dict[str, str],
                                release_settings: Dict[str, ReleaseMatchSetting],
                                mdb: databases.Database,
                                pdb: databases.Database,
                                cache: Optional[aiomcache.Client],
                                ) -> Tuple[List[MinedPullRequest], List[MinedPullRequest]]:
    now = datetime.now(tz=timezone.utc)
    merged_at_key = PullRequest.merged_at.key
    closed_at_key = PullRequest.closed_at.key
    node_id_key = PullRequest.node_id.key
    remined = {}
    done = []
    for pr in prs_time_machine:
        if (pr.release[Release.published_at.key] is None and
                (not pd.isnull(pr.pr[merged_at_key]) or pd.isnull(pr.pr[closed_at_key]))):
            remined[pr.pr[node_id_key]] = pr
        else:
            done.append(pr)
    tasks = []
    if done:
        # updated_at can be outside of `time_to` and missed in the cache
        tasks.append(mdb.fetch_all(
            select([PullRequest.node_id, PullRequest.updated_at])
            .where(PullRequest.node_id.in_([pr.pr[node_id_key] for pr in done]))))
    if remined:
        tasks.extend([
            read_sql_query(select([PullRequest])
                           .where(PullRequest.node_id.in_(remined))
                           .order_by(PullRequest.node_id),
                           mdb, PullRequest, index=node_id_key),
            # `time_to` is in the place of `time_from` because we know that these PRs
            # were not released before `time_to`
            load_releases(repos, branches, default_branches, time_to, now, release_settings,
                          mdb, pdb, cache),
        ])
    if tasks:
        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in task_results:
            if isinstance(r, Exception):
                raise r from None
    else:
        task_results = []
    if done:
        updates = task_results[0]
        updates = {p[0]: p[1] for p in updates}
        updated_at_key = PullRequest.updated_at.key
        for pr in done:
            ts = updates[pr.pr[node_id_key]]
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            pr.pr[updated_at_key] = ts
    if remined:
        prs, releases = task_results[-2:]
        releases, matched_bys = releases
        dfs = await PullRequestMiner.mine_by_ids(
            prs, [], now, releases, matched_bys, default_branches,
            release_settings, mdb, pdb, cache)
        prs_today = list(PullRequestMiner(prs, *dfs))
    else:
        prs_today = []
    # reorder prs_time_machine without really changing the contents
    prs_time_machine = [remined[pr.pr[node_id_key]] for pr in prs_today] + done
    prs_today += done
    return prs_today, prs_time_machine


@sentry_span
async def filter_pull_requests(properties: Set[Property],
                               time_from: datetime,
                               time_to: datetime,
                               repos: Set[str],
                               participants: Participants,
                               labels: Set[str],
                               exclude_inactive: bool,
                               release_settings: Dict[str, ReleaseMatchSetting],
                               mdb: databases.Database,
                               pdb: databases.Database,
                               cache: Optional[aiomcache.Client],
                               ) -> List[PullRequestListItem]:
    """Filter GitHub pull requests according to the specified criteria.

    We call _filter_pull_requests() to ignore all but the first result. We've got
    @cached.postprocess inside and it requires the wrapped function to return all the relevant
    post-load dependencies.

    :param repos: List of repository names without the service prefix.
    """
    prs, _ = await _filter_pull_requests(
        properties, time_from, time_to, repos, participants, labels, exclude_inactive,
        release_settings, mdb, pdb, cache)
    return prs


def _postprocess_filtered_prs(result: Tuple[List[PullRequestListItem], Set[str]],
                              labels: Set[str], **_):
    prs, cached_labels = result
    if cached_labels == labels:
        return prs, labels
    if cached_labels and (not labels or labels - cached_labels):
        raise CancelCache()
    prs = [pr for pr in prs if labels.intersection({label.name for label in (pr.labels or [])})]
    return prs, labels


@cached(
    exptime=PullRequestMiner.CACHE_TTL,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda time_from, time_to, repos, properties, participants, exclude_inactive, release_settings, **_: (  # noqa
        time_from.timestamp(),
        time_to.timestamp(),
        ",".join(sorted(repos)),
        ",".join(s.name.lower() for s in sorted(properties)),
        sorted((k.name.lower(), sorted(v)) for k, v in participants.items()),
        exclude_inactive,
        release_settings,
    ),
    postprocess=_postprocess_filtered_prs,
)
async def _filter_pull_requests(properties: Set[Property],
                                time_from: datetime,
                                time_to: datetime,
                                repos: Set[str],
                                participants: Participants,
                                labels: Set[str],
                                exclude_inactive: bool,
                                release_settings: Dict[str, ReleaseMatchSetting],
                                mdb: databases.Database,
                                pdb: databases.Database,
                                cache: Optional[aiomcache.Client],
                                ) -> Tuple[List[PullRequestListItem], Set[str]]:
    assert isinstance(properties, set)
    assert isinstance(repos, set)
    assert isinstance(labels, set)
    log = logging.getLogger("%s.filter_pull_requests" % metadata.__package__)
    # required to efficiently use the cache with timezones
    date_from, date_to = coarsen_time_interval(time_from, time_to)
    branches, default_branches = await extract_branches(repos, mdb, cache)
    tasks = (
        PullRequestMiner.mine(
            date_from, date_to, time_from, time_to, repos, participants, labels, branches,
            default_branches, exclude_inactive, release_settings, mdb, pdb, cache),
        load_precomputed_done_times_filters(
            time_from, time_to, repos, participants, labels, default_branches, exclude_inactive,
            release_settings, pdb),
    )
    miner_time_machine, done_times = await asyncio.gather(*tasks, return_exceptions=True)
    if isinstance(miner_time_machine, Exception):
        raise miner_time_machine from None
    if isinstance(done_times, Exception):
        raise done_times from None
    prs_time_machine = list(miner_time_machine)
    if time_to < datetime.now(tz=timezone.utc):
        prs_today, prs_time_machine = await _refresh_prs_to_today(
            prs_time_machine, time_to, repos, branches, default_branches, release_settings,
            mdb, pdb, cache)
    else:
        prs_today = prs_time_machine
    miner = PullRequestListMiner(prs_time_machine, prs_today, done_times, properties, time_from)
    with sentry_sdk.start_span(op="PullRequestListMiner.__iter__"):
        prs = list(miner)
    set_pdb_hits(pdb, "filter_pull_requests/times", miner.precomputed_hits)
    set_pdb_misses(pdb, "filter_pull_requests/times", miner.precomputed_misses)
    if miner.calculated_released_times:
        await store_precomputed_done_times(*zip(*miner.calculated_released_times),
                                           default_branches, release_settings, pdb)
    log.debug("return %d PRs", len(prs))
    return prs, labels


@sentry_span
@cached(
    exptime=PullRequestMiner.CACHE_TTL,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda prs, release_settings, **_: (  # noqa
        ";".join("%s:%s" % (repo, ",".join(map(str, sorted(numbers))))
                 for repo, numbers in sorted(prs.items())),
        release_settings,
    ),
)
async def fetch_pull_requests(prs: Dict[str, Set[int]],
                              release_settings: Dict[str, ReleaseMatchSetting],
                              mdb: databases.Database,
                              pdb: databases.Database,
                              cache: Optional[aiomcache.Client],
                              ) -> List[PullRequestListItem]:
    """
    List GitHub pull requests by repository and numbers.

    :params prs: For each repository name without the prefix, there is a set of PR numbers to list.
    """
    branches, default_branches = await extract_branches(prs, mdb, cache)
    filters = [and_(PullRequest.repository_full_name == repo, PullRequest.number.in_(numbers))
               for repo, numbers in prs.items()]
    tasks = [
        read_sql_query(select([PullRequest])
                       .where(or_(*filters))
                       .order_by(PullRequest.node_id),
                       mdb, PullRequest, index=PullRequest.node_id.key),
        load_precomputed_done_times_reponums(prs, default_branches, release_settings, pdb),
    ]
    prs_df, done_times = await asyncio.gather(*tasks, return_exceptions=True)
    for r in (prs, done_times):
        if isinstance(r, Exception):
            raise r from None
    if prs_df.empty:
        return []
    now = datetime.now(timezone.utc)
    rel_time_from = prs_df[PullRequest.merged_at.key].min()
    if rel_time_from == rel_time_from:
        releases, matched_bys = await load_releases(
            prs, branches, default_branches, rel_time_from, now, release_settings, mdb, pdb, cache)
        unreleased = await discover_unreleased_prs(
            prs_df, releases, matched_bys, default_branches, release_settings, pdb)
    else:
        releases, matched_bys, unreleased = dummy_releases_df(), {}, []
    dfs = await PullRequestMiner.mine_by_ids(
        prs_df, unreleased, now, releases, matched_bys, default_branches,
        release_settings, mdb, pdb, cache)
    prs = list(PullRequestMiner(prs_df, *dfs))
    miner = PullRequestListMiner(
        prs, prs, done_times, set(Property), prs_df[PullRequest.created_at.key].min())
    with sentry_sdk.start_span(op="PullRequestListMiner.__iter__"):
        prs = list(miner)
    set_pdb_hits(pdb, "filter_pull_requests/times", miner.precomputed_hits)
    set_pdb_misses(pdb, "filter_pull_requests/times", miner.precomputed_misses)
    return prs
