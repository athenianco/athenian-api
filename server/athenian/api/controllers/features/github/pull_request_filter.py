import asyncio
from dataclasses import fields
from datetime import datetime, timedelta, timezone
import logging
import pickle
from typing import Callable, Collection, Dict, Generator, Iterable, List, Optional, Sequence, \
    Set, Tuple

import aiomcache
import numpy as np
import pandas as pd
import sentry_sdk
from sqlalchemy import and_, select, union_all

from athenian.api import COROUTINE_YIELD_EVERY_ITER, list_with_yield, metadata
from athenian.api.async_utils import gather, read_sql_query
from athenian.api.cache import cached, CancelCache
from athenian.api.controllers.datetime_utils import coarsen_time_interval
from athenian.api.controllers.features.github.check_run_metrics import \
    MergedPRsWithFailedChecksCounter
from athenian.api.controllers.features.github.pull_request_metrics import AllCounter, \
    MergingPendingCounter, MergingTimeCalculator, ReleasePendingCounter, ReleaseTimeCalculator, \
    ReviewPendingCounter, ReviewTimeCalculator, StagePendingDependencyCalculator, \
    WorkInProgressPendingCounter, WorkInProgressTimeCalculator
from athenian.api.controllers.features.metric import Metric
from athenian.api.controllers.features.metric_calculator import MetricCalculator
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.bots import bots
from athenian.api.controllers.miners.github.branches import BranchMiner
from athenian.api.controllers.miners.github.check_run import mine_check_runs
from athenian.api.controllers.miners.github.commit import BRANCH_FETCH_COMMITS_COLUMNS, \
    fetch_precomputed_commit_history_dags, fetch_repository_commits_no_branch_dates
from athenian.api.controllers.miners.github.dag_accelerated import searchsorted_inrange
from athenian.api.controllers.miners.github.deployment import load_included_deployments
from athenian.api.controllers.miners.github.precomputed_prs import \
    DonePRFactsLoader, MergedPRFactsLoader, remove_ambiguous_prs, \
    store_merged_unreleased_pull_request_facts, store_open_pull_request_facts, \
    store_precomputed_done_facts
from athenian.api.controllers.miners.github.pull_request import ImpossiblePullRequest, \
    PRDataFrames, PullRequestFactsMiner, PullRequestMiner, ReviewResolution
from athenian.api.controllers.miners.github.release_load import dummy_releases_df, ReleaseLoader
from athenian.api.controllers.miners.github.release_match import load_commit_dags
from athenian.api.controllers.miners.types import Deployment, Label, MinedPullRequest, \
    PRParticipants, PullRequestEvent, PullRequestFacts, PullRequestJIRAIssueItem, \
    PullRequestListItem, PullRequestStage
from athenian.api.controllers.prefixer import PrefixerPromise
from athenian.api.controllers.settings import ReleaseMatch, ReleaseSettings
from athenian.api.db import add_pdb_misses, ParallelDatabase, set_pdb_hits, set_pdb_misses
from athenian.api.defer import defer
from athenian.api.models.metadata.github import CheckRun, PullRequest, PullRequestCommit, \
    PullRequestLabel, PullRequestReview, PullRequestReviewComment, Release
from athenian.api.models.metadata.jira import Issue
from athenian.api.tracing import sentry_span
from athenian.api.typing_utils import df_from_structs


class PullRequestListMiner:
    """Collect various PR metadata for displaying PRs on the frontend."""

    _no_time_from = datetime(year=1970, month=1, day=1)
    log = logging.getLogger("%s.PullRequestListMiner" % metadata.__version__)

    class DummyAllCounter(AllCounter):
        """Fool StagePendingDependencyCalculator so that it does not exclude any PRs."""

        dtype = int

        def _analyze(self,
                     facts: pd.DataFrame,
                     min_times: np.ndarray,
                     max_times: np.ndarray,
                     **kwargs) -> np.ndarray:
            return np.full((len(min_times), len(facts)), 1)

        def _value(self, samples: np.ndarray) -> Metric[int]:
            raise AssertionError("This method should never be called.")

    def __init__(self,
                 prs: List[MinedPullRequest],
                 dfs: PRDataFrames,
                 facts: Dict[str, PullRequestFacts],
                 events: Set[PullRequestEvent],
                 stages: Set[PullRequestStage],
                 time_from: datetime,
                 time_to: datetime,
                 with_time_machine: bool):
        """Initialize a new instance of `PullRequestListMiner`."""
        self._prs = prs
        self._dfs = dfs
        self._facts = facts
        self._events = events
        self._stages = stages
        self._calcs, self._counter_deps = self.create_stage_calcs()
        assert isinstance(time_from, datetime)
        assert time_from.tzinfo == timezone.utc
        assert isinstance(time_to, datetime)
        assert time_to.tzinfo == timezone.utc
        self._time_from = time_from.replace(tzinfo=None)
        self._time_to = time_to.replace(tzinfo=None)
        self._with_time_machine = with_time_machine

    @classmethod
    def create_stage_calcs(cls) -> Tuple[Dict[str, Dict[str, MetricCalculator[int]]],
                                         Iterable[MetricCalculator]]:
        """Intialize PR metric calculators needed to calculate the stage timings."""
        all_counter = cls.DummyAllCounter(quantiles=(0, 1))
        pending_counter = StagePendingDependencyCalculator(all_counter, quantiles=(0, 1))
        return {
            "wip": {
                "time": WorkInProgressTimeCalculator(quantiles=(0, 1)),
                "pending_count": WorkInProgressPendingCounter(
                    pending_counter, quantiles=(0, 1)),
            },
            "review": {
                "time": ReviewTimeCalculator(quantiles=(0, 1)),
                "pending_count": ReviewPendingCounter(
                    pending_counter, quantiles=(0, 1)),
            },
            "merge": {
                "time": MergingTimeCalculator(quantiles=(0, 1)),
                "pending_count": MergingPendingCounter(
                    pending_counter, quantiles=(0, 1)),
            },
            "release": {
                "time": ReleaseTimeCalculator(quantiles=(0, 1)),
                "pending_count": ReleasePendingCounter(
                    pending_counter, quantiles=(0, 1)),
            },
        }, (all_counter, pending_counter)

    @classmethod
    def _collect_events_and_stages(cls,
                                   facts: PullRequestFacts,
                                   hard_events: Dict[PullRequestEvent, bool],
                                   time_from: datetime,
                                   ) -> Tuple[Set[PullRequestEvent], Set[PullRequestStage]]:
        events = set()
        stages = set()
        if facts.done:
            if facts.force_push_dropped:
                stages.add(PullRequestStage.FORCE_PUSH_DROPPED)
            stages.add(PullRequestStage.DONE)
        elif facts.merged:
            stages.add(PullRequestStage.RELEASING)
        elif facts.approved:
            stages.add(PullRequestStage.MERGING)
        elif facts.first_review_request:
            stages.add(PullRequestStage.REVIEWING)
        else:
            stages.add(PullRequestStage.WIP)
        if facts.created >= time_from:
            events.add(PullRequestEvent.CREATED)
        if hard_events[PullRequestEvent.COMMITTED]:
            events.add(PullRequestEvent.COMMITTED)
        if hard_events[PullRequestEvent.REVIEWED]:
            events.add(PullRequestEvent.REVIEWED)
        if facts.first_review_request_exact and facts.first_review_request_exact >= time_from:
            events.add(PullRequestEvent.REVIEW_REQUESTED)
        if facts.approved and facts.approved >= time_from:
            events.add(PullRequestEvent.APPROVED)
        if facts.merged and facts.merged >= time_from:
            events.add(PullRequestEvent.MERGED)
        if not facts.merged and facts.closed and facts.closed >= time_from:
            events.add(PullRequestEvent.REJECTED)
        if facts.released and facts.released >= time_from:
            events.add(PullRequestEvent.RELEASED)
        if hard_events[PullRequestEvent.CHANGES_REQUESTED]:
            events.add(PullRequestEvent.CHANGES_REQUESTED)
        return events, stages

    def _compile(self,
                 pr: MinedPullRequest,
                 facts: PullRequestFacts,
                 stage_timings: Dict[str, timedelta],
                 hard_events_time_machine: Dict[PullRequestEvent, Set[str]],
                 hard_events_now: Dict[PullRequestEvent, Set[str]],
                 ) -> Optional[PullRequestListItem]:
        """
        Match the PR to the required participants and properties and produce PullRequestListItem.

        We return None if the PR does not match.
        """
        facts_now = facts
        pr_node_id = pr.pr[PullRequest.node_id.name]
        if self._with_time_machine:
            facts_time_machine = facts.truncate(self._time_to)
            events_time_machine, stages_time_machine = self._collect_events_and_stages(
                facts_time_machine,
                {k: (pr_node_id in v) for k, v in hard_events_time_machine.items()},
                self._time_from)
            if self._stages or self._events:
                stages_pass = self._stages and self._stages.intersection(stages_time_machine)
                events_pass = self._events and self._events.intersection(events_time_machine)
                if not (stages_pass or events_pass):
                    return None
        else:
            events_time_machine = stages_time_machine = None
        events_now, stages_now = self._collect_events_and_stages(
            facts_now,
            {k: (pr_node_id in v) for k, v in hard_events_now.items()},
            self._no_time_from)
        author = pr.pr[PullRequest.user_login.name]
        external_reviews_mask = pr.reviews[PullRequestReview.user_login.name].values != author
        external_review_times = \
            pr.reviews[PullRequestReview.created_at.name].values[external_reviews_mask]
        first_review = pd.Timestamp(external_review_times.min(), tz=timezone.utc) \
            if len(external_review_times) > 0 else None
        review_comments = (
            pr.review_comments[PullRequestReviewComment.user_login.name].values != author
        ).sum()
        delta_comments = len(pr.review_comments) - review_comments
        reviews = external_reviews_mask.sum()
        updated_at = pr.pr[PullRequest.updated_at.name]
        assert updated_at == updated_at
        if pr.labels.empty:
            labels = None
        else:
            labels = [
                Label(name=name, description=description, color=color)
                for name, description, color in zip(
                    pr.labels[PullRequestLabel.name.name].values,
                    pr.labels[PullRequestLabel.description.name].values,
                    pr.labels[PullRequestLabel.color.name].values,
                )
            ]
        if pr.jiras.empty:
            jira = None
        else:
            jira = [
                PullRequestJIRAIssueItem(id=key,
                                         title=title,
                                         epic=epic,
                                         labels=labels or None,
                                         type=itype)
                for (key, title, epic, labels, itype) in zip(
                    pr.jiras.index.values,
                    pr.jiras[Issue.title.name].values,
                    pr.jiras["epic"].values,
                    pr.jiras[Issue.labels.name].values,
                    pr.jiras[Issue.type.name].values,
                )
            ]
        deployments = np.sort(pr.deployments.index.values) if len(pr.deployments.index) else None
        return PullRequestListItem(
            node_id=pr_node_id,
            repository=pr.pr[PullRequest.repository_full_name.name],
            number=pr.pr[PullRequest.number.name],
            title=pr.pr[PullRequest.title.name],
            size_added=pr.pr[PullRequest.additions.name],
            size_removed=pr.pr[PullRequest.deletions.name],
            files_changed=pr.pr[PullRequest.changed_files.name],
            created=pr.pr[PullRequest.created_at.name],
            updated=updated_at,
            closed=self._dt64_to_pydt(facts_now.closed),
            comments=len(pr.comments) + delta_comments,
            commits=len(pr.commits),
            review_requested=self._dt64_to_pydt(facts_now.first_review_request_exact),
            first_review=first_review,
            approved=self._dt64_to_pydt(facts_now.approved),
            review_comments=review_comments,
            reviews=reviews,
            merged=self._dt64_to_pydt(facts_now.merged),
            merged_with_failed_check_runs=None,
            released=self._dt64_to_pydt(facts_now.released),
            release_url=pr.release[Release.url.name],
            events_now=events_now,
            stages_now=stages_now,
            events_time_machine=events_time_machine,
            stages_time_machine=stages_time_machine,
            stage_timings=stage_timings,
            participants=pr.participant_logins(),
            labels=labels,
            jira=jira,
            deployments=deployments,
        )

    def __iter__(self) -> Generator[PullRequestListItem, None, None]:
        """Iterate over individual pull requests."""
        if not self._prs:
            return
        stage_timings = self._calc_stage_timings()
        hard_events_time_machine, hard_events_now = self._calc_hard_events()
        facts = self._facts
        node_id_key = PullRequest.node_id.name
        for i, pr in enumerate(self._prs):
            pr_stage_timings = {}
            for k in self._calcs:
                seconds = stage_timings[k][i]
                if seconds is not None:
                    pr_stage_timings[k] = seconds.item()
            item = self._compile(pr, facts[pr.pr[node_id_key]], pr_stage_timings,
                                 hard_events_time_machine, hard_events_now)
            if item is not None:
                yield item

    @sentry_span
    def _calc_stage_timings(self) -> Dict[str, Sequence[int]]:
        facts = self._facts
        if len(facts) == 0 or len(self._prs) == 0:
            return {k: [] for k in self._calcs}
        node_id_key = PullRequest.node_id.name
        df_facts = df_from_structs(
            (facts[pr.pr[node_id_key]] for pr in self._prs), length=len(self._prs))
        return self.calc_stage_timings(df_facts, self._calcs, self._counter_deps)

    @classmethod
    def calc_stage_timings(cls,
                           df_facts: pd.DataFrame,
                           calcs: Dict[str, Dict[str, MetricCalculator[int]]],
                           counter_deps: Iterable[MetricCalculator],
                           ) -> Dict[str, Sequence[int]]:
        """
        Calculate PR stage timings.

        :return: Map from PR node IDs to stage timings (in seconds).
        """
        now = datetime.now(tz=timezone.utc)
        dtype = df_facts["created"].dtype
        no_time_from = np.array([cls._no_time_from], dtype=dtype)
        now = np.array([now.replace(tzinfo=None)], dtype=dtype)
        stage_timings = {}
        # makes `samples` empty, we don't need them
        empty_group_mask = np.zeros((1, len(df_facts)), dtype=bool)
        for dep in counter_deps:
            dep(df_facts, no_time_from, now, None, empty_group_mask)
        for k, calcs in calcs.items():
            time_calc, pending_counter = calcs["time"], calcs["pending_count"]
            pending_counter(df_facts, no_time_from, now, None, empty_group_mask)

            kwargs = {
                "override_event_time": now - np.timedelta64(timedelta(seconds=1)),  # < time_max
                "override_event_indexes": np.nonzero(pending_counter.peek[0])[0],
            }
            if k == "review":
                kwargs["allow_unclosed"] = True
            time_calc(df_facts, no_time_from, now, None, empty_group_mask, **kwargs)
            stage_timings[k] = time_calc.peek[0]
        return stage_timings

    @sentry_span
    def _calc_hard_events(self) -> Tuple[Dict[PullRequestEvent, Set[str]],
                                         Dict[PullRequestEvent, Set[str]]]:
        events_time_machine = {}
        events_now = {}
        dfs = self._dfs
        time_from = np.datetime64(self._time_from)
        time_to = np.datetime64(self._time_to)

        index = dfs.commits.index.get_level_values(0).values
        committed_date = dfs.commits[PullRequestCommit.committed_date.name].values
        events_now[PullRequestEvent.COMMITTED] = set(index[committed_date == committed_date])
        if self._with_time_machine:
            events_time_machine[PullRequestEvent.COMMITTED] = \
                set(index[(committed_date >= time_from) & (committed_date < time_to)])

        prr_ulk = PullRequestReview.user_login.name
        prr_sak = PullRequestReview.submitted_at.name
        reviews = dfs.reviews[[prr_ulk, prr_sak]].droplevel(1)
        original_reviews_index = reviews.index.values
        reviews = reviews.join(dfs.prs[[PullRequest.user_login.name, PullRequest.closed_at.name]],
                               rsuffix="_pr")
        index = reviews.index.values
        submitted_at = reviews[prr_sak].values
        closed_at = reviews[PullRequest.closed_at.name].values
        not_closed_mask = closed_at != closed_at
        closed_at[not_closed_mask] = submitted_at[not_closed_mask]
        closed_at = closed_at.astype(submitted_at.dtype)
        mask = (
            (submitted_at <= closed_at)
            &
            (reviews[prr_ulk].values != reviews[PullRequest.user_login.name + "_pr"].values)
        )
        events_now[PullRequestEvent.REVIEWED] = set(index[mask])
        if self._with_time_machine:
            mask &= (submitted_at >= time_from) & (submitted_at < time_to)
            events_time_machine[PullRequestEvent.REVIEWED] = set(index[mask])

        submitted_at = dfs.reviews[prr_sak].values
        # no need to check submitted_at <= closed_at because GitHub disallows that
        mask = dfs.reviews[PullRequestReview.state.name].values == \
            ReviewResolution.CHANGES_REQUESTED.value
        events_now[PullRequestEvent.CHANGES_REQUESTED] = set(original_reviews_index[mask])
        if self._with_time_machine:
            mask &= (submitted_at >= time_from) & (submitted_at < time_to)
            events_time_machine[PullRequestEvent.CHANGES_REQUESTED] = \
                set(original_reviews_index[mask])

        return events_time_machine, events_now

    @staticmethod
    def _dt64_to_pydt(dt: Optional[np.datetime64]) -> Optional[datetime]:
        if dt is None:
            return None
        return dt.item().replace(tzinfo=timezone.utc)


@sentry_span
async def filter_pull_requests(events: Set[PullRequestEvent],
                               stages: Set[PullRequestStage],
                               time_from: datetime,
                               time_to: datetime,
                               repos: Set[str],
                               participants: PRParticipants,
                               labels: LabelFilter,
                               jira: JIRAFilter,
                               exclude_inactive: bool,
                               release_settings: ReleaseSettings,
                               updated_min: Optional[datetime],
                               updated_max: Optional[datetime],
                               prefixer: PrefixerPromise,
                               account: int,
                               meta_ids: Tuple[int, ...],
                               mdb: ParallelDatabase,
                               pdb: ParallelDatabase,
                               rdb: ParallelDatabase,
                               cache: Optional[aiomcache.Client],
                               ) -> Tuple[List[PullRequestListItem], Dict[str, Deployment]]:
    """Filter GitHub pull requests according to the specified criteria.

    We call _filter_pull_requests() to ignore all but the first result. We've got
    @cached.postprocess inside and it requires the wrapped function to return all the relevant
    post-load dependencies.

    :param repos: List of repository names without the service prefix.
    :param updated_min: The real time_from since when we fetch PRs.
    :param updated_max: The real time_to until when we fetch PRs.
    """
    prs, deployments, _, _ = await _filter_pull_requests(
        events, stages, time_from, time_to, repos, participants, labels, jira,
        exclude_inactive, release_settings, updated_min, updated_max,
        prefixer, account, meta_ids, mdb, pdb, rdb, cache)
    return prs, deployments


def _postprocess_filtered_prs(result: Tuple[List[PullRequestListItem],
                                            Dict[str, Deployment],
                                            LabelFilter,
                                            JIRAFilter],
                              labels: LabelFilter, jira: JIRAFilter, **_):
    prs, deployments, cached_labels, cached_jira = result
    if (not cached_labels.compatible_with(labels) or
            not cached_jira.compatible_with(jira)):
        raise CancelCache()
    if labels:
        prs = _filter_by_labels(prs, labels, _extract_pr_labels)
    if jira:
        if jira.labels:
            prs = _filter_by_labels(prs, jira.labels, _extract_jira_labels)
        if jira.epics:
            prs = _filter_by_jira_epics(prs, jira.epics)
        if jira.issue_types:
            prs = _filter_by_jira_issue_types(prs, jira.issue_types)
    left_deployments = [pr.deployments for pr in prs if pr.deployments]
    if left_deployments:
        left_deployments = np.unique(np.concatenate(left_deployments))
        deployments = {k: deployments[k] for k in left_deployments if k in deployments}
    else:
        deployments = {}
    return prs, deployments, labels, jira


def _extract_pr_labels(pr: PullRequestListItem) -> Optional[Set[str]]:
    if pr.labels is None:
        return None
    return {lbl.name for lbl in pr.labels}


def _extract_jira_labels(pr: PullRequestListItem) -> Optional[Set[str]]:
    if pr.jira is None:
        return None
    labels = set()
    for issue in pr.jira:
        if issue.labels is not None:
            labels.update(s.lower() for s in issue.labels)
    return labels or None


def _filter_by_labels(prs: List[PullRequestListItem],
                      labels: LabelFilter,
                      labels_getter: Callable[[PullRequestListItem], Optional[Set[str]]],
                      ) -> List[PullRequestListItem]:
    if labels.include:
        singles, multiples = LabelFilter.split(labels.include)
        singles = set(singles)
        for i, g in enumerate(multiples):
            multiples[i] = set(g)
        new_prs = []
        for pr in prs:
            pr_labels = labels_getter(pr)
            if pr_labels is None or not singles.intersection(pr_labels):
                continue
            for group in multiples:
                if not pr_labels.issuperset(group):
                    break
            else:
                new_prs.append(pr)
        prs = new_prs
    if labels.exclude:
        new_prs = []
        for pr in prs:
            pr_labels = labels_getter(pr)
            if pr_labels is not None and labels.exclude.intersection(pr_labels):
                continue
            new_prs.append(pr)
        prs = new_prs
    return prs


def _filter_by_jira_epics(prs: List[PullRequestListItem],
                          epics: Set[str],
                          ) -> List[PullRequestListItem]:
    new_prs = []
    for pr in prs:
        if pr.jira is None:
            continue
        for issue in pr.jira:
            if issue.epic in epics:
                new_prs.append(pr)
                break
    return new_prs


def _filter_by_jira_issue_types(prs: List[PullRequestListItem],
                                types: Set[str],
                                ) -> List[PullRequestListItem]:
    new_prs = []
    for pr in prs:
        if pr.jira is None:
            continue
        for issue in pr.jira:
            if issue.type.lower() in types:
                new_prs.append(pr)
                break
    return new_prs


@sentry_span
async def _load_failed_check_runs_for_prs(time_from: datetime,
                                          time_to: datetime,
                                          repos: Collection[str],
                                          labels: LabelFilter,
                                          jira: JIRAFilter,
                                          meta_ids: Tuple[int, ...],
                                          mdb: ParallelDatabase,
                                          cache: Optional[aiomcache.Client],
                                          ) -> Tuple[np.ndarray, List[np.ndarray]]:
    df = await mine_check_runs(
        time_from, time_to, repos, [], labels, jira, meta_ids, mdb, cache)
    index, pull_requests, failure_mask = \
        MergedPRsWithFailedChecksCounter.find_prs_merged_with_failed_check_runs(df)
    check_run_names = df[CheckRun.name.name].values[index.values]
    failed_check_run_names = check_run_names[failure_mask]
    failed_pull_requests = pull_requests[failure_mask]
    unique_prs, index, counts = np.unique(
        failed_pull_requests, return_counts=True, return_inverse=True)
    check_runs = np.split(failed_check_run_names[np.argsort(index)], np.cumsum(counts[:-1]))
    return unique_prs, check_runs


@sentry_span
async def _append_merged_with_failed_check_runs(check_runs_task: Optional[asyncio.Task],
                                                prs: List[PullRequestListItem],
                                                ) -> None:
    if check_runs_task is None:
        return
    await check_runs_task
    failed_prs, check_run_names = check_runs_task.result()
    if len(failed_prs):
        node_ids = np.fromiter((pr.node_id for pr in prs), int, len(prs))
        order = np.argsort(node_ids)
        node_ids = node_ids[order]
        order = np.argsort(order)
        if len(node_ids) > len(failed_prs):
            found = searchsorted_inrange(node_ids, failed_prs)
            failed_indexes = np.flatnonzero(node_ids[found] == failed_prs)
            pr_indexes = found[failed_indexes]
            check_run_indexes = failed_indexes
        else:
            found = searchsorted_inrange(failed_prs, node_ids)
            failed_indexes = np.flatnonzero(failed_prs[found] == node_ids)
            pr_indexes = failed_indexes
            check_run_indexes = found[failed_indexes]
        for pr_index, check_run_index in zip(pr_indexes, check_run_indexes):
            prs[order[pr_index]].merged_with_failed_check_runs = \
                check_run_names[check_run_index].tolist()


@cached(
    exptime=PullRequestMiner.CACHE_TTL,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda time_from, time_to, repos, events, stages, participants, exclude_inactive, release_settings, updated_min, updated_max, **_: (  # noqa
        time_from.timestamp(),
        time_to.timestamp(),
        ",".join(sorted(repos)),
        ",".join(s.name.lower() for s in sorted(events)),
        ",".join(s.name.lower() for s in sorted(stages)),
        sorted((k.name.lower(), sorted(v)) for k, v in participants.items()),
        exclude_inactive,
        updated_min.timestamp() if updated_min is not None else None,
        updated_max.timestamp() if updated_max is not None else None,
        release_settings,
    ),
    postprocess=_postprocess_filtered_prs,
)
async def _filter_pull_requests(events: Set[PullRequestEvent],
                                stages: Set[PullRequestStage],
                                time_from: datetime,
                                time_to: datetime,
                                repos: Set[str],
                                participants: PRParticipants,
                                labels: LabelFilter,
                                jira: JIRAFilter,
                                exclude_inactive: bool,
                                release_settings: ReleaseSettings,
                                updated_min: Optional[datetime],
                                updated_max: Optional[datetime],
                                prefixer: PrefixerPromise,
                                account: int,
                                meta_ids: Tuple[int, ...],
                                mdb: ParallelDatabase,
                                pdb: ParallelDatabase,
                                rdb: ParallelDatabase,
                                cache: Optional[aiomcache.Client],
                                ) -> Tuple[List[PullRequestListItem],
                                           Dict[str, Deployment],
                                           LabelFilter,
                                           JIRAFilter]:
    assert isinstance(events, set)
    assert isinstance(stages, set)
    assert isinstance(repos, set)
    assert (updated_min is None) == (updated_max is None)
    log = logging.getLogger("%s.filter_pull_requests" % metadata.__package__)
    # required to efficiently use the cache with timezones
    date_from, date_to = coarsen_time_interval(time_from, time_to)
    if updated_min is not None:
        coarsen_time_interval(updated_min, updated_max)
    check_runs_task = asyncio.create_task(_load_failed_check_runs_for_prs(
        updated_min or time_from, updated_max or time_to,
        repos, labels, jira, meta_ids, mdb, cache))
    branches, default_branches = await BranchMiner.extract_branches(repos, meta_ids, mdb, cache)
    tasks = (
        PullRequestMiner.mine(
            date_from, date_to, time_from, time_to, repos, participants,
            labels, jira, branches, default_branches, exclude_inactive, release_settings,
            prefixer, account, meta_ids, mdb, pdb, rdb, cache,
            truncate=False, updated_min=updated_min, updated_max=updated_max),
        DonePRFactsLoader.load_precomputed_done_facts_filters(
            time_from, time_to, repos, participants, labels, default_branches,
            exclude_inactive, release_settings, prefixer, account, pdb),
    )
    (pr_miner, unreleased_facts, matched_bys, unreleased_prs_event), (facts, ambiguous) = \
        await gather(*tasks)
    add_pdb_misses(pdb, "load_precomputed_done_facts_filters/ambiguous",
                   remove_ambiguous_prs(facts, ambiguous, matched_bys))
    # we want the released PR facts to overwrite the others
    facts, unreleased_facts = unreleased_facts, facts
    facts.update(unreleased_facts)
    del unreleased_facts

    deployment_names = pr_miner.dfs.deployments.index.get_level_values(1).unique()
    deployments = asyncio.create_task(load_included_deployments(
        deployment_names, account, meta_ids, mdb, rdb, cache),
        name=f"load_included_deployments({len(deployment_names)})")
    prs = await list_with_yield(pr_miner, "PullRequestMiner.__iter__")

    with sentry_sdk.start_span(op="PullRequestFactsMiner.__call__"):
        facts_miner = PullRequestFactsMiner(await bots(mdb))
        missed_done_facts = []
        missed_open_facts = []
        missed_merged_unreleased_facts = []

        async def store_missed_done_facts():
            nonlocal missed_done_facts
            await defer(store_precomputed_done_facts(
                *zip(*missed_done_facts), default_branches, release_settings, account, pdb),
                "store_precomputed_done_facts(%d)" % len(missed_done_facts))
            missed_done_facts = []

        async def store_missed_open_facts():
            nonlocal missed_open_facts
            await defer(store_open_pull_request_facts(missed_open_facts, account, pdb),
                        "store_open_pull_request_facts(%d)" % len(missed_open_facts))
            missed_open_facts = []

        async def store_missed_merged_unreleased_facts():
            nonlocal missed_merged_unreleased_facts
            await defer(store_merged_unreleased_pull_request_facts(
                missed_merged_unreleased_facts, matched_bys, default_branches,
                release_settings, account, pdb, unreleased_prs_event),
                "store_merged_unreleased_pull_request_facts(%d)" %
                len(missed_merged_unreleased_facts))
            missed_merged_unreleased_facts = []

        fact_evals = 0
        hit_facts_counter = 0
        missed_done_facts_counter = missed_open_facts_counter = \
            missed_merged_unreleased_facts_counter = 0
        bad_prs = []
        for i, pr in enumerate(prs):
            node_id = pr.pr[PullRequest.node_id.name]
            if node_id not in facts:
                fact_evals += 1
                if (fact_evals + 1) % COROUTINE_YIELD_EVERY_ITER == 0:
                    await asyncio.sleep(0)
                try:
                    facts[node_id] = pr_facts = facts_miner(pr)
                except ImpossiblePullRequest:
                    bad_prs.insert(0, i)  # reversed order
                    continue
                if pr_facts.released or pr_facts.closed and not pr_facts.merged:
                    missed_done_facts_counter += 1
                    missed_done_facts.append((pr, pr_facts))
                    if (len(missed_done_facts) + 1) % 100 == 0:
                        await store_missed_done_facts()
                elif not pr_facts.closed:
                    missed_open_facts_counter += 1
                    missed_open_facts.append((pr, pr_facts))
                    if (len(missed_open_facts) + 1) % 100 == 0:
                        await store_missed_open_facts()
                elif pr_facts.merged and not pr_facts.released:
                    missed_merged_unreleased_facts_counter += 1
                    missed_merged_unreleased_facts.append((pr, pr_facts))
                    if (len(missed_merged_unreleased_facts) + 1) % 100 == 0:
                        await store_missed_merged_unreleased_facts()
            else:
                hit_facts_counter += 1
        if missed_done_facts:
            await store_missed_done_facts()
        if missed_open_facts:
            await store_missed_open_facts()
        if missed_merged_unreleased_facts:
            await store_missed_merged_unreleased_facts()
        if bad_prs:
            # the order is already reversed
            for i in bad_prs:
                del prs[i]
        set_pdb_hits(pdb, "filter_pull_requests/facts", hit_facts_counter)
        set_pdb_misses(pdb, "filter_pull_requests/done_facts", missed_done_facts_counter)
        set_pdb_misses(pdb, "filter_pull_requests/open_facts", missed_open_facts_counter)
        set_pdb_misses(pdb, "filter_pull_requests/merged_unreleased_facts",
                       missed_merged_unreleased_facts_counter)
        log.info("total fact evals: %d", fact_evals)

    prs = await list_with_yield(
        PullRequestListMiner(prs, pr_miner.dfs, facts, events, stages, time_from, time_to, True),
        "PullRequestListMiner.__iter__",
    )
    await _append_merged_with_failed_check_runs(check_runs_task, prs)
    await deployments
    log.debug("return %d PRs", len(prs))
    return prs, deployments.result(), labels, jira


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
                              release_settings: ReleaseSettings,
                              prefixer: PrefixerPromise,
                              account: int,
                              meta_ids: Tuple[int, ...],
                              mdb: ParallelDatabase,
                              pdb: ParallelDatabase,
                              rdb: ParallelDatabase,
                              cache: Optional[aiomcache.Client],
                              ) -> Tuple[List[PullRequestListItem], Dict[str, Deployment]]:
    """
    List GitHub pull requests by repository and numbers.

    :params prs: For each repository name without the prefix, there is a set of PR numbers to list.
    """
    mined_prs, dfs, facts, _, deployments_task, check_runs_task = await _fetch_pull_requests(
        prs, release_settings, prefixer, account, meta_ids, mdb, pdb, rdb, cache)
    if not mined_prs:
        return [], {}
    miner = PullRequestListMiner(
        mined_prs, dfs, facts, set(), set(),
        datetime(1970, 1, 1, tzinfo=timezone.utc), datetime.now(timezone.utc), False)
    prs = await list_with_yield(miner, "PullRequestListMiner.__iter__")
    await _append_merged_with_failed_check_runs(check_runs_task, prs)
    await deployments_task
    return prs, deployments_task.result()


async def _fetch_pull_requests(prs: Dict[str, Set[int]],
                               release_settings: ReleaseSettings,
                               prefixer: PrefixerPromise,
                               account: int,
                               meta_ids: Tuple[int, ...],
                               mdb: ParallelDatabase,
                               pdb: ParallelDatabase,
                               rdb: ParallelDatabase,
                               cache: Optional[aiomcache.Client],
                               ) -> Tuple[List[MinedPullRequest],
                                          PRDataFrames,
                                          Dict[str, PullRequestFacts],
                                          Dict[str, ReleaseMatch],
                                          asyncio.Task,
                                          Optional[asyncio.Task]]:
    branches, default_branches = await BranchMiner.extract_branches(prs, meta_ids, mdb, cache)
    filters = [and_(PullRequest.repository_full_name == repo,
                    PullRequest.number.in_(numbers),
                    PullRequest.acc_id.in_(meta_ids))
               for repo, numbers in prs.items()]
    queries = [select([PullRequest]).where(f).order_by(PullRequest.node_id) for f in filters]
    tasks = [
        read_sql_query(union_all(*queries) if len(queries) > 1 else queries[0],  # sqlite sucks
                       mdb, PullRequest, index=PullRequest.node_id.name),
        DonePRFactsLoader.load_precomputed_done_facts_reponums(
            prs, default_branches, release_settings, prefixer, account, pdb),
    ]
    prs_df, (facts, ambiguous) = await gather(*tasks)
    if (max_merged_at := prs_df[PullRequest.merged_at.name].max()) == max_merged_at:
        check_runs_task = asyncio.create_task(_load_failed_check_runs_for_prs(
            prs_df[PullRequest.created_at.name].min() - timedelta(hours=1),
            max_merged_at + timedelta(days=1),
            prs.keys(), LabelFilter.empty(), JIRAFilter.empty(), meta_ids, mdb, cache))
    else:
        check_runs_task = None
    unwrapped = await unwrap_pull_requests(
        prs_df, facts, ambiguous, True, branches, default_branches, release_settings,
        prefixer, account, meta_ids, mdb, pdb, rdb, cache)
    return unwrapped + (check_runs_task,)


async def unwrap_pull_requests(prs_df: pd.DataFrame,
                               precomputed_done_facts: Dict[str, PullRequestFacts],
                               precomputed_ambiguous_done_facts: Dict[str, List[str]],
                               with_jira: bool,
                               branches: pd.DataFrame,
                               default_branches: Dict[str, str],
                               release_settings: ReleaseSettings,
                               prefixer: PrefixerPromise,
                               account: int,
                               meta_ids: Tuple[int, ...],
                               mdb: ParallelDatabase,
                               pdb: ParallelDatabase,
                               rdb: ParallelDatabase,
                               cache: Optional[aiomcache.Client],
                               resolve_rebased: bool = True,
                               with_deployments: bool = True,
                               ) -> Tuple[List[MinedPullRequest],
                                          PRDataFrames,
                                          Dict[str, PullRequestFacts],
                                          Dict[str, ReleaseMatch],
                                          Optional[asyncio.Task]]:
    """
    Fetch all the missing information about PRs in a dataframe.

    :param prs_df: dataframe with PullRequest-s.
    :param precomputed_done_facts: Preloaded precomputed facts of done PRs (explicit).
    :param precomputed_ambiguous_done_facts: Preloaded precomputed facts of done PRs (implicit).
    :param with_jira: Value indicating whether to load the mapped JIRA issues.
    :param branches: Branches of the relevant repositories.
    :param default_branches: Default branches of the relevant repositories.
    :param release_settings: Account's release settings.
    :param account: State DB account ID.
    :param meta_ids: GitHub account IDs.
    :param mdb: Metadata DB.
    :param pdb: Precomputed DB.
    :param cache: Optional memcached client.
    :return: Everything that's necessary for PullRequestListMiner.
    """
    if prs_df.empty:
        async def noop():
            return {}

        return (
            [],
            PRDataFrames(*(pd.DataFrame() for _ in fields(PRDataFrames))),
            {},
            {},
            asyncio.create_task(noop(), name="noop") if with_deployments else None,
        )
    if resolve_rebased:
        dags = await fetch_precomputed_commit_history_dags(
            prs_df[PullRequest.repository_full_name.name].unique(), account, pdb, cache)
        dags = await fetch_repository_commits_no_branch_dates(
            dags, branches, BRANCH_FETCH_COMMITS_COLUMNS, True, account, meta_ids, mdb, pdb, cache)
        prs_df = await PullRequestMiner.mark_dead_prs(
            prs_df, branches, dags, meta_ids, mdb, PullRequest)
    facts, ambiguous = precomputed_done_facts, precomputed_ambiguous_done_facts
    PullRequestMiner.adjust_pr_closed_merged_timestamps(prs_df)
    now = datetime.now(timezone.utc)
    if rel_time_from := prs_df[PullRequest.merged_at.name].nonemin():
        milestone_prs = prs_df[[PullRequest.merge_commit_sha.name,
                                PullRequest.merge_commit_id.name,
                                PullRequest.merged_at.name,
                                PullRequest.repository_full_name.name]]
        milestone_prs.columns = [
            Release.sha.name, Release.commit_id.name, Release.published_at.name,
            Release.repository_full_name.name,
        ]
        milestone_releases = dummy_releases_df().append(milestone_prs.reset_index(drop=True))
        milestone_releases = milestone_releases.take(np.where(
            milestone_releases[Release.sha.name].notnull())[0])
        releases, matched_bys = await ReleaseLoader.load_releases(
            prs_df[PullRequest.repository_full_name.name].unique(), branches, default_branches,
            rel_time_from, now, release_settings, prefixer,
            account, meta_ids, mdb, pdb, rdb, cache)
        add_pdb_misses(pdb, "load_precomputed_done_facts_reponums/ambiguous",
                       remove_ambiguous_prs(facts, ambiguous, matched_bys))
        tasks = [
            load_commit_dags(
                releases.append(milestone_releases), account, meta_ids, mdb, pdb, cache),
            # not nonemax() here! we want NaT-s inside load_merged_unreleased_pull_request_facts
            MergedPRFactsLoader.load_merged_unreleased_pull_request_facts(
                prs_df, releases[Release.published_at.name].max(), LabelFilter.empty(),
                matched_bys, default_branches, release_settings, prefixer, account, pdb),
        ]
        dags, unreleased = await gather(*tasks)
    else:
        releases, matched_bys, unreleased = dummy_releases_df(), {}, {}
        dags = await fetch_precomputed_commit_history_dags(
            prs_df[PullRequest.repository_full_name.name].unique(), account, pdb, cache)
    dfs, _, _ = await PullRequestMiner.mine_by_ids(
        prs_df, unreleased, now, releases, matched_bys, branches, default_branches, dags,
        release_settings, prefixer, account, meta_ids, mdb, pdb, cache, with_jira=with_jira)
    if with_deployments:
        deployment_names = dfs.deployments.index.get_level_values(1).unique()
        deployments = asyncio.create_task(load_included_deployments(
            deployment_names, account, meta_ids, mdb, rdb, cache),
            name=f"load_included_deployments({len(deployment_names)})")
    else:
        deployments = None
    prs = await list_with_yield(PullRequestMiner(dfs), "PullRequestMiner.__iter__")
    for k, v in unreleased.items():
        if k not in facts:
            facts[k] = v

    filtered_prs = []
    with sentry_sdk.start_span(op="PullRequestFactsMiner.__call__",
                               description=str(len(prs))):
        facts_miner = PullRequestFactsMiner(await bots(mdb))
        pdb_misses = 0
        for pr in prs:
            if (node_id := pr.pr[PullRequest.node_id.name]) not in facts:
                try:
                    facts[node_id] = facts_miner(pr)
                except ImpossiblePullRequest:
                    continue
                finally:
                    pdb_misses += 1
            filtered_prs.append(pr)

    set_pdb_hits(pdb, "fetch_pull_requests/facts", len(filtered_prs) - pdb_misses)
    set_pdb_misses(pdb, "fetch_pull_requests/facts", pdb_misses)
    return filtered_prs, dfs, facts, matched_bys, deployments
