import asyncio
from dataclasses import fields as dataclass_fields
from datetime import datetime, timedelta, timezone
import logging
import pickle
from typing import Callable, Collection, Dict, Generator, Iterable, KeysView, List, Optional, \
    Sequence, Set, Tuple, Union

import aiomcache
import numpy as np
import pandas as pd
import sentry_sdk
from sqlalchemy import and_, select, union_all

from athenian.api import COROUTINE_YIELD_EVERY_ITER, list_with_yield, metadata
from athenian.api.async_utils import gather, read_sql_query
from athenian.api.cache import cached, CancelCache, short_term_exptime
from athenian.api.controllers.datetime_utils import coarsen_time_interval
from athenian.api.controllers.features.github.check_run_metrics import \
    MergedPRsWithFailedChecksCounter
from athenian.api.controllers.features.github.pull_request_metrics import AllCounter, \
    DeploymentPendingMarker, DeploymentTimeCalculator, EnvironmentsMarker, MergingPendingCounter, \
    MergingTimeCalculator, ReleasePendingCounter, ReleaseTimeCalculator, ReviewPendingCounter, \
    ReviewTimeCalculator, StagePendingDependencyCalculator, WorkInProgressPendingCounter, \
    WorkInProgressTimeCalculator
from athenian.api.controllers.features.github.unfresh_pull_request_metrics import \
    UnfreshPullRequestFactsFetcher
from athenian.api.controllers.features.metric import Metric
from athenian.api.controllers.features.metric_calculator import MetricCalculator
from athenian.api.controllers.logical_repos import drop_logical_repo
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.branches import BranchMiner
from athenian.api.controllers.miners.github.check_run import mine_check_runs
from athenian.api.controllers.miners.github.commit import BRANCH_FETCH_COMMITS_COLUMNS, \
    fetch_precomputed_commit_history_dags, fetch_repository_commits_no_branch_dates
from athenian.api.controllers.miners.github.dag_accelerated import searchsorted_inrange
from athenian.api.controllers.miners.github.deployment_light import \
    fetch_repository_environments, load_included_deployments
from athenian.api.controllers.miners.github.logical import split_logical_prs
from athenian.api.controllers.miners.github.precomputed_prs import \
    DonePRFactsLoader, MergedPRFactsLoader, remove_ambiguous_prs, \
    store_merged_unreleased_pull_request_facts, store_open_pull_request_facts, \
    store_precomputed_done_facts
from athenian.api.controllers.miners.github.pull_request import ImpossiblePullRequest, \
    PRDataFrames, PullRequestFactsMiner, PullRequestMiner, ReviewResolution
from athenian.api.controllers.miners.github.release_load import dummy_releases_df, ReleaseLoader
from athenian.api.controllers.miners.github.release_match import load_commit_dags
from athenian.api.controllers.miners.types import Deployment, DeploymentConclusion, Label, \
    MinedPullRequest, PRParticipants, PullRequestEvent, PullRequestFacts, \
    PullRequestFactsMap, PullRequestJIRAIssueItem, PullRequestListItem, PullRequestStage
from athenian.api.controllers.prefixer import Prefixer
from athenian.api.controllers.settings import LogicalRepositorySettings, ReleaseMatch, \
    ReleaseSettings
from athenian.api.db import add_pdb_misses, Database, set_pdb_hits, set_pdb_misses
from athenian.api.defer import defer
from athenian.api.models.metadata.github import CheckRun, PullRequest, PullRequestCommit, \
    PullRequestLabel, PullRequestReview, PullRequestReviewComment, Release
from athenian.api.models.metadata.jira import Issue
from athenian.api.tracing import sentry_span
from athenian.api.typing_utils import df_from_structs


EventMap = Dict[PullRequestEvent, Set[Union[int, Tuple[int, str]]]]


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
                 facts: PullRequestFactsMap,
                 events: Set[PullRequestEvent],
                 stages: Set[PullRequestStage],
                 time_from: datetime,
                 time_to: datetime,
                 with_time_machine: bool,
                 events_environment: Optional[str],
                 environments: Dict[str, List[str]]):
        """Initialize a new instance of `PullRequestListMiner`."""
        self._prs = prs
        self._dfs = dfs
        self._facts = facts
        assert isinstance(events, set)
        assert isinstance(stages, set)
        self._events = events
        self._stages = stages
        self._environment = events_environment
        self._environments, environments_order = np.unique(
            list(environments), return_inverse=True)
        self._calcs, self._counter_deps = self.create_stage_calcs(environments, environments_order)
        assert isinstance(time_from, datetime)
        assert time_from.tzinfo == timezone.utc
        assert isinstance(time_to, datetime)
        assert time_to.tzinfo == timezone.utc
        self._time_from = time_from.replace(tzinfo=None)
        self._time_to = time_to.replace(tzinfo=None)
        self._with_time_machine = with_time_machine

    @classmethod
    def create_stage_calcs(cls,
                           environments: Dict[str, List[str]],
                           environments_order: Optional[Sequence[int]] = None,
                           ) -> Tuple[Dict[str, Dict[str, List[MetricCalculator[int]]]],
                                      Iterable[MetricCalculator]]:
        """Intialize PR metric calculators needed to calculate the stage timings."""
        quantiles = dict(quantiles=(0, 1))
        ordered_envs = np.empty(len(environments), dtype=object)
        repos_in_env = np.empty(len(environments), dtype=object)
        if environments_order is None:
            environments_order = range(len(environments))
        for env_index, (env, repos) in zip(environments_order, environments.items()):
            repos_in_env[env_index] = repos
            ordered_envs[env_index] = env
        del environments
        del environments_order
        if len(ordered_envs) == 0:
            ordered_envs = [None]
        all_counter = cls.DummyAllCounter(**quantiles)
        pendinger = StagePendingDependencyCalculator(all_counter, **quantiles)
        env_marker = EnvironmentsMarker(**quantiles, environments=ordered_envs)
        calcs = {
            "wip": {
                "time": [WorkInProgressTimeCalculator(**quantiles)],
                "pending": [WorkInProgressPendingCounter(
                    pendinger, **quantiles)],
            },
            "review": {
                "time": [ReviewTimeCalculator(**quantiles)],
                "pending": [ReviewPendingCounter(
                    pendinger, **quantiles)],
            },
            "merge": {
                "time": [MergingTimeCalculator(**quantiles)],
                "pending": [MergingPendingCounter(
                    pendinger, **quantiles)],
            },
            "release": {
                "time": [ReleaseTimeCalculator(**quantiles)],
                "pending": [ReleasePendingCounter(
                    pendinger, **quantiles)],
            },
            "deploy": {
                "time": DeploymentTimeCalculator(
                    env_marker, **quantiles, environments=ordered_envs,
                ).split(),
                "pending": DeploymentPendingMarker(
                    env_marker, **quantiles, environments=ordered_envs, repositories=repos_in_env,
                ).split(),
                "deps": [env_marker],
            },
        }
        return calcs, (all_counter, pendinger)

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
        if hard_events[PullRequestEvent.DEPLOYED]:
            events.add(PullRequestEvent.DEPLOYED)
            stages.add(PullRequestStage.DEPLOYED)
        return events, stages

    def _compile(self,
                 pr: MinedPullRequest,
                 facts: PullRequestFacts,
                 stage_timings: Dict[str, Union[timedelta, Dict[str, timedelta]]],
                 hard_events_time_machine: EventMap,
                 hard_events_now: EventMap,
                 ) -> Optional[PullRequestListItem]:
        """
        Match the PR to the required participants and properties and produce PullRequestListItem.

        We return None if the PR does not match.
        """
        facts_now = facts
        pr_node_id = pr.pr[PullRequest.node_id.name]
        repo = pr.pr[PullRequest.repository_full_name.name]
        if self._with_time_machine:
            facts_time_machine = facts.truncate(self._time_to)
            events_time_machine, stages_time_machine = self._collect_events_and_stages(
                facts_time_machine,
                {k: (pr_node_id in v or (pr_node_id, repo) in v)
                 for k, v in hard_events_time_machine.items()},
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
            {k: (pr_node_id in v or (pr_node_id, repo) in v) for k, v in hard_events_now.items()},
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
        deployments = pr.deployments.index.values if len(pr.deployments.index) else None
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
            merged_with_failed_check_runs=None,  # filled by _append_merged_with_failed_check_runs
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
        repo_name_key = PullRequest.repository_full_name.name
        for i, pr in enumerate(self._prs):
            pr_stage_timings = {}
            for k in self._calcs:
                key_stage_timings = stage_timings[k]
                if k == "deploy":
                    deploy_timings = {}
                    for env, timings in zip(self._environments, key_stage_timings):
                        if (timing := timings[i].item()) is not None:
                            deploy_timings[env] = timing
                    if deploy_timings:
                        pr_stage_timings[k] = deploy_timings
                else:
                    if (timing := key_stage_timings[0][i].item()) is not None:
                        pr_stage_timings[k] = timing
            item = self._compile(pr,
                                 facts[(pr.pr[node_id_key], pr.pr[repo_name_key])],
                                 pr_stage_timings,
                                 hard_events_time_machine,
                                 hard_events_now)
            if item is not None:
                yield item

    @sentry_span
    def _calc_stage_timings(self) -> Dict[str, List[np.ndarray]]:
        facts = self._facts
        if len(facts) == 0 or len(self._prs) == 0:
            return {k: [] for k in self._calcs}
        node_id_key = PullRequest.node_id.name
        repo_name_key = PullRequest.repository_full_name.name
        df_facts = df_from_structs(
            (facts[(pr.pr[node_id_key], pr.pr[repo_name_key])] for pr in self._prs),
            length=len(self._prs))
        return self.calc_stage_timings(df_facts, self._calcs, self._counter_deps)

    @classmethod
    def calc_stage_timings(cls,
                           df_facts: pd.DataFrame,
                           calcs: Dict[str, Dict[str, List[MetricCalculator[int]]]],
                           counter_deps: Iterable[MetricCalculator],
                           ) -> Dict[str, List[np.ndarray]]:  # np.ndarray[int]
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
        for stage, stage_calcs in calcs.items():
            stage_timings[stage] = []
            for dep in stage_calcs.get("deps", []):
                dep(df_facts, no_time_from, now, None, empty_group_mask)
            for time_calc, pendinger in zip(stage_calcs["time"], stage_calcs["pending"]):
                pendinger(df_facts, no_time_from, now, None, empty_group_mask)
                kwargs = {
                    # -1 second to appear less than time_max
                    "override_event_time": now - np.timedelta64(timedelta(seconds=1)),
                    "override_event_indexes": np.flatnonzero(pendinger.peek[0] == 1),
                }
                if stage == "review":
                    kwargs["allow_unclosed"] = True
                time_calc(df_facts, no_time_from, now, None, empty_group_mask, **kwargs)
                stage_timings[stage].append(time_calc.peek[0])
        return stage_timings

    @sentry_span
    def _calc_hard_events(self) -> Tuple[EventMap, EventMap]:
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
        reviews = dfs.prs[[PullRequest.user_login.name, PullRequest.closed_at.name]].join(
            reviews, on="node_id", lsuffix="_pr", how="inner")
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

        if len(self._environments) == 0:
            events_time_machine[PullRequestEvent.DEPLOYED] = \
                events_now[PullRequestEvent.DEPLOYED] = set()
            return events_time_machine, events_now

        if self._environment is None:
            dep_index = 0  # whatever in range
        else:
            dep_index = searchsorted_inrange(self._environments, [self._environment])[0]
        if self._environments[dep_index] == self._environment:
            prs = dfs.prs.index.values
            deployed = np.full(len(prs), None, "datetime64[s]")
            dep_peek = self._calcs["deploy"]["time"][dep_index].calcs[0].peek
            mask = (dep_peek.environments.item() & (1 << dep_index)).astype(bool)
            successful = dep_peek.conclusions.item()[dep_index] == DeploymentConclusion.SUCCESS
            deployed[mask] = dep_peek.finished.item()[dep_index][successful]
            events_time_machine[PullRequestEvent.DEPLOYED] = \
                set(prs[deployed < np.array(self._time_to, dtype=deployed.dtype)])
            events_now[PullRequestEvent.DEPLOYED] = set(prs[mask])
        else:
            events_time_machine[PullRequestEvent.DEPLOYED] = \
                events_now[PullRequestEvent.DEPLOYED] = set()

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
                               environment: Optional[str],
                               exclude_inactive: bool,
                               bots: Set[str],
                               release_settings: ReleaseSettings,
                               logical_settings: LogicalRepositorySettings,
                               updated_min: Optional[datetime],
                               updated_max: Optional[datetime],
                               prefixer: Prefixer,
                               account: int,
                               meta_ids: Tuple[int, ...],
                               mdb: Database,
                               pdb: Database,
                               rdb: Database,
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
        environment, exclude_inactive, bots, release_settings, logical_settings,
        updated_min, updated_max, prefixer, account, meta_ids, mdb, pdb, rdb, cache)
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
    left_deployments = [pr.deployments for pr in prs
                        if pr.deployments is not None and len(pr.deployments)]
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
                                          logical_settings: LogicalRepositorySettings,
                                          meta_ids: Tuple[int, ...],
                                          mdb: Database,
                                          cache: Optional[aiomcache.Client],
                                          ) -> Tuple[np.ndarray, List[np.ndarray]]:
    df = await mine_check_runs(
        time_from, time_to, repos, [], labels, jira, True, logical_settings, meta_ids, mdb, cache)
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
    exptime=short_term_exptime,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda time_from, time_to, repos, events, stages, participants, environment, exclude_inactive, release_settings, logical_settings, updated_min, updated_max, **_: (  # noqa
        time_from.timestamp(),
        time_to.timestamp(),
        ",".join(sorted(repos)),
        ",".join(s.name.lower() for s in sorted(events)),
        ",".join(s.name.lower() for s in sorted(stages)),
        sorted((k.name.lower(), sorted(v)) for k, v in participants.items()),
        environment,
        exclude_inactive,
        updated_min.timestamp() if updated_min is not None else None,
        updated_max.timestamp() if updated_max is not None else None,
        release_settings,
        logical_settings,
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
                                environment: Optional[str],
                                exclude_inactive: bool,
                                bots: Set[str],
                                release_settings: ReleaseSettings,
                                logical_settings: LogicalRepositorySettings,
                                updated_min: Optional[datetime],
                                updated_max: Optional[datetime],
                                prefixer: Prefixer,
                                account: int,
                                meta_ids: Tuple[int, ...],
                                mdb: Database,
                                pdb: Database,
                                rdb: Database,
                                cache: Optional[aiomcache.Client],
                                ) -> Tuple[List[PullRequestListItem],
                                           Dict[str, Deployment],
                                           LabelFilter,
                                           JIRAFilter]:
    assert isinstance(events, set)
    assert isinstance(stages, set)
    assert isinstance(repos, set)
    log = logging.getLogger("%s.filter_pull_requests" % metadata.__package__)
    # required to efficiently use the cache with timezones
    date_from, date_to = coarsen_time_interval(time_from, time_to)
    if updated_min is not None:
        assert updated_max is not None
        coarsen_time_interval(updated_min, updated_max)
    else:
        assert updated_max is None
    check_runs_task = asyncio.create_task(_load_failed_check_runs_for_prs(
        updated_min or time_from, updated_max or time_to,
        repos, labels, jira, logical_settings, meta_ids, mdb, cache))
    environments_task = asyncio.create_task(fetch_repository_environments(
        repos, prefixer, account, rdb, cache))
    branches, default_branches = await BranchMiner.extract_branches(
        repos, prefixer, meta_ids, mdb, cache)
    tasks = (
        PullRequestMiner.mine(
            date_from, date_to, time_from, time_to, repos, participants, labels, jira, True,
            branches, default_branches, exclude_inactive, release_settings, logical_settings,
            prefixer, account, meta_ids, mdb, pdb, rdb, cache,
            truncate=False, updated_min=updated_min, updated_max=updated_max),
        DonePRFactsLoader.load_precomputed_done_facts_filters(
            time_from, time_to, repos, participants, labels, default_branches,
            exclude_inactive, release_settings, prefixer, account, pdb),
    )
    (
        (pr_miner, unreleased_facts, matched_bys, unreleased_prs_event),
        (released_facts, ambiguous),
    ) = await gather(*tasks)
    add_pdb_misses(pdb, "load_precomputed_done_facts_filters/ambiguous",
                   remove_ambiguous_prs(released_facts, ambiguous, matched_bys))
    # we want the released PR facts to overwrite the others
    facts = {**unreleased_facts, **released_facts}
    del unreleased_facts, released_facts

    deployment_names = pr_miner.dfs.deployments.index.get_level_values(1).unique()
    deps_task = asyncio.create_task(_load_deployments(
        deployment_names, facts, logical_settings, prefixer,
        account, meta_ids, mdb, pdb, rdb, cache),
        name=f"filter_pull_requests/_load_deployments({len(deployment_names)})")
    prs = await list_with_yield(pr_miner, "PullRequestMiner.__iter__")

    with sentry_sdk.start_span(op="PullRequestFactsMiner.__call__"):
        facts_miner = PullRequestFactsMiner(bots)
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
            node_id, repo = \
                pr.pr[PullRequest.node_id.name], pr.pr[PullRequest.repository_full_name.name]
            if (node_id, repo) not in facts:
                fact_evals += 1
                if (fact_evals + 1) % COROUTINE_YIELD_EVERY_ITER == 0:
                    await asyncio.sleep(0)
                try:
                    facts[(node_id, repo)] = pr_facts = facts_miner(pr)
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

    _, include_deps_task = await gather(environments_task, deps_task)
    prs = await list_with_yield(
        PullRequestListMiner(
            prs, pr_miner.dfs, facts, events, stages, time_from, time_to, True, environment,
            environments_task.result()),
        "PullRequestListMiner.__iter__",
    )
    await gather(_append_merged_with_failed_check_runs(check_runs_task, prs),
                 include_deps_task)
    log.debug("return %d PRs", len(prs))
    return prs, include_deps_task.result(), labels, jira


@sentry_span
async def _load_deployments(new_names: Sequence[str],
                            precomputed_facts: PullRequestFactsMap,
                            logical_settings: LogicalRepositorySettings,
                            prefixer: Prefixer,
                            account: int,
                            meta_ids: Tuple[int, ...],
                            mdb: Database,
                            pdb: Database,
                            rdb: Database,
                            cache: Optional[aiomcache.Client],
                            ) -> asyncio.Task:
    deps = await PullRequestMiner.fetch_pr_deployments(
        {node_id for node_id, _ in precomputed_facts}, prefixer, account, pdb, rdb)
    log = logging.getLogger(f"{metadata.__package__}.filter_pull_requests.load_deployments")
    UnfreshPullRequestFactsFetcher.append_deployments(
        precomputed_facts, deps, logical_settings.has_logical_prs(), log)
    names = np.unique(np.concatenate([new_names, deps.index.get_level_values(1).values]))
    return asyncio.create_task(
        load_included_deployments(names, account, meta_ids, mdb, rdb, cache),
        name="filter_pull_requests/load_included_deployments",
    )


@sentry_span
@cached(
    exptime=short_term_exptime,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda prs, release_settings, logical_settings, environment, **_: (
        ";".join("%s:%s" % (repo, ",".join(map(str, sorted(numbers))))
                 for repo, numbers in sorted(prs.items())),
        release_settings,
        logical_settings,
        environment,
    ),
)
async def fetch_pull_requests(prs: Dict[str, Set[int]],
                              bots: Set[str],
                              release_settings: ReleaseSettings,
                              logical_settings: LogicalRepositorySettings,
                              environment: Optional[str],
                              prefixer: Prefixer,
                              account: int,
                              meta_ids: Tuple[int, ...],
                              mdb: Database,
                              pdb: Database,
                              rdb: Database,
                              cache: Optional[aiomcache.Client],
                              ) -> Tuple[List[PullRequestListItem], Dict[str, Deployment]]:
    """
    List GitHub pull requests by repository and numbers.

    :params prs: For each repository name without the prefix, there is a set of PR numbers to list.
    """
    (mined_prs, dfs, facts, _, deployments_task, check_runs_task), repo_envs = await gather(
        _fetch_pull_requests(
            prs, bots, release_settings, logical_settings, prefixer, account, meta_ids,
            mdb, pdb, rdb, cache),
        fetch_repository_environments(prs, prefixer, account, rdb, cache),
    )
    if not mined_prs:
        return [], {}
    miner = PullRequestListMiner(
        mined_prs, dfs, facts, set(), set(),
        datetime(1970, 1, 1, tzinfo=timezone.utc), datetime.now(timezone.utc),
        False, environment, repo_envs)
    prs = await list_with_yield(miner, "PullRequestListMiner.__iter__")
    await _append_merged_with_failed_check_runs(check_runs_task, prs)
    await deployments_task
    return prs, deployments_task.result()


async def _fetch_pull_requests(prs: Dict[str, Set[int]],
                               bots: Set[str],
                               release_settings: ReleaseSettings,
                               logical_settings: LogicalRepositorySettings,
                               prefixer: Prefixer,
                               account: int,
                               meta_ids: Tuple[int, ...],
                               mdb: Database,
                               pdb: Database,
                               rdb: Database,
                               cache: Optional[aiomcache.Client],
                               ) -> Tuple[List[MinedPullRequest],
                                          PRDataFrames,
                                          PullRequestFactsMap,
                                          Dict[str, ReleaseMatch],
                                          asyncio.Task,
                                          Optional[asyncio.Task]]:
    branches, default_branches = await BranchMiner.extract_branches(
        prs, prefixer, meta_ids, mdb, cache)
    filters = [and_(PullRequest.repository_full_name == drop_logical_repo(repo),
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
            prs.keys(), LabelFilter.empty(), JIRAFilter.empty(),
            logical_settings, meta_ids, mdb, cache))
    else:
        check_runs_task = None
    unwrapped = await unwrap_pull_requests(
        prs_df, facts, ambiguous, True, branches, default_branches,
        bots, release_settings, logical_settings, prefixer, account, meta_ids,
        mdb, pdb, rdb, cache, repositories=prs.keys())
    return unwrapped + (check_runs_task,)


async def unwrap_pull_requests(prs_df: pd.DataFrame,
                               precomputed_done_facts: PullRequestFactsMap,
                               precomputed_ambiguous_done_facts: Dict[str, List[int]],
                               with_jira: bool,
                               branches: pd.DataFrame,
                               default_branches: Dict[str, str],
                               bots: Set[str],
                               release_settings: ReleaseSettings,
                               logical_settings: LogicalRepositorySettings,
                               prefixer: Prefixer,
                               account: int,
                               meta_ids: Tuple[int, ...],
                               mdb: Database,
                               pdb: Database,
                               rdb: Database,
                               cache: Optional[aiomcache.Client],
                               resolve_rebased: bool = True,
                               repositories: Optional[Union[Set[str], KeysView[str]]] = None,
                               ) -> Tuple[List[MinedPullRequest],
                                          PRDataFrames,
                                          PullRequestFactsMap,
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
            PRDataFrames(*(pd.DataFrame() for _ in dataclass_fields(PRDataFrames))),
            {},
            {},
            asyncio.create_task(noop(), name="noop"),
        )
    if repositories is None:
        repositories = logical_settings.with_logical_repos(
            prs_df[PullRequest.repository_full_name.name].values)
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
            rel_time_from, now, release_settings, logical_settings, prefixer,
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
    for k, v in unreleased.items():
        if k not in facts:
            facts[k] = v
    dfs, _, _ = await PullRequestMiner.mine_by_ids(
        prs_df, unreleased, repositories, now, releases, matched_bys, branches, default_branches,
        dags, release_settings, logical_settings, prefixer, account, meta_ids,
        mdb, pdb, rdb, cache, with_jira=with_jira)
    deployment_names = dfs.deployments.index.get_level_values(1).unique()
    deployments_task = asyncio.create_task(_load_deployments(
        deployment_names, facts, logical_settings, prefixer,
        account, meta_ids, mdb, pdb, rdb, cache),
        name=f"load_included_deployments({len(deployment_names)})")

    dfs.prs = split_logical_prs(dfs.prs, dfs.labels, repositories, logical_settings)
    prs = await list_with_yield(PullRequestMiner(dfs), "PullRequestMiner.__iter__")

    filtered_prs = []
    with sentry_sdk.start_span(op="PullRequestFactsMiner.__call__",
                               description=str(len(prs))):
        facts_miner = PullRequestFactsMiner(bots)
        pdb_misses = 0
        for pr in prs:
            node_id, repo = \
                pr.pr[PullRequest.node_id.name], pr.pr[PullRequest.repository_full_name.name]
            if (node_id, repo) not in facts:
                try:
                    facts[(node_id, repo)] = facts_miner(pr)
                except ImpossiblePullRequest:
                    continue
                finally:
                    pdb_misses += 1
            filtered_prs.append(pr)

    set_pdb_hits(pdb, "fetch_pull_requests/facts", len(filtered_prs) - pdb_misses)
    set_pdb_misses(pdb, "fetch_pull_requests/facts", pdb_misses)
    if deployments_task is not None:
        await deployments_task
        deployments_task = deployments_task.result()
    return filtered_prs, dfs, facts, matched_bys, deployments_task
