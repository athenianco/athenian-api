import asyncio
from collections.abc import Collection
from dataclasses import fields as dataclass_fields
from datetime import datetime, timedelta, timezone
import logging
import pickle
from typing import Callable, Generator, Iterable, KeysView, Optional, Sequence

import aiomcache
import numpy as np
import numpy.typing as npt
import pandas as pd
import sentry_sdk
from sqlalchemy import and_, select, union_all

from athenian.api import metadata
from athenian.api.async_utils import (
    COROUTINE_YIELD_EVERY_ITER,
    gather,
    list_with_yield,
    read_sql_query,
)
from athenian.api.cache import CancelCache, cached, short_term_exptime
from athenian.api.db import Database, add_pdb_misses, set_pdb_hits, set_pdb_misses
from athenian.api.defer import defer
from athenian.api.internal.datetime_utils import coarsen_time_interval
from athenian.api.internal.features.github.pull_request_filter_accelerated import (
    collect_events_and_stages,
)
from athenian.api.internal.features.github.pull_request_metrics import (
    AllCounter,
    DeploymentPendingMarker,
    DeploymentTimeCalculator,
    EnvironmentsMarker,
    MergingPendingCounter,
    MergingTimeCalculator,
    ReleasePendingCounter,
    ReleaseTimeCalculator,
    ReviewPendingCounter,
    ReviewTimeCalculator,
    StagePendingDependencyCalculator,
    WorkInProgressPendingCounter,
    WorkInProgressTimeCalculator,
)
from athenian.api.internal.features.github.unfresh_pull_request_metrics import (
    UnfreshPullRequestFactsFetcher,
)
from athenian.api.internal.features.metric import Metric
from athenian.api.internal.features.metric_calculator import MetricCalculator
from athenian.api.internal.logical_repos import drop_logical_repo
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.branches import BranchMiner
from athenian.api.internal.miners.github.commit import (
    BRANCH_FETCH_COMMITS_COLUMNS,
    fetch_precomputed_commit_history_dags,
    fetch_repository_commits,
)
from athenian.api.internal.miners.github.deployment_light import (
    fetch_repository_environments,
    load_included_deployments,
)
from athenian.api.internal.miners.github.logical import split_logical_prs
from athenian.api.internal.miners.github.precomputed_prs import (
    DonePRFactsLoader,
    MergedPRFactsLoader,
    remove_ambiguous_prs,
    store_merged_unreleased_pull_request_facts,
    store_open_pull_request_facts,
    store_precomputed_done_facts,
)
from athenian.api.internal.miners.github.pull_request import (
    ImpossiblePullRequest,
    PRDataFrames,
    PullRequestFactsMiner,
    PullRequestMiner,
    ReviewResolution,
)
from athenian.api.internal.miners.github.release_load import ReleaseLoader, dummy_releases_df
from athenian.api.internal.miners.github.release_match import load_commit_dags
from athenian.api.internal.miners.jira.issue import PullRequestJiraMapper
from athenian.api.internal.miners.participation import PRParticipants
from athenian.api.internal.miners.types import (
    Deployment,
    JIRAEntityToFetch,
    Label,
    LoadedJIRADetails,
    MinedPullRequest,
    PullRequestEvent,
    PullRequestFacts,
    PullRequestFactsMap,
    PullRequestJIRAIssueItem,
    PullRequestListItem,
    PullRequestStage,
)
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.settings import LogicalRepositorySettings, ReleaseMatch, ReleaseSettings
from athenian.api.models.metadata.github import (
    PullRequest,
    PullRequestCommit,
    PullRequestLabel,
    PullRequestReview,
    PullRequestReviewComment,
    Release,
)
from athenian.api.models.metadata.jira import Issue
from athenian.api.native.mi_heap_destroy_stl_allocator import make_mi_heap_allocator_capsule
from athenian.api.object_arrays import is_not_null
from athenian.api.tracing import sentry_span
from athenian.api.typing_utils import df_from_structs

EventMap = dict[PullRequestEvent, set[int | tuple[int, str]]]


class PullRequestListMiner:
    """Collect various PR metadata for displaying PRs on the frontend."""

    _no_time_from = np.datetime64("1970-01-01", "s")
    log = logging.getLogger("%s.PullRequestListMiner" % metadata.__version__)

    class DummyAllCounter(AllCounter):
        """Fool StagePendingDependencyCalculator so that it does not exclude any PRs."""

        dtype = int

        def _analyze(
            self,
            facts: pd.DataFrame,
            min_times: np.ndarray,
            max_times: np.ndarray,
            **kwargs,
        ) -> np.ndarray:
            return np.full((len(min_times), len(facts)), 1)

        def _value(self, samples: np.ndarray) -> Metric[int]:
            raise AssertionError("This method should never be called.")

    def __init__(
        self,
        prs: list[MinedPullRequest],
        dfs: PRDataFrames,
        facts: PullRequestFactsMap,
        events: set[PullRequestEvent],
        stages: set[PullRequestStage],
        time_from: datetime,
        time_to: datetime,
        with_time_machine: bool,
        environments: dict[str, list[str]],
    ):
        """Initialize a new instance of `PullRequestListMiner`."""
        self._prs = prs
        self._dfs = dfs
        self._facts = facts
        assert isinstance(events, set)
        assert isinstance(stages, set)
        self._events = events
        self._stages = stages
        self._environments = environments
        self._calcs, self._counter_deps = self.create_stage_calcs(environments)
        assert isinstance(time_from, datetime)
        assert time_from.tzinfo == timezone.utc
        assert isinstance(time_to, datetime)
        assert time_to.tzinfo == timezone.utc
        self._time_from = time_from.replace(tzinfo=None)
        self._time_to = time_to.replace(tzinfo=None)
        self._with_time_machine = with_time_machine

    @classmethod
    def create_stage_calcs(
        cls,
        environments: dict[str, list[str]],
        stages: Collection[str] = ("wip", "review", "merge", "release", "deploy"),
    ) -> tuple[dict[str, dict[str, list[MetricCalculator[int]]]], Iterable[MetricCalculator]]:
        """Intialize PR metric calculators needed to calculate the stage timings."""
        quantiles = {"quantiles": (0, 1)}
        ordered_envs = np.empty(len(environments), dtype=object)
        repos_in_env = np.empty(len(environments), dtype=object)
        for env_index, (env, repos) in enumerate(environments.items()):
            repos_in_env[env_index] = repos
            ordered_envs[env_index] = env
        del environments
        if len(ordered_envs) == 0:
            ordered_envs = [None]
        all_counter = cls.DummyAllCounter(**quantiles)
        pendinger = StagePendingDependencyCalculator(all_counter, **quantiles)
        env_marker = EnvironmentsMarker(**quantiles, environments=ordered_envs)
        calcs = {}
        if "wip" in stages:
            calcs["wip"] = {
                "time": [WorkInProgressTimeCalculator(**quantiles)],
                "pending": [WorkInProgressPendingCounter(pendinger, **quantiles)],
            }
        if "review" in stages:
            calcs["review"] = {
                "time": [ReviewTimeCalculator(**quantiles)],
                "pending": [ReviewPendingCounter(pendinger, **quantiles)],
            }
        if "merge" in stages:
            calcs["merge"] = {
                "time": [MergingTimeCalculator(**quantiles)],
                "pending": [MergingPendingCounter(pendinger, **quantiles)],
            }
        if "release" in stages:
            calcs["release"] = {
                "time": [ReleaseTimeCalculator(**quantiles)],
                "pending": [ReleasePendingCounter(pendinger, **quantiles)],
            }
        if "deploy" in stages:
            calcs["deploy"] = {
                "time": DeploymentTimeCalculator(
                    env_marker, **quantiles, environments=ordered_envs,
                ).split(),
                # yes, we will return redundant environments for logical repositories
                # this is an acceptable shortcut
                "pending": DeploymentPendingMarker(
                    env_marker,
                    **quantiles,
                    environments=ordered_envs,
                    repositories=repos_in_env,
                    drop_logical=True,
                ).split(),
                "deps": [env_marker],
            }

        return calcs, (all_counter, pendinger)

    @classmethod
    def _collect_events_and_stages(
        cls,
        facts: PullRequestFacts,
        hard_events: dict[PullRequestEvent, bool],
        time_from: np.datetime64,
    ) -> tuple[set[PullRequestEvent], set[PullRequestStage]]:
        return collect_events_and_stages(facts, hard_events, time_from)

    # profiling indicates that we spend a lot of time getting these
    PullRequest_node_id_name = PullRequest.node_id.name
    PullRequest_user_login_name = PullRequest.user_login.name
    PullRequest_repository_full_name_name = PullRequest.repository_full_name.name
    PullRequest_number_name = PullRequest.number.name
    PullRequest_title_name = PullRequest.title.name
    PullRequest_additions_name = PullRequest.additions.name
    PullRequest_deletions_name = PullRequest.deletions.name
    PullRequest_changed_files_name = PullRequest.changed_files.name
    PullRequest_created_at_name = PullRequest.created_at.name
    PullRequestReview_user_login_name = PullRequestReview.user_login.name
    PullRequestReview_created_at_name = PullRequestReview.created_at.name
    PullRequestReviewComment_user_login_name = PullRequestReviewComment.user_login.name
    PullRequest_updated_at_name = PullRequest.updated_at.name
    PullRequestLabel_name_name = PullRequestLabel.name.name
    PullRequestLabel_description_name = PullRequestLabel.description.name
    PullRequestLabel_color_name = PullRequestLabel.color.name
    Release_url_name = Release.url.name
    Issue_title_name = Issue.title.name
    Issue_labels_name = Issue.labels.name
    Issue_type_name = Issue.type.name

    def _compile(
        self,
        pr: MinedPullRequest,
        facts: PullRequestFacts,
        stage_timings: dict[str, timedelta | dict[str, timedelta]],
        hard_events_time_machine: EventMap,
        hard_events_now: EventMap,
        alloc,
    ) -> Optional[PullRequestListItem]:
        """
        Match the PR to the required participants and properties and produce PullRequestListItem.

        We return None if the PR does not match.
        """
        facts_now = facts
        pr_node_id = pr.pr[self.PullRequest_node_id_name]
        repo = pr.pr[self.PullRequest_repository_full_name_name]
        if self._with_time_machine:
            facts_time_machine = facts.truncate(np.datetime64(self._time_to, "s"))
            events_time_machine, stages_time_machine = self._collect_events_and_stages(
                facts_time_machine,
                {
                    k: (pr_node_id in v or (pr_node_id, repo) in v)
                    for k, v in hard_events_time_machine.items()
                },
                np.datetime64(self._time_from, "s"),
            )
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
            self._no_time_from,
        )
        author = pr.pr[self.PullRequest_user_login_name]
        external_reviews_mask = pr.reviews[self.PullRequestReview_user_login_name].values != author
        external_review_times = pr.reviews[self.PullRequestReview_created_at_name].values[
            external_reviews_mask
        ]
        first_review = (
            pd.Timestamp(external_review_times.min(), tz=timezone.utc)
            if len(external_review_times) > 0
            else None
        )
        review_comments = (
            pr.review_comments[self.PullRequestReviewComment_user_login_name].values != author
        ).sum()
        delta_comments = len(pr.review_comments) - review_comments
        reviews = external_reviews_mask.sum()
        updated_at = pr.pr[self.PullRequest_updated_at_name]
        assert updated_at == updated_at
        if pr.labels.empty:
            labels = None
        else:
            labels = [
                Label(name=name, description=description, color=color)
                for name, description, color in zip(
                    pr.labels[self.PullRequestLabel_name_name].values,
                    pr.labels[self.PullRequestLabel_description_name].values,
                    pr.labels[self.PullRequestLabel_color_name].values,
                )
            ]
        if pr.jiras.empty:
            jira = None
        else:
            jira = [
                PullRequestJIRAIssueItem(
                    id=key, title=title, epic=epic, labels=labels or None, type=itype,
                )
                for (key, title, epic, labels, itype) in zip(
                    pr.jiras.index.values,
                    pr.jiras[self.Issue_title_name].values,
                    pr.jiras["epic"].values,
                    pr.jiras[self.Issue_labels_name].values,
                    pr.jiras[self.Issue_type_name].values,
                )
            ]
        deployments = (
            pr.deployments.index.get_level_values(1).values if len(pr.deployments.index) else None
        )
        return PullRequestListItem(
            node_id=pr_node_id,
            repository=pr.pr[self.PullRequest_repository_full_name_name],
            number=pr.pr[self.PullRequest_number_name],
            title=pr.pr[self.PullRequest_title_name],
            size_added=pr.pr[self.PullRequest_additions_name],
            size_removed=pr.pr[self.PullRequest_deletions_name],
            files_changed=pr.pr[self.PullRequest_changed_files_name],
            created=pr.pr[self.PullRequest_created_at_name],
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
            merged_with_failed_check_runs=facts_now.merged_with_failed_check_runs.tolist(),
            released=self._dt64_to_pydt(facts_now.released),
            release_url=pr.release[self.Release_url_name],
            events_now=events_now,
            stages_now=stages_now,
            events_time_machine=events_time_machine,
            stages_time_machine=stages_time_machine,
            stage_timings=stage_timings,
            participants=pr.participant_nodes(alloc),
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
        alloc = make_mi_heap_allocator_capsule()
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
            item = self._compile(
                pr,
                facts[(pr.pr[node_id_key], pr.pr[repo_name_key])],
                pr_stage_timings,
                hard_events_time_machine,
                hard_events_now,
                alloc,
            )
            if item is not None:
                yield item

    @sentry_span
    def _calc_stage_timings(self) -> dict[str, list[np.ndarray]]:
        facts = self._facts
        if len(facts) == 0 or len(self._prs) == 0:
            return {k: [] for k in self._calcs}
        node_id_key = PullRequest.node_id.name
        repo_name_key = PullRequest.repository_full_name.name
        df_facts = df_from_structs(
            (facts[(pr.pr[node_id_key], pr.pr[repo_name_key])] for pr in self._prs),
            length=len(self._prs),
        )
        return self.calc_stage_timings(df_facts, self._calcs, self._counter_deps)

    @classmethod
    @sentry_span
    def calc_stage_timings(
        cls,
        df_facts: pd.DataFrame,
        calcs: dict[str, dict[str, list[MetricCalculator[int]]]],
        counter_deps: Iterable[MetricCalculator],
    ) -> dict[str, list[np.ndarray]]:  # np.ndarray[int]
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
    def _calc_hard_events(self) -> tuple[EventMap, EventMap]:
        events_time_machine = {}
        events_now = {}
        dfs = self._dfs
        time_from = np.datetime64(self._time_from)
        time_to = np.datetime64(self._time_to)

        index = dfs.commits.index.get_level_values(0).values
        committed_date = dfs.commits[PullRequestCommit.committed_date.name].values
        events_now[PullRequestEvent.COMMITTED] = set(index[committed_date == committed_date])
        if self._with_time_machine:
            events_time_machine[PullRequestEvent.COMMITTED] = set(
                index[(committed_date >= time_from) & (committed_date < time_to)],
            )

        prr_ulk = PullRequestReview.user_login.name
        prr_sak = PullRequestReview.submitted_at.name
        reviews = dfs.reviews[[prr_ulk, prr_sak]].droplevel(1)
        original_reviews_index = reviews.index.values
        reviews = dfs.prs[[PullRequest.user_login.name, PullRequest.closed_at.name]].join(
            reviews, on="node_id", lsuffix="_pr", how="inner",
        )
        index = reviews.index.values
        submitted_at = reviews[prr_sak].values
        closed_at = reviews[PullRequest.closed_at.name].values
        not_closed_mask = closed_at != closed_at
        closed_at[not_closed_mask] = submitted_at[not_closed_mask]
        closed_at = closed_at.astype(submitted_at.dtype)
        mask = (submitted_at <= closed_at) & (
            reviews[prr_ulk].values != reviews[PullRequest.user_login.name + "_pr"].values
        )
        events_now[PullRequestEvent.REVIEWED] = set(index[mask])
        if self._with_time_machine:
            mask &= (submitted_at >= time_from) & (submitted_at < time_to)
            events_time_machine[PullRequestEvent.REVIEWED] = set(index[mask])

        submitted_at = dfs.reviews[prr_sak].values
        # no need to check submitted_at <= closed_at because GitHub disallows that
        mask = (
            dfs.reviews[PullRequestReview.state.name].values
            == ReviewResolution.CHANGES_REQUESTED.value
        )
        events_now[PullRequestEvent.CHANGES_REQUESTED] = set(original_reviews_index[mask])
        if self._with_time_machine:
            mask &= (submitted_at >= time_from) & (submitted_at < time_to)
            events_time_machine[PullRequestEvent.CHANGES_REQUESTED] = set(
                original_reviews_index[mask],
            )

        if len(self._environments) == 0:
            events_time_machine[PullRequestEvent.DEPLOYED] = events_now[
                PullRequestEvent.DEPLOYED
            ] = set()
            return events_time_machine, events_now

        events_time_machine[PullRequestEvent.DEPLOYED] = set()
        events_now[PullRequestEvent.DEPLOYED] = set()
        if self._environments:
            for dep_index in range(len(self._environments)):
                prs = dfs.prs.index.values
                deployed = self._calcs["deploy"]["time"][dep_index].calc_deployed(
                    columns=(dfs.prs[PullRequest.merged_at.name].values, prs),
                )
                events_time_machine[PullRequestEvent.DEPLOYED] |= set(
                    prs[deployed < np.array(self._time_to, dtype=deployed.dtype)],
                )
                events_now[PullRequestEvent.DEPLOYED] |= set(prs[deployed == deployed])

        return events_time_machine, events_now

    @staticmethod
    def _dt64_to_pydt(dt: Optional[np.datetime64]) -> Optional[datetime]:
        if dt is None:
            return None
        return dt.item().replace(tzinfo=timezone.utc)


@sentry_span
async def filter_pull_requests(
    events: set[PullRequestEvent],
    stages: set[PullRequestStage],
    time_from: datetime,
    time_to: datetime,
    repos: set[str],
    participants: PRParticipants,
    labels: LabelFilter,
    jira: JIRAFilter,
    environments: Optional[Sequence[str]],
    exclude_inactive: bool,
    bots: set[str],
    release_settings: ReleaseSettings,
    logical_settings: LogicalRepositorySettings,
    updated_min: Optional[datetime],
    updated_max: Optional[datetime],
    prefixer: Prefixer,
    account: int,
    meta_ids: tuple[int, ...],
    mdb: Database,
    pdb: Database,
    rdb: Database,
    cache: Optional[aiomcache.Client],
) -> tuple[list[PullRequestListItem], dict[str, Deployment]]:
    """Filter GitHub pull requests according to the specified criteria.

    We call _filter_pull_requests() to ignore all but the first result. We've got
    @cached.postprocess inside and it requires the wrapped function to return all the relevant
    post-load dependencies.

    :param repos: List of repository names without the service prefix.
    :param updated_min: The real time_from since when we fetch PRs.
    :param updated_max: The real time_to until when we fetch PRs.
    """
    prs, deployments, _, _ = await _filter_pull_requests(
        events,
        stages,
        time_from,
        time_to,
        repos,
        participants,
        labels,
        jira,
        environments,
        exclude_inactive,
        bots,
        release_settings,
        logical_settings,
        updated_min,
        updated_max,
        prefixer,
        account,
        meta_ids,
        mdb,
        pdb,
        rdb,
        cache,
    )
    return prs, deployments


def _postprocess_filtered_prs(
    result: tuple[list[PullRequestListItem], dict[str, Deployment], LabelFilter, JIRAFilter],
    labels: LabelFilter,
    jira: JIRAFilter,
    **_,
):
    prs, deployments, cached_labels, cached_jira = result
    if not cached_labels.compatible_with(labels) or not cached_jira.compatible_with(jira):
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
    left_deployments = [
        pr.deployments for pr in prs if pr.deployments is not None and len(pr.deployments)
    ]
    if left_deployments:
        left_deployments = np.unique(np.concatenate(left_deployments))
        deployments = {k: deployments[k] for k in left_deployments if k in deployments}
    else:
        deployments = {}
    return prs, deployments, labels, jira


def _extract_pr_labels(pr: PullRequestListItem) -> Optional[set[str]]:
    if pr.labels is None:
        return None
    return {lbl.name for lbl in pr.labels}


def _extract_jira_labels(pr: PullRequestListItem) -> Optional[set[str]]:
    if pr.jira is None:
        return None
    labels = set()
    for issue in pr.jira:
        if issue.labels is not None:
            labels.update(s.lower() for s in issue.labels)
    return labels or None


def _filter_by_labels(
    prs: list[PullRequestListItem],
    labels: LabelFilter,
    labels_getter: Callable[[PullRequestListItem], Optional[set[str]]],
) -> list[PullRequestListItem]:
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


def _filter_by_jira_epics(
    prs: list[PullRequestListItem],
    epics: set[str],
) -> list[PullRequestListItem]:
    new_prs = []
    for pr in prs:
        if pr.jira is None:
            continue
        for issue in pr.jira:
            if issue.epic in epics:
                new_prs.append(pr)
                break
    return new_prs


def _filter_by_jira_issue_types(
    prs: list[PullRequestListItem],
    types: set[str],
) -> list[PullRequestListItem]:
    new_prs = []
    for pr in prs:
        if pr.jira is None:
            continue
        for issue in pr.jira:
            if issue.type.lower() in types:
                new_prs.append(pr)
                break
    return new_prs


@cached(
    exptime=short_term_exptime,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda time_from, time_to, repos, events, stages, participants, environments, exclude_inactive, release_settings, logical_settings, updated_min, updated_max, **_: (  # noqa
        time_from.timestamp(),
        time_to.timestamp(),
        ",".join(sorted(repos)),
        ",".join(s.name.lower() for s in sorted(events)),
        ",".join(s.name.lower() for s in sorted(stages)),
        sorted((k.name.lower(), sorted(v)) for k, v in participants.items()),
        ",".join(sorted(environments or [])),
        exclude_inactive,
        updated_min.timestamp() if updated_min is not None else None,
        updated_max.timestamp() if updated_max is not None else None,
        release_settings,
        logical_settings,
    ),
    postprocess=_postprocess_filtered_prs,
)
async def _filter_pull_requests(
    events: set[PullRequestEvent],
    stages: set[PullRequestStage],
    time_from: datetime,
    time_to: datetime,
    repos: set[str],
    participants: PRParticipants,
    labels: LabelFilter,
    jira: JIRAFilter,
    environments: Optional[Sequence[str]],
    exclude_inactive: bool,
    bots: set[str],
    release_settings: ReleaseSettings,
    logical_settings: LogicalRepositorySettings,
    updated_min: Optional[datetime],
    updated_max: Optional[datetime],
    prefixer: Prefixer,
    account: int,
    meta_ids: tuple[int, ...],
    mdb: Database,
    pdb: Database,
    rdb: Database,
    cache: Optional[aiomcache.Client],
) -> tuple[list[PullRequestListItem], dict[str, Deployment], LabelFilter, JIRAFilter]:
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
    environments_task = asyncio.create_task(
        fetch_repository_environments(
            repos,
            environments,
            prefixer,
            account,
            rdb,
            cache,
            time_from=time_from,
            time_to=time_to,
        ),
    )
    branches, default_branches = await BranchMiner.load_branches(
        repos, prefixer, meta_ids, mdb, cache,
    )

    async def done_flow():
        released_facts, ambiguous = await DonePRFactsLoader.load_precomputed_done_facts_filters(
            time_from,
            time_to,
            repos,
            participants,
            labels,
            default_branches,
            exclude_inactive,
            release_settings,
            prefixer,
            account,
            pdb,
        )
        deps = await PullRequestMiner.fetch_pr_deployments(
            {node_id for node_id, _ in released_facts}, account, pdb, rdb,
        )
        UnfreshPullRequestFactsFetcher.append_deployments(released_facts, deps, log)
        return released_facts, ambiguous, deps

    (
        (pr_miner, unreleased_facts, matched_bys, unreleased_prs_event),
        (released_facts, ambiguous, done_deps),
    ) = await gather(
        PullRequestMiner.mine(
            date_from,
            date_to,
            time_from,
            time_to,
            repos,
            participants,
            labels,
            jira,
            True,
            branches,
            default_branches,
            exclude_inactive,
            release_settings,
            logical_settings,
            prefixer,
            account,
            meta_ids,
            mdb,
            pdb,
            rdb,
            cache,
            truncate=False,
            updated_min=updated_min,
            updated_max=updated_max,
        ),
        done_flow(),
    )
    add_pdb_misses(
        pdb,
        "load_precomputed_done_facts_filters/ambiguous",
        remove_ambiguous_prs(released_facts, ambiguous, matched_bys),
    )
    UnfreshPullRequestFactsFetcher.append_deployments(
        unreleased_facts, pr_miner.dfs.deployments, log,
    )
    # we want the released PR facts to overwrite the others
    facts = {**unreleased_facts, **released_facts}
    # precomputed facts miss `jira` property; we need it inside df_from_structs()
    PullRequestJiraMapper.apply_empty_to_pr_facts(facts)
    del unreleased_facts, released_facts
    deployment_names = np.unique(
        np.concatenate(
            [
                pr_miner.dfs.deployments.index.get_level_values(2).values,
                done_deps.index.get_level_values(1).values,
            ],
        ),
    )
    del done_deps
    deps_task = asyncio.create_task(
        load_included_deployments(
            deployment_names, logical_settings, prefixer, account, meta_ids, mdb, rdb, cache,
        ),
        name="filter_pull_requests/load_included_deployments",
    )

    prs = await list_with_yield(pr_miner, "PullRequestMiner.__iter__")

    with sentry_sdk.start_span(op="PullRequestFactsMiner.__call__"):
        facts_miner = PullRequestFactsMiner(bots)
        missed_done_facts = []
        missed_open_facts = []
        missed_merged_unreleased_facts = []

        async def store_missed_done_facts():
            nonlocal missed_done_facts
            await defer(
                store_precomputed_done_facts(
                    *zip(*missed_done_facts),
                    time_to,
                    default_branches,
                    release_settings,
                    account,
                    pdb,
                ),
                "store_precomputed_done_facts(%d)" % len(missed_done_facts),
            )
            missed_done_facts = []

        async def store_missed_open_facts():
            nonlocal missed_open_facts
            await defer(
                store_open_pull_request_facts(missed_open_facts, account, pdb),
                "store_open_pull_request_facts(%d)" % len(missed_open_facts),
            )
            missed_open_facts = []

        async def store_missed_merged_unreleased_facts():
            nonlocal missed_merged_unreleased_facts
            await defer(
                store_merged_unreleased_pull_request_facts(
                    missed_merged_unreleased_facts,
                    time_to,
                    matched_bys,
                    default_branches,
                    release_settings,
                    account,
                    pdb,
                    unreleased_prs_event,
                ),
                "store_merged_unreleased_pull_request_facts(%d)"
                % len(missed_merged_unreleased_facts),
            )
            missed_merged_unreleased_facts = []

        fact_evals = 0
        hit_facts_counter = 0
        missed_done_facts_counter = (
            missed_open_facts_counter
        ) = missed_merged_unreleased_facts_counter = 0
        bad_prs = []
        for i, pr in enumerate(prs):
            node_id, repo = (
                pr.pr[PullRequest.node_id.name],
                pr.pr[PullRequest.repository_full_name.name],
            )
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
            left_dfs_mask = np.ones(len(prs), dtype=bool)
            left_dfs_mask[bad_prs] = False
            new_dfs = {key: df for key, df in pr_miner.dfs.items()}
            new_dfs["prs"] = pr_miner.dfs.prs.take(np.flatnonzero(left_dfs_mask))
            pr_miner = PullRequestMiner(PRDataFrames(**new_dfs))
            # the order is already reversed
            for i in bad_prs:
                del prs[i]
        set_pdb_hits(pdb, "filter_pull_requests/facts", hit_facts_counter)
        set_pdb_misses(pdb, "filter_pull_requests/done_facts", missed_done_facts_counter)
        set_pdb_misses(pdb, "filter_pull_requests/open_facts", missed_open_facts_counter)
        set_pdb_misses(
            pdb,
            "filter_pull_requests/merged_unreleased_facts",
            missed_merged_unreleased_facts_counter,
        )
        log.info("total fact evals: %d", fact_evals)

    prs, included_deps = await gather(
        list_with_yield(
            PullRequestListMiner(
                prs,
                pr_miner.dfs,
                facts,
                events,
                stages,
                time_from,
                time_to,
                True,
                await environments_task,
            ),
            "PullRequestListMiner.__iter__",
        ),
        deps_task,
    )
    log.debug("return %d PRs", len(prs))
    return prs, included_deps, labels, jira


@sentry_span
async def fetch_pr_deployments(
    pr_node_ids: Collection[int],
    logical_settings: LogicalRepositorySettings,
    prefixer: Prefixer,
    account: int,
    meta_ids: tuple[int, ...],
    mdb: Database,
    pdb: Database,
    rdb: Database,
    cache: Optional[aiomcache.Client],
) -> tuple[pd.DataFrame, asyncio.Task]:
    """Load PR deployments by PR node IDs and schedule the task to load details about each \
    deployment."""
    deps = await PullRequestMiner.fetch_pr_deployments(pr_node_ids, account, pdb, rdb)
    dep_names = deps.index.get_level_values(2).unique()
    included_task = asyncio.create_task(
        load_included_deployments(
            dep_names, logical_settings, prefixer, account, meta_ids, mdb, rdb, cache,
        ),
        name=f"fetch_pull_requests/load_included_deployments({len(dep_names)})",
    )
    return deps, included_task


@sentry_span
@cached(
    exptime=short_term_exptime,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda prs, release_settings, logical_settings, environments, **_: (
        ";".join(
            "%s:%s" % (repo, ",".join(map(str, sorted(numbers))))
            for repo, numbers in sorted(prs.items())
        ),
        release_settings,
        logical_settings,
        ",".join(sorted(environments if environments is not None else [])),
    ),
)
async def fetch_pull_requests(
    prs: dict[str, set[int]],
    bots: set[str],
    release_settings: ReleaseSettings,
    logical_settings: LogicalRepositorySettings,
    environments: Optional[Sequence[str]],
    prefixer: Prefixer,
    account: int,
    meta_ids: tuple[int, ...],
    mdb: Database,
    pdb: Database,
    rdb: Database,
    cache: Optional[aiomcache.Client],
) -> tuple[list[PullRequestListItem], dict[str, Deployment]]:
    """
    List GitHub pull requests by repository and numbers.

    :params prs: For each repository name without the prefix, there is a set of PR numbers to list.
    """
    mined_prs, dfs, facts, _, deployments_task = await _fetch_pull_requests(
        prs,
        bots,
        release_settings,
        logical_settings,
        prefixer,
        account,
        meta_ids,
        mdb,
        pdb,
        rdb,
        cache,
    )
    if not mined_prs:
        return [], {}
    repo_envs = await fetch_repository_environments(
        prs,
        environments,
        prefixer,
        account,
        rdb,
        cache,
        time_from=dfs.prs[PullRequest.created_at.name].min(),
        time_to=dfs.prs[PullRequest.created_at.name].max(),
    )
    miner = PullRequestListMiner(
        mined_prs,
        dfs,
        facts,
        set(),
        set(),
        datetime(1970, 1, 1, tzinfo=timezone.utc),
        datetime.now(timezone.utc),
        False,
        repo_envs,
    )
    prs = await list_with_yield(miner, "PullRequestListMiner.__iter__")
    await deployments_task
    return prs, deployments_task.result()


async def _fetch_pull_requests(
    prs: dict[str, set[int]],
    bots: set[str],
    release_settings: ReleaseSettings,
    logical_settings: LogicalRepositorySettings,
    prefixer: Prefixer,
    account: int,
    meta_ids: tuple[int, ...],
    mdb: Database,
    pdb: Database,
    rdb: Database,
    cache: Optional[aiomcache.Client],
) -> tuple[
    list[MinedPullRequest],
    PRDataFrames,
    PullRequestFactsMap,
    dict[str, ReleaseMatch],
    Optional[asyncio.Task],
]:
    assert prs
    branches, default_branches = await BranchMiner.load_branches(
        prs, prefixer, meta_ids, mdb, cache,
    )
    filters = [
        and_(
            PullRequest.repository_full_name == drop_logical_repo(repo),
            PullRequest.number.in_(numbers),
            PullRequest.acc_id.in_(meta_ids),
        )
        for repo, numbers in prs.items()
    ]
    selects = [select(PullRequest).where(f) for f in filters]
    # execute UNION ALL of at most 10 SELECTs to help PostgreSQL parallelization
    select_batches = [selects[i : i + 10] for i in range(0, len(selects), 10)]
    queries = [union_all(*b) if len(b) > 1 else b[0] for b in select_batches]  # sqlite sucks
    tasks = [
        read_sql_query(query, mdb, PullRequest, index=PullRequest.node_id.name)
        for query in queries
    ]
    tasks.append(
        DonePRFactsLoader.load_precomputed_done_facts_reponums(
            prs, default_branches, release_settings, prefixer, account, pdb,
        ),
    )

    *prs_dfs, (facts, ambiguous) = await gather(*tasks)
    prs_df = pd.concat(prs_dfs) if len(prs_dfs) > 1 else prs_dfs[0]
    PullRequestMiner.adjust_pr_closed_merged_timestamps(prs_df)

    return await unwrap_pull_requests(
        prs_df,
        facts,
        ambiguous,
        None,
        None,
        JIRAEntityToFetch.ISSUES,
        branches,
        default_branches,
        bots,
        release_settings,
        logical_settings,
        prefixer,
        account,
        meta_ids,
        mdb,
        pdb,
        rdb,
        cache,
        repositories=prs.keys(),
    )


async def unwrap_pull_requests(
    prs_df: pd.DataFrame,
    precomputed_done_facts: PullRequestFactsMap,
    precomputed_ambiguous_done_facts: dict[str, list[int]],
    check_runs_task: Optional[asyncio.Task],
    deployments_task: Optional[asyncio.Task],
    with_jira: JIRAEntityToFetch | int,
    branches: pd.DataFrame,
    default_branches: dict[str, str],
    bots: set[str],
    release_settings: ReleaseSettings,
    logical_settings: LogicalRepositorySettings,
    prefixer: Prefixer,
    account: int,
    meta_ids: tuple[int, ...],
    mdb: Database,
    pdb: Database,
    rdb: Database,
    cache: Optional[aiomcache.Client],
    resolve_rebased: bool = True,
    repositories: Optional[set[str] | KeysView[str]] = None,
) -> tuple[
    list[MinedPullRequest],
    PRDataFrames,
    PullRequestFactsMap,
    dict[str, ReleaseMatch],
    asyncio.Task,
]:
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

    if check_runs_task is None:
        closed_pr_mask = prs_df[PullRequest.closed_at.name].notnull().values
        check_runs_task = asyncio.create_task(
            PullRequestMiner.fetch_pr_check_runs(
                prs_df.index.values[closed_pr_mask],
                prs_df.index.values[~closed_pr_mask],
                account,
                meta_ids,
                mdb,
                pdb,
                cache,
            ),
            name=f"unwrap_pull_requests/fetch_pr_check_runs({len(prs_df)})",
        )
    if deployments_task is None:
        merged_pr_ids = prs_df.index.values[prs_df[PullRequest.merged_at.name].notnull().values]
        deployments_task = asyncio.create_task(
            fetch_pr_deployments(
                merged_pr_ids,
                logical_settings,
                prefixer,
                account,
                meta_ids,
                mdb,
                pdb,
                rdb,
                cache,
            ),
            name=f"unwrap_pull_requests/fetch_pr_deployments({len(merged_pr_ids)})",
        )

    if repositories is None:
        repositories = logical_settings.with_logical_prs(
            prs_df[PullRequest.repository_full_name.name].values,
        )
    if resolve_rebased:
        dags = await fetch_precomputed_commit_history_dags(
            prs_df[PullRequest.repository_full_name.name].unique(), account, pdb, cache,
        )
        dags = await fetch_repository_commits(
            dags, branches, BRANCH_FETCH_COMMITS_COLUMNS, True, account, meta_ids, mdb, pdb, cache,
        )
        prs_df = await PullRequestMiner.mark_dead_prs(
            prs_df, branches, dags, account, meta_ids, mdb, pdb, PullRequest,
        )
    facts, ambiguous = precomputed_done_facts, precomputed_ambiguous_done_facts
    now = datetime.now(timezone.utc)
    if rel_time_from := prs_df[PullRequest.merged_at.name].nonemin():
        milestone_prs = prs_df[
            [
                PullRequest.merge_commit_sha.name,
                PullRequest.merge_commit_id.name,
                PullRequest.merged_at.name,
                PullRequest.repository_full_name.name,
            ]
        ]
        milestone_prs.columns = [
            Release.sha.name,
            Release.commit_id.name,
            Release.published_at.name,
            Release.repository_full_name.name,
        ]
        milestone_releases = dummy_releases_df().append(milestone_prs.reset_index(drop=True))
        milestone_releases = milestone_releases.take(
            np.flatnonzero(is_not_null(milestone_releases[Release.sha.name].values)),
        )
        releases, matched_bys = await ReleaseLoader.load_releases(
            prs_df[PullRequest.repository_full_name.name].unique(),
            branches,
            default_branches,
            rel_time_from,
            now,
            release_settings,
            logical_settings,
            prefixer,
            account,
            meta_ids,
            mdb,
            pdb,
            rdb,
            cache,
        )
        add_pdb_misses(
            pdb,
            "load_precomputed_done_facts_reponums/ambiguous",
            remove_ambiguous_prs(facts, ambiguous, matched_bys),
        )
        dags, unreleased = await gather(
            load_commit_dags(
                releases.append(milestone_releases), account, meta_ids, mdb, pdb, cache,
            ),
            # not nonemax() here! we want NaT-s inside load_merged_unreleased_pull_request_facts
            MergedPRFactsLoader.load_merged_unreleased_pull_request_facts(
                prs_df,
                releases[Release.published_at.name].max(),
                LabelFilter.empty(),
                matched_bys,
                default_branches,
                release_settings,
                prefixer,
                account,
                pdb,
            ),
        )
    else:
        releases, matched_bys, unreleased = dummy_releases_df(), {}, {}
        dags = await fetch_precomputed_commit_history_dags(
            prs_df[PullRequest.repository_full_name.name].unique(), account, pdb, cache,
        )
    for k, v in unreleased.items():
        if k not in facts:
            facts[k] = v
    empty_jira = LoadedJIRADetails.empty()
    for v in facts.values():
        if v.jira is None:
            v.jira = empty_jira
    (dfs, _, _), pr_check_runs, (pr_deployments, deployments_task) = await gather(
        PullRequestMiner.mine_by_ids(
            prs_df,
            unreleased,
            repositories,
            now,
            releases,
            matched_bys,
            branches,
            default_branches,
            dags,
            release_settings,
            logical_settings,
            prefixer,
            account,
            meta_ids,
            mdb,
            pdb,
            rdb,
            cache,
            with_jira=with_jira,
            skip_check_runs=True,
            skip_deployments=True,
        ),
        check_runs_task,
        deployments_task,
    )

    dfs.prs = split_logical_prs(dfs.prs, dfs.labels, repositories, logical_settings)
    dfs.check_runs = pr_check_runs
    dfs.deployments = pr_deployments
    log = logging.getLogger(f"{metadata.__package__}.unwrap_pull_requests/append_deployments")
    UnfreshPullRequestFactsFetcher.append_deployments(facts, dfs.deployments, log)

    prs = await list_with_yield(PullRequestMiner(dfs), "PullRequestMiner.__iter__")

    filtered_prs = []
    with sentry_sdk.start_span(op="PullRequestFactsMiner.__call__", description=str(len(prs))):
        facts_miner = PullRequestFactsMiner(bots)
        pdb_misses = 0
        for pr in prs:
            node_id, repo = (
                pr.pr[PullRequest.node_id.name],
                pr.pr[PullRequest.repository_full_name.name],
            )
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
    if with_jira == JIRAEntityToFetch.NOTHING:
        # we need it inside df_from_structs()
        PullRequestJiraMapper.apply_empty_to_pr_facts(facts)
    return filtered_prs, dfs, facts, matched_bys, deployments_task


def pr_facts_stages_masks(pr_facts: pd.DataFrame) -> npt.NDArray[int]:
    """Given a df of PullRequestFacts columns return the masks representing their stages.

    Each mask is an integer where the bit at position PullRequestEvent.STAGE.value is 1.
    This function returns the same stages collected by
    PullRequestListMiner._collect_events_and_stages.

    Use `pr_stages_mask()` to build a comparable bitmask starting from a set of stage names.
    """
    # 8 stages, use stage int value in the enum as position in the bitmask
    masks = np.zeros(len(pr_facts), np.uint8)
    if pr_facts.empty:
        return masks

    force_push_dropped_mask = pr_facts.done.values & pr_facts.force_push_dropped.values
    masks[force_push_dropped_mask] |= _pr_stage_enum_mask(PullRequestStage.FORCE_PUSH_DROPPED)

    release_ignored_mask = pr_facts.done.values & pr_facts.release_ignored.values
    # stages are prioritized, so update masks only when it's still 0
    masks[(masks == 0) & release_ignored_mask] |= _pr_stage_enum_mask(
        PullRequestStage.RELEASE_IGNORED,
    )
    # DONE is an exception, it lives together with FORCE_PUSH_DROPPED or RELEASE_IGNORED_MASK
    masks[pr_facts.done.values] |= _pr_stage_enum_mask(PullRequestStage.DONE)
    masks[(masks == 0) & ~np.isnat(pr_facts.merged.values)] |= _pr_stage_enum_mask(
        PullRequestStage.RELEASING,
    )
    masks[(masks == 0) & ~np.isnat(pr_facts.approved.values)] |= _pr_stage_enum_mask(
        PullRequestStage.MERGING,
    )
    masks[(masks == 0) & ~np.isnat(pr_facts.first_review_request.values)] |= _pr_stage_enum_mask(
        PullRequestStage.REVIEWING,
    )
    # everything is at least wip
    masks[masks == 0] = _pr_stage_enum_mask(PullRequestStage.WIP)

    return masks


def pr_stages_mask(stages: Collection[str]) -> int:
    """Return the bitmask representing a set of pull request stages.

    `stages` are strings included in the web PullRequestStage enum.

    """
    mask = 0
    for stage in stages:
        stage_enum = getattr(PullRequestStage, stage.upper())
        mask |= _pr_stage_enum_mask(stage_enum)
    return mask


def _pr_stage_enum_mask(stage: PullRequestStage) -> int:
    """Return the bitmask for a single pull request stage expressed as an enum."""
    return 1 << (stage.value - 1)
