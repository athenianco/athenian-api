import asyncio
from datetime import datetime, timedelta, timezone
import logging
import pickle
from typing import Callable, Dict, Generator, List, Optional, Sequence, Set, Tuple

import aiomcache
import databases
import numpy as np
import pandas as pd
import sentry_sdk
from sqlalchemy import and_, or_, select

from athenian.api import COROUTINE_YIELD_EVERY_ITER, list_with_yield, metadata
from athenian.api.async_read_sql_query import read_sql_query
from athenian.api.cache import cached, CancelCache
from athenian.api.controllers.datetime_utils import coarsen_time_interval
from athenian.api.controllers.features.github.pull_request_metrics import AllCounter, \
    MergingPendingCounter, \
    MergingTimeCalculator, ReleasePendingCounter, ReleaseTimeCalculator, ReviewPendingCounter, \
    ReviewTimeCalculator, StagePendingDependencyCalculator, WorkInProgressPendingCounter, \
    WorkInProgressTimeCalculator
from athenian.api.controllers.features.metric import Metric
from athenian.api.controllers.features.metric_calculator import df_from_dataclasses
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.bots import bots
from athenian.api.controllers.miners.github.branches import extract_branches
from athenian.api.controllers.miners.github.precomputed_prs import \
    load_merged_unreleased_pull_request_facts, load_precomputed_done_facts_filters, \
    load_precomputed_done_facts_reponums, store_merged_unreleased_pull_request_facts, \
    store_open_pull_request_facts, store_precomputed_done_facts
from athenian.api.controllers.miners.github.pull_request import ImpossiblePullRequest, \
    PRDataFrames, PullRequestFactsMiner, PullRequestMiner, ReviewResolution
from athenian.api.controllers.miners.github.release import dummy_releases_df, \
    fetch_precomputed_commit_history_dags, load_commit_dags, load_releases
from athenian.api.controllers.miners.types import Label, MinedPullRequest, PRParticipants, \
    PullRequestEvent, PullRequestFacts, PullRequestJIRAIssueItem, PullRequestListItem, \
    PullRequestStage
from athenian.api.controllers.settings import ReleaseMatch, ReleaseMatchSetting
from athenian.api.db import set_pdb_hits, set_pdb_misses
from athenian.api.defer import defer
from athenian.api.models.metadata import PREFIXES
from athenian.api.models.metadata.github import PullRequest, PullRequestCommit, PullRequestLabel, \
    PullRequestReview, PullRequestReviewComment, Release
from athenian.api.models.metadata.jira import Issue
from athenian.api.tracing import sentry_span


class PullRequestListMiner:
    """Collect various PR metadata for displaying PRs on the frontend."""

    _prefix = PREFIXES["github"]
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
        all_counter = self.DummyAllCounter(quantiles=(0, 1))
        pending_counter = StagePendingDependencyCalculator(all_counter, quantiles=(0, 1))
        self._counter_deps = [all_counter, pending_counter]
        self._calcs = {
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
        }
        self._no_time_from = datetime(year=1970, month=1, day=1, tzinfo=timezone.utc)
        assert isinstance(time_from, datetime)
        self._time_from = time_from
        self._time_to = time_to
        self._now = datetime.now(tz=timezone.utc)
        self._with_time_machine = with_time_machine

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
        pr_node_id = pr.pr[PullRequest.node_id.key]
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
        author = pr.pr[PullRequest.user_login.key]
        external_reviews_mask = pr.reviews[PullRequestReview.user_login.key].values != author
        external_review_times = pr.reviews[PullRequestReview.created_at.key].values[
            external_reviews_mask]
        first_review = pd.Timestamp(external_review_times.min(), tz=timezone.utc) \
            if len(external_review_times) > 0 else None
        review_comments = (
            pr.review_comments[PullRequestReviewComment.user_login.key].values != author
        ).sum()
        delta_comments = len(pr.review_comments) - review_comments
        reviews = external_reviews_mask.sum()
        updated_at = pr.pr[PullRequest.updated_at.key]
        assert updated_at == updated_at
        if pr.labels.empty:
            labels = None
        else:
            labels = [
                Label(name=name, description=description, color=color)
                for name, description, color in zip(
                    pr.labels[PullRequestLabel.name.key].values,
                    pr.labels[PullRequestLabel.description.key].values,
                    pr.labels[PullRequestLabel.color.key].values,
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
                    pr.jiras[Issue.title.key].values,
                    pr.jiras["epic"].values,
                    pr.jiras[Issue.labels.key].values,
                    pr.jiras[Issue.type.key].values,
                )
            ]
        return PullRequestListItem(
            repository=self._prefix + pr.pr[PullRequest.repository_full_name.key],
            number=pr.pr[PullRequest.number.key],
            title=pr.pr[PullRequest.title.key],
            size_added=pr.pr[PullRequest.additions.key],
            size_removed=pr.pr[PullRequest.deletions.key],
            files_changed=pr.pr[PullRequest.changed_files.key],
            created=pr.pr[PullRequest.created_at.key],
            updated=updated_at,
            closed=facts_now.closed,
            comments=len(pr.comments) + delta_comments,
            commits=len(pr.commits),
            review_requested=facts_now.first_review_request_exact,
            first_review=first_review,
            approved=facts_now.approved,
            review_comments=review_comments,
            reviews=reviews,
            merged=facts_now.merged,
            released=facts_now.released,
            release_url=pr.release[Release.url.key],
            events_now=events_now,
            stages_now=stages_now,
            events_time_machine=events_time_machine,
            stages_time_machine=stages_time_machine,
            stage_timings=stage_timings,
            participants=pr.participants(),
            labels=labels,
            jira=jira,
        )

    @sentry_span
    def _calc_stage_details(self) -> Dict[str, Sequence[int]]:
        facts = self._facts
        if len(facts) == 0 or len(self._prs) == 0:
            return {k: [] for k in self._calcs}
        node_id_key = PullRequest.node_id.key
        df_facts = df_from_dataclasses((facts[pr.pr[node_id_key]] for pr in self._prs),
                                       length=len(self._prs))
        dtype = df_facts["created"].dtype
        no_time_from = np.array([self._no_time_from.replace(tzinfo=None)], dtype=dtype)
        now = np.array([self._now.replace(tzinfo=None)], dtype=dtype)
        stage_timings = {}
        empty_group = [np.array([], dtype=int)]  # makes `samples` empty, we don't need them
        for dep in self._counter_deps:
            dep(df_facts, no_time_from, now, empty_group)
        for k, calcs in self._calcs.items():
            time_calc, pending_counter = calcs["time"], calcs["pending_count"]
            pending_counter(df_facts, no_time_from, now, empty_group)

            kwargs = {
                "override_event_time": now - np.timedelta64(timedelta(seconds=1)),  # < time_max
                "override_event_indexes": np.nonzero(pending_counter.peek[0])[0],
            }
            if k == "review":
                kwargs["allow_unclosed"] = True
            time_calc(df_facts, no_time_from, now, empty_group, **kwargs)
            stage_timings[k] = time_calc.peek[0]
        return stage_timings

    @sentry_span
    def _calc_hard_events(self) -> Tuple[Dict[PullRequestEvent, Set[str]],
                                         Dict[PullRequestEvent, Set[str]]]:
        events_time_machine = {}
        events_now = {}
        dfs = self._dfs
        time_from = np.datetime64(self._time_from.replace(tzinfo=None))
        time_to = np.datetime64(self._time_to.replace(tzinfo=None))

        index = dfs.commits.index.get_level_values(0).values
        committed_date = dfs.commits[PullRequestCommit.committed_date.key].values
        events_now[PullRequestEvent.COMMITTED] = set(index[committed_date == committed_date])
        if self._with_time_machine:
            events_time_machine[PullRequestEvent.COMMITTED] = \
                set(index[(committed_date >= time_from) & (committed_date < time_to)])

        prr_ulk = PullRequestReview.user_login.key
        prr_sak = PullRequestReview.submitted_at.key
        reviews = dfs.reviews[[prr_ulk, prr_sak]].droplevel(1)
        original_reviews_index = reviews.index.values
        reviews = reviews.join(dfs.prs[[PullRequest.user_login.key, PullRequest.closed_at.key]],
                               rsuffix="_pr")
        index = reviews.index.values
        submitted_at = reviews[prr_sak].values
        closed_at = reviews[PullRequest.closed_at.key].values.astype(submitted_at.dtype)
        not_closed_mask = closed_at != closed_at
        closed_at[not_closed_mask] = submitted_at[not_closed_mask]
        mask = (
            (submitted_at <= closed_at)
            &
            (reviews[prr_ulk].values != reviews[PullRequest.user_login.key + "_pr"].values)
        )
        events_now[PullRequestEvent.REVIEWED] = set(index[mask])
        if self._with_time_machine:
            mask &= (submitted_at >= time_from) & (submitted_at < time_to)
            events_time_machine[PullRequestEvent.REVIEWED] = set(index[mask])

        submitted_at = dfs.reviews[prr_sak].values
        # no need to check submitted_at <= closed_at because GitHub disallows that
        mask = dfs.reviews[PullRequestReview.state.key].values == \
            ReviewResolution.CHANGES_REQUESTED.value
        events_now[PullRequestEvent.CHANGES_REQUESTED] = set(original_reviews_index[mask])
        if self._with_time_machine:
            mask &= (submitted_at >= time_from) & (submitted_at < time_to)
            events_time_machine[PullRequestEvent.CHANGES_REQUESTED] = \
                set(original_reviews_index[mask])

        return events_time_machine, events_now

    def __iter__(self) -> Generator[PullRequestListItem, None, None]:
        """Iterate over individual pull requests."""
        stage_timings = self._calc_stage_details()
        hard_events_time_machine, hard_events_now = self._calc_hard_events()
        facts = self._facts
        node_id_key = PullRequest.node_id.key
        for i, pr in enumerate(self._prs):
            pr_stage_timings = {}
            for k in self._calcs:
                seconds = stage_timings[k][i]
                if seconds is not None:
                    pr_stage_timings[k] = timedelta(seconds=seconds)
            item = self._compile(pr, facts[pr.pr[node_id_key]], pr_stage_timings,
                                 hard_events_time_machine, hard_events_now)
            if item is not None:
                yield item


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
                               release_settings: Dict[str, ReleaseMatchSetting],
                               updated_min: Optional[datetime],
                               updated_max: Optional[datetime],
                               limit: int,
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
    prs, _, _ = await _filter_pull_requests(
        events, stages, time_from, time_to, repos, participants, labels, jira, exclude_inactive,
        release_settings, updated_min, updated_max, limit, mdb, pdb, cache)
    return prs


def _postprocess_filtered_prs(result: Tuple[List[PullRequestListItem], LabelFilter, JIRAFilter],
                              labels: LabelFilter, jira: JIRAFilter, **_):
    prs, cached_labels, cached_jira = result
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
    return prs, labels, jira


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


@cached(
    exptime=PullRequestMiner.CACHE_TTL,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda time_from, time_to, repos, events, stages, participants, exclude_inactive, release_settings, updated_min, updated_max, limit, **_: (  # noqa
        time_from.timestamp(),
        time_to.timestamp(),
        ",".join(sorted(repos)),
        ",".join(s.name.lower() for s in sorted(events)),
        ",".join(s.name.lower() for s in sorted(stages)),
        sorted((k.name.lower(), sorted(v)) for k, v in participants.items()),
        exclude_inactive,
        updated_min.timestamp() if updated_min is not None else None,
        updated_max.timestamp() if updated_max is not None else None,
        limit,
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
                                release_settings: Dict[str, ReleaseMatchSetting],
                                updated_min: Optional[datetime],
                                updated_max: Optional[datetime],
                                limit: int,
                                mdb: databases.Database,
                                pdb: databases.Database,
                                cache: Optional[aiomcache.Client],
                                ) -> Tuple[List[PullRequestListItem], LabelFilter, JIRAFilter]:
    assert isinstance(events, set)
    assert isinstance(stages, set)
    assert isinstance(repos, set)
    assert (updated_min is None) == (updated_max is None)
    log = logging.getLogger("%s.filter_pull_requests" % metadata.__package__)
    # required to efficiently use the cache with timezones
    date_from, date_to = coarsen_time_interval(time_from, time_to)
    if updated_min is not None:
        coarsen_time_interval(updated_min, updated_max)
    branches, default_branches = await extract_branches(repos, mdb, cache)
    tasks = (
        PullRequestMiner.mine(
            date_from, date_to, time_from, time_to, repos, participants, labels, jira, branches,
            default_branches, exclude_inactive, release_settings, mdb, pdb, cache,
            truncate=False, updated_min=updated_min, updated_max=updated_max, limit=limit),
        load_precomputed_done_facts_filters(
            time_from, time_to, repos, participants, labels, default_branches,
            exclude_inactive, release_settings, pdb),
    )
    pr_miner, facts = await asyncio.gather(*tasks, return_exceptions=True)
    for r in (pr_miner, facts):
        if isinstance(r, Exception):
            raise r from None
    pr_miner, unreleased_facts, matched_bys, unreleased_prs_event = pr_miner
    # we want the released PR facts to overwrite the others
    facts, unreleased_facts = unreleased_facts, facts
    facts.update(unreleased_facts)
    del unreleased_facts
    facts = {k: v for k, (_, v) in facts.items()}

    prs = await list_with_yield(pr_miner, "PullRequestMiner.__iter__")

    with sentry_sdk.start_span(op="PullRequestFactsMiner.__call__"):
        facts_miner = PullRequestFactsMiner(await bots(mdb))
        missed_done_facts = []
        missed_open_facts = []
        missed_merged_unreleased_facts = []

        async def store_missed_done_facts():
            nonlocal missed_done_facts
            await defer(store_precomputed_done_facts(
                *zip(*missed_done_facts), default_branches, release_settings, pdb),
                "store_precomputed_done_facts(%d)" % len(missed_done_facts))
            missed_done_facts = []

        async def store_missed_open_facts():
            nonlocal missed_open_facts
            await defer(store_open_pull_request_facts(missed_open_facts, pdb),
                        "store_open_pull_request_facts(%d)" % len(missed_open_facts))
            missed_open_facts = []

        async def store_missed_merged_unreleased_facts():
            nonlocal missed_merged_unreleased_facts
            await defer(store_merged_unreleased_pull_request_facts(
                missed_merged_unreleased_facts, matched_bys, default_branches,
                release_settings, pdb, unreleased_prs_event),
                "store_merged_unreleased_pull_request_facts(%d)" %
                len(missed_merged_unreleased_facts))
            missed_merged_unreleased_facts = []

        fact_evals = 0
        hit_facts_counter = 0
        missed_done_facts_counter = missed_open_facts_counter = \
            missed_merged_unreleased_facts_counter = 0
        bad_prs = []
        for i, pr in enumerate(prs):
            node_id = pr.pr[PullRequest.node_id.key]
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
                    missed_done_facts.append((pr, (None, pr_facts)))
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

    log.debug("return %d PRs", len(prs))
    return prs, labels, jira


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
    mined_prs, dfs, facts, _ = await _fetch_pull_requests(prs, release_settings, mdb, pdb, cache)
    if not mined_prs:
        return []
    miner = PullRequestListMiner(
        mined_prs, dfs, facts, set(), set(),
        datetime(1970, 1, 1, tzinfo=timezone.utc), datetime.now(timezone.utc), False)
    return await list_with_yield(miner, "PullRequestListMiner.__iter__")


async def _fetch_pull_requests(prs: Dict[str, Set[int]],
                               release_settings: Dict[str, ReleaseMatchSetting],
                               mdb: databases.Database,
                               pdb: databases.Database,
                               cache: Optional[aiomcache.Client],
                               ) -> Tuple[List[MinedPullRequest],
                                          PRDataFrames,
                                          Dict[str, PullRequestFacts],
                                          Dict[str, ReleaseMatch]]:
    branches, default_branches = await extract_branches(prs, mdb, cache)
    filters = [and_(PullRequest.repository_full_name == repo, PullRequest.number.in_(numbers))
               for repo, numbers in prs.items()]
    tasks = [
        read_sql_query(select([PullRequest])
                       .where(or_(*filters))
                       .order_by(PullRequest.node_id),
                       mdb, PullRequest, index=PullRequest.node_id.key),
        load_precomputed_done_facts_reponums(prs, default_branches, release_settings, pdb),
    ]
    prs_df, facts = await asyncio.gather(*tasks, return_exceptions=True)
    for r in (prs, facts):
        if isinstance(r, Exception):
            raise r from None
    if prs_df.empty:
        return [], PRDataFrames(*(pd.DataFrame() for _ in range(9))), {}, {}
    now = datetime.now(timezone.utc)
    if rel_time_from := prs_df[PullRequest.merged_at.key].nonemin():
        milestone_prs = prs_df[[PullRequest.merge_commit_sha.key,
                                PullRequest.merge_commit_id.key,
                                PullRequest.merged_at.key,
                                PullRequest.repository_full_name.key]]
        milestone_prs.columns = [
            Release.sha.key, Release.commit_id.key, Release.published_at.key,
            Release.repository_full_name.key,
        ]
        milestone_releases = dummy_releases_df().append(milestone_prs.reset_index(drop=True))
        milestone_releases = milestone_releases.take(np.where(
            milestone_releases[Release.sha.key].notnull())[0])
        releases, matched_bys = await load_releases(
            prs, branches, default_branches, rel_time_from, now, release_settings, mdb, pdb, cache)
        tasks = [
            load_commit_dags(releases.append(milestone_releases), mdb, pdb, cache),
            # not nonemax() here! we want NaT-s inside load_merged_unreleased_pull_request_facts
            load_merged_unreleased_pull_request_facts(
                prs_df, releases[Release.published_at.key].max(), LabelFilter.empty(),
                matched_bys, default_branches, release_settings, pdb),
        ]
        dags, unreleased = await asyncio.gather(*tasks, return_exceptions=True)
        for r in (dags, unreleased):
            if isinstance(r, Exception):
                raise r from None
    else:
        releases, matched_bys, unreleased = dummy_releases_df(), {}, {}
        dags = await fetch_precomputed_commit_history_dags(
            prs_df[PullRequest.repository_full_name.key].unique(), pdb, cache)
    dfs, _, _ = await PullRequestMiner.mine_by_ids(
        prs_df, unreleased, now, releases, matched_bys, branches, default_branches, dags,
        release_settings, mdb, pdb, cache)
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
            if (node_id := pr.pr[PullRequest.node_id.key]) not in facts:
                try:
                    facts[node_id] = None, facts_miner(pr)
                except ImpossiblePullRequest:
                    continue
                finally:
                    pdb_misses += 1
            filtered_prs.append(pr)

    facts = {k: v for k, (_, v) in facts.items()}
    set_pdb_hits(pdb, "fetch_pull_requests/facts", len(filtered_prs) - pdb_misses)
    set_pdb_misses(pdb, "fetch_pull_requests/facts", pdb_misses)
    return filtered_prs, dfs, facts, matched_bys
