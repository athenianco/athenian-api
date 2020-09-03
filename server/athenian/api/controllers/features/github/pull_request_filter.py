import asyncio
from datetime import datetime, timedelta, timezone
import logging
import pickle
from typing import Dict, Generator, List, Optional, Set, Tuple

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
from athenian.api.controllers.features.github.pull_request_metrics import \
    MergingTimeCalculator, ReleaseTimeCalculator, ReviewTimeCalculator, \
    WorkInProgressTimeCalculator
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.bots import bots
from athenian.api.controllers.miners.github.branches import extract_branches
from athenian.api.controllers.miners.github.precomputed_prs import discover_unreleased_prs, \
    load_precomputed_done_facts_filters, load_precomputed_done_facts_reponums, \
    store_merged_unreleased_pull_request_facts, store_open_pull_request_facts, \
    store_precomputed_done_facts
from athenian.api.controllers.miners.github.pull_request import ImpossiblePullRequest, \
    PullRequestFactsMiner, PullRequestMiner, ReviewResolution
from athenian.api.controllers.miners.github.release import dummy_releases_df, \
    fetch_precomputed_commit_history_dags, load_commit_dags, load_releases
from athenian.api.controllers.miners.types import Label, MinedPullRequest, Participants, \
    Property, PullRequestFacts, PullRequestListItem
from athenian.api.controllers.settings import ReleaseMatchSetting
from athenian.api.db import set_pdb_hits, set_pdb_misses
from athenian.api.defer import defer, wait_deferred
from athenian.api.models.metadata import PREFIXES
from athenian.api.models.metadata.github import PullRequest, PullRequestCommit, PullRequestLabel, \
    PullRequestReview, PullRequestReviewComment, Release
from athenian.api.tracing import sentry_span


class PullRequestListMiner:
    """Collect various PR metadata for displaying PRs on the frontend."""

    _prefix = PREFIXES["github"]
    log = logging.getLogger("%s.PullRequestListMiner" % metadata.__version__)

    def __init__(self,
                 prs: List[MinedPullRequest],
                 facts: Dict[str, PullRequestFacts],
                 properties: Set[Property],
                 time_from: datetime,
                 time_to: datetime):
        """Initialize a new instance of `PullRequestListMiner`."""
        self._prs = prs
        self._facts = facts
        self._properties = properties
        self._calcs = {
            "wip": (WorkInProgressTimeCalculator(quantiles=(0, 1)), Property.WIP),
            "review": (ReviewTimeCalculator(quantiles=(0, 1)), Property.REVIEWING),
            "merge": (MergingTimeCalculator(quantiles=(0, 1)), Property.MERGING),
            "release": (ReleaseTimeCalculator(quantiles=(0, 1)), Property.RELEASING),
        }
        self._no_time_from = datetime(year=1970, month=1, day=1, tzinfo=timezone.utc)
        assert isinstance(time_from, datetime)
        self._time_from = time_from
        self._time_to = time_to
        self._now = datetime.now(tz=timezone.utc)

    @classmethod
    def _collect_properties(cls,
                            facts: PullRequestFacts,
                            pr: MinedPullRequest,
                            time_from: datetime,
                            ) -> Set[Property]:
        np_time_from = np.datetime64(time_from.replace(tzinfo=None))
        author = pr.pr[PullRequest.user_login.key]
        props = set()
        if facts.done:
            if facts.force_push_dropped:
                props.add(Property.FORCE_PUSH_DROPPED)
            props.add(Property.DONE)
        elif facts.merged:
            props.add(Property.RELEASING)
        elif facts.approved:
            props.add(Property.MERGING)
        elif facts.first_review_request:
            props.add(Property.REVIEWING)
        else:
            props.add(Property.WIP)
        if facts.created.best > time_from:
            props.add(Property.CREATED)
        if (pr.commits[PullRequestCommit.committed_date.key].values > np_time_from).any():
            props.add(Property.COMMIT_HAPPENED)
        review_submitted_ats = pr.reviews[PullRequestReview.submitted_at.key].values
        if ((review_submitted_ats > np_time_from)
                & (pr.reviews[PullRequestReview.user_login.key].values != author)).any():
            props.add(Property.REVIEW_HAPPENED)
        if facts.first_review_request.value is not None and \
                facts.first_review_request.value > time_from:
            props.add(Property.REVIEW_REQUEST_HAPPENED)
        if facts.approved and facts.approved.best > time_from:
            props.add(Property.APPROVE_HAPPENED)
        if facts.merged and facts.merged.best > time_from:
            props.add(Property.MERGE_HAPPENED)
        if not facts.merged and facts.closed and facts.closed.best > time_from:
            props.add(Property.REJECTION_HAPPENED)
        if facts.released and facts.released.best > time_from:
            props.add(Property.RELEASE_HAPPENED)
        review_states = pr.reviews[PullRequestReview.state.key]
        if ((review_states.values == ReviewResolution.CHANGES_REQUESTED.value)
                & (review_submitted_ats > np_time_from)).any():
            props.add(Property.CHANGES_REQUEST_HAPPENED)
        return props

    def _compile(self,
                 pr: MinedPullRequest,
                 facts: PullRequestFacts,
                 ) -> Optional[PullRequestListItem]:
        """
        Match the PR to the required participants and properties and produce PullRequestListItem.

        We return None if the PR does not match.
        """
        pr_today = pr
        pr_time_machine = pr.truncate(
            self._time_to, ignore=("review_comments", "review_requests", "comments"))
        facts_today = facts
        facts_time_machine = facts.truncate(self._time_to)
        props_time_machine = self._collect_properties(
            facts_time_machine, pr_time_machine, self._time_from)
        if not self._properties.intersection(props_time_machine):
            return None
        props_today = self._collect_properties(facts_today, pr_today, self._no_time_from)
        author = pr_today.pr[PullRequest.user_id.key]
        external_reviews_mask = pr_today.reviews[PullRequestReview.user_id.key].values != author
        external_review_times = pr_today.reviews[PullRequestReview.created_at.key].values[
            external_reviews_mask]
        first_review = pd.Timestamp(external_review_times.min(), tz=timezone.utc) \
            if len(external_review_times) > 0 else None
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
            calc(facts_today, no_time_from, now, **kwargs)
            stage_timings[k] = calc.peek
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
            closed=facts_today.closed.best,
            comments=len(pr_today.comments) + delta_comments,
            commits=len(pr_today.commits),
            review_requested=facts_today.first_review_request.value,
            first_review=first_review,
            approved=facts_today.approved.best,
            review_comments=review_comments,
            reviews=reviews,
            merged=facts_today.merged.best,
            released=facts_today.released.best,
            release_url=pr_today.release[Release.url.key],
            properties=props_today,
            stage_timings=stage_timings,
            participants=pr_today.participants(),
            labels=labels,
        )

    def __iter__(self) -> Generator[PullRequestListItem, None, None]:
        """Iterate over individual pull requests."""
        facts = self._facts
        node_id_key = PullRequest.node_id.key
        for pr in self._prs:
            item = self._compile(pr, facts[pr.pr[node_id_key]])
            if item is not None:
                yield item


@sentry_span
async def filter_pull_requests(properties: Set[Property],
                               time_from: datetime,
                               time_to: datetime,
                               repos: Set[str],
                               participants: Participants,
                               labels: LabelFilter,
                               jira: JIRAFilter,
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
    prs, _, _ = await _filter_pull_requests(
        properties, time_from, time_to, repos, participants, labels, jira, exclude_inactive,
        release_settings, mdb, pdb, cache)
    return prs


def _postprocess_filtered_prs(result: Tuple[List[PullRequestListItem], LabelFilter, JIRAFilter],
                              labels: LabelFilter, jira: JIRAFilter, **_):
    prs, cached_labels, cached_jira = result
    if (not cached_labels.compatible_with(labels) or
            not cached_jira.compatible_with(jira)):
        raise CancelCache()
    if labels.include:
        prs = [pr for pr in prs
               if labels.include.intersection({label.name for label in (pr.labels or [])})]
    if labels.exclude:
        prs = [pr for pr in prs
               if not labels.exclude.intersection({label.name for label in (pr.labels or [])})]
    return prs, labels, jira


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
    version=2,
)
async def _filter_pull_requests(properties: Set[Property],
                                time_from: datetime,
                                time_to: datetime,
                                repos: Set[str],
                                participants: Participants,
                                labels: LabelFilter,
                                jira: JIRAFilter,
                                exclude_inactive: bool,
                                release_settings: Dict[str, ReleaseMatchSetting],
                                mdb: databases.Database,
                                pdb: databases.Database,
                                cache: Optional[aiomcache.Client],
                                ) -> Tuple[List[PullRequestListItem], LabelFilter, JIRAFilter]:
    assert isinstance(properties, set)
    assert isinstance(repos, set)
    log = logging.getLogger("%s.filter_pull_requests" % metadata.__package__)
    # required to efficiently use the cache with timezones
    date_from, date_to = coarsen_time_interval(time_from, time_to)
    branches, default_branches = await extract_branches(repos, mdb, cache)
    tasks = (
        PullRequestMiner.mine(
            date_from, date_to, time_from, time_to, repos, participants, labels, jira, branches,
            default_branches, exclude_inactive, release_settings, mdb, pdb, cache,
            truncate=False),
        load_precomputed_done_facts_filters(
            time_from, time_to, repos, participants, labels, default_branches, exclude_inactive,
            release_settings, pdb),
    )
    pr_miner, facts = await asyncio.gather(*tasks, return_exceptions=True)
    for r in (pr_miner, facts):
        if isinstance(r, Exception):
            raise r from None
    pr_miner, unreleased_facts, matched_bys = pr_miner
    # we want the released PR facts to overwrite the others
    facts, unreleased_facts = unreleased_facts, facts
    facts.update(unreleased_facts)
    del unreleased_facts

    prs = await list_with_yield(pr_miner, "PullRequestMiner.__iter__")

    with sentry_sdk.start_span(op="PullRequestFactsMiner.__call__"):
        facts_miner = PullRequestFactsMiner(await bots(mdb))
        missed_done_facts = []
        missed_open_facts = []
        missed_merged_unreleased_facts = []

        async def store_missed_done_facts():
            await defer(store_precomputed_done_facts(
                *zip(*missed_done_facts), default_branches, release_settings, pdb),
                "store_precomputed_done_facts(%d)" % len(missed_done_facts))

        async def store_missed_open_facts():
            await defer(store_open_pull_request_facts(missed_open_facts, pdb),
                        "store_open_pull_request_facts(%d)" % len(missed_open_facts))

        async def store_missed_merged_unreleased_facts():
            if pdb.url.dialect == "sqlite":
                await wait_deferred()  # wait for update_unreleased_prs
            await defer(store_merged_unreleased_pull_request_facts(
                missed_merged_unreleased_facts, matched_bys, default_branches,
                release_settings, pdb),
                "store_merged_unreleased_pull_request_facts(%d)" %
                len(missed_merged_unreleased_facts))

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
                    missed_done_facts.append((pr, pr_facts))
                    if (len(missed_done_facts) + 1) % 100 == 0:
                        await store_missed_done_facts()
                elif not pr_facts.closed:
                    missed_open_facts_counter += 1
                    missed_open_facts.append((pr.pr, pr_facts))
                    if (len(missed_open_facts) + 1) % 100 == 0:
                        await store_missed_open_facts()
                elif pr_facts.merged and not pr_facts.released:
                    missed_merged_unreleased_facts_counter += 1
                    missed_merged_unreleased_facts.append((pr.pr, pr_facts))
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
        PullRequestListMiner(prs, facts, properties, time_from, time_to),
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
        return []
    now = datetime.now(timezone.utc)
    rel_time_from = prs_df[PullRequest.merged_at.key].min()
    if rel_time_from == rel_time_from:
        releases, matched_bys = await load_releases(
            prs, branches, default_branches, rel_time_from, now, release_settings, mdb, pdb, cache)
        tasks = [
            load_commit_dags(releases, mdb, pdb, cache),
            discover_unreleased_prs(
                prs_df, releases[Release.published_at.key].max(), matched_bys, default_branches,
                release_settings, pdb),
        ]
        dags, unreleased = await asyncio.gather(*tasks, return_exceptions=True)
        for r in (dags, unreleased):
            if isinstance(r, Exception):
                raise r from None
    else:
        releases, matched_bys, unreleased = dummy_releases_df(), {}, {}
        dags = await fetch_precomputed_commit_history_dags(
            prs_df[PullRequest.repository_full_name.key].unique(), pdb, cache)
    dfs, _ = await PullRequestMiner.mine_by_ids(
        prs_df, unreleased, now, releases, matched_bys, branches, default_branches, dags,
        release_settings, mdb, pdb, cache)
    prs = await list_with_yield(PullRequestMiner(dfs), "PullRequestMiner.__iter__")
    for k, v in unreleased.items():
        if v is not None and k not in facts:
            facts[k] = v
    with sentry_sdk.start_span(op="PullRequestFactsMiner.__call__",
                               description=str(len(prs))):
        facts_miner = PullRequestFactsMiner(await bots(mdb))
        pdb_misses = 0
        for pr in prs:
            node_id = pr.pr[PullRequest.node_id.key]
            if node_id not in facts:
                facts[node_id] = facts_miner(pr)
                pdb_misses += 1
    miner = PullRequestListMiner(
        prs, facts, set(Property), prs_df[PullRequest.created_at.key].min(), now)
    prs = await list_with_yield(miner, "PullRequestListMiner.__iter__")
    set_pdb_hits(pdb, "filter_pull_requests/facts", len(prs) - pdb_misses)
    set_pdb_misses(pdb, "filter_pull_requests/facts", pdb_misses)
    return prs
