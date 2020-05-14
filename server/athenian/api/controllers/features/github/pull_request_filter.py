from datetime import datetime, timedelta, timezone
from itertools import chain
import pickle
from typing import Collection, Dict, Generator, Iterable, Mapping, Optional, Set

import aiomcache
import databases
import pandas as pd
from sqlalchemy import select

from athenian.api.async_read_sql_query import read_sql_query
from athenian.api.cache import cached
from athenian.api.controllers.datetime_utils import coarsen_time_interval
from athenian.api.controllers.features.github.pull_request_metrics import \
    MergingTimeCalculator, ReleaseTimeCalculator, ReviewTimeCalculator, \
    WorkInProgressTimeCalculator
from athenian.api.controllers.miners.github.pull_request import dtmin, ImpossiblePullRequest, \
    MinedPullRequest, PullRequestMiner, PullRequestTimes, PullRequestTimesMiner, ReviewResolution
from athenian.api.controllers.miners.pull_request_list_item import ParticipationKind, Property, \
    PullRequestListItem
from athenian.api.controllers.settings import ReleaseMatchSetting
from athenian.api.models.metadata import PREFIXES
from athenian.api.models.metadata.github import PullRequest, PullRequestCommit, \
    PullRequestReview, PullRequestReviewComment, PullRequestReviewRequest, Release


class PullRequestListMiner:
    """Collect various PR metadata for displaying PRs on the frontend."""

    _prefix = PREFIXES["github"]

    def __init__(self,
                 prs_time_machine: Iterable[MinedPullRequest],
                 prs_today: Iterable[MinedPullRequest],
                 properties: Set[Property],
                 participants: Dict[ParticipationKind, Set[str]],
                 time_from: datetime):
        """Initialize a new instance of `PullRequestListMiner`."""
        self._prs_time_machine = prs_time_machine
        self._prs_today = prs_today
        self._times_miner = PullRequestTimesMiner()
        self._properties = properties
        self._participants = participants
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

    def _match_participants(self, pr: MinedPullRequest) -> bool:
        """Check the PR participants for compatibility with self.participants.

        :return: True whether the PR satisfies the participants filter, otherwise False.
        """
        if not self._participants:
            return True

        participants = pr.participants()
        for k, v in self._participants.items():
            if participants.get(k, set()).intersection(v):
                return True
        return False

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
        if ((pr.reviews[PullRequestReview.submitted_at.key] > time_from)
                & (pr.reviews[PullRequestReview.user_login.key] != author)).any():
            props.add(Property.REVIEW_HAPPENED)
        if times.first_review_request.value is not None and \
                times.first_review_request.value > time_from:
            props.add(Property.REVIEW_REQUEST_HAPPENED)
        if times.approved and times.approved.best > time_from:
            props.add(Property.APPROVE_HAPPENED)
        if times.merged and times.merged.best > time_from:
            props.add(Property.MERGE_HAPPENED)
        if times.released and times.released.best > time_from:
            props.add(Property.RELEASE_HAPPENED)
        if ((pr.reviews[PullRequestReview.state.key] == ReviewResolution.CHANGES_REQUESTED.value)
                & (pr.reviews[PullRequestReview.submitted_at.key] > time_from)).any():  # noqa
            props.add(Property.CHANGES_REQUEST_HAPPENED)
        return props

    def _compile(self,
                 pr_time_machine: MinedPullRequest,
                 times_time_machine: PullRequestTimes,
                 pr_today: MinedPullRequest,
                 times_today: PullRequestTimes,
                 ) -> Optional[PullRequestListItem]:
        """
        Match the PR to the required participants and properties.

        :param pr_time_machine: PR's metadata as of `time_to`.
        :param pr_today: Today's version of the PR's metadata.
        :param times_time_machine: Facts about the PR corresponding to [`time_from`, `time_to`].
        :param times_today: Facts about the PR as of datetime.now().
        """
        assert pr_time_machine.pr[PullRequest.node_id.key] == pr_today.pr[PullRequest.node_id.key]
        if not self._match_participants(pr_time_machine):
            return None
        props_time_machine = self._collect_properties(
            times_time_machine, pr_time_machine, self._time_from)
        if not self._properties.intersection(props_time_machine):
            return None
        props_today = self._collect_properties(times_today, pr_today, self._no_time_from)
        for p in range(Property.WIP, Property.DONE + 1):
            p = Property(p)
            if p in props_time_machine:
                props_today.add(p)
            else:
                try:
                    props_today.remove(p)
                except KeyError:
                    pass
        review_requested = \
            dtmin(pr_today.review_requests[PullRequestReviewRequest.created_at.key].max())
        author = pr_today.pr[PullRequest.user_id.key]
        review_comments = (
            pr_today.review_comments[PullRequestReviewComment.user_id.key].values != author
        ).sum()
        delta_comments = len(pr_today.review_comments) - review_comments
        reviews = (pr_today.reviews[PullRequestReview.user_id.key].values != author).sum()
        stage_timings = {}
        no_time_from = self._no_time_from
        now = self._now
        for k, (calc, prop) in self._calcs.items():
            kwargs = {} if k != "review" else {"allow_unclosed": True}
            if prop in props_today:
                kwargs["override_event_time"] = now - timedelta(seconds=1)  # < time_max
            stage_timings[k] = calc.analyze(times_today, no_time_from, now, **kwargs)
        updated_at = pr_today.pr[PullRequest.updated_at.key]
        assert updated_at == updated_at
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
            review_requested=review_requested,
            approved=times_today.approved.best,
            review_comments=review_comments,
            reviews=reviews,
            merged=times_today.merged.best,
            released=times_today.released.best,
            release_url=pr_today.release[Release.url.key],
            properties=props_today,
            stage_timings=stage_timings,
            participants=pr_today.participants(),
        )

    def __iter__(self) -> Generator[PullRequestListItem, None, None]:
        """Iterate over the individual pull requests."""
        for pr_time_machine, pr_today in zip(self._prs_time_machine, self._prs_today):
            try:
                item = self._compile(pr_time_machine, self._times_miner(pr_time_machine),
                                     pr_today, self._times_miner(pr_today))
            except ImpossiblePullRequest:
                continue
            if item is not None:
                yield item


@cached(
    exptime=PullRequestMiner.CACHE_TTL,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda time_from, time_to, repos, properties, participants, release_settings, **_: (
        time_from.timestamp(),
        time_to.timestamp(),
        ",".join(sorted(repos)),
        ",".join(s.name.lower() for s in sorted(set(properties))),
        sorted((k.name.lower(), sorted(set(v))) for k, v in participants.items()),
        release_settings,
    ),
)
async def filter_pull_requests(properties: Collection[Property],
                               time_from: datetime,
                               time_to: datetime,
                               repos: Collection[str],
                               release_settings: Dict[str, ReleaseMatchSetting],
                               participants: Mapping[ParticipationKind, Collection[str]],
                               db: databases.Database,
                               cache: Optional[aiomcache.Client],
                               ) -> Iterable[PullRequestListItem]:
    """Filter GitHub pull requests according to the specified criteria.

    :param repos: List of repository names without the service prefix.
    """
    # required to efficiently use the cache with timezones
    date_from, date_to = coarsen_time_interval(time_from, time_to)
    everybody = {p.split("/", 1)[1] for p in chain.from_iterable(participants.values())}
    miner_time_machine = await PullRequestMiner.mine(
        date_from, date_to, time_from, time_to, repos, release_settings, everybody, db, cache)
    prs_time_machine = list(miner_time_machine)

    now = datetime.now(tz=timezone.utc) + timedelta(days=1)
    tomorrow = datetime(year=now.year, month=now.month, day=now.day, tzinfo=now.tzinfo)

    if time_to != tomorrow:
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
        if done:
            # updated_at can be outside of `time_to` and missed in the cache
            updates = await db.fetch_all(
                select([PullRequest.node_id, PullRequest.updated_at])
                .where(PullRequest.node_id.in_([pr.pr[node_id_key] for pr in done])))
            updates = {p[0]: p[1] for p in updates}
            updated_at_key = PullRequest.updated_at.key
            for pr in done:
                ts = updates[pr.pr[node_id_key]]
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                pr.pr[updated_at_key] = ts
        if remined:
            prs = await read_sql_query(select([PullRequest])
                                       .where(PullRequest.node_id.in_(remined))
                                       .order_by(PullRequest.node_id),
                                       db, PullRequest, index=node_id_key)
            dfs = await PullRequestMiner.mine_by_ids(
                prs, prs[PullRequest.created_at.key].min(), tomorrow, release_settings, db, cache)
            prs_today = list(PullRequestMiner(prs, *dfs))
        else:
            prs_today = []
        prs_time_machine = [remined[pr.pr[node_id_key]] for pr in prs_today] + done
        prs_today += done
    else:
        prs_today = prs_time_machine
    properties = set(properties)
    participants = {k: set(v) for k, v in participants.items()}
    return PullRequestListMiner(prs_time_machine, prs_today, properties, participants, time_from)
