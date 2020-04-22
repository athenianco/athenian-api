from datetime import date, datetime, timezone
import pickle
from typing import Collection, Dict, Generator, List, Mapping, Optional, Set

import aiomcache
import databases
import pandas as pd

from athenian.api.cache import cached
from athenian.api.controllers.features.github.pull_request_metrics import \
    MergingTimeCalculator, ReleaseTimeCalculator, ReviewTimeCalculator, \
    WorkInProgressTimeCalculator
from athenian.api.controllers.miners.github.pull_request import MinedPullRequest, \
    PullRequestMiner, PullRequestTimesMiner, ReviewResolution
from athenian.api.controllers.miners.pull_request_list_item import ParticipationKind, Property, \
    PullRequestListItem
from athenian.api.controllers.settings import ReleaseMatchSetting
from athenian.api.models.metadata.github import PullRequest, PullRequestComment, \
    PullRequestCommit, PullRequestReview, PullRequestReviewRequest, Release
from athenian.api.models.web import PullRequestParticipant


class PullRequestListMiner(PullRequestTimesMiner):
    """Collect various PR metadata for displaying PRs on the frontend."""

    def __init__(self, *args, **kwargs):
        """Initialize a new instance of `PullRequestListMiner`."""
        super().__init__(*args, **kwargs)
        self._properties = set()
        self._participants = {}
        self._time_from = pd.NaT
        self._time_to = pd.NaT
        self._calcs = {
            "wip": WorkInProgressTimeCalculator(),
            "review": ReviewTimeCalculator(),
            "merge": MergingTimeCalculator(),
            "release": ReleaseTimeCalculator(),
        }

    @property
    def properties(self) -> Set[Property]:
        """Return the required PR properties."""
        return self._properties

    @properties.setter
    def properties(self, value: Collection[Property]):
        """Set the required PR properties."""
        self._properties = set(value)

    @property
    def participants(self) -> Dict[ParticipationKind, Set[str]]:
        """Return the required PR participants."""
        return self._participants

    @participants.setter
    def participants(self, value: Mapping[ParticipationKind, Collection[str]]):
        """Set the required PR participants."""
        self._participants = {k: set(v) for k, v in value.items()}

    @property
    def time_from(self) -> datetime:
        """Return the minimum of the allowed events time span."""
        return self._time_from

    @time_from.setter
    def time_from(self, value: datetime):
        """Set the minimum of the allowed events time span."""
        assert isinstance(value, datetime) and value.tzinfo is not None
        self._time_from = value

    @property
    def time_to(self) -> datetime:
        """Return the minimum of the allowed events time span."""
        return self._time_to

    @time_to.setter
    def time_to(self, value: datetime):
        """Set the maximum of the allowed events time span."""
        assert isinstance(value, datetime) and value.tzinfo is not None
        self._time_to = value

    def _match_participants(self, yours: Mapping[ParticipationKind, Set[str]]) -> bool:
        """Check the PR particpants for compatibility with self.participants.

        :return: True whether the PR satisfies the participants filter, otherwise False.
        """
        if not self.participants:
            return True
        for k, v in self.participants.items():
            if yours.get(k, set()).intersection(v):
                return True
        return False

    def _compile(self, pr: MinedPullRequest) -> Optional[PullRequestListItem]:
        """Match the PR to the required participants and properties."""
        prefix = "github.com/"
        author = pr.pr[PullRequest.user_login.key]
        merger = pr.pr[PullRequest.merged_by_login.key]
        releaser = pr.release[Release.author.key]
        participants = {
            ParticipationKind.AUTHOR: {prefix + author} if author else set(),
            ParticipationKind.REVIEWER: {
                (prefix + u) for u in pr.reviews[PullRequestReview.user_login.key] if u},
            ParticipationKind.COMMENTER: {
                (prefix + u) for u in pr.comments[PullRequestComment.user_login.key] if u},
            ParticipationKind.COMMIT_COMMITTER: {
                (prefix + u) for u in pr.commits[PullRequestCommit.committer_login.key] if u},
            ParticipationKind.COMMIT_AUTHOR: {
                (prefix + u) for u in pr.commits[PullRequestCommit.author_login.key] if u},
            ParticipationKind.MERGER: {prefix + merger} if merger else set(),
            ParticipationKind.RELEASER: {prefix + releaser} if releaser else set(),
        }
        try:
            participants[ParticipationKind.REVIEWER].remove(prefix + author)
        except (KeyError, TypeError):
            pass
        if not self._match_participants(participants):
            return None
        times = super()._compile(pr)
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
        time_from = self.time_from
        if times.created.best > time_from:
            props.add(Property.CREATED)
        if (pr.commits[PullRequestCommit.committed_date.key] > time_from).any():
            props.add(Property.COMMIT_HAPPENED)
        if (pr.reviews[PullRequestReview.submitted_at.key] > time_from).any():
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
        if not self.properties.intersection(props):
            return None
        review_requested = \
            pr.review_requests[PullRequestReviewRequest.created_at.key].max() or None
        time_to = min(self.time_to, datetime.now(timezone.utc))
        stage_timings = {}
        no_time_from = datetime(year=1970, month=1, day=1, tzinfo=timezone.utc)
        for k, calc in self._calcs.items():
            kwargs = {} if k != "review" else {"allow_unclosed": True}
            stage_timings[k] = calc.analyze(times, no_time_from, time_to, **kwargs)
        for prop, stage in ((Property.WIP, "wip"), (Property.REVIEWING, "review"),
                            (Property.MERGING, "merge"), (Property.RELEASING, "release")):
            if prop in props:
                stage_timings[stage] = self._calcs[stage].analyze(
                    times, no_time_from, time_to, override_event_time=time_to)
        return PullRequestListItem(
            repository=prefix + pr.pr[PullRequest.repository_full_name.key],
            number=pr.pr[PullRequest.number.key],
            title=pr.pr[PullRequest.title.key],
            size_added=pr.pr[PullRequest.additions.key],
            size_removed=pr.pr[PullRequest.deletions.key],
            files_changed=pr.pr[PullRequest.changed_files.key],
            created=pr.pr[PullRequest.created_at.key],
            updated=pr.pr[PullRequest.updated_at.key],
            closed=times.closed.best,
            comments=len(pr.comments),
            commits=len(pr.commits),
            review_requested=review_requested,
            approved=times.approved.best,
            review_comments=len(pr.review_comments),
            merged=times.merged.best,
            released=times.released.best,
            release_url=pr.release[Release.url.key],
            properties=props,
            stage_timings=stage_timings,
            participants=participants,
        )

    def __iter__(self) -> Generator[PullRequestListItem, None, None]:
        """Iterate over the individual pull requests."""
        for pr in PullRequestMiner.__iter__(self):
            item = self._compile(pr)
            if item is not None:
                yield item


@cached(
    exptime=PullRequestListMiner.CACHE_TTL,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda date_from, date_to, repos, properties, participants, **_: (
        date_from.toordinal(),
        date_to.toordinal(),
        ",".join(sorted(repos)),
        ",".join(s.name.lower() for s in sorted(set(properties))),
        sorted((k.name.lower(), sorted(set(v))) for k, v in participants.items()),
    ),
)
async def filter_pull_requests(
        properties: Collection[Property],
        date_from: date,
        date_to: date,
        repos: Collection[str],
        release_settings: Dict[str, ReleaseMatchSetting],
        participants: Mapping[ParticipationKind, Collection[str]],
        db: databases.Database, cache: Optional[aiomcache.Client],
) -> List[PullRequestListItem]:
    """Filter GitHub pull requests according to the specified criteria.

    :param repos: List of repository names without the service prefix.
    """
    assert isinstance(date_from, date) and not isinstance(date_from, datetime)
    assert isinstance(date_to, date) and not isinstance(date_to, datetime)
    miner = await PullRequestListMiner.mine(
        date_from, date_to, repos, release_settings,
        participants.get(PullRequestParticipant.STATUS_AUTHOR, []), db, cache)
    miner.properties = properties
    miner.participants = participants
    miner.time_from = pd.Timestamp(date_from, tzinfo=timezone.utc)
    miner.time_to = pd.Timestamp(date_to, tzinfo=timezone.utc)
    return list(miner)
