from datetime import datetime, timedelta
from enum import auto, IntEnum
from typing import Any, Dict, List, Mapping, Optional, Set, Union

import numpy as np
import pandas as pd

from athenian.api.controllers.settings import ReleaseMatch
from athenian.api.models.metadata.github import PullRequest, PullRequestComment, \
    PullRequestCommit, PullRequestReview, Release
from athenian.api.typing_utils import dataclass


class PRParticipationKind(IntEnum):
    """Developer relationship with a pull request.

    These values are written to the precomputed DB, so be careful with changing them.
    """

    AUTHOR = 1
    REVIEWER = 2
    COMMENTER = 3
    COMMIT_AUTHOR = 4
    COMMIT_COMMITTER = 5
    MERGER = 6
    RELEASER = 7


PRParticipants = Mapping[PRParticipationKind, Set[str]]


class ReleaseParticipationKind(IntEnum):
    """Developer relationship with a release."""

    PR_AUTHOR = 1
    COMMIT_AUTHOR = 2
    RELEASER = 3


ReleaseParticipants = Mapping[ReleaseParticipationKind, List[str]]


class Property(IntEnum):
    """PR's modelled lifecycle stage or corresponding events between `time_from` and `time_to`."""

    # stages begin
    WIP = auto()
    REVIEWING = auto()
    MERGING = auto()
    RELEASING = auto()
    FORCE_PUSH_DROPPED = auto()
    DONE = auto()
    # stages end
    # events begin
    CREATED = auto()
    COMMIT_HAPPENED = auto()
    REVIEW_HAPPENED = auto()
    APPROVE_HAPPENED = auto()
    REVIEW_REQUEST_HAPPENED = auto()
    CHANGES_REQUEST_HAPPENED = auto()
    MERGE_HAPPENED = auto()
    REJECTION_HAPPENED = auto()
    RELEASE_HAPPENED = auto()
    # events end


class PullRequestEvent(IntEnum):
    """PR's modelled lifecycle event."""

    CREATED = auto()
    COMMITTED = auto()
    REVIEWED = auto()
    APPROVED = auto()
    REVIEW_REQUESTED = auto()
    CHANGES_REQUESTED = auto()
    MERGED = auto()
    REJECTED = auto()
    RELEASED = auto()


class PullRequestStage(IntEnum):
    """PR's modelled lifecycle stage."""

    WIP = auto()
    REVIEWING = auto()
    MERGING = auto()
    RELEASING = auto()
    FORCE_PUSH_DROPPED = auto()
    DONE = auto()


@dataclass(slots=True, frozen=True)
class Label:
    """Pull request label."""

    name: str
    description: Optional[str]
    color: str


@dataclass(slots=True, frozen=True)
class PullRequestJIRAIssueItem:
    """JIRA PR properties."""

    id: str
    title: str
    labels: Optional[Set[str]]
    epic: Optional[str]
    type: str


@dataclass(slots=True, frozen=True)
class PullRequestListItem:
    """General PR properties used to list PRs on the frontend."""

    repository: str
    number: int
    title: str
    size_added: int
    size_removed: int
    files_changed: int
    created: pd.Timestamp
    updated: pd.Timestamp
    closed: Optional[pd.Timestamp]
    comments: int
    commits: int
    review_requested: Optional[pd.Timestamp]
    first_review: Optional[pd.Timestamp]
    approved: Optional[pd.Timestamp]
    review_comments: int
    reviews: int
    merged: Optional[pd.Timestamp]
    released: Optional[pd.Timestamp]
    release_url: str
    stage_timings: Dict[str, timedelta]
    events_time_machine: Optional[Set[PullRequestEvent]]
    stages_time_machine: Optional[Set[PullRequestStage]]
    events_now: Set[PullRequestEvent]
    stages_now: Set[PullRequestStage]
    participants: PRParticipants
    labels: List[Label]
    jira: Optional[List[PullRequestJIRAIssueItem]]


@dataclass(slots=True, frozen=True)
class MinedPullRequest:
    """All the relevant information we are able to load from the metadata DB about a PR.

    All the DataFrame-s have a two-layered index:
    1. pull request id
    2. own id
    The artificial first index layer makes it is faster to select data belonging to a certain PR.
    """

    pr: Dict[str, Any]
    commits: pd.DataFrame
    reviews: pd.DataFrame
    review_comments: pd.DataFrame
    review_requests: pd.DataFrame
    comments: pd.DataFrame
    release: Dict[str, Any]
    labels: pd.DataFrame
    jiras: pd.DataFrame

    def participants(self) -> PRParticipants:
        """Collect unique developer logins that are mentioned in this pull request."""
        author = self.pr[PullRequest.user_login.key]
        merger = self.pr[PullRequest.merged_by_login.key]
        releaser = self.release[Release.author.key]
        participants = {
            PRParticipationKind.AUTHOR: {author} if author else set(),
            PRParticipationKind.REVIEWER: self._extract_people(
                self.reviews, PullRequestReview.user_login.key),
            PRParticipationKind.COMMENTER: self._extract_people(
                self.comments, PullRequestComment.user_login.key),
            PRParticipationKind.COMMIT_COMMITTER: self._extract_people(
                self.commits, PullRequestCommit.committer_login.key),
            PRParticipationKind.COMMIT_AUTHOR: self._extract_people(
                self.commits, PullRequestCommit.author_login.key),
            PRParticipationKind.MERGER: {merger} if merger else set(),
            PRParticipationKind.RELEASER: {releaser} if releaser else set(),
        }
        reviewers = participants[PRParticipationKind.REVIEWER]
        if author in reviewers:
            reviewers.remove(author)
        return participants

    @staticmethod
    def _extract_people(df: pd.DataFrame, col: str) -> Set[str]:
        values = df[col].values
        return set(values[np.where(values)[0]])


@dataclass(slots=True, frozen=True, eq=False, first_mutable="jira_id")
class PullRequestFacts:
    """Various PR event timestamps and other properties."""

    created: pd.Timestamp
    first_commit: Optional[pd.Timestamp]
    work_began: pd.Timestamp
    last_commit_before_first_review: Optional[pd.Timestamp]
    last_commit: Optional[pd.Timestamp]
    merged: Optional[pd.Timestamp]
    closed: Optional[pd.Timestamp]
    first_comment_on_first_review: Optional[pd.Timestamp]
    first_review_request: Optional[pd.Timestamp]
    first_review_request_exact: Optional[pd.Timestamp]
    approved: Optional[pd.Timestamp]
    last_review: Optional[pd.Timestamp]
    released: Optional[pd.Timestamp]
    done: bool
    reviews: np.ndarray
    activity_days: np.ndarray
    size: int
    force_push_dropped: bool
    # Mutable optional fields go below.
    jira_id: Optional[str] = None
    repository_full_name: Optional[str] = None
    author: Optional[str] = None
    merger: Optional[str] = None
    releaser: Optional[str] = None

    def max_timestamp(self) -> pd.Timestamp:
        """Find the maximum timestamp contained in the struct."""
        if self.released is not None:
            return self.released
        if self.closed is not None:
            return self.closed
        return max(t for t in (self.created, self.first_commit, self.last_commit,
                               self.first_review_request, self.last_review)
                   if t is not None)

    def truncate(self, dt: Union[pd.Timestamp, datetime]) -> "PullRequestFacts":
        """Create a copy of the facts without timestamps bigger than or equal to `dt`."""
        changed = []
        for k, v in self.items():  # do not use dataclasses.asdict() - very slow
            if isinstance(v, pd.Timestamp) and v >= dt:
                changed.append(k)
        if not changed:
            return self
        dikt = dict(self)
        for k in changed:
            dikt[k] = None
        dikt["done"] = dikt["released"] or dikt["force_push_dropped"] or (
            dikt["closed"] and not dikt["merged"])
        return PullRequestFacts(**dikt)

    def validate(self) -> None:
        """Ensure that there are no NaNs."""
        for k, v in self.items():  # do not use dataclasses.asdict() - very slow
            if isinstance(v, np.ndarray):
                assert (v == v).all(), k
            else:
                assert v == v, k

    def __eq__(self, other) -> bool:
        """Compare this object to another."""
        if self is other:
            return True

        if self.__class__ is not other.__class__:
            raise NotImplementedError(
                f"Cannot compare {self.__class__} and {other.__class__}")

        for k, v in self.items():
            v_other = getattr(other, k)
            if v is v_other:
                continue

            if isinstance(v, np.ndarray) and isinstance(v_other, np.ndarray):
                if not np.array_equal(v, v_other):
                    return False
            elif v != v_other:
                return False

        return True

    def __str__(self) -> str:
        """Format for human-readability."""
        # do not use dataclasses.asdict() - very slow
        return "{\n\t%s\n}" % ",\n\t".join("%s: %s" % (k, v) for k, v in self.items())

    def __lt__(self, other: "PullRequestFacts") -> bool:
        """Order by `work_began`."""
        if self.work_began != other.work_began:
            return self.work_began < other.work_began
        return self.created < other.created


def nonemin(*args: Union[pd.Timestamp, type(None)]) -> Optional[pd.Timestamp]:
    """Find the minimum of several dates handling NaNs gracefully."""
    if all(arg is None for arg in args):
        return None
    return min(arg for arg in args if arg)


def nonemax(*args: Union[pd.Timestamp, type(None)]) -> Optional[pd.Timestamp]:
    """Find the maximum of several dates handling NaNs gracefully."""
    if all(arg is None for arg in args):
        return None
    return max(arg for arg in args if arg)


@dataclass(slots=True, frozen=True, first_mutable="repository_full_name")
class ReleaseFacts:
    """Various release properties and statistics."""

    published: datetime
    publisher: str
    matched_by: ReleaseMatch
    age: timedelta
    additions: int
    deletions: int
    commits_count: int
    prs: Dict[str, np.ndarray]
    commit_authors: List[str]
    # Mutable optional fields go below.
    repository_full_name: Optional[str] = None

    def max_timestamp(self) -> datetime:
        """Find the maximum timestamp contained in the struct."""
        return self.published
