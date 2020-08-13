import dataclasses
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import auto, IntEnum
from typing import Any, Dict, Generic, List, Mapping, Optional, Set, TypeVar, Union

import numpy as np
import pandas as pd

from athenian.api.models.metadata.github import PullRequest, PullRequestComment, \
    PullRequestCommit, PullRequestReview, PullRequestReviewComment, PullRequestReviewRequest, \
    Release


class ParticipationKind(IntEnum):
    """The way the developer relates to a pull request.

    These values are written to the precomputed DB, so be careful with changing them.
    """

    AUTHOR = 1
    REVIEWER = 2
    COMMENTER = 3
    COMMIT_AUTHOR = 4
    COMMIT_COMMITTER = 5
    MERGER = 6
    RELEASER = 7


Participants = Mapping[ParticipationKind, Set[str]]


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


@dataclass(frozen=True)
class Label:
    """Pull request label."""

    name: str
    description: Optional[str]
    color: str


@dataclass(frozen=True)
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
    properties: Set[Property]
    participants: Participants
    labels: List[Label]


@dataclasses.dataclass(frozen=True)
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

    def participants(self) -> Participants:
        """Collect unique developer logins that are mentioned in this pull request."""
        author = self.pr[PullRequest.user_login.key]
        merger = self.pr[PullRequest.merged_by_login.key]
        releaser = self.release[Release.author.key]
        participants = {
            ParticipationKind.AUTHOR: {author} if author else set(),
            ParticipationKind.REVIEWER: self._extract_people(
                self.reviews, PullRequestReview.user_login.key),
            ParticipationKind.COMMENTER: self._extract_people(
                self.comments, PullRequestComment.user_login.key),
            ParticipationKind.COMMIT_COMMITTER: self._extract_people(
                self.commits, PullRequestCommit.committer_login.key),
            ParticipationKind.COMMIT_AUTHOR: self._extract_people(
                self.commits, PullRequestCommit.author_login.key),
            ParticipationKind.MERGER: {merger} if merger else set(),
            ParticipationKind.RELEASER: {releaser} if releaser else set(),
        }
        try:
            participants[ParticipationKind.REVIEWER].remove(author)
        except (KeyError, TypeError):
            pass
        return participants

    @staticmethod
    def _extract_people(df: pd.DataFrame, col: str) -> Set[str]:
        values = df[col].values
        return set(values[np.where(values)[0]])

    def truncate(self, dt: Union[pd.Timestamp, datetime],
                 ignore=tuple()) -> "MinedPullRequest":
        """
        Create a copy of the PR data without timestamps bigger than or equal to `dt`.

        :param ignore: Field names to not truncate.
        """
        pr = self.pr
        assert pr[PullRequest.created_at.key] < dt
        closed_at = pr[PullRequest.closed_at.key]
        if closed_at is not None and closed_at >= dt:
            pr = pr.copy()
            pr[PullRequest.closed_at.key] = None
            pr[PullRequest.merged_at.key] = None
        # we ignore PullRequest.updated_at
        release = self.release
        published_at = release[Release.published_at.key]
        if published_at is not None and published_at >= dt:
            release = {k: None for k in release}
        dfs = {}
        dt = np.datetime64(dt.replace(tzinfo=None))
        for name, col in (("commits", PullRequestCommit.committed_date),
                          ("review_requests", PullRequestReviewRequest.created_at),
                          ("review_comments", PullRequestReviewComment.created_at),
                          ("reviews", PullRequestReview.created_at),
                          ("comments", PullRequestComment.created_at)):
            df = getattr(self, name)  # type: pd.DataFrame
            if name not in ignore:
                left = np.where(df[col.key].values < dt)[0]
                if len(left) < len(df):
                    df = df.take(left)
            dfs[name] = df
        return MinedPullRequest(pr=pr, release=release, labels=self.labels, **dfs)


T = TypeVar("T")


class Fallback(Generic[T]):
    """
    A value with a "plan B".

    The idea is to return the backup in `Fallback.best` if the primary value is absent (None).
    We can check whether the primary value exists by `Fallback.value is None`.
    """

    def __init__(self, value: Optional[T], fallback: Union[None, T, "Fallback[T]"]):
        """Initialize a new instance of `Fallback`."""
        if value != value:  # NaN check
            value = None
        self.__value = value
        self.__fallback = fallback

    @property
    def best(self) -> Optional[T]:
        """The "best effort" value, either the primary or the backup one."""  # noqa: D401
        if self.__value is not None:
            return self.__value
        if isinstance(self.__fallback, Fallback):
            return self.__fallback.best
        return self.__fallback

    def __str__(self) -> str:
        """str()."""
        return "Fallback(%s, %s)" % (self.value, self.best)

    def __repr__(self) -> str:
        """repr()."""
        return "Fallback(%r, %r)" % (self.value, self.best)

    def __bool__(self) -> bool:
        """Return the value indicating whether there is any value, either primary or backup."""
        return self.best is not None

    def __lt__(self, other: "Fallback[T]") -> bool:
        """Implement <."""
        if not self or not other:
            raise ArithmeticError
        return self.best < other.best

    def __eq__(self, other: "Fallback[T]") -> bool:
        """Implement ==."""
        return self.best == other.best

    def __le__(self, other: "Fallback[T]") -> bool:
        """Implement <=."""
        if not self or not other:
            raise ArithmeticError
        return self.best <= other.best

    @property
    def value(self) -> Optional[T]:
        """The primary value."""  # noqa: D401
        return self.__value

    @classmethod
    def max(cls, *args: "Fallback[T]") -> "Fallback[T]":
        """Calculate the maximum of several Fallback.best-s."""
        return cls.agg(max, *args)

    @classmethod
    def min(cls, *args: "Fallback[T]") -> "Fallback[T]":
        """Calculate the minimum of several Fallback.best-s."""
        return cls.agg(min, *args)

    @classmethod
    def agg(cls, func: callable, *args: "Fallback[T]") -> "Fallback[T]":
        """Calculate an aggregation of several Fallback.best-s."""
        try:
            return cls(func(arg.best for arg in args if arg.best is not None), None)
        except ValueError:
            return cls(None, None)


DT = Union[pd.Timestamp, datetime, None]


@dataclasses.dataclass(frozen=True)
class PullRequestFacts:
    """Various PR event timestamps and other properties."""

    @property
    def work_began(self) -> Fallback[DT]:  # PR_B   noqa: D102
        return Fallback.min(self.created, self.first_commit)

    created: Fallback[DT]                                # PR_C
    first_commit: Fallback[DT]                           # PR_CC
    last_commit_before_first_review: Fallback[DT]        # PR_CFR
    last_commit: Fallback[DT]                            # PR_LC
    merged: Fallback[DT]                                 # PR_M
    closed: Fallback[DT]                                 # PR_CL
    first_comment_on_first_review: Fallback[DT]          # PR_W
    first_review_request: Fallback[DT]                   # PR_S
    approved: Fallback[DT]                               # PR_A
    last_review: Fallback[DT]                            # PR_LR
    first_passed_checks: Fallback[DT]                    # PR_VS
    last_passed_checks: Fallback[DT]                     # PR_VF
    released: Fallback[DT]                               # PR_R
    size: int

    def max_timestamp(self) -> DT:
        """Find the maximum timestamp contained in the struct."""
        return Fallback.max(*(v for v in vars(self).values()
                              if isinstance(v, Fallback))).best
        # do not use dataclasses.asdict() - very slow

    def truncate(self, dt: Union[pd.Timestamp, datetime]) -> "PullRequestFacts":
        """Create a copy of the facts without timestamps bigger than or equal to `dt`."""
        dikt = {}
        changed = False
        for k, v in vars(self).items():  # do not use dataclasses.asdict() - very slow
            if not isinstance(v, Fallback):
                dikt[k] = v
                continue
            if v:
                if v.best < dt:
                    if v.value is None or v.value < dt:
                        dikt[k] = v
                    else:
                        dikt[k] = Fallback(None, v.best)
                        changed = True
                else:
                    dikt[k] = Fallback(None, None)
                    changed = True
            else:
                dikt[k] = v
        if not changed:
            return self
        return PullRequestFacts(**dikt)

    def __str__(self) -> str:
        """Format for human-readability."""
        return "{\n\t%s\n}" % ",\n\t".join(
            "%s: %s" % (k, v.best if isinstance(v, Fallback) else v)
            for k, v in vars(self).items())  # do not use dataclasses.asdict() - very slow

    def __lt__(self, other: "PullRequestFacts") -> bool:
        """Order by `work_began`."""
        return self.work_began.best < other.work_began.best

    def __hash__(self) -> int:
        """Implement hash()."""
        return hash(str(self))


def dtmin(*args: Union[DT, float]) -> DT:
    """Find the minimum of several dates handling NaNs gracefully."""
    if all((arg != arg) for arg in args):
        return None
    return min(arg for arg in args if arg == arg)


def dtmax(*args: Union[DT, float]) -> DT:
    """Find the maximum of several dates handling NaNs gracefully."""
    if all((arg != arg) for arg in args):
        return None
    return max(arg for arg in args if arg == arg)
