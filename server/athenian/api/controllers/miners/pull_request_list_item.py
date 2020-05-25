from dataclasses import dataclass
from datetime import timedelta
from enum import auto, IntEnum
from typing import Collection, Dict, Mapping, Optional

import pandas as pd


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


class Property(IntEnum):
    """PR's modelled lifecycle stage or corresponding events between `time_from` and `time_to`."""

    # stages begin
    WIP = auto()
    REVIEWING = auto()
    MERGING = auto()
    RELEASING = auto()
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
    RELEASE_HAPPENED = auto()
    # events end


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
    properties: Collection[Property]
    participants: Mapping[ParticipationKind, Collection[str]]
