from dataclasses import dataclass
from enum import IntEnum
from typing import Mapping, Sequence

import pandas as pd


class ParticipationKind(IntEnum):
    """The way the developer relates to a pull request."""

    AUTHOR = 1
    REVIEWER = 2
    COMMENTER = 3
    COMMIT_AUTHOR = 4
    COMMIT_COMMITTER = 5
    MERGER = 6


class Stage(IntEnum):
    """Modelled evolution pipeline stage."""

    WIP = 1
    REVIEW = 2
    MERGE = 3
    RELEASE = 4
    DONE = 5


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
    comments: int
    commits: int
    review_requested: bool
    review_comments: int
    merged: bool
    stage: Stage
    participants: Mapping[ParticipationKind, Sequence[str]]
