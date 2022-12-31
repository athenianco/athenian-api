from enum import IntEnum, auto
from typing import Mapping, Sequence


class PRParticipationKind(IntEnum):
    """Developer relationship with a pull request.

    These values are written to the precomputed DB and used in nativ eextensions,
    so you can only append new values without changing the old.
    """

    AUTHOR = auto()
    REVIEWER = auto()
    COMMENTER = auto()
    COMMIT_AUTHOR = auto()
    COMMIT_COMMITTER = auto()
    MERGER = auto()
    RELEASER = auto()


PRParticipants = Mapping[PRParticipationKind, set[str]]


class ReleaseParticipationKind(IntEnum):
    """Developer relationship with a release."""

    PR_AUTHOR = auto()
    COMMIT_AUTHOR = auto()
    RELEASER = auto()


ReleaseParticipants = Mapping[ReleaseParticipationKind, Sequence[int]]


class JIRAParticipationKind(IntEnum):
    """User relationship with a JIRA issue."""

    ASSIGNEE = auto()
    REPORTER = auto()
    COMMENTER = auto()


JIRAParticipants = Mapping[JIRAParticipationKind, Sequence[str]]
