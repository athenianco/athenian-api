from typing import Optional

from athenian.api.internal.miners.types import JIRAParticipants, JIRAParticipationKind
from athenian.api.models.web.base_model_ import Model


class JIRAFilterWith(Model):
    """Group of JIRA issue participant names split by role."""

    assignees: Optional[list[Optional[str]]]
    reporters: Optional[list[str]]
    commenters: Optional[list[str]]

    def validate_reporters(self, reporters: Optional[list[str]]) -> Optional[list[str]]:
        """Sets the reporters of this JIRAFilterWith.

        Selected issue reporter users.

        :param reporters: The reporters of this JIRAFilterWith.
        """
        if reporters is not None:
            for i, reporter in enumerate(reporters):
                if reporter is None:
                    raise ValueError("`reporters[%d]` cannot be null" % i)
        return reporters

    def validate_commenters(self, commenters: Optional[list[str]]) -> Optional[list[str]]:
        """Sets the commenters of this JIRAFilterWith.

        Selected issue commenter users.

        :param commenters: The commenters of this JIRAFilterWith.
        """
        if commenters is not None:
            for i, commenter in enumerate(commenters):
                if commenter is None:
                    raise ValueError("`commenters[%d]` cannot be null" % i)
        return commenters

    def as_participants(self) -> JIRAParticipants:
        """Convert to the internal representation."""
        result = {}
        if self.reporters:
            result[JIRAParticipationKind.REPORTER] = [*{*self.reporters}]
        if self.assignees:
            result[JIRAParticipationKind.ASSIGNEE] = [*{*self.assignees}]
        if self.commenters:
            result[JIRAParticipationKind.COMMENTER] = [*{*self.commenters}]
        return result
