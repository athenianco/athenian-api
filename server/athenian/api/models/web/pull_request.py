from datetime import datetime
from typing import Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.linked_jira_issue import LinkedJIRAIssue
from athenian.api.models.web.pull_request_event import PullRequestEvent
from athenian.api.models.web.pull_request_label import PullRequestLabel
from athenian.api.models.web.pull_request_participant import PullRequestParticipant
from athenian.api.models.web.pull_request_stage import PullRequestStage
from athenian.api.models.web.stage_timings import StageTimings


class PullRequest(Model):
    """
    Details of a pull request.

    All the attributes, stats and events correspond to "today", not `date_to`, *except for
    the PR stages `wip`, `reviewing`, `merging`, `releasing`*, which correspond to `date_to`.
    """

    repository: str
    number: int
    title: str
    size_added: int
    size_removed: int
    files_changed: int
    created: datetime
    updated: datetime
    closed: Optional[datetime]
    comments: int
    commits: int
    review_requested: Optional[datetime]
    first_review: Optional[datetime]
    approved: Optional[datetime]
    review_comments: Optional[int]
    reviews: Optional[int]
    merged: Optional[datetime]
    merged_with_failed_check_runs: Optional[list[str]]
    released: Optional[datetime]
    release_url: Optional[str]
    stage_timings: StageTimings
    events_time_machine: Optional[list[str]]
    stages_time_machine: Optional[list[str]]
    events_now: list[str]
    stages_now: list[str]
    participants: list[PullRequestParticipant]
    labels: Optional[list[PullRequestLabel]]
    jira: Optional[list[LinkedJIRAIssue]]
    deployments: Optional[list[str]]

    def __lt__(self, other: "PullRequest") -> bool:
        """Compute self < other."""
        return self.updated < other.updated

    def validate_events_time_machine(
        self,
        events_time_machine: Optional[list[str]],
    ) -> Optional[list[str]]:
        """
        Sets events_time_machine of this PullRequest.

        list of PR events which happened until `date_to`. `date_from` does not matter.

        :param events_time_machine: The events_time_machine of this PullRequest.
        """
        for prop in events_time_machine or []:
            if prop not in PullRequestEvent:
                raise ValueError("Invalid `events_time_machine`: %s" % events_time_machine)

        return events_time_machine

    def validate_stages_time_machine(
        self,
        stages_time_machine: Optional[list[str]],
    ) -> Optional[list[str]]:
        """Sets the stages_time_machine of this PullRequest.

        :param stages_time_machine: The stages_time_machine of this PullRequest.
        """
        for prop in stages_time_machine or []:
            if prop not in PullRequestStage:
                raise ValueError("Invalid `stages_time_machine`: %s" % stages_time_machine)

        return stages_time_machine

    def validate_events_now(self, events_now: list[str]) -> list[str]:
        """
        Sets events_now of this PullRequest.

        list of PR events that ever happened.

        :param events_now: The events_now of this PullRequest.
        """
        if events_now is None:
            raise ValueError("`events_now` must not be null")
        for prop in events_now:
            if prop not in PullRequestEvent:
                raise ValueError("Invalid `events_now`: %s" % events_now)

        return events_now

    def validate_stages_now(self, stages_now: list[str]) -> list[str]:
        """Sets the stages_now of this PullRequest.

        :param stages_now: The stages_now of this PullRequest.
        """
        if stages_now is None:
            raise ValueError("`stages_now` must not be null")
        for prop in stages_now:
            if prop not in PullRequestStage:
                raise ValueError("Invalid `stages_now`: %s" % stages_now)

        return stages_now
