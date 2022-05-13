from datetime import datetime
from typing import List, Optional

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

    attribute_types = {
        "repository": str,
        "number": int,
        "title": str,
        "size_added": int,
        "size_removed": int,
        "files_changed": int,
        "created": datetime,
        "updated": datetime,
        "closed": datetime,
        "comments": int,
        "commits": int,
        "review_requested": datetime,
        "first_review": datetime,
        "approved": datetime,
        "review_comments": int,
        "reviews": int,
        "merged": datetime,
        "merged_with_failed_check_runs": Optional[List[str]],
        "released": Optional[datetime],
        "release_url": Optional[str],
        "stage_timings": StageTimings,
        "events_time_machine": Optional[List[str]],
        "stages_time_machine": Optional[List[str]],
        "events_now": List[str],
        "stages_now": List[str],
        "participants": List[PullRequestParticipant],
        "labels": Optional[List[PullRequestLabel]],
        "jira": Optional[List[LinkedJIRAIssue]],
    }

    attribute_map = {
        "repository": "repository",
        "number": "number",
        "title": "title",
        "size_added": "size_added",
        "size_removed": "size_removed",
        "files_changed": "files_changed",
        "created": "created",
        "updated": "updated",
        "closed": "closed",
        "comments": "comments",
        "commits": "commits",
        "review_requested": "review_requested",
        "first_review": "first_review",
        "approved": "approved",
        "review_comments": "review_comments",
        "reviews": "reviews",
        "merged": "merged",
        "merged_with_failed_check_runs": "merged_with_failed_check_runs",
        "released": "released",
        "release_url": "release_url",
        "stage_timings": "stage_timings",
        "events_time_machine": "events_time_machine",
        "stages_time_machine": "stages_time_machine",
        "events_now": "events_now",
        "stages_now": "stages_now",
        "participants": "participants",
        "labels": "labels",
        "jira": "jira",
    }

    def __init__(
        self,
        repository: Optional[str] = None,
        number: Optional[int] = None,
        title: Optional[str] = None,
        size_added: Optional[int] = None,
        size_removed: Optional[int] = None,
        files_changed: Optional[int] = None,
        created: Optional[datetime] = None,
        updated: Optional[datetime] = None,
        closed: Optional[datetime] = None,
        comments: Optional[int] = None,
        commits: Optional[int] = None,
        review_requested: Optional[datetime] = None,
        first_review: Optional[datetime] = None,
        approved: Optional[datetime] = None,
        review_comments: Optional[int] = None,
        reviews: Optional[int] = None,
        merged: Optional[datetime] = None,
        merged_with_failed_check_runs: Optional[List[str]] = None,
        released: Optional[datetime] = None,
        release_url: Optional[str] = None,
        stage_timings: Optional[StageTimings] = None,
        events_time_machine: Optional[List[str]] = None,
        stages_time_machine: Optional[List[str]] = None,
        events_now: Optional[List[str]] = None,
        stages_now: Optional[List[str]] = None,
        participants: Optional[List[PullRequestParticipant]] = None,
        labels: Optional[List[PullRequestLabel]] = None,
        jira: Optional[List[LinkedJIRAIssue]] = None,
    ):
        """PullRequest - a model defined in OpenAPI

        :param repository: The repository of this PullRequest.
        :param title: The title of this PullRequest.
        :param size_added: The size_added of this PullRequest.
        :param size_removed: The size_removed of this PullRequest.
        :param files_changed: The files_changed of this PullRequest.
        :param created: The created of this PullRequest.
        :param updated: The updated of this PullRequest.
        :param closed: The closed of this PullRequest.
        :param comments: The comments of this PullRequest.
        :param commits: The commits of this PullRequest.
        :param review_requested: The review_requested of this PullRequest.
        :param first_review: The first_review of this PullRequest.
        :param approved: The approved of this PullRequest.
        :param review_comments: The review_comments of this PullRequest.
        :param reviews: The reviews of this PullRequest.
        :param merged: The merged of this PullRequest.
        :param released: The released of this PullRequest.
        :param release_url: The release URL of this PullRequest.
        :param stage_timings: The stage timings of this PullRequest.
        :param events_time_machine: The events_time_machine of this PullRequest.
        :param stages_time_machine: The stages_time_machine of this PullRequest.
        :param events_now: The events_now of this PullRequest.
        :param stages_now: The stages_now of this PullRequest.
        :param participants: The participants of this PullRequest.
        :param labels: The labels of this PullRequest.
        :param jira: The jira of this PullRequest.
        """
        self._repository = repository
        self._number = number
        self._title = title
        self._size_added = size_added
        self._size_removed = size_removed
        self._files_changed = files_changed
        self._created = created
        self._updated = updated
        self._closed = closed
        self._comments = comments
        self._commits = commits
        self._review_requested = review_requested
        self._first_review = first_review
        self._approved = approved
        self._review_comments = review_comments
        self._reviews = reviews
        self._merged = merged
        self._merged_with_failed_check_runs = merged_with_failed_check_runs
        self._released = released
        self._release_url = release_url
        self._stage_timings = stage_timings
        self._events_time_machine = events_time_machine
        self._stages_time_machine = stages_time_machine
        self._events_now = events_now
        self._stages_now = stages_now
        self._participants = participants
        self._labels = labels
        self._jira = jira

    def __lt__(self, other: "PullRequest") -> bool:
        """Compute self < other."""
        return self.updated < other.updated

    @property
    def repository(self) -> str:
        """Gets the repository of this PullRequest.

        PR is/was open in this repository.

        :return: The repository of this PullRequest.
        :rtype: str
        """
        return self._repository

    @repository.setter
    def repository(self, repository: str):
        """Sets the repository of this PullRequest.

        PR is/was open in this repository.

        :param repository: The repository of this PullRequest.
        """
        if repository is None:
            raise ValueError("Invalid value for `repository`, must not be `None`")

        self._repository = repository

    @property
    def number(self) -> int:
        """Gets the number of this PullRequest.

        PR number.

        :return: The number of this PullRequest.
        """
        return self._number

    @number.setter
    def number(self, number: int):
        """Sets the number of this PullRequest.

        PR number.

        :param number: The number of this PullRequest.
        """
        if number is None:
            raise ValueError("Invalid value for `number`, must not be `None`")

        self._number = number

    @property
    def title(self) -> str:
        """Gets the title of this PullRequest.

        Title of the PR.

        :return: The title of this PullRequest.
        """
        return self._title

    @title.setter
    def title(self, title: str):
        """Sets the title of this PullRequest.

        Title of the PR.

        :param title: The title of this PullRequest.
        """
        if title is None:
            raise ValueError("Invalid value for `title`, must not be `None`")

        self._title = title

    @property
    def size_added(self) -> int:
        """Gets the size_added of this PullRequest.

        Overall number of lines added.

        :return: The size_added of this PullRequest.
        """
        return self._size_added

    @size_added.setter
    def size_added(self, size_added: int):
        """Sets the size_added of this PullRequest.

        Overall number of lines added.

        :param size_added: The size_added of this PullRequest.
        """
        if size_added is None:
            raise ValueError("Invalid value for `size_added`, must not be `None`")

        self._size_added = size_added

    @property
    def size_removed(self) -> int:
        """Gets the size_removed of this PullRequest.

        Overall number of lines removed.

        :return: The size_removed of this PullRequest.
        """
        return self._size_removed

    @size_removed.setter
    def size_removed(self, size_removed: int):
        """Sets the size_removed of this PullRequest.

        Overall number of lines removed.

        :param size_removed: The size_removed of this PullRequest.
        """
        if size_removed is None:
            raise ValueError("Invalid value for `size_removed`, must not be `None`")

        self._size_removed = size_removed

    @property
    def files_changed(self) -> int:
        """Gets the files_changed of this PullRequest.

        Number of files changed in this PR.

        :return: The files_changed of this PullRequest.
        """
        return self._files_changed

    @files_changed.setter
    def files_changed(self, files_changed: int):
        """Sets the files_changed of this PullRequest.

        Number of files changed in this PR.

        :param files_changed: The files_changed of this PullRequest.
        """
        if files_changed is None:
            raise ValueError("Invalid value for `files_changed`, must not be `None`")

        self._files_changed = files_changed

    @property
    def created(self) -> datetime:
        """Gets the created of this PullRequest.

        When this PR was created.

        :return: The created of this PullRequest.
        """
        return self._created

    @created.setter
    def created(self, created: datetime):
        """Sets the created of this PullRequest.

        When this PR was created.

        :param created: The created of this PullRequest.
        """
        if created is None:
            raise ValueError("Invalid value for `created`, must not be `None`")

        self._created = created

    @property
    def updated(self) -> datetime:
        """Gets the updated of this PullRequest.

        When this PR was last updated.

        :return: The updated of this PullRequest.
        """
        return self._updated

    @updated.setter
    def updated(self, updated: datetime):
        """Sets the updated of this PullRequest.

        When this PR was last updated.

        :param updated: The updated of this PullRequest.
        """
        if updated is None:
            raise ValueError("Invalid value for `updated`, must not be `None`")

        self._updated = updated

    @property
    def closed(self) -> Optional[datetime]:
        """Gets the closed of this PullRequest.

        When this PR was closed.

        :return: The closed of this PullRequest.
        """
        return self._closed

    @closed.setter
    def closed(self, closed: Optional[datetime]):
        """Sets the closed of this PullRequest.

        When this PR was closed.

        :param closed: The closed of this PullRequest.
        """
        self._closed = closed

    @property
    def comments(self) -> int:
        """Gets the comments of this PullRequest.

        Number of *regular* (not review) comments in this PR.

        :return: The comments of this PullRequest.
        """
        return self._comments

    @comments.setter
    def comments(self, comments: int):
        """Sets the comments of this PullRequest.

        Number of *regular* (not review) comments in this PR.

        :param comments: The comments of this PullRequest.
        """
        if comments is None:
            raise ValueError("Invalid value for `comments`, must not be `None`")

        self._comments = comments

    @property
    def commits(self) -> int:
        """Gets the commits of this PullRequest.

        Number of commits in this PR.

        :return: The commits of this PullRequest.
        """
        return self._commits

    @commits.setter
    def commits(self, commits: int):
        """Sets the commits of this PullRequest.

        Number of commits in this PR.

        :param commits: The commits of this PullRequest.
        """
        if commits is None:
            raise ValueError("Invalid value for `commits`, must not be `None`")

        self._commits = commits

    @property
    def review_requested(self) -> datetime:
        """Gets the review_requested of this PullRequest.

        When was the first time the author of this PR requested a review.

        :return: The review_requested of this PullRequest.
        """
        return self._review_requested

    @review_requested.setter
    def review_requested(self, review_requested: Optional[datetime]):
        """Sets the review_requested of this PullRequest.

        When was the first time the author of this PR requested a review.

        :param review_requested: The review_requested of this PullRequest.
        """
        self._review_requested = review_requested

    @property
    def first_review(self) -> datetime:
        """Gets the first_review of this PullRequest.

        When the first review of this PR happened.

        :return: The first_review of this PullRequest.
        """
        return self._first_review

    @first_review.setter
    def first_review(self, first_review: Optional[datetime]):
        """Sets the first_review of this PullRequest.

        When the first review of this PR happened.

        :param first_review: The first_review of this PullRequest.
        """
        self._first_review = first_review

    @property
    def approved(self) -> datetime:
        """Gets the approved of this PullRequest.

        When this PR was approved.

        :return: The approved of this PullRequest.
        """
        return self._approved

    @approved.setter
    def approved(self, approved: Optional[datetime]):
        """Sets the approved of this PullRequest.

        When this PR was approved.

        :param approved: The approved of this PullRequest.
        """
        self._approved = approved

    @property
    def review_comments(self) -> int:
        """Gets the review_comments of this PullRequest.

        Number of review comments this PR received. A review comment is left at
        a specific line in a specific file. In other words: review summaries are
        *not* considered review comments; refer to `reviews`. Comments by the PR
        author are considered as `comments`, not as `review_comments`.

        :return: The review_comments of this PullRequest.
        """
        return self._review_comments

    @review_comments.setter
    def review_comments(self, review_comments: Optional[int]):
        """Sets the review_comments of this PullRequest.

        Number of review comments this PR received. A review comment is left at
        a specific line in a specific file. In other words: review summaries are
        *not* considered review comments; refer to `reviews`. Comments by the PR
        author are considered as `comments`, not as `review_comments`.

        :param review_comments: The review_comments of this PullRequest.
        """
        self._review_comments = review_comments

    @property
    def reviews(self) -> int:
        """Gets the reviews of this PullRequest.

        Number of times this PR was reviewed. Reviews by the PR author are not taken into account.

        :return: The reviews of this PullRequest.
        """
        return self._reviews

    @reviews.setter
    def reviews(self, reviews: Optional[int]):
        """Sets the reviews of this PullRequest.

        Number of times this PR was reviewed. Reviews by the PR author are not taken into account.

        :param reviews: The reviews of this PullRequest.
        """
        self._reviews = reviews

    @property
    def merged(self) -> datetime:
        """Gets the merged of this PullRequest.

        When this PR was merged.

        :return: The merged of this PullRequest.
        """
        return self._merged

    @merged.setter
    def merged(self, merged: datetime):
        """Sets the merged of this PullRequest.

        When this PR was merged.

        :param merged: The merged of this PullRequest.
        """
        self._merged = merged

    @property
    def merged_with_failed_check_runs(self) -> Optional[List[str]]:
        """Gets the merged_with_failed_check_runs of this PullRequest.

        PR was merged with these failed check runs.

        :return: The merged_with_failed_check_runs of this PullRequest.
        """
        return self._merged_with_failed_check_runs

    @merged_with_failed_check_runs.setter
    def merged_with_failed_check_runs(self, merged_with_failed_check_runs: Optional[List[str]]):
        """Sets the merged_with_failed_check_runs of this PullRequest.

        PR was merged with these failed check runs.

        :param merged_with_failed_check_runs: The merged_with_failed_check_runs of this \
                                              PullRequest.
        """
        self._merged_with_failed_check_runs = merged_with_failed_check_runs

    @property
    def released(self) -> datetime:
        """Gets the released of this PullRequest.

        When this PR was released.

        :return: The released of this PullRequest.
        """
        return self._released

    @released.setter
    def released(self, released: datetime):
        """Sets the released of this PullRequest.

        When this PR was released.

        :param released: The released of this PullRequest.
        """
        self._released = released

    @property
    def release_url(self) -> str:
        """Gets the release_url of this PullRequest.

        URL of the earliest release that includes this merged PR.

        :return: The release_url of this PullRequest.
        """
        return self._release_url

    @release_url.setter
    def release_url(self, release_url: str):
        """Sets the release_url of this PullRequest.

        URL of the earliest release that includes this merged PR.

        :param release_url: The release_url of this PullRequest.
        """
        self._release_url = release_url

    @property
    def stage_timings(self) -> StageTimings:
        """
        Gets the modelled pipeline stage timings of this PullRequest.

        :return: The stage timings of this PullRequest.
        """
        return self._stage_timings

    @stage_timings.setter
    def stage_timings(self, stage_timings: StageTimings):
        """
        Sets the modelled pipeline stage timings of this PullRequest.

        :param stage_timings: The stage timings of this PullRequest.
        """
        if stage_timings is None:
            raise ValueError("Invalid value for `stage_timings`, must not be `None`")

        self._stage_timings = stage_timings

    @property
    def events_time_machine(self) -> Optional[List[str]]:
        """
        Gets events_time_machine of this PullRequest.

        List of PR events which happened until `date_to`. `date_from` does not matter.

        :return: The events_time_machine of this PullRequest.
        """
        return self._events_time_machine

    @events_time_machine.setter
    def events_time_machine(self, events_time_machine: Optional[List[str]]):
        """
        Sets events_time_machine of this PullRequest.

        List of PR events which happened until `date_to`. `date_from` does not matter.

        :param events_time_machine: The events_time_machine of this PullRequest.
        """
        for prop in (events_time_machine or []):
            if prop not in PullRequestEvent:
                raise ValueError("Invalid `events_time_machine`: %s" % events_time_machine)

        self._events_time_machine = events_time_machine

    @property
    def stages_time_machine(self) -> Optional[List[str]]:
        """Gets the stages_time_machine of this PullRequest.

        :return: The stages_time_machine of this PullRequest.
        """
        return self._stages_time_machine

    @stages_time_machine.setter
    def stages_time_machine(self, stages_time_machine: Optional[List[str]]):
        """Sets the stages_time_machine of this PullRequest.

        :param stages_time_machine: The stages_time_machine of this PullRequest.
        """
        for prop in (stages_time_machine or []):
            if prop not in PullRequestStage:
                raise ValueError("Invalid `stages_time_machine`: %s" % stages_time_machine)

        self._stages_time_machine = stages_time_machine

    @property
    def events_now(self) -> List[str]:
        """
        Gets events_now of this PullRequest.

        List of PR events that ever happened.

        :return: The events_now of this PullRequest.
        """
        return self._events_now

    @events_now.setter
    def events_now(self, events_now: List[str]):
        """
        Sets events_now of this PullRequest.

        List of PR events that ever happened.

        :param events_now: The events_now of this PullRequest.
        """
        if events_now is None:
            raise ValueError("`events_now` must not be null")
        for prop in events_now:
            if prop not in PullRequestEvent:
                raise ValueError("Invalid `events_now`: %s" % events_now)

        self._events_now = events_now

    @property
    def stages_now(self) -> List[str]:
        """Gets the stages_now of this PullRequest.

        :return: The stages_now of this PullRequest.
        """
        return self._stages_now

    @stages_now.setter
    def stages_now(self, stages_now: List[str]):
        """Sets the stages_now of this PullRequest.

        :param stages_now: The stages_now of this PullRequest.
        """
        if stages_now is None:
            raise ValueError("`stages_now` must not be null")
        for prop in stages_now:
            if prop not in PullRequestStage:
                raise ValueError("Invalid `stages_now`: %s" % stages_now)

        self._stages_now = stages_now

    @property
    def participants(self) -> List[PullRequestParticipant]:
        """Gets the participants of this PullRequest.

        List of developers related to this PR.

        :return: The participants of this PullRequest.
        """
        return self._participants

    @participants.setter
    def participants(self, participants: List[PullRequestParticipant]):
        """Sets the participants of this PullRequest.

        List of developers related to this PR.

        :param participants: The participants of this PullRequest.
        """
        if participants is None:
            raise ValueError("Invalid value for `participants`, must not be `None`")

        self._participants = participants

    @property
    def labels(self) -> Optional[List[PullRequestLabel]]:
        """Gets the labels of this PullRequest.

        List of developers related to this PR.

        :return: The labels of this PullRequest.
        """
        return self._labels

    @labels.setter
    def labels(self, labels: Optional[List[PullRequestLabel]]):
        """Sets the labels of this PullRequest.

        List of developers related to this PR.

        :param labels: The labels of this PullRequest.
        """
        self._labels = labels

    @property
    def jira(self) -> Optional[List[LinkedJIRAIssue]]:
        """Gets the jira of this PullRequest.

        List of JIRA issues linked to this PR.

        :return: The jira of this PullRequest.
        """
        return self._jira

    @jira.setter
    def jira(self, jira: Optional[List[LinkedJIRAIssue]]):
        """Sets the jira of this PullRequest.

        List of JIRA issues linked to this PR.

        :param jira: The jira of this PullRequest.
        """
        self._jira = jira
