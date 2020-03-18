from datetime import datetime
from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.pull_request_participant import PullRequestParticipant
from athenian.api.models.web.pull_request_pipeline_stage import PullRequestPipelineStage
from athenian.api.models.web.pull_request_property import PullRequestProperty


class PullRequest(Model):
    """Details of a pull request."""

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
        review_requested: Optional[bool] = None,
        review_comments: Optional[int] = None,
        merged: Optional[datetime] = None,
        released: Optional[datetime] = None,
        release_url: Optional[str] = None,
        stage: Optional[str] = None,
        properties: Optional[List[str]] = None,
        participants: Optional[List[PullRequestParticipant]] = None,
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
        :param review_comments: The review_comments of this PullRequest.
        :param merged: The merged of this PullRequest.
        :param released: The released of this PullRequest.
        :param release_url: The release URL of this PullRequest.
        :param stage: The stage of this PullRequest.
        :param properties: The properties of this PullRequest.
        :param participants: The participants of this PullRequest.
        """
        self.openapi_types = {
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
            "review_requested": bool,
            "review_comments": int,
            "merged": datetime,
            "released": datetime,
            "release_url": str,
            "stage": str,
            "properties": List[str],
            "participants": List[PullRequestParticipant],
        }

        self.attribute_map = {
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
            "review_comments": "review_comments",
            "merged": "merged",
            "released": "released",
            "release_url": "release_url",
            "stage": "stage",
            "properties": "properties",
            "participants": "participants",
        }

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
        self._review_comments = review_comments
        self._merged = merged
        self._released = released
        self._release_url = release_url
        self._stage = stage
        self._properties = properties
        self._participants = participants

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

        :param commits: The updated of this PullRequest.
        """
        if commits is None:
            raise ValueError("Invalid value for `commits`, must not be `None`")

        self._commits = commits

    @property
    def review_requested(self) -> bool:
        """Gets the review_requested of this PullRequest.

        Value indicating whether the author of this PR requested a review.

        :return: The review_requested of this PullRequest.
        """
        return self._review_requested

    @review_requested.setter
    def review_requested(self, review_requested: bool):
        """Sets the review_requested of this PullRequest.

        Value indicating whether the author of this PR requested a review.

        :param review_requested: The updated of this PullRequest.
        """
        if review_requested is None:
            raise ValueError("Invalid value for `review_requested`, must not be `None`")

        self._review_requested = review_requested

    @property
    def review_comments(self) -> int:
        """Gets the review_comments of this PullRequest.

        Value indicating whether this PR received at least one review.

        :return: The review_comments of this PullRequest.
        """
        return self._review_comments

    @review_comments.setter
    def review_comments(self, review_comments: int):
        """Sets the review_comments of this PullRequest.

        Value indicating whether this PR received at least one review.

        :param review_comments: The updated of this PullRequest.
        """
        if review_comments is None:
            raise ValueError("Invalid value for `review_comments`, must not be `None`")

        self._review_comments = review_comments

    @property
    def merged(self) -> datetime:
        """Gets the merged of this PullRequest.

        Value indicating whether this PR was merged.

        :return: The merged of this PullRequest.
        """
        return self._merged

    @merged.setter
    def merged(self, merged: datetime):
        """Sets the merged of this PullRequest.

        Value indicating whether this PR was merged.

        :param merged: The updated of this PullRequest.
        """
        self._merged = merged

    @property
    def released(self) -> datetime:
        """Gets the released of this PullRequest.

        Value indicating whether this PR was released.

        :return: The released of this PullRequest.
        """
        return self._released

    @released.setter
    def released(self, released: datetime):
        """Sets the released of this PullRequest.

        Value indicating whether this PR was released.

        :param released: The updated of this PullRequest.
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

        :param release_url: The updated of this PullRequest.
        """
        self._release_url = release_url

    @property
    def stage(self) -> str:
        """
        Gets the modelled pipeline stage of this PullRequest.

        :return: The stage of this PullRequest.
        """
        return self._stage

    @stage.setter
    def stage(self, stage: str):
        """
        Sets the modelled pipeline stage of this PullRequest.

        :param stage: The stage of this PullRequest.
        """
        if stage not in PullRequestPipelineStage.ALL:
            raise ValueError("Invalid stage: %s" % stage)

        self._stage = stage

    @property
    def properties(self) -> List[str]:
        """
        Gets the modelled pipeline properties of this PullRequest.

        :return: The properties of this PullRequest.
        """
        return self._properties

    @properties.setter
    def properties(self, properties: List[str]):
        """
        Sets the modelled pipeline properties of this PullRequest.

        :param properties: The properties of this PullRequest.
        """
        for prop in properties:
            if prop not in PullRequestProperty.ALL:
                raise ValueError("Invalid properties: %s" % properties)

        self._properties = properties

    @property
    def participants(self) -> List[PullRequestParticipant]:
        """Gets the participants of this PullRequest.

        List of developers related to this PR, always including the author.

        :return: The participants of this PullRequest.
        """
        return self._participants

    @participants.setter
    def participants(self, participants: List[PullRequestParticipant]):
        """Sets the participants of this PullRequest.

        List of developers related to this PR, always including the author.

        :param participants: The participants of this PullRequest.
        """
        if participants is None:
            raise ValueError("Invalid value for `participants`, must not be `None`")

        self._participants = participants
