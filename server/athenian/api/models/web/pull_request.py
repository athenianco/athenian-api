from datetime import datetime
from typing import List, Optional

from athenian.api import serialization
from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.pull_request_participant import PullRequestParticipant
from athenian.api.models.web.pull_request_pipeline_stage import PullRequestPipelineStage


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
        stage: Optional[str] = None,
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
        :param stage: The stage of this PullRequest.
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
            "stage": str,
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
            "stage": "stage",
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
        self._stage = stage
        self._participants = participants

    @classmethod
    def from_dict(cls, dikt: dict) -> "PullRequest":
        """Returns the dict as a model

        :param dikt: A dict.
        :return: The PullRequest of this PullRequest.
        """
        return serialization.deserialize_model(dikt, cls)

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
