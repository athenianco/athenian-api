from typing import List, Optional

from athenian.api.models.web.base_model_ import Model


class PullRequestParticipant(Model):
    """Developer and their role in the PR."""

    STATUS_AUTHOR = "author"
    STATUS_REVIEWER = "reviewer"
    STATUS_COMMIT_AUTHOR = "commit_author"
    STATUS_COMMIT_COMMITTER = "commit_committer"
    STATUS_COMMENTER = "commenter"
    STATUS_MERGER = "merger"
    STATUS_RELEASER = "releaser"
    STATUSES = {
        STATUS_AUTHOR,
        STATUS_REVIEWER,
        STATUS_COMMIT_AUTHOR,
        STATUS_COMMIT_COMMITTER,
        STATUS_COMMENTER,
        STATUS_MERGER,
        STATUS_RELEASER,
    }
    attribute_types = {"id": str, "status": List[str]}
    attribute_map = {"id": "id", "status": "status"}

    def __init__(self, id: Optional[str] = None, status: Optional[List[str]] = None):
        """PullRequestParticipant - a model defined in OpenAPI

        :param id: The id of this PullRequestParticipant.
        :param status: The status of this PullRequestParticipant.
        """
        self._id = id
        self._status = status

    def __lt__(self, other: "PullRequestParticipant") -> bool:
        """Compute self < other."""
        return self.id < other.id

    @property
    def id(self) -> str:
        """Gets the id of this PullRequestParticipant.

        Person identifier.

        :return: The id of this PullRequestParticipant.
        """
        return self._id

    @id.setter
    def id(self, id: str):
        """Sets the id of this PullRequestParticipant.

        Person identifier.

        :param id: The id of this PullRequestParticipant.
        """
        if id is None:
            raise ValueError("Invalid value for `id`, must not be `None`")

        self._id = id

    @property
    def status(self) -> List[str]:
        """Gets the status of this PullRequestParticipant.

        :return: The status of this PullRequestParticipant.
        """
        return self._status

    @status.setter
    def status(self, status: List[str]):
        """Sets the status of this PullRequestParticipant.

        :param status: The status of this PullRequestParticipant.
        """
        for v in status:
            if v not in self.STATUSES:
                raise ValueError(
                    "Invalid value for `status` (%s), must be one of %s" % (v, self.STATUSES)
                )

        self._status = status
