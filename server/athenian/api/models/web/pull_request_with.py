from typing import List, Optional

from athenian.api import serialization
from athenian.api.models.web.base_model_ import Model


class PullRequestWith(Model):
    """Triage PRs by various developer participation."""

    def __init__(
        self,
        author: Optional[List[str]] = None,
        reviewer: Optional[List[str]] = None,
        commit_author: Optional[List[str]] = None,
        commit_committer: Optional[List[str]] = None,
        commenter: Optional[List[str]] = None,
        merger: Optional[List[str]] = None,
    ):
        """PullRequestWith - a model defined in OpenAPI

        :param author: The author of this PullRequestWith.
        :param reviewer: The reviewer of this PullRequestWith.
        :param commit_author: The commit_author of this PullRequestWith.
        :param commit_committer: The commit_committer of this PullRequestWith.
        :param commenter: The commenter of this PullRequestWith.
        :param merger: The merger of this PullRequestWith.
        """
        self.openapi_types = {
            "author": List[str],
            "reviewer": List[str],
            "commit_author": List[str],
            "commit_committer": List[str],
            "commenter": List[str],
            "merger": List[str],
        }

        self.attribute_map = {
            "author": "author",
            "reviewer": "reviewer",
            "commit_author": "commit-author",
            "commit_committer": "commit-committer",
            "commenter": "commenter",
            "merger": "merger",
        }

        self._author = author
        self._reviewer = reviewer
        self._commit_author = commit_author
        self._commit_committer = commit_committer
        self._commenter = commenter
        self._merger = merger

    @classmethod
    def from_dict(cls, dikt: dict) -> "PullRequestWith":
        """Returns the dict as a model

        :param dikt: A dict.
        :return: The PullRequestWith of this PullRequestWith.
        """
        return serialization.deserialize_model(dikt, cls)

    @property
    def author(self) -> Optional[List[str]]:
        """Gets the author of this PullRequestWith.

        :return: The author of this PullRequestWith.
        """
        return self._author

    @author.setter
    def author(self, author: Optional[List[str]]):
        """Sets the author of this PullRequestWith.

        :param author: The author of this PullRequestWith.
        """
        self._author = author

    @property
    def reviewer(self) -> Optional[List[str]]:
        """Gets the reviewer of this PullRequestWith.

        :return: The reviewer of this PullRequestWith.
        """
        return self._reviewer

    @reviewer.setter
    def reviewer(self, reviewer: Optional[List[str]]):
        """Sets the reviewer of this PullRequestWith.

        :param reviewer: The reviewer of this PullRequestWith.
        """
        self._reviewer = reviewer

    @property
    def commit_author(self) -> Optional[List[str]]:
        """Gets the commit_author of this PullRequestWith.

        :return: The commit_author of this PullRequestWith.
        """
        return self._commit_author

    @commit_author.setter
    def commit_author(self, commit_author: Optional[List[str]]):
        """Sets the commit_author of this PullRequestWith.

        :param commit_author: The commit_author of this PullRequestWith.
        """
        self._commit_author = commit_author

    @property
    def commit_committer(self) -> Optional[List[str]]:
        """Gets the commit_committer of this PullRequestWith.

        :return: The commit_committer of this PullRequestWith.
        """
        return self._commit_committer

    @commit_committer.setter
    def commit_committer(self, commit_committer: Optional[List[str]]):
        """Sets the commit_committer of this PullRequestWith.

        :param commit_committer: The commit_committer of this PullRequestWith.
        """
        self._commit_committer = commit_committer

    @property
    def commenter(self) -> Optional[List[str]]:
        """Gets the commenter of this PullRequestWith.

        :return: The commenter of this PullRequestWith.
        """
        return self._commenter

    @commenter.setter
    def commenter(self, commenter: Optional[List[str]]):
        """Sets the commenter of this PullRequestWith.

        :param commenter: The commenter of this PullRequestWith.
        """
        self._commenter = commenter

    @property
    def merger(self) -> Optional[List[str]]:
        """Gets the merger of this PullRequestWith.

        :return: The merger of this PullRequestWith.
        """
        return self._merger

    @merger.setter
    def merger(self, merger: Optional[List[str]]):
        """Sets the merger of this PullRequestWith.

        :param merger: The merger of this PullRequestWith.
        """
        self._merger = merger
