from typing import Optional

from athenian.api.models.web.base_model_ import Model


class DeveloperUpdates(Model):
    """
    Various developer contributions statistics over the specified time period.

    Note: any of these properties may be missing if there was no such activity.
    """

    attribute_types = {
        "prs": int,
        "reviewer": int,
        "commit_author": int,
        "commit_committer": int,
        "commenter": int,
        "releaser": int,
    }

    attribute_map = {
        "prs": "prs",
        "reviewer": "reviewer",
        "commit_author": "commit_author",
        "commit_committer": "commit_committer",
        "commenter": "commenter",
        "releaser": "releaser",
    }

    def __init__(
        self,
        prs: Optional[int] = None,
        reviewer: Optional[int] = None,
        commit_author: Optional[int] = None,
        commit_committer: Optional[int] = None,
        commenter: Optional[int] = None,
        releaser: Optional[int] = None,
    ):
        """DeveloperUpdates - a model defined in OpenAPI

        :param prs: The prs of this DeveloperUpdates.
        :param reviewer: The reviewer of this DeveloperUpdates.
        :param commit_author: The commit_author of this DeveloperUpdates.
        :param commit_committer: The commit_committer of this DeveloperUpdates.
        :param commenter: The commenter of this DeveloperUpdates.
        :param merger: The merger of this DeveloperUpdates.
        :param releaser: The releaser of this DeveloperUpdates.
        """
        self._prs = prs
        self._reviewer = reviewer
        self._commit_author = commit_author
        self._commit_committer = commit_committer
        self._commenter = commenter
        self._releaser = releaser

    @property
    def prs(self) -> int:
        """Gets the prs of this DeveloperUpdates.

        How many PRs authored by this developer updated. Note: this is not the same as the number \
        of PRs this developer opened. Note: the update origin is not necessarily this developer.

        :return: The prs of this DeveloperUpdates.
        """
        return self._prs

    @prs.setter
    def prs(self, prs: int):
        """Sets the prs of this DeveloperUpdates.

        How many PRs authored by this developer updated. Note: this is not the same as the number \
        of PRs this developer opened. Note: the update origin is not necessarily this developer.

        :param prs: The prs of this DeveloperUpdates.
        """
        self._prs = prs

    @property
    def reviewer(self) -> int:
        """Gets the reviewer of this DeveloperUpdates.

        How many reviews this developer submitted. Note: this is not the same as the number of
        unique PRs reviewed.

        :return: The reviewer of this DeveloperUpdates.
        """
        return self._reviewer

    @reviewer.setter
    def reviewer(self, reviewer: int):
        """Sets the reviewer of this DeveloperUpdates.

        How many reviews this developer submitted. Note: this is not the same as the number of
        unique PRs reviewed.

        :param reviewer: The reviewer of this DeveloperUpdates.
        """
        self._reviewer = reviewer

    @property
    def commit_author(self) -> int:
        """Gets the commit_author of this DeveloperUpdates.

        How many commits this developer prsed.

        :return: The commit_author of this DeveloperUpdates.
        """
        return self._commit_author

    @commit_author.setter
    def commit_author(self, commit_author: int):
        """Sets the commit_author of this DeveloperUpdates.

        How many commits this developer prsed.

        :param commit_author: The commit_author of this DeveloperUpdates.
        """
        self._commit_author = commit_author

    @property
    def commit_committer(self) -> int:
        """Gets the commit_committer of this DeveloperUpdates.

        How many commits this developer pushed.

        :return: The commit_committer of this DeveloperUpdates.
        """
        return self._commit_committer

    @commit_committer.setter
    def commit_committer(self, commit_committer: int):
        """Sets the commit_committer of this DeveloperUpdates.

        How many commits this developer pushed.

        :param commit_committer: The commit_committer of this DeveloperUpdates.
        """
        self._commit_committer = commit_committer

    @property
    def commenter(self) -> int:
        """Gets the commenter of this DeveloperUpdates.

        How many regular PR comments this developer left. Note: issues are not taken into account,
        only the PRs.

        :return: The commenter of this DeveloperUpdates.
        """
        return self._commenter

    @commenter.setter
    def commenter(self, commenter: int):
        """Sets the commenter of this DeveloperUpdates.

        How many regular PR comments this developer left. Note: issues are not taken into account,
        only the PRs.

        :param commenter: The commenter of this DeveloperUpdates.
        """
        self._commenter = commenter

    @property
    def releaser(self) -> int:
        """Gets the releaser of this DeveloperUpdates.

        How many releases this developer created.

        :return: The releaser of this DeveloperUpdates.
        """
        return self._releaser

    @releaser.setter
    def releaser(self, releaser: int):
        """Sets the releaser of this DeveloperUpdates.

        How many releases this developer created.

        :param releaser: The releaser of this DeveloperUpdates.
        """
        self._releaser = releaser
