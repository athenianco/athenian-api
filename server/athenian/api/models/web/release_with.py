from typing import List, Optional

from athenian.api.models.web.base_model_ import MappingModel


class ReleaseWith(MappingModel):
    """Release contribution roles."""

    attribute_types = {
        "pr_author": Optional[List[str]],
        "commit_author": Optional[List[str]],
        "releaser": Optional[List[str]],
    }

    attribute_map = {
        "pr_author": "pr_author",
        "commit_author": "commit_author",
        "releaser": "releaser",
    }

    def __init__(
        self,
        pr_author: Optional[List[str]] = None,
        commit_author: Optional[List[str]] = None,
        releaser: Optional[List[str]] = None,
    ):
        """ReleaseWith - a model defined in OpenAPI

        :param pr_author: The pr_author of this ReleaseWith.
        :param commit_author: The commit_author of this ReleaseWith.
        :param releaser: The releaser of this ReleaseWith.
        """
        self._pr_author = pr_author
        self._commit_author = commit_author
        self._releaser = releaser

    @property
    def pr_author(self) -> Optional[List[str]]:
        """Gets the pr_author of this ReleaseWith.

        Authors of released pull requests.

        :return: The pr_author of this ReleaseWith.
        """
        return self._pr_author

    @pr_author.setter
    def pr_author(self, pr_author: Optional[List[str]]):
        """Sets the pr_author of this ReleaseWith.

        Authors of released pull requests.

        :param pr_author: The pr_author of this ReleaseWith.
        """
        self._pr_author = pr_author

    @property
    def commit_author(self) -> Optional[List[str]]:
        """Gets the commit_author of this ReleaseWith.

        Authors of released commits.

        :return: The commit_author of this ReleaseWith.
        """
        return self._commit_author

    @commit_author.setter
    def commit_author(self, commit_author: Optional[List[str]]):
        """Sets the commit_author of this ReleaseWith.

        Authors of released commits.

        :param commit_author: The commit_author of this ReleaseWith.
        """
        self._commit_author = commit_author

    @property
    def releaser(self) -> Optional[List[str]]:
        """Gets the releaser of this ReleaseWith.

        Release publishers.

        :return: The releaser of this ReleaseWith.
        """
        return self._releaser

    @releaser.setter
    def releaser(self, releaser: Optional[List[str]]):
        """Sets the releaser of this ReleaseWith.

        Release publishers.

        :param releaser: The releaser of this ReleaseWith.
        """
        self._releaser = releaser
