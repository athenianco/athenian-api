from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.release_match_strategy import ReleaseMatchStrategy


class ReleaseMatchRequest(Model):
    """Release matching rule setting."""

    openapi_types = {
        "account": int,
        "repositories": List[str],
        "branches": str,
        "tags": str,
        "match": str,
    }

    attribute_map = {
        "account": "account",
        "repositories": "repositories",
        "branches": "branches",
        "tags": "tags",
        "match": "match",
    }

    def __init__(
        self,
        account: Optional[int] = None,
        repositories: Optional[List[str]] = None,
        branches: Optional[str] = None,
        tags: Optional[str] = None,
        match: Optional[str] = None,
    ):
        """ReleaseMatchRequest - a model defined in OpenAPI

        :param account: The account of this ReleaseMatchRequest.
        :param repositories: The repositories of this ReleaseMatchRequest.
        :param branches: The branches of this ReleaseMatchRequest.
        :param tags: The tags of this ReleaseMatchRequest.
        :param match: The match of this ReleaseMatchRequest.
        """
        self._account = account
        self._repositories = repositories
        self._branches = branches
        self._tags = tags
        self._match = match

    @property
    def account(self) -> int:
        """Gets the account of this ReleaseMatchRequest.

        :return: The account of this ReleaseMatchRequest.
        """
        return self._account

    @account.setter
    def account(self, account: int):
        """Sets the account of this ReleaseMatchRequest.

        :param account: The account of this ReleaseMatchRequest.
        """
        if account is None:
            raise ValueError("Invalid value for `account`, must not be `None`")

        self._account = account

    @property
    def repositories(self) -> List[str]:
        """Gets the repositories of this ReleaseMatchRequest.

        A set of repositories. An empty list results an empty response in contrary to DeveloperSet.
        Duplicates are automatically ignored.

        :return: The repositories of this ReleaseMatchRequest.
        """
        return self._repositories

    @repositories.setter
    def repositories(self, repositories: List[str]):
        """Sets the repositories of this ReleaseMatchRequest.

        A set of repositories. An empty list results an empty response in contrary to DeveloperSet.
        Duplicates are automatically ignored.

        :param repositories: The repositories of this ReleaseMatchRequest.
        """
        if repositories is None:
            raise ValueError("Invalid value for `repositories`, must not be `None`")

        self._repositories = repositories

    @property
    def branches(self) -> str:
        """Gets the branches of this ReleaseMatchRequest.

        Regular expression to match branch names.
        Reference: https://www.postgresql.org/docs/current/functions-matching.html#FUNCTIONS-SIMILARTO-REGEXP

        :return: The branches of this ReleaseMatchRequest.
        """  # noqa
        return self._branches

    @branches.setter
    def branches(self, branches: str):
        """Sets the branches of this ReleaseMatchRequest.

        Regular expression to match branch names.
        Reference: https://www.postgresql.org/docs/current/functions-matching.html#FUNCTIONS-SIMILARTO-REGEXP

        :param branches: The branches of this ReleaseMatchRequest.
        """  # noqa
        self._branches = branches

    @property
    def tags(self) -> str:
        """Gets the tags of this ReleaseMatchRequest.

        Regular expression to match tag names.
        Reference: https://www.postgresql.org/docs/current/functions-matching.html#FUNCTIONS-SIMILARTO-REGEXP

        :return: The tags of this ReleaseMatchRequest.
        """  # noqa
        return self._tags

    @tags.setter
    def tags(self, tags: str):
        """Sets the tags of this ReleaseMatchRequest.

        Regular expression to match tag names.
        Reference: https://www.postgresql.org/docs/current/functions-matching.html#FUNCTIONS-SIMILARTO-REGEXP

        :param tags: The tags of this ReleaseMatchRequest.
        """  # noqa
        self._tags = tags

    @property
    def match(self) -> str:
        """Gets the match of this ReleaseMatchRequest.

        Workflow choice: consider certain branch merges or tags as releases.

        :return: The match of this ReleaseMatchRequest.
        """
        return self._match

    @match.setter
    def match(self, match: str):
        """Sets the match of this ReleaseMatchRequest.

        Workflow choice: consider certain branch merges or tags as releases.

        :param match: The match of this ReleaseMatchRequest.
        """
        if match not in ReleaseMatchStrategy:
            raise ValueError(
                "Invalid value for `match` (%s), must be one of %s" %
                (match, list(ReleaseMatchStrategy)))

        self._match = match
