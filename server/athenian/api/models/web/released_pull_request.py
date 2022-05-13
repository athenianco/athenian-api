from typing import List, Optional

from athenian.api.models.web.base_model_ import Model


class ReleasedPullRequest(Model):
    """Details about a pull request listed in `/filter/releases`."""

    attribute_types = {
        "number": int,
        "title": str,
        "additions": int,
        "deletions": int,
        "author": str,
        "jira": Optional[List[str]],
    }

    attribute_map = {
        "number": "number",
        "title": "title",
        "additions": "additions",
        "deletions": "deletions",
        "author": "author",
        "jira": "jira",
    }

    def __init__(
        self,
        number: Optional[int] = None,
        title: Optional[str] = None,
        additions: Optional[int] = None,
        deletions: Optional[int] = None,
        author: Optional[str] = None,
        jira: List[str] = None,
    ):
        """ReleasedPullRequest - a model defined in OpenAPI

        :param number: The number of this ReleasedPullRequest.
        :param title: The title of this ReleasedPullRequest.
        :param additions: The additions of this ReleasedPullRequest.
        :param deletions: The deletions of this ReleasedPullRequest.
        :param author: The author of this ReleasedPullRequest.
        :param jira: The jira of this ReleasedPullRequest.
        """
        self._number = number
        self._title = title
        self._additions = additions
        self._deletions = deletions
        self._author = author
        self._jira = jira

    @property
    def number(self) -> int:
        """Gets the number of this ReleasedPullRequest.

        :return: The number of this ReleasedPullRequest.
        """
        return self._number

    @number.setter
    def number(self, number: int):
        """Sets the number of this ReleasedPullRequest.

        :param number: The number of this ReleasedPullRequest.
        """
        if number is None:
            raise ValueError("Invalid value for `number`, must not be `None`")

        self._number = number

    @property
    def title(self) -> str:
        """Gets the title of this ReleasedPullRequest.

        :return: The title of this ReleasedPullRequest.
        """
        return self._title

    @title.setter
    def title(self, title: str):
        """Sets the title of this ReleasedPullRequest.

        :param title: The title of this ReleasedPullRequest.
        """
        if title is None:
            raise ValueError("Invalid value for `title`, must not be `None`")

        self._title = title

    @property
    def additions(self) -> int:
        """Gets the additions of this ReleasedPullRequest.

        :return: The additions of this ReleasedPullRequest.
        """
        return self._additions

    @additions.setter
    def additions(self, additions: int):
        """Sets the additions of this ReleasedPullRequest.

        :param additions: The additions of this ReleasedPullRequest.
        """
        if additions is None:
            raise ValueError("Invalid value for `additions`, must not be `None`")

        self._additions = additions

    @property
    def deletions(self) -> int:
        """Gets the deletions of this ReleasedPullRequest.

        :return: The deletions of this ReleasedPullRequest.
        """
        return self._deletions

    @deletions.setter
    def deletions(self, deletions: int):
        """Sets the deletions of this ReleasedPullRequest.

        :param deletions: The deletions of this ReleasedPullRequest.
        """
        if deletions is None:
            raise ValueError("Invalid value for `deletions`, must not be `None`")

        self._deletions = deletions

    @property
    def author(self) -> Optional[str]:
        """Gets the author of this ReleasedPullRequest.

        User name which uniquely identifies any developer on any service provider.
        The format matches the profile URL without the protocol part.

        :return: The author of this ReleasedPullRequest.
        """
        return self._author

    @author.setter
    def author(self, author: Optional[str]):
        """Sets the author of this ReleasedPullRequest.

        User name which uniquely identifies any developer on any service provider.
        The format matches the profile URL without the protocol part.

        :param author: The author of this ReleasedPullRequest.
        """
        self._author = author

    @property
    def jira(self) -> Optional[List[str]]:
        """Gets the jira of this ReleasedPullRequest.

        Mapped JIRA issue IDs.

        :return: The jira of this ReleasedPullRequest.
        """
        return self._jira

    @jira.setter
    def jira(self, jira: Optional[List[str]]):
        """Sets the jira of this ReleasedPullRequest.

        Mapped JIRA issue IDs.

        :param jira: The jira of this ReleasedPullRequest.
        """
        self._jira = jira
