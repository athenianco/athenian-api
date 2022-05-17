from datetime import datetime
from typing import Optional

from athenian.api.models.web.base_model_ import Model


class JIRALabel(Model):
    """Details about a JIRA label."""

    attribute_types = {
        "title": str,
        "last_used": datetime,
        "issues_count": int,
        "kind": str,
    }

    attribute_map = {
        "title": "title",
        "last_used": "last_used",
        "issues_count": "issues_count",
        "kind": "kind",
    }

    def __init__(
        self,
        title: Optional[str] = None,
        last_used: Optional[datetime] = None,
        issues_count: Optional[int] = None,
        kind: Optional[str] = None,
    ):
        """JIRALabel - a model defined in OpenAPI

        :param title: The title of this JIRALabel.
        :param last_used: The last_used of this JIRALabel.
        :param issues_count: The issues_count of this JIRALabel.
        :param kind: The kind of this JIRALabel.
        """
        self._title = title
        self._last_used = last_used
        self._issues_count = issues_count
        self._kind = kind

    def __lt__(self, other: "JIRALabel") -> bool:
        """Support sorting."""
        return self._title < other._title

    @property
    def title(self) -> str:
        """Gets the title of this JIRALabel.

        :return: The title of this JIRALabel.
        """
        return self._title

    @title.setter
    def title(self, title: str):
        """Sets the title of this JIRALabel.

        :param title: The title of this JIRALabel.
        """
        if title is None:
            raise ValueError("Invalid value for `title`, must not be `None`")

        self._title = title

    @property
    def last_used(self) -> datetime:
        """Gets the last_used of this JIRALabel.

        When this label was last assigned to an issue.

        :return: The last_used of this JIRALabel.
        """
        return self._last_used

    @last_used.setter
    def last_used(self, last_used: datetime):
        """Sets the last_used of this JIRALabel.

        When this label was last assigned to an issue.

        :param last_used: The last_used of this JIRALabel.
        """
        if last_used is None:
            raise ValueError("Invalid value for `last_used`, must not be `None`")

        self._last_used = last_used

    @property
    def issues_count(self) -> int:
        """Gets the issues_count of this JIRALabel.

        In how many issues (in the specified time interval) this label was used.

        :return: The issues_count of this JIRALabel.
        """
        return self._issues_count

    @issues_count.setter
    def issues_count(self, issues_count: int):
        """Sets the issues_count of this JIRALabel.

        In how many issues (in the specified time interval) this label was used.

        :param issues_count: The issues_count of this JIRALabel.
        """
        if issues_count is None:
            raise ValueError("Invalid value for `issues_count`, must not be `None`")

        self._issues_count = issues_count

    @property
    def kind(self) -> str:
        """Gets the kind of this JIRALabel.

        Label kind - \"Label\", \"Component\", etc.

        :return: The kind of this JIRALabel.
        """
        return self._kind

    @kind.setter
    def kind(self, kind: str):
        """Sets the kind of this JIRALabel.

        Label kind - \"Label\", \"Component\", etc.

        :param kind: The kind of this JIRALabel.
        """
        self._kind = kind
