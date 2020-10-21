from typing import Dict, List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.jira_epic import JIRAEpic
from athenian.api.models.web.jira_label import JIRALabel
from athenian.api.models.web.jira_priority import JIRAPriority


class FoundJIRAStuff(Model):
    """Response from `/filter/jira`: found JIRA epics, labels, issue types, priorities, and \
    mentioned users."""

    openapi_types = {
        "epics": List[JIRAEpic],
        "labels": List[JIRALabel],
        "issue_types": List[str],
        "priorities": List[JIRAPriority],
        "users": Dict[str, str],
    }
    attribute_map = {
        "epics": "epics",
        "labels": "labels",
        "issue_types": "issue_types",
        "priorities": "priorities",
        "users": "users",
    }

    def __init__(self,
                 epics: Optional[List[JIRAEpic]] = None,
                 labels: Optional[List[JIRALabel]] = None,
                 issue_types: Optional[List[str]] = None,
                 priorities: Optional[List[JIRAPriority]] = None,
                 users: Optional[Dict[str, str]] = None):
        """FoundJIRAStuff - a model defined in OpenAPI

        :param epics: The epics of this FoundJIRAStuff.
        :param labels: The labels of this FoundJIRAStuff.
        :param issue_types: The issue_types of this FoundJIRAStuff.
        :param priorities: The priorities of this FoundJIRAStuff.
        :param users: The users of this FoundJIRAStuff.
        """
        self._epics = epics
        self._labels = labels
        self._issue_types = issue_types
        self._priorities = priorities
        self._users = users

    @property
    def epics(self) -> List[JIRAEpic]:
        """Gets the epics of this FoundJIRAStuff.

        :return: The epics of this FoundJIRAStuff.
        """
        return self._epics

    @epics.setter
    def epics(self, epics: List[JIRAEpic]):
        """Sets the epics of this FoundJIRAStuff.

        :param epics: The epics of this FoundJIRAStuff.
        """
        if epics is None:
            raise ValueError("Invalid value for `epics`, must not be `None`")

        self._epics = epics

    @property
    def labels(self) -> List[JIRALabel]:
        """Gets the labels of this FoundJIRAStuff.

        :return: The labels of this FoundJIRAStuff.
        """
        return self._labels

    @labels.setter
    def labels(self, labels: List[JIRALabel]):
        """Sets the labels of this FoundJIRAStuff.

        :param labels: The labels of this FoundJIRAStuff.
        """
        if labels is None:
            raise ValueError("Invalid value for `labels`, must not be `None`")

        self._labels = labels

    @property
    def issue_types(self) -> List[str]:
        """Gets the issue_types of this FoundJIRAStuff.

        Types of the updated issues.

        :return: The issue_types of this FoundJIRAStuff.
        """
        return self._issue_types

    @issue_types.setter
    def issue_types(self, issue_types: List[str]):
        """Sets the issue_types of this FoundJIRAStuff.

        Types of the updated issues.

        :param issue_types: The issue_types of this FoundJIRAStuff.
        """
        if issue_types is None:
            raise ValueError("Invalid value for `issue_types`, must not be `None`")

        self._issue_types = issue_types

    @property
    def priorities(self) -> List[JIRAPriority]:
        """Gets the priorities of this FoundJIRAStuff.

        Issue priority names sorted by importance in ascending order.

        :return: The priorities of this FoundJIRAStuff.
        """
        return self._priorities

    @priorities.setter
    def priorities(self, priorities: List[JIRAPriority]):
        """Sets the priorities of this FoundJIRAStuff.

        Issue priority names sorted by importance in ascending order.

        :param priorities: The priorities of this FoundJIRAStuff.
        """
        if priorities is None:
            raise ValueError("Invalid value for `priorities`, must not be `None`")

        self._priorities = priorities

    @property
    def users(self) -> Dict[str, str]:
        """Gets the users of this FoundJIRAStuff.

        Mentioned users in the found JIRA issues mapped to their avatar URLs.

        :return: The users of this FoundJIRAStuff.
        """
        return self._users

    @users.setter
    def users(self, users: Dict[str, str]):
        """Sets the users of this FoundJIRAStuff.

        Mentioned users in the found JIRA issues mapped to their avatar URLs.

        :param users: The users of this FoundJIRAStuff.
        """
        if users is None:
            raise ValueError("Invalid value for `users`, must not be `None`")

        self._users = users
