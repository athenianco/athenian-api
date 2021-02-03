from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.jira_epic import JIRAEpic
from athenian.api.models.web.jira_issue_type import JIRAIssueType
from athenian.api.models.web.jira_label import JIRALabel
from athenian.api.models.web.jira_priority import JIRAPriority
from athenian.api.models.web.jira_status import JIRAStatus
from athenian.api.models.web.jira_user import JIRAUser


class FoundJIRAStuff(Model):
    """Response from `/filter/jira`: found JIRA epics, labels, issue types, priorities, and \
    mentioned users."""

    openapi_types = {
        "epics": Optional[List[JIRAEpic]],
        "labels": Optional[List[JIRALabel]],
        "issue_types": Optional[List[JIRAIssueType]],
        "priorities": Optional[List[JIRAPriority]],
        "statuses": Optional[List[JIRAStatus]],
        "users": Optional[List[JIRAUser]],
    }
    attribute_map = {
        "epics": "epics",
        "labels": "labels",
        "issue_types": "issue_types",
        "priorities": "priorities",
        "statuses": "statuses",
        "users": "users",
    }

    def __init__(self,
                 epics: Optional[List[JIRAEpic]] = None,
                 labels: Optional[List[JIRALabel]] = None,
                 issue_types: Optional[List[JIRAIssueType]] = None,
                 priorities: Optional[List[JIRAPriority]] = None,
                 statuses: Optional[List[JIRAStatus]] = None,
                 users: Optional[List[JIRAUser]] = None):
        """FoundJIRAStuff - a model defined in OpenAPI

        :param epics: The epics of this FoundJIRAStuff.
        :param labels: The labels of this FoundJIRAStuff.
        :param issue_types: The issue_types of this FoundJIRAStuff.
        :param priorities: The priorities of this FoundJIRAStuff.
        :param statuses: The statuses of this FoundJIRAStuff.
        :param users: The users of this FoundJIRAStuff.
        """
        self._epics = epics
        self._labels = labels
        self._issue_types = issue_types
        self._priorities = priorities
        self._statuses = statuses
        self._users = users

    @property
    def epics(self) -> Optional[List[JIRAEpic]]:
        """Gets the epics of this FoundJIRAStuff.

        :return: The epics of this FoundJIRAStuff.
        """
        return self._epics

    @epics.setter
    def epics(self, epics: Optional[List[JIRAEpic]]):
        """Sets the epics of this FoundJIRAStuff.

        :param epics: The epics of this FoundJIRAStuff.
        """
        self._epics = epics

    @property
    def labels(self) -> Optional[List[JIRALabel]]:
        """Gets the labels of this FoundJIRAStuff.

        :return: The labels of this FoundJIRAStuff.
        """
        return self._labels

    @labels.setter
    def labels(self, labels: Optional[List[JIRALabel]]):
        """Sets the labels of this FoundJIRAStuff.

        :param labels: The labels of this FoundJIRAStuff.
        """
        self._labels = labels

    @property
    def issue_types(self) -> Optional[List[JIRAIssueType]]:
        """Gets the issue_types of this FoundJIRAStuff.

        Types of the updated issues.

        :return: The issue_types of this FoundJIRAStuff.
        """
        return self._issue_types

    @issue_types.setter
    def issue_types(self, issue_types: Optional[List[JIRAIssueType]]):
        """Sets the issue_types of this FoundJIRAStuff.

        Types of the updated issues.

        :param issue_types: The issue_types of this FoundJIRAStuff.
        """
        self._issue_types = issue_types

    @property
    def priorities(self) -> Optional[List[JIRAPriority]]:
        """Gets the priorities of this FoundJIRAStuff.

        Issue priority names sorted by importance in ascending order.

        :return: The priorities of this FoundJIRAStuff.
        """
        return self._priorities

    @priorities.setter
    def priorities(self, priorities: Optional[List[JIRAPriority]]):
        """Sets the priorities of this FoundJIRAStuff.

        Issue priority names sorted by importance in ascending order.

        :param priorities: The priorities of this FoundJIRAStuff.
        """
        self._priorities = priorities

    @property
    def statuses(self) -> Optional[List[JIRAStatus]]:
        """Gets the statuses of this FoundJIRAStuff.

        Mentioned issue statuses sorted by name.

        :return: The statuses of this FoundJIRAStuff.
        """
        return self._statuses

    @statuses.setter
    def statuses(self, statuses: Optional[List[JIRAStatus]]):
        """Sets the statuses of this FoundJIRAStuff.

        Mentioned issue statuses sorted by name.

        :param statuses: The statuses of this FoundJIRAStuff.
        """
        self._statuses = statuses

    @property
    def users(self) -> Optional[List[JIRAUser]]:
        """Gets the users of this FoundJIRAStuff.

        Mentioned users in the found JIRA issues.

        :return: The users of this FoundJIRAStuff.
        """
        return self._users

    @users.setter
    def users(self, users: Optional[List[JIRAUser]]):
        """Sets the users of this FoundJIRAStuff.

        Mentioned users in the found JIRA issues.

        :param users: The users of this FoundJIRAStuff.
        """
        self._users = users
