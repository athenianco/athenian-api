from typing import Dict, List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.deployment_notification import DeploymentNotification
from athenian.api.models.web.jira_epic import JIRAEpic
from athenian.api.models.web.jira_issue import JIRAIssue
from athenian.api.models.web.jira_issue_type import JIRAIssueType
from athenian.api.models.web.jira_label import JIRALabel
from athenian.api.models.web.jira_priority import JIRAPriority
from athenian.api.models.web.jira_status import JIRAStatus
from athenian.api.models.web.jira_user import JIRAUser
from athenian.api.typing_utils import VerbatimOptional


class FilteredJIRAStuff(Model):
    """Response from `/filter/jira`: found JIRA epics, labels, issue types, priorities, and \
    mentioned users."""

    attribute_types = {
        "epics": VerbatimOptional[List[JIRAEpic]],
        "issues": VerbatimOptional[List[JIRAIssue]],
        "labels": VerbatimOptional[List[JIRALabel]],
        "issue_types": VerbatimOptional[List[JIRAIssueType]],
        "priorities": VerbatimOptional[List[JIRAPriority]],
        "statuses": VerbatimOptional[List[JIRAStatus]],
        "users": VerbatimOptional[List[JIRAUser]],
        "deployments": VerbatimOptional[Dict[str, DeploymentNotification]],
    }
    attribute_map = {
        "epics": "epics",
        "issues": "issues",
        "labels": "labels",
        "issue_types": "issue_types",
        "priorities": "priorities",
        "statuses": "statuses",
        "users": "users",
        "deployments": "deployments",
    }

    def __init__(self,
                 epics: Optional[List[JIRAEpic]] = None,
                 issues: Optional[List[JIRAIssue]] = None,
                 labels: Optional[List[JIRALabel]] = None,
                 issue_types: Optional[List[JIRAIssueType]] = None,
                 priorities: Optional[List[JIRAPriority]] = None,
                 statuses: Optional[List[JIRAStatus]] = None,
                 users: Optional[List[JIRAUser]] = None,
                 deployments: Optional[Dict[str, DeploymentNotification]] = None):
        """FilteredJIRAStuff - a model defined in OpenAPI

        :param epics: The epics of this FilteredJIRAStuff.
        :param issues: The issues of this FilteredJIRAStuff.
        :param labels: The labels of this FilteredJIRAStuff.
        :param issue_types: The issue_types of this FilteredJIRAStuff.
        :param priorities: The priorities of this FilteredJIRAStuff.
        :param statuses: The statuses of this FilteredJIRAStuff.
        :param users: The users of this FilteredJIRAStuff.
        :param deployments: The deployments of this FilteredJIRAStuff.
        """
        self._epics = epics
        self._issues = issues
        self._labels = labels
        self._issue_types = issue_types
        self._priorities = priorities
        self._statuses = statuses
        self._users = users
        self._deployments = deployments

    @property
    def epics(self) -> Optional[List[JIRAEpic]]:
        """Gets the epics of this FilteredJIRAStuff.

        :return: The epics of this FilteredJIRAStuff.
        """
        return self._epics

    @epics.setter
    def epics(self, epics: Optional[List[JIRAEpic]]):
        """Sets the epics of this FilteredJIRAStuff.

        :param epics: The epics of this FilteredJIRAStuff.
        """
        self._epics = epics

    @property
    def issues(self) -> Optional[List[JIRAIssue]]:
        """Gets the issues of this FilteredJIRAStuff.

        :return: The issues of this FilteredJIRAStuff.
        """
        return self._issues

    @issues.setter
    def issues(self, issues: Optional[List[JIRAIssue]]):
        """Sets the issues of this FilteredJIRAStuff.

        :param issues: The issues of this FilteredJIRAStuff.
        """
        self._issues = issues

    @property
    def labels(self) -> Optional[List[JIRALabel]]:
        """Gets the labels of this FilteredJIRAStuff.

        :return: The labels of this FilteredJIRAStuff.
        """
        return self._labels

    @labels.setter
    def labels(self, labels: Optional[List[JIRALabel]]):
        """Sets the labels of this FilteredJIRAStuff.

        :param labels: The labels of this FilteredJIRAStuff.
        """
        self._labels = labels

    @property
    def issue_types(self) -> Optional[List[JIRAIssueType]]:
        """Gets the issue_types of this FilteredJIRAStuff.

        Types of the updated issues.

        :return: The issue_types of this FilteredJIRAStuff.
        """
        return self._issue_types

    @issue_types.setter
    def issue_types(self, issue_types: Optional[List[JIRAIssueType]]):
        """Sets the issue_types of this FilteredJIRAStuff.

        Types of the updated issues.

        :param issue_types: The issue_types of this FilteredJIRAStuff.
        """
        self._issue_types = issue_types

    @property
    def priorities(self) -> Optional[List[JIRAPriority]]:
        """Gets the priorities of this FilteredJIRAStuff.

        Issue priority names sorted by importance in ascending order.

        :return: The priorities of this FilteredJIRAStuff.
        """
        return self._priorities

    @priorities.setter
    def priorities(self, priorities: Optional[List[JIRAPriority]]):
        """Sets the priorities of this FilteredJIRAStuff.

        Issue priority names sorted by importance in ascending order.

        :param priorities: The priorities of this FilteredJIRAStuff.
        """
        self._priorities = priorities

    @property
    def statuses(self) -> Optional[List[JIRAStatus]]:
        """Gets the statuses of this FilteredJIRAStuff.

        Mentioned issue statuses sorted by name.

        :return: The statuses of this FilteredJIRAStuff.
        """
        return self._statuses

    @statuses.setter
    def statuses(self, statuses: Optional[List[JIRAStatus]]):
        """Sets the statuses of this FilteredJIRAStuff.

        Mentioned issue statuses sorted by name.

        :param statuses: The statuses of this FilteredJIRAStuff.
        """
        self._statuses = statuses

    @property
    def users(self) -> Optional[List[JIRAUser]]:
        """Gets the users of this FilteredJIRAStuff.

        Mentioned users in the found JIRA issues.

        :return: The users of this FilteredJIRAStuff.
        """
        return self._users

    @users.setter
    def users(self, users: Optional[List[JIRAUser]]):
        """Sets the users of this FilteredJIRAStuff.

        Mentioned users in the found JIRA issues.

        :param users: The users of this FilteredJIRAStuff.
        """
        self._users = users

    @property
    def deployments(self) -> Optional[Dict[str, DeploymentNotification]]:
        """Gets the deployments of this FilteredJIRAStuff.

        Mentioned deployments.

        :return: The deployments of this FilteredJIRAStuff.
        """
        return self._deployments

    @deployments.setter
    def deployments(self, deployments: Optional[Dict[str, DeploymentNotification]]):
        """Sets the deployments of this FilteredJIRAStuff.

        Mentioned deployments.

        :param deployments: The deployments of this FilteredJIRAStuff.
        """
        self._deployments = deployments
