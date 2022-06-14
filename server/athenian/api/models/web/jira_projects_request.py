from typing import Dict, Optional

from athenian.api.models.web.base_model_ import Model


class JIRAProjectsRequest(Model):
    """Enable or disable a JIRA project."""

    attribute_types = {
        "account": int,
        "projects": Dict[str, bool],
    }

    attribute_map = {
        "account": "account",
        "projects": "projects",
    }

    def __init__(self, account: Optional[int] = None, projects: Optional[Dict[str, bool]] = None):
        """JIRAProjectsRequest - a model defined in OpenAPI

        :param account: The account of this JIRAProjectsRequest.
        :param projects: The projects of this JIRAProjectsRequest.
        """
        self._account = account
        self._projects = projects

    @property
    def account(self) -> int:
        """Gets the account of this JIRAProjectsRequest.

        Changed account ID.

        :return: The account of this JIRAProjectsRequest.
        """
        return self._account

    @account.setter
    def account(self, account: int):
        """Sets the account of this JIRAProjectsRequest.

        Changed account ID.

        :param account: The account of this JIRAProjectsRequest.
        """
        if account is None:
            raise ValueError("Invalid value for `account`, must not be `None`")

        self._account = account

    @property
    def projects(self) -> Dict[str, bool]:
        """Gets the projects of this JIRAProjectsRequest.

        Map from project projects to their enabled/disabled flags.

        :return: The projects of this JIRAProjectsRequest.
        """
        return self._projects

    @projects.setter
    def projects(self, projects: Dict[str, bool]):
        """Sets the projects of this JIRAProjectsRequest.

        Map from project projects to their enabled/disabled flags.

        :param projects: The projects of this JIRAProjectsRequest.
        """
        if projects is None:
            raise ValueError("Invalid value for `projects`, must not be `None`")

        self._projects = projects
