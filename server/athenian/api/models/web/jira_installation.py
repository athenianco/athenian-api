from typing import List, Optional

from athenian.api.models.web.base_model_ import Model


class JIRAInstallation(Model):
    """Information about a link with JIRA."""

    attribute_types = {"url": str, "projects": List[str]}
    attribute_map = {"url": "url", "projects": "projects"}

    def __init__(self,
                 url: Optional[str] = None,
                 projects: Optional[List[str]] = None):
        """JIRAInstallation - a model defined in OpenAPI

        :param url: The url of this JIRAInstallation.
        :param projects: The projects of this JIRAInstallation.
        """
        self._url = url
        self._projects = projects

    @property
    def url(self) -> str:
        """Gets the url of this JIRAInstallation.

        JIRA base URL.

        :return: The url of this JIRAInstallation.
        """
        return self._url

    @url.setter
    def url(self, url: str):
        """Sets the url of this JIRAInstallation.

        JIRA base URL.

        :param url: The url of this JIRAInstallation.
        """
        if url is None:
            raise ValueError("Invalid value for `url`, must not be `None`")

        self._url = url

    @property
    def projects(self) -> List[str]:
        """Gets the projects of this JIRAInstallation.

        List of accessible project keys.

        :return: The projects of this JIRAInstallation.
        """
        return self._projects

    @projects.setter
    def projects(self, projects: List[str]):
        """Sets the projects of this JIRAInstallation.

        List of accessible project keys.

        :param projects: The projects of this JIRAInstallation.
        """
        if projects is None:
            raise ValueError("Invalid value for `projects`, must not be `None`")

        self._projects = projects
