from datetime import datetime
from typing import Optional

from athenian.api.models.web.base_model_ import Model


class ReleaseNotification(Model):
    """Push message about a custom release event."""

    attribute_types = {
        "repository": str,
        "commit": str,
        "name": Optional[str],
        "author": Optional[str],
        "url": Optional[str],
        "published_at": Optional[datetime],
    }

    attribute_map = {
        "repository": "repository",
        "commit": "commit",
        "name": "name",
        "author": "author",
        "url": "url",
        "published_at": "published_at",
    }

    def __init__(
        self,
        repository: Optional[str] = None,
        commit: Optional[str] = None,
        name: Optional[str] = None,
        author: Optional[str] = None,
        url: Optional[str] = None,
        published_at: Optional[datetime] = None,
    ):
        """ReleaseNotification - a model defined in OpenAPI

        :param repository: The repository of this ReleaseNotification.
        :param commit: The commit of this ReleaseNotification.
        :param name: The name of this ReleaseNotification.
        :param author: The author of this ReleaseNotification.
        :param url: The url of this ReleaseNotification.
        :param published_at: The published_at of this ReleaseNotification.
        """
        self._repository = repository
        self._commit = commit
        self._name = name
        self._author = author
        self._url = url
        self._published_at = published_at

    @property
    def repository(self) -> str:
        """Gets the repository of this ReleaseNotification.

        Repository name which uniquely identifies any repository in any service provider.
        The format matches the repository URL without the protocol part. No \".git\" should be
        appended. We support a special syntax for repository sets: \"{reposet id}\" adds all
        the repositories from the given set.

        :return: The repository of this ReleaseNotification.
        """
        return self._repository

    @repository.setter
    def repository(self, repository: str):
        """Sets the repository of this ReleaseNotification.

        Repository name which uniquely identifies any repository in any service provider.
        The format matches the repository URL without the protocol part. No \".git\" should be
        appended. We support a special syntax for repository sets: \"{reposet id}\" adds all
        the repositories from the given set.

        :param repository: The repository of this ReleaseNotification.
        """
        if repository is None:
            raise ValueError("Invalid value for `repository`, must not be `None`")

        self._repository = repository

    @property
    def commit(self) -> str:
        """Gets the commit of this ReleaseNotification.

        Commit hash, either short (7 chars) or long (40 chars) form.

        :return: The commit of this ReleaseNotification.
        """
        return self._commit

    @commit.setter
    def commit(self, commit: str):
        """Sets the commit of this ReleaseNotification.

        Commit hash, either short (7 chars) or long (40 chars) form.

        :param commit: The commit of this ReleaseNotification.
        """
        if commit is None:
            raise ValueError("Invalid value for `commit`, must not be `None`")
        if len(commit) > 40:
            raise ValueError(
                "Invalid value for `commit`, length must be less than or equal to `40`")
        if len(commit) < 7:
            raise ValueError(
                "Invalid value for `commit`, length must be greater than or equal to `7`")

        self._commit = commit

    @property
    def name(self) -> Optional[str]:
        """Gets the name of this ReleaseNotification.

        Release name.

        :return: The name of this ReleaseNotification.
        """
        return self._name

    @name.setter
    def name(self, name: Optional[str]):
        """Sets the name of this ReleaseNotification.

        Release name.

        :param name: The name of this ReleaseNotification.
        """
        self._name = name

    @property
    def author(self) -> Optional[str]:
        """Gets the author of this ReleaseNotification.

        Release author.

        :return: The author of this ReleaseNotification.
        """
        return self._author

    @author.setter
    def author(self, author: Optional[str]):
        """Sets the author of this ReleaseNotification.

        Release author.

        :param author: The author of this ReleaseNotification.
        """
        self._author = author

    @property
    def url(self) -> Optional[str]:
        """Gets the url of this ReleaseNotification.

        Release URL.

        :return: The url of this ReleaseNotification.
        """
        return self._url

    @url.setter
    def url(self, url: Optional[str]):
        """Sets the url of this ReleaseNotification.

        Release URL.

        :param url: The url of this ReleaseNotification.
        """
        self._url = url

    @property
    def published_at(self) -> Optional[datetime]:
        """Gets the published_at of this ReleaseNotification.

        When the release was created. If missing, set to `now()`.

        :return: The published_at of this ReleaseNotification.
        """
        return self._published_at

    @published_at.setter
    def published_at(self, published_at: Optional[datetime]):
        """Sets the published_at of this ReleaseNotification.

        When the release was created. If missing, set to `now()`.

        :param published_at: The published_at of this ReleaseNotification.
        """
        self._published_at = published_at
