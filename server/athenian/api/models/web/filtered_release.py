from datetime import datetime, timedelta
from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.released_pull_request import ReleasedPullRequest


class FilteredRelease(Model):
    """Various information about a repository release."""

    openapi_types = {
        "name": str,
        "repository": str,
        "url": str,
        "published": datetime,
        "age": timedelta,
        "added_lines": int,
        "deleted_lines": int,
        "commits": int,
        "publisher": str,
        "commit_authors": List[str],
        "prs": List[ReleasedPullRequest],
        "deployments": Optional[List[str]],
    }

    attribute_map = {
        "name": "name",
        "repository": "repository",
        "url": "url",
        "published": "published",
        "age": "age",
        "added_lines": "added_lines",
        "deleted_lines": "deleted_lines",
        "commits": "commits",
        "publisher": "publisher",
        "commit_authors": "commit_authors",
        "prs": "prs",
        "deployments": "deployments",
    }

    def __init__(
        self,
        name: Optional[str] = None,
        repository: Optional[str] = None,
        url: Optional[str] = None,
        published: Optional[datetime] = None,
        age: Optional[timedelta] = None,
        added_lines: Optional[int] = None,
        deleted_lines: Optional[int] = None,
        commits: Optional[int] = None,
        publisher: Optional[str] = None,
        commit_authors: Optional[List[str]] = None,
        prs: Optional[List[ReleasedPullRequest]] = None,
        deployments: Optional[List[str]] = None,
    ):
        """FilteredRelease - a model defined in OpenAPI

        :param name: The name of this FilteredRelease.
        :param repository: The repository of this FilteredRelease.
        :param url: The url of this FilteredRelease.
        :param published: The published of this FilteredRelease.
        :param age: The age of this FilteredRelease.
        :param added_lines: The added_lines of this FilteredRelease.
        :param deleted_lines: The deleted_lines of this FilteredRelease.
        :param commits: The commits of this FilteredRelease.
        :param publisher: The publisher of this FilteredRelease.
        :param commit_authors: The publisher of this FilteredRelease.
        :param deployments: The deployments of this FilteredRelease.
        """
        self._name = name
        self._repository = repository
        self._url = url
        self._published = published
        self._age = age
        self._added_lines = added_lines
        self._deleted_lines = deleted_lines
        self._commits = commits
        self._publisher = publisher
        self._commit_authors = commit_authors
        self._prs = prs
        self._deployments = deployments

    @property
    def name(self) -> str:
        """Gets the name of this FilteredRelease.

        Title of the release.

        :return: The name of this FilteredRelease.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Sets the name of this FilteredRelease.

        Title of the release.

        :param name: The name of this FilteredRelease.
        """
        if name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")

        self._name = name

    @property
    def repository(self) -> str:
        """Gets the repository of this FilteredRelease.

        Name of the repository where the release exists.

        :return: The repository of this FilteredRelease.
        """
        return self._repository

    @repository.setter
    def repository(self, repository: str):
        """Sets the repository of this FilteredRelease.

        Name of the repository where the release exists.

        :param repository: The repository of this FilteredRelease.
        """
        if repository is None:
            raise ValueError("Invalid value for `repository`, must not be `None`")

        self._repository = repository

    @property
    def url(self) -> str:
        """Gets the url of this FilteredRelease.

        Link to the release.

        :return: The url of this FilteredRelease.
        """
        return self._url

    @url.setter
    def url(self, url: str):
        """Sets the url of this FilteredRelease.

        Link to the release.

        :param url: The url of this FilteredRelease.
        """
        if url is None:
            raise ValueError("Invalid value for `url`, must not be `None`")

        self._url = url

    @property
    def published(self) -> datetime:
        """Gets the published of this FilteredRelease.

        When the release was created.

        :return: The published of this FilteredRelease.
        """
        return self._published

    @published.setter
    def published(self, published: datetime):
        """Sets the published of this FilteredRelease.

        When the release was created.

        :param published: The published of this FilteredRelease.
        """
        if published is None:
            raise ValueError("Invalid value for `published`, must not be `None`")

        self._published = published

    @property
    def age(self) -> timedelta:
        """Gets the age of this FilteredRelease.

        When the release was created.

        :return: The age of this FilteredRelease.
        """
        return self._age

    @age.setter
    def age(self, age: timedelta):
        """Sets the age of this FilteredRelease.

        When the release was created.

        :param age: The age of this FilteredRelease.
        """
        if age is None:
            raise ValueError("Invalid value for `age`, must not be `None`")

        self._age = age

    @property
    def added_lines(self) -> int:
        """Gets the added_lines of this FilteredRelease.

        Cumulative number of lines inserted since the previous release.

        :return: The added_lines of this FilteredRelease.
        """
        return self._added_lines

    @added_lines.setter
    def added_lines(self, added_lines: int):
        """Sets the added_lines of this FilteredRelease.

        Cumulative number of lines inserted since the previous release.

        :param added_lines: The added_lines of this FilteredRelease.
        """
        if added_lines is None:
            raise ValueError("Invalid value for `added_lines`, must not be `None`")

        self._added_lines = added_lines

    @property
    def deleted_lines(self) -> int:
        """Gets the deleted_lines of this FilteredRelease.

        Cumulative number of lines removed since the previous release.

        :return: The deleted_lines of this FilteredRelease.
        """
        return self._deleted_lines

    @deleted_lines.setter
    def deleted_lines(self, deleted_lines: int):
        """Sets the deleted_lines of this FilteredRelease.

        Cumulative number of lines removed since the previous release.

        :param deleted_lines: The deleted_lines of this FilteredRelease.
        """
        if deleted_lines is None:
            raise ValueError("Invalid value for `deleted_lines`, must not be `None`")

        self._deleted_lines = deleted_lines

    @property
    def commits(self) -> int:
        """Gets the commits of this FilteredRelease.

        Number of commits since the previous release.

        :return: The commits of this FilteredRelease.
        """
        return self._commits

    @commits.setter
    def commits(self, commits: int):
        """Sets the commits of this FilteredRelease.

        Number of commits since the previous release.

        :param commits: The commits of this FilteredRelease.
        """
        if commits is None:
            raise ValueError("Invalid value for `commits`, must not be `None`")

        self._commits = commits

    @property
    def publisher(self) -> str:
        """Gets the publisher of this FilteredRelease.

        Login of the person who created the release.

        :return: The publisher of this FilteredRelease.
        """
        return self._publisher

    @publisher.setter
    def publisher(self, publisher: str):
        """Sets the publisher of this FilteredRelease.

        Login of the person who created the release.

        :param publisher: The publisher of this FilteredRelease.
        """
        if publisher is None:
            raise ValueError("Invalid value for `publisher`, must not be `None`")

        self._publisher = publisher

    @property
    def commit_authors(self) -> List[str]:
        """Gets the commit commit_authors of this FilteredRelease.

        Profile picture of the person who created the release.

        :return: The commit commit_authors of this FilteredRelease.
        """
        return self._commit_authors

    @commit_authors.setter
    def commit_authors(self, commit_authors: List[str]):
        """Sets the commit commit_authors of this FilteredRelease.

        Profile picture of the person who created the release.

        :param commit_authors: The commit commit_authors of this FilteredRelease.
        """
        if commit_authors is None:
            raise ValueError("Invalid value for `commit_authors`, must not be `None`")

        self._commit_authors = commit_authors

    @property
    def prs(self) -> List[ReleasedPullRequest]:
        """Gets the commit prs of this FilteredRelease.

        List of released pull requests.

        :return: The commit prs of this FilteredRelease.
        """
        return self._prs

    @prs.setter
    def prs(self, prs: List[ReleasedPullRequest]):
        """Sets the commit prs of this FilteredRelease.

        List of released pull requests.

        :param prs: The commit prs of this FilteredRelease.
        """
        if prs is None:
            raise ValueError("Invalid value for `prs`, must not be `None`")

        self._prs = prs

    @property
    def deployments(self) -> Optional[List[str]]:
        """Gets the deployments of this FilteredRelease.

        Deployments with this release.

        :return: The deployments of this FilteredRelease.
        """
        return self._deployments

    @deployments.setter
    def deployments(self, deployments: Optional[List[str]]):
        """Sets the deployments of this FilteredRelease.

        Deployments with this release.

        :param deployments: The deployments of this FilteredRelease.
        """
        self._deployments = deployments
