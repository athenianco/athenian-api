from typing import List, Optional

from athenian.api.models.web.base_model_ import Model


class ReleaseNames(Model):
    """Repository name and a list of release names in that repository."""

    attribute_types = {"repository": str, "names": List[str]}
    attribute_map = {"repository": "repository", "names": "names"}

    def __init__(self,
                 repository: Optional[str] = None,
                 names: Optional[List[str]] = None):
        """ReleaseNames - a model defined in OpenAPI

        :param repository: The repository of this ReleaseNames.
        :param names: The names of this ReleaseNames.
        """
        self._repository = repository
        self._names = names

    @property
    def repository(self) -> str:
        """Gets the repository of this ReleaseNames.

        Repository name which uniquely identifies any repository in any service provider.
        The format matches the repository URL without the protocol part. No \".git\" should be
        appended. We support a special syntax for repository sets: \"{reposet id}\" adds all
        the repositories from the given set.

        :return: The repository of this ReleaseNames.
        """
        return self._repository

    @repository.setter
    def repository(self, repository: str):
        """Sets the repository of this ReleaseNames.

        Repository name which uniquely identifies any repository in any service provider.
        The format matches the repository URL without the protocol part. No \".git\" should be
        appended. We support a special syntax for repository sets: \"{reposet id}\" adds all
        the repositories from the given set.

        :param repository: The repository of this ReleaseNames.
        """
        if repository is None:
            raise ValueError("Invalid value for `repository`, must not be `None`")

        self._repository = repository

    @property
    def names(self) -> List[str]:
        """Gets the names of this ReleaseNames.

        List of release names. For tag releases, those are the tag names. For branch releases,
        those are commit hashes.

        :return: The names of this ReleaseNames.
        """
        return self._names

    @names.setter
    def names(self, names: List[str]):
        """Sets the names of this ReleaseNames.

        List of release names. For tag releases, those are the tag names. For branch releases,
        those are commit hashes.

        :param names: The names of this ReleaseNames.
        """
        if names is None:
            raise ValueError("Invalid value for `names`, must not be `None`")

        self._names = names
