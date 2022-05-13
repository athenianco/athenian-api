from typing import List

from athenian.api.models.web.base_model_ import Model


class PullRequestNumbers(Model):
    """Repository name and a list of PR numbers in that repository."""

    attribute_types = {"repository": str, "numbers": List[int]}
    attribute_map = {"repository": "repository", "numbers": "numbers"}

    def __init__(self, repository: str = None, numbers: List[int] = None):
        """PullRequestNumbers - a model defined in OpenAPI

        :param repository: The repository of this PullRequestNumbers.
        :param numbers: The numbers of this PullRequestNumbers.
        """
        self._repository = repository
        self._numbers = numbers

    @property
    def repository(self) -> str:
        """Gets the repository of this PullRequestNumbers.

        Repository name which uniquely identifies any repository in any service provider.
        The format matches the repository URL without the protocol part. No \".git\" should be
        appended. We support a special syntax for repository sets: \"{reposet id}\" adds all
        the repositories from the given set.

        :return: The repository of this PullRequestNumbers.
        """
        return self._repository

    @repository.setter
    def repository(self, repository: str):
        """Sets the repository of this PullRequestNumbers.

        Repository name which uniquely identifies any repository in any service provider.
        The format matches the repository URL without the protocol part. No \".git\" should be
        appended. We support a special syntax for repository sets: \"{reposet id}\" adds all
        the repositories from the given set.

        :param repository: The repository of this PullRequestNumbers.
        """
        if repository is None:
            raise ValueError("Invalid value for `repository`, must not be `None`")

        self._repository = repository

    @property
    def numbers(self) -> List[int]:
        """Gets the numbers of this PullRequestNumbers.

        :return: The numbers of this PullRequestNumbers.
        """
        return self._numbers

    @numbers.setter
    def numbers(self, numbers: List[int]):
        """Sets the numbers of this PullRequestNumbers.

        :param numbers: The numbers of this PullRequestNumbers.
        """
        if numbers is None:
            raise ValueError("Invalid value for `numbers`, must not be `None`")

        self._numbers = numbers
