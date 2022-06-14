from typing import Optional

from athenian.api.models.web.base_model_ import Model


class DeployedComponent(Model):
    """Definition of the deployed software unit."""

    attribute_types = {"repository": str, "reference": str}

    attribute_map = {"repository": "repository", "reference": "reference"}

    def __init__(
        self,
        repository: Optional[str] = None,
        reference: Optional[str] = None,
    ):
        """DeployedComponent - a model defined in OpenAPI

        :param repository: The repository of this DeployedComponent.
        :param reference: The reference of this DeployedComponent.
        """
        self._repository = repository
        self._reference = reference

    @property
    def repository(self) -> str:
        """Gets the repository of this DeployedComponent.

        Repository name which uniquely identifies any repository in any service provider.
        The format matches the repository URL without the protocol part. No \".git\" should be
        appended. We support a special syntax for repository sets: \"{reposet id}\" adds all
        the repositories from the given set.

        :return: The repository of this DeployedComponent.
        """
        return self._repository

    @repository.setter
    def repository(self, repository: str):
        """Sets the repository of this DeployedComponent.

        Repository name which uniquely identifies any repository in any service provider.
        The format matches the repository URL without the protocol part. No \".git\" should be
        appended. We support a special syntax for repository sets: \"{reposet id}\" adds all
        the repositories from the given set.

        :param repository: The repository of this DeployedComponent.
        """
        if repository is None:
            raise ValueError("Invalid value for `repository`, must not be `None`")

        self._repository = repository

    @property
    def reference(self) -> str:
        """Gets the reference of this DeployedComponent.

        We accept three ways to define a Git reference: 1. Tag name. 2. Full commit hash
        (40 characters). 3. Short commit hash (7 characters).  We ignore the reference while we
        cannot find it in our database. There can be two reasons: - There is a mistake or a typo
        in the provided data. - We are temporarily unable to synchronize with GitHub.

        :return: The reference of this DeployedComponent.
        """
        return self._reference

    @reference.setter
    def reference(self, reference: str):
        """Sets the reference of this DeployedComponent.

        We accept three ways to define a Git reference: 1. Tag name. 2. Full commit hash
        (40 characters). 3. Short commit hash (7 characters).  We ignore the reference while we
        cannot find it in our database. There can be two reasons: - There is a mistake or a typo
        in the provided data. - We are temporarily unable to synchronize with GitHub.

        :param reference: The reference of this DeployedComponent.
        """
        if reference is None:
            raise ValueError("Invalid value for `reference`, must not be `None`")
        if not reference:
            raise ValueError(
                "Invalid value for `reference`, length must be greater than or equal to `1`"
            )

        self._reference = reference
