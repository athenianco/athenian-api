from typing import List, Optional

from athenian.api.models.web.base_model_ import Model


class DeploymentWith(Model):
    """Deployment contribution roles. The aggregation is `OR` everywhere."""

    openapi_types = {
        "pr_author": Optional[List],
        "commit_author": Optional[List],
        "releaser": Optional[List],
        "deployer": Optional[List],
    }

    attribute_map = {
        "pr_author": "pr_author",
        "commit_author": "commit_author",
        "releaser": "releaser",
        "deployer": "deployer",
    }

    def __init__(
        self,
        pr_author: Optional[List] = None,
        commit_author: Optional[List] = None,
        releaser: Optional[List] = None,
        deployer: Optional[List] = None,
    ):
        """DeploymentWith - a model defined in OpenAPI

        :param pr_author: The pr_author of this DeploymentWith.
        :param commit_author: The commit_author of this DeploymentWith.
        :param releaser: The releaser of this DeploymentWith.
        :param deployer: The deployer of this DeploymentWith.
        """
        self._pr_author = pr_author
        self._commit_author = commit_author
        self._releaser = releaser
        self._deployer = deployer

    @property
    def pr_author(self) -> Optional[List]:
        """Gets the pr_author of this DeploymentWith.

        :return: The pr_author of this DeploymentWith.
        """
        return self._pr_author

    @pr_author.setter
    def pr_author(self, pr_author: Optional[List]):
        """Sets the pr_author of this DeploymentWith.

        :param pr_author: The pr_author of this DeploymentWith.
        """
        self._pr_author = pr_author

    @property
    def commit_author(self) -> Optional[List]:
        """Gets the commit_author of this DeploymentWith.

        :return: The commit_author of this DeploymentWith.
        """
        return self._commit_author

    @commit_author.setter
    def commit_author(self, commit_author: Optional[List]):
        """Sets the commit_author of this DeploymentWith.

        :param commit_author: The commit_author of this DeploymentWith.
        """
        self._commit_author = commit_author

    @property
    def releaser(self) -> Optional[List]:
        """Gets the releaser of this DeploymentWith.

        :return: The releaser of this DeploymentWith.
        """
        return self._releaser

    @releaser.setter
    def releaser(self, releaser: Optional[List]):
        """Sets the releaser of this DeploymentWith.

        :param releaser: The releaser of this DeploymentWith.
        """
        self._releaser = releaser

    @property
    def deployer(self) -> Optional[List]:
        """Gets the deployer of this DeploymentWith.

        :return: The deployer of this DeploymentWith.
        """
        return self._deployer

    @deployer.setter
    def deployer(self, deployer: Optional[List]):
        """Sets the deployer of this DeploymentWith.

        :param deployer: The deployer of this DeploymentWith.
        """
        self._deployer = deployer
