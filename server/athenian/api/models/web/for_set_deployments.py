from typing import List, Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.common_deployment_properties import CommonDeploymentProperties
from athenian.api.models.web.deployment_with import DeploymentWith
from athenian.api.models.web.for_set import make_common_pull_request_filters, RepositoryGroupsMixin


class _ForSetDeployments(Model, RepositoryGroupsMixin):
    """Request body of `/metrics/deployments`, the deployments selector."""

    openapi_types = {
        "repositories": Optional[List[str]],
        "repogroups": Optional[List[List[int]]],
        "with_": Optional[DeploymentWith],
        "withgroups": Optional[List[DeploymentWith]],
        "environments": Optional[List[List[str]]],
    }

    attribute_map = {
        "repositories": "repositories",
        "repogroups": "repogroups",
        "with_": "with",
        "withgroups": "withgroups",
        "environments": "environments",
    }

    def __init__(
        self,
        repositories: Optional[List[str]] = None,
        repogroups: Optional[List[List[int]]] = None,
        with_: Optional[DeploymentWith] = None,
        withgroups: Optional[List[DeploymentWith]] = None,
        environments: Optional[List[List[str]]] = None,
    ):
        """ForSetDeployments - a model defined in OpenAPI

        :param repositories: The repositories of this ForSetDeployments.
        :param repogroups: The repogroups of this ForSetDeployments.
        :param with_: The with_ of this ForSetDeployments.
        :param withgroups: The withgroups of this ForSetDeployments.
        :param environments: The environments of this ForSetDeployments.
        """
        self._repositories = repositories
        self._repogroups = repogroups
        self._with_ = with_
        self._withgroups = withgroups
        self._environments = environments

    @property
    def with_(self) -> Optional[DeploymentWith]:
        """Gets the with_ of this ForSetDeployments.

        :return: The with_ of this ForSetDeployments.
        """
        return self._with_

    @with_.setter
    def with_(self, with_: Optional[DeploymentWith]):
        """Sets the with_ of this ForSetDeployments.

        :param with_: The with_ of this ForSetDeployments.
        """
        self._with_ = with_

    @property
    def withgroups(self) -> Optional[List[DeploymentWith]]:
        """Gets the withgroups of this ForSetDeployments.

        Alternative to `with` - calculate metrics for distinct filters separately.

        :return: The withgroups of this ForSetDeployments.
        """
        return self._withgroups

    @withgroups.setter
    def withgroups(self, withgroups: Optional[List[DeploymentWith]]):
        """Sets the withgroups of this ForSetDeployments.

        Alternative to `with` - calculate metrics for distinct filters separately.

        :param withgroups: The withgroups of this ForSetDeployments.
        """
        self._withgroups = withgroups

    @property
    def environments(self) -> Optional[List[List[str]]]:
        """Gets the environments of this ForSetDeployments.

        List of environment groups for which to calculate the metrics.

        :return: The environments of this ForSetDeployments.
        """
        return self._environments

    @environments.setter
    def environments(self, environments: Optional[List[List[str]]]):
        """Sets the environments of this ForSetDeployments.

        List of environment groups for which to calculate the metrics.

        :param environments: The environments of this ForSetDeployments.
        """
        self._environments = environments

    def select_withgroup(self, index: int) -> "ForSetDeployments":
        """Change `with` to point at the specified group and clear `withgroups`."""
        fs = self.copy()
        if self.withgroups is None:
            if index > 0:
                raise IndexError("%d is out of range (no withgroups)" % index)
            return fs
        if index >= len(self.withgroups):
            raise IndexError("%d is out of range (max is %d)" % (index, len(self.withgroups)))
        fs.withgroups = None
        fs.with_ = self.withgroups[index]
        return fs

    def select_envgroup(self, index: int) -> "ForSetDeployments":
        """Change `environments` to contain only the specified group."""
        fs = self.copy()
        if self.environments is None:
            if index > 0:
                raise IndexError("%d is out of range (no environments)" % index)
            return fs
        if index >= len(self.environments):
            raise IndexError("%d is out of range (max is %d)" % (index, len(self.environments)))
        fs.environments = self.environments[index]
        return fs


ForSetDeployments = AllOf(_ForSetDeployments,
                          make_common_pull_request_filters("pr_"),
                          CommonDeploymentProperties,
                          name="ForSetDeployments",
                          module=__name__)
