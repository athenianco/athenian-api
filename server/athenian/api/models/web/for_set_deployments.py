from typing import Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.common_deployment_properties import CommonDeploymentProperties
from athenian.api.models.web.deployment_with import DeploymentWith
from athenian.api.models.web.for_set_common import (
    RepositoryGroupsMixin,
    make_common_pull_request_filters,
)


class _ForSetDeployments(Model, RepositoryGroupsMixin):
    """Request body of `/metrics/deployments`, the deployments selector."""

    repositories: Optional[list[str]]
    repogroups: Optional[list[list[int]]]
    with_: (Optional[DeploymentWith], "with")
    withgroups: Optional[list[DeploymentWith]]
    environments: Optional[list[str]]
    envgroups: Optional[list[list[str]]]

    def validate_repositories(self, repositories: Optional[list[str]]) -> Optional[list[str]]:
        if repositories is None:
            return repositories
        return super().validate_repositories(repositories)

    def select_withgroup(self, index: int) -> "ForSetDeployments":
        """Change `with` to point at the specified group and clear `withgroups`."""
        fs = self.copy()
        if not self.withgroups:
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
        if self.environments:
            if index >= len(self.environments):
                raise IndexError(
                    "%d is out of range (max is %d)" % (index, len(self.environments)),
                )
            fs.environments = [self.environments[index]]
            return fs
        if self.envgroups:
            if index >= len(self.envgroups):
                raise IndexError("%d is out of range (max is %d)" % (index, len(self.envgroups)))
            fs.environments = self.envgroups[index]
            return fs
        if index > 0:
            raise IndexError("%d is out of range (no environments/envgroups)" % index)
        return fs


ForSetDeployments = AllOf(
    _ForSetDeployments,
    make_common_pull_request_filters("pr_"),
    CommonDeploymentProperties,
    name="ForSetDeployments",
    module=__name__,
)
