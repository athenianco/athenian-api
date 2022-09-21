from typing import Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.common_deployment_properties import CommonDeploymentProperties
from athenian.api.models.web.common_filter_properties import CommonFilterProperties
from athenian.api.models.web.deployment_conclusion import DeploymentConclusion
from athenian.api.models.web.deployment_with import DeploymentWith
from athenian.api.models.web.for_set_common import make_common_pull_request_filters


class _FilterDeploymentsRequest(Model, sealed=False):
    """Filters to select the deployments in `/filter/deployments`."""

    in_: (Optional[list[str]], "in")
    with_: (Optional[DeploymentWith], "with")
    environments: Optional[list[str]]
    conclusions: Optional[list[str]]

    def validate_conclusions(self, conclusions: Optional[list[str]]) -> Optional[list[str]]:
        """Sets the conclusions of this FilterDeploymentsRequest.

        Deployments must execute with any of these conclusions.

        :param conclusions: The conclusions of this FilterDeploymentsRequest.
        """
        if conclusions is not None:
            for value in conclusions:
                if value not in DeploymentConclusion:
                    raise ValueError(
                        f'Value "{value}" must be one of {list(DeploymentConclusion)}',
                    )
        return conclusions


FilterDeploymentsRequest = AllOf(
    _FilterDeploymentsRequest,
    CommonFilterProperties,
    make_common_pull_request_filters("pr_"),
    CommonDeploymentProperties,
    name="FilterDeploymentsRequest",
    module=__name__,
)
