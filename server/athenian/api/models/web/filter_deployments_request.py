from typing import List, Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.common_deployment_properties import CommonDeploymentProperties
from athenian.api.models.web.common_filter_properties import CommonFilterProperties
from athenian.api.models.web.deployment_conclusion import DeploymentConclusion
from athenian.api.models.web.deployment_with import DeploymentWith
from athenian.api.models.web.for_set_pull_requests import make_common_pull_request_filters


class _FilterDeploymentsRequest(Model, sealed=False):
    """Filters to select the deployments in `/filter/deployments`."""

    openapi_types = {
        "in_": List[str],
        "with_": DeploymentWith,
        "environments": List[str],
        "conclusions": List[str],
    }

    attribute_map = {
        "in_": "in",
        "with_": "with",
        "environments": "environments",
        "conclusions": "conclusions",
    }

    def __init__(
        self,
        in_: Optional[List[str]] = None,
        with_: Optional[DeploymentWith] = None,
        environments: Optional[List[str]] = None,
        conclusions: Optional[List[str]] = None,
    ):
        """FilterDeploymentsRequest - a model defined in OpenAPI

        :param in_: The in_ of this FilterDeploymentsRequest.
        :param with_: The with_ of this FilterDeploymentsRequest.
        :param environments: The environments of this FilterDeploymentsRequest.
        :param conclusions: The conclusions of this FilterDeploymentsRequest.
        """
        self._in_ = in_
        self._with_ = with_
        self._environments = environments
        self._conclusions = conclusions

    @property
    def in_(self) -> List[str]:
        """Gets the in_ of this FilterDeploymentsRequest.

        Set of repositories. An empty list raises a bad response 400. Duplicates are automatically
        ignored.

        :return: The in_ of this FilterDeploymentsRequest.
        """
        return self._in_

    @in_.setter
    def in_(self, in_: List[str]):
        """Sets the in_ of this FilterDeploymentsRequest.

        Set of repositories. An empty list raises a bad response 400. Duplicates are automatically
        ignored.

        :param in_: The in_ of this FilterDeploymentsRequest.
        """
        self._in_ = in_

    @property
    def with_(self) -> DeploymentWith:
        """Gets the with_ of this FilterDeploymentsRequest.

        :return: The with_ of this FilterDeploymentsRequest.
        """
        return self._with_

    @with_.setter
    def with_(self, with_: DeploymentWith):
        """Sets the with_ of this FilterDeploymentsRequest.

        :param with_: The with_ of this FilterDeploymentsRequest.
        """
        self._with_ = with_

    @property
    def environments(self) -> List[str]:
        """Gets the environments of this FilterDeploymentsRequest.

        Deployments must belong to one of these environments.

        :return: The environments of this FilterDeploymentsRequest.
        """
        return self._environments

    @environments.setter
    def environments(self, environments: List[str]):
        """Sets the environments of this FilterDeploymentsRequest.

        Deployments must belong to one of these environments.

        :param environments: The environments of this FilterDeploymentsRequest.
        """
        self._environments = environments

    @property
    def conclusions(self) -> Optional[List[str]]:
        """Gets the conclusions of this FilterDeploymentsRequest.

        Deployments must execute with any of these conclusions.

        :return: The conclusions of this FilterDeploymentsRequest.
        """
        return self._conclusions

    @conclusions.setter
    def conclusions(self, conclusions: Optional[List[str]]):
        """Sets the conclusions of this FilterDeploymentsRequest.

        Deployments must execute with any of these conclusions.

        :param conclusions: The conclusions of this FilterDeploymentsRequest.
        """
        if conclusions is not None:
            for value in conclusions:
                if value not in DeploymentConclusion:
                    raise ValueError(
                        f'Value "{value}" must be one of {list(DeploymentConclusion)}')
        self._conclusions = conclusions


FilterDeploymentsRequest = AllOf(_FilterDeploymentsRequest,
                                 CommonFilterProperties,
                                 make_common_pull_request_filters("pr_"),
                                 CommonDeploymentProperties,
                                 name="FilterDeploymentsRequest", module=__name__)
