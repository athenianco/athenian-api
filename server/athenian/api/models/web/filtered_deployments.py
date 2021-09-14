from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.filtered_deployment import FilteredDeployment
from athenian.api.models.web.release_set import ReleaseSetInclude


class FilteredDeployments(Model):
    """Found deployments, response from `/filter/deployments`."""

    openapi_types = {
        "include": ReleaseSetInclude,
        "deployments": List[FilteredDeployment],
    }

    attribute_map = {"include": "include", "deployments": "deployments"}

    def __init__(
        self,
        include: Optional[ReleaseSetInclude] = None,
        deployments: Optional[List[FilteredDeployment]] = None,
    ):
        """FilteredDeployments - a model defined in OpenAPI

        :param include: The include of this FilteredDeployments.
        :param deployments: The deployments of this FilteredDeployments.
        """
        self._include = include
        self._deployments = deployments

    @property
    def include(self) -> ReleaseSetInclude:
        """Gets the include of this FilteredDeployments.

        All people and JIRA issues mentioned in the deployments.

        :return: The include of this FilteredDeployments.
        """
        return self._include

    @include.setter
    def include(self, include: ReleaseSetInclude):
        """Sets the include of this FilteredDeployments.

        All people and JIRA issues mentioned in the deployments.

        :param include: The include of this FilteredDeployments.
        """
        if include is None:
            raise ValueError("Invalid value for `include`, must not be `None`")

        self._include = include

    @property
    def deployments(self) -> List[FilteredDeployment]:
        """Gets the deployments of this FilteredDeployments.

        List of matched deployments.

        :return: The deployments of this FilteredDeployments.
        """
        return self._deployments

    @deployments.setter
    def deployments(self, deployments: List[FilteredDeployment]):
        """Sets the deployments of this FilteredDeployments.

        List of matched deployments.

        :param deployments: The deployments of this FilteredDeployments.
        """
        if deployments is None:
            raise ValueError("Invalid value for `deployments`, must not be `None`")

        self._deployments = deployments
