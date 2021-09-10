from typing import Any, Dict, List, Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.common_filter_properties import CommonFilterProperties
from athenian.api.models.web.deployment_conclusion import DeploymentConclusion
from athenian.api.models.web.for_set import make_common_pull_request_filters
from athenian.api.models.web.release_with import ReleaseWith


class _FilterDeploymentsRequest(Model):
    """Filters to select the deployments in `/filter/deployments`."""

    openapi_types = {
        "in_": List[str],
        "with_": ReleaseWith,
        "environments": List[str],
        "conclusions": List[str],
        "with_labels": object,
        "without_labels": object,
    }

    attribute_map = {
        "in_": "in",
        "with_": "with",
        "environments": "environments",
        "conclusions": "conclusions",
        "with_labels": "with_labels",
        "without_labels": "without_labels",
    }

    __enable_slots__ = False

    def __init__(
        self,
        in_: Optional[List[str]] = None,
        with_: Optional[ReleaseWith] = None,
        environments: Optional[List[str]] = None,
        conclusions: Optional[List[str]] = None,
        with_labels: Optional[Dict[str, Any]] = None,
        without_labels: Optional[Dict[str, Any]] = None,
    ):
        """FilterDeploymentsRequest - a model defined in OpenAPI

        :param in_: The in_ of this FilterDeploymentsRequest.
        :param with_: The with_ of this FilterDeploymentsRequest.
        :param environments: The environments of this FilterDeploymentsRequest.
        :param conclusions: The conclusions of this FilterDeploymentsRequest.
        :param with_labels: The with_labels of this FilterDeploymentsRequest.
        :param without_labels: The without_labels of this FilterDeploymentsRequest.
        """
        self._in_ = in_
        self._with_ = with_
        self._environments = environments
        self._conclusions = conclusions
        self._with_labels = with_labels
        self._without_labels = without_labels

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
    def with_(self) -> ReleaseWith:
        """Gets the with_ of this FilterDeploymentsRequest.

        :return: The with_ of this FilterDeploymentsRequest.
        """
        return self._with_

    @with_.setter
    def with_(self, with_: ReleaseWith):
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

    @property
    def with_labels(self) -> Dict[str, Any]:
        """Gets the with_labels of this FilterDeploymentsRequest.

        Deployments should contain at least one of the specified label values. `null` matches any
        label value and effectively checks the label existence.

        :return: The with_labels of this FilterDeploymentsRequest.
        """
        return self._with_labels

    @with_labels.setter
    def with_labels(self, with_labels: Dict[str, Any]):
        """Sets the with_labels of this FilterDeploymentsRequest.

        Deployments should contain at least one of the specified label values. `null` matches any
        label value and effectively checks the label existence.

        :param with_labels: The with_labels of this FilterDeploymentsRequest.
        """
        self._with_labels = with_labels

    @property
    def without_labels(self) -> Dict[str, Any]:
        """Gets the without_labels of this FilterDeploymentsRequest.

        Deployments may not contain the specified label values. `null` matches any label value and
        effectively ensures that the label does not exist.

        :return: The without_labels of this FilterDeploymentsRequest.
        """
        return self._without_labels

    @without_labels.setter
    def without_labels(self, without_labels: Dict[str, Any]):
        """Sets the without_labels of this FilterDeploymentsRequest.

        Deployments may not contain the specified label values. `null` matches any label value and
        effectively ensures that the label does not exist.

        :param without_labels: The without_labels of this FilterDeploymentsRequest.
        """
        self._without_labels = without_labels


FilterDeploymentsRequest = AllOf(_FilterDeploymentsRequest,
                                 CommonFilterProperties,
                                 make_common_pull_request_filters("pr_"),
                                 name="FilterDeploymentsRequest", module=__name__)
