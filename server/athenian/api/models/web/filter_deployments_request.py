from typing import Any, Dict, List, Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.common_filter_properties import CommonFilterProperties
from athenian.api.models.web.deployment_conclusion import DeploymentConclusion
from athenian.api.models.web.for_set import make_common_pull_request_filters
from athenian.api.models.web.release_with import ReleaseWith


class _FilterDeploymentsRequest(Model):
    """Filters to select the deployments in `/filter/deployments`."""

    openapi_types = {
        "_in": List[str],
        "_with": ReleaseWith,
        "environments": List[str],
        "conclusions": List[DeploymentConclusion],
        "with_labels": object,
        "without_labels": object,
    }

    attribute_map = {
        "_in": "in",
        "_with": "with",
        "environments": "environments",
        "conclusions": "conclusions",
        "with_labels": "with_labels",
        "without_labels": "without_labels",
    }

    __enable_slots__ = False

    def __init__(
        self,
        _in: Optional[List[str]] = None,
        _with: Optional[ReleaseWith] = None,
        environments: Optional[List[str]] = None,
        conclusions: Optional[List[DeploymentConclusion]] = None,
        with_labels: Optional[Dict[str, Any]] = None,
        without_labels: Optional[Dict[str, Any]] = None,
    ):
        """FilterDeploymentsRequest - a model defined in OpenAPI

        :param _in: The _in of this FilterDeploymentsRequest.
        :param _with: The _with of this FilterDeploymentsRequest.
        :param environments: The environments of this FilterDeploymentsRequest.
        :param conclusions: The conclusions of this FilterDeploymentsRequest.
        :param with_labels: The with_labels of this FilterDeploymentsRequest.
        :param without_labels: The without_labels of this FilterDeploymentsRequest.
        """
        self.__in = _in
        self.__with = _with
        self._environments = environments
        self._conclusions = conclusions
        self._with_labels = with_labels
        self._without_labels = without_labels

    @property
    def _in(self) -> List[str]:
        """Gets the _in of this FilterDeploymentsRequest.

        Set of repositories. An empty list raises a bad response 400. Duplicates are automatically
        ignored.

        :return: The _in of this FilterDeploymentsRequest.
        """
        return self.__in

    @_in.setter
    def _in(self, _in: List[str]):
        """Sets the _in of this FilterDeploymentsRequest.

        Set of repositories. An empty list raises a bad response 400. Duplicates are automatically
        ignored.

        :param _in: The _in of this FilterDeploymentsRequest.
        """
        self.__in = _in

    @property
    def _with(self) -> ReleaseWith:
        """Gets the _with of this FilterDeploymentsRequest.

        :return: The _with of this FilterDeploymentsRequest.
        """
        return self.__with

    @_with.setter
    def _with(self, _with: ReleaseWith):
        """Sets the _with of this FilterDeploymentsRequest.

        :param _with: The _with of this FilterDeploymentsRequest.
        """
        self.__with = _with

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
    def conclusions(self) -> List[DeploymentConclusion]:
        """Gets the conclusions of this FilterDeploymentsRequest.

        Deployments must execute with any of these conclusions.

        :return: The conclusions of this FilterDeploymentsRequest.
        """
        return self._conclusions

    @conclusions.setter
    def conclusions(self, conclusions: List[DeploymentConclusion]):
        """Sets the conclusions of this FilterDeploymentsRequest.

        Deployments must execute with any of these conclusions.

        :param conclusions: The conclusions of this FilterDeploymentsRequest.
        """
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
