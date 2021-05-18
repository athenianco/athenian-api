from typing import List, Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.common_filter_properties import CommonFilterProperties
from athenian.api.models.web.common_metrics_properties import QuantilesMixin
from athenian.api.models.web.jira_filter import JIRAFilter


class _FilterCodeChecksRequest(Model, QuantilesMixin):
    """Request body of `/filter/code_checks`."""

    openapi_types = {
        "in_": List[str],
        "triggered_by": Optional[List[str]],
        "jira": Optional[JIRAFilter],
        "quantiles": Optional[List[float]],
    }

    attribute_map = {
        "in_": "in",
        "triggered_by": "triggered_by",
        "jira": "jira",
        "quantiles": "quantiles",
    }

    __enable_slots__ = False

    def __init__(
        self,
        in_: Optional[List[str]] = None,
        triggered_by: Optional[List[str]] = None,
        jira: Optional[JIRAFilter] = None,
        quantiles: Optional[List[float]] = None,
    ):
        """FilterCodeChecksRequest - a model defined in OpenAPI

        :param in_: The in_ of this FilterCodeChecksRequest.
        :param triggered_by: The triggered_by of this FilterCodeChecksRequest.
        :param jira: The jira of this FilterCodeChecksRequest.
        :param quantiles: The quantiles of this FilterCodeChecksRequest.
        """
        self._in_ = in_
        self._triggered_by = triggered_by
        self._jira = jira
        self._quantiles = quantiles

    @property
    def in_(self) -> List[str]:
        """Gets the in_ of this FilterCodeChecksRequest.

        Set of repositories. An empty list raises a bad response 400. Duplicates are automatically
        ignored.

        :return: The in_ of this FilterCodeChecksRequest.
        """
        return self._in_

    @in_.setter
    def in_(self, in_: List[str]):
        """Sets the in_ of this FilterCodeChecksRequest.

        Set of repositories. An empty list raises a bad response 400. Duplicates are automatically
        ignored.

        :param in_: The in_ of this FilterCodeChecksRequest.
        """
        if in_ is None:
            raise ValueError("Invalid value for `in_`, must not be `None`")

        self._in_ = in_

    @property
    def triggered_by(self) -> List[str]:
        """Gets the triggered_by of this FilterCodeChecksRequest.

        Check runs must be triggered by commits pushed by these people. When it is impossible
        to determine who pushed, e.g. in legacy API based checks, they are committers.

        :return: The triggered_by of this FilterCodeChecksRequest.
        """
        return self._triggered_by

    @triggered_by.setter
    def triggered_by(self, triggered_by: List[str]):
        """Sets the triggered_by of this FilterCodeChecksRequest.

        Check runs must be triggered by commits pushed by these people. When it is impossible
        to determine who pushed, e.g. in legacy API based checks, they are committers.

        :param triggered_by: The triggered_by of this FilterCodeChecksRequest.
        """
        self._triggered_by = triggered_by

    @property
    def jira(self) -> Optional[JIRAFilter]:
        """Gets the jira of this FilterCodeChecksRequest.

        :return: The jira of this FilterCodeChecksRequest.
        """
        return self._jira

    @jira.setter
    def jira(self, jira: Optional[JIRAFilter]):
        """Sets the jira of this FilterCodeChecksRequest.

        :param jira: The jira of this FilterCodeChecksRequest.
        """
        self._jira = jira


FilterCodeChecksRequest = AllOf(_FilterCodeChecksRequest,
                                CommonFilterProperties,
                                name="FilterCodeChecksRequest",
                                module=__name__)
