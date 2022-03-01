from typing import List, Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.common_metrics_properties import QuantilesMixin
from athenian.api.models.web.filter_jira_common import FilterJIRACommon
from athenian.api.models.web.jira_filter_with import JIRAFilterWith
from athenian.api.models.web.jira_histogram_definition import JIRAHistogramDefinition


class _JIRAHistogramsRequest(Model, QuantilesMixin, sealed=False):
    """Request of `/histograms/jira`."""

    openapi_types = {
        "epics": Optional[List[str]],
        "with_": Optional[List[JIRAFilterWith]],
        "histograms": List[JIRAHistogramDefinition],
        "quantiles": Optional[List[float]],
    }

    attribute_map = {
        "epics": "epics",
        "with_": "with",
        "histograms": "histograms",
        "quantiles": "quantiles",
    }

    def __init__(
        self,
        epics: Optional[List[str]] = None,
        with_: Optional[List[JIRAFilterWith]] = None,
        histograms: Optional[List[JIRAHistogramDefinition]] = None,
        quantiles: Optional[List[float]] = None,
    ):
        """JIRAHistogramsRequest - a model defined in OpenAPI

        :param epics: The epics of this JIRAHistogramsRequest.
        :param with_: The with of this JIRAHistogramsRequest.
        :param histograms: The histograms of this JIRAHistogramsRequest.
        :param quantiles: The quantiles of this JIRAHistogramsRequest.
        """
        self._epics = epics
        self._with_ = with_
        self._histograms = histograms
        self._quantiles = quantiles

    @property
    def epics(self) -> Optional[List[str]]:
        """Gets the epics of this JIRAHistogramsRequest.

        Selected issue epics.

        :return: The epics of this JIRAHistogramsRequest.
        """
        return self._epics

    @epics.setter
    def epics(self, epics: Optional[List[str]]):
        """Sets the epics of this JIRAHistogramsRequest.

        Selected issue epics.

        :param epics: The epics of this JIRAHistogramsRequest.
        """
        self._epics = epics

    @property
    def with_(self) -> Optional[List[JIRAFilterWith]]:
        """Gets the with_ of this JIRAHistogramsRequest.

        Groups of issue participants. The histograms will be calculated for each group.

        :return: The with_ of this JIRAHistogramsRequest.
        """
        return self._with_

    @with_.setter
    def with_(self, with_: Optional[List[JIRAFilterWith]]):
        """Sets the with_ of this JIRAHistogramsRequest.

        Groups of issue participants. The histograms will be calculated for each group.

        :param with_: The with_ of this JIRAHistogramsRequest.
        """
        self._with_ = with_

    @property
    def histograms(self) -> List[JIRAHistogramDefinition]:
        """Gets the histograms of this JIRAHistogramsRequest.

        Histogram parameters for each wanted topic.

        :return: The histograms of this JIRAHistogramsRequest.
        """
        return self._histograms

    @histograms.setter
    def histograms(self, histograms: List[JIRAHistogramDefinition]):
        """Sets the histograms of this JIRAHistogramsRequest.

        Histogram parameters for each wanted topic.

        :param histograms: The histograms of this JIRAHistogramsRequest.
        """
        if histograms is None:
            raise ValueError("Invalid value for `histograms`, must not be `None`")
        self._histograms = histograms


JIRAHistogramsRequest = AllOf(_JIRAHistogramsRequest, FilterJIRACommon,
                              name="JIRAHistogramsRequest", module=__name__)
