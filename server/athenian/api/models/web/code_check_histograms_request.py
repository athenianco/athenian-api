from typing import List, Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.code_check_histogram_definition import \
    CodeCheckHistogramDefinition
from athenian.api.models.web.common_filter_properties import CommonFilterProperties
from athenian.api.models.web.common_metrics_properties import QuantilesMixin
from athenian.api.models.web.for_set_code_checks import ForSetCodeChecks


class _CodeCheckHistogramsRequest(Model, QuantilesMixin):
    """Request of `/histograms/code_checks`."""

    openapi_types = {
        "for_": List[ForSetCodeChecks],
        "histograms": List[CodeCheckHistogramDefinition],
        "quantiles": List[float],
    }

    attribute_map = {
        "for_": "for",
        "histograms": "histograms",
        "quantiles": "quantiles",
    }

    __enable_slots__ = False

    def __init__(
        self,
        for_: Optional[List[ForSetCodeChecks]] = None,
        histograms: Optional[List[CodeCheckHistogramDefinition]] = None,
        quantiles: Optional[List[float]] = None,
    ):
        """CodeCheckHistogramsRequest - a model defined in OpenAPI

        :param for_: The for_ of this CodeCheckHistogramsRequest.
        :param histograms: The histograms of this CodeCheckHistogramsRequest.
        :param quantiles: The quantiles of this CodeCheckHistogramsRequest.
        """
        self._for_ = for_
        self._histograms = histograms
        self._quantiles = quantiles

    @property
    def for_(self) -> List[ForSetCodeChecks]:
        """Gets the for_ of this CodeCheckHistogramsRequest.

        Sets of developers and repositories for which to calculate the histograms. The aggregation
        is `AND` between repositories and developers. The aggregation is `OR` inside both
        repositories and developers.

        :return: The for_ of this CodeCheckHistogramsRequest.
        """
        return self._for_

    @for_.setter
    def for_(self, for_: List[ForSetCodeChecks]):
        """Sets the for_ of this CodeCheckHistogramsRequest.

        Sets of developers and repositories for which to calculate the histograms. The aggregation
        is `AND` between repositories and developers. The aggregation is `OR` inside both
        repositories and developers.

        :param for_: The for_ of this CodeCheckHistogramsRequest.
        """
        if for_ is None:
            raise ValueError("Invalid value for `for_`, must not be `None`")

        self._for_ = for_

    @property
    def histograms(self) -> List[CodeCheckHistogramDefinition]:
        """Gets the histograms of this CodeCheckHistogramsRequest.

        Histogram parameters for each wanted topic.

        :return: The histograms of this CodeCheckHistogramsRequest.
        """
        return self._histograms

    @histograms.setter
    def histograms(self, histograms: List[CodeCheckHistogramDefinition]):
        """Sets the histograms of this CodeCheckHistogramsRequest.

        Histogram parameters for each wanted topic.

        :param histograms: The histograms of this CodeCheckHistogramsRequest.
        """
        if histograms is None:
            raise ValueError("Invalid value for `histograms`, must not be `None`")

        self._histograms = histograms


CodeCheckHistogramsRequest = AllOf(_CodeCheckHistogramsRequest,
                                   CommonFilterProperties,
                                   name="CodeCheckHistogramsRequest",
                                   module=__name__)
