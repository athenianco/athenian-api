from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.calculated_linear_metric_values import \
    CalculatedLinearMetricValues
from athenian.api.models.web.for_set_code_checks import ForSetCodeChecks


class CalculatedCodeCheckMetricsItem(Model):
    """Series of calculated metrics for a specific set of repositories and commit authors."""

    openapi_types = {
        "for_": ForSetCodeChecks,
        "granularity": str,
        "check_runs": Optional[int],
        "suites_ratio": Optional[float],
        "values": List[CalculatedLinearMetricValues],
    }

    attribute_map = {
        "for_": "for",
        "granularity": "granularity",
        "check_runs": "check_runs",
        "suites_ratio": "suites_ratio",
        "values": "values",
    }

    def __init__(
        self,
        for_: Optional[ForSetCodeChecks] = None,
        granularity: Optional[str] = None,
        check_runs: Optional[int] = None,
        suites_ratio: Optional[float] = None,
        values: Optional[List[CalculatedLinearMetricValues]] = None,
    ):
        """CalculatedCodeCheckMetricsItem - a model defined in OpenAPI

        :param for_: The for_ of this CalculatedCodeCheckMetricsItem.
        :param granularity: The granularity of this CalculatedCodeCheckMetricsItem.
        :param check_runs: The check_runs of this CalculatedCodeCheckMetricsItem.
        :param suites_ratio: The suites_ratio of this CalculatedCodeCheckMetricsItem.
        :param values: The values of this CalculatedCodeCheckMetricsItem.
        """
        self._for_ = for_
        self._granularity = granularity
        self._check_runs = check_runs
        self._suites_ratio = suites_ratio
        self._values = values

    @property
    def for_(self) -> ForSetCodeChecks:
        """Gets the for_ of this CalculatedCodeCheckMetricsItem.

        :return: The for_ of this CalculatedCodeCheckMetricsItem.
        """
        return self._for_

    @for_.setter
    def for_(self, for_: ForSetCodeChecks):
        """Sets the for_ of this CalculatedCodeCheckMetricsItem.

        :param for_: The for_ of this CalculatedCodeCheckMetricsItem.
        """
        if for_ is None:
            raise ValueError("Invalid value for `for_`, must not be `None`")

        self._for_ = for_

    @property
    def granularity(self) -> str:
        """Gets the granularity of this CalculatedCodeCheckMetricsItem.

        How often the metrics are reported. The value must satisfy the following regular
        expression: /^all|(([1-9]\\d* )?(aligned )?(day|week|month|year))$/. \"all\" produces
        a single interval [`date_from`, `date_to`]. \"aligned week/month/year\" produces intervals
        cut by calendar week/month/year borders, for example, when `date_from` is `2020-01-15` and
        `date_to` is `2020-03-10`, the intervals will be
        `2020-01-15` - `2020-02-01` - `2020-03-01` - `2020-03-10`.

        :return: The granularity of this CalculatedCodeCheckMetricsItem.
        """
        return self._granularity

    @granularity.setter
    def granularity(self, granularity: str):
        """Sets the granularity of this CalculatedCodeCheckMetricsItem.

        How often the metrics are reported. The value must satisfy the following regular
        expression: /^all|(([1-9]\\d* )?(aligned )?(day|week|month|year))$/. \"all\" produces
        a single interval [`date_from`, `date_to`]. \"aligned week/month/year\" produces intervals
        cut by calendar week/month/year borders, for example, when `date_from` is `2020-01-15` and
        `date_to` is `2020-03-10`, the intervals will be
        `2020-01-15` - `2020-02-01` - `2020-03-01` - `2020-03-10`.

        :param granularity: The granularity of this CalculatedCodeCheckMetricsItem.
        """
        self._granularity = granularity

    @property
    def check_runs(self) -> Optional[int]:
        """Gets the check_runs of this CalculatedCodeCheckMetricsItem.

        We calculated metrics for check suites with this number of runs. Not null only if the user
        specified `split_by_check_runs = true`.

        :return: The check_runs of this CalculatedCodeCheckMetricsItem.
        """
        return self._check_runs

    @check_runs.setter
    def check_runs(self, check_runs: Optional[int]):
        """Sets the check_runs of this CalculatedCodeCheckMetricsItem.

        We calculated metrics for check suites with this number of runs. Not null only if the user
        specified `split_by_check_runs = true`.

        :param check_runs: The check_runs of this CalculatedCodeCheckMetricsItem.
        """
        self._check_runs = check_runs

    @property
    def suites_ratio(self) -> Optional[float]:
        """Gets the suites_ratio of this CalculatedCodeCheckMetricsItem.

        Number of check suites with `check_runs` number of check runs divided by the overall number
        of check suites. Not null only if the user specified `split_by_check_runs = true`.

        :return: The suites_ratio of this CalculatedCodeCheckMetricsItem.
        """
        return self._suites_ratio

    @suites_ratio.setter
    def suites_ratio(self, suites_ratio: Optional[float]):
        """Sets the suites_ratio of this CalculatedCodeCheckMetricsItem.

        Number of check suites with `check_runs` number of check runs divided by the overall number
        of check suites. Not null only if the user specified `split_by_check_runs = true`.

        :param suites_ratio: The suites_ratio of this CalculatedCodeCheckMetricsItem.
        """
        if suites_ratio is not None and suites_ratio > 1:
            raise ValueError(
                "Invalid value for `suites_ratio`, must be a value less than or equal to `1`")
        if suites_ratio is not None and suites_ratio < 0:
            raise ValueError(
                "Invalid value for `suites_ratio`, must be a value greater than or equal to `0`")

        self._suites_ratio = suites_ratio

    @property
    def values(self) -> List[CalculatedLinearMetricValues]:
        """Gets the values of this CalculatedCodeCheckMetricsItem.

        The sequence steps from `date_from` till `date_to` by `granularity`.

        :return: The values of this CalculatedCodeCheckMetricsItem.
        """
        return self._values

    @values.setter
    def values(self, values: List[CalculatedLinearMetricValues]):
        """Sets the values of this CalculatedCodeCheckMetricsItem.

        The sequence steps from `date_from` till `date_to` by `granularity`.

        :param values: The values of this CalculatedCodeCheckMetricsItem.
        """
        if values is None:
            raise ValueError("Invalid value for `values`, must not be `None`")

        self._values = values
