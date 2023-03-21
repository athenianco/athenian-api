"""Common tools for the search controllers."""
import abc
from datetime import datetime
from typing import Sequence, Type, TypeVar

import medvedi as md
import numpy as np
import numpy.typing as npt

from athenian.api.internal.features.metric_calculator import (
    DEFAULT_QUANTILE_STRIDE,
    MetricCalculatorEnsemble,
)
from athenian.api.models.web import OrderByDirection, OrderByExpression


class OrderBy(metaclass=abc.ABCMeta):
    """Handles the order by expressions."""

    @abc.abstractmethod
    def apply_expression(
        self,
        expr: OrderByExpression,
        current_indexes: npt.NDArray[int],
    ) -> tuple[npt.NDArray[int], npt.NDArray[int]]:
        """Parse an expression and return a tuple with ordered indexes and discard indexes.

        `current_indexes` is the current order of the pull request facts.
        Returns:
        - the new ordered indexes to select sorted items
        - an array with the indexes of element to discard from end result due to expression.

        """

    @classmethod
    def _ordered_indexes(
        cls,
        expr: OrderByExpression,
        current_indexes: npt.NDArray[int],
        values: npt.NDArray,
        nulls: npt.NDArray[int],
    ) -> npt.NDArray[int]:
        """Return the new ordered indexes using the given values and nulls indexes."""
        notnulls_values = values[~nulls]
        if expr.direction == OrderByDirection.DESCENDING.value:
            notnulls_values = cls._negate_values(notnulls_values)

        indexes_notnull = current_indexes[~nulls][np.argsort(notnulls_values, kind="stable")]

        res_parts = [indexes_notnull, current_indexes[nulls]]
        if expr.nulls_first:
            res_parts = res_parts[::-1]
        result = np.concatenate(res_parts)
        return result

    @classmethod
    def _discard_mask(cls, expr: OrderByExpression, nulls: npt.NDArray[int]) -> npt.NDArray[int]:
        if len(nulls) and expr.exclude_nulls:
            return np.flatnonzero(nulls)
        else:
            return np.array([], dtype=int)

    @classmethod
    def _negate_values(cls, values: npt.NDArray) -> npt.NDArray:
        return -values


class OrderByMetrics(OrderBy):
    """Handles order by metric values.

    Expressions about a metric field can be fed into this object with `apply_expression`.
    """

    def __init__(self, calc_ensemble: MetricCalculatorEnsemble):
        """Init the `OrderByMetrics` which will use the given the metrics calculator."""
        self._calc_ensemble = calc_ensemble

    def apply_expression(
        self,
        expr: OrderByExpression,
        current_indexes: npt.NDArray[int],
    ) -> tuple[npt.NDArray[int], npt.NDArray[int]]:
        """Apply the order by expression."""
        calc = self._calc_ensemble[expr.field][0]
        values = calc.peek[0][current_indexes]

        if calc.has_nan:
            nulls = values != values
        else:
            nulls = values == calc.nan
        ordered_indexes = self._ordered_indexes(expr, current_indexes, values, nulls)
        discard = self._discard_mask(expr, nulls)
        return ordered_indexes, discard


class OrderByValues(OrderBy):
    """handles order by values extract through a custom function."""

    def apply_expression(
        self,
        expr: OrderByExpression,
        current_indexes: npt.NDArray[int],
    ) -> tuple[npt.NDArray[int], npt.NDArray[int]]:
        """Apply the order by expression."""
        values = self._get_values(expr)[current_indexes]
        nulls = values != values
        discard = self._discard_mask(expr, nulls)
        ordered_indexes = self._ordered_indexes(expr, current_indexes, values, nulls)
        return ordered_indexes, discard

    @abc.abstractmethod
    def _get_values(self, expr: OrderByExpression) -> npt.NDArray[np.datetime64]:
        """Extract the values to use in ordering."""


CalcualtorT = TypeVar("CalcualtorT", bound=MetricCalculatorEnsemble)


def build_metrics_calculator_ensemble(
    facts: md.DataFrame,
    metrics: Sequence[str],
    time_from: datetime,
    time_to: datetime,
    Calculator: Type[CalcualtorT],
) -> CalcualtorT | None:
    """Build the MetricCalculatorEnsemble for the given metrics.

    The metrics calculation will also be executed.
    """
    if not metrics:
        return None
    min_times, max_times = (
        np.array([t.replace(tzinfo=None)], dtype="datetime64[us]") for t in (time_from, time_to)
    )
    calc_ensemble = Calculator(*metrics, quantiles=(0, 1), quantile_stride=DEFAULT_QUANTILE_STRIDE)
    groups = np.full((1, len(facts)), True, bool)
    calc_ensemble(facts, min_times, max_times, groups)
    return calc_ensemble
