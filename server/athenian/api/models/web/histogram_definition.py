from datetime import timedelta
from typing import Any, List, Optional, Type

from athenian.api.models.web.base_model_ import Enum, Model
from athenian.api.models.web.histogram_scale import HistogramScale
from athenian.api.serialization import deserialize_timedelta


def HistogramDefinition(M: Enum, module_name: str) -> Type[Model]:
    """
    Create a new "HistogramDefinition" class with the given metric type.

    :param M: Metric enum.
    :param module_name: Place the created class in this module, set it to `__name__`.
    """

    class HistogramDefinition(Model):
        """Histogram parameters for each measured topic."""

        metric: M
        scale: Optional[HistogramScale]
        bins: Optional[int]
        ticks: Optional[List[object]]

        def validate_scale(self, scale: Optional[str]) -> Optional[str]:
            """Sets the scale of this HistogramDefinition.

            Histogram's X axis scale.

            :param scale: The scale of this HistogramDefinition.
            """
            if scale is not None and scale not in HistogramScale:
                raise ValueError('"scale" must be one of %s' % list(HistogramScale))

            return scale

        def validate_bins(self, bins: Optional[int]) -> Optional[int]:
            """Sets the bins of this HistogramDefinition.

            Number of bars in the histogram. 0 or null means automatic.

            :param bins: The bins of this HistogramDefinition.
            """
            if bins is not None and bins < 0:
                raise ValueError(
                    "Invalid value for `bins`, must be a value greater than or equal to `0`",
                )

            return bins

        def validate_ticks(self, ticks: Optional[list[Any]]) -> Optional[list[Any]]:
            """Sets the ticks of this HistogramDefinition.

            Alternatively to `bins` and `scale`, set the X axis bar borders manually.
            Only one of two may be specified. The ticks are automatically prepended
            the distribution minimum and appended the distribution maximum.

            :param ticks: The ticks of this HistogramDefinition.
            """
            if ticks is not None and len(ticks) == 0:
                raise ValueError("`ticks` must contain at least one element")

            if ticks and isinstance(ticks[0], str):
                parsed_ticks = []
                max_tick = timedelta(days=365 * 2)
                for tick in ticks:
                    parsed_tick = deserialize_timedelta(tick)
                    if parsed_tick > max_tick:
                        raise ValueError(f"Too big timedelta {tick}, the max is {max_tick}")
                    parsed_ticks.append(parsed_tick)
                ticks = parsed_ticks

            return ticks

    HistogramDefinition.__module__ = module_name
    return HistogramDefinition
