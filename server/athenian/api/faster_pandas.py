from datetime import timezone
from functools import lru_cache

from pandas import Series, set_option
from pandas.core.arrays import DatetimeArray, datetimes
from pandas.core.arrays.datetimelike import DatetimeLikeArrayMixin
from pandas.core.base import IndexOpsMixin
from pandas.core.dtypes import common
from pandas.core.dtypes.dtypes import DatetimeTZDtype


def nan_to_none_return(func):
    """Decorate to replace returned NaN-s with None-s."""
    def wrapped_nan_to_none_return(self, *args, **kwargs):
        r = getattr(self, func)(*args, **kwargs)
        if r != r:
            return None
        return r

    return wrapped_nan_to_none_return


def patch_pandas():
    """Patch pandas internals to increase performance on small DataFrame-s."""
    set_option("mode.chained_assignment", "raise")
    IndexOpsMixin.nonemin = nan_to_none_return("min")
    IndexOpsMixin.nonemax = nan_to_none_return("max")

    common.pandas_dtype = lru_cache()(common.pandas_dtype)
    datetimes.pandas_dtype = common.pandas_dtype
    common.is_dtype_equal = lru_cache()(common.is_dtype_equal)
    datetimes.is_dtype_equal = common.is_dtype_equal

    DatetimeTZDtype.utc = DatetimeTZDtype(tz=timezone.utc)

    def cached_utc_new(cls, *args, **kwargs):
        if not args and not kwargs:
            return object.__new__(cls)
        if not args and kwargs == {"tz": timezone.utc}:
            return cls.utc
        obj = object.__new__(cls)
        obj.__init__(*args, **kwargs)
        return obj

    DatetimeTZDtype.__new__ = cached_utc_new

    original_take = DatetimeLikeArrayMixin.take

    def fast_take(self, indices, *, allow_fill=False, fill_value=None, axis: int = 0):
        if len(indices) and indices.min() < 0:
            return original_take(
                self, indices, allow_fill=allow_fill, fill_value=fill_value, axis=axis)
        return original_take(self, indices, allow_fill=False, axis=axis)

    DatetimeLikeArrayMixin.take = fast_take

    original_tz_convert = DatetimeArray.tz_convert

    def fast_tz_convert(self, tz):
        if tz is None:
            return self
        return original_tz_convert(self, tz)

    DatetimeArray.tz_convert = fast_tz_convert

    datetimes._validate_dt64_dtype = lru_cache()(datetimes._validate_dt64_dtype)

    # https://github.com/pandas-dev/pandas/issues/35768
    original_series_take = Series.take

    def safe_take(self, indices, axis=0, is_copy=None, **kwargs) -> Series:
        kwargs.pop("fill_value", None)
        kwargs.pop("allow_fill", None)
        return original_series_take(self, indices, axis=axis, is_copy=is_copy, **kwargs)

    Series.take = safe_take
