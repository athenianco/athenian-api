from datetime import timezone
from functools import lru_cache, wraps
from typing import List, Optional

import numpy as np
from pandas import Series, set_option
from pandas.core import algorithms
from pandas.core.arrays import DatetimeArray, datetimes
from pandas.core.arrays.datetimelike import DatetimeLikeArrayMixin
from pandas.core.base import IndexOpsMixin
from pandas.core.dtypes import common
from pandas.core.dtypes.cast import maybe_cast_to_datetime
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas.core.internals.construction
from pandas.core.internals.construction import DtypeObj, lib, Scalar


def nan_to_none_return(func):
    """Decorate to replace returned NaN-s with None-s."""
    @wraps(func)
    def wrapped_nan_to_none_return(*args, **kwargs):
        r = func(*args, **kwargs)
        if r != r:
            return None
        return r

    return wrapped_nan_to_none_return


def patch_pandas():
    """Patch pandas internals to increase performance on small DataFrame-s."""
    set_option("mode.chained_assignment", "raise")
    obj_dtype = np.dtype("O")

    def _convert_object_array(
            content: List[Scalar], coerce_float: bool = False, dtype: Optional[DtypeObj] = None,
    ) -> List[Scalar]:
        # safe=True avoids converting nullable integers to floats
        def convert(arr):
            if dtype != obj_dtype:
                arr = lib.maybe_convert_objects(arr, try_float=coerce_float, safe=True)
                arr = maybe_cast_to_datetime(arr, dtype)
            return arr

        arrays = [convert(arr) for arr in content]

        return arrays

    pandas.core.internals.construction._convert_object_array = _convert_object_array
    IndexOpsMixin.nonemin = nan_to_none_return(IndexOpsMixin.min)
    IndexOpsMixin.nonemax = nan_to_none_return(IndexOpsMixin.max)

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

    def fast_take(self, indices, allow_fill=False, fill_value=None):
        if len(indices) and indices.min() < 0:
            return original_take(self, indices, allow_fill=allow_fill, fill_value=fill_value)
        return original_take(self, indices, allow_fill=False)

    DatetimeLikeArrayMixin.take = fast_take

    original_tz_convert = DatetimeArray.tz_convert

    def fast_tz_convert(self, tz):
        if tz is None:
            return self
        return original_tz_convert(self, tz)

    DatetimeArray.tz_convert = fast_tz_convert

    original_get_take_nd_function = algorithms._get_take_nd_function
    cached_get_take_nd_function = lru_cache()(algorithms._get_take_nd_function)

    def _get_take_nd_function(ndim: int, arr_dtype, out_dtype, axis: int = 0, mask_info=None):
        if mask_info is None or not mask_info[1]:
            return cached_get_take_nd_function(ndim, arr_dtype, out_dtype, axis)
        return original_get_take_nd_function(ndim, arr_dtype, out_dtype, axis, mask_info)

    algorithms._get_take_nd_function = _get_take_nd_function

    datetimes._validate_dt64_dtype = lru_cache()(datetimes._validate_dt64_dtype)

    # https://github.com/pandas-dev/pandas/issues/35768
    original_series_take = Series.take

    def safe_take(self, indices, axis=0, is_copy=None, **kwargs) -> Series:
        kwargs.pop("fill_value", None)
        kwargs.pop("allow_fill", None)
        return original_series_take(self, indices, axis=axis, is_copy=is_copy, **kwargs)

    Series.take = safe_take
