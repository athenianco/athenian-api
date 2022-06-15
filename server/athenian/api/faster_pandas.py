from datetime import timezone
from functools import lru_cache, wraps
from typing import List, Optional
import warnings

import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from pandas import DataFrame, Index, MultiIndex, Series, set_option
    from pandas.core import algorithms
    from pandas.core.arrays import DatetimeArray, datetimes
    from pandas.core.arrays.datetimelike import DatetimeLikeArrayMixin
    from pandas.core.base import IndexOpsMixin
    from pandas.core.dtypes import common
    from pandas.core.dtypes.cast import maybe_cast_to_datetime
    from pandas.core.dtypes.common import is_dtype_equal
    from pandas.core.dtypes.dtypes import DatetimeTZDtype
    from pandas.core.dtypes.inference import is_array_like
    from pandas.core.dtypes.missing import na_value_for_dtype
    import pandas.core.internals.construction
    from pandas.core.internals.construction import DtypeObj, Scalar, lib
    from pandas.core.reshape.merge import _MergeOperation, _should_fill


def nan_to_none_return(func):
    """Decorate to replace returned NaN-s with None-s."""

    @wraps(func)
    def wrapped_nan_to_none_return(*args, **kwargs):
        r = func(*args, **kwargs)
        if r != r:
            return None
        return r

    return wrapped_nan_to_none_return


def _disable_consolidate(self):
    self._consolidate_inplace = lambda: None
    self._mgr._consolidate_inplace = lambda: None


def patch_pandas():
    """
    Patch pandas internals to increase performance on small DataFrame-s.

    Look: Pandas sucks. I mean it. Every minor release breaks the public API, performance is awful,
    maintainers are ignorant, etc. But we don't have an alternative given our human resources.

    So:
    - Patch certain functions to improve the performance for our use-cases.
    - Backport bugs.
    - Dream about a better package with a similar API.
    """
    set_option("mode.chained_assignment", "raise")
    obj_dtype = np.dtype("O")

    DataFrame.disable_consolidate = _disable_consolidate

    # not required for 1.3.0+
    # backport https://github.com/pandas-dev/pandas/pull/34414
    _MergeOperation._maybe_add_join_keys = _maybe_add_join_keys

    def _convert_object_array(
        content: List[Scalar],
        coerce_float: bool = False,
        dtype: Optional[DtypeObj] = None,
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


def _maybe_add_join_keys(self, result, left_indexer, right_indexer):

    left_has_missing = None
    right_has_missing = None

    keys = zip(self.join_names, self.left_on, self.right_on)
    for i, (name, lname, rname) in enumerate(keys):
        if not _should_fill(lname, rname):
            continue

        take_left, take_right = None, None

        if name in result:

            if left_indexer is not None and right_indexer is not None:
                if name in self.left:

                    if left_has_missing is None:
                        left_has_missing = (left_indexer == -1).any()

                    if left_has_missing:
                        take_right = self.right_join_keys[i]

                        if not is_dtype_equal(result[name].dtype, self.left[name].dtype):
                            take_left = self.left[name]._values

                elif name in self.right:

                    if right_has_missing is None:
                        right_has_missing = (right_indexer == -1).any()

                    if right_has_missing:
                        take_left = self.left_join_keys[i]

                        if not is_dtype_equal(result[name].dtype, self.right[name].dtype):
                            take_right = self.right[name]._values

        elif left_indexer is not None and is_array_like(self.left_join_keys[i]):
            take_left = self.left_join_keys[i]
            take_right = self.right_join_keys[i]

        if take_left is not None or take_right is not None:

            if take_left is None:
                lvals = result[name]._values
            else:
                lfill = na_value_for_dtype(take_left.dtype)
                lvals = algorithms.take_1d(take_left, left_indexer, fill_value=lfill)

            if take_right is None:
                rvals = result[name]._values
            else:
                rfill = na_value_for_dtype(take_right.dtype)
                rvals = algorithms.take_1d(take_right, right_indexer, fill_value=rfill)

            # if we have an all missing left_indexer
            # make sure to just use the right values
            mask = left_indexer == -1
            if mask.all():
                key_col = Index(rvals)  # <<< https://github.com/pandas-dev/pandas/pull/34414
            else:
                key_col = Index(lvals).where(~mask, rvals)

            if result._is_label_reference(name):
                result[name] = key_col
            elif result._is_level_reference(name):
                if isinstance(result.index, MultiIndex):
                    key_col.name = name
                    idx_list = [
                        result.index.get_level_values(level_name)
                        if level_name != name
                        else key_col
                        for level_name in result.index.names
                    ]

                    result.set_index(idx_list, inplace=True)
                else:
                    result.index = Index(key_col, name=name)
            else:
                result.insert(i, name or f"key_{i}", key_col)
