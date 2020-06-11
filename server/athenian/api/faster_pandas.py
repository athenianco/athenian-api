from datetime import timezone
from functools import lru_cache

from pandas.core import algorithms
from pandas.core.arrays import datetimes
from pandas.core.arrays.datetimelike import DatetimeLikeArrayMixin
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.internals import Block


def patch_pandas():
    """Patch pandas internals to increase performance on small DataFrame-s."""
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
        return original_take(self, indices, allow_fill=False)

    DatetimeLikeArrayMixin.take = fast_take
    original_ftype_getter = Block.ftype.getter
    ftype_cache = {}

    def fast_ftype(self):
        key = (self.dtype, self._ftype)
        try:
            return ftype_cache[key]
        except KeyError:
            ftype_cache[key] = ftype = original_ftype_getter(self)
            return ftype

    Block.ftype = property(fast_ftype)

    original_get_take_nd_function = algorithms._get_take_nd_function
    cached_get_take_nd_function = lru_cache()(algorithms._get_take_nd_function)

    def _get_take_nd_function(ndim: int, arr_dtype, out_dtype, axis: int = 0, mask_info=None):
        if mask_info is None or not mask_info[1]:
            return cached_get_take_nd_function(ndim, arr_dtype, out_dtype, axis)
        return original_get_take_nd_function(ndim, arr_dtype, out_dtype, axis, mask_info)

    algorithms._get_take_nd_function = _get_take_nd_function

    datetimes._validate_dt64_dtype = lru_cache()(datetimes._validate_dt64_dtype)
