from datetime import timezone

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
    ftype_cache = {}
    original_ftype_getter = Block.ftype.getter

    def fast_ftype(self):
        if getattr(self.values, "_pandas_ftype", False):
            dtype = self.dtype.subtype
        else:
            dtype = self.dtype
        try:
            return ftype_cache[(dtype, self._ftype)]
        except KeyError:
            ftype_cache[(dtype, self._ftype)] = r = original_ftype_getter(self)
            return r

    Block.ftype = property(fast_ftype)
