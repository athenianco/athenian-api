import dataclasses
from datetime import datetime, timedelta
from itertools import chain
import types
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    KeysView,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

import numpy as np
from numpy import typing as npt
import pandas as pd
from pandas.core.dtypes.cast import tslib
from pandas.core.internals.blocks import (
    Block,
    _extract_bool_array,
    get_block_type as get_block_type_original,
    lib as blocks_lib,
    make_block as make_block_original,
)
from pandas.core.internals.managers import BlockManager
import sentry_sdk

from athenian.api.to_object_arrays import array_from_buffer, is_not_null
from athenian.api.tracing import sentry_span

OriginalSpecialForm = type(Any)


class _VerbatimUnion(OriginalSpecialForm, _root=True):
    __slots__ = ("__verbatim__",)

    def __init__(self):
        self._getitem = Union._getitem
        self._name = Union._name
        self.__verbatim__ = True


VerbatimUnion = _VerbatimUnion()


@OriginalSpecialForm
def VerbatimOptional(self, parameters):
    """Alternative Optional that prevents coercing (), [], and {} attributes to null during \
    serialization."""
    return VerbatimUnion[Optional[parameters].__args__]


VerbatimUnionTypes = (VerbatimUnion, Union)


def is_generic(klass: type):
    """Determine whether klass is a generic class."""
    # UnionType sadly doesn't define __origin__
    return hasattr(klass, "__origin__") or isinstance(klass, types.UnionType)


def is_dict(klass: type):
    """Determine whether klass is a dict."""
    return getattr(klass, "__origin__", None) == dict


def is_list(klass: type):
    """Determine whether klass is a list."""
    return getattr(klass, "__origin__", None) == list


def is_union(klass: object) -> bool:
    """Determine whether klass is a Union."""
    return (
        # old style typing.Union[T0, T1]
        getattr(klass, "__origin__", None) in VerbatimUnionTypes
        # new style T0 | T1
        or isinstance(klass, types.UnionType)
    )


def is_optional(klass: type):
    """Determine whether klass is an Optional."""
    return (
        is_union(klass) and len(klass.__args__) == 2 and issubclass(klass.__args__[1], type(None))
    )


def wraps(wrapper, wrappee):
    """Alternative to functools.wraps() for async functions."""  # noqa: D402
    wrapper.__name__ = wrappee.__name__
    wrapper.__qualname__ = wrappee.__qualname__
    wrapper.__module__ = wrappee.__module__
    wrapper.__doc__ = wrappee.__doc__
    wrapper.__annotations__ = wrappee.__annotations__
    wrapper.__wrapped__ = wrappee
    return wrapper


NST = TypeVar("NST")


class NumpyStruct(Mapping[str, Any]):
    """
    Constrained dataclass based on numpy structured array.

    We divide the fields into two groups: mutable and immutable.
    The mutable fields are stored as regular class members and discarded from serialization.
    The immutable fields are not materialized explicitly. Instead, they are taken from numpy
    structured array (`_arr`) that references an arbitrary memory buffer (`_data`).
    Serialization of the class is as simple as exposing the underlying memory buffer outside.

    We support variable-length sub-arrays using the special notation `[<array dtype>]`. That way
    the arrays are appended to `_data`, and `_arr` points to them by pairs (offset, length).
    """

    dtype: np.dtype
    nested_dtypes: Mapping[str, np.dtype]

    def __init__(self, data: Union[bytes, bytearray, memoryview, np.ndarray], **optional: Any):
        """Initialize a new instance of NumpyStruct from raw memory and the (perhaps incomplete) \
        mapping of mutable field values."""
        if isinstance(data, (np.ndarray, np.void)):
            assert data.shape == () or data.shape == (1,)
            data = data.reshape(1)
            self._data = data.view(np.uint8).data
            self._arr = data
        else:
            self._data = data
            self._arr = None
        for attr in self.__slots__[2:]:
            setattr(self, attr, optional.get(attr))

    @classmethod
    def from_fields(cls: NST, **kwargs: Any) -> NST:
        """Initialize a new instance of NumpyStruct from the dict of [im]mutable field values."""
        arr = np.zeros(1, cls.dtype)
        extra_bytes = []
        offset = cls.dtype.itemsize
        for field_name, (field_dtype, _) in cls.dtype.fields.items():
            value = kwargs.pop(field_name)
            try:
                nested_dtype = cls.nested_dtypes[field_name]
            except KeyError:
                dtype_char = field_dtype.char
                if value is None and (dtype_char == "S" or dtype_char == "U"):
                    value = ""
                if dtype_char == "M" and isinstance(value, datetime):
                    value = value.replace(tzinfo=None)
                arr[field_name] = np.asarray(value, field_dtype)
            else:
                if is_str := (
                    (is_ascii := _dtype_is_ascii(nested_dtype)) or nested_dtype.char in ("S", "U")
                ):
                    if isinstance(value, np.ndarray):
                        if value.dtype == np.dtype(object):
                            nan_mask = value == np.array([None])
                        else:
                            nan_mask = np.full(len(value), False)
                    else:
                        nan_mask = np.fromiter(
                            (v is None for v in value), dtype=np.bool_, count=len(value),
                        )
                    if is_ascii:
                        nested_dtype = np.dtype("S")
                value = np.asarray(value, nested_dtype)
                assert len(value.shape) == 1, "we don't support arrays of more than 1 dimension"
                if is_str and nan_mask.any():
                    if not value.flags.writeable:
                        value = value.copy()
                    value[nan_mask] = ""
                extra_bytes.append(data := value.view(np.byte).data)
                pointer = [offset, len(value)]
                if is_str and (is_ascii or nested_dtype.itemsize == 0):
                    pointer.append(
                        value.dtype.itemsize // np.dtype(nested_dtype.char + "1").itemsize,
                    )
                arr[field_name] = pointer
                offset += len(data)
        main_view = arr.view(np.byte).data
        if not extra_bytes:
            return cls(main_view)
        return cls(b"".join((main_view, *extra_bytes)), **kwargs)

    @property
    def data(self) -> bytes:
        """Return the underlying memory."""
        return self._data

    @property
    def array(self) -> np.ndarray:
        """Return the underlying numpy array that wraps `data`."""
        if self._arr is None:
            self._arr = array_from_buffer(self.data, self.dtype, count=1)
        return self._arr

    @property
    def coerced_data(self) -> memoryview:
        """Return prefix of `data` with nested immutable objects excluded."""
        return memoryview(self.data)[: self.dtype.itemsize]

    def __getitem__(self, item: str) -> Any:
        """Implement self[]."""
        return getattr(self, item)

    def __setitem__(self, key: str, value: Any) -> None:
        """Implement self[] = ..."""
        setattr(self, key, value)

    def __len__(self) -> int:
        """Implement len()."""
        return len(self.dtype) + len(self.__slots__) - 2

    def __iter__(self) -> Iterator[str]:
        """Implement iter()."""
        return iter(chain(self.dtype.names, self.__slots__[2:]))

    def __hash__(self) -> int:
        """Implement hash()."""
        return hash(self._data)

    def __str__(self) -> str:
        """Format for human-readability."""
        return "{\n\t%s\n}" % ",\n\t".join("%s: %s" % (k, v) for k, v in self.items())

    def __repr__(self) -> str:
        """Implement repr()."""
        kwargs = {k: v for k in self.__slots__[2:] if (v := getattr(self, k)) is not None}
        if kwargs:
            kwargs_str = ", ".join(f"{k}={repr(v)}" for k, v in kwargs.items()) + ", "
        else:
            kwargs_str = ""
        return f"{type(self).__name__}({kwargs_str}data={repr(bytes(self._data))})"

    def __sentry_repr__(self) -> str:
        """Override {}.__repr__() in Sentry."""
        return str(self)

    def __eq__(self, other) -> bool:
        """Compare this object to another."""
        if self is other:
            return True

        if self.__class__ is not other.__class__:
            raise NotImplementedError(f"Cannot compare {self.__class__} and {other.__class__}")

        return self.data == other.data

    def __getstate__(self) -> dict[str, Any]:
        """Support pickle.dump()."""
        data = self.data
        return {
            "data": bytes(data) if not isinstance(data, (bytes, bytearray)) else data,
            **{attr: getattr(self, attr) for attr in self.__slots__[2:]},
        }

    def __setstate__(self, state: dict[str, Any]):
        """Support pickle.load()."""
        self.__init__(**state)

    def copy(self) -> "NumpyStruct":
        """Clone the instance."""
        return type(self)(self.data, **{attr: getattr(self, attr) for attr in self.__slots__[2:]})

    @staticmethod
    def _generate_get(
        name: str,
        type_: Union[str, np.dtype, list[Union[str, np.dtype]]],
    ) -> Callable[["NumpyStruct"], Any]:
        if _dtype_is_ascii(type_):
            type_ = str
        elif isinstance(type_, list):
            type_ = np.ndarray
        elif (char := np.dtype(type_).char) == "U":
            type_ = np.str_
        elif char == "S":
            type_ = np.bytes_
        elif char == "V":
            type_ = np.ndarray

        def get_field(self) -> Optional[type_]:
            if self._arr is None:
                self._arr = array_from_buffer(self.data, self.dtype, count=1)
            value = self._arr[name][0]
            if (nested_dtype := self.nested_dtypes.get(name)) is None:
                if value != value:
                    return None
                if type_ is str:
                    value = value.decode()
                if type_ in (str, np.str_):
                    value = value or None
                return value
            if (_dtype_is_ascii(nested_dtype) and (char := "S")) or (
                (((char := nested_dtype.char) == "S") or (char == "U"))
                and nested_dtype.itemsize == 0
            ):
                offset, count, itemsize = value[0], value[1], value[2]  # performance
                nested_dtype = np.dtype(f"{char}{itemsize}")
            else:
                offset, count = value[0], value[1]  # performance
            return array_from_buffer(self.data, nested_dtype, count=count, offset=offset)

        get_field.__name__ = name
        return get_field


def _dtype_is_ascii(dtype: Union[str, np.dtype]) -> bool:
    return (dtype is ascii) or (isinstance(dtype, str) and dtype.startswith("ascii"))


def numpy_struct(cls: type) -> Type[NumpyStruct]:
    """
    Decorate a class to transform it to a NumpyStruct.

    The decorated class must define two sub-classes: `dtype` and `optional`.
    The former annotates numpy-friendly immutable fields. The latter annotates mutable fields.
    """
    dtype = cls.Immutable.__annotations__
    dtype_tuples = []
    nested_dtypes = {}
    for k, v in dtype.items():
        if isinstance(v, list):
            assert len(v) == 1, "Array must be specified as `[dtype]`."
            nested_dtype = v[0]
            if not (is_ascii := _dtype_is_ascii(nested_dtype)):
                nested_dtype = np.dtype(nested_dtype)
            nested_dtypes[k] = nested_dtype
            if is_ascii or (nested_dtype.char in ("S", "U") and nested_dtype.itemsize == 0):
                # save the characters count
                dtype_tuples.append((k, np.int32, 3))
            else:
                dtype_tuples.append((k, np.int32, 2))
        elif _dtype_is_ascii(v):
            dtype_tuples.append((k, "S" + v[6:-1]))
        else:
            dtype_tuples.append((k, v))
    try:
        optional = cls.Optional.__annotations__
    except AttributeError:
        optional = {}
    field_names = NamedTuple(
        f"{cls.__name__}FieldNames",
        [(k, str) for k in chain(dtype, optional)],
    )(*chain(dtype, optional))
    base = type(
        cls.__name__ + "Base",
        (NumpyStruct,),
        {k: property(NumpyStruct._generate_get(k, v)) for k, v in dtype.items()},
    )
    body = {
        "__slots__": ("_data", "_arr", *optional),
        "dtype": np.dtype(dtype_tuples),
        "nested_dtypes": nested_dtypes,
        "f": field_names,
    }
    struct_cls = type(cls.__name__, (cls, base), body)
    struct_cls.__module__ = cls.__module__
    cls.__name__ += "Origin"
    return struct_cls


class IntBlock(Block):
    """
    Custom Pandas block to carry S and U dtypes.

    The name hacks the internals to recognize the block downstream.
    """

    __slots__ = ()
    _can_hold_na = False

    @property
    def fill_value(self):
        """Return an empty string."""
        return self.values.dtype.type()

    def take_nd(
        self,
        indexer,
        axis: int = 0,
        new_mgr_locs=None,
        fill_value=blocks_lib.no_default,
    ):
        """Take values according to indexer and return them as a block."""
        new_values = self.values.take(indexer, axis=axis)

        if new_mgr_locs is None:
            new_mgr_locs = self.mgr_locs

        return self.make_block_same_class(new_values, new_mgr_locs)

    def putmask(
        self,
        mask,
        new,
        inplace: bool = False,
        axis: int = 0,
        transpose: bool = False,
    ) -> list["Block"]:
        """Specialize DataFrame.where()."""
        mask = _extract_bool_array(mask)
        new_values = self.values if inplace else self.values.copy()
        if isinstance(new, np.ndarray) and len(new) == len(mask):
            new = new[mask]
        mask = mask.reshape(new_values.shape)
        new_values[mask] = new
        return [self.make_block_same_class(new_values, placement=self.mgr_locs)]


def get_block_type(values, dtype=None):
    """Add block type exclusion for fixed-length bytes and strings."""
    if (dtype or values.dtype).kind in ("S", "U"):
        return IntBlock
    return get_block_type_original(values, dtype)


def make_block(values, placement, klass=None, ndim=None, dtype=None):
    """Override the block class if we are S or U."""
    if (
        klass is not None
        and klass.__name__ == "IntBlock"
        and klass is not IntBlock
        and values.dtype.kind in ("S", "U")
    ):
        if isinstance(values, pd.Series):
            values = values.values
        klass = IntBlock
    return make_block_original(values, placement, klass=klass, ndim=ndim, dtype=dtype)


pd.core.internals.blocks.get_block_type = get_block_type
pd.core.internals.blocks.make_block = make_block
pd.core.internals.managers.get_block_type = get_block_type
pd.core.internals.managers.make_block = make_block


original_index_new = pd.Index.__new__


def _string_friendly_index_new(
    cls: pd.Index,
    data=None,
    dtype=None,
    copy=False,
    name=None,
    tupleize_cols=True,
    **kwargs,
) -> pd.Index:
    if dtype is None and isinstance(data, np.ndarray) and data.dtype.kind in ("S", "U"):
        # otherwise, pandas will coerce to object dtype; we know better
        if copy:
            data = data.copy()
        return cls._simple_new(data, name)
    return original_index_new(
        cls, data=data, dtype=dtype, copy=copy, name=name, tupleize_cols=tupleize_cols, **kwargs,
    )


pd.Index.__new__ = _string_friendly_index_new


def create_data_frame_from_arrays(
    arrays_typed: Sequence[np.ndarray],
    arrays_obj: np.ndarray,
    names_typed: list[str],
    names_obj: list[str],
    size: int,
) -> pd.DataFrame:
    """
    Create a new Pandas DataFrame from two parts: sequence of typed arrays and an object array.

    This is much more efficient than calling pd.DataFrame(dict) or pd.DataFrame.from_records()
    because we construct the block manager directly and avoid memory copies and various integrity
    checks.
    """
    assert len(arrays_typed) == len(names_typed)
    if names_obj:
        assert len(arrays_obj) == len(names_obj)
    range_index = pd.RangeIndex(stop=size)
    blocks = [
        make_block(np.atleast_2d(arrays_typed[i]), placement=[i])
        for i, arr in enumerate(arrays_typed)
    ]
    if names_obj:
        blocks.append(
            make_block(arrays_obj, placement=np.arange(len(arrays_obj)) + len(arrays_typed)),
        )
    manager = BlockManager(blocks, [pd.Index(names_typed + names_obj), range_index])
    return pd.DataFrame(manager, columns=names_typed + names_obj, copy=False)


@sentry_span
def df_from_structs(items: Iterable[NumpyStruct], length: Optional[int] = None) -> pd.DataFrame:
    """
    Combine several NumpyStruct-s to a Pandas DataFrame.

    :param items: A collection, a generator, an iterator - all are accepted.
    :param length: In case `items` does not support `len()`, specify the number of structs \
                   for better performance.
    :return: Pandas DataFrame with columns set to struct fields.
    """
    try:
        if length is None:
            length = len(items)  # type: ignore
    except TypeError:
        # slower branch without pre-allocation
        items_iter = iter(items)
        try:
            first_item = next(items_iter)
        except StopIteration:
            return pd.DataFrame()
        assert isinstance(first_item, NumpyStruct)
        inspector = _NumpyStructInspector(first_item, None)
        indirect_items = inspector.indirect_columns.items()
        coerced_datas = inspector.coerced_datas
        columns = inspector.columns
        direct_columns = inspector.direct_columns
        length = 1
        for item in items_iter:
            length += 1
            coerced_datas.append(item.coerced_data)

            for k in direct_columns:
                columns[k].append(getattr(item, k))

            for field, subfields in indirect_items:
                subitem = getattr(item, field)
                for k, subfield in subfields:
                    columns[k].append(getattr(subitem, subfield))

        table_array = array_from_buffer(
            b"".join(coerced_datas), dtype=inspector.dtype, count=len(coerced_datas),
        )
        del coerced_datas
    else:
        items_iter = iter(items)
        try:
            first_item = next(items_iter)
        except StopIteration:
            return pd.DataFrame()

        assert isinstance(first_item, NumpyStruct)
        inspector = _NumpyStructInspector(first_item, length)

        coerced_pos = inspector.dtype.itemsize
        coerced_datas_buf = inspector.get_coerced_datas_buf(length)
        columns = inspector.columns

        indirect_items = inspector.indirect_columns.items()
        i = 0
        for i, item in enumerate(items_iter, 1):
            coerced_data = item.coerced_data
            coerced_datas_buf[coerced_pos : coerced_pos + inspector.dtype.itemsize] = coerced_data
            coerced_pos += inspector.dtype.itemsize

            for k in inspector.direct_columns:
                columns[k][i] = getattr(item, k)

            for field, subfields in indirect_items:
                subitem = getattr(item, field)
                for k, subfield in subfields:
                    columns[k][i] = getattr(subitem, subfield)

        table_array = array_from_buffer(coerced_datas_buf, dtype=inspector.dtype, count=i + 1)
        del coerced_datas_buf

    inspector.delete_data()
    typed_names = [name for name in inspector.dtype.names if name not in inspector.nested_fields]
    # must take a copy to make each strided array contiguous
    typed_arrs = [table_array[name].copy() for name in typed_names]
    obj_names = list(columns)
    if (obj_arr := inspector.obj_arr) is None:
        obj_arr = np.empty((len(columns), length), dtype=object)
        for i, (k, v) in enumerate(columns.items()):
            obj_arr[i] = v
            columns[k] = obj_arr[i]
    del table_array

    column_types = {}
    try:
        optionals = first_item.Optional
    except AttributeError:
        pass  # no Optional
    else:

        def set_dtype(k: str, v: Any) -> None:
            if not isinstance(v, type) or not issubclass(
                v, (datetime, np.datetime64, bool, int, float),
            ):
                # we can only unbox types that have a "NaN" value
                v = object
            column_types[k] = v

        for k, v in optionals.__annotations__.items():
            if (sub_ks := inspector.indirect_columns.get(k)) is not None:
                for full_k, sub_k in sub_ks:
                    set_dtype(full_k, v.__annotations__[sub_k])
            else:
                set_dtype(k, v)

    left_untyped = []
    for i, (k, v) in enumerate(columns.items()):
        column_type = column_types.get(k, object)
        typed = False
        if issubclass(column_type, datetime):
            v = tslib.array_to_datetime(np.array(v, dtype=object), utc=True, errors="raise")[0]
            typed = True
        elif issubclass(column_type, timedelta):
            v = np.array(v, dtype="timedelta64[s]")
            typed = True
        elif np.dtype(column_type) != np.dtype(object):
            if not issubclass(column_type, (bool, int)) or is_not_null(v).all():
                v = np.array(v, dtype=column_type)
                typed = True
        if typed:
            typed_names.append(k)
            typed_arrs.append(v)
        else:
            left_untyped.append(i)
    if len(left_untyped) < len(obj_arr):
        obj_arr = obj_arr[left_untyped]
        obj_names = [obj_names[i] for i in left_untyped]

    df = create_data_frame_from_arrays(typed_arrs, obj_arr, typed_names, obj_names, length)
    sentry_sdk.Hub.current.scope.span.description = str(len(df))
    return df


class _NumpyStructInspector:
    def __init__(self, first_item: NumpyStruct, length: Optional[int]):
        columns = {}
        direct_columns = []
        dtype = [(f, v[0]) for f, v in first_item.dtype.fields.items()]
        direct_nested_fields = first_item.nested_dtypes
        coerced_datas = [first_item.coerced_data]
        indirect_columns = {}

        for k, v in first_item.items():
            if k not in first_item.dtype.names or k in direct_nested_fields:
                if dataclasses.is_dataclass(v):
                    for subfield, subvalue in dataclass_asdict(v).items():
                        full_k = f"{k}_{subfield}"
                        indirect_columns.setdefault(k, []).append((full_k, subfield))
                        columns[full_k] = [subvalue]
                else:
                    columns[k] = [v]
                    direct_columns.append(k)

        self._dtype = np.dtype(dtype)  # dtype for coerced_datas
        self._coerced_datas = coerced_datas
        if length is not None:
            self._obj_arr = obj_arr = np.empty((len(columns), length), dtype=object)
            for i, (k, v) in enumerate(columns.items()):
                columns[k] = sub_arr = obj_arr[i]
                sub_arr[0] = v[0]
        else:
            self._obj_arr = None
        self._columns = columns
        self._direct_columns = direct_columns
        self._nested_fields = direct_nested_fields.keys()
        self._indirect_columns = indirect_columns

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def coerced_datas(self) -> list[memoryview]:
        return self._coerced_datas

    @property
    def columns(self) -> dict[str, list[Any] | npt.NDArray[object]]:
        return self._columns

    @property
    def obj_arr(self) -> Optional[npt.NDArray[object]]:
        return self._obj_arr

    @property
    def direct_columns(self) -> list[str]:
        return self._direct_columns

    @property
    def nested_fields(self) -> KeysView[str]:
        return self._nested_fields

    @property
    def indirect_columns(self) -> dict[str, list[tuple[str, str]]]:
        return self._indirect_columns

    def get_coerced_datas_buf(self, length: int) -> bytearray:
        assert self._coerced_datas is not None
        itemsize = self._dtype.itemsize
        first_coerced_data = self._coerced_datas[0]
        buf = bytearray(itemsize * length)
        buf[:itemsize] = first_coerced_data
        return buf

    def delete_data(self) -> None:
        self._coerced_datas = None


def dataclass_asdict(dataclass_obj: Any) -> Mapping[str, Any]:
    """Convert a dataclass instance to a dict.

    This is lighter than stdlib dataclasses.asdict since fields are not recursed and
    complex values are not deep-copied.
    Simply field name/value couples will be the values of the dict.
    Dataclass fields order is preserved in the returned dict.
    """
    return {f.name: getattr(dataclass_obj, f.name) for f in dataclasses.fields(dataclass_obj)}
