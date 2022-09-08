import asyncio
from datetime import datetime, timezone
from itertools import chain
import logging
import textwrap
from typing import Any, Awaitable, Iterable, Optional, Sequence, Type, Union

import numpy as np
import pandas as pd
from pandas.core.dtypes.cast import OutOfBoundsDatetime, tslib
from pandas.core.internals.blocks import (
    Block,
    _extract_bool_array,
    get_block_type as get_block_type_original,
    lib as blocks_lib,
    make_block as make_block_original,
)
from pandas.core.internals.managers import BlockManager
import sentry_sdk
from sqlalchemy import BigInteger, Boolean, Column, DateTime, Integer, SmallInteger, String
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.sql.elements import Label
from sqlalchemy.sql.selectable import GenerativeSelect

from athenian.api import metadata
from athenian.api.db import Database, DatabaseLike, is_postgresql
from athenian.api.models.metadata.github import Base as MetadataBase
from athenian.api.models.persistentdata.models import Base as PerdataBase
from athenian.api.models.precomputed.models import GitHubBase as PrecomputedBase
from athenian.api.models.state.models import Base as StateBase
from athenian.api.to_object_arrays import is_null, to_object_arrays_split
from athenian.api.tracing import MAX_SENTRY_STRING_LENGTH


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


async def read_sql_query(
    sql: GenerativeSelect,
    con: DatabaseLike,
    columns: Union[
        Sequence[str],
        Sequence[InstrumentedAttribute],
        MetadataBase,
        PerdataBase,
        PrecomputedBase,
        StateBase,
    ],
    index: Optional[Union[str, Sequence[str], InstrumentedAttribute]] = None,
    soft_limit: Optional[int] = None,
) -> pd.DataFrame:
    """Read SQL query into a DataFrame.

    Returns a DataFrame corresponding to the result set of the query.
    Optionally provide an `index_col` parameter to use one of the
    columns as the index, otherwise default integer index will be used.

    Parameters
    ----------
    sql     : SQLAlchemy query object to be executed.
    con     : async SQLAlchemy database engine.
    columns : list of the resulting columns names, column objects or the model if SELECT *
    index   : Name(s) of the index column(s).
    soft_limit
            : Load this number of rows at maximum.

    Returns
    -------
    DataFrame

    Notes
    -----
    Any datetime values with time zone information parsed via the `parse_dates`
    parameter will be converted to UTC.
    """
    if isinstance(index, InstrumentedAttribute):
        index = index.name
    if await is_postgresql(con):
        if not isinstance(columns, Sequence):
            columns = columns.__table__.columns
        return await _read_sql_query_numpy(sql, con, columns, index=index, soft_limit=soft_limit)
    return await _read_sql_query_records(sql, con, columns, index=index, soft_limit=soft_limit)


async def _read_sql_query_numpy(
    sql: GenerativeSelect,
    con: DatabaseLike,
    columns: Union[Sequence[str], Sequence[InstrumentedAttribute]],
    index: Optional[Union[str, Sequence[str]]] = None,
    soft_limit: Optional[int] = None,
) -> pd.DataFrame:
    dtype, (int_erase_nulls, int_reset_nulls, str_erase_nulls, str_reset_nulls) = _build_dtype(
        columns,
    )
    sql.dtype = dtype
    try:
        data, nulls = await _fetch_query(sql, con)
    finally:
        sql.dtype = None
    blocks = {}
    rows_count = 0
    for i, arr in enumerate(data):
        blocks.setdefault(arr.dtype, [arr.base, []])[1].append(i)
        rows_count = len(arr)
    if nulls and (int_erase_nulls or int_reset_nulls or str_erase_nulls or str_reset_nulls):
        null_items, null_cols = np.unravel_index(nulls, (rows_count, len(dtype)))
        order = np.argsort(null_cols)
        null_items = null_items[order]
        null_cols = null_cols[order]
        unique_null_cols, offsets, counts = np.unique(
            null_cols, return_index=True, return_counts=True,
        )
        col_map = {name: i for i, name in enumerate(dtype.names)}
        remain_mask = None
        if int_erase_nulls or str_erase_nulls:
            for col in chain(int_erase_nulls, str_erase_nulls):
                pos = np.searchsorted(unique_null_cols, (col_index := col_map[col]))
                if pos < len(unique_null_cols) and unique_null_cols[pos] == col_index:
                    if remain_mask is None:
                        remain_mask = np.ones(rows_count, dtype=bool)
                    remain_mask[null_items[offsets[pos] : offsets[pos] + counts[pos]]] = False
        for reset_cols, reset_val in ((int_reset_nulls, 0), (str_reset_nulls, b"")):
            for col in reset_cols:
                pos = np.searchsorted(unique_null_cols, (col_index := col_map[col]))
                if pos < len(unique_null_cols) and unique_null_cols[pos] == col_index:
                    data[col][null_items[offsets[pos] : offsets[pos] + counts[pos]]] = reset_val
        if remain_mask is not None:
            for ptrs in blocks.values():
                ptrs[0] = ptrs[0][:, remain_mask]
            rows_count = remain_mask.sum()
    if soft_limit is not None and rows_count > soft_limit:
        rows_count = soft_limit
        for ptrs in blocks.values():
            ptrs[0] = ptrs[0][:, :soft_limit]
    pd_blocks = [make_block(block, placement=indexes) for block, indexes in blocks.values()]
    dtype_names = dtype.names
    block_mgr = BlockManager(pd_blocks, [pd.Index(dtype_names), pd.RangeIndex(stop=rows_count)])
    frame = pd.DataFrame(block_mgr, columns=dtype_names, copy=False)
    for column, (child_dtype, _) in dtype.fields.items():
        if child_dtype.kind == "M":
            try:
                frame[column] = frame[column].dt.tz_localize(timezone.utc)
            except (AttributeError, TypeError):
                continue
    if index is not None:
        frame.set_index(index, inplace=True)
    return frame


def _build_dtype(
    columns: Sequence[InstrumentedAttribute],
) -> tuple[np.dtype, tuple[set[str], set[str], set[str], set[str]]]:
    body = []
    int_erase_nulls = set()
    int_reset_nulls = set()
    str_erase_nulls = set()
    str_reset_nulls = set()
    for c in columns:
        if isinstance(c, str):
            body.append((c, object))
            continue
        if isinstance(c.type, DateTime) or (
            isinstance(c.type, type) and issubclass(c.type, DateTime)
        ):
            body.append((c.name, "datetime64[ns]"))
        elif isinstance(c.type, Boolean) or (
            isinstance(c.type, type) and issubclass(c.type, Boolean)
        ):
            body.append((c.name, bool))
        elif (
            isinstance(c.type, Integer)
            or (isinstance(c.type, type) and issubclass(c.type, Integer))
        ) and (
            (
                info := getattr(
                    c,
                    "info",
                    {} if not isinstance(c, Label) else getattr(c.element, "info", {}),
                )
            ).get("erase_nulls", False)
            or info.get("reset_nulls", False)
            or (
                not getattr(c, "nullable", False)
                and (not isinstance(c, Label) or not getattr(c.element, "nullable", False))
            )
        ):
            info = getattr(
                c, "info", {} if not isinstance(c, Label) else getattr(c.element, "info", {}),
            )
            if info.get("erase_nulls", False):
                int_erase_nulls.add(c.name)
            elif info.get("reset_nulls", False):
                int_reset_nulls.add(c.name)
            sql_type = c.type if isinstance(c.type, type) else type(c.type)
            if sql_type == BigInteger:
                int_dtype = np.int64
            elif sql_type == SmallInteger:
                int_dtype = np.int16
            elif sql_type == Integer:
                int_dtype = np.int32
            else:
                raise AssertionError(f"Unsupported integer type: {sql_type.__name__}")
            body.append((c.name, int_dtype))
        elif (
            isinstance(c.type, String) or (isinstance(c.type, type) and issubclass(c.type, String))
        ) and (
            info := getattr(
                c, "info", {} if not isinstance(c, Label) else getattr(c.element, "info", {}),
            )
        ).get(
            "dtype", False,
        ):
            dtype = np.dtype(info["dtype"])
            if info.get("erase_nulls"):
                str_erase_nulls.add(c.name)
            if dtype.kind == "S":
                str_reset_nulls.add(c.name)
            body.append((c.name, dtype))
        else:
            body.append((c.name, object))

    dtype = np.dtype(body, metadata={"blocks": True})
    return dtype, (int_erase_nulls, int_reset_nulls, str_erase_nulls, str_reset_nulls)


async def _read_sql_query_records(
    sql: GenerativeSelect,
    con: DatabaseLike,
    columns: Union[
        Sequence[str],
        Sequence[InstrumentedAttribute],
        MetadataBase,
        PerdataBase,
        PrecomputedBase,
        StateBase,
    ],
    index: Optional[Union[str, Sequence[str]]] = None,
    soft_limit: Optional[int] = None,
) -> pd.DataFrame:
    data = await _fetch_query(sql, con)
    if soft_limit is not None and len(data) > soft_limit:
        data = data[:soft_limit]
    return _wrap_sql_query(data, columns, index)


async def _fetch_query(
    sql: GenerativeSelect,
    con: DatabaseLike,
) -> Union[list[Sequence[Any]], tuple[np.ndarray, list[int]]]:
    try:
        data = await con.fetch_all(query=sql)
    except Exception as e:
        sql.dtype = None
        try:
            sql = str(sql)
        except Exception:
            sql = repr(sql)
        sql = textwrap.shorten(sql, MAX_SENTRY_STRING_LENGTH - 500)
        logging.getLogger("%s.read_sql_query" % metadata.__package__).error(
            "%s: %s; %s", type(e).__name__, e, sql,
        )
        raise e from None
    return data


def _create_block_manager_from_arrays(
    arrays_typed: Sequence[np.ndarray],
    arrays_obj: np.ndarray,
    names_typed: list[str],
    names_obj: list[str],
    size: int,
) -> BlockManager:
    assert len(arrays_typed) == len(names_typed)
    assert len(arrays_obj) == len(names_obj)
    range_index = pd.RangeIndex(stop=size)
    blocks = [
        make_block(np.atleast_2d(arrays_typed[i]), placement=[i])
        for i, arr in enumerate(arrays_typed)
    ]
    blocks.append(make_block(arrays_obj, placement=np.arange(len(arrays_obj)) + len(arrays_typed)))
    return BlockManager(blocks, [pd.Index(names_typed + names_obj), range_index])


def _wrap_sql_query(
    data: list[Sequence[Any]],
    columns: Union[Sequence[str], Sequence[InstrumentedAttribute], MetadataBase, StateBase],
    index: Optional[Union[str, Sequence[str]]] = None,
) -> pd.DataFrame:
    """Turn the fetched DB records to a pandas DataFrame."""
    try:
        columns = columns.__table__.columns
    except AttributeError:
        pass
    dt_columns = _extract_datetime_columns(columns)
    int_columns = _extract_integer_columns(columns)
    bool_columns = _extract_boolean_columns(columns)
    fixed_str_columns = _extract_fixed_string_columns(columns)
    columns = [(c.name if not isinstance(c, str) else c) for c in columns]

    typed_cols_indexes = []
    typed_cols_names = []
    obj_cols_indexes = []
    obj_cols_names = []
    for i, column in enumerate(columns):
        if (
            column in dt_columns
            or column in int_columns
            or column in bool_columns
            or column in fixed_str_columns
        ):
            cols_indexes = typed_cols_indexes
            cols_names = typed_cols_names
        else:
            cols_indexes = obj_cols_indexes
            cols_names = obj_cols_names
        cols_indexes.append(i)
        cols_names.append(column)
    log = logging.getLogger(f"{metadata.__package__}.wrap_sql_query")
    # we used to have pd.DataFrame.from_records + bunch of convert_*() in relevant columns
    # the current approach is faster for several reasons:
    # 1. avoid an expensive copy of the object dtype columns in the BlockManager construction
    # 2. call tslib.array_to_datetime directly without surrounding Pandas bloat
    # 3. convert to int in the numpy domain and thus do not have to mess with indexes
    #
    # an ideal conversion would be loading columns directly from asyncpg but that requires
    # quite some changes in their internals
    with sentry_sdk.start_span(op="wrap_sql_query/convert", description=str(size := len(data))):
        data_typed, data_obj = to_object_arrays_split(data, typed_cols_indexes, obj_cols_indexes)
        converted_typed = []
        remain_mask = None
        for column, values in zip(typed_cols_names, data_typed):
            if column in dt_columns:
                converted_typed.append(_convert_datetime(values))
            elif column in int_columns:
                values, discarded = _convert_integer(values, column, *int_columns[column], log)
                converted_typed.append(values)
                if discarded is not None:
                    if remain_mask is None:
                        remain_mask = np.ones(len(data), dtype=bool)
                    remain_mask[discarded] = False
            elif column in bool_columns:
                converted_typed.append(values.astype(bool))
            elif column in fixed_str_columns:
                values[is_null(values)] = np.dtype(fixed_str_columns[column]).type()
                converted_typed.append(values.astype(fixed_str_columns[column]))
            else:
                raise AssertionError("impossible: typed columns are either dt or int")
        if remain_mask is not None:
            size = remain_mask.sum()
            converted_typed = [arr[remain_mask] for arr in converted_typed]
            data_obj = data_obj[:, remain_mask]
    with sentry_sdk.start_span(op="wrap_sql_query/pd.DataFrame()", description=str(size)):
        block_mgr = _create_block_manager_from_arrays(
            converted_typed, data_obj, typed_cols_names, obj_cols_names, size,
        )
        frame = pd.DataFrame(block_mgr, columns=typed_cols_names + obj_cols_names, copy=False)
        for column in dt_columns:
            try:
                frame[column] = frame[column].dt.tz_localize(timezone.utc)
            except (AttributeError, TypeError):
                continue
        if index is not None:
            frame.set_index(index, inplace=True)
    return frame


def _extract_datetime_columns(columns: Iterable[Union[Column, str]]) -> set[str]:
    return {
        c.name
        for c in columns
        if not isinstance(c, str)
        and (
            isinstance(c.type, DateTime)
            or (isinstance(c.type, type) and issubclass(c.type, DateTime))
        )
    }


def _extract_boolean_columns(columns: Iterable[Union[Column, str]]) -> set[str]:
    return {
        c.name
        for c in columns
        if not isinstance(c, str)
        and (
            isinstance(c.type, Boolean)
            or (isinstance(c.type, type) and issubclass(c.type, Boolean))
        )
    }


def _extract_integer_columns(
    columns: Iterable[Union[Column, str]],
) -> dict[str, tuple[bool, bool]]:
    return {
        c.name: (info.get("erase_nulls", False), info.get("reset_nulls", False))
        for c in columns
        if not isinstance(c, str)
        and (
            isinstance(c.type, Integer)
            or (isinstance(c.type, type) and issubclass(c.type, Integer))
        )
        and (
            (
                info := getattr(
                    c, "info", {} if not isinstance(c, Label) else getattr(c.element, "info", {}),
                )
            ).get("erase_nulls", False)
            or info.get("reset_nulls", False)
            or (
                not getattr(c, "nullable", False)
                and (not isinstance(c, Label) or not getattr(c.element, "nullable", False))
            )
        )
    }


def _extract_fixed_string_columns(
    columns: Iterable[Union[Column, str]],
) -> dict[str, tuple[str, bool]]:
    return {
        c.name: info["dtype"]
        for c in columns
        if (
            not isinstance(c, str)
            and (
                isinstance(c.type, String)
                or (isinstance(c.type, type) and issubclass(c.type, String))
            )
            and (
                info := getattr(
                    c, "info", {} if not isinstance(c, Label) else getattr(c.element, "info", {}),
                )
            ).get("dtype", False)
        )
    }


def _convert_datetime(arr: np.ndarray) -> np.ndarray:
    # None converts to NaT
    try:
        ts, offset = tslib.array_to_datetime(arr, utc=True, errors="raise")
        assert offset is None
    except OutOfBoundsDatetime:
        # TODO(vmarkovtsev): copy the function and set OOB values to NaT
        # this comparison is very slow but still faster than removing tzinfo and taking np.array()
        arr[arr == datetime(1, 1, 1)] = None
        arr[arr == datetime(1, 1, 1, tzinfo=timezone.utc)] = None
        try:
            return _convert_datetime(arr)
        except OutOfBoundsDatetime as e:
            raise e from None
    # 0 converts to 1970-01-01T00:00:00
    ts[ts == np.zeros(1, ts.dtype)[0]] = None
    return ts


def postprocess_datetime(
    frame: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Ensure *inplace* that all the timestamps inside the dataframe are valid UTC or NaT.

    :return: Fixed dataframe - the same instance as `frame`.
    """
    utc_dt1 = datetime(1, 1, 1, tzinfo=timezone.utc)
    dt1 = datetime(1, 1, 1)
    if columns is not None:
        obj_cols = dt_cols = columns
    else:
        obj_cols = frame.select_dtypes(include=[object])
        dt_cols = frame.select_dtypes(include=["datetime"])
    for col in obj_cols:
        fc = frame[col]
        if utc_dt1 in fc:
            fc.replace(utc_dt1, pd.NaT, inplace=True)
        if dt1 in fc:
            fc.replace(dt1, pd.NaT, inplace=True)
    for col in dt_cols:
        fc = frame[col]
        if 0 in fc:
            fc.replace(0, pd.NaT, inplace=True)
        try:
            frame[col] = fc.dt.tz_localize(timezone.utc)
        except (AttributeError, TypeError):
            continue
    return frame


def _convert_integer(
    arr: np.ndarray,
    name: str,
    erase_null: bool,
    reset_null: bool,
    log: logging.Logger,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    nulls = None
    while True:
        try:
            return arr.astype(int), nulls if not reset_null else None
        except TypeError as e:
            nulls = is_null(arr)
            if not (nulls.any() and (erase_null or reset_null)):
                raise ValueError(f"Column {name} is not all-integer") from e
            log.warning("fetched nulls instead of integers in %s", name)
            arr[nulls] = 0


async def gather(
    *coros_or_futures: Optional[Awaitable],
    op: Optional[str] = None,
    description: Optional[str] = None,
    catch: Type[BaseException] = Exception,
) -> tuple[Any, ...]:
    """Return a future aggregating results/exceptions from the given coroutines/futures.

    This is equivalent to `asyncio.gather(*coros_or_futures, return_exceptions=True)` with
    subsequent exception forwarding and skipping None-s.

    :param op: Wrap the execution in a Sentry span with this `op`.
    :param description: Sentry span description.
    :param catch: Forward exceptions of this type.
    :return: tuple with awaited results. If some awaitable was None, the corresponding result \
             is None, too.
    """
    __tracebackhide__ = True  # noqa: F841

    async def dummy() -> None:
        return None

    async def body():
        __tracebackhide__ = True  # noqa: F841
        nonlocal coros_or_futures
        if len(coros_or_futures) == 0:
            return ()
        coros_or_futures = [(cf if cf is not None else dummy()) for cf in coros_or_futures]
        if len(coros_or_futures) == 1:
            return (await coros_or_futures[0],)
        results = await asyncio.gather(*coros_or_futures, return_exceptions=True)
        for r in results:
            if isinstance(r, catch):
                raise r from None
        return results

    if op is not None:
        with sentry_sdk.start_span(op=op, description=description):
            return await body()
    return await body()


async def read_sql_query_with_join_collapse(
    query: GenerativeSelect,
    db: Database,
    columns: Union[
        Sequence[str],
        Sequence[InstrumentedAttribute],
        MetadataBase,
        PerdataBase,
        PrecomputedBase,
        StateBase,
    ],
    index: Optional[Union[str, Sequence[str]]] = None,
    soft_limit: Optional[int] = None,
) -> pd.DataFrame:
    """Enforce the predefined JOIN order in read_sql_query()."""
    query = query.with_statement_hint("set(join_collapse_limit 1)")
    return await read_sql_query(query, db, columns=columns, index=index, soft_limit=soft_limit)


# Allow other coroutines to execute every Nth iteration in long loops
COROUTINE_YIELD_EVERY_ITER = 250


async def list_with_yield(iterable: Iterable[Any], sentry_op: str) -> list[Any]:
    """Drain an iterable to a list, tracing the loop in Sentry and respecting other coroutines."""
    with sentry_sdk.start_span(op=sentry_op) as span:
        things = []
        for i, thing in enumerate(iterable):
            if (i + 1) % COROUTINE_YIELD_EVERY_ITER == 0:
                await asyncio.sleep(0)
            things.append(thing)
        try:
            span.description = str(i)
        except UnboundLocalError:
            pass
    return things
