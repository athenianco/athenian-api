import asyncio
from itertools import chain
import logging
import textwrap
from typing import Any, Awaitable, Iterable, Optional, Sequence, Type, Union

import medvedi as md
from medvedi.accelerators import is_not_null, is_null
import numpy as np
from numpy import typing as npt
import sentry_sdk
import sqlalchemy as sa
from sqlalchemy import REAL, BigInteger, Boolean, Column, DateTime, Integer, SmallInteger, String
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.sql.elements import Label
from sqlalchemy.sql.selectable import GenerativeSelect

from athenian.api import metadata
from athenian.api.db import Database, DatabaseLike, is_postgresql
from athenian.api.models.metadata.github import Base as MetadataBase
from athenian.api.models.persistentdata.models import Base as PerdataBase
from athenian.api.models.precomputed.models import GitHubBase as PrecomputedBase
from athenian.api.models.state.models import Base as StateBase
from athenian.api.object_arrays import to_object_arrays
from athenian.api.tracing import MAX_SENTRY_STRING_LENGTH


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
) -> md.DataFrame:
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
    if not isinstance(columns, Sequence):
        columns = columns.__table__.columns
    if await is_postgresql(con):
        return await _read_sql_query_numpy(sql, con, columns, index=index, soft_limit=soft_limit)
    return await _read_sql_query_records(sql, con, columns, index=index, soft_limit=soft_limit)


async def _read_sql_query_numpy(
    sql: GenerativeSelect,
    con: DatabaseLike,
    columns: Sequence[str] | Sequence[InstrumentedAttribute],
    index: Optional[Union[str, Sequence[str]]] = None,
    soft_limit: Optional[int] = None,
) -> md.DataFrame:
    dtype, (int_erase_nulls, int_reset_nulls, str_erase_nulls, str_reset_nulls) = infer_dtype(
        columns,
    )
    sql.dtype = dtype
    try:
        data, nulls = await _fetch_query(sql, con)
    finally:
        sql.dtype = None
    rows_count = len(data[0]) if len(columns) else 0
    return _columns_to_dataframe(
        dict(zip(dtype.names, data)),
        rows_count,
        dtype,
        int_erase_nulls,
        int_reset_nulls,
        str_erase_nulls,
        str_reset_nulls,
        *np.unravel_index(np.asarray(nulls, dtype=int), (rows_count, len(dtype))),
        index,
        soft_limit,
    )


def _columns_to_dataframe(
    columns: dict[str, np.ndarray],
    rows_count: int,
    dtype: np.dtype,
    int_erase_nulls: set[str],
    int_reset_nulls: set[str],
    str_erase_nulls: set[str],
    str_reset_nulls: set[str],
    null_items: npt.NDArray[int],
    null_cols: npt.NDArray[int],
    index: Optional[Union[str, Sequence[str]]],
    soft_limit: Optional[int],
) -> md.DataFrame:
    if len(null_items) and (
        int_erase_nulls or int_reset_nulls or str_erase_nulls or str_reset_nulls
    ):
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
                    columns[col][null_items[offsets[pos] : offsets[pos] + counts[pos]]] = reset_val
        if remain_mask is not None:
            for key, column in columns.items():
                columns[key] = column[remain_mask]
            rows_count = remain_mask.sum()
    if soft_limit is not None and rows_count > soft_limit:
        for key, column in columns.items():
            columns[key] = column[:soft_limit]

    return md.DataFrame(columns, index=index)


def infer_dtype(
    columns: Sequence[InstrumentedAttribute],
) -> tuple[np.dtype, tuple[set[str], set[str], set[str], set[str]]]:
    """Detect the joint structured dtype of the specified columns."""
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
            body.append((c.name, "datetime64[us]"))
        elif isinstance(c.type, Boolean) or (
            isinstance(c.type, type) and issubclass(c.type, Boolean)
        ):
            body.append((c.name, bool))
        elif (
            isinstance(c.type, Integer)
            or (isinstance(c.type, type) and issubclass(c.type, Integer))
        ) and (
            (info := _get_col_info(c)).get("erase_nulls", False)
            or info.get("reset_nulls", False)
            or not _get_col_nullable(c)
        ):
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
        ) and ((info := _get_col_info(c)).get("dtype", False)):
            dtype = np.dtype(info["dtype"])
            if info.get("erase_nulls"):
                str_erase_nulls.add(c.name)
            if dtype.kind == "S":
                str_reset_nulls.add(c.name)
            body.append((c.name, dtype))
        elif isinstance(c.type, REAL) or (isinstance(c.type, type) and issubclass(c.type, REAL)):
            body.append((c.name, np.float32))
        else:
            body.append((c.name, object))

    dtype = np.dtype(body, metadata={"blocks": True})
    return dtype, (int_erase_nulls, int_reset_nulls, str_erase_nulls, str_reset_nulls)


async def _read_sql_query_records(
    sql: GenerativeSelect,
    con: DatabaseLike,
    columns: Sequence[str] | Sequence[InstrumentedAttribute],
    index: Optional[Union[str, Sequence[str]]] = None,
    soft_limit: Optional[int] = None,
) -> md.DataFrame:
    rows = await _fetch_query(sql, con)

    dtype, (int_erase_nulls, int_reset_nulls, str_erase_nulls, str_reset_nulls) = infer_dtype(
        columns,
    )

    data = to_object_arrays(rows, len(columns))
    nulls = np.flatnonzero(is_null(data.ravel()))

    columns = {}
    for (name, (column_dtype, *_)), arr in zip(dtype.fields.items(), data):
        if arr_len := len(arr):
            try:
                if column_dtype.kind in ("S", "U"):
                    raise TypeError
                casted = arr.astype(column_dtype)
            except TypeError:
                if column_dtype.kind in ("f", "M", "m"):
                    casted = np.full(arr_len, None, column_dtype)
                else:
                    casted = np.zeros(arr_len, column_dtype)
                if (mask := is_not_null(arr)).any():
                    casted[mask] = arr[mask].astype(column_dtype)
        else:
            casted = np.array([], dtype=column_dtype)
        columns[name] = casted

    null_cols, null_items = np.unravel_index(nulls, (len(dtype), len(rows)))
    return _columns_to_dataframe(
        columns,
        len(rows),
        dtype,
        int_erase_nulls,
        int_reset_nulls,
        str_erase_nulls,
        str_reset_nulls,
        null_items,
        null_cols,
        index,
        soft_limit,
    )


async def _fetch_query(
    sql: GenerativeSelect,
    con: DatabaseLike,
) -> Union[list[Sequence[Any]], tuple[np.ndarray, list[int]]]:
    try:
        rows = await con.fetch_all(query=sql)
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
    return rows


def _extract_col_property(col: Column, prop: str, default: Any) -> Any:
    if isinstance(table := getattr(col, "table", None), sa.sql.Alias):
        return _extract_col_property(getattr(table.element.c, col.name, None), prop, default)
    try:
        return getattr(col, prop)
    except AttributeError:
        if isinstance(col, Label):
            return _extract_col_property(col.element, prop, default)
        return default


def _get_col_info(col: Column):
    """Get the `info` attribute of the column, unwrapping Label and Alias if needed."""
    return _extract_col_property(col, "info", {})


def _get_col_nullable(col: Column):
    """Get the `nullable` attribute of the column, unwrapping Label and Alias if needed."""
    return _extract_col_property(col, "nullable", False)


class CatchNothing(Exception):
    """Suppress all errors inside gather()."""


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
) -> md.DataFrame:
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
