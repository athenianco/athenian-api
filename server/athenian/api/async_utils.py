import asyncio
from datetime import datetime, timezone
import logging
import textwrap
from typing import Any, Collection, Iterable, Optional, Sequence, Tuple, Type, Union

import numpy as np
import pandas as pd
import sentry_sdk
from sqlalchemy import Column, DateTime, Integer
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.sql import ClauseElement
from sqlalchemy.sql.elements import Label

from athenian.api import metadata
from athenian.api.db import Database, DatabaseLike
from athenian.api.models.metadata.github import Base as MetadataBase
from athenian.api.models.persistentdata.models import Base as PerdataBase
from athenian.api.models.precomputed.models import GitHubBase as PrecomputedBase
from athenian.api.models.state.models import Base as StateBase
from athenian.api.tracing import MAX_SENTRY_STRING_LENGTH


async def read_sql_query(sql: ClauseElement,
                         con: DatabaseLike,
                         columns: Union[Sequence[str], Sequence[InstrumentedAttribute],
                                        MetadataBase, PerdataBase, PrecomputedBase, StateBase],
                         index: Optional[Union[str, Sequence[str]]] = None,
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

    Returns
    -------
    DataFrame

    Notes
    -----
    Any datetime values with time zone information parsed via the `parse_dates`
    parameter will be converted to UTC.
    """
    try:
        data = await con.fetch_all(query=sql)
    except Exception as e:
        try:
            sql = str(sql)
        except Exception:
            sql = repr(sql)
        sql = textwrap.shorten(sql, MAX_SENTRY_STRING_LENGTH - 500)
        logging.getLogger("%s.read_sql_query" % metadata.__package__).error(
            "%s: %s; %s", type(e).__name__, e, sql)
        raise e from None
    return wrap_sql_query(data, columns, index)


def wrap_sql_query(data: Sequence[Iterable[Any]],
                   columns: Union[Sequence[str], Sequence[InstrumentedAttribute],
                                  MetadataBase, StateBase],
                   index: Optional[Union[str, Sequence[str]]] = None,
                   ) -> pd.DataFrame:
    """Turn the fetched DB records to a pandas DataFrame."""
    try:
        columns[0]
    except TypeError:
        dt_columns = _extract_datetime_columns(columns.__table__.columns)
        i_columns = _extract_integer_columns(columns.__table__.columns)
        columns = [c.name for c in columns.__table__.columns]
    else:
        dt_columns = _extract_datetime_columns(columns)
        i_columns = _extract_integer_columns(columns)
        columns = [(c.name if not isinstance(c, str) else c) for c in columns]
    with sentry_sdk.start_span(op="pd.DataFrame.from_records", description=str(len(data))):
        frame = pd.DataFrame.from_records(data, columns=columns, coerce_float=True)
    with sentry_sdk.start_span(op="postprocess"):
        frame = postprocess_datetime(frame, columns=dt_columns)
        frame = postprocess_integer(frame, columns=i_columns)
        if index is not None:
            frame.set_index(index, inplace=True)
    return frame


def _extract_datetime_columns(columns: Iterable[Union[Column, str]]) -> Collection[str]:
    return [
        c.name for c in columns
        if not isinstance(c, str) and (
            isinstance(c.type, DateTime) or
            (isinstance(c.type, type) and issubclass(c.type, DateTime))
        )
    ]


def _extract_integer_columns(columns: Iterable[Union[Column, str]],
                             ) -> Collection[Tuple[str, bool]]:
    return [
        (
            c.name, getattr(
                c, "info", {} if not isinstance(c, Label) else getattr(c.element, "info", {}),
            ).get("erase_nulls", False),
        )
        for c in columns
        if not isinstance(c, str) and (
            isinstance(c.type, Integer) or
            (isinstance(c.type, type) and issubclass(c.type, Integer))
        )
        and not getattr(c, "nullable", False)
        and (not isinstance(c, Label) or (
            (not getattr(c.element, "nullable", False))
            and (not getattr(c, "nullable", False))
        ))
    ]


def postprocess_datetime(frame: pd.DataFrame,
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


def postprocess_integer(frame: pd.DataFrame, columns: Iterable[Tuple[str, int]]) -> pd.DataFrame:
    """Ensure *inplace* that all the integers inside the dataframe are not objects.

    :return: Fixed dataframe, a potentially different instance.
    """
    dirty_index = False
    log = None
    for col, erase_null in columns:
        while True:
            try:
                frame[col] = frame[col].astype(int, copy=False)
                break
            except TypeError as e:
                nulls = frame[col].isnull().values
                if not nulls.any():
                    raise ValueError(f"Column {col} is not all-integer") from e
                if not erase_null:
                    raise ValueError(f"Column {col} is not all-integer\n"
                                     f"{frame.loc[nulls].to_dict('records')}") from e
                if log is None:
                    log = logging.getLogger(f"{metadata.__package__}.read_sql_query")
                log.error("fetched nulls instead of integers in %s: %s",
                          col, frame.loc[nulls].to_dict("records"))
                frame = frame.take(np.flatnonzero(~nulls))
                dirty_index = True
    if dirty_index:
        frame.reset_index(drop=True, inplace=True)
    return frame


async def gather(*coros_or_futures,
                 op: Optional[str] = None,
                 description: Optional[str] = None,
                 catch: Type[BaseException] = Exception,
                 ) -> Tuple[Any, ...]:
    """Return a future aggregating results/exceptions from the given coroutines/futures.

    This is equivalent to `asyncio.gather(*coros_or_futures, return_exceptions=True)` with
    subsequent exception forwarding.

    :param op: Wrap the execution in a Sentry span with this `op`.
    :param description: Sentry span description.
    :param catch: Forward exceptions of this type.
    """
    async def body():
        if len(coros_or_futures) == 0:
            return tuple()
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
        query: ClauseElement,
        db: Database,
        columns: Union[Sequence[str], Sequence[InstrumentedAttribute],
                       MetadataBase, PerdataBase, PrecomputedBase, StateBase],
        index: Optional[Union[str, Sequence[str]]] = None,
) -> pd.DataFrame:
    """Enforce the predefined JOIN order in read_sql_query()."""
    query = query.with_statement_hint("Set(join_collapse_limit 1)")
    return await read_sql_query(query, db, columns=columns, index=index)
