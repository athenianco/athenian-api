from datetime import datetime, timezone
import logging
import textwrap
from typing import Collection, Iterable, Mapping, Optional, Sequence, Union

import pandas as pd
from sqlalchemy import Column, DateTime
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.sql import ClauseElement

from athenian.api import metadata
from athenian.api.models.metadata.github import Base as MetadataBase
from athenian.api.models.state.models import Base as StateBase
from athenian.api.tracing import MAX_SENTRY_STRING_LENGTH
from athenian.api.typing_utils import DatabaseLike


async def read_sql_query(sql: ClauseElement,
                         con: DatabaseLike,
                         columns: Union[Sequence[str], Sequence[InstrumentedAttribute],
                                        MetadataBase, StateBase],
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


def wrap_sql_query(data: Iterable[Mapping],
                   columns: Union[Sequence[str], Sequence[InstrumentedAttribute],
                                  MetadataBase, StateBase],
                   index: Optional[Union[str, Sequence[str]]] = None,
                   ) -> pd.DataFrame:
    """Turn the fetched DB records to a pandas DataFrame."""
    try:
        probe = columns[0]
    except TypeError:
        dt_columns = _extract_datetime_columns(columns.__table__.columns)
        columns = [c.name for c in columns.__table__.columns]
    else:
        if not isinstance(probe, str):
            dt_columns = _extract_datetime_columns(columns)
            columns = [c.key for c in columns]
        else:
            dt_columns = None
    frame = pd.DataFrame.from_records((r.values() for r in data),
                                      columns=columns, coerce_float=True)
    if index is not None:
        frame.set_index(index, inplace=True)
    return postprocess_datetime(frame, columns=dt_columns)


def _extract_datetime_columns(columns: Iterable[Column]) -> Collection[str]:
    return [c.name for c in columns if isinstance(c.type, DateTime)
            or (isinstance(c.type, type) and issubclass(c.type, DateTime))]


def postprocess_datetime(frame: pd.DataFrame,
                         columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
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
