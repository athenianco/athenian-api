from datetime import datetime, timezone
from typing import Sequence, Union

import databases
import pandas as pd
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.sql import ClauseElement

from athenian.api.models.metadata.github import Base as MetadataBase
from athenian.api.models.state.models import Base as StateBase


async def read_sql_query(sql: ClauseElement,
                         con: Union[databases.Database, databases.core.Connection],
                         columns: Union[Sequence[str], Sequence[InstrumentedAttribute],
                                        MetadataBase, StateBase],
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

    Returns
    -------
    DataFrame

    Notes
    -----
    Any datetime values with time zone information parsed via the `parse_dates`
    parameter will be converted to UTC.
    """
    data = await con.fetch_all(query=sql)
    try:
        probe = columns[0]
    except TypeError:
        columns = [c.name for c in columns.__table__.columns]
    else:
        if not isinstance(probe, str):
            columns = [c.key for c in columns]
    frame = pd.DataFrame.from_records(data, columns=columns, coerce_float=True)
    for col in frame.select_dtypes(include=[object]):
        frame[col].replace(datetime(1, 1, 1, tzinfo=timezone.utc), pd.NaT, inplace=True)
        frame[col].replace(datetime(1, 1, 1), pd.NaT, inplace=True)
    for col in frame.select_dtypes(include=["datetime"]):
        frame[col].replace(0, pd.NaT, inplace=True)
        try:
            frame[col] = frame[col].dt.tz_localize(timezone.utc)
        except TypeError:
            continue
    return frame
