import databases
import pandas as pd
from sqlalchemy.sql import ClauseElement


async def read_sql_query(sql: ClauseElement, con: databases.Database) -> pd.DataFrame:
    """Read SQL query into a DataFrame.

    Returns a DataFrame corresponding to the result set of the query.
    Optionally provide an `index_col` parameter to use one of the
    columns as the index, otherwise default integer index will be used.

    Parameters
    ----------
    sql : SQLAlchemy query object to be executed.
    con : async SQLAlchemy database engine.

    Returns
    -------
    DataFrame

    Notes
    -----
    Any datetime values with time zone information parsed via the `parse_dates`
    parameter will be converted to UTC.
    """
    data = await con.fetch_all(query=sql)
    columns = data[0].keys() if data else []
    frame = pd.DataFrame.from_records(data, columns=columns, coerce_float=True)
    return frame
