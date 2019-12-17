import databases
import pandas
from pandas.io.sql import _wrap_result
from sqlalchemy.sql import ClauseElement


async def read_sql_query(
    sql: ClauseElement,
    con: databases.Database,
    index_col=None,
    coerce_float=True,
    parse_dates=None,
) -> pandas.DataFrame:
    """Read SQL query into a DataFrame.

    Returns a DataFrame corresponding to the result set of the query.
    Optionally provide an `index_col` parameter to use one of the
    columns as the index, otherwise default integer index will be used.

    Parameters
    ----------
    sql : SQLAlchemy query object to be executed.
    con : async SQLAlchemy database engine.
    index_col : string or list of strings, optional, default: None
        Column(s) to set as index(MultiIndex).
    coerce_float : boolean, default True
        Attempts to convert values of non-string, non-numeric objects (like
        decimal.Decimal) to floating point. Useful for SQL result sets.
    parse_dates : list or dict, default: None
        - List of column names to parse as dates.
        - Dict of ``{column_name: format string}`` where format string is
          strftime compatible in case of parsing string times, or is one of
          (D, s, ns, ms, us) in case of parsing integer timestamps.
        - Dict of ``{column_name: arg dict}``, where the arg dict corresponds
          to the keyword arguments of :func:`pandas.to_datetime`
          Especially useful with databases without native Datetime support,
          such as SQLite.

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
    frame = _wrap_result(
        data,
        columns,
        index_col=index_col,
        coerce_float=coerce_float,
        parse_dates=parse_dates,
    )
    return frame
