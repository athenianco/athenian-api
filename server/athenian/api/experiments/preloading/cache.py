import asyncio
from typing import Dict, List, Optional, Union

import pandas as pd
import sqlalchemy as sa
from sqlalchemy import select

from athenian.api.async_utils import read_sql_query
from athenian.api.typing_utils import DatabaseLike


def _humanize_size(v):
    unit = ""
    for u in ["B", "KiB", "MiB", "GiB"]:
        unit = u
        if v < 1024:
            break
        v = v / 1024

    return f"{round(v, 2)} {unit}"


class CachedDataFrame:
    """A DataFrame that is loaded and can be refreshed from the database.

    This can be read as a normal `pandas.DataFrame`.

    """

    def __init__(
        self,
        id_: str,
        db: DatabaseLike,
        cols: Optional[List[sa.Column]] = None,
        categorical_cols: Optional[List[sa.Column]] = None,
        identifier_cols: Optional[List[sa.Column]] = None,
        filtering_clause: Optional[sa.sql.ClauseElement] = None,
        debug_memory: Optional[bool] = False,
    ):
        """Initialize a `CachedDataFrame`.

        The wrapped `pandas.DataFrame` can be accessed through the `df` property and
        its dtypes are processed in order to reduce the memory usage.

        :param cols: Columns to include.
        :param categorical_cols: Columns that can be cast to `category` dtype
        :param identifier_cols: Columns that can be cast to bytes, `S` dtype
        """
        self._id = id_
        self._cols = cols or ["*"]
        self._categorical_cols = categorical_cols or []
        self._identifier_cols = identifier_cols or []
        self._filtering_clause = filtering_clause
        self._db = db
        self._debug_memory = debug_memory
        self._mem = {}
        self._df = None

    @property
    def id_(self) -> str:
        """Return the id of the `CachedDataFrame`."""
        return self._id

    @property
    def df(self) -> pd.DataFrame:
        """Return the wrapped `pandas.DataFrame`."""
        assert self._df is not None, "Need to await refresh() at least once."
        return self._df

    async def refresh(self) -> "CachedDataFrame":
        """Refresh the DataFrame from the database."""
        query = select(self._cols)
        if self._filtering_clause:
            query = query.where(self._filtering_clause)

        df = await read_sql_query(query, self._db, self._cols)
        if self._debug_memory:
            size = df.memory_usage(index=True, deep=True)
            self._mem["raw"] = {"series": size, "total": sum(size)}

        self._df = self._squeeze(df)
        if self._debug_memory:
            size = self._df.memory_usage(index=True, deep=True)
            self._mem["processed"] = {"series": size, "total": sum(size)}
            self._mem["percentage"] = {
                "series": self._mem["processed"]["series"]
                / self._mem["raw"]["series"]
                * 100,
                "total": self._mem["processed"]["total"]
                / self._mem["raw"]["total"]
                * 100,
            }

        return self

    def memory_usage(
        self, total: bool = False, human: bool = False,
    ) -> Union[pd.Series, int, str]:
        """Return information about the memory usage."""
        s = self._df.memory_usage(index=True, deep=True)
        if total:
            size = sum(s)
            return size if not human else _humanize_size(size)
        else:
            return s if not human else s.apply(_humanize_size)

    def _squeeze(self, df: pd.DataFrame) -> pd.DataFrame:
        def _cast_category(df: pd.DataFrame, cols: List[sa.Column]) -> pd.DataFrame:
            for c in cols:
                df[c.name] = df[c.name].astype("category")
            return df

        def _cast_identifiers(df: pd.DataFrame, cols: List[sa.Column]) -> pd.DataFrame:
            for c in cols:
                df[c.name] = df[c.name].astype("S")
            return df

        return df.pipe(_cast_category, self._categorical_cols).pipe(
            _cast_identifiers, self._identifier_cols,
        )

    def __str__(self):
        """Return the summary of the `CachedDataFrame`."""

        def fmt_cols(cols):
            return ", ".join(c.name for c in cols)

        metadata = [
            f"Table: {self._db.url}",
            f"Columns: {fmt_cols(self._cols)}",
            f"Categorical columns: {fmt_cols(self._categorical_cols)}",
            f"Identifier columns: {fmt_cols(self._identifier_cols)}",
        ]

        lines = metadata + ["DataFrame", str(self._df)]
        return "\n".join(lines)


class MemoryCache:
    """A singleton in-memory cache for instances of `CachedDataFrame`."""

    _instance = None

    def __init__(
        self,
        mdb: DatabaseLike,
        pdb: DatabaseLike,
        options: Dict[str, Dict],
        debug_memory: Optional[bool] = False,
    ):
        """Initialize a `MemoryCache`."""
        self._mdb = mdb
        self._pdb = pdb
        self._options = options
        self._debug_memory = debug_memory
        self._dfs = {}

    @classmethod
    def get_instance(
        cls,
        mdb: Optional[DatabaseLike] = None,
        pdb: Optional[DatabaseLike] = None,
        options: Optional[Dict[str, Dict]] = None,
        debug_memory: Optional[bool] = False,
    ) -> "MemoryCache":
        """Return the singleton instance."""
        if cls._instance is None:
            cls._instance = cls(mdb, pdb, options, debug_memory=debug_memory)
        return cls._instance

    @property
    def dfs(self) -> Dict[str, CachedDataFrame]:
        """Return all the `CachedDataFrame`s."""
        return self._dfs

    def memory_usage(
        self, total: bool = False, human: bool = False,
    ) -> Union[int, Dict]:
        """Return information about the memory usage."""
        size_by_df = {
            k: df.memory_usage(total=True, human=human and not total)
            for k, df in self.dfs.items()
        }

        if total:
            size = sum(size_by_df.values())
            return size if not human else _humanize_size(size)
        else:
            return size_by_df

    async def refresh(self, id_: Optional[str] = None):
        """Refresh the DataFrames from the database."""
        if id_:
            await self._dfs[id_].refresh()
        else:
            await asyncio.gather(*[df.refresh() for df in self._dfs.values()])

    async def load(self):
        """Load all the DataFrames from the database."""
        for id_, opts in self._options.items():
            opts["debug_memory"] = self._debug_memory
            self._dfs[id_] = CachedDataFrame(id_, **opts)

        await asyncio.gather(*[cdf.refresh() for cdf in self._dfs.values()])


def build_memory_cache_options(mdb: DatabaseLike, pdb: DatabaseLike) -> Dict[str, Dict]:
    """Return the options for the MemoryCache."""
    # TODO
    return {}
    # Example configuration:
    # {
    #     "prs": {
    #         "db": mdb,
    #         "cols": [
    #             PullRequest.acc_id,
    #             PullRequest.node_id,
    #             PullRequest.closed,
    #             PullRequest.closed_at,
    #             PullRequest.created_at,
    #             PullRequest.merged,
    #             PullRequest.merged_at,
    #             PullRequest.merged_by_login,
    #             PullRequest.repository_full_name,
    #             PullRequest.updated_at,
    #             PullRequest.user_login,
    #             PullRequest.hidden,
    #         ],
    #         "categorical_cols": [
    #             PullRequest.acc_id,
    #             PullRequest.merged_by_login,
    #             PullRequest.repository_full_name,
    #             PullRequest.user_login,
    #         ],
    #         "identifier_cols": [PullRequest.node_id],
    #     },
    # }
