import asyncio
from enum import Enum
import logging
from typing import Collection, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import select
from sqlalchemy.orm.attributes import InstrumentedAttribute

from athenian.api import metadata
from athenian.api.async_utils import gather, read_sql_query
from athenian.api.models.metadata.github import NodePullRequestJiraIssues, PullRequest
from athenian.api.models.precomputed.models import GitHubRelease as PrecomputedRelease
from athenian.api.typing_utils import DatabaseLike


def _humanize_size(v):
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

    def filter(
        self,
        mask: pd.Series,
        columns: Optional[Collection[str]] = None,
        index: Optional[str] = None,
        uncast: bool = True,
    ) -> pd.DataFrame:
        """Filter the wrapped `pandas.Datarame` with the provided mask.

        :param mask: boolean mask for filtering the wrapped DataFrame.
        :param columns: the columns to select.
        :param index: the column to set as index.
        :param uncast: whether to uncast the columns dtypes, see the `_squeeze` method.

        """
        df = self._df.take(np.flatnonzero(mask))
        if columns:
            df = df[columns]
        if uncast:
            for col in self._identifier_cols:
                try:
                    df[col.key] = df[col.key].apply(lambda s: s.decode("utf8"))
                except KeyError:
                    continue

            for col in self._categorical_cols:
                try:
                    df[col.key] = df[col.key].astype("object").replace({np.nan: None})
                except KeyError:
                    continue
        if index:
            df.set_index(index, inplace=True)

        return df

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
    """An in-memory cache of a specific db for instances of `CachedDataFrame`."""

    def __init__(
        self,
        db: DatabaseLike,
        options: Dict[str, Dict],
        debug_memory: Optional[bool] = False,
    ):
        """Initialize a `MemoryCache`."""
        self._db = db
        self._options = options
        self._debug_memory = debug_memory
        self._dfs = {
            id_: CachedDataFrame(id_, **opts, db=self._db, debug_memory=self._debug_memory)
            for id_, opts in self._options.items()
        }

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
            await gather(*[df.refresh() for df in self._dfs.values()])


class MemoryCachePreloader:
    """Responsible for dealing with `MemoryCache` preloading and graceful shutdown."""

    _log = logging.getLogger(f"{metadata.__package__}.MemoryCachePreloader")

    def __init__(self):
        """Initialize a `MemoryCachePreloader`."""
        self._task = None
        self._debug_mode = self._log.isEnabledFor(logging.DEBUG)

    def preload(
        self, debug_memory: Optional[bool] = None, **dbs: Dict[str, DatabaseLike],
    ):
        """Preloads the `MemoryCache`."""
        self._task = asyncio.create_task(self._preload_all(debug_memory, **dbs))
        self._task.add_done_callback(self._done)

    async def shutdown(self, *_):
        """Graceful handles the eventually running preloading task."""
        if not self._task:
            return

        if not self._task.done():
            self._log.info("Cancelling MemoryCache preloading")
            self._task.cancel()

    async def _preload_all(self, debug_memory, **dbs: Dict[str, DatabaseLike]):
        self._log.info("Preloading MemoryCache")
        debug_mode = self._debug_mode if debug_memory is None else debug_memory
        tasks = []
        for db_name, opts in get_memory_cache_options().items():
            db = dbs[db_name]
            db.cache = mc = MemoryCache(db, opts, debug_memory=debug_mode)
            tasks.append(mc.refresh())

        await gather(*tasks)

    def _done(self, *_):
        if self._task.cancelled():
            self._log.info("MemoryCache preloading cancelled")
        elif self._task.exception():
            raise self._task.exception()
        else:
            self._log.info("MemoryCache ready")


class MCID(str, Enum):
    """Identifiers of the cached tables from metadata DB."""

    prs = PullRequest.__table__.fullname
    jira_mapping = NodePullRequestJiraIssues.__table__.fullname


class PCID(str, Enum):
    """Identifiers of the cached tables from precomputed DB."""

    releases = PrecomputedRelease.__table__.fullname


def get_memory_cache_options() -> Dict[str, Dict[str, Dict[str, List[InstrumentedAttribute]]]]:
    """Return the options for the MemoryCache."""
    return {
        "mdb": {
            MCID.prs.value: {
                "cols": [
                    PullRequest.acc_id,
                    PullRequest.node_id,
                    PullRequest.number,
                    PullRequest.closed,
                    PullRequest.closed_at,
                    PullRequest.created_at,
                    PullRequest.merged,
                    PullRequest.merge_commit_id,
                    PullRequest.merge_commit_sha,
                    PullRequest.merged_at,
                    PullRequest.merged_by_login,
                    PullRequest.repository_full_name,
                    PullRequest.updated_at,
                    PullRequest.user_login,
                    PullRequest.additions,
                    PullRequest.deletions,
                ],
                "categorical_cols": [
                    PullRequest.acc_id,
                    PullRequest.merged_by_login,
                    PullRequest.repository_full_name,
                    PullRequest.user_login,
                ],
                "identifier_cols": [
                    PullRequest.node_id,
                    PullRequest.merge_commit_id,
                    PullRequest.merge_commit_sha,
                ],
            },
            MCID.jira_mapping.value: {
                "cols": [
                    NodePullRequestJiraIssues.node_id,
                    NodePullRequestJiraIssues.node_acc,
                    NodePullRequestJiraIssues.jira_id,
                ],
                "categorical_cols": [
                    NodePullRequestJiraIssues.node_acc,
                    NodePullRequestJiraIssues.jira_id,
                ],
                "identifier_cols": [NodePullRequestJiraIssues.node_id],
            },
        },
        "pdb": {
            PCID.releases.value: {
                "cols": [
                    PrecomputedRelease.release_match,
                    PrecomputedRelease.repository_full_name,
                    PrecomputedRelease.published_at,
                    PrecomputedRelease.acc_id,
                ],
                "categorical_cols": [
                    PrecomputedRelease.acc_id,
                    PrecomputedRelease.repository_full_name,
                    PrecomputedRelease.release_match,
                ],
            },
        },
    }
