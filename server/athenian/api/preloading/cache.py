import asyncio
from enum import Enum
import logging
from typing import Collection, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import sentry_sdk
import sqlalchemy as sa
from sqlalchemy import select
from sqlalchemy.orm.attributes import InstrumentedAttribute

from athenian.api import metadata
from athenian.api.async_utils import gather, read_sql_query
from athenian.api.models.metadata.github import Branch, NodePullRequestJiraIssues, \
    PullRequest
from athenian.api.models.precomputed.models import \
    GitHubDonePullRequestFacts, \
    GitHubRelease as PrecomputedRelease, \
    GitHubReleaseMatchTimespan as PrecomputedGitHubReleaseMatchTimespan
from athenian.api.models.state.models import AccountFeature, AccountGitHubAccount, Feature
from athenian.api.typing_utils import DatabaseLike


PRELOADING_FEATURE_FLAG_NAME = "github_features_entries_preloading"


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
        sharding: Optional[Dict] = None,
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
        self._sharding = sharding
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

    @property
    def filtering_clause(self) -> sa.sql.ClauseElement:
        """Return the filtering clause used for refreshing the data."""
        return self._filtering_clause

    @filtering_clause.setter
    def filtering_clause(self, fc: sa.sql.ClauseElement):
        self._filtering_clause = fc

    async def refresh(self) -> "CachedDataFrame":
        """Refresh the DataFrame from the database."""
        query = select(self._cols)
        if self._filtering_clause is not None:
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
        with sentry_sdk.start_span(op=f"CachedDataFrame.filter/{self._id}"):
            df = self._df.take(np.flatnonzero(mask))
            if columns:
                df = df[columns]
            if uncast:
                for col in self._identifier_cols:
                    try:
                        df[col.key] = df[col.key].values.astype("U")
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
        sdb: DatabaseLike,
        db: DatabaseLike,
        options: Dict[str, Dict],
        debug_memory: Optional[bool] = False,
    ):
        """Initialize a `MemoryCache`."""
        self._sdb = sdb
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

    async def refresh(self, id_: Optional[str] = None) -> None:
        """Refresh the DataFrames from the database."""
        if id_:
            df = self._dfs[id_]
            df.filtering_clause = await self._build_filtering_clause(id_)
            await df.refresh()
        else:
            tasks = []
            accounts = await self._get_enabled_accounts()
            for id_, df in self._dfs.items():
                df.filtering_clause = await self._build_filtering_clause(id_, accounts=accounts)
                tasks.append(df.refresh())

            await gather(*tasks)

    async def _build_filtering_clause(
            self, id_: str, accounts: Optional[Dict[str, Collection[int]]] = None,
    ) -> sa.sql.ClauseElement:
        if not accounts:
            accounts = await self._get_enabled_accounts()

        opts = self._options[id_]
        sharding_opts = opts["sharding"]
        filtering_clause = sharding_opts["column"].in_(accounts[sharding_opts["key"]])
        if fc := opts.get("filtering_clause") is not None:
            filtering_clause = sa.and_(filtering_clause, fc)

        return filtering_clause

    async def _get_enabled_accounts(self) -> Dict[str, List[int]]:
        query = (
            sa.select([AccountFeature.account_id])
            .select_from(sa.join(Feature, AccountFeature,
                                 Feature.id == AccountFeature.feature_id))
            .where(sa.and_(AccountFeature.enabled,
                           Feature.name == PRELOADING_FEATURE_FLAG_NAME))
        )
        acc_ids = [r[0] for r in await self._sdb.fetch_all(query)]
        query = sa.select([AccountGitHubAccount.id]).where(
            AccountGitHubAccount.account_id.in_(acc_ids))
        meta_ids = [r[0] for r in await self._sdb.fetch_all(query)]

        return {"acc_id": acc_ids, "meta_id": meta_ids}


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
            db.cache = mc = MemoryCache(dbs["sdb"], db, opts, debug_memory=debug_mode)
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
    branches = Branch.__table__.fullname


class PCID(str, Enum):
    """Identifiers of the cached tables from precomputed DB."""

    releases = PrecomputedRelease.__table__.fullname
    releases_match_timespan = PrecomputedGitHubReleaseMatchTimespan.__table__.fullname
    done_pr_facts = GitHubDonePullRequestFacts.__table__.fullname


def get_memory_cache_options() -> Dict[str, Dict[str, Dict[str, List[InstrumentedAttribute]]]]:
    """Return the options for the MemoryCache."""
    return {
        "mdb": {
            MCID.prs.value: {
                "sharding": {
                    "column": PullRequest.acc_id,
                    "key": "meta_id",
                },
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
                "sharding": {
                    "column": NodePullRequestJiraIssues.node_acc,
                    "key": "meta_id",
                },
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
            MCID.branches.value: {
                "sharding": {
                    "column": Branch.acc_id,
                    "key": "meta_id",
                },
                "cols": [
                    Branch.repository_full_name,
                    Branch.acc_id,
                    Branch.is_default,
                    Branch.branch_name,
                    Branch.commit_sha,
                    Branch.commit_id,
                ],
                "categorical_cols": [
                    Branch.repository_full_name,
                    Branch.acc_id,
                ],
                "identifier_cols": [
                    Branch.commit_sha,
                    Branch.commit_id,
                ],
            },
        },
        "pdb": {
            PCID.releases.value: {
                "sharding": {
                    "column": PrecomputedRelease.acc_id,
                    "key": "acc_id",
                },
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
            PCID.releases_match_timespan.value: {
                "sharding": {
                    "column": PrecomputedGitHubReleaseMatchTimespan.acc_id,
                    "key": "acc_id",
                },
                "cols": [
                    PrecomputedGitHubReleaseMatchTimespan.release_match,
                    PrecomputedGitHubReleaseMatchTimespan.repository_full_name,
                    PrecomputedGitHubReleaseMatchTimespan.acc_id,
                    PrecomputedGitHubReleaseMatchTimespan.time_from,
                    PrecomputedGitHubReleaseMatchTimespan.time_to,
                ],
                "categorical_cols": [
                    PrecomputedGitHubReleaseMatchTimespan.release_match,
                    PrecomputedGitHubReleaseMatchTimespan.repository_full_name,
                    PrecomputedGitHubReleaseMatchTimespan.acc_id,
                ],
            },
            PCID.done_pr_facts.value: {
                "sharding": {
                    "column": GitHubDonePullRequestFacts.acc_id,
                    "key": "acc_id",
                },
                "filtering_clause": (
                    GitHubDonePullRequestFacts.format_version ==
                    GitHubDonePullRequestFacts.__table__.columns[
                        GitHubDonePullRequestFacts.format_version.key].default.arg,
                ),
                "cols": [
                    GitHubDonePullRequestFacts.pr_node_id,
                    GitHubDonePullRequestFacts.repository_full_name,
                    GitHubDonePullRequestFacts.release_match,
                    GitHubDonePullRequestFacts.acc_id,
                    GitHubDonePullRequestFacts.pr_created_at,
                    GitHubDonePullRequestFacts.pr_done_at,
                    GitHubDonePullRequestFacts.activity_days,
                    GitHubDonePullRequestFacts.data,
                    GitHubDonePullRequestFacts.author,
                    GitHubDonePullRequestFacts.merger,
                    GitHubDonePullRequestFacts.releaser,
                    GitHubDonePullRequestFacts.reviewers,
                    GitHubDonePullRequestFacts.commenters,
                    GitHubDonePullRequestFacts.commit_authors,
                    GitHubDonePullRequestFacts.commit_committers,
                ],
                "categorical_cols": [
                    GitHubDonePullRequestFacts.repository_full_name,
                    GitHubDonePullRequestFacts.release_match,
                    GitHubDonePullRequestFacts.acc_id,
                    GitHubDonePullRequestFacts.author,
                    GitHubDonePullRequestFacts.merger,
                    GitHubDonePullRequestFacts.releaser,
                    GitHubDonePullRequestFacts.reviewers,
                    GitHubDonePullRequestFacts.commenters,
                    GitHubDonePullRequestFacts.commit_authors,
                    GitHubDonePullRequestFacts.commit_committers,
                ],
                "identifier_cols": [
                    GitHubDonePullRequestFacts.pr_node_id,
                ],
            },
        },
    }
