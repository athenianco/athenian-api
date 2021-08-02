from datetime import datetime, timezone
from enum import Enum
import logging
from typing import Collection, Dict, Iterable, List, NamedTuple, Optional, Union

import databases
import numpy as np
import pandas as pd
import prometheus_client
import sentry_sdk
import sqlalchemy as sa
from sqlalchemy import select
from sqlalchemy.orm.attributes import InstrumentedAttribute

from athenian.api import aiocron
from athenian.api import metadata
from athenian.api.async_utils import gather, read_sql_query
from athenian.api.models.metadata.github import Branch, NodePullRequestJiraIssues, \
    PullRequest
from athenian.api.models.precomputed.models import \
    GitHubDonePullRequestFacts, GitHubMergedPullRequestFacts, GitHubOpenPullRequestFacts, \
    GitHubRelease as PrecomputedRelease, \
    GitHubReleaseMatchTimespan as PrecomputedGitHubReleaseMatchTimespan
from athenian.api.models.state.models import AccountFeature, AccountGitHubAccount, Feature


PRELOADING_FEATURE_FLAG_NAME = "github_features_entries_preloading"

Gauges = NamedTuple("Gauges", [("memory", prometheus_client.Gauge),
                               ("timing", prometheus_client.Gauge)])


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

    _log = logging.getLogger(f"{metadata.__package__}.CachedDataFrame")

    def __init__(
        self,
        id_: str,
        db: databases.Database,
        sharding: Dict,
        cols: Optional[List[sa.Column]] = None,
        categorical_cols: Optional[List[sa.Column]] = None,
        identifier_cols: Optional[List[sa.Column]] = None,
        filtering_clause: Optional[sa.sql.ClauseElement] = None,
        coerce_cols: Optional[Dict] = None,
        gauge: Optional[prometheus_client.Gauge] = None,
        debug_memory: Optional[bool] = False,
    ):
        """Initialize a `CachedDataFrame`.

        The wrapped `pandas.DataFrame` can be accessed through the `df` property and
        its dtypes are processed in order to reduce the memory usage.

        :param cols: Columns to include.
        :param categorical_cols: Columns that can be cast to `category` dtype
        :param identifier_cols: Columns that can be cast to bytes, `S` dtype
        """
        assert isinstance(db, databases.Database)
        self._id = id_
        self._cols = cols or ["*"]
        self._categorical_cols = categorical_cols or []
        self._identifier_cols = identifier_cols or []
        self._filtering_clause = filtering_clause
        self._sharding = sharding
        self._coerce_cols = coerce_cols or {}
        self._db = db
        self._debug_memory = debug_memory
        self._mem = {}
        self._columns = None
        self._sharded_dfs = {}
        self._metrics = {}
        self._gauge = gauge

    @property
    def id_(self) -> str:
        """Return the id of the `CachedDataFrame`."""
        return self._id

    @property
    def filtering_clause(self) -> sa.sql.ClauseElement:
        """Return the filtering clause used for refreshing the data."""
        return self._filtering_clause

    @filtering_clause.setter
    def filtering_clause(self, fc: sa.sql.ClauseElement):
        self._filtering_clause = fc

    def get_dfs(self, sharding_keys: Iterable[Union[int, str]]) -> pd.DataFrame:
        """Return the wrapped `pandas.DataFrame`."""
        assert self._columns is not None, "Need to await refresh() at least once."

        def _build_empty_df():
            return pd.DataFrame(columns=self._columns)

        if not self._sharded_dfs:
            return _build_empty_df()

        return pd.concat(self._sharded_dfs.get(sk, _build_empty_df())
                         for sk in sorted(sharding_keys))

    async def refresh(self) -> "CachedDataFrame":
        """Refresh the DataFrame from the database."""
        query = select(self._cols)
        if self._filtering_clause is not None:
            query = query.where(self._filtering_clause)

        self._log.debug("Refreshing %s", self._id)
        raw_full_df = await read_sql_query(query, self._db, self._cols)

        if self._debug_memory:
            sizes = raw_full_df.memory_usage(index=True, deep=True)
            self._mem["raw"] = {"series": sizes, "total": sizes.sum()}

        squeezed_full_df = self._squeeze(self._coerce(raw_full_df))
        del raw_full_df

        if self._debug_memory or self._gauge is not None:
            sizes = squeezed_full_df.memory_usage(index=True, deep=True)
            for key, val in sizes.items():
                self._gauge.labels(
                    metadata.__package__, metadata.__version__,
                    self._db.url.database, self._id, key,
                ).set(val)
            self._gauge.labels(
                metadata.__package__, metadata.__version__,
                self._db.url.database, self._id, "__all__",
            ).set(sizes.sum())
        if self._debug_memory:
            self._mem["processed"] = {"series": sizes, "total": sizes.sum()}
            self._mem["percentage"] = {
                "series": self._mem["processed"]["series"]
                / self._mem["raw"]["series"]
                * 100,
                "total": self._mem["processed"]["total"]
                / self._mem["raw"]["total"]
                * 100,
            }

        self._columns = squeezed_full_df.columns.to_list()
        self._sharded_dfs = dict(tuple(
            squeezed_full_df.groupby(self._sharding["column"].name)))

        return self

    def filter(
        self,
        sharding_keys: Iterable[Union[int, str]],
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
            df = self.get_dfs(sharding_keys).take(np.flatnonzero(mask))
            if columns:
                df = df[columns]
            if uncast:
                for col in self._identifier_cols:
                    try:
                        df[col.name] = df[col.name].values.astype("U")
                    except KeyError:
                        continue

                for col in self._categorical_cols:
                    try:
                        df[col.name] = df[col.name].astype("object").replace({np.nan: None})
                    except KeyError:
                        continue
            if index:
                df.set_index(index, inplace=True)

            return df

    def memory_usage(
        self, total: bool = False, human: bool = False,
    ) -> Union[pd.Series, int, str]:
        """Return information about the memory usage."""
        if not self._sharded_dfs:
            s = [0]
        else:
            s = pd.concat(self._sharded_dfs.values()).memory_usage(index=True, deep=True)
        if total:
            size = sum(s)
            return size if not human else _humanize_size(size)
        else:
            return s if not human else s.apply(_humanize_size)

    def _coerce(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, coerce_opts in self._coerce_cols.items():
            if isinstance(coerce_opts, dict):
                if not (coerce_func := coerce_opts.get(self._db.url.dialect)):
                    continue
            else:
                coerce_func = coerce_opts

            df[col.name] = coerce_func(df[col.name])

        return df

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

    _log = logging.getLogger(f"{metadata.__package__}.MemoryCache")

    def __init__(
        self,
        sdb: databases.Database,
        db: databases.Database,
        options: Dict[str, Dict],
        gauges: Optional[Gauges],
        debug_memory: Optional[bool],
    ):
        """Initialize a `MemoryCache`."""
        assert isinstance(db, databases.Database)
        self._sdb = sdb
        self._db = db
        self._options = options
        self._debug_memory = debug_memory
        self._gauge = gauges.timing if gauges else None
        self._enabled_accounts = {}
        self._dfs = {
            id_: CachedDataFrame(
                id_, **opts, db=self._db, gauge=gauges.memory if gauges else None,
                debug_memory=self._debug_memory)
            for id_, opts in self._options.items()
        }

    @property
    def dfs(self) -> Dict[str, CachedDataFrame]:
        """Return all the `CachedDataFrame`s."""
        return self._dfs

    def is_account_loaded(self, acc_id: int) -> bool:
        """Return whether the data for the given account is loaded."""
        for enabled_accounts in self._enabled_accounts.values():
            if acc_id not in enabled_accounts["acc_id"]:
                return False

        return True

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
        start = datetime.utcnow()
        table = id_ or "__all__"
        accounts = await self._get_enabled_accounts()
        if id_:
            refreshed = 1
            df = self._dfs[id_]
            df.filtering_clause = await self._build_filtering_clause(id_, accounts)
            await df.refresh()
            self._enabled_accounts[id_] = accounts
        else:
            tasks = []
            refreshed = len(self._dfs)
            for id_, df in self._dfs.items():
                df.filtering_clause = await self._build_filtering_clause(id_, accounts)
                tasks.append(df.refresh())

            await gather(*tasks)

            for id_ in self._dfs.keys():
                self._enabled_accounts[id_] = accounts

        elapsed = (datetime.utcnow() - start).total_seconds()
        if self._gauge is not None:
            self._gauge.labels(
                metadata.__package__, metadata.__version__,
                self._db.url.database, table,
            ).set(elapsed)

        memory_used = self.memory_usage(total=True, human=True)
        self._log.info("Refreshed %d tables in %s in %f seconds, total memory: %s",
                       refreshed, self._db.url.database, elapsed, memory_used)

    async def _build_filtering_clause(
        self, id_: str, accounts: Dict[str, Collection[int]],
    ) -> sa.sql.ClauseElement:
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

    def __init__(self,
                 refresh_frequency_minutes: int,
                 max_early_refresh_minutes: Optional[int] = None,
                 prometheus_registry: Optional[prometheus_client.CollectorRegistry] = None,
                 debug_memory: Optional[bool] = None):
        """Initialize a `MemoryCachePreloader`."""
        self._refresh_frequency_minutes = refresh_frequency_minutes
        self._max_early_refresh_minutes = (
            max_early_refresh_minutes or refresh_frequency_minutes // 5
        )
        self._refresh_jobs = []
        if debug_memory is None:
            self._debug_memory = self._log.isEnabledFor(logging.DEBUG)
        else:
            self._debug_memory = debug_memory

        if prometheus_registry is not None:
            memory_gauge = prometheus_client.Gauge(
                "memory_cache_consumed_memory", "Consumed memory by MemoryCaches",
                ["app_name", "version", "db", "table", "column"],
                registry=prometheus_registry,
            )
            timing_gauge = prometheus_client.Gauge(
                "memory_cache_preloading_refresh_time_seconds",
                "Time required for refreshing the MemoryCaches",
                ["app_name", "version", "db", "table"],
                registry=prometheus_registry,
            )
            gauges = Gauges(memory_gauge, timing_gauge)
        else:
            gauges = None

        self._gauges = gauges

    async def preload(self, **dbs: databases.Database) -> None:
        """
        Load in memory the tables configured in `get_memory_cache_options()`.

        We set each db's "cache" attribute to the loaded `MemoryCache` object.
        """
        self._log.debug("Starting to preload DB tables in memory")
        tasks = []
        sdb = dbs["sdb"]
        for db_name, opts in get_memory_cache_options().items():
            db = dbs[db_name]
            db.cache = mc = MemoryCache(sdb, db, opts, self._gauges, self._debug_memory)
            tasks.append(mc.refresh())

        await gather(*tasks)
        self._log.info("Finished preloading DB tables in memory")

        self._log.debug("Scheduling refresh for preloaded DB tables in memory")
        for db_name in get_memory_cache_options():
            self._refresh_jobs.append(
                aiocron.crontab(
                    f"*/{self._refresh_frequency_minutes} * * * *",
                    func=_refresh, args=(dbs[db_name], ),
                    max_early_expiration_seconds=self._max_early_refresh_minutes),
            )

        self._log.info("Scheduled refresh for preloaded DB tables in memory each %d min",
                       self._refresh_frequency_minutes)

    async def stop(self) -> None:
        """Stop the refreshing jobs."""
        for j in self._refresh_jobs:
            j.stop()

        self._log.info("Refresh jobs for preloaded DB tables stopped")


async def _refresh(db: databases.Database) -> None:
    await db.cache.refresh()


def parse_sqlite_timestamps(col_series):
    """Parse strings as timestamps because SQLite does not have a native TIMESTAMP column type."""
    return [
        [datetime.strptime(v, "%Y-%m-%d").replace(tzinfo=timezone.utc)
         for v in r] for r in col_series
    ]


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
    open_pr_facts = GitHubOpenPullRequestFacts.__table__.fullname
    merged_pr_facts = GitHubMergedPullRequestFacts.__table__.fullname


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
                    PullRequest.merged_by_id,
                    PullRequest.repository_full_name,
                    PullRequest.updated_at,
                    PullRequest.user_login,
                    PullRequest.user_node_id,
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
                "identifier_cols": [],
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
                    PrecomputedRelease.node_id,
                    PrecomputedRelease.commit_id,
                    PrecomputedRelease.sha,
                    PrecomputedRelease.release_match,
                    PrecomputedRelease.repository_full_name,
                    PrecomputedRelease.published_at,
                    PrecomputedRelease.acc_id,
                    PrecomputedRelease.author,
                    PrecomputedRelease.url,
                ],
                "categorical_cols": [
                    PrecomputedRelease.acc_id,
                    PrecomputedRelease.repository_full_name,
                    PrecomputedRelease.release_match,
                ],
                "identifier_cols": [
                    PrecomputedRelease.sha,
                    PrecomputedRelease.url,
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
            PCID.open_pr_facts.value: {
                "sharding": {
                    "column": GitHubOpenPullRequestFacts.acc_id,
                    "key": "acc_id",
                },
                "filtering_clause": (
                    GitHubOpenPullRequestFacts.format_version ==
                    GitHubOpenPullRequestFacts.__table__.columns[
                        GitHubOpenPullRequestFacts.format_version.key].default.arg,
                ),
                "cols": [
                    GitHubOpenPullRequestFacts.pr_node_id,
                    GitHubOpenPullRequestFacts.repository_full_name,
                    GitHubOpenPullRequestFacts.acc_id,
                    GitHubOpenPullRequestFacts.pr_created_at,
                    GitHubOpenPullRequestFacts.pr_updated_at,
                    GitHubOpenPullRequestFacts.activity_days,
                    GitHubOpenPullRequestFacts.number,
                    GitHubOpenPullRequestFacts.data,
                ],
                "categorical_cols": [
                    GitHubOpenPullRequestFacts.repository_full_name,
                    GitHubOpenPullRequestFacts.acc_id,
                ],
                "identifier_cols": [],
                "coerce_cols": {
                    GitHubOpenPullRequestFacts.activity_days: {
                        "sqlite": parse_sqlite_timestamps,
                    },
                },
            },
            PCID.merged_pr_facts.value: {
                "sharding": {
                    "column": GitHubMergedPullRequestFacts.acc_id,
                    "key": "acc_id",
                },
                "filtering_clause": (
                    GitHubMergedPullRequestFacts.format_version ==
                    GitHubMergedPullRequestFacts.__table__.columns[
                        GitHubMergedPullRequestFacts.format_version.key].default.arg,
                ),
                "cols": [
                    GitHubMergedPullRequestFacts.pr_node_id,
                    GitHubMergedPullRequestFacts.repository_full_name,
                    GitHubMergedPullRequestFacts.data,
                    GitHubMergedPullRequestFacts.author,
                    GitHubMergedPullRequestFacts.merger,
                    GitHubMergedPullRequestFacts.checked_until,
                    GitHubMergedPullRequestFacts.acc_id,
                    GitHubMergedPullRequestFacts.labels,
                    GitHubMergedPullRequestFacts.activity_days,
                    GitHubMergedPullRequestFacts.release_match,
                ],
                "categorical_cols": [
                    GitHubMergedPullRequestFacts.repository_full_name,
                    GitHubMergedPullRequestFacts.acc_id,
                    GitHubMergedPullRequestFacts.release_match,
                ],
                "identifier_cols": [],
                "coerce_cols": {
                    GitHubMergedPullRequestFacts.activity_days: {
                        "sqlite": parse_sqlite_timestamps,
                    },
                },
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
                    GitHubDonePullRequestFacts.acc_id,
                    GitHubDonePullRequestFacts.repository_full_name,
                    GitHubDonePullRequestFacts.pr_created_at,
                    GitHubDonePullRequestFacts.pr_done_at,
                    GitHubDonePullRequestFacts.author,
                    GitHubDonePullRequestFacts.merger,
                    GitHubDonePullRequestFacts.releaser,
                    GitHubDonePullRequestFacts.commenters,
                    GitHubDonePullRequestFacts.reviewers,
                    GitHubDonePullRequestFacts.commit_authors,
                    GitHubDonePullRequestFacts.commit_committers,
                    GitHubDonePullRequestFacts.pr_node_id,
                    GitHubDonePullRequestFacts.release_match,
                    GitHubDonePullRequestFacts.activity_days,
                    GitHubDonePullRequestFacts.data,
                ],
                "categorical_cols": [
                    GitHubDonePullRequestFacts.acc_id,
                    GitHubDonePullRequestFacts.repository_full_name,
                    GitHubDonePullRequestFacts.release_match,
                ],
                "identifier_cols": [],
                "coerce_cols": {
                    GitHubDonePullRequestFacts.activity_days: {
                        "sqlite": parse_sqlite_timestamps,
                    },
                },
            },
        },
    }
