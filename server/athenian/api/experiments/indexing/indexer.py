import bisect
from datetime import datetime, timezone
from enum import Enum
from functools import reduce
from typing import Dict

import numpy as np
import pandas as pd

from athenian.api.async_utils import gather


class LayerLevel(Enum):
    """Represent a the granularity level of a layer in the indexer."""

    MINUTE = 0
    HOUR = 1
    DAY = 2
    MONTH = 3
    QUARTER = 4

    @classmethod
    def levels(cls):
        """Return the levels mapping by values."""
        return {v.value: v for k, v in cls.__members__.items()}

    @classmethod
    def min(cls):
        """Return the minimum level."""
        levels = cls.levels()
        return levels[min(levels)]

    @classmethod
    def max(cls):
        """Return the maximum level."""
        levels = cls.levels()
        return levels[max(levels)]

    def get_deeper_level(self):
        """Return the next lower level."""
        return self.levels()[self.value - 1]

    def get_shallower_level(self):
        """Return the next higher level."""
        return self.levels()[self.value + 1]


class BaseIndexer:
    """Base logic of an indexer."""

    __instance = None
    __slots__ = ["_layers", "_virtual_indexes", "_last_update"]

    def __init__(
        self,
        layers: Dict[LayerLevel, np.ndarray],
        virtual_indexes: Dict[LayerLevel, np.ndarray],
        last_update: datetime,
    ):
        """Initialize an indexer with the provided input."""
        self._layers = layers
        self._virtual_indexes = virtual_indexes
        self._last_update = last_update

    @property
    def last_update(self):
        """Return the last update of the indexer."""
        return self._last_update

    @classmethod
    async def get_instance(cls, mdb_conn):
        """Return or create the singleton indexer instance."""
        if cls.__instance is not None:
            return cls.__instance

        return await cls.create(mdb_conn)

    @classmethod
    async def create(cls, mdb_conn):
        """Create an indexer."""
        return cls(*(await cls._get_init_params(mdb_conn)))

    async def full_refresh(self, mdb_conn):
        """Update the indexer by resetting the data and fully refreshing the layers."""
        layers, virtual_indexes, last_update = await self._get_init_params(mdb_conn)
        self._layers = layers
        self._virtual_indexes = virtual_indexes
        self._last_update = last_update

    async def refresh(self, mdb_conn):
        """Update the indexer by adding new data."""
        # same query, but created_at >= last_update
        raise NotImplementedError()

    def search(self, date_from, date_to, return_counts=False):
        """Seach for the data in the provided interval."""
        vi_index_from = self._indexify(date_from)
        vi_index_to = self._indexify(date_to)

        v = self._search_in_layer(LayerLevel.max(), vi_index_from, vi_index_to)
        return np.unique(v[np.nonzero(v)], return_counts=return_counts)

    def _search_in_layer(self, layer_level, vi_index_from, vi_index_to):
        index_from = bisect.bisect_left(self._virtual_indexes[layer_level], vi_index_from)
        index_to = bisect.bisect_left(self._virtual_indexes[layer_level], vi_index_to) - 1

        if layer_level.value == 0:
            index_to += 1
            if index_from >= index_to:
                return np.array([], dtype="object")

            postings_lists = self._layers[layer_level][index_from:index_to]
            v = np.concatenate(postings_lists)
            print(
                f"Found {len(postings_lists)} to concatenate in layer "
                f"{layer_level.name} (total items: {len(v)})",
            )
            return v

        if index_from >= index_to:
            return self._search_in_layer(
                layer_level.get_deeper_level(), vi_index_from, vi_index_to,
            )

        left = self._search_in_layer(
            layer_level.get_deeper_level(),
            vi_index_from,
            self._virtual_indexes[layer_level][index_from],
        )
        right = self._search_in_layer(
            layer_level.get_deeper_level(),
            self._virtual_indexes[layer_level][index_to],
            vi_index_to,
        )
        postings_lists = self._layers[layer_level][index_from:index_to]
        center = np.concatenate(postings_lists)
        v = np.concatenate([left, center, right])
        print(
            f"Found {len(postings_lists)} to concatenate in layer "
            f"{layer_level.name} (total items: {len(center)} | grand total items: {len(v)})",
        )
        return v

    @classmethod
    async def _get_init_params(cls, mdb_conn):
        # can be improved by initializing with a fixed amount of time
        # and go back on demand
        def layer(df):
            return df["values"].to_numpy()

        def virtual_indexes(df):
            return df["timestamp"].apply(cls._indexify).to_numpy().astype(np.uint32)

        last_update = datetime.utcnow().replace(tzinfo=timezone.utc)
        print("Fetching data...")
        items = await cls._fetch_data(mdb_conn)
        print("Data fetched")

        print("Preparing dataframe...")
        df = pd.DataFrame(items).rename(columns={"timestamp": "timestamp_m"})

        df["timestamp_h"] = df["timestamp_m"].apply(lambda t: t.replace(minute=0))
        df["timestamp_d"] = df["timestamp_h"].apply(lambda t: t.replace(hour=0))
        df["timestamp_mo"] = df["timestamp_d"].apply(lambda t: t.replace(day=1))
        df["timestamp_q"] = df["timestamp_mo"].apply(
            lambda t: t.replace(month={0: 1, 1: 4, 2: 7, 3: 10}[(t.month - 1) // 3]),
        )
        print("Dataframe prepared!")

        layer_level = LayerLevel.MINUTE
        print(f"Building for layer {layer_level}...")
        df_m = df[["timestamp_m", "values"]]
        df_m = df_m.rename(columns={"timestamp_m": "timestamp"})

        layers = {LayerLevel.MINUTE: layer(df_m)}
        v_indexes = {LayerLevel.MINUTE: virtual_indexes(df_m)}
        print(f"{layer_level} built")

        for layer_level, col in [
            (LayerLevel.HOUR, "timestamp_h"),
            (LayerLevel.DAY, "timestamp_d"),
            (LayerLevel.MONTH, "timestamp_mo"),
            (LayerLevel.QUARTER, "timestamp_q"),
        ]:
            print(f"Building for layer {layer_level}...")
            df_level = df.groupby(col).agg({"values": sum}).reset_index()
            df_level["values"] = df_level["values"].apply(lambda l: list(set(l)))
            df_level = df_level.rename(columns={col: "timestamp"})

            layers[layer_level] = layer(df_level)
            v_indexes[layer_level] = virtual_indexes(df_level)

            print(f"{layer_level} built")

        # the memory could be further squeezed by storing list of numbers on each level
        # and remap them in the end so that the strings are stored once only
        return layers, v_indexes, last_update

    @classmethod
    def _indexify(cls, date: datetime):
        first_timestamp = datetime.utcfromtimestamp(0).replace(tzinfo=timezone.utc)
        # How many minutes from `first_timestamp`?
        return int((date - first_timestamp).total_seconds()) // 60

    @classmethod
    async def _fetch_data(cls, mdb_conn):
        async with mdb_conn.connection() as mdb:
            rows = await mdb.fetch_all(cls.sql_query)

        return [cls._parse_row(r) for r in rows]

    @classmethod
    def _parse_row(cls, row):
        d = dict(row)
        return {
            "timestamp": d.pop("timestamp"),
            "values": d.pop("values"),
            # "extra": d,
        }


class AuthorIndexer(BaseIndexer):
    """Indexer for PRs' authors."""

    sql_query = """
SELECT
    timestamp, array_agg(distinct user_login) AS values
FROM (
    SELECT
        user_login,
        DATE_TRUNC('hour', created_at) + date_part('minute', created_at)::int / 1 * interval '1 min' AS timestamp
    FROM github_pull_requests_compat
) q
GROUP BY timestamp
ORDER BY timestamp;
    """  # noqa: E501


class ReviewerIndexer(BaseIndexer):
    """Indexer for PRs' reviewers."""

    sql_query = """
SELECT
    timestamp, array_agg(distinct user_login) AS values
FROM (
    SELECT
        user_login,
        DATE_TRUNC('hour', submitted_at) + date_part('minute', submitted_at)::int / 1 * interval '1 min' AS timestamp
    FROM github_pull_request_reviews_compat
) q
GROUP BY timestamp
ORDER BY timestamp;
    """  # noqa: E501


class CommitAuthorIndexer(BaseIndexer):
    """Indexer for PRs' commits authors."""

    sql_query = """
SELECT
    timestamp, array_agg(distinct author_login) AS values
FROM (
    SELECT
        author_login,
        DATE_TRUNC('hour', committed_date) + date_part('minute', committed_date)::int / 1 * interval '1 min' AS timestamp
    FROM github_push_commits_compat
) q
GROUP BY timestamp
ORDER BY timestamp;
    """  # noqa: E501


class CommitCommitterIndexer(BaseIndexer):
    """Indexer for PRs' commits committers."""

    sql_query = """
SELECT
    timestamp, array_agg(distinct committer_login) AS values
FROM (
    SELECT
        committer_login,
        DATE_TRUNC('hour', committed_date) + date_part('minute', committed_date)::int / 1 * interval '1 min' AS timestamp
    FROM github_push_commits_compat
) q
GROUP BY timestamp
ORDER BY timestamp;
    """  # noqa: E501


class CommenterIndexer(BaseIndexer):
    """Indexer for PRs' commenters."""

    sql_query = """
SELECT
    timestamp, array_agg(distinct user_login) AS values
FROM (
    SELECT
        user_login,
        DATE_TRUNC('hour', created_at) + date_part('minute', created_at)::int / 1 * interval '1 min' AS timestamp
    FROM github_pull_request_comments_compat
) q
GROUP BY timestamp
ORDER BY timestamp;
    """  # noqa: E501


class MergerIndexer(BaseIndexer):
    """Indexer for PRs' mergers."""

    sql_query = """
SELECT
    timestamp, array_agg(distinct merged_by_login) AS values
FROM (
    SELECT
        merged_by_login,
        DATE_TRUNC('hour', merged_at) + date_part('minute', merged_at)::int / 1 * interval '1 min' AS timestamp
    FROM github_pull_requests_compat
    WHERE merged = true
) q
GROUP BY timestamp
ORDER BY timestamp;
    """  # noqa: E501


class ReleaserIndexer(BaseIndexer):
    """Indexer for PRs' releasers."""

    sql_query = """
SELECT
    timestamp, array_agg(distinct author) AS values
FROM (
    SELECT
        author,
        DATE_TRUNC('hour', published_at) + date_part('minute', published_at)::int / 1 * interval '1 min' AS timestamp
    FROM github_releases_compat
) q
GROUP BY timestamp
ORDER BY timestamp;
    """  # noqa: E501


class ContributorsIndexer:
    """Indexer for PRs' contributors as a composite of other indexers."""

    indexers = [
        AuthorIndexer,
        ReviewerIndexer,
        CommitAuthorIndexer,
        CommitCommitterIndexer,
        CommenterIndexer,
        MergerIndexer,
        ReleaserIndexer,
    ]

    def __init__(self, indexers):
        """Initialize a contribs indexer with the provided sub-indexers."""
        self._indexers = indexers

    @classmethod
    async def create(cls, mdb_conn):
        """Create a contributors indexer."""
        tasks = [ind_cls.get_instance(mdb_conn) for ind_cls in cls.indexers]
        indexers = await gather(*tasks)
        return cls(indexers)

    def search(self, date_from, date_to, op, return_counts=False):
        """Search for contriburos in the provided interval and `op` logic."""
        if op.strip().upper() == "AND":
            func = np.intersect1d
        elif op.strip().upper() == "OR":
            func = np.union1d
        else:
            raise ValueError(f"Unsupported operator `{op}`")

        # handle return counts = true
        return reduce(
            func,
            [
                ind.search(date_from, date_to, return_counts=return_counts)
                for ind in self._indexers
            ],
        )
