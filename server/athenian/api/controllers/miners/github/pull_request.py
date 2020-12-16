import asyncio
from dataclasses import dataclass, fields as dataclass_fields
from datetime import date, datetime, timezone
from enum import Enum
from itertools import chain
import logging
import pickle
from typing import Collection, Dict, Generator, Iterable, Iterator, List, Optional, Sequence, \
    Set, Tuple

import aiomcache
import databases
import numpy as np
import pandas as pd
from pandas.core.common import flatten
from sqlalchemy import sql
from sqlalchemy.orm import aliased
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.sql.elements import BinaryExpression

from athenian.api import metadata
from athenian.api.async_utils import gather, read_sql_query
from athenian.api.cache import cached, CancelCache
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.precomputed_prs import \
    discover_inactive_merged_unreleased_prs, load_merged_unreleased_pull_request_facts, \
    load_open_pull_request_facts, update_unreleased_prs
from athenian.api.controllers.miners.github.release_match import map_prs_to_releases, \
    map_releases_to_prs
from athenian.api.controllers.miners.github.released_pr import matched_by_column
from athenian.api.controllers.miners.jira.issue import generate_jira_prs_query
from athenian.api.controllers.miners.types import MinedPullRequest, nonemax, nonemin, \
    PRParticipants, PRParticipationKind, PullRequestFacts
from athenian.api.controllers.settings import ReleaseMatch, ReleaseMatchSetting
from athenian.api.db import add_pdb_misses
from athenian.api.defer import AllEvents, defer
from athenian.api.models.metadata.github import Base, NodePullRequestJiraIssues, PullRequest, \
    PullRequestComment, PullRequestCommit, PullRequestLabel, PullRequestReview, \
    PullRequestReviewComment, PullRequestReviewRequest, Release
from athenian.api.models.metadata.jira import Component, Issue
from athenian.api.tracing import sentry_span
from athenian.api.typing_utils import DatabaseLike


@dataclass
class PRDataFrames:
    """Set of dataframes with all the PR data we can reach."""

    prs: pd.DataFrame
    commits: pd.DataFrame
    releases: pd.DataFrame
    jiras: pd.DataFrame
    reviews: pd.DataFrame
    review_comments: pd.DataFrame
    review_requests: pd.DataFrame
    comments: pd.DataFrame
    labels: pd.DataFrame

    def __iter__(self) -> Iterator[pd.DataFrame]:
        """Iterate the contained dataframes."""
        return iter(getattr(self, f.name) for f in dataclass_fields(self))


class PullRequestMiner:
    """Load all the information related to Pull Requests from the metadata DB. Iterate over it \
    to access individual PR objects."""

    CACHE_TTL = 5 * 60
    log = logging.getLogger("%s.PullRequestMiner" % metadata.__package__)

    def __init__(self, dfs: PRDataFrames):
        """Initialize a new instance of `PullRequestMiner`."""
        self._dfs = dfs

    @property
    def dfs(self) -> PRDataFrames:
        """Return the bound dataframes with PR information."""
        return self._dfs

    def drop(self, node_ids: Collection[str]) -> pd.Index:
        """
        Remove PRs from the given collection of PR node IDs in-place.

        Node IDs don't have to be all present.

        :return: Actually removed node IDs.
        """
        removed = self._dfs.prs.index.intersection(node_ids)
        if removed.empty:
            return removed
        self._dfs.prs.drop(removed, inplace=True)
        for df in self._dfs:
            df.drop(removed, inplace=True, errors="ignore",
                    level=0 if isinstance(df.index, pd.MultiIndex) else None)
        return removed

    def _deserialize_mine_cache(buffer: bytes) -> Tuple[PRDataFrames,
                                                        Dict[str, Tuple[str, PullRequestFacts]],
                                                        Set[str],
                                                        PRParticipants,
                                                        LabelFilter,
                                                        JIRAFilter,
                                                        Dict[str, ReleaseMatch],
                                                        asyncio.Event]:
        stuff = pickle.loads(buffer)
        event = asyncio.Event()
        event.set()
        return (*stuff, event)

    @sentry_span
    def _postprocess_cached_prs(
            result: Tuple[PRDataFrames,
                          Dict[str, Tuple[str, PullRequestFacts]],
                          Set[str],
                          PRParticipants,
                          LabelFilter,
                          JIRAFilter,
                          Dict[str, ReleaseMatch],
                          asyncio.Event],
            date_to: date,
            repositories: Set[str],
            participants: PRParticipants,
            labels: LabelFilter,
            jira: JIRAFilter,
            pr_blacklist: Optional[Tuple[Collection[str], Dict[str, List[str]]]],
            truncate: bool,
            **_) -> Tuple[PRDataFrames,
                          Dict[str, Tuple[str, PullRequestFacts]],
                          Set[str],
                          PRParticipants,
                          LabelFilter,
                          JIRAFilter,
                          Dict[str, ReleaseMatch],
                          asyncio.Event]:
        dfs, _, cached_repositories, cached_participants, cached_labels, cached_jira, _, _ = result
        cls = PullRequestMiner
        if (repositories - cached_repositories or
                not cls._check_participants_compatibility(cached_participants, participants) or
                not cached_labels.compatible_with(labels) or
                not cached_jira.compatible_with(jira)):
            raise CancelCache()
        to_remove = set()
        if pr_blacklist is not None:
            to_remove.update(pr_blacklist[0])
        to_remove.update(dfs.prs.index.take(np.where(
            np.in1d(dfs.prs[PullRequest.repository_full_name.key].values,
                    list(repositories), assume_unique=True, invert=True),
        )[0]))
        time_to = None if truncate else pd.Timestamp(date_to, tzinfo=timezone.utc)
        to_remove.update(cls._find_drop_by_participants(dfs, participants, time_to))
        to_remove.update(cls._find_drop_by_labels(dfs, labels))
        to_remove.update(cls._find_drop_by_jira(dfs, jira))
        cls._drop(dfs, to_remove)
        return result

    @classmethod
    @sentry_span
    @cached(
        exptime=lambda cls, **_: cls.CACHE_TTL,
        serialize=lambda r: pickle.dumps(r[:-1]),
        deserialize=_deserialize_mine_cache,
        key=lambda date_from, date_to, exclude_inactive, release_settings, updated_min, updated_max, pr_blacklist, truncate, **_: (  # noqa
            date_from.toordinal(), date_to.toordinal(), exclude_inactive, release_settings,
            updated_min.timestamp() if updated_min is not None else None,
            updated_max.timestamp() if updated_max is not None else None,
            ",".join(sorted(pr_blacklist[0]) if pr_blacklist is not None else []), truncate,
        ),
        postprocess=_postprocess_cached_prs,
    )
    async def _mine(cls,
                    date_from: date,
                    date_to: date,
                    repositories: Set[str],
                    participants: PRParticipants,
                    labels: LabelFilter,
                    jira: JIRAFilter,
                    branches: pd.DataFrame,
                    default_branches: Dict[str, str],
                    exclude_inactive: bool,
                    release_settings: Dict[str, ReleaseMatchSetting],
                    updated_min: Optional[datetime],
                    updated_max: Optional[datetime],
                    pr_blacklist: Optional[Tuple[Collection[str], Dict[str, List[str]]]],
                    truncate: bool,
                    meta_ids: Tuple[int, ...],
                    mdb: databases.Database,
                    pdb: databases.Database,
                    cache: Optional[aiomcache.Client],
                    ) -> Tuple[PRDataFrames,
                               Dict[str, Tuple[str, PullRequestFacts]],
                               Set[str],
                               PRParticipants,
                               LabelFilter,
                               JIRAFilter,
                               Dict[str, ReleaseMatch],
                               asyncio.Event]:
        assert isinstance(date_from, date) and not isinstance(date_from, datetime)
        assert isinstance(date_to, date) and not isinstance(date_to, datetime)
        assert isinstance(repositories, set)
        assert (updated_min is None) == (updated_max is None)
        time_from, time_to = (pd.Timestamp(t, tzinfo=timezone.utc) for t in (date_from, date_to))
        if pr_blacklist is not None:
            pr_blacklist, ambiguous = pr_blacklist
            if len(pr_blacklist) > 0:
                pr_blacklist = PullRequest.node_id.notin_any_values(pr_blacklist)
            else:
                pr_blacklist = None
        # the heaviest task should always go first
        tasks = [
            map_releases_to_prs(
                repositories, branches, default_branches, time_from, time_to,
                participants.get(PRParticipationKind.AUTHOR, []),
                participants.get(PRParticipationKind.MERGER, []),
                jira, release_settings, updated_min, updated_max,
                meta_ids, mdb, pdb, cache, pr_blacklist, truncate),
            cls.fetch_prs(
                time_from, time_to, repositories, participants, labels, jira,
                exclude_inactive, pr_blacklist, meta_ids, mdb, cache,
                updated_min=updated_min, updated_max=updated_max),
        ]
        # the following is a very rough approximation regarding updated_min/max:
        # we load all of none of the inactive merged PRs
        if not exclude_inactive and (updated_min is None or updated_min <= time_from):
            tasks.append(cls._fetch_inactive_merged_unreleased_prs(
                time_from, time_to, repositories, participants, labels, jira, default_branches,
                release_settings, meta_ids, mdb, pdb, cache))
        else:
            async def dummy_unreleased():
                return pd.DataFrame()
            tasks.append(dummy_unreleased())
        (released_prs, releases, matched_bys, dags), prs, unreleased = await gather(*tasks)
        concatenated = [prs, released_prs, unreleased]
        missed_prs = {}
        if pr_blacklist is not None:
            for repo, pr_node_ids in ambiguous.items():
                if matched_bys[repo] == ReleaseMatch.tag:
                    missed_prs[repo] = pr_node_ids
        if missed_prs:
            add_pdb_misses(pdb, "PullRequestMiner.mine/blacklist",
                           sum(len(v) for v in missed_prs.values()))
            # These PRs are released by branch and not by tag, and we require by tag.
            # Now fetch only them, respecting the filters.
            # TODO(vmarkovtsev): do not load the releases from scratch in map_releases_to_prs()
            inverse_pr_blacklist = PullRequest.node_id.in_(
                list(chain.from_iterable(missed_prs.values())))
            tasks = [
                map_releases_to_prs(
                    missed_prs, branches, default_branches, time_from, time_to,
                    participants.get(PRParticipationKind.AUTHOR, []),
                    participants.get(PRParticipationKind.MERGER, []),
                    jira, release_settings, updated_min, updated_max,
                    meta_ids, mdb, pdb, cache, inverse_pr_blacklist, truncate),
                cls.fetch_prs(
                    time_from, time_to, missed_prs, participants, labels, jira,
                    exclude_inactive, inverse_pr_blacklist, meta_ids, mdb, cache,
                    updated_min=updated_min, updated_max=updated_max),
            ]
            (missed_released_prs, _, _, _), missed_prs = await gather(*tasks)
            concatenated.extend([missed_released_prs, missed_prs])
        prs = pd.concat(concatenated, copy=False)
        prs = prs[~prs.index.duplicated()]
        prs.sort_index(level=0, inplace=True, sort_remaining=False)

        tasks = [
            # bypass the useless inner caching by calling _mine_by_ids directly
            cls._mine_by_ids(
                prs, unreleased.index, time_to, releases, matched_bys, branches,
                default_branches, dags, release_settings, meta_ids,
                mdb, pdb, cache, truncate=truncate),
            load_open_pull_request_facts(prs, pdb),
        ]
        (dfs, unreleased_facts, unreleased_prs_event), open_facts = await gather(
            *tasks, op="PullRequestMiner.mine/external_data")

        to_drop = cls._find_drop_by_participants(dfs, participants, None if truncate else time_to)
        to_drop |= cls._find_drop_by_labels(dfs, labels)
        if exclude_inactive:
            to_drop |= cls._find_drop_by_inactive(dfs, time_from, time_to)
        cls._drop(dfs, to_drop)

        facts = open_facts
        for k, v in unreleased_facts.items():  # merged unreleased PR precomputed facts
            if v is not None:  # it can be None because the pdb table is filled in two steps
                facts[k] = v
        # we don't care about the precomputed facts, they are here for the reference

        return dfs, facts, repositories, participants, labels, jira, matched_bys, \
            unreleased_prs_event

    _deserialize_mine_cache = staticmethod(_deserialize_mine_cache)
    _postprocess_cached_prs = staticmethod(_postprocess_cached_prs)

    def _deserialize_mine_by_ids_cache(
            buffer: bytes) -> Tuple[PRDataFrames,
                                    Dict[str, Tuple[str, PullRequestFacts]],
                                    asyncio.Event]:
        dfs, facts = pickle.loads(buffer)
        event = asyncio.Event()
        event.set()
        return dfs, facts, event

    @classmethod
    @cached(
        exptime=lambda cls, **_: cls.CACHE_TTL,
        serialize=lambda r: pickle.dumps(r[:-1]),
        deserialize=_deserialize_mine_by_ids_cache,
        key=lambda prs, unreleased, releases, time_to, truncate=True, **_: (
            ",".join(prs.index), ",".join(unreleased),
            ",".join(releases[Release.id.key].values), time_to.timestamp(),
            truncate,
        ),
    )
    async def mine_by_ids(cls,
                          prs: pd.DataFrame,
                          unreleased: Collection[str],
                          time_to: datetime,
                          releases: pd.DataFrame,
                          matched_bys: Dict[str, ReleaseMatch],
                          branches: pd.DataFrame,
                          default_branches: Dict[str, str],
                          dags: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                          release_settings: Dict[str, ReleaseMatchSetting],
                          meta_ids: Tuple[int, ...],
                          mdb: databases.Database,
                          pdb: databases.Database,
                          cache: Optional[aiomcache.Client],
                          truncate: bool = True,
                          ) -> Tuple[PRDataFrames,
                                     Dict[str, Tuple[str, PullRequestFacts]],
                                     asyncio.Event]:
        """
        Fetch PR metadata for certain PRs.

        :param prs: pandas DataFrame with fetched PullRequest-s. Only the details about those PRs \
                    will be loaded from the DB.
        :param truncate: Do not load anything after `time_to`.
        :return: 1. List of mined DataFrame-s. \
                 2. mapping to pickle-d PullRequestFacts for unreleased merged PR. \
                 3. Synchronization for updating the pdb table with merged unreleased PRs.
        """
        return await cls._mine_by_ids(
            prs, unreleased, time_to, releases, matched_bys, branches, default_branches,
            dags, release_settings, meta_ids, mdb, pdb, cache, truncate=truncate)

    _deserialize_mine_by_ids_cache = staticmethod(_deserialize_mine_by_ids_cache)

    @classmethod
    @sentry_span
    async def _mine_by_ids(cls,
                           prs: pd.DataFrame,
                           unreleased: Collection[str],
                           time_to: datetime,
                           releases: pd.DataFrame,
                           matched_bys: Dict[str, ReleaseMatch],
                           branches: pd.DataFrame,
                           default_branches: Dict[str, str],
                           dags: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                           release_settings: Dict[str, ReleaseMatchSetting],
                           meta_ids: Tuple[int, ...],
                           mdb: databases.Database,
                           pdb: databases.Database,
                           cache: Optional[aiomcache.Client],
                           truncate: bool = True,
                           ) -> Tuple[PRDataFrames,
                                      Dict[str, Tuple[str, PullRequestFacts]],
                                      asyncio.Event]:
        node_ids = prs.index if len(prs) > 0 else set()
        facts = {}  # precomputed PullRequestFacts about merged unreleased PRs
        unreleased_prs_event: asyncio.Event = None
        merged_unreleased_indexes = []

        @sentry_span
        async def fetch_reviews():
            return await cls._read_filtered_models(
                PullRequestReview, node_ids, time_to, meta_ids, mdb,
                columns=[PullRequestReview.submitted_at, PullRequestReview.state,
                         PullRequestReview.user_login],
                created_at=truncate)

        @sentry_span
        async def fetch_review_comments():
            return await cls._read_filtered_models(
                PullRequestReviewComment, node_ids, time_to, meta_ids, mdb,
                columns=[PullRequestReviewComment.created_at, PullRequestReviewComment.user_login],
                created_at=truncate)

        @sentry_span
        async def fetch_review_requests():
            return await cls._read_filtered_models(
                PullRequestReviewRequest, node_ids, time_to, meta_ids, mdb,
                columns=[PullRequestReviewRequest.created_at],
                created_at=truncate)

        @sentry_span
        async def fetch_comments():
            return await cls._read_filtered_models(
                PullRequestComment, node_ids, time_to, meta_ids, mdb,
                columns=[PullRequestComment.created_at, PullRequestComment.user_login],
                created_at=truncate)

        @sentry_span
        async def fetch_commits():
            return await cls._read_filtered_models(
                PullRequestCommit, node_ids, time_to, meta_ids, mdb,
                columns=[PullRequestCommit.authored_date, PullRequestCommit.committed_date,
                         PullRequestCommit.author_login, PullRequestCommit.committer_login],
                created_at=truncate)

        @sentry_span
        async def map_releases():
            if truncate:
                merged_mask = (prs[PullRequest.merged_at.key] < time_to).values
                nonlocal merged_unreleased_indexes
                merged_unreleased_indexes = np.where(prs[PullRequest.merged_at.key] >= time_to)[0]
            else:
                merged_mask = prs[PullRequest.merged_at.key].notnull()
            merged_mask &= ~prs.index.isin(unreleased)
            merged_prs = prs.take(np.where(merged_mask)[0])
            subtasks = [map_prs_to_releases(
                merged_prs, releases, matched_bys, branches, default_branches, time_to,
                dags, release_settings, meta_ids, mdb, pdb, cache),
                load_merged_unreleased_pull_request_facts(
                    prs.take(np.where(~merged_mask)[0]),
                    nonemax(releases[Release.published_at.key].nonemax(), time_to),
                    LabelFilter.empty(), matched_bys, default_branches, release_settings, pdb)]
            df_facts, other_facts = await gather(*subtasks)
            nonlocal facts
            nonlocal unreleased_prs_event
            df, facts, unreleased_prs_event = df_facts
            facts.update(other_facts)
            return df

        @sentry_span
        async def fetch_labels():
            return await cls._read_filtered_models(
                PullRequestLabel, node_ids, time_to, meta_ids, mdb,
                columns=[sql.func.lower(PullRequestLabel.name).label(PullRequestLabel.name.key),
                         PullRequestLabel.description,
                         PullRequestLabel.color],
                created_at=False)

        @sentry_span
        async def fetch_jira():
            _map = aliased(NodePullRequestJiraIssues, name="m")
            _issue = aliased(Issue, name="i")
            _issue_epic = aliased(Issue, name="e")
            selected = [
                PullRequest.node_id, _issue.key, _issue.title, _issue.type, _issue.status,
                _issue.created, _issue.updated, _issue.resolved, _issue.labels, _issue.components,
                _issue.acc_id, _issue_epic.key.label("epic"),
            ]
            df = await read_sql_query(
                sql.select(selected).select_from(sql.join(
                    PullRequest, sql.join(
                        _map, sql.join(_issue, _issue_epic, sql.and_(
                            _issue.epic_id == _issue_epic.id,
                            _issue.acc_id == _issue_epic.acc_id), isouter=True),
                        sql.and_(_map.jira_id == _issue.id,
                                 _map.jira_acc == _issue.acc_id)),
                    sql.and_(PullRequest.node_id == _map.node_id,
                             PullRequest.acc_id == _map.node_acc),
                )).where(sql.and_(PullRequest.node_id.in_(node_ids),
                                  PullRequest.acc_id.in_(meta_ids))),
                mdb, columns=selected, index=[PullRequest.node_id.key, _issue.key.key])
            if df.empty:
                df.drop([Issue.acc_id.key, Issue.components.key], inplace=True, axis=1)
                return df
            components = df[[Issue.acc_id.key, Issue.components.key]] \
                .groupby(Issue.acc_id.key, sort=False).aggregate(lambda s: set(flatten(s)))
            rows = await mdb.fetch_all(
                sql.select([Component.acc_id, Component.id, Component.name])
                .where(sql.or_(*(sql.and_(Component.id.in_(vals),
                                          Component.acc_id == int(acc))
                                 for acc, vals in zip(components.index.values,
                                                      components[Issue.components.key].values)))))
            cmap = {}
            for r in rows:
                cmap.setdefault(r[0], {})[r[1]] = r[2].lower()
            df[Issue.labels.key] = (
                df[Issue.labels.key].apply(lambda i: [s.lower() for s in (i or [])])
                +
                df[[Issue.acc_id.key, Issue.components.key]]
                .apply(lambda row: ([cmap[row[Issue.acc_id.key]][c]
                                    for c in row[Issue.components.key]]
                                    if row[Issue.components.key] is not None else []),
                       axis=1)
            )
            df.drop([Issue.acc_id.key, Issue.components.key], inplace=True, axis=1)
            return df

        # the order is important: it provides the best performance
        # we launch coroutines from the heaviest to the lightest
        dfs = await gather(
            fetch_commits(),
            map_releases(),
            fetch_jira(),
            fetch_reviews(),
            fetch_review_comments(),
            fetch_review_requests(),
            fetch_comments(),
            fetch_labels())
        dfs = PRDataFrames(prs, *dfs)
        if len(merged_unreleased_indexes):
            # if we truncate and there are PRs merged after `time_to`
            merged_unreleased_prs = prs.take(merged_unreleased_indexes)
            label_matches = np.nonzero(np.in1d(
                dfs.labels.index.get_level_values(0).values.astype("U"),
                merged_unreleased_prs.index.values.astype("U")))[0]
            labels = {}
            for k, v in zip(dfs.labels.index.values[label_matches],
                            dfs.labels[PullRequestLabel.name.key].take(label_matches).values):
                try:
                    labels[k].append(v)
                except KeyError:
                    labels[k] = [v]
            other_unreleased_prs_event = asyncio.Event()
            combined_unreleased_prs_event = AllEvents(
                unreleased_prs_event, other_unreleased_prs_event)
            unreleased_prs_event = combined_unreleased_prs_event
            await defer(update_unreleased_prs(
                merged_unreleased_prs, pd.DataFrame(), time_to, labels, matched_bys,
                default_branches, release_settings, pdb, other_unreleased_prs_event),
                "update_unreleased_prs/truncate(%d)" % len(merged_unreleased_indexes))
        return dfs, facts, unreleased_prs_event

    @classmethod
    @sentry_span
    async def mine(cls,
                   date_from: date,
                   date_to: date,
                   time_from: datetime,
                   time_to: datetime,
                   repositories: Set[str],
                   participants: PRParticipants,
                   labels: LabelFilter,
                   jira: JIRAFilter,
                   branches: pd.DataFrame,
                   default_branches: Dict[str, str],
                   exclude_inactive: bool,
                   release_settings: Dict[str, ReleaseMatchSetting],
                   meta_ids: Tuple[int, ...],
                   mdb: databases.Database,
                   pdb: databases.Database,
                   cache: Optional[aiomcache.Client],
                   updated_min: Optional[datetime] = None,
                   updated_max: Optional[datetime] = None,
                   pr_blacklist: Optional[Tuple[Collection[str], Dict[str, List[str]]]] = None,
                   truncate: bool = True,
                   ) -> Tuple["PullRequestMiner",
                              Dict[str, Tuple[str, PullRequestFacts]],
                              Dict[str, ReleaseMatch],
                              asyncio.Event]:
        """
        Mine metadata about pull requests according to the numerous filters.

        :param meta_ids: Metadata (GitHub) account IDs.
        :param date_from: Fetch PRs created starting from this date, inclusive.
        :param date_to: Fetch PRs created ending with this date, inclusive.
        :param time_from: Precise timestamp of since when PR events are allowed to happen.
        :param time_to: Precise timestamp of until when PR events are allowed to happen.
        :param repositories: PRs must belong to these repositories (prefix excluded).
        :param participants: PRs must have these user IDs in the specified participation roles \
                             (OR aggregation). An empty dict means everybody.
        :param labels: PRs must be labeled according to this filter's include & exclude sets.
        :param jira: JIRA filters for those PRs that are matched with JIRA issues.
        :param branches: Preloaded DataFrame with branches in the specified repositories.
        :param default_branches: Mapping from repository names to their default branch names.
        :param exclude_inactive: Ors must have at least one event in the given time frame.
        :param release_settings: Release match settings of the account.
        :param updated_min: PRs must have the last update timestamp not older than it.
        :param updated_max: PRs must have the last update timestamp not newer than or equal to it.
        :param mdb: Metadata db instance.
        :param pdb: Precomputed db instance.
        :param cache: memcached client to cache the collected data.
        :param pr_blacklist: completely ignore the existence of these PR node IDs. \
                             The second tuple element is the ambiguous PRs: released by branch \
                             while there were no tag releases and the strategy is `tag_or_branch`.
        :param truncate: activate the "time machine" and erase everything after `time_to`.
        :return: 1. New `PullRequestMiner` with the PRs satisfying to the specified filters. \
                 2. Precomputed facts about unreleased pull requests. \
                    This is an optimization which breaks the abstraction a bit. \
                 3. `matched_bys` - release matches for each repository. \
                 4. Synchronization for updating the pdb table with merged unreleased PRs. \
                    Another abstraction leakage that we have to deal with.
        """
        date_from_with_time = datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc)
        date_to_with_time = datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc)
        assert time_from >= date_from_with_time
        assert time_to <= date_to_with_time
        dfs, facts, _, _, _, _, matched_bys, event = await cls._mine(
            date_from, date_to, repositories, participants, labels, jira, branches,
            default_branches, exclude_inactive, release_settings, updated_min, updated_max,
            pr_blacklist, truncate, meta_ids, mdb, pdb, cache)
        cls._truncate_prs(dfs, time_from, time_to)
        return cls(dfs), facts, matched_bys, event

    @classmethod
    @sentry_span
    async def fetch_prs(cls,
                        time_from: datetime,
                        time_to: datetime,
                        repositories: Set[str],
                        participants: PRParticipants,
                        labels: LabelFilter,
                        jira: JIRAFilter,
                        exclude_inactive: bool,
                        pr_blacklist: Optional[BinaryExpression],
                        meta_ids: Tuple[int, ...],
                        mdb: databases.Database,
                        cache: Optional[aiomcache.Client],
                        columns=PullRequest,
                        updated_min: Optional[datetime] = None,
                        updated_max: Optional[datetime] = None,
                        ) -> pd.DataFrame:
        """
        Query pull requests from mdb that satisfy the given filters.

        Note: we cannot filter by regular PR labels here due to the DB schema limitations,
        so the caller is responsible for fetching PR labels and filtering by them afterward.
        Besides, we cannot filter by participation roles different from AUTHOR and MERGER.
        """
        assert (updated_min is None) == (updated_max is None)
        filters = [
            sql.case(
                [(PullRequest.closed, PullRequest.closed_at)],
                else_=sql.text("'3000-01-01'"),
            ) >= time_from,
            PullRequest.created_at < time_to,
            PullRequest.acc_id.in_(meta_ids),
            PullRequest.hidden.is_(False),
            PullRequest.repository_full_name.in_(repositories),
        ]
        if exclude_inactive and updated_min is None:
            # this does not provide 100% guarantee because it can be after time_to,
            # we need to properly filter later
            filters.append(PullRequest.updated_at >= time_from)
        if updated_min is not None:
            filters.append(PullRequest.updated_at.between(updated_min, updated_max))
        if pr_blacklist is not None:
            filters.append(pr_blacklist)
        if len(participants) == 1:
            if PRParticipationKind.AUTHOR in participants:
                filters.append(PullRequest.user_login.in_(
                    participants[PRParticipationKind.AUTHOR]))
            elif PRParticipationKind.MERGER in participants:
                filters.append(
                    PullRequest.merged_by_login.in_(participants[PRParticipationKind.MERGER]))
        elif len(participants) == 2 and PRParticipationKind.AUTHOR in participants and \
                PRParticipationKind.MERGER in participants:
            filters.append(sql.or_(
                PullRequest.user_login.in_(participants[PRParticipationKind.AUTHOR]),
                PullRequest.merged_by_login.in_(participants[PRParticipationKind.MERGER]),
            ))
        if columns is PullRequest:
            selected_columns = [PullRequest]
            remove_acc_id = False
        else:
            selected_columns = columns = list(columns)
            if remove_acc_id := (PullRequest.acc_id not in selected_columns):
                selected_columns.append(PullRequest.acc_id)
        if not jira:
            query = sql.select(selected_columns).where(sql.and_(*filters))
        else:
            query = await generate_jira_prs_query(
                filters, jira, mdb, cache, columns=selected_columns)
        if (embedded_labels_query := labels and labels.include and not labels.exclude and
                mdb.url.dialect in ("postgres", "postgresql")):
            singles, multiples = LabelFilter.split(labels.include)
            if multiples:
                in_items = set(singles + list(chain.from_iterable(multiples)))
                # we cannot be sure about multiples, but we do know that any PRs without any
                # mentioned label are not suitable
                embedded_labels_query = False
            else:
                in_items = singles
            pr_node_id = sql.column("m.node_id", is_literal=True)
            pr_acc_id = sql.column("m.acc_id", is_literal=True)
            query = sql.select([sql.column("m.*", is_literal=True)])\
                .distinct(pr_node_id) \
                .select_from(sql.join(
                    query.alias("m"), PullRequestLabel,
                    sql.and_(pr_node_id == PullRequestLabel.pull_request_node_id,
                             pr_acc_id == PullRequestLabel.acc_id))) \
                .where(sql.func.lower(PullRequestLabel.name).in_(in_items))
        prs = await read_sql_query(query, mdb, columns, index=PullRequest.node_id.key)
        if remove_acc_id:
            del prs[PullRequest.acc_id.key]
        if PullRequest.closed.key in prs:
            cls.adjust_pr_closed_merged_timestamps(prs)
        if not labels or embedded_labels_query:
            return prs
        lcols = [
            PullRequestLabel.pull_request_node_id,
            sql.func.lower(PullRequestLabel.name).label(PullRequestLabel.name.key),
            PullRequestLabel.description,
            PullRequestLabel.color,
        ]
        df_labels = await read_sql_query(
            sql.select(lcols)
            .where(sql.and_(PullRequestLabel.pull_request_node_id.in_(prs.index),
                            PullRequestLabel.acc_id.in_(meta_ids))),
            mdb, lcols, index=PullRequestLabel.pull_request_node_id.key)
        left = cls._find_left_by_labels(
            df_labels.index, df_labels[PullRequestLabel.name.key].values, labels)
        prs = prs.take(np.where(prs.index.isin(left))[0])
        return prs

    @staticmethod
    def adjust_pr_closed_merged_timestamps(prs_df: pd.DataFrame) -> None:
        """Force set `closed_at` and `merged_at` to NULL if not `closed`. Remove `closed`."""
        not_closed = ~prs_df[PullRequest.closed.key].values
        prs_df.loc[not_closed, PullRequest.closed_at.key] = pd.NaT
        prs_df.loc[not_closed, PullRequest.merged_at.key] = pd.NaT
        prs_df.drop(columns=PullRequest.closed.key, inplace=True)

    @classmethod
    @sentry_span
    async def _fetch_inactive_merged_unreleased_prs(
            cls,
            time_from: datetime,
            time_to: datetime,
            repos: Collection[str],
            participants: PRParticipants,
            labels: LabelFilter,
            jira: JIRAFilter,
            default_branches: Dict[str, str],
            release_settings: Dict[str, ReleaseMatchSetting],
            meta_ids: Tuple[int, ...],
            mdb: databases.Database,
            pdb: databases.Database,
            cache: Optional[aiomcache.Client]) -> pd.DataFrame:
        node_ids, _ = await discover_inactive_merged_unreleased_prs(
            time_from, time_to, repos, participants, labels, default_branches, release_settings,
            pdb, cache)
        if not jira:
            return await read_sql_query(sql.select([PullRequest])
                                        .where(PullRequest.node_id.in_(node_ids)),
                                        mdb, PullRequest, index=PullRequest.node_id.key)
        return await cls.filter_jira(node_ids, jira, meta_ids, mdb, cache)

    @classmethod
    @sentry_span
    async def filter_jira(cls,
                          pr_node_ids: Iterable[str],
                          jira: JIRAFilter,
                          meta_ids: Tuple[int, ...],
                          mdb: databases.Database,
                          cache: Optional[aiomcache.Client],
                          columns=PullRequest) -> pd.DataFrame:
        """Filter PRs by JIRA properties."""
        assert jira
        filters = [PullRequest.node_id.in_(pr_node_ids)]
        query = await generate_jira_prs_query(filters, jira, mdb, cache, columns=columns)
        return await read_sql_query(query, mdb, columns, index=PullRequest.node_id.key)

    @staticmethod
    def _check_participants_compatibility(cached_participants: PRParticipants,
                                          participants: PRParticipants) -> bool:
        if not cached_participants:
            return True
        if not participants:
            return False
        for k, v in participants.items():
            if v - cached_participants.get(k, set()):
                return False
        return True

    @classmethod
    @sentry_span
    def _remove_spurious_prs(cls, time_from: datetime, dfs: PRDataFrames) -> None:
        old_releases = np.where(dfs.releases[Release.published_at.key] < time_from)[0]
        if len(old_releases) == 0:
            return
        cls._drop(dfs, dfs.releases.index[old_releases])

    @classmethod
    def _drop(cls, dfs: PRDataFrames, pr_ids: Collection[str]) -> None:
        if len(pr_ids) == 0:
            return
        for df in dfs:
            df.drop(pr_ids,
                    level=0 if isinstance(df.index, pd.MultiIndex) else None,
                    inplace=True,
                    errors="ignore")

    @classmethod
    @sentry_span
    def _find_drop_by_participants(cls,
                                   dfs: PRDataFrames,
                                   participants: PRParticipants,
                                   time_to: Optional[datetime],
                                   ) -> pd.Index:
        if not participants:
            return pd.Index([])
        if time_to is not None:
            for df_name, col in (("commits", PullRequestCommit.committed_date),
                                 ("reviews", PullRequestReview.created_at),
                                 ("review_comments", PullRequestReviewComment.created_at),
                                 ("review_requests", PullRequestReviewRequest.created_at),
                                 ("comments", PullRequestComment.created_at)):
                df = getattr(dfs, df_name)
                setattr(dfs, df_name, df.take(np.where(df[col.key] < time_to)[0]))
        passed = []
        dict_iter = (
            (dfs.prs, PullRequest.user_login, None, PRParticipationKind.AUTHOR),
            (dfs.prs, PullRequest.merged_by_login, PullRequest.merged_at, PRParticipationKind.MERGER),  # noqa
            (dfs.releases, Release.author, Release.published_at, PRParticipationKind.RELEASER),
        )
        for df, part_col, date_col, pk in dict_iter:
            col_parts = participants.get(pk)
            if not col_parts:
                continue
            mask = df[part_col.key].isin(col_parts)
            if time_to is not None and date_col is not None:
                mask &= df[date_col.key] < time_to
            passed.append(df.index.take(np.where(mask)[0]))
        reviewers = participants.get(PRParticipationKind.REVIEWER)
        if reviewers:
            ulkr = PullRequestReview.user_login.key
            ulkp = PullRequest.user_login.key
            user_logins = pd.merge(dfs.reviews[ulkr].droplevel(1), dfs.prs[ulkp],
                                   left_index=True, right_index=True, how="left", copy=False)
            ulkr += "_x"
            ulkp += "_y"
            passed.append(user_logins.index.take(np.where(
                (user_logins[ulkr] != user_logins[ulkp]) & user_logins[ulkr].isin(reviewers),
            )[0]).unique())
        for df, col, pk in (
                (dfs.comments, PullRequestComment.user_login, PRParticipationKind.COMMENTER),
                (dfs.commits, PullRequestCommit.author_login, PRParticipationKind.COMMIT_AUTHOR),
                (dfs.commits, PullRequestCommit.committer_login, PRParticipationKind.COMMIT_COMMITTER)):  # noqa
            col_parts = participants.get(pk)
            if not col_parts:
                continue
            passed.append(df.index.get_level_values(0).take(np.where(
                df[col.key].isin(col_parts))[0]).unique())
        while len(passed) > 1:
            new_passed = []
            for i in range(0, len(passed), 2):
                if i + 1 < len(passed):
                    new_passed.append(passed[i].union(passed[i + 1]))
                else:
                    new_passed.append(passed[i])
            passed = new_passed
        return dfs.prs.index.difference(passed[0])

    @classmethod
    @sentry_span
    def _find_drop_by_labels(cls, dfs: PRDataFrames, labels: LabelFilter) -> pd.Index:
        if not labels:
            return pd.Index([])
        df_labels_index = dfs.labels.index.get_level_values(0)
        df_labels_names = dfs.labels[PullRequestLabel.name.key].values
        left = cls._find_left_by_labels(df_labels_index, df_labels_names, labels)
        return dfs.prs.index.difference(left)

    @classmethod
    def _find_left_by_labels(cls,
                             df_labels_index: pd.Index,
                             df_labels_names: Sequence[str],
                             labels: LabelFilter) -> pd.Index:
        left_include = left_exclude = None
        if labels.include:
            singles, multiples = LabelFilter.split(labels.include)
            left_include = df_labels_index.take(
                np.where(np.in1d(df_labels_names, singles))[0],
            ).unique()
            for group in multiples:
                passed = df_labels_index
                for label in group:
                    passed = passed.intersection(
                        df_labels_index.take(np.where(df_labels_names == label)))
                    if passed.empty:
                        break
                left_include = left_include.union(passed)
        if labels.exclude:
            left_exclude = df_labels_index.difference(df_labels_index.take(
                np.where(np.in1d(df_labels_names, list(labels.exclude)))[0],
            ).unique())
        if labels.include:
            if labels.exclude:
                left = left_include.intersection(left_exclude)
            else:
                left = left_include
        else:
            left = left_exclude
        return left

    @classmethod
    @sentry_span
    def _find_drop_by_jira(cls, dfs: PRDataFrames, jira: JIRAFilter) -> pd.Index:
        if not jira:
            return pd.Index([])
        left = []
        jira_index = dfs.jiras.index.get_level_values(0)
        if jira.labels:
            df_labels_names = dfs.jiras[Issue.labels.key].values
            df_labels_index = pd.Index(np.repeat(jira_index, [len(v) for v in df_labels_names]))
            df_labels_names = list(pd.core.common.flatten(df_labels_names))
            left.append(cls._find_left_by_labels(df_labels_index, df_labels_names, jira.labels))
        if jira.epics:
            left.append(jira_index.take(np.where(
                dfs.jiras["epic"].isin(jira.epics))[0]).unique())
        if jira.issue_types:
            left.append(dfs.jiras.index.get_level_values(0).take(np.where(
                dfs.jiras[Issue.type.key].str.lower().isin(jira.issue_types))[0]).unique())
        result = left[0]
        for other in left[1:]:
            result = result.intersection(other)
        return dfs.prs.index.difference(result)

    @classmethod
    @sentry_span
    def _find_drop_by_inactive(cls,
                               dfs: PRDataFrames,
                               time_from: datetime,
                               time_to: datetime) -> pd.Index:
        activities = [
            dfs.prs[PullRequest.created_at.key],
            dfs.prs[PullRequest.closed_at.key],
            dfs.commits[PullRequestCommit.committed_date.key],
            dfs.review_requests[PullRequestReviewRequest.created_at.key],
            dfs.reviews[PullRequestReview.created_at.key],
            dfs.comments[PullRequestComment.created_at.key],
            dfs.releases[Release.published_at.key],
        ]
        for df in activities:
            if df.index.nlevels > 1:
                df.index = df.index.droplevel(1)
            df.name = "timestamp"
        activities = pd.concat(activities, copy=False)
        active_prs = activities.index.take(np.where(
            activities.between(time_from, time_to))[0]).drop_duplicates()
        inactive_prs = dfs.prs.index.difference(active_prs)
        return inactive_prs

    @staticmethod
    async def _read_filtered_models(model_cls: Base,
                                    node_ids: Collection[str],
                                    time_to: datetime,
                                    meta_ids: Tuple[int, ...],
                                    mdb: DatabaseLike,
                                    columns: Optional[List[InstrumentedAttribute]] = None,
                                    created_at=True,
                                    ) -> pd.DataFrame:
        if columns is not None:
            columns = [model_cls.pull_request_node_id, model_cls.node_id] + columns
        filters = [model_cls.pull_request_node_id.in_(node_ids),
                   model_cls.acc_id.in_(meta_ids)]
        if created_at:
            filters.append(model_cls.created_at < time_to)
        df = await read_sql_query(
            sql.select(columns or [model_cls]).where(sql.and_(*filters)),
            con=mdb,
            columns=columns or model_cls,
            index=[model_cls.pull_request_node_id.key, model_cls.node_id.key])
        return df

    @classmethod
    @sentry_span
    def _truncate_prs(cls,
                      dfs: PRDataFrames,
                      time_from: datetime,
                      time_to: datetime,
                      ) -> None:
        """
        Remove PRs outside of the given time range.

        This is used to correctly handle timezone offsets.
        """
        # filter out PRs which were released before `time_from`
        unreleased = dfs.releases.index.take(np.where(
            dfs.releases[Release.published_at.key] < time_from)[0])
        # closed but not merged in `[date_from, time_from]`
        unrejected = dfs.prs.index.take(np.where(
            (dfs.prs[PullRequest.closed_at.key] < time_from) &
            dfs.prs[PullRequest.merged_at.key].isnull())[0])
        # created in `[time_to, date_to]`
        uncreated = dfs.prs.index.take(np.where(
            dfs.prs[PullRequest.created_at.key] >= time_to)[0])
        to_remove = unreleased.union(unrejected.union(uncreated))
        cls._drop(dfs, to_remove)

    def __len__(self) -> int:
        """Return the number of loaded pull requests."""
        return len(self._dfs.prs)

    def __iter__(self) -> Generator[MinedPullRequest, None, None]:
        """Iterate over the individual pull requests."""
        df_fields = [f.name for f in dataclass_fields(MinedPullRequest) if f.name != "pr"]
        dfs = []
        grouped_df_iters = []
        index_backup = []
        for k in df_fields:
            plural = k.endswith("s")
            df = getattr(self._dfs, k if plural else (k + "s"))
            dfs.append(df)
            # our very own groupby() allows us to call take() with reduced overhead
            node_ids = df.index.get_level_values(0).values
            if df.index.nlevels > 1:
                # this is not really required but it makes iteration deterministic
                order_keys = (node_ids + df.index.get_level_values(1).values).astype("U")
                node_ids = node_ids.astype("U")
            else:
                order_keys = node_ids = node_ids.astype("U")
            node_ids_order = np.argsort(order_keys)
            node_ids = node_ids[node_ids_order]
            node_ids_backtrack = np.arange(0, len(df))[node_ids_order]
            node_ids_unique_counts = np.unique(node_ids, return_counts=True)[1]
            node_ids_group_counts = np.zeros(len(node_ids_unique_counts) + 1, dtype=int)
            np.cumsum(node_ids_unique_counts, out=node_ids_group_counts[1:])
            keys = node_ids[node_ids_group_counts[:-1]]
            groups = np.split(node_ids_backtrack, node_ids_group_counts[1:-1])
            grouped_df_iters.append(iter(zip(keys, groups)))
            if plural:
                index_backup.append(df.index)
                df.index = df.index.droplevel(0)
            else:
                index_backup.append(None)
        try:
            grouped_df_states = []
            for i in grouped_df_iters:
                try:
                    grouped_df_states.append(next(i))
                except StopIteration:
                    grouped_df_states.append((None, None))
            empty_df_cache = {}
            pr_columns = [PullRequest.node_id.key]
            pr_columns.extend(self._dfs.prs.columns)
            if not self._dfs.prs.index.is_monotonic_increasing:
                raise IndexError("PRs index must be pre-sorted ascending: "
                                 "prs.sort_index(inplace=True)")
            for pr_tuple in self._dfs.prs.itertuples():
                pr_node_id = pr_tuple.Index
                items = {"pr": dict(zip(pr_columns, pr_tuple))}
                for i, (k, (state_pr_node_id, gdf), git, df) in enumerate(zip(
                        df_fields, grouped_df_states, grouped_df_iters, dfs)):
                    if state_pr_node_id == pr_node_id:
                        if not k.endswith("s"):
                            # much faster than df.iloc[gdf[0]]
                            gdf = {c: v for c, v in zip(df.columns, df._data.fast_xs(gdf[0]))}
                        else:
                            gdf = df.take(gdf)
                        items[k] = gdf
                        try:
                            grouped_df_states[i] = next(git)
                        except StopIteration:
                            grouped_df_states[i] = None, None
                    else:
                        if k.endswith("s"):
                            try:
                                items[k] = empty_df_cache[k]
                            except KeyError:
                                items[k] = empty_df_cache[k] = df.iloc[:0].copy()
                        else:
                            items[k] = {c: None for c in df.columns}
                yield MinedPullRequest(**items)
        finally:
            for df, index in zip(dfs, index_backup):
                if index is not None:
                    df.index = index


class ReviewResolution(Enum):
    """Possible review "state"-s in the metadata DB."""

    APPROVED = "APPROVED"
    CHANGES_REQUESTED = "CHANGES_REQUESTED"
    COMMENTED = "COMMENTED"


class ImpossiblePullRequest(Exception):
    """Raised by PullRequestFactsMiner._compile() on broken PRs."""


class PullRequestFactsMiner:
    """Extract the pull request event timestamps from MinedPullRequest-s."""

    log = logging.getLogger("%s.PullRequestFactsMiner" % metadata.__package__)
    dummy_reviews = pd.Series(["INVALID", pd.NaT],
                              index=[PullRequestReview.state.key,
                                     PullRequestReview.submitted_at.key])

    def __init__(self, bots: Set[str]):
        """Require the set of bots to be preloaded."""
        self._bots = np.sort(list(bots))

    def __call__(self, pr: MinedPullRequest) -> PullRequestFacts:
        """
        Extract the pull request event timestamps from a MinedPullRequest.

        May raise ImpossiblePullRequest if the PR has an "impossible" state like
        created after closed.
        """
        created = pr.pr[PullRequest.created_at.key]
        if created != created:
            raise ImpossiblePullRequest()
        merged = pr.pr[PullRequest.merged_at.key]
        if merged != merged:
            merged = None
        closed = pr.pr[PullRequest.closed_at.key]
        if closed != closed:
            closed = None
        if merged and not closed:
            self.log.error("[DEV-508] PR %s (%s#%d) is merged at %s but not closed",
                           pr.pr[PullRequest.node_id.key],
                           pr.pr[PullRequest.repository_full_name.key],
                           pr.pr[PullRequest.number.key],
                           merged)
            closed = merged
        # we don't need these indexes
        pr.comments.reset_index(inplace=True, drop=True)
        pr.reviews.reset_index(inplace=True, drop=True)
        first_commit = pr.commits[PullRequestCommit.authored_date.key].nonemin()
        # yes, first_commit uses authored_date while last_commit uses committed_date
        last_commit = pr.commits[PullRequestCommit.committed_date.key].nonemax()
        # convert to "U" dtype to enable sorting in np.in1d
        authored_comments = pr.comments[PullRequestReviewComment.user_login.key].values.astype("U")
        external_comments_times = pr.comments[PullRequestComment.created_at.key].take(
            np.where((authored_comments != pr.pr[PullRequest.user_login.key]) &
                     np.in1d(authored_comments, self._bots, invert=True))[0])
        first_comment = nonemin(
            pr.review_comments[PullRequestReviewComment.created_at.key].nonemin(),
            pr.reviews[PullRequestReview.submitted_at.key].nonemin(),
            external_comments_times.nonemin())
        if closed and first_comment and first_comment > closed:
            first_comment = None
        first_comment_on_first_review = first_comment or merged
        if first_comment_on_first_review:
            committed_dates = pr.commits[PullRequestCommit.committed_date.key]
            last_commit_before_first_review = committed_dates.take(np.where(
                committed_dates <= first_comment_on_first_review)[0]).nonemax()
            if not (last_commit_before_first_review_own := bool(last_commit_before_first_review)):
                last_commit_before_first_review = first_comment_on_first_review
            # force pushes that were lost
            first_commit = nonemin(first_commit, last_commit_before_first_review)
            last_commit = nonemax(last_commit, first_commit)
            first_review_request_backup = nonemin(
                nonemax(created, last_commit_before_first_review),
                first_comment_on_first_review)
        else:
            last_commit_before_first_review = None
            last_commit_before_first_review_own = False
            first_review_request_backup = None
        first_review_request = first_review_request_exact = \
            pr.review_requests[PullRequestReviewRequest.created_at.key].nonemin()
        if first_review_request_backup and first_review_request and \
                first_review_request > first_comment_on_first_review:
            # we cannot request a review after we received a review
            first_review_request = first_review_request_backup
        else:
            first_review_request = first_review_request or first_review_request_backup
        # ensure that the first review request is not earlier than the last commit before
        # the first review
        if last_commit_before_first_review_own and \
                last_commit_before_first_review > first_review_request:
            first_review_request = last_commit_before_first_review or first_review_request
        review_submitted_ats = pr.reviews[PullRequestReview.submitted_at.key]

        if closed:
            # it is possible to approve/reject after closing the PR
            # you start the review, then somebody closes the PR, then you submit the review
            try:
                last_review = pd.Timestamp(review_submitted_ats.values[
                    (review_submitted_ats.values <= closed.to_numpy())
                ].max(), tz=timezone.utc)
            except ValueError:
                last_review = None
            last_review = nonemax(
                last_review,
                external_comments_times.take(np.where(
                    external_comments_times <= closed)[0]).nonemax())
        else:
            last_review = review_submitted_ats.nonemax() or \
                nonemin(external_comments_times.nonemax())
        if not first_review_request:
            assert not last_review, pr.pr[PullRequest.node_id.key]
        if merged:
            reviews_before_merge = \
                pr.reviews[PullRequestReview.submitted_at.key].values <= merged.to_numpy()
            if reviews_before_merge.all():
                reviews_before_merge = pr.reviews
            else:
                reviews_before_merge = pr.reviews.take(np.where(reviews_before_merge)[0])
                reviews_before_merge.reset_index(drop=True, inplace=True)
        else:
            reviews_before_merge = pr.reviews
        # the most recent review for each reviewer
        if reviews_before_merge.empty:
            # express lane
            grouped_reviews = self.dummy_reviews
        elif reviews_before_merge[PullRequestReview.user_login.key].nunique() == 1:
            # fast lane
            grouped_reviews = reviews_before_merge._ixs(
                reviews_before_merge[PullRequestReview.submitted_at.key].values.argmax())
        else:
            # the most recent review for each reviewer
            latest_review_ixs = [
                ixs[0] for ixs in
                reviews_before_merge[[PullRequestReview.user_login.key,
                                      PullRequestReview.submitted_at.key]]
                .take(np.where(reviews_before_merge[PullRequestReview.state.key] !=
                               ReviewResolution.COMMENTED.value)[0])
                .sort_values([PullRequestReview.submitted_at.key], ascending=False)
                .groupby(PullRequestReview.user_login.key, sort=False)
                .grouper.groups.values()
            ]
            grouped_reviews = {
                k: reviews_before_merge[k].take(latest_review_ixs)
                for k in (PullRequestReview.state.key, PullRequestReview.submitted_at.key)}
        grouped_reviews_states = grouped_reviews[PullRequestReview.state.key]
        if isinstance(grouped_reviews_states, str):
            changes_requested = grouped_reviews_states == ReviewResolution.CHANGES_REQUESTED.value
        else:
            changes_requested = (
                grouped_reviews_states.values == ReviewResolution.CHANGES_REQUESTED.value
            ).any()
        if changes_requested:
            # merged with negative reviews
            approved = None
        else:
            if isinstance(grouped_reviews_states, str):
                if grouped_reviews_states == ReviewResolution.APPROVED.value:
                    approved = grouped_reviews[PullRequestReview.submitted_at.key]
                else:
                    approved = None
            else:
                approved = grouped_reviews[PullRequestReview.submitted_at.key].take(
                    np.where(grouped_reviews_states == ReviewResolution.APPROVED.value)[0],
                ).nonemax()
            if approved and closed:
                # similar to last_review
                approved = min(approved, closed)
        released = pr.release[Release.published_at.key]
        if released != released:
            released = None
        additions = pr.pr[PullRequest.additions.key]
        deletions = pr.pr[PullRequest.deletions.key]
        if additions is None or deletions is None:
            self.log.error("NULL in PR additions or deletions: %s (%s#%d): +%s -%s",
                           pr.pr[PullRequest.node_id.key],
                           pr.pr[PullRequest.repository_full_name.key],
                           pr.pr[PullRequest.number.key],
                           additions, deletions)
            raise ImpossiblePullRequest()
        size = additions + deletions
        force_push_dropped = pr.release[matched_by_column] == ReleaseMatch.force_push_drop
        done = bool(released or force_push_dropped or (closed and not merged))
        work_began = nonemin(created, first_commit)
        ts_dtype = "datetime64[ns]"
        reviews = np.sort(reviews_before_merge[PullRequestReview.submitted_at.key].values) \
            .astype(ts_dtype)
        activity_days = np.concatenate([
            np.array([created, closed, released], dtype=ts_dtype),
            pr.commits[PullRequestCommit.committed_date.key].values,
            pr.review_requests[PullRequestReviewRequest.created_at.key].values,
            pr.reviews[PullRequestReview.created_at.key].values,
            pr.comments[PullRequestComment.created_at.key].values,
        ]).astype("datetime64[D]")
        activity_days = \
            np.unique(activity_days[activity_days == activity_days]).astype(ts_dtype)
        facts = PullRequestFacts(
            created=created,
            first_commit=first_commit,
            work_began=work_began,
            last_commit_before_first_review=last_commit_before_first_review,
            last_commit=last_commit,
            merged=merged,
            first_comment_on_first_review=first_comment_on_first_review,
            first_review_request=first_review_request,
            first_review_request_exact=first_review_request_exact,
            last_review=last_review,
            approved=approved,
            released=released,
            closed=closed,
            done=done,
            reviews=reviews,
            activity_days=activity_days,
            size=size,
            force_push_dropped=force_push_dropped,
        )
        self._validate(facts, pr.pr[PullRequest.htmlurl.key])
        return facts

    def _validate(self, facts: PullRequestFacts, url: str) -> None:
        """Run sanity checks to ensure consistency."""
        facts.validate()
        if not facts.closed:
            return
        if facts.last_commit and facts.last_commit > facts.closed:
            self.log.error("%s is impossible: closed %s but last commit %s: delta %s",
                           url, facts.closed, facts.last_commit,
                           facts.closed - facts.last_commit)
            raise ImpossiblePullRequest()
        if facts.created > facts.closed:
            self.log.error("%s is impossible: closed %s but created %s: delta %s",
                           url, facts.closed, facts.created,
                           facts.closed - facts.created)
            raise ImpossiblePullRequest()
        if facts.merged and facts.released and facts.merged > facts.released:
            self.log.error("%s is impossible: merged %s but released %s: delta %s",
                           url, facts.merged, facts.released,
                           facts.released - facts.merged)
            raise ImpossiblePullRequest()
