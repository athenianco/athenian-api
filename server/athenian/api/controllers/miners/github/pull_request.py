import asyncio
import dataclasses
from datetime import date, datetime, timezone
from enum import Enum
import logging
import pickle
from typing import Any, Collection, Dict, Generator, Generic, List, Mapping, Optional, Set, \
    Tuple, TypeVar, Union

import aiomcache
import databases
import numpy as np
import pandas as pd
from sqlalchemy import select, sql
from sqlalchemy.orm.attributes import InstrumentedAttribute

from athenian.api import metadata
from athenian.api.async_read_sql_query import read_sql_query
from athenian.api.cache import cached, CancelCache
from athenian.api.controllers.miners.github.hardcoded import BOTS
from athenian.api.controllers.miners.github.release import map_prs_to_releases, \
    map_releases_to_prs
from athenian.api.controllers.miners.pull_request_list_item import ParticipationKind
from athenian.api.controllers.settings import ReleaseMatchSetting
from athenian.api.models.metadata import PREFIXES
from athenian.api.models.metadata.github import Base, PullRequest, PullRequestComment, \
    PullRequestCommit, PullRequestReview, PullRequestReviewComment, PullRequestReviewRequest, \
    Release


@dataclasses.dataclass(frozen=True)
class MinedPullRequest:
    """All the relevant information we are able to load from the metadata DB about a PR.

    All the DataFrame-s have a two-layered index:
    1. pull request id
    2. own id
    The artificial first index layer makes it is faster to select data belonging to a certain PR.
    """

    pr: Dict[str, Any]
    reviews: pd.DataFrame
    review_comments: pd.DataFrame
    review_requests: pd.DataFrame
    comments: pd.DataFrame
    commits: pd.DataFrame
    release: Dict[str, Any]

    def participants(self, with_prefix=True) -> Mapping[ParticipationKind, Set[str]]:
        """
        Collect unique developer logins which are mentioned in this pull request.

        :param with_prefix: Value indicating whether to prepend "github.com/" to the returned \
                            logins.
        """
        prefix = PREFIXES["github"] if with_prefix else ""
        author = self.pr[PullRequest.user_login.key]
        merger = self.pr[PullRequest.merged_by_login.key]
        releaser = self.release[Release.author.key]
        participants = {
            ParticipationKind.AUTHOR: {prefix + author} if author else set(),
            ParticipationKind.REVIEWER: self._extract_people(
                self.reviews, PullRequestReview.user_login.key, prefix),
            ParticipationKind.COMMENTER: self._extract_people(
                self.comments, PullRequestComment.user_login.key, prefix),
            ParticipationKind.COMMIT_COMMITTER: self._extract_people(
                self.commits, PullRequestCommit.committer_login.key, prefix),
            ParticipationKind.COMMIT_AUTHOR: self._extract_people(
                self.commits, PullRequestCommit.author_login.key, prefix),
            ParticipationKind.MERGER: {prefix + merger} if merger else set(),
            ParticipationKind.RELEASER: {prefix + releaser} if releaser else set(),
        }
        try:
            participants[ParticipationKind.REVIEWER].remove(prefix + author)
        except (KeyError, TypeError):
            pass
        return participants

    @staticmethod
    def _extract_people(df: pd.DataFrame, col: str, prefix: str) -> Set[str]:
        return set(prefix + df[col].values[np.where(df[col].values)[0]])


class PullRequestMiner:
    """Load all the information related to Pull Requests from the metadata DB. Iterate over it \
    with individual PR tuples."""

    CACHE_TTL = 5 * 60
    log = logging.getLogger("%s.PullRequestMiner" % metadata.__package__)

    def __init__(self, prs: pd.DataFrame, reviews: pd.DataFrame, review_comments: pd.DataFrame,
                 review_requests: pd.DataFrame, comments: pd.DataFrame, commits: pd.DataFrame,
                 releases: pd.DataFrame):
        """Initialize a new instance of `PullRequestMiner`."""
        self._prs = prs
        self._reviews = reviews
        self._review_comments = review_comments
        self._review_requests = review_requests
        self._comments = comments
        self._commits = commits
        self._releases = releases

    def _postprocess_cached_prs(result: Tuple[List[pd.DataFrame], Set[str], Set[str]],
                                repositories: Collection[str],
                                developers: Collection[str],
                                pr_blacklist: Optional[Collection[str]] = None,
                                **_) -> Tuple[List[pd.DataFrame], Set[str], Set[str]]:
        dfs, cached_repositories, cached_developers = result
        if set(repositories) - cached_repositories:
            raise CancelCache()
        if cached_developers and (not developers or set(developers) - cached_developers):
            raise CancelCache()
        to_remove = set()
        if pr_blacklist:
            to_remove.update(pr_blacklist)
        prs = dfs[0]
        if not isinstance(repositories, set):
            repositories = set(repositories)
        if not isinstance(developers, set):
            developers = set(developers)
        to_remove.update(prs.index.take(np.where(
            np.in1d(prs[PullRequest.repository_full_name.key].values,
                    list(repositories), assume_unique=True, invert=True),
        )[0]))
        to_remove.update(PullRequestMiner._find_drop_by_developers(dfs, developers))
        PullRequestMiner._drop(dfs, to_remove)
        return result

    @classmethod
    @cached(
        exptime=lambda cls, **_: cls.CACHE_TTL,
        serialize=pickle.dumps,
        deserialize=pickle.loads,
        key=lambda date_from, date_to, exclude_inactive, release_settings, **_: (
            date_from.toordinal(), date_to.toordinal(), exclude_inactive, release_settings,
        ),
        postprocess=_postprocess_cached_prs,
    )
    async def _mine(cls,
                    date_from: date,
                    date_to: date,
                    repositories: Collection[str],
                    developers: Collection[str],
                    exclude_inactive: bool,
                    release_settings: Dict[str, ReleaseMatchSetting],
                    db: databases.Database,
                    cache: Optional[aiomcache.Client],
                    pr_blacklist: Optional[Collection[str]] = None,
                    ) -> Tuple[List[pd.DataFrame], Set[str], Set[str]]:
        assert isinstance(date_from, date) and not isinstance(date_from, datetime)
        assert isinstance(date_to, date) and not isinstance(date_to, datetime)
        if not isinstance(repositories, set):
            repositories = set(repositories)
        if not isinstance(developers, set):
            developers = set(developers)
        time_from, time_to = (pd.Timestamp(t, tzinfo=timezone.utc) for t in (date_from, date_to))
        filters = [
            sql.and_(sql.or_(PullRequest.closed_at.is_(None),
                             PullRequest.closed_at >= time_from),
                     PullRequest.created_at < time_to,
                     PullRequest.hidden.is_(False)),
            PullRequest.repository_full_name.in_(repositories),
        ]
        prs = await read_sql_query(select([PullRequest]).where(sql.and_(*filters)),
                                   db, PullRequest, index=PullRequest.node_id.key)
        released_prs = await map_releases_to_prs(
            repositories, time_from, time_to, release_settings, db, cache)
        prs = pd.concat([prs, released_prs], sort=False)
        prs = prs[~prs.index.duplicated()]
        if pr_blacklist:
            old_len = len(prs)
            prs.drop(pr_blacklist, inplace=True, errors="ignore")
            cls.log.info("PR blacklist cut the number of PRs from %d to %d", old_len, len(prs))
        prs.sort_index(level=0, inplace=True, sort_remaining=False)
        cls._truncate_timestamps(prs, time_to)
        # bypass the useless inner caching by calling __wrapped__ directly
        dfs = await cls.mine_by_ids.__wrapped__(
            cls, prs, time_from, time_to, release_settings, db, cache)
        dfs = [prs, *dfs]
        cls._drop(dfs, cls._find_drop_by_developers(dfs, developers))
        return dfs, repositories, developers

    _postprocess_cached_prs = staticmethod(_postprocess_cached_prs)

    @classmethod
    @cached(
        exptime=lambda cls, **_: cls.CACHE_TTL,
        serialize=pickle.dumps,
        deserialize=pickle.loads,
        key=lambda prs, time_from, time_to, release_settings, **_: (
            ",".join(prs.index), time_from.timestamp(), time_to.timestamp(), release_settings,
        ),
    )
    async def mine_by_ids(cls,
                          prs: pd.DataFrame,
                          time_from: datetime,
                          time_to: datetime,
                          release_settings: Dict[str, ReleaseMatchSetting],
                          db: databases.Database,
                          cache: Optional[aiomcache.Client],
                          ) -> List[pd.DataFrame]:
        """
        Fetch PR metadata for certain PRs.

        :param prs: pandas DataFrame with fetched PullRequest-s. Only the details about those PRs \
                    will be loaded from the DB.
        """
        node_ids = prs.index if len(prs) > 0 else set()

        async def fetch_reviews():
            return await cls._read_filtered_models(
                db, PullRequestReview, node_ids, time_to,
                columns=[PullRequestReview.submitted_at, PullRequestReview.user_id,
                         PullRequestReview.state, PullRequestReview.user_login])

        async def fetch_review_comments():
            return await cls._read_filtered_models(
                db, PullRequestReviewComment, node_ids, time_to,
                columns=[PullRequestReviewComment.created_at, PullRequestReviewComment.user_id])

        async def fetch_review_requests():
            return await cls._read_filtered_models(
                db, PullRequestReviewRequest, node_ids, time_to,
                columns=[PullRequestReviewRequest.created_at])

        async def fetch_comments():
            return await cls._read_filtered_models(
                db, PullRequestComment, node_ids, time_to,
                columns=[PullRequestComment.created_at, PullRequestComment.user_id,
                         PullRequestComment.user_login])

        async def fetch_commits():
            return await cls._read_filtered_models(
                db, PullRequestCommit, node_ids, time_to,
                columns=[PullRequestCommit.authored_date, PullRequestCommit.committed_date,
                         PullRequestCommit.author_login, PullRequestCommit.committer_login])

        async def map_releases():
            merged_prs = prs[prs[PullRequest.merged_at.key] <= time_to]
            return await map_prs_to_releases(
                merged_prs, time_from, time_to, release_settings, db, cache)

        return await asyncio.gather(
            fetch_reviews(), fetch_review_comments(), fetch_review_requests(), fetch_comments(),
            fetch_commits(), map_releases())

    @classmethod
    def _remove_spurious_prs(cls,
                             time_from: datetime,
                             prs: pd.DataFrame,
                             reviews: pd.DataFrame,
                             review_comments: pd.DataFrame,
                             review_requests: pd.DataFrame,
                             comments: pd.DataFrame,
                             commits: pd.DataFrame,
                             releases: pd.DataFrame):
        old_releases = np.where(releases[Release.published_at.key] < time_from)[0]
        if len(old_releases) == 0:
            return
        cls._drop((prs, reviews, review_comments, review_requests, comments, commits, releases),
                  releases.index[old_releases])

    @classmethod
    def _drop(cls, dfs: Collection[pd.DataFrame], pr_ids: Collection[str]) -> None:
        if len(pr_ids) == 0:
            return
        for df in dfs:
            df.drop(pr_ids,
                    level=0 if isinstance(df.index, pd.MultiIndex) else None,
                    inplace=True,
                    errors="ignore")

    @classmethod
    def _find_drop_by_developers(cls, dfs: List[pd.DataFrame], developers: Set[str]) -> Set[str]:
        if not developers:
            return set()
        developers = np.asarray(list(developers))
        prs, reviews, _, _, comments, commits, releases = dfs
        passed_pr_ids = set(prs.index.take(np.where(
            np.in1d(prs[PullRequest.user_login.key].values, developers, assume_unique=True),
        )[0]))
        if len(passed_pr_ids) == len(prs):
            return set()
        # how about mergers?
        passed_pr_ids.update(prs.index.take(np.where(
            np.in1d(prs[PullRequest.merged_by_login.key].values, developers, assume_unique=True),
        )[0]))
        if len(passed_pr_ids) == len(prs):
            return set()
        # how about reviewers?
        passed_pr_ids.update(reviews.index.get_level_values(0).take(np.where(
            np.in1d(reviews[PullRequestReview.user_login.key].values, developers,
                    assume_unique=True),
        )[0]))
        if len(passed_pr_ids) == len(prs):
            return set()
        # how about commenters?
        passed_pr_ids.update(comments.index.get_level_values(0).take(np.where(
            np.in1d(comments[PullRequestComment.user_login.key].values, developers,
                    assume_unique=True),
        )[0]))
        if len(passed_pr_ids) == len(prs):
            return set()
        # how about commits?
        passed_pr_ids.update(commits.index.get_level_values(0).take(np.where(
            np.in1d(commits[PullRequestCommit.author_login.key].values, developers,
                    assume_unique=True)
            | np.in1d(commits[PullRequestCommit.committer_login.key].values, developers,
                      assume_unique=True),
        )[0]))
        if len(passed_pr_ids) == len(prs):
            return set()
        # how about releasers?
        passed_pr_ids.update(releases.index.take(np.where(
            np.in1d(releases[Release.author.key].values, developers, assume_unique=True),
        )[0]))
        if len(passed_pr_ids) == len(prs):
            return set()
        return set(prs.index.values) - passed_pr_ids

    @classmethod
    async def mine(cls,
                   date_from: date,
                   date_to: date,
                   time_from: datetime,
                   time_to: datetime,
                   repositories: Collection[str],
                   developers: Collection[str],
                   exclude_inactive: bool,
                   release_settings: Dict[str, ReleaseMatchSetting],
                   db: databases.Database,
                   cache: Optional[aiomcache.Client],
                   pr_blacklist: Optional[Collection[str]] = None,
                   ) -> "PullRequestMiner":
        """
        Create a new `PullRequestMiner` from the metadata DB according to the specified filters.

        :param date_from: Fetch PRs created starting from this date, inclusive.
        :param date_to: Fetch PRs created ending with this date, inclusive.
        :param time_from: Precise timestamp of since when PR events are allowed to happen.
        :param time_to: Precise timestamp of until when PR events are allowed to happen.
        :param repositories: PRs must belong to these repositories (prefix excluded).
        :param developers: PRs must be authored by these user IDs. An empty list means everybody.
        :param exclude_inactive: Ors must have at least one event in the given time frame.
        :param release_settings: Release match settings of the account.
        :param db: Metadata db instance.
        :param cache: memcached client to cache the collected data.
        :param pr_blacklist: completely ignore the existence of these PR node IDs.
        """
        date_from_with_time = datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc)
        date_to_with_time = datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc)
        assert time_from >= date_from_with_time
        assert time_to <= date_to_with_time
        dfs, _, _ = await cls._mine(date_from, date_to, repositories, developers, exclude_inactive,
                                    release_settings, db, cache, pr_blacklist=pr_blacklist)
        cls._truncate_prs(dfs, time_from, time_to)
        return cls(*dfs)

    @staticmethod
    async def _read_filtered_models(conn: Union[databases.core.Connection, databases.Database],
                                    model_cls: Base,
                                    node_ids: Collection[str],
                                    time_to: datetime,
                                    columns: Optional[List[InstrumentedAttribute]] = None,
                                    ) -> pd.DataFrame:
        if columns is not None:
            columns = [model_cls.pull_request_node_id, model_cls.node_id] + columns
        df = await read_sql_query(select(columns or [model_cls]).where(
            sql.and_(model_cls.pull_request_node_id.in_(node_ids),
                     model_cls.created_at < time_to)),
            con=conn,
            columns=columns or model_cls,
            index=[model_cls.pull_request_node_id.key, model_cls.node_id.key])
        return df

    @classmethod
    def _truncate_prs(cls, dfs: List[pd.DataFrame], time_from: datetime, time_to: datetime,
                      ) -> None:
        """
        Remove PRs outside of the given time range.

        This is used to correctly handle timezone offsets.
        """
        prs, releases = dfs[0], dfs[-1]
        # filter out PRs which were released before `time_from`
        unreleased = releases.index.take(np.where(
            releases[Release.published_at.key] < time_from)[0])
        # closed but not merged in `[date_from, time_from]`
        unrejected = prs.index.take(np.where(
            (prs[PullRequest.closed_at.key] < time_from) &
            prs[PullRequest.merged_at.key].isnull())[0])
        # created in `[time_to, date_to]`
        uncreated = prs.index.take(np.where(
            prs[PullRequest.created_at.key] >= time_to)[0])
        to_remove = unreleased.union(unrejected.union(uncreated))
        cls._drop(dfs, to_remove)
        for df in dfs:
            cls._truncate_timestamps(df, time_to)

    @staticmethod
    def _truncate_timestamps(df: pd.DataFrame, upto: datetime):
        """Set all the timestamps after `upto` to NaT to avoid "future leakages"."""
        for col in df.select_dtypes(include=[object]):
            try:
                df.loc[df[col] > upto, col] = pd.NaT
            except TypeError:
                continue
        for col in df.select_dtypes(include=["datetime"]):
            df.loc[df[col] > upto, col] = pd.NaT

    def __iter__(self) -> Generator[MinedPullRequest, None, None]:
        """Iterate over the individual pull requests."""
        df_fields = list(MinedPullRequest.__dataclass_fields__)
        df_fields.remove("pr")
        dfs = []
        grouped_df_iters = []
        for k in df_fields:
            df = getattr(self, "_" + (k if k.endswith("s") else k + "s"))
            dfs.append(df)
            grouped_df_iters.append(iter(df.groupby(level=0, sort=True, as_index=False)))
        grouped_df_states = []
        for i in grouped_df_iters:
            try:
                grouped_df_states.append(next(i))
            except StopIteration:
                grouped_df_states.append((None, None))
        empty_df_cache = {}
        pr_columns = [PullRequest.node_id.key]
        pr_columns.extend(self._prs.columns)
        if not self._prs.index.is_monotonic_increasing:
            raise IndexError("PRs index must be pre-sorted ascending: "
                             "prs.sort_index(inplace=True)")
        for pr_tuple in self._prs.itertuples():
            pr_node_id = pr_tuple.Index
            items = {"pr": dict(zip(pr_columns, pr_tuple))}
            for i, (k, (state_pr_node_id, gdf), git, df) in enumerate(zip(
                    df_fields, grouped_df_states, grouped_df_iters, dfs)):
                if state_pr_node_id == pr_node_id:
                    if not k.endswith("s"):
                        # much faster than gdf.iloc[0]
                        gdf = {c: v for c, v in zip(gdf.columns, gdf._data.fast_xs(0))}
                    else:
                        gdf.index = gdf.index.droplevel(0)
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


T = TypeVar("T")


class Fallback(Generic[T]):
    """
    A value with a "plan B".

    The idea is to return the backup in `Fallback.best` if the primary value is absent (None).
    We can check whether the primary value exists by `Fallback.value is None`.
    """

    def __init__(self, value: Optional[T], fallback: Union[None, T, "Fallback[T]"]):
        """Initialize a new instance of `Fallback`."""
        if value != value:  # NaN check
            value = None
        self.__value = value
        self.__fallback = fallback

    @property
    def best(self) -> Optional[T]:
        """The "best effort" value, either the primary or the backup one."""  # noqa: D401
        if self.__value is not None:
            return self.__value
        if isinstance(self.__fallback, Fallback):
            return self.__fallback.best
        return self.__fallback

    def __str__(self) -> str:
        """str()."""
        return "Fallback(%s, %s)" % (self.value, self.best)

    def __repr__(self) -> str:
        """repr()."""
        return "Fallback(%r, %r)" % (self.value, self.best)

    def __bool__(self) -> bool:
        """Return the value indicating whether there is any value, either primary or backup."""
        return self.best is not None

    def __lt__(self, other: "Fallback[T]") -> bool:
        """Implement <."""
        if not self or not other:
            raise ArithmeticError
        return self.best < other.best

    def __eq__(self, other: "Fallback[T]") -> bool:
        """Implement ==."""
        return self.best == other.best

    def __le__(self, other: "Fallback[T]") -> bool:
        """Implement <=."""
        if not self or not other:
            raise ArithmeticError
        return self.best <= other.best

    @property
    def value(self) -> Optional[T]:
        """The primary value."""  # noqa: D401
        return self.__value

    @classmethod
    def max(cls, *args: "Fallback[T]") -> "Fallback[T]":
        """Calculate the maximum of several Fallback.best-s."""
        return cls.agg(max, *args)

    @classmethod
    def min(cls, *args: "Fallback[T]") -> "Fallback[T]":
        """Calculate the minimum of several Fallback.best-s."""
        return cls.agg(min, *args)

    @classmethod
    def agg(cls, func: callable, *args: "Fallback[T]") -> "Fallback[T]":
        """Calculate an aggregation of several Fallback.best-s."""
        try:
            return cls(func(arg.best for arg in args if arg.best is not None), None)
        except ValueError:
            return cls(None, None)


DT = Union[pd.Timestamp, datetime, None]


@dataclasses.dataclass(frozen=True)
class PullRequestTimes:
    """Various PR update timestamps."""

    @property
    def work_began(self) -> Fallback[DT]:  # PR_B   noqa: D102
        return Fallback.min(self.created, self.first_commit)

    created: Fallback[DT]                                # PR_C
    first_commit: Fallback[DT]                           # PR_CC
    last_commit_before_first_review: Fallback[DT]        # PR_CFR
    last_commit: Fallback[DT]                            # PR_LC
    merged: Fallback[DT]                                 # PR_M
    closed: Fallback[DT]                                 # PR_CL
    first_comment_on_first_review: Fallback[DT]          # PR_W
    first_review_request: Fallback[DT]                   # PR_S
    approved: Fallback[DT]                               # PR_A
    last_review: Fallback[DT]                            # PR_LR
    first_passed_checks: Fallback[DT]                    # PR_VS
    last_passed_checks: Fallback[DT]                     # PR_VF
    released: Fallback[DT]                               # PR_R

    def max_timestamp(self) -> DT:
        """Find the maximum timestamp contained in the struct."""
        return Fallback.max(*self.__dict__.values()).best

    def __str__(self) -> str:
        """Format for human-readability."""
        return "{\n\t%s\n}" % ",\n\t".join(
            "%s: %s" % (k, v.best) for k, v in dataclasses.asdict(self).items())

    def __lt__(self, other: "PullRequestTimes") -> bool:
        """Order by `work_began`."""
        return self.work_began.best < other.work_began.best

    def __hash__(self) -> int:
        """Implement hash()."""
        return hash(str(self))


class ReviewResolution(Enum):
    """Possible review "state"-s in the metadata DB."""

    APPROVED = "APPROVED"
    CHANGES_REQUESTED = "CHANGES_REQUESTED"
    COMMENTED = "COMMENTED"


class ImpossiblePullRequest(Exception):
    """Raised by PullRequestTimesMiner._compile() on broken PRs."""


class PullRequestTimesMiner:
    """Extract the pull request event timestamps from MinedPullRequest-s."""

    log = logging.getLogger("%s.PullRequestTimesMiner" % metadata.__package__)

    def __call__(self, pr: MinedPullRequest) -> PullRequestTimes:
        """
        Extract the pull request event timestamps from a MinedPullRequest.

        May raise ImpossiblePullRequest if the PR has an "impossible" state like
        created after closed.
        """
        created_at = Fallback(pr.pr[PullRequest.created_at.key], None)
        merged_at = Fallback(pr.pr[PullRequest.merged_at.key], None)
        closed_at = Fallback(pr.pr[PullRequest.closed_at.key], None)
        # we don't need the indexes
        pr.comments.reset_index(inplace=True, drop=True)
        pr.reviews.reset_index(inplace=True, drop=True)
        first_commit = Fallback(pr.commits[PullRequestCommit.authored_date.key].min(), None)
        # yes, first_commit uses authored_date while last_commit uses committed_date
        last_commit = Fallback(pr.commits[PullRequestCommit.committed_date.key].max(), None)
        authored_comments = pr.comments[PullRequestReviewComment.user_id.key]
        external_comments_times = pr.comments[PullRequestComment.created_at.key].take(
            np.where((authored_comments != pr.pr[PullRequest.user_id.key]) &
                     ~authored_comments.isin(BOTS))[0])
        first_comment = dtmin(
            pr.review_comments[PullRequestReviewComment.created_at.key].min(),
            pr.reviews[PullRequestReview.submitted_at.key].min(),
            external_comments_times.min())
        if closed_at and first_comment is not None and first_comment > closed_at.best:
            first_comment = None
        first_comment_on_first_review = Fallback(first_comment, merged_at)
        if first_comment_on_first_review:
            committed_dates = pr.commits[PullRequestCommit.committed_date.key]
            last_commit_before_first_review = Fallback(
                committed_dates.take(np.where(
                    committed_dates <= first_comment_on_first_review.best)[0]).max(),
                first_comment_on_first_review)
            # force pushes that were lost
            first_commit = Fallback.min(first_commit, last_commit_before_first_review)
            last_commit = Fallback.max(last_commit, first_commit)
            first_review_request_backup = Fallback.min(
                Fallback.max(created_at, last_commit_before_first_review),
                first_comment_on_first_review)
        else:
            last_commit_before_first_review = Fallback(None, None)
            first_review_request_backup = None
        first_review_request = pr.review_requests[PullRequestReviewRequest.created_at.key].min()
        if first_review_request_backup and first_review_request == first_review_request and \
                first_review_request > first_comment_on_first_review.best:
            # we cannot request a review after we received a review
            first_review_request = Fallback(first_review_request_backup.best, None)
        else:
            first_review_request = Fallback(first_review_request, first_review_request_backup)
        # ensure that the first review request is not earlier than the last commit before
        # the first review
        if last_commit_before_first_review.value is not None and \
                last_commit_before_first_review > first_review_request:
            first_review_request = Fallback(
                last_commit_before_first_review.value, first_review_request)
        review_submitted_ats = pr.reviews[PullRequestReview.submitted_at.key]
        if closed_at:
            not_review_comments = \
                pr.reviews[PullRequestReview.state.key].values != ReviewResolution.COMMENTED.value
            # it is possible to approve/reject after closing the PR
            # you start the review, then somebody closes the PR, then you submit the review
            last_review_at = review_submitted_ats.take(
                np.where((review_submitted_ats.values <= closed_at.best.to_numpy()) |
                         not_review_comments)[0]).max()
            if last_review_at == last_review_at:
                # we don't want dtmin() here - what if there was no review at all?
                last_review_at = min(last_review_at, closed_at.best)
            last_review = Fallback(
                last_review_at,
                dtmin(external_comments_times.take(np.where(
                    external_comments_times <= closed_at.best)[0]).max()))
        else:
            last_review = Fallback(review_submitted_ats.max(),
                                   dtmin(external_comments_times.max()))
        if merged_at:
            reviews_before_merge = \
                pr.reviews[PullRequestReview.submitted_at.key].values <= merged_at.best.to_numpy()
            if reviews_before_merge.all():
                reviews_before_merge = pr.reviews
            else:
                reviews_before_merge = pr.reviews.take(np.where(reviews_before_merge)[0])
        else:
            reviews_before_merge = pr.reviews
        # the most recent review for each reviewer
        if reviews_before_merge.empty:
            grouped_reviews = reviews_before_merge
        elif reviews_before_merge[PullRequestReview.user_id.key].nunique() == 1:
            # fast lane
            grouped_reviews = reviews_before_merge.take([
                reviews_before_merge[PullRequestReview.submitted_at.key].values.argmax()])
        else:
            grouped_reviews = reviews_before_merge \
                .take(np.where(reviews_before_merge[PullRequestReview.state.key] !=
                               ReviewResolution.COMMENTED.value)[0]) \
                .sort_values([PullRequestReview.submitted_at.key],
                             ascending=False, ignore_index=True) \
                .groupby(PullRequestReview.user_id.key, sort=False, as_index=False) \
                .head(1)  # the most recent review for each reviewer
        if (grouped_reviews[PullRequestReview.state.key].values
                == ReviewResolution.CHANGES_REQUESTED.value).any():
            # merged with negative reviews
            approved_at_value = None
        else:
            approved_at_value = grouped_reviews[PullRequestReview.submitted_at.key].take(
                np.where(grouped_reviews[PullRequestReview.state.key].values ==
                         ReviewResolution.APPROVED.value)[0]).max()
            if approved_at_value == approved_at_value and closed_at:
                # similar to last_review
                approved_at_value = min(approved_at_value, closed_at.best)
        approved_at = Fallback(approved_at_value, None)
        last_passed_checks = Fallback(None, None)  # FIXME(vmarkovtsev): no CI info
        released_at = Fallback(pr.release[Release.published_at.key], None)
        times = PullRequestTimes(
            created=created_at,
            first_commit=first_commit,
            last_commit_before_first_review=last_commit_before_first_review,
            last_commit=last_commit,
            merged=merged_at,
            first_comment_on_first_review=first_comment_on_first_review,
            first_review_request=first_review_request,
            last_review=last_review,
            approved=approved_at,
            first_passed_checks=Fallback(None, None),  # FIXME(vmarkovtsev): no CI info
            last_passed_checks=last_passed_checks,
            released=released_at,
            closed=closed_at,
        )
        self._validate(times, pr.pr[PullRequest.htmlurl.key])
        return times

    def _validate(self, times: PullRequestTimes, url: str) -> None:
        """Run sanity checks to ensure consistency."""
        if not times.closed:
            return
        if times.last_commit and times.last_commit.best > times.closed.best:
            self.log.error("%s is impossible: closed %s but last commit %s: delta %s",
                           url, times.closed.best, times.last_commit.best,
                           times.closed.best - times.last_commit.best)
            raise ImpossiblePullRequest()
        if times.created.best > times.closed.best:
            self.log.error("%s is impossible: closed %s but created %s: delta %s",
                           url, times.closed.best, times.created.best,
                           times.closed.best - times.created.best)
            raise ImpossiblePullRequest()
        if times.merged and times.released and times.merged.best > times.released.best:
            self.log.error("%s is impossible: merged %s but released %s: delta %s",
                           url, times.merged.best, times.released.best,
                           times.released.best - times.merged.best)
            raise ImpossiblePullRequest()


def dtmin(*args: Union[DT, float]) -> DT:
    """Find the minimum of several dates handling NaNs gracefully."""
    if all((arg != arg) for arg in args):
        return None
    return min(arg for arg in args if arg == arg)
